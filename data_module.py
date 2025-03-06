import os
import torch
import numpy as np
import pandas as pd
from numba import njit
from skimage.io import imread
from torch.utils.data import Dataset

from utils import matrix_to_image, measure_time


class PathlossNormalizer:
    def __init__(
            self,
            min_pathloss=0,
            max_pathloss=160.0,
            min_antenna_gain=-55.0,
        ):
        # ImageNet stats for the first three channels
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.min_pathloss = min_pathloss
        self.max_pathloss = max_pathloss
        self.min_antenna_gain = min_antenna_gain

    def normalize_input(self, input_tensor):
        """
        Apply all normalizations to input tensor
        
        Args:
            input_tensor: 6-channel input tensor 
                          [reflectance, transmittance, fspl, distance, antenna_gain, frequency]
        
        Returns:
            Fully normalized input tensor
        """
        normalized = input_tensor.clone()

        # Reflectance & transmittance
        for i in range(2):
            normalized[i] = (normalized[i] / 255.0 - self.mean[i]) / self.std[i]
        
        # Log distance
        normalized[2] = np.log10(1 + normalized[2])

        # Antenna gain
        normalized[3] = normalized[3] / self.min_antenna_gain

        # Frequency channel
        normalized[4] = np.log10(normalized[4]) - 1.9  # "magic shift"

        # Free space pathloss channel
        normalized[5] = 2.0 * ((normalized[5] - self.min_pathloss) / (self.max_pathloss - self.min_pathloss)) - 1.0

        return normalized
    
    def normalize_output(self, y):
        """Scale pathloss into [-1, 1]."""
        if self.max_pathloss > self.min_pathloss:
            return 2.0 * ((y - self.min_pathloss) / (self.max_pathloss - self.min_pathloss)) - 1.0
        return y
    
    def denormalize_output(self, y):
        """Inverse of the above scaling."""
        return (y + 1.0) / 2.0 * (self.max_pathloss - self.min_pathloss) + self.min_pathloss



@njit
def calculate_transmittance_loss(energy_matrix, x_ant, y_ant, n_angles=360*128, radial_step=1.0, max_walls=10):
    """
    Approximate the loss from an antenna at (x_ant, y_ant) to every pixel in 'energy_matrix'
    using a polar (angle + radial stepping) algorithm.
    
    If two or more consecutive pixels on the same radial line have the same transmittance value, we only subtract it once.
    
    Stops calculation on a ray after encountering max_walls distinct wall values.
    """
    h, w = energy_matrix.shape
    dtheta = 2.0 * np.pi / n_angles
    output = np.zeros((h, w), dtype=energy_matrix.dtype)  # Initialize with zeros
    
    cos_vals = np.cos(np.arange(n_angles) * dtheta)
    sin_vals = np.sin(np.arange(n_angles) * dtheta)
    
    max_dist = np.sqrt(w*w + h*h)
    
    for i in range(n_angles):
        cos_t = cos_vals[i]
        sin_t = sin_vals[i]
        
        sum_loss = 0.0
        last_val = None  # Start with None to ensure first pixel is always properly processed
        wall_count = 0
        
        r = 0.0
        while r <= max_dist:
            x = x_ant + r * cos_t
            y = y_ant + r * sin_t
            
            px = int(round(x))
            py = int(round(y))
            
            if px < 0 or px >= w or py < 0 or py >= h:
                break
            
            val = energy_matrix[py, px]
            
            # First pixel handling
            if last_val is None:
                last_val = val
            
            if val != last_val and val > 0:  # Only count non-zero values as walls
                sum_loss += val
                last_val = val
                wall_count += 1
                
                if wall_count >= max_walls:
                    r_temp = r + radial_step
                    while r_temp <= max_dist:
                        x_temp = x_ant + r_temp * cos_t
                        y_temp = y_ant + r_temp * sin_t
                        
                        px_temp = int(round(x_temp))
                        py_temp = int(round(y_temp))
                        
                        if px_temp < 0 or px_temp >= w or py_temp < 0 or py_temp >= h:
                            break
                        
                        if output[py_temp, px_temp] == 0 or sum_loss < output[py_temp, px_temp]:
                            output[py_temp, px_temp] = sum_loss
                            
                        r_temp += radial_step
                    break
            elif val != last_val:
                last_val = val
            
            if output[py, px] == 0 or (sum_loss < output[py, px]):
                output[py, px] = sum_loss
                
            r += radial_step
    
    return output

# compiling numba function for better performance
calculate_transmittance_loss(np.array([[1]]), 0, 0)

class PathlossDataset(Dataset):
    """
    Pads input/output to 640 x 640 and returns a mask indicating the valid region.
    Includes a free-space pathloss calculation function for future use.
    """
    def __init__(self, file_list, input_path, output_path, positions_path, 
                 buildings_path, radiation_path, img_size=640, training=False):
        """
        Args:
            file_list: List of file IDs (b, ant, f, sp)
            input_path: Path to input images
            output_path: Path to ground truth pathloss maps
            positions_path: Path to antenna positions
            buildings_path: Path to building details
            radiation_path: Path to radiation patterns
            img_size: We will pad up to exactly 640 x 640
            training: If True, we normalize outputs to [-1,1]
        """
        self.file_list = file_list
        self.input_path = input_path
        self.output_path = output_path
        self.positions_path = positions_path
        self.buildings_path = buildings_path
        self.radiation_path = radiation_path
        self.training = training
        self.normalizer = PathlossNormalizer()
        
        # Frequencies
        self.freqs_MHz = [868, 1800, 3500]

        # We'll pad up to this size
        self.target_size = img_size

    def __len__(self):
        return len(self.file_list)

    def calculate_pathloss(
        self,
        W,
        H,
        x_ant,
        y_ant,
        antenna_gain,       # shape=(360,) antenna gain in dBi [0..359]
        freq_MHz,           # frequency in MHz
        grid_unit_meters=0.25,  # cell size in meters
        min_distance_m=0.25     # clamp distance below this
    ):
        """
        Example free-space pathloss calculation with distance clamping.
        """
        y_idx, x_idx = np.indices((H, W))

        dx = x_idx - x_ant
        dy = y_idx - y_ant

        dist_m = np.sqrt(dx*dx + dy*dy) * grid_unit_meters
        dist_clamped = np.maximum(dist_m, min_distance_m)

        fspl_db = 20.0 * np.log10(dist_clamped) + 20.0 * np.log10(freq_MHz) - 27.55

        pathloss_db = fspl_db - antenna_gain
        return pathloss_db
    
    def __getitem__(self, idx):
        # Parse file info
        b, ant, f, sp = self.file_list[idx]
        
        # File names
        input_file = f"B{b}_Ant{ant}_f{f}_S{sp}.png"
        output_file = f"B{b}_Ant{ant}_f{f}_S{sp}.png"
        position_file = f"Positions_B{b}_Ant{ant}_f{f}.csv"
        building_file = f"B{b}_Details.csv"
        radiation_file = f"Ant{ant}_Pattern.csv"
        
        # Check file existence
        if not all([
            os.path.exists(os.path.join(self.input_path, input_file)),
            os.path.exists(os.path.join(self.output_path, output_file)),
            os.path.exists(os.path.join(self.positions_path, position_file)),
            os.path.exists(os.path.join(self.buildings_path, building_file)),
            os.path.exists(os.path.join(self.radiation_path, radiation_file))
        ]):
            raise ValueError(f"Warning: Some files missing for B{b}_Ant{ant}_f{f}_S{sp}")

        # Read data files
        sampling_positions = pd.read_csv(os.path.join(self.positions_path, position_file))
        building_details = pd.read_csv(os.path.join(self.buildings_path, building_file))
        
        # Building dimensions
        W, H = building_details["W"].iloc[0], building_details["H"].iloc[0]
        
        # Get antenna position
        x_ant = sampling_positions["Y"].loc[int(sp)]
        y_ant = sampling_positions["X"].loc[int(sp)]
        
        # Load images
        input_img = imread(os.path.join(self.input_path, input_file))   # shape [H, W, channels]
        output_img = imread(os.path.join(self.output_path, output_file))# shape [H, W]
        freq_MHz = self.freqs_MHz[int(f)-1]
        
        # Load radiation pattern
        radiation_pattern = np.genfromtxt(os.path.join(self.radiation_path, radiation_file), delimiter=',')

        # Compute angle map
        x_grid = np.repeat(np.linspace(0, W-1, W), H, axis=0).reshape(W, H).T
        y_grid = np.repeat(np.linspace(0, H-1, H), W, axis=0).reshape(H, W)
        
        angles = -(180/np.pi) * np.arctan2((y_ant - y_grid), (x_ant - x_grid)) \
                 + 180 + sampling_positions['Azimuth'].iloc[int(sp)]
        angles = np.where(angles > 359, angles - 360, angles).astype(int)
        
        # Gains, distances
        antenna_gain = radiation_pattern[angles]

        reflectance = input_img[:, :, 0].astype(np.float32)
        transmittance = input_img[:, :, 1].astype(np.float32)
        distance = input_img[:, :, 2].astype(np.float32)


        free_space_pathloss = self.calculate_pathloss(
            W, H, x_ant, y_ant, antenna_gain, freq_MHz
        )

        # with measure_time():
        #     transmittance_loss = transmittance_loss(transmittance, x_ant, y_ant)

        transmittance_loss = calculate_transmittance_loss(transmittance, x_ant, y_ant)
        fs_plus_transmittance_loss = free_space_pathloss + transmittance_loss


        # evaluate_fspl(output_img, fs_plus_transmittance_loss)
        
        # matrix_to_image(
        #     output_img, 
        #     free_space_pathloss, 
        #     transmittance_loss, 
        #     fs_plus_transmittance_loss,
        #     titles=["Free Space Pathloss", "Transmittance Loss", "Free Space + Transmittance"]
            
        # )
        
        # Build input tensor: [6, H, W]
        input_tensor = np.zeros((6, H, W), dtype=np.float32)
        input_tensor[0] = reflectance  # reflectance
        input_tensor[1] =  transmittance # transmittance
        input_tensor[2] = distance # distance
        input_tensor[3] = antenna_gain
        input_tensor[4] = freq_MHz  # constant freq map
        input_tensor[5] = fs_plus_transmittance_loss

        # Ground truth pathloss: [1, H, W]
        output_tensor = output_img.astype(np.float32)[None, ...]  # channel dim

        # Convert to torch
        input_tensor = torch.from_numpy(input_tensor)
        output_tensor = torch.from_numpy(output_tensor)

        # Normalize input
        input_tensor = self.normalizer.normalize_input(input_tensor)
        # Normalize output if training
        if self.training:
            output_tensor = self.normalizer.normalize_output(output_tensor)

        # -------------
        #  PAD TO 640
        # -------------
        padH = self.target_size
        padW = self.target_size

        if H > padH or W > padW:
            raise ValueError(f"Cannot pad to {padH}: image is bigger ({H}x{W}). Either crop or pick a larger pad size.")
        
        padded_input = torch.zeros((6, padH, padW), dtype=input_tensor.dtype)
        padded_output = torch.zeros((1, padH, padW), dtype=output_tensor.dtype)

        padded_input[:, :H, :W] = input_tensor
        padded_output[:, :H, :W] = output_tensor

        # Create mask for valid region
        mask = torch.zeros((1, padH, padW), dtype=torch.bool)
        mask[:, :H, :W] = True

        return padded_input, padded_output, mask




