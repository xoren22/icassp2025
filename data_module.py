import os
import cv2
import math
import torch
import numpy as np
import pandas as pd
from skimage.io import imread
from torch.utils.data import Dataset


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

    def compute_pathloss_clamped(
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
        distance = np.sqrt((x_ant - x_grid)**2 + (y_ant - y_grid)**2)

        fspl = self.compute_pathloss_clamped(
            W, H, x_ant, y_ant, antenna_gain, freq_MHz
        )
        
        # Build input tensor: [6, H, W]
        input_tensor = np.zeros((6, H, W), dtype=np.float32)
        input_tensor[0] = input_img[:, :, 0].astype(np.float32)  # reflectance
        input_tensor[1] = input_img[:, :, 1].astype(np.float32)  # transmittance
        input_tensor[2] = distance # distance
        input_tensor[3] = antenna_gain
        input_tensor[4] = freq_MHz  # constant freq map
        input_tensor[5] = fspl

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


