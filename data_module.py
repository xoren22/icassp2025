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
            input_tensor: 5-channel input tensor [reflectance, transmittance, distance, antenna_gain, frequency]
        
        Returns:
            Fully normalized input tensor
        """
        normalized = input_tensor.clone()

        # First two channels: reflectance and transmittance
        for i in range(2):
            normalized[i] = (normalized[i] / 255.0 - self.mean[i]) / self.std[i]
        
        # normalize distance. Dirty but avoids log0 case
        normalized[2] = np.log10(1+normalized[2])

        normalized[3] = normalized[3] / self.min_antenna_gain
        normalized[4] = np.log10(normalized[4]) - 1.9 # magic constant, don't worry about it
        
        return normalized
    
    def normalize_output(self, y):
        if self.max_pathloss > self.min_pathloss:
            return 2.0 * ((y - self.min_pathloss) / (self.max_pathloss - self.min_pathloss)) - 1.0
        return y
    
    def denormalize_output(self, y):
        return (y + 1.0) / 2.0 * (self.max_pathloss - self.min_pathloss) + self.min_pathloss


class PathlossDataset(Dataset):    
    def __init__(self, file_list, input_path, output_path, positions_path, 
                 buildings_path, radiation_path, img_size=512, training=False
        ):
        """
        Args:
            file_list: List of file IDs (b, ant, f, sp)
            input_path: Path to input images
            output_path: Path to ground truth pathloss maps
            positions_path: Path to antenna positions
            buildings_path: Path to building details
            radiation_path: Path to radiation patterns
            img_size: Size to resize images to
            transform: Optional transforms to apply
            normalizer: PathlossNormalizer for normalization
        """
        self.file_list = file_list
        self.input_path = input_path
        self.output_path = output_path
        self.positions_path = positions_path
        self.buildings_path = buildings_path
        self.radiation_path = radiation_path
        self.training = training
        self.img_size = img_size
        self.normalizer = PathlossNormalizer()
        self.freqs_MHz = [868, 1800, 3500]

    def compute_pathloss_clamped(
        self,
        W, H,
        x_ant, y_ant,
        antenna_gain,   # shape=(360,), antenna gain in dBi [0..359]
        freq_MHz,               # frequency in MHz
        grid_unit_meters=0.25,  # cell size in meters
        min_distance_m=0.25,     # clamp distance below this
    ):

        y_idx, x_idx = np.indices((H, W))

        dx = x_idx - x_ant
        dy = y_idx - y_ant

        dist_m = np.sqrt(dx*dx + dy*dy) * grid_unit_meters
        dist_clamped = np.maximum(dist_m, min_distance_m)

        fspl_db = 20.0 * np.log10(dist_clamped) + 20.0 * np.log10(freq_MHz) - 27.55

        if self.img_size != W or self.img_size != H:
            fspl_db = cv2.resize(fspl_db, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        pathloss_db = fspl_db - antenna_gain

        return pathloss_db


    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Parse file info
        b, ant, f, sp = self.file_list[idx]
        
        # File names
        input_file = f"B{b}_Ant{ant}_f{f}_S{sp}.png"
        output_file = f"B{b}_Ant{ant}_f{f}_S{sp}.png"
        position_file = f"Positions_B{b}_Ant{ant}_f{f}.csv"
        building_file = f"B{b}_Details.csv"
        radiation_file = f"Ant{ant}_Pattern.csv"
        
        # Check if files exist, if not raise exception
        if not all([
            os.path.exists(os.path.join(self.input_path, input_file)),
            os.path.exists(os.path.join(self.output_path, output_file)),
            os.path.exists(os.path.join(self.positions_path, position_file)),
            os.path.exists(os.path.join(self.buildings_path, building_file)),
            os.path.exists(os.path.join(self.radiation_path, radiation_file))
        ]):
            raise ValueError(f"Warning: Some files missing for B{b}_Ant{ant}_f{f}_S{sp}")

        # Load required data files
        sampling_positions = pd.read_csv(os.path.join(self.positions_path, position_file))
        building_details = pd.read_csv(os.path.join(self.buildings_path, building_file))
        
        # Get building dimensions
        W, H = building_details["W"].iloc[0], building_details["H"].iloc[0]
        
        # Get antenna position
        x_ant = sampling_positions["Y"].loc[int(sp)]
        y_ant = sampling_positions["X"].loc[int(sp)]
        
        # Load input and output images
        input_img = imread(os.path.join(self.input_path, input_file))
        output_img = imread(os.path.join(self.output_path, output_file))
        
        freq_MHz = self.freqs_MHz[int(f)-1]
        if self.img_size != input_img.shape[0]:
            input_img = cv2.resize(input_img, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            output_img = cv2.resize(output_img, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
        
        radiation_pattern = np.genfromtxt(os.path.join(self.radiation_path, radiation_file), delimiter=',')

        x_grid = np.repeat(np.linspace(0, W-1, self.img_size), self.img_size, axis=0).reshape(self.img_size, self.img_size).transpose()
        y_grid = np.repeat(np.linspace(0, H-1, self.img_size), self.img_size, axis=0).reshape(self.img_size, self.img_size)
        
        angles = -(180/np.pi) * np.arctan2((y_ant - y_grid), (x_ant - x_grid)) + 180 + sampling_positions['Azimuth'].iloc[int(sp)]
        angles = np.where(angles >  359, angles - 360 , angles).astype(int)
        
        antenna_gain = radiation_pattern[angles]
        
        distance = np.sqrt(((x_ant - x_grid) * (W/self.img_size))**2 + ((y_ant - y_grid) * (H/self.img_size))**2)
        
        input_tensor = np.zeros((5, self.img_size, self.img_size), dtype=np.float32)
        input_tensor[0] = input_img[:, :, 0].astype(np.float32)  # Reflectance (raw uint8)
        input_tensor[1] = input_img[:, :, 1].astype(np.float32)  # Transmittance (raw uint8)
        input_tensor[2] = distance  # Distance
        input_tensor[3] = antenna_gain  # Raw antenna gain
        input_tensor[4] = np.full((self.img_size, self.img_size), freq_MHz)  # Frequency channel
        
        # Prepare ground truth pathloss
        output_tensor = output_img.astype(np.float32)
        
        # Convert to PyTorch tensors
        input_tensor = torch.from_numpy(input_tensor)
        output_tensor = torch.from_numpy(output_tensor).unsqueeze(0)  # Add channel dimension
        
        input_tensor = self.normalizer.normalize_input(input_tensor)
        if self.training:
            output_tensor = self.normalizer.normalize_output(output_tensor)

        # pl = self.compute_pathloss_clamped(
        #     W, H, x_ant, y_ant,
        #     antenna_gain, freq_MHz
        # )

        # matrix_to_image(pl, output_img)

        return input_tensor, output_tensor













def matrix_to_image(*matrices):
    from time import time
    import matplotlib.pyplot as plt
    
    n = len(matrices)
    if n < 2:
        raise ValueError("At least two matrices are required")
    
    # First matrix is free space pathloss, second is ground truth
    free_space_pathloss = matrices[0]
    ground_truth = matrices[1]
    
    # Calculate the difference matrix
    diff = ground_truth - free_space_pathloss
    
    # Create figure with n+1 subplots (n input matrices + 1 diff matrix)
    fig, axes = plt.subplots(1, n+1, figsize=(5*(n+1), 5))
    
    # Plot input matrices
    titles = ["Free Space Pathloss", "Ground Truth"] + [f"Matrix {i+3}" for i in range(n-2)]
    cmaps = ['viridis', 'plasma'] + ['inferno'] * (n-2)
    
    for i in range(n):
        im = axes[i].imshow(matrices[i], cmap=cmaps[i])
        axes[i].set_title(titles[i])
        fig.colorbar(im, ax=axes[i])
    
    # Plot difference matrix
    im_diff = axes[-1].imshow(diff, cmap='coolwarm')
    axes[-1].set_title("Diff")
    fig.colorbar(im_diff, ax=axes[-1])
    
    plt.tight_layout()
    plt.savefig(f"foo/{time()}.png")
    plt.close()
    
    return diff  # Return the diff matrix in case it's needed

