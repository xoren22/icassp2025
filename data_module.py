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

    def free_space_pathloss(self, d_m, freq_hz):
        c = 3e8  # Speed of light in m/s
        wavelength = c / freq_hz
        near_field_threshold = wavelength / (2 * math.pi)
        d_far = max(d_m, near_field_threshold)
        fspl_db = 20.0 * math.log10(4.0 * math.pi * d_far * freq_hz / c)

        return fspl_db    
    
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
        self.freqs_Hz = [868, 1800, 3500]
        
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
        
        freq = self.freqs_Hz[int(f)-1]
        if self.img_size != input_img.shape[0]:
            input_img = cv2.resize(input_img, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            output_img = cv2.resize(output_img, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
        
        radiation_pattern = np.genfromtxt(os.path.join(self.radiation_path, radiation_file), delimiter=',', skip_header=1)

        x_grid = np.repeat(np.linspace(0, W-1, self.img_size), self.img_size, axis=0).reshape(self.img_size, self.img_size).transpose()
        y_grid = np.repeat(np.linspace(0, H-1, self.img_size), self.img_size, axis=0).reshape(self.img_size, self.img_size)
        
        angles = -(180/np.pi) * np.arctan2((y_ant - y_grid), (x_ant - x_grid)) + 180 + sampling_positions['Azimuth'].iloc[int(sp)]
        angles = np.clip(angles % radiation_pattern.shape[0], 0, radiation_pattern.shape[0]-1).astype(int)
        antenna_gain = radiation_pattern[angles]
        
        distance = np.sqrt(((x_ant - x_grid) * (W/self.img_size))**2 + ((y_ant - y_grid) * (H/self.img_size))**2)
        
        input_tensor = np.zeros((5, self.img_size, self.img_size), dtype=np.float32)
        input_tensor[0] = input_img[:, :, 0].astype(np.float32)  # Reflectance (raw uint8)
        input_tensor[1] = input_img[:, :, 1].astype(np.float32)  # Transmittance (raw uint8)
        input_tensor[2] = distance  # Distance
        input_tensor[3] = antenna_gain  # Raw antenna gain
        input_tensor[4] = np.full((self.img_size, self.img_size), freq)  # Frequency channel
        
        # Prepare ground truth pathloss
        output_tensor = output_img.astype(np.float32)
        
        # Convert to PyTorch tensors
        input_tensor = torch.from_numpy(input_tensor)
        output_tensor = torch.from_numpy(output_tensor).unsqueeze(0)  # Add channel dimension
        
        input_tensor = self.normalizer.normalize_input(input_tensor)
        if self.training:
            output_tensor = self.normalizer.normalize_output(output_tensor)

        return input_tensor, output_tensor

