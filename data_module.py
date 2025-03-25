import os
import torch
import numpy as np
import pandas as pd
from typing import Tuple
from torch.utils.data import Dataset
from torchvision.io import read_image

from _types import RadarSample
from augmentations import normalize_size
from featurizer import calculate_antenna_gain, calculate_fspl, calculate_transmittance_loss


class PathlossNormalizer:
    def __init__(
            self,
            min_antenna_gain=-55.0,
        ):
        # ImageNet stats for the first three channels
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        self.min_antenna_gain = min_antenna_gain

    def normalize_input(self, input_tensor):
        normalized = input_tensor.clone()
        for i in range(2):
            normalized[i] = (normalized[i] / 255.0 - self.mean[i]) / self.std[i]
        normalized[2] = torch.log10(1 + normalized[2])
        normalized[3] = normalized[3] / self.min_antenna_gain
        normalized[4] = torch.log10(normalized[4]) - 1.9  # "magic shift"
        normalized[5] = normalized[5] / 100.0 # this feature mask has values < 300

        return normalized


class PathlossDataset(Dataset):
    def __init__(self, 
        file_list, 
        input_path, 
        positions_path, 
        buildings_path, 
        radiation_path, 
        output_path=None, 
        img_size=640, 
        initial_pixel_size=0.25, 
        training=False, load_output=True, augmentations=None,
    ):

        self.training = training
        self.file_list = file_list
        self.input_path = input_path
        self.load_output = load_output
        self.output_path = output_path
        self.positions_path = positions_path
        self.buildings_path = buildings_path
        self.radiation_path = radiation_path
        self.augmentations = augmentations
        self.initial_pixel_size = initial_pixel_size

        self.normalizer = PathlossNormalizer()
        
        self.target_size = img_size
        self.freqs_MHz = [868, 1800, 3500]


    def __len__(self):
        return len(self.file_list)

    def idx_to_ids(self, idx):
        return self.file_list[idx]

    def read_sample(self, b, ant, f, sp):
        input_file = f"B{b}_Ant{ant}_f{f}_S{sp}.png"
        output_file = f"B{b}_Ant{ant}_f{f}_S{sp}.png"
        radiation_file = f"Ant{ant}_Pattern.csv"
        position_file = f"Positions_B{b}_Ant{ant}_f{f}.csv"
        
        input_img = read_image(os.path.join(self.input_path, input_file)).float()
        C, H, W = input_img.shape
        
        if not self.load_output:
            output_img = None
        else:
            output_img = read_image(os.path.join(self.output_path, output_file)).float()
            if output_img.size(0) == 1:  # If single channel, remove channel dimension
                output_img = output_img.squeeze(0)
            
        freq_MHz = self.freqs_MHz[int(f)-1]
        sampling_positions = pd.read_csv(os.path.join(self.positions_path, position_file))
        x_ant, y_ant, azimuth = sampling_positions.loc[int(sp), ["Y", "X", "Azimuth"]]
        
        radiation_pattern_np = np.genfromtxt(os.path.join(self.radiation_path, radiation_file), delimiter=',')
        radiation_pattern = torch.from_numpy(radiation_pattern_np).float()

        sample = RadarSample(
            H=H,
            W=W,
            x_ant=x_ant,
            y_ant=y_ant,
            azimuth=azimuth,
            freq_MHz=freq_MHz,
            input_img=input_img,
            output_img=output_img,
            radiation_pattern=radiation_pattern,
            pixel_size=self.initial_pixel_size,
            mask=torch.ones_like(output_img),
        )
        
        return sample
    

    def __getitem__(self, idx):
        b, ant, f, sp = self.file_list[idx]
        sample = self.read_sample(b, ant, f, sp)
        sample = normalize_size(sample=sample, target_size=self.target_size)
        if self.augmentations is not None:
            sample = self.augmentations(sample)

        output_tensor = sample.output_img if sample.output_img is not None else None
        
        reflectance = sample.input_img[0]  # First channel
        transmittance = sample.input_img[1]  # Second channel
        distance = sample.input_img[2]  # Third channel

        radiation_pattern = sample.radiation_pattern
        
        antenna_gain = calculate_antenna_gain(
            radiation_pattern, 
            sample.W, 
            sample.H, 
            sample.azimuth, 
            sample.x_ant, 
            sample.y_ant
        )
        
        # Calculate free space path loss on CPU
        free_space_pathloss = calculate_fspl(
            dist_m=distance,
            freq_MHz=sample.freq_MHz,
            antenna_gain=antenna_gain, 
        )

        # Calculate transmittance loss on CPU
        transmittance_loss = calculate_transmittance_loss(
            transmittance, 
            sample.x_ant, 
            sample.y_ant
        )
        
        fs_plus_transmittance_loss = free_space_pathloss + transmittance_loss

        # Build input tensor: [7, H, W] - Already in correct (C, H, W) format
        input_tensor = torch.zeros((7, sample.H, sample.W), dtype=torch.float32, device=torch.device('cpu'))
        input_tensor[0] = reflectance  # reflectance
        input_tensor[1] = transmittance  # transmittance
        input_tensor[2] = distance  # distance
        input_tensor[3] = antenna_gain
        input_tensor[4] = torch.full((sample.H, sample.W), sample.freq_MHz, dtype=torch.float32, device=torch.device('cpu'))
        input_tensor[5] = fs_plus_transmittance_loss

        # Normalize the input tensor
        input_tensor = self.normalizer.normalize_input(input_tensor)
        
        # Add mask to the last channel
        mask = sample.mask
        input_tensor[6] = mask
        return input_tensor, output_tensor, mask   
