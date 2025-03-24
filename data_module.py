import os
import torch
import numpy as np
import pandas as pd
from typing import Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset
from torchvision.io import read_image

from featurizer import *
from augmentations import resize_db, resize_nearest, resize_linear


@dataclass
class RadarSample:
    H: int
    W: int
    x_ant: float
    y_ant: float
    azimuth: float
    freq_MHz: float
    input_img: torch.Tensor  # In format (C, H, W)
    output_img: torch.Tensor  # In format (H, W) or (1, H, W)
    radiation_pattern: torch.Tensor


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
    def __init__(self, file_list, input_path, positions_path, buildings_path, radiation_path, output_path=None, img_size=640, training=False, load_output=True):
        self.training = training
        self.file_list = file_list
        self.input_path = input_path
        self.load_output = load_output
        self.output_path = output_path
        self.positions_path = positions_path
        self.buildings_path = buildings_path
        self.radiation_path = radiation_path
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
        
        # Read input image directly as torch tensor (C, H, W)
        input_img = read_image(os.path.join(self.input_path, input_file)).to(torch.float32)
        C, H, W = input_img.shape
        
        if not self.load_output:
            output_img = None
        else:
            # Read output image directly as torch tensor
            output_img = read_image(os.path.join(self.output_path, output_file)).to(torch.float32)
            if output_img.size(0) == 1:  # If single channel, remove channel dimension
                output_img = output_img.squeeze(0)
            
        freq_MHz = self.freqs_MHz[int(f)-1]
        sampling_positions = pd.read_csv(os.path.join(self.positions_path, position_file))
        x_ant, y_ant, azimuth = sampling_positions.loc[int(sp), ["Y", "X", "Azimuth"]]
        
        # Convert CSV data to torch tensor
        # We could alternatively use pandas directly to torch.tensor if desired
        radiation_pattern_np = np.genfromtxt(os.path.join(self.radiation_path, radiation_file), delimiter=',')
        radiation_pattern = torch.from_numpy(radiation_pattern_np).to(torch.float32)

        sample = RadarSample(
            H,
            W,
            x_ant,
            y_ant,
            azimuth,
            freq_MHz,
            input_img,
            output_img,
            radiation_pattern,
        )
        
        return sample

    def _normalize_size(self, sample: RadarSample) -> Tuple[RadarSample, torch.Tensor]:
        C, H, W = sample.input_img.shape
        scale_factor = min(self.target_size / H, self.target_size / W)
        new_h, new_w = int(H * scale_factor), int(W * scale_factor)
        new_size = (new_h, new_w)
        
        reflectance = sample.input_img[0:1]  # First channel with dimension [1, H, W]
        transmittance = sample.input_img[1:2]  # Second channel with dimension [1, H, W]
        distance = sample.input_img[2:3]  # Third channel with dimension [1, H, W]
        
        reflectance_resized = resize_nearest(reflectance, new_size)
        transmittance_resized = resize_nearest(transmittance, new_size)
        distance_resized = resize_linear(distance, new_size)
        
        sample.x_ant *= scale_factor
        sample.y_ant *= scale_factor
        
        resized_input = torch.cat([reflectance_resized, transmittance_resized, distance_resized], dim=0)
        
        if sample.output_img is not None:
            sample.output_img = resize_db(sample.output_img.unsqueeze(0), new_size).squeeze(0)

        sample.input_img = torch.zeros((C, self.target_size, self.target_size), dtype=torch.float32)
        sample.input_img[:, :new_h, :new_w] = resized_input
        
        if sample.output_img is not None:
            padded_output = torch.zeros((self.target_size, self.target_size), dtype=torch.float32)
            padded_output[:new_h, :new_w] = sample.output_img
            sample.output_img = padded_output
        
        mask = torch.zeros((self.target_size, self.target_size), dtype=torch.float32)
        mask[:new_h, :new_w] = 1

        sample.H = sample.W = self.target_size
        
        return sample, mask

    def __getitem__(self, idx):
        b, ant, f, sp = self.file_list[idx]
        sample = self.read_sample(b, ant, f, sp)
        sample, mask = self._normalize_size(sample)

        output_tensor = sample.output_img if sample.output_img is not None else None
        
        reflectance = sample.input_img[0]  # First channel
        transmittance = sample.input_img[1]  # Second channel
        distance = sample.input_img[2]  # Third channel

        antenna_gain = calculate_antenna_gain(
            sample.radiation_pattern, 
            sample.W, 
            sample.H, 
            sample.azimuth, 
            sample.x_ant, 
            sample.y_ant
        )
        
        free_space_pathloss = calculate_fspl(
            sample.W, 
            sample.H, 
            sample.x_ant, 
            sample.y_ant, 
            antenna_gain, 
            sample.freq_MHz
        )

        transmittance_loss = calculate_transmittance_loss(
            transmittance, 
            sample.x_ant, 
            sample.y_ant
        )
        
        fs_plus_transmittance_loss = free_space_pathloss + transmittance_loss

        # Build input tensor: [7, H, W] - Already in correct (C, H, W) format
        input_tensor = torch.zeros((7, sample.H, sample.W), dtype=torch.float32)
        input_tensor[0] = reflectance  # reflectance
        input_tensor[1] = transmittance  # transmittance
        input_tensor[2] = distance  # distance
        input_tensor[3] = antenna_gain
        input_tensor[4] = torch.full((sample.H, sample.W), sample.freq_MHz, dtype=torch.float32)  # constant freq map
        input_tensor[5] = fs_plus_transmittance_loss

        # Normalize the input tensor
        input_tensor = self.normalizer.normalize_input(input_tensor)
        
        # Add mask to the last channel
        input_tensor[6] = mask

        return input_tensor, output_tensor, mask