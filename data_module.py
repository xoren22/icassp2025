import os
import torch
import numpy as np
import pandas as pd
from typing import Tuple
from skimage.io import imread
from dataclasses import dataclass
from torch.utils.data import Dataset

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
    input_img: np.ndarray  
    output_img: np.ndarray
    radiation_pattern: np.array


class PathlossNormalizer:
    def __init__(
            self,
            min_antenna_gain=-55.0,
        ):
        # ImageNet stats for the first three channels
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.min_antenna_gain = min_antenna_gain

    def normalize_input(self, input_tensor):
        normalized = input_tensor.clone()
        for i in range(2):
            normalized[i] = (normalized[i] / 255.0 - self.mean[i]) / self.std[i]
        normalized[2] = np.log10(1 + normalized[2])
        normalized[3] = normalized[3] / self.min_antenna_gain
        normalized[4] = np.log10(normalized[4]) - 1.9  # "magic shift"
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
        
        input_img = imread(os.path.join(self.input_path, input_file))   # shape [H, W, channels]
        H, W = input_img.shape[:2]
        if not self.load_output:
            output_img = None
        else:
            output_img = imread(os.path.join(self.output_path, output_file))
        freq_MHz = self.freqs_MHz[int(f)-1]
        sampling_positions = pd.read_csv(os.path.join(self.positions_path, position_file))
        x_ant, y_ant, azimuth = sampling_positions.loc[int(sp), ["Y", "X", "Azimuth"]]
        radiation_pattern = np.genfromtxt(os.path.join(self.radiation_path, radiation_file), delimiter=',')

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

    def _normalize_size(self, sample: RadarSample) -> Tuple[RadarSample, np.ndarray]:
        scale_factor = min(self.target_size / sample.H, self.target_size / sample.W)
        new_h, new_w = int(sample.H * scale_factor), int(sample.W * scale_factor)

        new_size = (new_h, new_w)
        reflectance = sample.input_img[:,:,0]
        transmittance = sample.input_img[:,:,1]
        distance = sample.input_img[:,:,2]
        
        distance = resize_linear(distance, new_size)
        reflectance = resize_nearest(reflectance, new_size)
        transmittance = resize_nearest(transmittance, new_size)
        
        sample.x_ant *= scale_factor
        sample.y_ant *= scale_factor
        resized_input = np.stack([reflectance, transmittance, distance], axis=2)
        if sample.output_img is not None:
            sample.output_img = resize_db(sample.output_img, new_size)
        
        sample.input_img = np.zeros((self.target_size, self.target_size, 3), dtype=np.float32)
        sample.input_img[:new_h, :new_w, :] = resized_input
        

        if sample.output_img is not None:
            padded_output = np.zeros((self.target_size, self.target_size), dtype=np.float32)
            padded_output[:new_h, :new_w] = sample.output_img
            sample.output_img = padded_output
        
        mask = np.zeros((self.target_size, self.target_size), dtype=np.float32)
        mask[:new_h, :new_w] = 1
        
        sample.H = sample.W = self.target_size  # Update to full padded dimensions
        
        return sample, mask

    def __getitem__(self, idx):
        b, ant, f, sp = self.file_list[idx]
        sample = self.read_sample(b, ant, f, sp)
        sample, mask = self._normalize_size(sample)

        output_tensor = torch.from_numpy(sample.output_img.astype(np.float32))
        reflectance = torch.from_numpy(sample.input_img[:,:,0]).to(torch.float32)
        transmittance = torch.from_numpy(sample.input_img[:,:,1]).to(torch.float32)
        distance = torch.from_numpy(sample.input_img[:,:,2]).to(torch.float32)

        # calculating new features from them
        antenna_gain = calculate_antenna_gain(sample.radiation_pattern, sample.W, sample.H, sample.azimuth, sample.x_ant, sample.y_ant)
        free_space_pathloss = calculate_fspl(
            sample.W, sample.H, sample.x_ant, sample.y_ant, antenna_gain, sample.freq_MHz
        )

        transmittance_loss = calculate_transmittance_loss(transmittance.numpy(), sample.x_ant, sample.y_ant)
        fs_plus_transmittance_loss = free_space_pathloss + transmittance_loss

        # Build input tensor: [7, H, W]
        input_tensor = np.zeros((7, sample.H, sample.W), dtype=np.float32)
        input_tensor[0] = reflectance  # reflectance
        input_tensor[1] = transmittance # transmittance
        input_tensor[2] = distance # distance
        input_tensor[3] = antenna_gain
        input_tensor[4] = sample.freq_MHz  # constant freq map
        input_tensor[5] = fs_plus_transmittance_loss

        input_tensor = torch.from_numpy(input_tensor)
        input_tensor = self.normalizer.normalize_input(input_tensor)
        
        # Add mask to the last channel
        input_tensor[6] = torch.from_numpy(mask).to(torch.float32)

        return input_tensor, output_tensor, mask