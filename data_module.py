import os
import torch
import numpy as np
import pandas as pd
from typing import Union
from torch.utils.data import Dataset
from torchvision.io import read_image

from featurizer import featurizer
from augmentations import normalize_size
from _types import RadarSample, RadarSampleInputs


class PathlossDataset(Dataset):
    def __init__(self, 
        inputs_list, 
        img_size=640, 
        initial_pixel_size=0.25, 
        training=False, load_output=True, augmentations=None,
    ):

        self.training = training
        self.inputs_list = inputs_list
        self.load_output = load_output
        self.augmentations = augmentations
        self.initial_pixel_size = initial_pixel_size

        self.featurizer = featurizer
    
        self.target_size = img_size


    def __len__(self):
        return len(self.inputs_list)

    def read_sample(self, inputs: Union[RadarSampleInputs, dict]):
        if isinstance(inputs, RadarSampleInputs):
            inputs = inputs.asdict()

        freq_MHz = inputs["freq_MHz"]
        input_file = inputs["input_file"]
        output_file = inputs["output_file"]
        position_file = inputs["position_file"]
        sampling_position = inputs["sampling_position"]
        radiation_pattern_file = inputs["radiation_pattern_file"]
        
        input_img = read_image(input_file).float()
        C, H, W = input_img.shape
        
        if not self.load_output:
            output_img = None
        else:
            output_img = read_image(output_file).float()
            if output_img.size(0) == 1:  # If single channel, remove channel dimension
                output_img = output_img.squeeze(0)
            
        sampling_positions = pd.read_csv(position_file)
        x_ant, y_ant, azimuth = sampling_positions.loc[int(sampling_position), ["Y", "X", "Azimuth"]]
        
        radiation_pattern_np = np.genfromtxt(radiation_pattern_file, delimiter=',')
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
        inp = self.inputs_list[idx]
        sample = self.read_sample(inp)
        sample = normalize_size(sample=sample, target_size=self.target_size)
        if self.augmentations is not None:
            sample = self.augmentations(sample)

        output_tensor = sample.output_img if sample.output_img is not None else None
        input_tensor = self.featurizer(sample=sample)
        mask = sample.mask

        return input_tensor, output_tensor, mask   
