import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union
from torch.utils.data import Dataset
from torchvision.io import read_image

from featurizer import featurize_inputs
from augmentations import normalize_size
from _types import RadarSample, RadarSampleInputs

INITIAL_PIXEL_SIZE = 0.25
IMG_TARGET_SIZE = 640

def read_sample(inputs: Union[RadarSampleInputs, dict]):
    if isinstance(inputs, RadarSampleInputs):
        inputs = inputs.asdict()

    ids = inputs["ids"]
    freq_MHz = inputs["freq_MHz"]
    input_file = inputs["input_file"]
    output_file = inputs.get("output_file")
    position_file = inputs["position_file"]
    sampling_position = inputs["sampling_position"]
    
    input_img = read_image(input_file).float()
    C, H, W = input_img.shape
    reflectance = input_img[0]
    transmittance = input_img[1]
    
    output_img = None
    if output_file:
        output_img = read_image(output_file).float()
        if output_img.size(0) == 1:  # If single channel, remove channel dimension
            output_img = output_img.squeeze(0)
        
    sampling_positions = pd.read_csv(position_file)
    x_ant, y_ant = sampling_positions.loc[int(sampling_position), ["Y", "X"]]

    if x_ant < 0 or x_ant >= W or y_ant < 0 or y_ant >= H:
        b, ant, f, sp = ids
        print(f"Antenna out of the bouds for B{b}_Ant{ant}_f{f}_S{sp}.png -- Skipping")
        return
    
    sample = RadarSample(
        H=H,
        W=W,
        ids=ids,
        x_ant=x_ant,
        y_ant=y_ant,
        freq_MHz=freq_MHz,
        reflectance=reflectance,
        transmittance=transmittance,
        output_img=output_img,
        pixel_size=INITIAL_PIXEL_SIZE,
        mask=torch.ones((H, W)),
    )
    
    return sample



class PathlossDataset(Dataset):
    def __init__(self, 
        inputs_list,  
        training=False, load_output=True, augmentations=None,
    ):

        self.training = training
        self.inputs_list = inputs_list
        self.load_output = load_output
        self.augmentations = augmentations

        self.target_size = IMG_TARGET_SIZE
        self.samples = self._preprocess_samples(self.inputs_list)
   
    def _preprocess_samples(self, inputs_list):
        samples = []
        for inp in tqdm(inputs_list, "Preprocessing samples: "):
            if sample := read_sample(inp):
                sample = normalize_size(sample=sample, target_size=self.target_size)
                samples.append(sample)
        return samples    


    def __len__(self):
        return len(self.inputs_list)


    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.augmentations is not None:
            sample = self.augmentations(sample)

        output_tensor = sample.output_img if sample.output_img is not None else None
        input_tensor = featurize_inputs(sample=sample)
        mask = sample.mask

        return input_tensor, output_tensor, mask   
