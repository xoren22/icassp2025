import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image

from featurizer import featurize_inputs
from augmentations import normalize_size
from _types import RadarSample, RadarSampleInputs

INITIAL_PIXEL_SIZE = 0.25
IMG_TARGET_SIZE = 640

def pad_sample(sample: RadarSample) -> RadarSample:
    C, H, W = sample.input_img.shape
    x_ant, y_ant = sample.x_ant, sample.y_ant
    
    pad_left   = int(max(0, -x_ant))
    pad_right  = int(max(0, x_ant - (W - 1)))
    pad_top    = int(max(0, -y_ant))
    pad_bottom = int(max(0, y_ant - (H - 1)))

    if not any([pad_left, pad_right, pad_top, pad_bottom]):
        return sample

    sample.input_img = F.pad(
        sample.input_img.unsqueeze(0),  # (C, H, W) -> (1, C, H, W)
        (pad_left, pad_right, pad_top, pad_bottom),
        value=0
    ).squeeze(0)  # -> (C, new_H, new_W)

    if sample.output_img is not None:
        sample.output_img = F.pad(
            sample.output_img.unsqueeze(0),  # (H, W) or (C, H, W)
            (pad_left, pad_right, pad_top, pad_bottom),
            value=0
        ).squeeze(0)

    sample.mask = F.pad(
        sample.mask.unsqueeze(0),  # (H, W) -> (1, H, W)
        (pad_left, pad_right, pad_top, pad_bottom),
        value=0
    ).squeeze(0)  # (new_H, new_W)
    
    sample.x_ant += pad_left
    sample.y_ant += pad_top
    _, new_H, new_W = sample.input_img.shape
    sample.H, sample.W = new_H, new_W
    return sample


def read_sample(inputs: Union[RadarSampleInputs, dict]):
    if isinstance(inputs, RadarSampleInputs):
        inputs = inputs.asdict()

    ids = inputs["ids"]
    freq_MHz = inputs["freq_MHz"]
    input_file = inputs["input_file"]
    output_file = inputs.get("output_file")
    position_file = inputs["position_file"]
    sampling_position = inputs["sampling_position"]
    radiation_pattern_file = inputs["radiation_pattern_file"]

    input_img = read_image(input_file).float()  # shape: (C, H, W)
    C, H, W = input_img.shape

    output_img = None
    if output_file:
        output_img = read_image(output_file).float()
        if output_img.size(0) == 1:
            output_img = output_img.squeeze(0)
    
    sampling_positions = pd.read_csv(position_file)
    x_ant, y_ant, azimuth = sampling_positions.loc[
        int(sampling_position), ["Y", "X", "Azimuth"]
    ]

    radiation_pattern_np = np.genfromtxt(radiation_pattern_file, delimiter=",")
    radiation_pattern = torch.from_numpy(radiation_pattern_np).float()

    mask = torch.ones((H, W), dtype=torch.float32)

    sample = RadarSample(
        H=H,
        W=W,
        x_ant=x_ant,
        y_ant=y_ant,
        azimuth=azimuth,
        freq_MHz=freq_MHz,
        input_img=input_img,
        output_img=output_img,
        pixel_size=0.25,
        mask=mask,
        ids=ids,
        radiation_pattern=radiation_pattern,
    )

    # Ensure the antenna is within bounds
    sample = pad_sample(sample)

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
            sample = read_sample(inp)
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
