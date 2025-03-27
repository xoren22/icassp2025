import torch
import os
from typing import Union, Dict

from utils import load_model
from featurizer import featurizer
from _types import RadarSampleInputs
from augmentations import normalize_size, resize_db
from data_module import read_sample, IMG_TARGET_SIZE, INITIAL_PIXEL_SIZE


class PathlossPredictor:
    def __init__(
        self,
        model=None,
        model_ckpt_path=None,
    ):
        if model is not None:
            self.model = model
        elif model_ckpt_path is not None:
            self.model = load_model(model_ckpt_path)
        else:
            raise ValueError("Either model or model_ckpt_path must be provided")
        self.model.eval()
    
    def predict(self, input_sample: Union[Dict, RadarSampleInputs]):
        sample = read_sample(input_sample)
        old_h, old_w = sample.H, sample.W
        sample = normalize_size(sample, IMG_TARGET_SIZE)
        
        mask = sample.mask
        scaling_factor = INITIAL_PIXEL_SIZE / sample.pixel_size
        norm_h, norm_w = int(old_h*scaling_factor), int(old_w*scaling_factor)
        
        input_tensor = featurizer(sample)
        with torch.no_grad():
            pred = self.model(input_tensor.unsqueeze(0)).squeeze(0)
        pred = pred[torch.where(mask == 1)].reshape((norm_h, norm_w))
        pred = resize_db(pred.unsqueeze(0), new_size=(old_h, old_w)).squeeze(0)

        return pred


if __name__ == "__main__":
    model_path = '/auto/home/xoren/icassp2025/models/best_model.pth'

    sample = {
        'freq_MHz': 868, 
        'ids': (1, 1, 1, 0),
        'sampling_position': 0, 
        'input_file': '/auto/home/xoren/icassp2025/data/Inputs/Task_2_ICASSP/B1_Ant1_f1_S0.png', 
        # 'output_file': '/auto/home/xoren/icassp2025/data/Outputs/Task_2_ICASSP/B1_Ant1_f1_S0.png', 
        'position_file': '/auto/home/xoren/icassp2025/data/Positions/Positions_B1_Ant1_f1.csv', 
        'radiation_pattern_file': '/auto/home/xoren/icassp2025/data/Radiation_Patterns/Ant1_Pattern.csv', 
    }

    model = PathlossPredictor(
        model_ckpt_path=model_path,
    )

    pred = model.predict(
        sample
    )

    print(pred.shape)