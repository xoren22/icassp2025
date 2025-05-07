# approx.py

import os
import math
import torch
import numpy as np
import pandas as pd
from numba import njit
from torchvision.io import read_image
from scipy.ndimage import gaussian_filter
from kaggle_eval import kaggle_async_eval
from dataclasses import dataclass, asdict
from typing import Union, Tuple, Optional, List


IMG_TARGET_SIZE = 640
INITIAL_PIXEL_SIZE = 0.25

@dataclass
class RadarSample:
    H: int
    W: int
    x_ant: float
    y_ant: float
    freq_MHz: float
    reflectance: torch.Tensor  # In format (H, W)
    transmittance: torch.Tensor  # In format (H, W)
    output_img: torch.Tensor  # In format (H, W)
    pixel_size: float = 0.25
    mask: Union[torch.Tensor, None] = None
    ids: Optional[List[Tuple[int, int, int, int]]] = None

    def copy(self):
        return RadarSample(
                    H=self.H,
                    W=self.W,
                    x_ant=self.x_ant,
                    y_ant=self.y_ant,
                    freq_MHz=self.freq_MHz,
                    reflectance=self.reflectance,  
                    transmittance=self.transmittance,  
                    output_img=self.output_img, 
                    pixel_size=self.pixel_size,
                    mask=self.mask,
                    ids=self.ids,
                )

@dataclass
class RadarSampleInputs:
    freq_MHz: float
    input_file: str
    output_file: Union[str, None]
    position_file: str
    sampling_position : int
    ids: Optional[Tuple[int, int, int, int]] = None

    def asdict(self):
        return asdict(self)
    
    def __post_init__(self):
        if self.ids and not all(isinstance(i, int) for i in self.ids):
            raise ValueError("All IDs must be integers")
        
        if not isinstance(self.freq_MHz, (int, float)):
            raise ValueError("freq_MHz must be a number")
        
        for path_attr in ['input_file', 'position_file']:
            path = getattr(self, path_attr)
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
    
 
 
def read_sample(inputs: Union[RadarSampleInputs, dict]):
    if isinstance(inputs, RadarSampleInputs):
        inputs = inputs.asdict()

    freq_MHz = inputs["freq_MHz"]
    input_file = inputs["input_file"]
    output_file = inputs.get("output_file")
    position_file = inputs["position_file"]
    sampling_position = inputs["sampling_position"]
    
    input_img = read_image(input_file).float()
    C, H, W = input_img.shape
    reflectance = input_img[0:1]
    transmittance = input_img[1:2]
    
    output_img = None
    if output_file:
        output_img = read_image(output_file).float()
        if output_img.size(0) == 1:  # If single channel, remove channel dimension
            output_img = output_img.squeeze(0)
        
    sampling_positions = pd.read_csv(position_file)
    x_ant, y_ant = sampling_positions.loc[int(sampling_position), ["Y", "X"]]
    
    sample = RadarSample(
        H=H,
        W=W,
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


@njit
def _calculate_transmittance_loss_numpy(transmittance_matrix, x_ant, y_ant, n_angles, radial_step, max_walls):
    h, w = transmittance_matrix.shape
    output = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    dtheta = 2.0 * np.pi / n_angles
    cos_vals = np.cos(np.arange(n_angles) * dtheta)
    sin_vals = np.sin(np.arange(n_angles) * dtheta)
    max_dist = np.sqrt(w*w + h*h)

    for i in range(n_angles):
        cos_t = cos_vals[i]
        sin_t = sin_vals[i]
        sum_loss = 0.0
        last_val = None
        wall_count = 0
        r = 0.0

        while r <= max_dist:
            x = x_ant + r * cos_t
            y = y_ant + r * sin_t
            px = int(round(x))
            py = int(round(y))

            if px < 0 or px >= w or py < 0 or py >= h:
                if last_val is None:          # antenna still outside → keep marching
                    r += radial_step
                    continue
                if last_val > 0:              # we were inside, add last wall
                    sum_loss += last_val
                    wall_count += 1
                break

            val = transmittance_matrix[py, px]
            if last_val is None:
                last_val = val

            if val != last_val:
                if last_val > 0 and val == 0:
                    sum_loss += last_val
                    wall_count += 1
                    if wall_count >= max_walls:
                        r_temp = r
                        while r_temp <= max_dist:
                            x_temp = x_ant + r_temp * cos_t
                            y_temp = y_ant + r_temp * sin_t
                            px_temp = int(round(x_temp))
                            py_temp = int(round(y_temp))
                            if px_temp < 0 or px_temp >= w or py_temp < 0 or py_temp >= h:
                                break
                            n = counts[py_temp, px_temp]                       # averaging
                            output[py_temp, px_temp] = (output[py_temp, px_temp] * n + sum_loss) / (n + 1)
                            counts[py_temp, px_temp] = n + 1
                            r_temp += radial_step
                        break
                last_val = val

            n = counts[py, px]                                               # averaging
            output[py, px] = (output[py, px] * n + sum_loss) / (n + 1)
            counts[py, px] = n + 1

            if wall_count >= max_walls:
                break

            r += radial_step

    return output


def calculate_transmittance_loss(transmittance_matrix, x_ant, y_ant, n_angles=360*128, radial_step=1.0, max_walls=10, smooth=True):
    transmittance_np = transmittance_matrix.cpu().numpy()
    output_np = _calculate_transmittance_loss_numpy(transmittance_np, x_ant, y_ant, n_angles, radial_step, max_walls)
    if smooth:
        sigma_px = 5.0 
        output_np = gaussian_filter(output_np, sigma=sigma_px, mode='reflect')
    return torch.from_numpy(output_np).to(device=torch.device('cpu'))


def calculate_fspl(
    dist_m,               # distance in meters (torch tensor)
    freq_MHz,             # frequency in MHz
    antenna_gain,         # shape=(360,) antenna gain in dBi [0..359]
    min_dist_m=0.125,     # clamp distance below this
):
    dist_clamped = torch.clamp(dist_m, min=min_dist_m)
    freq_tensor = torch.tensor(freq_MHz, device=torch.device('cpu'))
    fspl_db = 20.0 * torch.log10(dist_clamped) + 20.0 * torch.log10(freq_tensor) - 27.55
    pathloss_db = fspl_db - antenna_gain
    
    return pathloss_db


def calculate_distance(x_ant, y_ant, H, W, pixel_size):
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing='ij'
    )
    return torch.sqrt((x_grid - x_ant)**2 + (y_grid - y_ant)**2) * pixel_size


class Approx:
    def approximate(self, sample : RadarSample) -> torch.Tensor:
        trans = sample.transmittance  # second channel = transmittance
        dist = calculate_distance(sample.x_ant, sample.y_ant, sample.H, sample.W, sample.pixel_size)
        trans_loss = calculate_transmittance_loss(trans, sample.x_ant, sample.y_ant, smooth=True)

        fspl = calculate_fspl(
            dist_m=dist,
            freq_MHz=sample.freq_MHz,
            antenna_gain=torch.zeros_like(dist)
        )
        approx = trans_loss + fspl
        approx = torch.floor(approx)          # round down
        approx = torch.clamp(approx, max=160.0)   # global clip

        return approx
    
    def predict(self, samples):
        samples = [read_sample(s) for s in samples]
        predictions = [self.approximate(s) for s in samples]

        return predictions

BASE_DIR = os.path.dirname(__file__)

kaggle_async_eval(
    epoch=1,
    model=Approx(),
    csv_save_path=f"{BASE_DIR}/../Desktop/Task1.csv",
    submission_message="Preds",
    competition='iprm-task-1',
)