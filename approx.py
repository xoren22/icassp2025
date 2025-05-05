# approx.py

import os
import torch
import numpy as np
import pandas as pd
from numba import njit
from torchvision.io import read_image
from scipy.ndimage import gaussian_filter
from kaggle_eval import kaggle_async_eval
from dataclasses import dataclass, asdict
from typing import Union, Tuple, Optional, List


@dataclass
class RadarSample:
    H: int
    W: int
    x_ant: List[float]
    y_ant: List[float]
    azimuth: List[float]
    freq_MHz: List[float]
    input_img: torch.Tensor    # Format (C, H, W), includes transmittance channel
    output_img: torch.Tensor   # (H, W) or (1, H, W) ground-truth pathloss
    radiation_pattern: List[torch.Tensor]
    pixel_size: float = 0.25
    mask: Union[torch.Tensor, None] = None
    ids: Optional[List[Tuple[int, int, int, int]]] = None

@dataclass
class RadarSampleInputs:
    freq_MHz: float
    input_file: str
    output_file: Union[str, None]
    position_file: str
    radiation_pattern_file: str
    sampling_position: int
    id_prefix : object
    ids: Optional[Tuple[int, int, int, int]] = None

    def asdict(self):
        return asdict(self)

    def __post_init__(self):
        # Basic sanity checks
        if self.ids and not all(isinstance(i, int) for i in self.ids):
            raise ValueError("All IDs must be integers.")
        if not isinstance(self.freq_MHz, (int, float)):
            raise ValueError("freq_MHz must be numeric.")

        # Ensure file existence (comment out if you have some missing paths on purpose)
        for path_attr in ['input_file', 'position_file', 'radiation_pattern_file']:
            path = getattr(self, path_attr)
            if path and not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

def read_sample(inputs: Union[RadarSampleInputs, dict]) -> RadarSample:
    """
    Reads one sample from the provided file paths:
      - input_file (transmittance + possibly other channels)
      - output_file (ground-truth pathloss)
      - position_file (CSV with X, Y, Azimuth)
      - radiation_pattern_file
    """
    if isinstance(inputs, RadarSampleInputs):
        inputs = inputs.asdict()

    freq_MHz = inputs["freq_MHz"]
    input_file = inputs["input_file"]
    output_file = inputs.get("output_file")
    position_file = inputs["position_file"]
    radiation_pattern_file = inputs["radiation_pattern_file"]
    sampling_position = inputs["sampling_position"]
    ids = inputs.get("ids")

    input_img = read_image(input_file).float()[:2]  # Keep first 2 channels
    C, H, W = input_img.shape

    output_img = None
    if output_file:
        output_img = read_image(output_file).float()
        if output_img.dim() == 3 and output_img.size(0) == 1:
            output_img = output_img.squeeze(0)  # (1,H,W) => (H,W)

    # Position CSV
    df_positions = pd.read_csv(position_file)
    x_ant, y_ant, azimuth = df_positions.loc[sampling_position, ["Y", "X", "Azimuth"]]

    # Radiation pattern
    radiation_pattern_np = np.genfromtxt(radiation_pattern_file, delimiter=',')
    radiation_pattern = torch.from_numpy(radiation_pattern_np).float()

    # Build the RadarSample
    sample = RadarSample(
        H=H,
        W=W,
        x_ant=[x_ant],
        y_ant=[y_ant],
        azimuth=[azimuth],
        freq_MHz=[freq_MHz],
        input_img=input_img,
        output_img=output_img,
        radiation_pattern=[radiation_pattern],
        pixel_size=0.25,
        mask=torch.ones((H, W)),
        ids=ids,
    )
    return sample


@njit
def _calculate_transmittance_loss_numpy(
    transmittance_matrix,
    x_ant,
    y_ant,
    n_angles=360*128,
    radial_step=1.0,
    max_walls=10
):
    """
    Numba-accelerated function that casts 'n_angles' rays from (x_ant, y_ant).
    On each crossing from positive->zero in transmittance_matrix, we add path-loss
    to sum_loss. If sum_loss exceeds 160, we clip to 160 and stop the ray.
    """
    h, w = transmittance_matrix.shape
    output  = np.zeros((h, w), dtype=np.float32)
    counts  = np.zeros((h, w), dtype=np.float32)

    dtheta = 2.0 * np.pi / n_angles
    max_dist = np.sqrt(w*w + h*h)
    cos_vals = np.cos(np.arange(n_angles) * dtheta)
    sin_vals = np.sin(np.arange(n_angles) * dtheta)

    for i in range(int(n_angles)):
        cos_t = cos_vals[i]
        sin_t = sin_vals[i]
        sum_loss  = 0.0
        last_val  = None
        wall_count = 0
        r = 0.0

        while r <= max_dist:
            x = x_ant + r * cos_t
            y = y_ant + r * sin_t

            px = int(round(x))
            py = int(round(y))

            # Out of bounds => optionally add last_val, then break
            if px < 0 or px >= w or py < 0 or py >= h:
                if last_val is not None and last_val > 0:
                    sum_loss += last_val
                    # Check for 160 limit
                    if sum_loss > 160:
                        sum_loss = 160
                break

            val = transmittance_matrix[py, px]
            if last_val is None:
                last_val = val

            # Detect crossing from positive->zero => add last_val
            if val != last_val:
                if last_val > 0 and val == 0:
                    sum_loss += last_val
                    # If exceeding 160, stop the ray
                    if sum_loss > 160:
                        sum_loss = 160
                        break
                    wall_count += 1
                    if wall_count >= max_walls:
                        # fill remainder with sum_loss
                        r_temp = r
                        while r_temp <= max_dist:
                            x_temp = x_ant + r_temp * cos_t
                            y_temp = y_ant + r_temp * sin_t
                            px_temp = int(round(x_temp))
                            py_temp = int(round(y_temp))
                            if px_temp < 0 or px_temp >= w or py_temp < 0 or py_temp >= h:
                                break
                            # average sum_loss into that pixel
                            if counts[py_temp, px_temp] == 0:
                                output[py_temp, px_temp] = sum_loss
                                counts[py_temp, px_temp] = 1
                            else:
                                old_val = output[py_temp, px_temp]
                                old_count = counts[py_temp, px_temp]
                                output[py_temp, px_temp] = (old_val*old_count + sum_loss) / (old_count+1)
                                counts[py_temp, px_temp] += 1
                            r_temp += radial_step
                        break
                last_val = val

            # Average current sum_loss into (px, py)
            if counts[py, px] == 0:
                output[py, px] = sum_loss
                counts[py, px] = 1
            else:
                old_val = output[py, px]
                old_count = counts[py, px]
                output[py, px] = (old_val*old_count + sum_loss) / (old_count+1)
                counts[py, px] += 1

            if wall_count >= max_walls or sum_loss > 160:
                # Check for 160 limit
                if sum_loss > 160:
                    sum_loss = 160
                break

            r += radial_step

    return output

def calculate_transmittance_loss(
    transmittance_matrix: torch.Tensor,
    x_ant: float,
    y_ant: float,
    n_angles=360*128,
    radial_step=1.0,
    max_walls=10
) -> torch.Tensor:
    trans_np = transmittance_matrix.cpu().numpy().astype(np.float32)
    output_np = _calculate_transmittance_loss_numpy(
        trans_np, x_ant, y_ant, n_angles, radial_step, max_walls
    )

    np.clip(output_np, 0.0, 160.0, out=output_np)

    return torch.from_numpy(output_np)

def calculate_distance(x_ant, y_ant, H, W, pixel_size):
    """
    Returns a (H,W) tensor with the distance from (x_ant, y_ant) to each pixel center.
    """
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing='ij'
    )
    dist = torch.sqrt((x_grid - x_ant)**2 + (y_grid - y_ant)**2)
    dist_m = dist * pixel_size
    return dist_m

def calculate_fspl(
    dist_m: torch.Tensor,
    freq_MHz: float,
    antenna_gain: torch.Tensor,
    min_dist_m: float = 0.125
):
    """
    FSPL(dB) = 20 log10(dist) + 20 log10(freq) - 27.55, minus antenna_gain.
    """
    dist_clamped = torch.clamp(dist_m, min=min_dist_m)
    freq_tensor = torch.tensor(freq_MHz, dtype=torch.float32)
    fspl_db = 20.0 * torch.log10(dist_clamped) + 20.0 * torch.log10(freq_tensor) - 27.55
    return fspl_db - antenna_gain


class Approx:
    def approximate(self, sample : RadarSample) -> torch.Tensor:
        trans = sample.input_img[1]  # second channel = transmittance
        dist = calculate_distance(sample.x_ant[0], sample.y_ant[0], sample.H, sample.W, sample.pixel_size)
        trans_loss = calculate_transmittance_loss(trans, sample.x_ant[0], sample.y_ant[0])
        trans_loss_smooth = torch.from_numpy(gaussian_filter(trans_loss.numpy(), sigma=5.0, mode='reflect'))

        fspl = calculate_fspl(
            dist_m=dist,
            freq_MHz=sample.freq_MHz[0],
            antenna_gain=torch.zeros_like(dist)
        )
        approx = trans_loss_smooth + fspl
        approx = torch.round(approx-0.5)

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