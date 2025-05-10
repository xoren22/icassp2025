import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import njit, prange
from torchvision.io import read_image
import matplotlib.pyplot as plt
 
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass, asdict
from typing import Union, Tuple, Optional, List

from kaggle_eval import kaggle_async_eval


IMG_TARGET_SIZE = 640
INITIAL_PIXEL_SIZE = 0.25


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
    pixel_size: float = 0.25
    mask: Union[torch.Tensor, None] = None
    ids: Optional[List[Tuple[int, int, int, int]]] = None

    def copy(self):
        return RadarSample(
                    self.H,
                    self.W,
                    self.x_ant,
                    self.y_ant,
                    self.azimuth,
                    self.freq_MHz,
                    self.input_img,  
                    self.output_img, 
                    self.pixel_size,
                    self.mask,
                    self.ids,
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
    
    output_img = None
    if output_file:
        output_img = read_image(output_file).float()
        if output_img.size(0) == 1:  # If single channel, remove channel dimension
            output_img = output_img.squeeze(0)
        
    sampling_positions = pd.read_csv(position_file)
    x_ant, y_ant, azimuth = sampling_positions.loc[int(sampling_position), ["Y", "X", "Azimuth"]]
    
    sample = RadarSample(
        H=H,
        W=W,
        x_ant=x_ant,
        y_ant=y_ant,
        azimuth=azimuth,
        freq_MHz=freq_MHz,
        input_img=input_img,
        output_img=output_img,
        pixel_size=INITIAL_PIXEL_SIZE,
        mask=torch.ones((H, W)),
    )

    if 0 > sample.x_ant >= sample.W or 0 > sample.y_ant >= sample.H:
        print(f"Warning: antenna coords out of range. (x_ant={sample.x_ant}, y_ant={sample.y_ant}), (W={sample.W}, H={sample.H}) -> clamping to valid range.")
    
    return sample


def calculate_fspl(
    dist_m,               # distance in meters (torch tensor)
    freq_MHz,             # frequency in MHz
    min_dist_m=0.125,     # clamp distance below this
):
    dist_clamped = np.maximum(dist_m, min_dist_m)
    fspl_db = 20.0 * np.log10(dist_clamped) + 20.0 * np.log10(freq_MHz) - 27.55

    return fspl_db



@njit(parallel=True, fastmath=True, nogil=True, boundscheck=False)
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

            if px < 0 or px >= w or py < 0 or py >= h:
                # antenna still outside → step forward until we hit the map
                if last_val is None:
                    r += radial_step
                    continue
                # already inside → leave as before
                if last_val > 0:
                    sum_loss += last_val
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



@njit(parallel=True, fastmath=True)
def _calculate_hybrid_loss_mc_numpy_fast(
    reflectance_matrix: np.ndarray,
    transmittance_matrix: np.ndarray,
    x_ant: float,
    y_ant: float,
    n_angles: int = 360*64,
    radial_step: float = 1.0,
    max_reflect: int = 5,
    max_transmit: int = 10,
    reflection_prob: float = 0.5,
    samples_per_angle: int = 8,
    max_loss: float = 160.0
) -> np.ndarray:
    h, w = reflectance_matrix.shape
    # initialize to max_loss
    output = np.full((h, w), max_loss, dtype=np.float32)

    # precompute dirs
    two_pi = 2.0 * np.pi
    dtheta = two_pi / n_angles
    cos_vals = np.cos(np.arange(n_angles) * dtheta)
    sin_vals = np.sin(np.arange(n_angles) * dtheta)
    max_dist = np.hypot(w, h)

    # precompute RNG array once
    rng_count = n_angles * samples_per_angle * (max_reflect + max_transmit)
    rng = (np.arange(rng_count, dtype=np.int64) * 1103515245 + 12345) & 0x7FFFFFFF
    rng = (rng / np.float32(2**31))

    # Parallel over angles
    for i in prange(n_angles):
        dx0 = cos_vals[i]
        dy0 = sin_vals[i]
        base_rng_i = i * samples_per_angle * (max_reflect + max_transmit)

        for s in range(samples_per_angle):
            # ray state
            x = x_ant
            y = y_ant
            dx = dx0
            dy = dy0
            sum_loss = 0.0
            last_val = -1.0  # sentinel
            refl_ct = 0
            trans_ct = 0
            rng_idx = base_rng_i + s * (max_reflect + max_transmit)

            # march until both caps or loss cap hit
            while True:
                # trace along this direction to next boundary
                traveled = 0.0
                hit_px = hit_py = -1

                while traveled <= max_dist:
                    x += dx * radial_step
                    y += dy * radial_step
                    traveled += radial_step

                    px = int(round(x))
                    py = int(round(y))
                    if px < 0 or px >= w or py < 0 or py >= h:
                        # out of image
                        traveled = max_dist + 1.0
                        break

                    # update best (min) loss so far
                    if sum_loss < output[py, px]:
                        output[py, px] = sum_loss

                    # check for boundary
                    val = reflectance_matrix[py, px]
                    if last_val < 0.0:
                        last_val = val
                    if val != last_val:
                        hit_px, hit_py = px, py
                        break
                    last_val = val

                if hit_px < 0:
                    break  # left image

                # decide branch
                if refl_ct >= max_reflect and trans_ct >= max_transmit:
                    break
                elif refl_ct >= max_reflect:
                    branch = 0
                elif trans_ct >= max_transmit:
                    branch = 1
                else:
                    branch = 1 if rng[rng_idx] < reflection_prob else 0
                    rng_idx += 1

                # apply loss + update
                if branch == 1:
                    # reflect
                    sum_loss += reflectance_matrix[hit_py, hit_px]
                    refl_ct += 1
                    # estimate normal (4-nbr) and reflect dir
                    nx = 0.0; ny = 0.0
                    # inline neighbor checks
                    if hit_px > 0 and reflectance_matrix[hit_py, hit_px-1] != val:
                        nx -= 1.0
                    if hit_px < w-1 and reflectance_matrix[hit_py, hit_px+1] != val:
                        nx += 1.0
                    if hit_py > 0 and reflectance_matrix[hit_py-1, hit_px] != val:
                        ny -= 1.0
                    if hit_py < h-1 and reflectance_matrix[hit_py+1, hit_px] != val:
                        ny += 1.0
                    norm = np.hypot(nx, ny)
                    if norm > 0.0:
                        nx /= norm; ny /= norm
                    else:
                        nx, ny = -dx, -dy
                    # reflect vector
                    dot = dx*nx + dy*ny
                    dx -= 2.0 * dot * nx
                    dy -= 2.0 * dot * ny
                    mag = np.hypot(dx, dy)
                    if mag > 0.0:
                        dx /= mag; dy /= mag
                else:
                    # transmit
                    sum_loss += transmittance_matrix[hit_py, hit_px]
                    trans_ct += 1
                    # direction unchanged

                # cap-check
                if sum_loss >= max_loss:
                    break

                # continue from this boundary
                x = hit_px
                y = hit_py

    return output


def _calculate_reflectance_eff_numpy(
    reflectance_matrix: np.ndarray,
    transmittance_matrix: np.ndarray,
    x_ant: float,
    y_ant: float,
    n_angles: int = 360*128,
    radial_step: float = 1.0,
    max_walls: int = 5,
    reflection_prob: float = 0.5
) -> np.ndarray:
    """
    Single‐ray effective attenuation: each wall interface
    contributes an expected loss a_eff = p_refl*R + (1-p_refl)*T.
    March one ray per angle, accumulate these a_eff losses,
    and record the per-pixel minimum accumulated loss.
    """
    h, w = reflectance_matrix.shape
    # initialize to a large value
    output = np.full((h, w), np.inf, dtype=np.float32)

    two_pi = 2.0 * np.pi
    dtheta = two_pi / n_angles
    cosv = np.cos(np.arange(n_angles) * dtheta)
    sinv = np.sin(np.arange(n_angles) * dtheta)
    max_dist = np.hypot(w, h)

    for i in prange(n_angles):
        dx = cosv[i]
        dy = sinv[i]
        sum_loss = 0.0
        last_val = -1.0  # sentinel: not yet on material
        r = 0.0
        # march until edge
        while r <= max_dist:
            x = x_ant + r * dx
            y = y_ant + r * dy
            px = int(x + 0.5)
            py = int(y + 0.5)
            if px < 0 or px >= w or py < 0 or py >= h:
                break
            val = reflectance_matrix[py, px]
            # on interface if crossing from material->air or air->material
            if last_val >= 0.0 and val != last_val:
                # expected attenuation at this interface
                R = reflectance_matrix[py, px]
                T = transmittance_matrix[py, px]
                a_eff = reflection_prob * R + (1.0 - reflection_prob) * T
                sum_loss += a_eff
                # count walls and stop after max_walls
                last_val = val
                # record this loss at the interface pixel
                if sum_loss < output[py, px]:
                    output[py, px] = sum_loss
                # optionally stop is reached walls cap
                # but effective doesn't branch so continue
            last_val = val
            # record cumulative loss at every pixel
            if sum_loss < output[py, px]:
                output[py, px] = sum_loss
            r += radial_step

    return output


@njit(parallel=True, fastmath=True, nogil=True, boundscheck=False)



class Approx:
    def approximate(self, sample : RadarSample) -> torch.Tensor:
        ref, trans, dist = sample.input_img.cpu().numpy()
        x_ant, y_ant = sample.x_ant, sample.y_ant

        fspl = calculate_fspl(dist_m=dist, freq_MHz=sample.freq_MHz)

        ref_feat = _calculate_hybrid_loss_mc_numpy_fast(ref, trans, x_ant, y_ant, n_angles=360*128, samples_per_angle=32, radial_step=1.0, max_reflect=5, max_transmit=10, reflection_prob=0.5, max_loss=160)
        ref_feat = np.minimum(ref_feat+fspl, 160.0)
        
        trans_feat = _calculate_transmittance_loss_numpy(trans, x_ant, y_ant, n_angles=360*128, radial_step=1.0, max_walls=10)
        trans_feat = gaussian_filter(trans_feat, sigma=5.0, mode='reflect')
        trans_feat = np.minimum(trans_feat+fspl, 160.0)

        eff_feat = _calculate_reflectance_eff_numpy(ref, trans, x_ant, y_ant)
        eff_feat = np.minimum(eff_feat+fspl, 160.0)

        approx = np.floor(np.minimum(trans_feat, eff_feat))

        return torch.from_numpy(approx)
    
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