import torch
import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter

from _types import RadarSample

# Import combined approximator utilities without duplicating code
from approx import (
    load_precomputed_normals_for_building,
    calculate_combined_loss_with_normals,
    N_ANGLES,
)


@njit
def _calculate_transmittance_loss_numpy(transmittance_matrix, x_ant, y_ant, n_angles, radial_step, max_walls):
    """
    Numpy implementation for numba optimization.
    This function must stay as numpy for numba to work.
    """
    h, w = transmittance_matrix.shape
    dtheta = 2.0 * np.pi / n_angles
    output = np.zeros((h, w), dtype=transmittance_matrix.dtype)
    
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
                if last_val is not None and last_val > 0:
                    sum_loss += last_val
                    wall_count += 1
                    if wall_count >= max_walls:
                        pass  # Already out of bounds, so we do nothing more
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
                            
                            if output[py_temp, px_temp] == 0 or sum_loss < output[py_temp, px_temp]:
                                output[py_temp, px_temp] = sum_loss
                            r_temp += radial_step
                        break
                last_val = val
            
            if output[py, px] == 0 or (sum_loss < output[py, px]):
                output[py, px] = sum_loss
            
            r += radial_step
    
    return output

_calculate_transmittance_loss_numpy(np.array([[1]]), 0, 0, 2, 2, 2)


def calculate_transmittance_loss(transmittance_matrix, x_ant, y_ant, n_angles=360*128*1, radial_step=1.0, max_walls=10, smooth=True):
    transmittance_np = transmittance_matrix.cpu().numpy()
    output_np = _calculate_transmittance_loss_numpy(transmittance_np, x_ant, y_ant, n_angles, radial_step, max_walls)
    if smooth:
        output_np = gaussian_filter(output_np, sigma=2.0, mode='reflect')
    return torch.from_numpy(output_np).to(device=torch.device('cpu'))


def calculate_combined_feature(sample: RadarSample, n_angles: int = N_ANGLES) -> torch.Tensor:
    """
    Compute the combined pathloss approximation feature (FSPL + walls) using
    the precomputed wall normals for the building associated with `sample`.
    Returns a CPU float32 tensor of shape (H, W).
    """
    reflectance = sample.input_img[0].cpu().numpy()
    transmittance = sample.input_img[1].cpu().numpy()
    x_ant, y_ant, f = float(sample.x_ant), float(sample.y_ant), float(sample.freq_MHz)

    # Determine building id from sample.ids (tuple like (b, ant, f, sp))
    b_id = int(sample.ids[0]) if sample.ids is not None else 0
    nx_img, ny_img = load_precomputed_normals_for_building(b_id, reflectance, transmittance)

    feat, _ = calculate_combined_loss_with_normals(
        reflectance.astype(np.float64),
        transmittance.astype(np.float64),
        nx_img, ny_img,
        x_ant, y_ant, f,
        n_angles=n_angles,
    )
    feat = np.nan_to_num(np.minimum(feat, 160.0), nan=160.0, posinf=160.0, neginf=0.0)
    return torch.from_numpy(feat.astype(np.float32))


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


def calculate_antenna_gain(radiation_pattern, W, H, azimuth, x_ant, y_ant):
    """
    Calculate antenna gain across a grid based on radiation pattern and antenna orientation.
    Works with torch tensors.
    """
    x_grid = torch.arange(W, device=torch.device('cpu')).expand(H, W)
    y_grid = torch.arange(H, device=torch.device('cpu')).view(-1, 1).expand(H, W)
    angles = -(180/torch.pi) * torch.atan2((y_ant - y_grid), (x_ant - x_grid)) + 180 + azimuth
    angles = torch.where(angles > 359, angles - 360, angles).to(torch.long)
    antenna_gain = radiation_pattern[angles]
    
    return antenna_gain



def normalize_input(input_tensor):
    """Normalize channels to reasonable ranges; returns a new tensor."""
    normalized = torch.empty_like(input_tensor)

    # Reflectance [0,150] -> [0,1]
    ch = input_tensor[0].clone()
    ch.clamp_(min=0, max=150)
    normalized[0] = ch / 150.0

    # Transmittance [0,150] -> [0,1]
    ch = input_tensor[1].clone()
    ch.clamp_(min=0, max=150)
    normalized[1] = ch / 150.0

    # Distance [0,1000] -> [0,1]
    ch = input_tensor[2].clone()
    ch.clamp_(min=0, max=1000)
    normalized[2] = ch / 1000.0

    # Frequency [868,3500] -> [0,1]
    ch = input_tensor[3].clone()
    ch.clamp_(min=868, max=3500)
    normalized[3] = (ch - 868.0) / (3500.0 - 868.0)

    # Approx feature [0,160] -> [0,1]
    ch = input_tensor[4].clone()
    ch = torch.nan_to_num(ch, nan=160.0, posinf=160.0, neginf=0.0)
    ch.clamp_(min=0, max=160)
    normalized[4] = ch / 160.0

    # Mask passthrough
    normalized[5] = input_tensor[5]

    return normalized


def featurize_inputs(sample: RadarSample, feature_type: str = "transmittance") -> torch.Tensor:
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

    if feature_type == "combined":
        approx_feature = calculate_combined_feature(sample)
    else:
        # Calculate transmittance loss on CPU
        transmittance_loss = calculate_transmittance_loss(
            transmittance, 
            sample.x_ant, 
            sample.y_ant
        )
        approx_feature = free_space_pathloss + transmittance_loss

    # Build input tensor: [6, H, W]
    input_tensor = torch.zeros((6, sample.H, sample.W), dtype=torch.float32, device=torch.device('cpu'))
    input_tensor[0] = reflectance  # reflectance
    input_tensor[1] = transmittance  # transmittance
    input_tensor[2] = free_space_pathloss  # distance
    input_tensor[3] = torch.full((sample.H, sample.W), sample.freq_MHz, dtype=torch.float32, device=torch.device('cpu'))
    input_tensor[4] = approx_feature

    input_tensor = normalize_input(input_tensor)

    mask = sample.mask
    input_tensor[5] = mask
    
    return input_tensor