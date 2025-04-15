import torch
import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter

from _types import RadarSample


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
    normalized = input_tensor.clone()
    normalized[0] = normalized[0] / 10.0 # reflecatnace is ~12
    normalized[1] = normalized[1] / 10.0 # transmittance is ~12
    normalized[2] = normalized[2]  / 35.0 # max fspl are in range 25~43
    normalized[3] = torch.log10(normalized[3]) - np.log10([868, 1800, 3500]).mean()  # frequency in MHz
    normalized[4] = normalized[4] / 100.0 # fspl+tranmittance has values < 300
    # normalized[5] is the mask and is left as is
    return normalized


def featurize_inputs(sample: RadarSample) -> torch.Tensor:
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

    # Calculate transmittance loss on CPU
    transmittance_loss = calculate_transmittance_loss(
        transmittance, 
        sample.x_ant, 
        sample.y_ant
    )
    
    fs_plus_transmittance_loss = free_space_pathloss + transmittance_loss

    # Build input tensor: [6, H, W] - Already in correct (C, H, W) format
    input_tensor = torch.zeros((6, sample.H, sample.W), dtype=torch.float32, device=torch.device('cpu'))
    input_tensor[0] = reflectance  # reflectance
    input_tensor[1] = transmittance  # transmittance
    input_tensor[2] = free_space_pathloss  # distance
    input_tensor[3] = torch.full((sample.H, sample.W), sample.freq_MHz, dtype=torch.float32, device=torch.device('cpu'))
    input_tensor[4] = fs_plus_transmittance_loss

    input_tensor = normalize_input(input_tensor)

    mask = sample.mask
    input_tensor[5] = mask
    
    return input_tensor