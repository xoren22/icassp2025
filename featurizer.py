import torch
import numpy as np
from numba import njit

from _types import RadarSample
from utils import calculate_distance, combine_incoherent_sum_db


@njit
def _calculate_transmittance_loss_numpy(transmittance_matrix, x_ant, y_ant, n_angles=360*128, radial_step=1.0, max_walls=10):
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
    
    for i in range(int(n_angles)):
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


def calculate_transmittance_loss(transmittance_matrix, x_ant, y_ant, n_angles=360*128/1, radial_step=1.0, max_walls=10):
    transmittance_np = transmittance_matrix.cpu().numpy()
    output_np = _calculate_transmittance_loss_numpy(transmittance_np, x_ant, y_ant, n_angles, radial_step, max_walls)
    return torch.from_numpy(output_np).to(device=torch.device('cpu'))


# _calculate_transmittance_loss_numpy(np.array([[1]]), 0, 0)


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
    min_antenna_gain = -55.0
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    normalized = input_tensor.clone()
    for i in range(2):
        normalized[i] = (normalized[i] / 255.0 - mean[i]) / std[i]
    normalized[2] = normalized[2]  / 100.0 # fspl channel
    normalized[3] = normalized[3] / min_antenna_gain
    normalized[4] = torch.log10(normalized[4]) - 1.9  # "magic shift"
    normalized[5] = normalized[5] / 100.0 # this feature mask has values < 300

    return normalized


def featurize_inputs(sample: RadarSample) -> torch.Tensor:
    reflectance    = sample.input_img[0]  # channel 0
    transmittance  = sample.input_img[1]  # channel 1
    mask           = sample.mask

    x_ants = sample.x_ant
    y_ants = sample.y_ant
    freqs  = sample.freq_MHz
    azims  = sample.azimuth
    rads   = sample.radiation_pattern

    # Collect per-antenna dB maps
    fspl_maps   = []
    transm_maps = []
    gain_maps   = []

    # Loop over antennas
    for i in range(len(x_ants)):
        # 1) Compute distance from this antenna
        dist_i = calculate_distance(
            x_ant=x_ants[i],
            y_ant=y_ants[i],
            H=sample.H,
            W=sample.W,
            pixel_size=sample.pixel_size
        )
        
        # 2) Calculate antenna gain map
        gain_i = calculate_antenna_gain(
            radiation_pattern=rads[i],
            W=sample.W,
            H=sample.H,
            azimuth=azims[i],
            x_ant=x_ants[i],
            y_ant=y_ants[i],
        )

        # 3) Calculate FSPL (which subtracts antenna gain internally)
        #    pass the distance & freq for this antenna
        fspl_db_i = calculate_fspl(
            dist_m=dist_i,
            freq_MHz=freqs[i],
            antenna_gain=gain_i,   # shape=(H,W) after broadcast in the function
        )
        fspl_maps.append(fspl_db_i)

        # 4) Calculate transmittance loss
        transm_loss_i = calculate_transmittance_loss(
            transmittance_matrix=transmittance,
            x_ant=x_ants[i],
            y_ant=y_ants[i],
        )
        transm_maps.append(transm_loss_i)
        gain_maps.append(gain_i)

    combined_fspl_db   = combine_incoherent_sum_db(fspl_maps)
    combined_transm_db = combine_incoherent_sum_db(transm_maps)
    combined_gain_db   = combine_incoherent_sum_db(gain_maps)
    fs_plus_transmittance_loss = combined_fspl_db + combined_transm_db
    input_tensor = torch.zeros((7, sample.H, sample.W), dtype=torch.float32, device=torch.device('cpu'))

    input_tensor[0] = reflectance         # reflectance
    input_tensor[1] = transmittance       # transmittance
    input_tensor[2] = combined_fspl_db    # replaced old "distance" with combined FSPL
    input_tensor[3] = combined_gain_db    # combined antenna gain
    input_tensor[4] = torch.full((sample.H, sample.W), np.mean(freqs), dtype=torch.float32)
    input_tensor[5] = fs_plus_transmittance_loss
    input_tensor[6] = mask

    input_tensor = normalize_input(input_tensor)

    return input_tensor


