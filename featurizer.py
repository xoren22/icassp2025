import torch
import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter

from _types import RadarSample
from config import OUTPUT_SCALER
from utils import calculate_distance


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

def calculate_transmittance_loss(transmittance_matrix, x_ant, y_ant, n_angles=360*1, radial_step=1.0, max_walls=10, smooth=True):
    transmittance_np = transmittance_matrix.cpu().numpy()
    output_np = _calculate_transmittance_loss_numpy(transmittance_np, x_ant, y_ant, n_angles, radial_step, max_walls)
    if smooth:
        output_np = gaussian_filter(output_np, sigma=5.0, mode='reflect')
    return torch.from_numpy(output_np).to(device=torch.device('cpu'))


def calculate_fspl(
    dist_m,               # distance in meters (torch tensor)
    freq_MHz,             # frequency in MHz
):
    freq_tensor = torch.tensor(freq_MHz, device=torch.device('cpu'))
    fspl_db = 20.0 * torch.log10(dist_m) + 20.0 * torch.log10(freq_tensor) - 27.55
    
    return fspl_db

def normalize_input(input_tensor):
    normalized = input_tensor.clone()
    normalized[0] = (normalized[0] - 9.0) / 9.0 # reflecatnace is ~12
    normalized[1] = (normalized[1] - 9.0) / 9.0 # transmittance is ~12
    normalized[2] = torch.log10(normalized[2]) - 1.0 # distance
    normalized[3] = torch.log10(normalized[3]) - 3.0 # frequency
    normalized[4] = (normalized[4] - 87.0) / 73.0 # free space pathloss
    normalized[5] = (normalized[5] - 87.0) / 73.0 # tranmittance+fspl loss
    # normalized[6] is the mask and is left as is
    return normalized


def featurize_inputs(sample: RadarSample) -> torch.Tensor:
    reflectance = sample.reflectance  # First channel
    transmittance = sample.transmittance  # Second channel
    distance = calculate_distance(
        W=sample.W,
        H=sample.H,
        x_ant=sample.x_ant,
        y_ant=sample.y_ant,
        pixel_size=sample.pixel_size,
    )

    # Calculate free space path loss on CPU
    free_space_pathloss = calculate_fspl(
        dist_m=distance,
        freq_MHz=sample.freq_MHz,
    )

    # Calculate transmittance loss on CPU
    transmittance_loss = calculate_transmittance_loss(
        transmittance, 
        sample.x_ant, 
        sample.y_ant
    )
    
    # Build input tensor: [6, H, W] - Already in correct (C, H, W) format
    input_tensor = torch.zeros((7, sample.H, sample.W), dtype=torch.float32, device=torch.device('cpu'))
    input_tensor[0] = reflectance  # reflectance
    input_tensor[1] = transmittance  # transmittance
    input_tensor[2] = distance  # distance
    input_tensor[3] = torch.full((sample.H, sample.W), sample.freq_MHz, dtype=torch.float32, device=torch.device('cpu'))
    input_tensor[4] = free_space_pathloss
    input_tensor[5] = transmittance_loss
    input_tensor[6] = sample.mask
    
    input_tensor = normalize_input(input_tensor)

    return input_tensor