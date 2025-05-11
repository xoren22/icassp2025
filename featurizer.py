import torch
import numpy as np
from numba import njit, prange
from scipy.ndimage import gaussian_filter

from _types import RadarSample
from config import OUTPUT_SCALER
from utils import calculate_distance


@njit(fastmath=True)
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

def calculate_hybrid_loss_mc(reflectance, transmittance, x_ant, y_ant):
    reflectance_np = reflectance.cpu().numpy()
    transmittance_np = transmittance.cpu().numpy()
    output_np = _calculate_hybrid_loss_mc_numpy_fast(reflectance_np, transmittance_np, x_ant, y_ant)
    return torch.from_numpy(output_np).to(device=torch.device('cpu'))


@njit(fastmath=True)
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

def calculate_reflectance_eff_numpy(reflectance, transmittance, x_ant, y_ant):
    reflectance_np = reflectance.cpu().numpy()
    transmittance_np = transmittance.cpu().numpy()
    output_np = _calculate_reflectance_eff_numpy(reflectance_np, transmittance_np, x_ant, y_ant)
    return torch.from_numpy(output_np).to(device=torch.device('cpu'))


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
    normalized[6] = (normalized[6] - 87.0) / 73.0 # tranmittance+fspl loss
    normalized[7] = (normalized[7] - 87.0) / 73.0 # tranmittance+fspl loss
    # normalized[8] is the mask and is left as is
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

    ref_feat = calculate_hybrid_loss_mc(reflectance, transmittance, sample.x_ant, sample.y_ant)
    ref_feat = torch.minimum(ref_feat+free_space_pathloss, torch.tensor(160.0))
    
    eff_feat = calculate_reflectance_eff_numpy(reflectance, transmittance, sample.x_ant, sample.y_ant)
    eff_feat = torch.minimum(eff_feat+free_space_pathloss, torch.tensor(160.0))
    
    # Build input tensor: [6, H, W] - Already in correct (C, H, W) format
    input_tensor = torch.zeros((9, sample.H, sample.W), dtype=torch.float32, device=torch.device('cpu'))
    input_tensor[0] = reflectance  # reflectance
    input_tensor[1] = transmittance  # transmittance
    input_tensor[2] = distance  # distance
    input_tensor[3] = torch.full((sample.H, sample.W), sample.freq_MHz, dtype=torch.float32, device=torch.device('cpu'))
    input_tensor[4] = free_space_pathloss
    input_tensor[5] = transmittance_loss
    input_tensor[6] = ref_feat
    input_tensor[7] = eff_feat
    input_tensor[8] = sample.mask
    
    input_tensor = normalize_input(input_tensor)

    return input_tensor