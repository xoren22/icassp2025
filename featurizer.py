import numpy as np
from numba import njit

@njit
def calculate_transmittance_loss(transmittance_matrix, x_ant, y_ant, 
                                 n_angles=360*128/1, radial_step=1.0, max_walls=10):
    """
    Approximate the loss from an antenna at (x_ant, y_ant) to every pixel in 'transmittance_matrix'
    using a polar (angle + radial stepping) algorithm.
    
    Now we finalize a nonzero block only when we see the first zero after it,
    rather than incrementing sum_loss on the first pixel of that block.
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
            
            # Exiting the grid
            if px < 0 or px >= w or py < 0 or py >= h:
                # If we leave bounds while still in a nonzero block, finalize it.
                if last_val is not None and last_val > 0:
                    sum_loss += last_val
                    wall_count += 1
                    if wall_count >= max_walls:
                        pass  # Already out of bounds, so we do nothing more
                break
            
            val = transmittance_matrix[py, px]
            
            # Initialize last_val on first pixel
            if last_val is None:
                last_val = val
            
            # If the pixel value changed
            if val != last_val:
                # We only finalize if we are *leaving* a nonzero block and see a zero
                # i.e., last_val>0 and new val==0
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
            
            # Update output (still store the best/lowest sum_loss for this pixel)
            if output[py, px] == 0 or (sum_loss < output[py, px]):
                output[py, px] = sum_loss
            
            r += radial_step
    
    return output


# compiling numba function for better performance
calculate_transmittance_loss(np.array([[1]]), 0, 0)


def calculate_fspl(
    W,
    H,
    x_ant,
    y_ant,
    antenna_gain,           # shape=(360,) antenna gain in dBi [0..359]
    freq_MHz,               # frequency in MHz
    grid_unit_meters=0.25,  # cell size in meters
    min_distance_m=0.125,     # clamp distance below this
):
    """
    Example free-space pathloss calculation with distance clamping.
    """
    y_idx, x_idx = np.indices((H, W))

    dx = x_idx - x_ant
    dy = y_idx - y_ant

    dist_m = np.sqrt(dx**2 + dy**2) * grid_unit_meters
    dist_clamped = np.maximum(dist_m, min_distance_m)

    fspl_db = 20.0 * np.log10(dist_clamped) + 20.0 * np.log10(freq_MHz) - 27.55

    pathloss_db = fspl_db - antenna_gain
    return pathloss_db


def calculate_antenna_gain(radiation_pattern, W, H, azimuth, x_ant, y_ant):
    x_grid = np.repeat(np.linspace(0, W-1, W), H, axis=0).reshape(W, H).T
    y_grid = np.repeat(np.linspace(0, H-1, H), W, axis=0).reshape(H, W)
    
    angles = -(180/np.pi) * np.arctan2((y_ant - y_grid), (x_ant - x_grid)) + 180 + azimuth
    angles = np.where(angles > 359, angles - 360, angles).astype(int)
    antenna_gain = radiation_pattern[angles]
    return antenna_gain

