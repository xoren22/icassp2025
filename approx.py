import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit, prange

from helper import RadarSample, load_samples, compare_two_matrices, rmse, visualize_predictions

# ---------------------------------------------------------------------
#  FREE‐SPACE PATH LOSS
# ---------------------------------------------------------------------
def calculate_fspl(dist_m, freq_MHz, min_dist_m=0.125):
    dist_clamped = np.maximum(dist_m, min_dist_m)
    return 20.0 * np.log10(dist_clamped) + 20.0 * np.log10(freq_MHz) - 27.55

# ---------------------------------------------------------------------
#  NUMBA HELPERS FOR COMBINED TRACING
# ---------------------------------------------------------------------
@njit(inline='always')
def _step_until_wall(mat, x0, y0, dx, dy, radial_step, max_dist):
    h, w = mat.shape
    x, y = x0, y0
    last_val = mat[int(round(y0)), int(round(x0))]
    travelled = 0.0

    while travelled <= max_dist:
        x += dx * radial_step
        y += dy * radial_step
        travelled += radial_step

        px = int(round(x))
        py = int(round(y))
        if px < 0 or px >= w or py < 0 or py >= h:
            return -1, -1, travelled, last_val, last_val

        cur_val = mat[py, px]
        if cur_val != last_val:
            return px, py, travelled, last_val, cur_val

    return -1, -1, travelled, last_val, last_val

@njit(inline='always')
def _estimate_normal(refl_mat, px, py):
    h, w = refl_mat.shape
    val = refl_mat[py, px]
    nx = 0.0; ny = 0.0
    if px > 0   and refl_mat[py, px-1] != val: nx -= 1.0
    if px < w-1 and refl_mat[py, px+1] != val: nx += 1.0
    if py > 0   and refl_mat[py-1, px] != val: ny -= 1.0
    if py < h-1 and refl_mat[py+1, px] != val: ny += 1.0

    norm = np.hypot(nx, ny)
    if norm == 0.0:
        return 0.0, 0.0
    return nx / norm, ny / norm

@njit(inline='always')
def _reflect_dir(dx, dy, nx, ny):
    dot = dx*nx + dy*ny
    rx = dx - 2.0*dot*nx
    ry = dy - 2.0*dot*ny
    mag = np.hypot(rx, ry)
    if mag == 0.0:
        return -dx, -dy
    return rx/mag, ry/mag

# ---------------------------------------------------------------------
#  PAINT‐TO‐EDGE FALLBACK
# ---------------------------------------------------------------------

@njit(inline='always')
def _paint_to_edge(out_img, x0, y0, dx, dy, acc_loss, radial_step, max_dist):
    h, w = out_img.shape
    r = 0.0
    while r <= max_dist:
        ix = int(round(x0 + dx*r))
        iy = int(round(y0 + dy*r))
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return
        if acc_loss < out_img[iy, ix]:
            out_img[iy, ix] = acc_loss
        r += radial_step

# ---------------------------------------------------------------------
#  RECURSIVE TRACER
# ---------------------------------------------------------------------

@njit
def _trace_ray_recursive(
    refl_mat, trans_mat, out_img,
    x0, y0, dx, dy,
    trans_ct, refl_ct,
    acc_loss,
    radial_step, max_dist,
    max_trans, max_refl,
    max_loss
):
    if acc_loss >= max_loss:
        return

    px_hit, py_hit, travelled, _, _ = _step_until_wall(
        refl_mat, x0, y0, dx, dy, radial_step, max_dist
    )

    # paint the free‐space segment
    steps = int(travelled / radial_step) + 1
    for s in range(steps):
        xi = x0 + dx * radial_step * s
        yi = y0 + dy * radial_step * s
        ix = int(round(xi))
        iy = int(round(yi))
        if ix < 0 or ix >= out_img.shape[1] or iy < 0 or iy >= out_img.shape[0]:
            break
        if acc_loss < out_img[iy, ix]:
            out_img[iy, ix] = acc_loss

    # exited image?
    if px_hit < 0:
        return

    # —— transmission branch ——
    if trans_ct < max_trans:
        new_loss = acc_loss + trans_mat[py_hit, px_hit]
        _trace_ray_recursive(
            refl_mat, trans_mat, out_img,
            px_hit, py_hit, dx, dy,
            trans_ct+1, refl_ct,
            new_loss,
            radial_step, max_dist,
            max_trans, max_refl,
            max_loss
        )
    else:
        _paint_to_edge(out_img, px_hit, py_hit, dx, dy,
                       acc_loss, radial_step, max_dist)

    # —— reflection branch ——
    if refl_ct < max_refl:
        nx, ny = _estimate_normal(refl_mat, px_hit, py_hit)
        if nx != 0.0 or ny != 0.0:
            rdx, rdy = _reflect_dir(dx, dy, nx, ny)
            new_loss = acc_loss + refl_mat[py_hit, px_hit]
            _trace_ray_recursive(
                refl_mat, trans_mat, out_img,
                px_hit, py_hit, rdx, rdy,
                trans_ct, refl_ct+1,
                new_loss,
                radial_step, max_dist,
                max_trans, max_refl,
                max_loss
            )
    else:
        _paint_to_edge(out_img, px_hit, py_hit, dx, dy,
                       acc_loss, radial_step, max_dist)

# ---------------------------------------------------------------------
#  PUBLIC ENTRY-POINT
# ---------------------------------------------------------------------

@njit(parallel=True, fastmath=True, nogil=True, boundscheck=False)
def calculate_combined_loss(
    reflectance_mat, transmittance_mat,
    x_ant, y_ant,
    n_angles: int = 360*128,
    radial_step: float = 1.0,
    max_reflections: int = 3,
    max_transmissions: int = 10,
    max_loss: float = 160.0
) -> np.ndarray:
    h, w = reflectance_mat.shape
    out_img = np.full((h, w), max_loss, dtype=np.float32)

    dtheta   = 2.0 * np.pi / n_angles
    max_dist = np.hypot(w, h)
    cos_v    = np.cos(np.arange(n_angles) * dtheta)
    sin_v    = np.sin(np.arange(n_angles) * dtheta)

    for i in prange(n_angles):
        _trace_ray_recursive(
            reflectance_mat, transmittance_mat, out_img,
            x_ant, y_ant,
            cos_v[i], sin_v[i],
            0, 0,       # trans_ct, refl_ct
            0.0,        # initial loss
            radial_step, max_dist,
            max_transmissions, max_reflections,
            max_loss
        )

    return out_img


import numpy as np
from numba import njit, prange

# ───────────────────────────────────────────────────────────────
# helpers
# ───────────────────────────────────────────────────────────────
@njit(inline='always')
def _euclidean_distance(px: int, py: int, x_ant: float, y_ant: float, pixel_size: float = 0.25) -> float:
    """Pixel-to-antenna distance in the same units as pixel spacing (≈ 0.25 m)."""
    return np.hypot(px - x_ant, py - y_ant) * pixel_size

@njit(inline='always')
def _fspl(dist_m: float, freq_MHz: float, min_dist_m: float = 0.125) -> float:
    """Free-space path-loss in dB, clamped to avoid log(0)."""
    d = dist_m if dist_m > min_dist_m else min_dist_m
    return 20.0 * np.log10(d) + 20.0 * np.log10(freq_MHz) - 27.55

# ───────────────────────────────────────────────────────────────
# transmission-only tracer with FSPL baked in
# ───────────────────────────────────────────────────────────────
@njit(parallel=True, fastmath=True, nogil=True, boundscheck=False)
def calculate_transmission_loss_numpy(
    transmittance_matrix: np.ndarray,
    x_ant: float,
    y_ant: float,
    freq_MHz: float,
    n_angles: int = 360 * 128,
    radial_step: float = 1.0,
    max_walls: int = 10
) -> np.ndarray:
    """
    Casts `n_angles` rays from the antenna pixel.  Each time a ray crosses a wall
    (positive → zero in `transmittance_matrix`) it adds that wall’s loss.
    After all rays are done, FSPL for every pixel is added in one pass.
    """
    h,  w  = transmittance_matrix.shape
    out    = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)   # for averaging overlaps

    dtheta   = 2.0 * np.pi / n_angles
    max_dist = np.sqrt(w * w + h * h)
    cos_vals = np.cos(np.arange(n_angles) * dtheta)
    sin_vals = np.sin(np.arange(n_angles) * dtheta)

    for i in range(n_angles):
        ct, st   = cos_vals[i], sin_vals[i]
        sum_loss = 0.0
        last_val = None
        wall_ct  = 0
        r        = 0.0

        while r <= max_dist:
            x = x_ant + r * ct
            y = y_ant + r * st
            px = int(round(x))
            py = int(round(y))

            # ray outside → step in until inside
            if px < 0 or px >= w or py < 0 or py >= h:
                if last_val is None:
                    r += radial_step
                    continue
                if last_val > 0:
                    sum_loss += last_val
                    if sum_loss > 160.0:
                        sum_loss = 160.0
                break

            val = transmittance_matrix[py, px]
            if last_val is None:
                last_val = val

            # crossing from positive to zero ⇒ add wall loss
            if val != last_val:
                if last_val > 0 and val == 0:
                    sum_loss += last_val
                    if sum_loss > 160.0:
                        sum_loss = 160.0
                        break
                    wall_ct += 1
                    if wall_ct >= max_walls:
                        # fill the rest of the ray with current loss
                        r_tmp = r
                        while r_tmp <= max_dist:
                            x_t = x_ant + r_tmp * ct
                            y_t = y_ant + r_tmp * st
                            px_t = int(round(x_t))
                            py_t = int(round(y_t))
                            if px_t < 0 or px_t >= w or py_t < 0 or py_t >= h:
                                break
                            if counts[py_t, px_t] == 0:
                                out[py_t, px_t] = sum_loss
                                counts[py_t, px_t] = 1
                            else:
                                c = counts[py_t, px_t]
                                out[py_t, px_t] = (out[py_t, px_t] * c + sum_loss) / (c + 1)
                                counts[py_t, px_t] = c + 1
                            r_tmp += radial_step
                        break
                last_val = val

            # write current loss into this pixel (average if hit before)
            if counts[py, px] == 0:
                out[py, px] = sum_loss
                counts[py, px] = 1
            else:
                c = counts[py, px]
                out[py, px] = (out[py, px] * c + sum_loss) / (c + 1)
                counts[py, px] = c + 1

            if wall_ct >= max_walls or sum_loss >= 160.0:
                if sum_loss > 160.0:
                    sum_loss = 160.0
                break

            r += radial_step

    # ─ add FSPL for every pixel ─
    for py in prange(h):
        for px in range(w):
            d    = _euclidean_distance(px, py, x_ant, y_ant)
            fspl = _fspl(d, freq_MHz)
            total = out[py, px] + fspl
            out[py, px] = 160.0 if total > 160.0 else total

    return out


# ---------------------------------------------------------------------
#  VIS & DEBUG
# ---------------------------------------------------------------------
def debug_combined_method(sample):
    ref, trans, dist = sample.input_img.cpu().numpy()
    x,y,freq_MHz = sample.x_ant, sample.y_ant, sample.freq_MHz
    fspl = calculate_fspl(dist, sample.freq_MHz)
    comb = calculate_combined_loss(ref, trans, x, y, n_angles=360*128, max_reflections=5, max_transmissions=10)
    tx   = calculate_transmission_loss_numpy(trans, x, y, freq_MHz, n_angles=360*128)
    print("Combined range:", comb.min(), comb.max())
    print("Trans range:  ", tx.min(), tx.max())
    print("Painted pixels diff:", np.sum(comb>0), np.sum(tx>0))
    return comb, tx, fspl

# ---------------------------------------------------------------------
#  WRAPPER CLASS
# ---------------------------------------------------------------------
class Approx:
    def __init__(self, method='combined'):
        self.method = method
    def approximate(self, sample: RadarSample) -> torch.Tensor:
        ref, trans, dist = sample.input_img.cpu().numpy()
        x,y,freq_MHz = sample.x_ant, sample.y_ant, sample.freq_MHz
        fspl = calculate_fspl(dist, sample.freq_MHz)
        if self.method=='combined':
            feat = calculate_combined_loss(ref, trans, x, y,
                                           n_angles=360*128,
                                           max_reflections=3,
                                           max_transmissions=10) + fspl
        else:
            feat = calculate_transmission_loss_numpy(trans, x, y, freq_MHz,
                                                     n_angles=360*128,
                                                     max_walls=10)
        out = np.minimum(feat, 160.0)
        return torch.from_numpy(np.floor(out))
    def predict(self, samples):
        return [self.approximate(s) for s in tqdm(samples, "predicting")]

# ---------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":  
    # 1) load a handful of samples  
    samples = load_samples(num_samples=5)  

    # 2) run both methods  
    comb_model = Approx(method='combined')  
    trans_model = Approx(method='transmission')  
    preds_comb  = comb_model.predict(samples)  
    preds_trans = trans_model.predict(samples)  

    # 3) zero-reflection sanity check  
    s0    = samples[0]  
    ref0, trans0, dist0 = s0.input_img.cpu().numpy()[0], s0.input_img.cpu().numpy()[1], s0.input_img.cpu().numpy()[2]  
    # (assuming your channels are [ref, trans, dist]; adjust indexing if needed)  
    x0, y0, freq_MHz = s0.x_ant, s0.y_ant, s0.freq_MHz
    fspl0   = calculate_fspl(dist0, s0.freq_MHz)  

    c0 = calculate_combined_loss(  
        ref0, trans0, x0, y0,  
        n_angles=360*128,  
        max_reflections=0,  
        max_transmissions=10  
    )  
    t0 = calculate_transmission_loss_numpy(  
        trans0, x0, y0, freq_MHz,
        n_angles=360*128,  
        radial_step=1.0,  
        max_walls=10  
    )  

    compare_two_matrices(  
        np.minimum(c0 + fspl0, 160.0),  
        np.minimum(t0 + fspl0, 160.0),  
        title1="Combined (0 reflections)",  
        title2="Transmission Only",  
        save_path="val.png"  
    )  

    # 4) compute & print RMSE  
    rmses_c = [rmse(p, s.output_img) for p, s in zip(preds_comb,  samples)]  
    rmses_t = [rmse(p, s.output_img) for p, s in zip(preds_trans, samples)]  
    print(f"RMSE (combined)   : {np.mean(rmses_c):.3f}")  
    print(f"RMSE (transmission): {np.mean(rmses_t):.3f}")  

    # 5) optional per-sample debug of the first one  
    debug_combined_method(samples[0])  

    # 6) final visualization  
    visualize_predictions(
        samples,
        [s.output_img for s in samples],     # ground truths
        preds_comb,                          # combined preds
        preds_trans,                         # transmission‐only preds
        n=3,                                 # how many rows to plot
        save_path="viz.png"
    )
