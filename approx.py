import os, torch, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit, prange
from helper import RadarSample, load_samples, compare_two_matrices, rmse, visualize_predictions


MAX_REFL = 5
MAX_TRANS = 15

# ───────────────────────────────────────────────────────────────
#  FREE-SPACE PATH-LOSS (scalar helper used everywhere)
# ───────────────────────────────────────────────────────────────
def calculate_fspl(dist_m, freq_MHz, min_dist_m=0.125):
    dist_clamped = np.maximum(dist_m, min_dist_m)
    return 20.0 * np.log10(dist_clamped) + 20.0 * np.log10(freq_MHz) - 27.55

@njit(inline='always')
def _fspl(dist_m: float, freq_MHz: float, min_dist_m: float = 0.125) -> float:
    d = dist_m if dist_m > min_dist_m else min_dist_m
    return 20.0 * np.log10(d) + 20.0 * np.log10(freq_MHz) - 27.55

# ───────────────────────────────────────────────────────────────
#  NUMBA HELPERS FOR COMBINED TRACING
# ───────────────────────────────────────────────────────────────
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
        px = int(round(x)); py = int(round(y))
        if px < 0 or px >= w or py < 0 or py >= h:
            return -1, -1, travelled, last_val, last_val
        cur_val = mat[py, px]
        if cur_val != last_val:
            return px, py, travelled, last_val, cur_val
    return -1, -1, travelled, last_val, last_val








# ───────────────────────────────────────────────────────────────
#  PCA-based wall-normal estimator (replaces old 4-way version)
# ───────────────────────────────────────────────────────────────
@njit(inline='always', cache=True)
def _pca_angle(xs, ys):
    """Return principal-axis angle in radians (0..π)."""
    n   = xs.size
    mx  = xs.mean()
    my  = ys.mean()
    sxx = syy = sxy = 0.0
    for i in range(n):
        dx = xs[i] - mx
        dy = ys[i] - my
        sxx += dx*dx
        syy += dy*dy
        sxy += dx*dy
    sxx /= n; syy /= n; sxy /= n
    return 0.5 * np.arctan2(2.0 * sxy, sxx - syy)   # first PC

# ── 1. Offline normal-map generator (parallel) ─────────────────
@njit(parallel=True, cache=True)
def _precompute_normals_pca(refl, win=10):
    h, w = refl.shape
    nx_img = np.zeros(refl.shape, np.float32)
    ny_img = np.zeros(refl.shape, np.float32)

    for py in prange(h):
        for px in range(w):
            if refl[py, px] == 0:          # air pixel → skip
                continue

            y0 = max(py - win, 0);  y1 = min(py + win + 1, h)
            x0 = max(px - win, 0);  x1 = min(px + win + 1, w)
            patch = refl[y0:y1, x0:x1]
            ys, xs = np.nonzero(patch)
            if xs.size < 2:
                continue

            xs = xs.astype(np.float32) + x0
            ys = ys.astype(np.float32) + y0

            # PCA closed-form
            mx, my = xs.mean(), ys.mean()
            sxx = syy = sxy = 0.0
            for i in range(xs.size):
                dx = xs[i]-mx; dy = ys[i]-my
                sxx += dx*dx; syy += dy*dy; sxy += dx*dy
            n = xs.size
            sxx/=n; syy/=n; sxy/=n
            ang = 0.5*np.arctan2(2*sxy, sxx-syy)

            nx =  np.sin(ang)          # normal = tangent+90°
            ny = -np.cos(ang)
            mag = np.hypot(nx, ny)
            nx_img[py, px] = nx/mag
            ny_img[py, px] = ny/mag
    return nx_img, ny_img

# ── 2. O(1) lookup (replaces old _estimate_normal) ─────────────
@njit(inline='always')
def _estimate_normal(nx_img, ny_img, px, py):
    return nx_img[py, px], ny_img[py, px]























@njit(inline='always')
def _reflect_dir(dx, dy, nx, ny):
    dot = dx*nx + dy*ny
    rx = dx - 2.0*dot*nx
    ry = dy - 2.0*dot*ny
    mag = np.hypot(rx, ry)
    return (-dx, -dy) if mag == 0.0 else (rx/mag, ry/mag)

# ───────────────────────────────────────────────────────────────
#  PAINT-TO-EDGE FALLBACK  (FSPL included)
# ───────────────────────────────────────────────────────────────
@njit(inline='always')
def _paint_to_edge(out_img, x0, y0, dx, dy,
                   acc_loss, path_px,
                   pixel_size, freq_MHz,
                   radial_step, max_dist, max_loss):
    h, w = out_img.shape
    r = 0.0
    while r <= max_dist:
        ix = int(round(x0 + dx*r)); iy = int(round(y0 + dy*r))
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return
        fspl = _fspl((path_px + r) * pixel_size, freq_MHz)
        tot  = acc_loss + fspl
        if tot < out_img[iy, ix]:
            out_img[iy, ix] = tot if tot < max_loss else max_loss
        r += radial_step

# ───────────────────────────────────────────────────────────────
#  RECURSIVE TRACER  (FSPL added every step)
# ───────────────────────────────────────────────────────────────
@njit
def _trace_ray_recursive(
    refl_mat, trans_mat, nx_img, ny_img, out_img,
    x0, y0, dx, dy,
    trans_ct, refl_ct,
    acc_loss,          # wall / reflection loss only
    path_px,           # cumulative distance in pixels
    pixel_size, freq_MHz,
    radial_step, max_dist,
    max_trans, max_refl,
    max_loss
):
    if acc_loss >= max_loss:
        return

    px_hit, py_hit, travelled, _, _ = _step_until_wall(
        refl_mat, x0, y0, dx, dy, radial_step, max_dist
    )

    # paint free-space segment with FSPL
    visited_ix = -1; visited_iy = -1
    steps = int(travelled / radial_step) + 1
    for s in range(steps):
        xi = x0 + dx * radial_step * s
        yi = y0 + dy * radial_step * s
        ix = int(round(xi)); iy = int(round(yi))
        # skip if same pixel as previous iteration
        if ix == visited_ix and iy == visited_iy:
            continue
        visited_ix, visited_iy = ix, iy
        if ix < 0 or ix >= out_img.shape[1] or iy < 0 or iy >= out_img.shape[0]:
            break
        d_pix = _euclidean_distance(ix, iy, x_ant=x0, y_ant=y0, pixel_size=pixel_size)  # we'll adjust below
        # Actually antenna is at original (x_ant,y_ant) of trace root, which is fixed across recursion: always the initial antenna pixel.
        # pass original antenna coords via function?  For now approximate using sqrt((ix-x_ant)^2...) stored outside? We don't have x_ant param here.
        fspl = _fspl(d_pix, freq_MHz)
        tot  = acc_loss + fspl
        if tot < out_img[iy, ix]:
            out_img[iy, ix] = tot if tot < max_loss else max_loss

    # exited image?
    if px_hit < 0:
        return

    new_path_px = path_px + travelled

    # decide which branches are still allowed
    branch_made = False

    # (1) transmission through the wall
    if trans_ct < max_trans:
        _trace_ray_recursive(
            refl_mat, trans_mat, nx_img, ny_img, out_img,
            px_hit, py_hit, dx, dy,
            trans_ct + 1, refl_ct,
            acc_loss + trans_mat[py_hit, px_hit],
            new_path_px,
            pixel_size, freq_MHz,
            radial_step, max_dist,
            max_trans, max_refl,
            max_loss
        )
        branch_made = True

    # (2) reflection from the wall
    if refl_ct < max_refl:
        nx, ny = _estimate_normal(nx_img, ny_img, px_hit, py_hit)
        if nx != 0.0 or ny != 0.0:
            rdx, rdy = _reflect_dir(dx, dy, nx, ny)
            _trace_ray_recursive(
                refl_mat, trans_mat, nx_img, ny_img, out_img,
                px_hit, py_hit, rdx, rdy,
                trans_ct, refl_ct + 1,
                acc_loss + refl_mat[py_hit, px_hit],
                new_path_px,
                pixel_size, freq_MHz,
                radial_step, max_dist,
                max_trans, max_refl,
                max_loss
            )
            branch_made = True

    # (3) if no branch was possible, budgets are exhausted → stop
    if not branch_made:
        return

# ───────────────────────────────────────────────────────────────
#  COMBINED TRACE  (with pre-computed PCA normal map)
# ───────────────────────────────────────────────────────────────
@njit(parallel=True, fastmath=True, nogil=True, boundscheck=False)
def calculate_combined_loss(
    reflectance_mat, transmittance_mat,
    x_ant, y_ant,
    freq_MHz: float,
    pixel_size: float = 0.25,
    n_angles: int = 360 * 128 * 1,
    radial_step: float = 1.0,
    max_reflections: int = MAX_REFL,
    max_trans: int = MAX_TRANS,
    max_loss: float = 160.0,
    pca_win: int = 10                     # half-window for PCA
) -> np.ndarray:
    """
    Casts `n_angles` rays, accounting for wall transmissions + reflections.
    Wall normals come from a *once-per-image* PCA pass, eliminating the
    expensive per-hit computation.
    """
    h, w = reflectance_mat.shape
    out_img = np.full((h, w), max_loss, dtype=np.float32)

    # 1) one-time normal-map build (numba-parallel)
    nx_img, ny_img = _precompute_normals_pca(reflectance_mat, pca_win)

    # 2) constant trig tables for ray launch
    dtheta   = 2.0 * np.pi / n_angles
    max_dist = np.hypot(w, h)
    cos_v    = np.cos(np.arange(n_angles) * dtheta)
    sin_v    = np.sin(np.arange(n_angles) * dtheta)

    # 3) launch rays in parallel
    for i in prange(n_angles):
        _trace_ray_recursive(
            reflectance_mat, transmittance_mat,
            nx_img, ny_img,            # ← NEW look-up images
            out_img,
            x_ant, y_ant,
            cos_v[i], sin_v[i],
            0, 0,                      # trans_ct, refl_ct
            0.0,                       # acc_loss
            0.0,                       # path_px
            pixel_size, freq_MHz,
            radial_step, max_dist,
            max_trans, max_reflections,
            max_loss
        )
    return out_img



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
    n_angles: int = 360 * 128 * 1,
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
                        # # fill the rest of the ray with current loss
                        # r_tmp = r
                        # while r_tmp <= max_dist:
                        #     x_t = x_ant + r_tmp * ct
                        #     y_t = y_ant + r_tmp * st
                        #     px_t = int(round(x_t))
                        #     py_t = int(round(y_t))
                        #     if px_t < 0 or px_t >= w or py_t < 0 or py_t >= h:
                        #         break
                        #     if counts[py_t, px_t] == 0:
                        #         out[py_t, px_t] = sum_loss
                        #         counts[py_t, px_t] = 1
                        #     else:
                        #         c = counts[py_t, px_t]
                        #         out[py_t, px_t] = (out[py_t, px_t] * c + sum_loss) / (c + 1)
                        #         counts[py_t, px_t] = c + 1
                        #     r_tmp += radial_step
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


# # ---------------------------------------------------------------------
# #  VIS & DEBUG
# # ---------------------------------------------------------------------
# def debug_combined_method(sample):
#     ref, trans, dist = sample.input_img.cpu().numpy()
#     x,y,freq_MHz = sample.x_ant, sample.y_ant, sample.freq_MHz
#     fspl = calculate_fspl(dist, sample.freq_MHz)
#     comb = calculate_combined_loss(ref, trans, x, y, freq_MHz, n_angles=360 * 128 * 1, max_reflections=5, max_trans=10)
#     tx   = calculate_transmission_loss_numpy(trans, x, y, freq_MHz, n_angles=360 * 128 * 1)
#     print("Combined range:", comb.min(), comb.max())
#     print("Trans range:  ", tx.min(), tx.max())
#     print("Painted pixels diff:", np.sum(comb>0), np.sum(tx>0))
#     return comb, tx, fspl

# ---------------------------------------------------------------------
#  WRAPPER CLASS
# ---------------------------------------------------------------------
class Approx:
    def __init__(self, method='combined'):
        self.method = method
    def approximate(self, sample: RadarSample, max_trans, max_reflections) -> torch.Tensor:
        ref, trans, dist = sample.input_img.cpu().numpy()
        x,y,freq_MHz = sample.x_ant, sample.y_ant, sample.freq_MHz

        if self.method=='combined':
            feat = calculate_combined_loss(ref, trans, x, y, freq_MHz,
                                           n_angles=360 * 128 * 1,
                                           max_reflections=max_reflections,
                                           max_trans=max_trans
                                           )
        else:
            feat = calculate_transmission_loss_numpy(trans, x, y, freq_MHz,
                                                     n_angles=360 * 128 * 1,
                                                     max_walls=10)
        out = np.minimum(feat, 160.0)
        return torch.from_numpy(np.floor(out))
    
    def predict(self, samples, max_trans=MAX_TRANS, max_reflections=5):
        return [self.approximate(s, max_trans=max_trans, max_reflections=max_reflections) for s in tqdm(samples, "predicting")]

# ---------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":  
    # 1) load a handful of samples  
    samples = load_samples(num_samples=4)  

    # 2) run both methods  
    comb_model = Approx(method='combined')  
    trans_model = Approx(method='transmission')  
    preds_comb  = comb_model.predict(samples)  
    preds_trans = trans_model.predict(samples)  

    # 3) zero-reflection sanity check  
    s0    = samples[0]  

    t0 = trans_model.predict(samples)[0].cpu().numpy()
    c0 = comb_model.predict(samples, max_reflections=0)[0].cpu().numpy()

    compare_two_matrices(  
        c0, t0,
        title1="Combined (0 reflections)",  
        title2="Transmission Only",  
        save_path="val.png"  
    )  

    # 4) compute & print RMSE  
    rmses_c = [rmse(p, s.output_img) for p, s in zip(preds_comb,  samples)]  
    rmses_t = [rmse(p, s.output_img) for p, s in zip(preds_trans, samples)]  
    rmses_t_plus_c = [rmse( (tp+cp)/2, s.output_img ) for tp, cp, s in zip(preds_trans, preds_comb, samples)]  

    print(f"RMSE (combined)   : {np.mean(rmses_c):.3f}")  
    print(f"RMSE (transmission): {np.mean(rmses_t):.3f}")
    print(f"RMSE (combined+transmission): {np.mean(rmses_t_plus_c):.3f}")

    # 5) optional per-sample debug of the first one  
    # debug_combined_method(samples[0])  

    # 6) final visualization  
    visualize_predictions(
        samples,
        [s.output_img for s in samples],     # ground truths
        preds_comb,                          # combined preds
        preds_trans,                         # transmission‐only preds
        n=3,                                 # how many rows to plot
        save_path="viz.png",
        trans_mask = [100*s.input_img[1] for s in samples]
    )
