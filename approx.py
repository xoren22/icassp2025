from tqdm import tqdm
import torch, numpy as np
from numba import njit, prange
from helper import (RadarSample, load_samples, rmse, visualize_predictions, compare_two_matrices, compare_hit_counts)

# ---------------------------------------------------------------------#
#  GLOBALS                                                             #
# ---------------------------------------------------------------------#
MAX_REFL  = 5            # reflection budget for normal runs
MAX_TRANS = 15           # transmission (wall) budget

# ---------------------------------------------------------------------#
#  NUMERIC BASICS                                                      #
# ---------------------------------------------------------------------#
def calculate_fspl(dist_m, freq_MHz, min_dist_m=0.125):
    dist_clamped = np.maximum(dist_m, min_dist_m)
    return 20.0*np.log10(dist_clamped) + 20.0*np.log10(freq_MHz) - 27.55

@njit(inline='always')
def _fspl(dist_m: float, freq_MHz: float, min_dist_m: float = 0.125) -> float:
    d = dist_m if dist_m > min_dist_m else min_dist_m
    return 20.0*np.log10(d) + 20.0*np.log10(freq_MHz) - 27.55

@njit(inline='always')
def _euclidean_distance(px, py, x_ant, y_ant, pixel_size=0.25):
    return np.hypot(px - x_ant, py - y_ant) * pixel_size

# ---------------------------------------------------------------------#
#  STEP-UNTIL-WALL                                                     #
# ---------------------------------------------------------------------#
@njit(inline='always')
def _step_until_wall(mat, x0, y0, dx, dy, radial_step, max_dist):
    h, w = mat.shape
    x, y = x0, y0
    px_prev = int(round(x0));  py_prev = int(round(y0))
    last_val = mat[py_prev, px_prev]
    travelled = 0.0

    while travelled <= max_dist:
        x += dx * radial_step
        y += dy * radial_step
        travelled += radial_step
        px, py = int(round(x)), int(round(y))

        if px < 0 or px >= w or py < 0 or py >= h:
            return -1, -1, -1, -1, travelled, last_val, last_val

        cur_val = mat[py, px]
        if cur_val != last_val:
            return px, py, px_prev, py_prev, travelled, last_val, cur_val

        px_prev, py_prev = px, py
        last_val = cur_val

    return -1, -1, -1, -1, travelled, last_val, last_val

# ---------------------------------------------------------------------#
#  PCA-BASED NORMALS (unchanged)                                       #
# ---------------------------------------------------------------------#
@njit(parallel=True, cache=True)
def _precompute_normals_pca(refl, win=10):
    h, w = refl.shape
    nx_img = np.zeros(refl.shape, np.float64)
    ny_img = np.zeros(refl.shape, np.float64)

    for py in prange(h):
        for px in range(w):
            if refl[py, px] == 0:
                continue
            y0 = max(py-win,0); y1 = min(py+win+1,h)
            x0 = max(px-win,0); x1 = min(px+win+1,w)
            patch = refl[y0:y1, x0:x1]
            ys, xs = np.nonzero(patch)
            if xs.size < 2:  # too few pixels
                continue
            xs = xs.astype(np.float64) + x0
            ys = ys.astype(np.float64) + y0
            mx, my = xs.mean(), ys.mean()
            sxx = syy = sxy = 0.0
            for i in range(xs.size):
                dx = xs[i]-mx; dy = ys[i]-my
                sxx += dx*dx; syy += dy*dy; sxy += dx*dy
            n = xs.size
            sxx/=n; syy/=n; sxy/=n
            ang = 0.5*np.arctan2(2*sxy, sxx-syy)
            nx =  np.sin(ang);  ny = -np.cos(ang)
            mag = np.hypot(nx, ny)
            nx_img[py, px] = nx/mag;  ny_img[py, px] = ny/mag
    return nx_img, ny_img

@njit(inline='always')
def _estimate_normal(nx_img, ny_img, px, py):
    return nx_img[py, px], ny_img[py, px]

@njit(inline='always')
def _reflect_dir(dx, dy, nx, ny):
    dot = dx*nx + dy*ny
    rx = dx - 2.0*dot*nx;  ry = dy - 2.0*dot*ny
    mag = np.hypot(rx, ry)
    return (-dx, -dy) if mag==0 else (rx/mag, ry/mag)

@njit
def _trace_ray_recursive(
    refl_mat, trans_mat, nx_img, ny_img,
    out_img, counts,                    # counts kept for stats
    x0, y0, dx, dy,
    trans_ct, refl_ct,
    acc_loss,                           # dB accumulated so far
    global_r,                           # px distance so far
    pixel_size, freq_MHz,
    radial_step, max_dist,
    max_trans, max_refl, max_loss
):
    # stop if already worse than the budget
    if acc_loss >= max_loss:
        return

    px_hit, py_hit, _, _, travelled, last_val, cur_val = _step_until_wall(
        trans_mat, x0, y0, dx, dy, radial_step, max_dist
    )

    # ─ paint current segment (min-dB rule) ─
    steps = int(travelled / radial_step) + 1
    for s in range(1, steps):                       # s=0 would double-paint
        xi = x0 + dx * radial_step * s
        yi = y0 + dy * radial_step * s
        ix = int(round(xi));  iy = int(round(yi))
        if ix < 0 or ix >= out_img.shape[1] or iy < 0 or iy >= out_img.shape[0]:
            break

        fspl = _fspl((global_r + radial_step * s) * pixel_size, freq_MHz)
        tot  = acc_loss + fspl
        if tot > max_loss:
            tot = max_loss

        # min-dB merge
        if tot < out_img[iy, ix]:
            out_img[iy, ix] = tot
        counts[iy, ix] += 1.0                     # keep hit statistics

    # left the map?
    if px_hit < 0:
        return

    # crossing a wall (air → wall exit)
    if last_val > 0. and cur_val == 0.:
        acc_loss += last_val
        trans_ct += 1
        if acc_loss >= max_loss or trans_ct > max_trans:
            return

    # advance position & distance
    new_x = x0 + dx * travelled
    new_y = y0 + dy * travelled
    new_r = global_r + travelled

    # ─ straight continuation ─
    _trace_ray_recursive(
        refl_mat, trans_mat, nx_img, ny_img,
        out_img, counts,
        new_x, new_y, dx, dy,
        trans_ct, refl_ct,
        acc_loss, new_r,
        pixel_size, freq_MHz,
        radial_step, max_dist,
        max_trans, max_refl, max_loss
    )

    # ─ reflection branch ─
    if refl_ct < max_refl:
        nx, ny = _estimate_normal(nx_img, ny_img, px_hit, py_hit)
        if nx != 0.0 or ny != 0.0:
            rdx, rdy = _reflect_dir(dx, dy, nx, ny)
            _trace_ray_recursive(
                refl_mat, trans_mat, nx_img, ny_img,
                out_img, counts,
                new_x, new_y, rdx, rdy,
                trans_ct, refl_ct + 1,
                acc_loss + refl_mat[py_hit, px_hit],
                new_r,
                pixel_size, freq_MHz,
                radial_step, max_dist,
                max_trans, max_refl, max_loss
            )


@njit(parallel=True, fastmath=True, nogil=True, boundscheck=False)
def calculate_combined_loss(
    reflectance_mat, transmittance_mat,
    x_ant, y_ant, freq_MHz,
    max_refl=MAX_REFL, max_trans=MAX_TRANS,
    pixel_size=0.25,
    n_angles=360*128*1, radial_step=1.0,
    max_loss=160.0, pca_win=10
):
    h, w = reflectance_mat.shape
    # float64 map initialised to worst loss so min() works
    out  = np.full((h, w), max_loss, np.float64)
    cnt  = np.zeros((h, w), np.float32)      # hit counter

    nx_img, ny_img = _precompute_normals_pca(reflectance_mat, pca_win)

    dtheta   = 2.0 * np.pi / n_angles
    max_dist = np.hypot(w, h)
    cos_v    = np.cos(np.arange(n_angles) * dtheta)
    sin_v    = np.sin(np.arange(n_angles) * dtheta)

    for i in prange(n_angles):
        _trace_ray_recursive(
            reflectance_mat, transmittance_mat,
            nx_img, ny_img,
            out, cnt,
            x_ant, y_ant,
            cos_v[i], sin_v[i],
            0, 0,                         # counters
            0.0, 0.0,                     # acc_loss, global_r
            pixel_size, freq_MHz,
            radial_step, max_dist,
            max_trans, max_refl, max_loss
        )

    # fill untouched pixels with direct-FSPL (keeps min-rule semantics)
    for py in prange(h):
        for px in range(w):
            if cnt[py, px] == 0:
                d    = _euclidean_distance(px, py, x_ant, y_ant, pixel_size)
                fspl = _fspl(d, freq_MHz)
                out[py, px] = fspl if fspl < max_loss else max_loss

    return out, cnt


# ---------------------------------------------------------------------#
#  TRANSMISSION-ONLY TRACE                                             #
# ---------------------------------------------------------------------#
@njit(parallel=True, fastmath=True, nogil=True, boundscheck=False)
def calculate_transmission_loss_numpy(trans_mat, x_ant, y_ant, freq_MHz, n_angles=360*128, radial_step=1.0, max_walls=MAX_TRANS, max_loss=160.0, pixel_size=0.25):

    h, w  = trans_mat.shape
    out   = np.full((h,w), max_loss, np.float64)  # Initialize to max_loss like combined method
    cnt   = np.zeros((h,w), np.float64)

    dtheta = 2.0*np.pi / n_angles
    max_dist = np.hypot(w, h)
    cos_v = np.cos(np.arange(n_angles)*dtheta)
    sin_v = np.sin(np.arange(n_angles)*dtheta)

    for i in range(n_angles):
        ct, st = cos_v[i], sin_v[i]
        sum_loss = 0.0; last_val = None; wall_ct = 0; r=0.0
        
        while r<=max_dist:
            x = x_ant + r*ct; y = y_ant + r*st
            px = int(round(x)); py = int(round(y))

            # outside?
            if px<0 or px>=w or py<0 or py>=h:
                if last_val is not None and last_val>0:
                    sum_loss = min(sum_loss+last_val, max_loss)
                break

            val = trans_mat[py,px]
            if last_val is None:
                last_val = val
            if val!=last_val and last_val>0 and val==0:
                sum_loss += last_val
                wall_ct += 1
                if sum_loss>=max_loss or wall_ct>=max_walls:
                    sum_loss = min(sum_loss, max_loss)
                    # Continue to fill this pixel before breaking
                    fspl = _fspl(r * pixel_size, freq_MHz)
                    tot = sum_loss + fspl
                    tot = max_loss if tot > max_loss else tot
                    # Use min-dB rule like combined method
                    if tot < out[py,px]:
                        out[py,px] = tot
                    cnt[py,px] += 1.0
                    break
            last_val = val

            # Calculate FSPL for this pixel
            fspl = _fspl(r * pixel_size, freq_MHz)
            tot = sum_loss + fspl
            tot = max_loss if tot > max_loss else tot
            
            # Use min-dB rule like combined method
            if tot < out[py,px]:
                out[py,px] = tot
            cnt[py,px] += 1.0
            r += radial_step

    # Fill untouched pixels with direct-FSPL (same as combined method)
    for py in prange(h):
        for px in range(w):
            if cnt[py, px] == 0:
                d    = _euclidean_distance(px, py, x_ant, y_ant, pixel_size)
                fspl = _fspl(d, freq_MHz)
                out[py,px] = fspl if fspl < max_loss else max_loss

    return out, cnt

# ---------------------------------------------------------------------#
#  APPROX WRAPPER                                                      #
# ---------------------------------------------------------------------#
class Approx:
    def __init__(self, method='combined'):
        self.method = method

    def approximate(self, sample: RadarSample,
                    max_trans=MAX_TRANS, max_refl=MAX_REFL):
        ref, trans, _ = sample.input_img.cpu().numpy()
        x, y, f = sample.x_ant, sample.y_ant, sample.freq_MHz

        if self.method == 'combined':
            feat, _ = calculate_combined_loss(ref, trans, x, y, f,
                                           max_refl=max_refl,
                                           max_trans=max_trans
                                        )
        elif self.method == 'beamtrace':
            from beamtrace import calculate_beamtrace_loss  # local import to avoid cyclic jit cost
            feat, _ = calculate_beamtrace_loss(ref, trans, x, y, f,
                                              max_refl=max_refl,
                                              max_trans=max_trans)
        else:
            feat, _ = calculate_transmission_loss_numpy(trans, x, y, f,
                                                     max_walls=max_trans)
        feat = np.minimum(feat, 160.0)
        return torch.from_numpy(np.floor(feat))

    def predict(self, samples, max_trans=MAX_TRANS, max_refl=MAX_REFL):
        return [self.approximate(s, max_trans, max_refl) for s in tqdm(samples, "predicting")]


# ---------------------------------------------------------------------#
#  MAIN                                                                #
# ---------------------------------------------------------------------#
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Approximation demo")
    parser.add_argument("-N", type=int, default=10, help="number of random samples to load")
    parser.add_argument("--compare0", action="store_true",
                        help="generate zero-reflection combined vs transmission comparison (val.png)")
    parser.add_argument("--hits", action="store_true",
                        help="generate hit-count plots (hit_counts.png)")
    args = parser.parse_args()

    N = args.N
    samples = load_samples(num_samples=N)

    # --- optional zero-reflection / hit-count diagnostics ---
    if args.compare0 or args.hits:
        s0 = samples[0]
        ref, trans, _ = s0.input_img.cpu().numpy()
        x, y, f = s0.x_ant, s0.y_ant, s0.freq_MHz

        tx_map, tx_cnt = calculate_transmission_loss_numpy(
            trans, x, y, f, n_angles=360*128,
            max_walls=MAX_TRANS)

        cmb_map, cmb_cnt = calculate_combined_loss(
            ref, trans, x, y, f,
            max_refl=0, max_trans=MAX_TRANS,
            n_angles=360*128)

        if args.compare0:
            compare_two_matrices(cmb_map, tx_map,
                                 title1="Combined (0 reflections)",
                                 title2="Transmission Only",
                                 save_path="val.png")

        if args.hits:
            compare_hit_counts(tx_cnt, cmb_cnt, save="hit_counts.png")

    # --- full-budget predictions for RMSE ---
    comb_model = Approx("combined")
    beam_model = Approx("beamtrace")

    preds_comb = comb_model.predict(samples, max_refl=MAX_REFL)
    preds_beam = beam_model.predict(samples, max_refl=MAX_REFL)

    rms_c = [rmse(p, s.output_img) for p, s in zip(preds_comb, samples)]
    rms_b = [rmse(p, s.output_img) for p, s in zip(preds_beam, samples)]

    print(f"RMSE (combined) : {np.mean(rms_c):.3f}")
    print(f"RMSE (beam)     : {np.mean(rms_b):.3f}")

    # ────────────────────────────────────────────────────────────
    # Visual comparison (generic): GT + each approximator + all
    # unique pairwise diff images. Diff is shown as |A-B| in gray.
    # ────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import itertools

    n_show = min(3, N)
    base_names = ["GT", "Combined", "Beam"]

    for idx in range(n_show):
        mats = [samples[idx].output_img, preds_comb[idx], preds_beam[idx]]

        # pairwise diffs
        diffs = []
        diff_titles = []
        for (a, b) in itertools.combinations(range(len(mats)), 2):
            diffs.append(np.abs(mats[a] - mats[b]))
            diff_titles.append(f"|{base_names[a][0]}-{base_names[b][0]}|")

        row_mats   = mats + diffs
        row_titles = base_names.copy()
        # append RMSE for approximators in titles
        row_titles[1] = f"Combined ({rmse(mats[1], mats[0]):.2f})"
        row_titles[2] = f"Beam ({rmse(mats[2], mats[0]):.2f})"
        row_titles.extend(diff_titles)

        if idx == 0:
            # create figure with dynamic column count after knowing sizes
            n_cols = len(row_mats)
            fig, axes = plt.subplots(n_show, n_cols, figsize=(4 * n_cols, 4 * n_show))

        for col, (mat, title) in enumerate(zip(row_mats, row_titles)):
            ax = axes[idx, col]
            if col < len(mats):
                im = ax.imshow(mat, vmax=160)
            else:
                im = ax.imshow(mat, cmap='gray')  # diff
            ax.set_title(title)
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.04)

    plt.tight_layout(); plt.savefig("viz.png", dpi=150); plt.close(fig)

    saved = ["viz.png"]
    if args.compare0:
        saved.append("val.png")
    if args.hits:
        saved.append("hit_counts.png")
    print("Saved:", "  ".join(saved))
