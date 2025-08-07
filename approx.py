from tqdm import tqdm
import torch, numpy as np
from numba import njit, prange
from helper import (RadarSample, load_samples, rmse, visualize_predictions, compare_two_matrices, compare_hit_counts)
import os

_WARMED_UP = False
# ---------------------------------------------------------------------#
#  NORMAL MAP LOADING                                                   #
# ---------------------------------------------------------------------#
def load_precomputed_normals_for_building(building_id: int, refl: np.ndarray, trans: np.ndarray):
    """
    Load precomputed wall normals from parsed_buildings/B{building_id}_normals.npz
    and return (nx_img, ny_img) as float64 arrays.

    Raises FileNotFoundError if the file is missing so the user can run
    parse_buildings.py first.
    """
    h, w = refl.shape
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parsed_dir = os.path.join(base_dir, "parsed_buildings")
    npz_path = os.path.join(parsed_dir, f"B{building_id}_normals.npz")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Precomputed normals not found for building {building_id}: {npz_path}. "
            f"Run parse_buildings.py to generate them."
        )

    data = np.load(npz_path)
    nx = np.ascontiguousarray(data["nx"], dtype=np.float64)
    ny = np.ascontiguousarray(data["ny"], dtype=np.float64)
    if nx.shape != (h, w) or ny.shape != (h, w):
        raise ValueError(
            f"Normal map shape mismatch for building {building_id}: expected {(h,w)}, got {nx.shape}"
        )
    return nx, ny

# ---------------------------------------------------------------------#
#  GLOBALS                                                             #
# ---------------------------------------------------------------------#
MAX_REFL  = 5            # reflection budget for normal runs
MAX_TRANS = 15           # transmission (wall) budget
N_ANGLES  = 360*32       # single place to control angular resolution for combined method
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

# Fast FSPL lookup table to avoid log10 in inner loops
@njit(cache=True)
def _build_fspl_lut(max_steps: int, pixel_size: float, freq_MHz: float, min_dist_m: float = 0.125):
    lut = np.empty(max_steps + 1, np.float64)
    for k in range(max_steps + 1):
        d = k * pixel_size
        if d < min_dist_m:
            d = min_dist_m
        lut[k] = 20.0*np.log10(d) + 20.0*np.log10(freq_MHz) - 27.55
    return lut

@njit(inline='always')
def _fspl_from_lut(lut: np.ndarray, step_index: int) -> float:
    # clamps index to [0, len-1]
    if step_index < 0:
        step_index = 0
    n = lut.shape[0]
    if step_index >= n:
        step_index = n - 1
    return lut[step_index]

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
#  NORMALS USAGE                                                       #
# ---------------------------------------------------------------------#
@njit(inline='always')
def _estimate_normal(nx_img, ny_img, px, py):
    return nx_img[py, px], ny_img[py, px]

@njit(inline='always')
def _reflect_dir(dx, dy, nx, ny):
    dot = dx*nx + dy*ny
    rx = dx - 2.0*dot*nx;  ry = dy - 2.0*dot*ny
    mag = np.hypot(rx, ry)
    return (-dx, -dy) if mag==0 else (rx/mag, ry/mag)

@njit(cache=True)
def _trace_ray_recursive(
    refl_mat, trans_mat, nx_img, ny_img,
    out_img, counts,                    # counts kept for stats
    x0, y0, dx, dy,
    trans_ct, refl_ct,
    acc_loss,                           # dB accumulated so far
    global_r,                           # px distance so far
    pixel_size, freq_MHz,
    radial_step, max_dist,
    max_trans, max_refl, max_loss,
    fspl_lut, use_lut, paint_stride
):
    # stop if already worse than the budget
    if acc_loss >= max_loss:
        return

    px_hit, py_hit, _, _, travelled, last_val, cur_val = _step_until_wall(
        trans_mat, x0, y0, dx, dy, radial_step, max_dist
    )

    # ─ paint current segment (min-dB rule) ─
    steps = int(travelled / radial_step) + 1
    for s in range(paint_stride, steps, paint_stride):  # s=0 would double-paint
        xi = x0 + dx * radial_step * s
        yi = y0 + dy * radial_step * s
        ix = int(round(xi));  iy = int(round(yi))
        if ix < 0 or ix >= out_img.shape[1] or iy < 0 or iy >= out_img.shape[0]:
            break

        if use_lut:
            k = int(global_r + s)  # valid when radial_step == 1.0
            fspl = _fspl_from_lut(fspl_lut, k)
        else:
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
        max_trans, max_refl, max_loss,
        fspl_lut, use_lut, paint_stride
    )

    # ─ reflection branch ─
    if refl_ct < max_refl:
        # Reflect only on reflective walls (guard against normals from pure transmission)
        refl_val = refl_mat[py_hit, px_hit]
        if refl_val > 0.0:
            nx, ny = _estimate_normal(nx_img, ny_img, px_hit, py_hit)
            if nx != 0.0 or ny != 0.0:
                rdx, rdy = _reflect_dir(dx, dy, nx, ny)
                _trace_ray_recursive(
                    refl_mat, trans_mat, nx_img, ny_img,
                    out_img, counts,
                    new_x, new_y, rdx, rdy,
                    trans_ct, refl_ct + 1,
                    acc_loss + refl_val,
                    new_r,
                    pixel_size, freq_MHz,
                    radial_step, max_dist,
                    max_trans, max_refl, max_loss,
                    fspl_lut, use_lut, paint_stride
                )


# Variant that uses provided normal maps (from precomputed data)
@njit(parallel=True, fastmath=True, nogil=True, boundscheck=False, cache=True)
def calculate_combined_loss_with_normals(
    reflectance_mat, transmittance_mat, nx_img, ny_img,
    x_ant, y_ant, freq_MHz,
    n_angles,
    max_refl=MAX_REFL, max_trans=MAX_TRANS,
    pixel_size=0.25,
    radial_step=1.0,
    max_loss=160.0,
    paint_stride=1,
    use_fspl_lut=True
):
    h, w = reflectance_mat.shape
    out  = np.full((h, w), max_loss, np.float32)
    cnt  = np.zeros((h, w), np.float32)

    dtheta   = 2.0 * np.pi / n_angles
    max_dist = np.hypot(w, h)
    cos_v    = np.cos(np.arange(n_angles) * dtheta)
    sin_v    = np.sin(np.arange(n_angles) * dtheta)

    # FSPL LUT only valid when radial_step == 1.0 (indexes by integer step count)
    use_lut = use_fspl_lut and (radial_step == 1.0)
    if use_lut:
        max_steps = int(max_dist) + 2
        fspl_lut = _build_fspl_lut(max_steps, pixel_size, freq_MHz)
    else:
        fspl_lut = np.zeros(1, np.float64)  # dummy to satisfy typing

    for i in prange(n_angles):
        _trace_ray_recursive(
            reflectance_mat, transmittance_mat,
            nx_img, ny_img,
            out, cnt,
            x_ant, y_ant,
            cos_v[i], sin_v[i],
            0, 0,
            0.0, 0.0,
            pixel_size, freq_MHz,
            radial_step, max_dist,
            max_trans, max_refl, max_loss,
            fspl_lut, use_lut, paint_stride
        )

    # fill untouched pixels with direct-FSPL
    for py in prange(h):
        for px in range(w):
            if cnt[py, px] == 0:
                d    = _euclidean_distance(px, py, x_ant, y_ant, pixel_size)
                fspl = _fspl(d, freq_MHz)
                out[py, px] = fspl if fspl < max_loss else max_loss

    return out, cnt

def _warmup_numba_once():
    global _WARMED_UP
    if _WARMED_UP:
        return
    h, w = 8, 8
    zero = np.zeros((h, w), np.float64)
    # simple walls: single vertical line
    trans = zero.copy(); trans[:, 4] = 1.0
    nx = zero.copy(); ny = zero.copy(); ny[:, 4] = 1.0
    calculate_combined_loss_with_normals(
        zero, trans, nx, ny,
        x_ant=4.0, y_ant=4.0, freq_MHz=1000.0,
        n_angles=N_ANGLES,
        max_refl=0, max_trans=1,
        radial_step=1.0,
        max_loss=160.0,
        paint_stride=1,
        use_fspl_lut=True
    )
    _WARMED_UP = True

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
        _warmup_numba_once()

    def approximate(self, sample: RadarSample,
                    max_trans=MAX_TRANS, max_refl=MAX_REFL):
        ref, trans, _ = sample.input_img.cpu().numpy()
        x, y, f = sample.x_ant, sample.y_ant, sample.freq_MHz
        # Precomputed normals
        building_id = sample.ids[0] if sample.ids else 0
        nx_img, ny_img = load_precomputed_normals_for_building(building_id, ref, trans)
        # Ensure contiguous dtypes for numba
        ref_c  = np.ascontiguousarray(ref, dtype=np.float64)
        trans_c = np.ascontiguousarray(trans, dtype=np.float64)

        if self.method == 'combined':
            feat, _ = calculate_combined_loss_with_normals(
                ref_c, trans_c, nx_img, ny_img,
                x, y, f,
                n_angles=N_ANGLES,
                max_refl=max_refl,
                max_trans=max_trans,
                radial_step=1.0,
                paint_stride=1,
                use_fspl_lut=True
            )
        elif self.method == 'combined_fast':
            # Aggressive speed settings; expect some accuracy loss
            fast_refl = max_refl if max_refl < 3 else 3
            feat, _ = calculate_combined_loss_with_normals(
                ref_c, trans_c, nx_img, ny_img,
                x, y, f,
                n_angles=N_ANGLES,
                max_refl=fast_refl,
                max_trans=max_trans,
                radial_step=1.0,
                paint_stride=8,
                use_fspl_lut=True
            )
        elif self.method == 'beamtrace':
            from beamtrace import calculate_beamtrace_loss_with_normals  # local import to avoid cyclic jit cost
            feat, _ = calculate_beamtrace_loss_with_normals(
                ref_c, trans_c, nx_img, ny_img, x, y, f,
                max_refl=max_refl, max_trans=max_trans)
        else:
            feat, _ = calculate_transmission_loss_numpy(trans_c, x, y, f, max_walls=max_trans)
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

        # use precomputed normals even for zero-reflection comparison
        b_id = s0.ids[0] if s0.ids else 0
        nx0, ny0 = load_precomputed_normals_for_building(b_id, ref, trans)
        cmb_map, cmb_cnt = calculate_combined_loss_with_normals(
            ref, trans, nx0, ny0, x, y, f,
            n_angles=N_ANGLES,
            max_refl=0, max_trans=MAX_TRANS,
            radial_step=1.0,
            paint_stride=1,
            use_fspl_lut=True)

        if args.compare0:
            compare_two_matrices(cmb_map, tx_map,
                                 title1="Combined (0 reflections)",
                                 title2="Transmission Only",
                                 save_path="val.png")

        if args.hits:
            compare_hit_counts(tx_cnt, cmb_cnt, save="hit_counts.png")

    # --- full-budget predictions for RMSE ---
    comb_model = Approx("combined")
    txonly_model = Approx("transmission")

    preds_comb = comb_model.predict(samples, max_refl=MAX_REFL)
    preds_tx   = txonly_model.predict(samples, max_refl=0)

    rms_c = [rmse(p, s.output_img) for p, s in zip(preds_comb, samples)]
    rms_t = [rmse(p, s.output_img) for p, s in zip(preds_tx, samples)]

    print(f"RMSE (combined) : {np.mean(rms_c):.3f}")
    print(f"RMSE (tx-only)  : {np.mean(rms_t):.3f}")

    # ────────────────────────────────────────────────────────────
    # Visual comparison (generic): GT + each approximator + all
    # unique pairwise diff images. Diff is shown as |A-B| in gray.
    # ────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import itertools

    n_show = min(3, N)
    base_names = ["GT", "Combined", "Tx-Only"]

    for idx in range(n_show):
        mats = [samples[idx].output_img, preds_comb[idx], preds_tx[idx]]

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
        row_titles[2] = f"Tx-Only ({rmse(mats[2], mats[0]):.2f})"
        row_titles.extend(diff_titles)

        if idx == 0:
            # create figure with dynamic column count after knowing sizes
            n_cols = len(row_mats)
            fig, axes = plt.subplots(n_show, n_cols, figsize=(4 * n_cols, 4 * n_show))
            import numpy as _np
            axes = _np.atleast_2d(axes)

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
