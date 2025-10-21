import os
os.environ["NUMBA_THREADING_LAYER"] = "tbb"
# Limit thread pools from BLAS/OpenMP libraries to avoid oversubscription/crashes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from tqdm import tqdm
import torch, numpy as np
from numba import njit, prange
from helper import (RadarSample, load_samples, rmse, visualize_predictions, compare_two_matrices, compare_hit_counts, read_sample)
from normal_parser import precompute_wall_angles_pca
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as _mp
from concurrent.futures import ThreadPoolExecutor

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
MAX_TRANS = 10           # transmission (wall) budget
N_ANGLES  = 360*128      # single place to control angular resolution for combined method

# Backfill configuration
BACKFILL_METHOD = "los"     # options: "los", "diffuse", or "fspl"
BACKFILL_PARAMS = {
    "iters": 60,           # for diffuse
    "lambda": 0.05,        # for diffuse
    "alpha": 1.0,          # for diffuse
}
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
@njit(cache=False)
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
#  BACKFILL (CPU, numpy)                                               #
# ---------------------------------------------------------------------#

def _compute_fspl_field(h: int, w: int, x_ant: float, y_ant: float, pixel_size: float, freq_MHz: float, max_loss: float):
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx - x_ant, yy - y_ant) * pixel_size
    d = np.maximum(d, 0.125)
    F = 20.0*np.log10(d) + 20.0*np.log10(freq_MHz) - 27.55
    return np.minimum(F, max_loss).astype(np.float32)


def _backfill_fspl(out: np.ndarray, cnt: np.ndarray, x_ant: float, y_ant: float, pixel_size: float, freq_MHz: float, max_loss: float) -> np.ndarray:
    mask0 = (cnt == 0)
    if not np.any(mask0):
        return out
    F = _compute_fspl_field(out.shape[0], out.shape[1], x_ant, y_ant, pixel_size, freq_MHz, max_loss)
    out2 = out.copy()
    out2[mask0] = F[mask0]
    return np.minimum(out2, max_loss)


def _backfill_diffuse_residual(out: np.ndarray, cnt: np.ndarray, x_ant: float, y_ant: float, pixel_size: float, freq_MHz: float, max_loss: float, *, iters: int = 60, lam: float = 0.05, alpha: float = 1.0) -> np.ndarray:
    h, w = out.shape
    mask0 = (cnt == 0)
    if not np.any(mask0):
        return out

    F = _compute_fspl_field(h, w, x_ant, y_ant, pixel_size, freq_MHz, max_loss)
    R = (out.astype(np.float32) - F).astype(np.float32)

    # neighbor weights from counts (normalized 0..1)
    cnt_f = cnt.astype(np.float32)
    cmax = float(cnt_f.max()) if cnt_f.size > 0 else 1.0
    norm = cnt_f / (cmax + 1e-6)
    # shifted weights per direction
    wL = 1.0 + alpha * np.pad(norm[:, 1:], ((0,0),(0,1)), mode='constant')  # left neighbor weight at (y,x)
    wR = 1.0 + alpha * np.pad(norm[:, :-1], ((0,0),(1,0)), mode='constant') # right
    wU = 1.0 + alpha * np.pad(norm[1:, :], ((0,1),(0,0)), mode='constant')  # up
    wD = 1.0 + alpha * np.pad(norm[:-1, :], ((1,0),(0,0)), mode='constant') # down

    # Precompute rolled R views for vectorized Jacobi
    for _ in range(iters):
        R_left  = np.pad(R[:, :-1], ((0,0),(1,0)), mode='edge')
        R_right = np.pad(R[:, 1:],  ((0,0),(0,1)), mode='edge')
        R_up    = np.pad(R[:-1, :], ((1,0),(0,0)), mode='edge')
        R_down  = np.pad(R[1:, :],  ((0,1),(0,0)), mode='edge')

        num = wL * R_left + wR * R_right + wU * R_up + wD * R_down
        den = (wL + wR + wU + wD) + lam
        R_new = R.copy()
        R_new[mask0] = (num[mask0] / den[mask0]).astype(np.float32)
        R = R_new

    out2 = F + R
    # enforce FSPL monotonic floor and cap
    out2 = np.maximum(out2, F)
    out2 = np.minimum(out2, max_loss)
    # preserve known pixels strictly
    out2[~mask0] = out[~mask0]
    return out2.astype(np.float32)


@njit(cache=False)
def _backfill_direct_los(out: np.ndarray, cnt: np.ndarray, trans_mat: np.ndarray, x_ant: float, y_ant: float, pixel_size: float, freq_MHz: float, max_loss: float) -> np.ndarray:
    h, w = out.shape
    out2 = out.copy()
    mask0 = (cnt == 0)

    for py in range(h):
        for px in range(w):
            if not mask0[py, px]:
                continue

            # DDA from antenna to (px,py)
            x0 = x_ant
            y0 = y_ant
            x1 = float(px)
            y1 = float(py)
            ddx = x1 - x0
            ddy = y1 - y0
            adx = ddx if ddx >= 0.0 else -ddx
            ady = ddy if ddy >= 0.0 else -ddy
            steps = int(adx if adx >= ady else ady)

            if steps <= 0:
                dxp = float(px) - x_ant
                dyp = float(py) - y_ant
                dist = np.hypot(dxp, dyp) * pixel_size
                if dist < 0.125:
                    dist = 0.125
                fspl = 20.0 * np.log10(dist) + 20.0 * np.log10(freq_MHz) - 27.55
                tot = fspl
                if tot > max_loss:
                    tot = max_loss
                out2[py, px] = tot
                cnt[py, px] = 1.0
                continue

            sx = ddx / steps
            sy = ddy / steps
            x = x0
            y = y0

            # Initialize last_val from starting position if in bounds
            ix0 = int(np.rint(x))
            iy0 = int(np.rint(y))
            if 0 <= ix0 < w and 0 <= iy0 < h:
                last_val = float(trans_mat[iy0, ix0])
            else:
                last_val = 0.0
            sum_loss = 0.0

            for s in range(steps + 1):
                ix = int(np.rint(x))
                iy = int(np.rint(y))
                if 0 <= ix < w and 0 <= iy < h:
                    val = float(trans_mat[iy, ix])
                    if s == 0:
                        last_val = val
                    if val != last_val and last_val > 0.0 and val == 0.0:
                        sum_loss += last_val
                        if sum_loss >= max_loss:
                            sum_loss = max_loss
                            break
                    last_val = val
                x += sx
                y += sy

            dxp = float(px) - x_ant
            dyp = float(py) - y_ant
            dist = np.hypot(dxp, dyp) * pixel_size
            if dist < 0.125:
                dist = 0.125
            fspl = 20.0 * np.log10(dist) + 20.0 * np.log10(freq_MHz) - 27.55
            tot = sum_loss + fspl
            if tot > max_loss:
                tot = max_loss
            out2[py, px] = tot
            cnt[py, px] = 1.0

    return out2.astype(np.float32)


def apply_backfill(out: np.ndarray, cnt: np.ndarray, x_ant: float, y_ant: float, pixel_size: float, freq_MHz: float, max_loss: float, method: str = BACKFILL_METHOD, params: dict | None = None, *, trans_mat: np.ndarray | None = None) -> np.ndarray:
    if params is None:
        params = BACKFILL_PARAMS
    if method == "fspl":
        return _backfill_fspl(out, cnt, x_ant, y_ant, pixel_size, freq_MHz, max_loss)
    elif method == "diffuse":
        iters = int(params.get("iters", 60))
        lam   = float(params.get("lambda", 0.05))
        alpha = float(params.get("alpha", 1.0))
        return _backfill_diffuse_residual(out, cnt, x_ant, y_ant, pixel_size, freq_MHz, max_loss, iters=iters, lam=lam, alpha=alpha)
    elif method == "los":
        if trans_mat is None:
            raise ValueError("LOS backfill requires trans_mat")
        return _backfill_direct_los(out, cnt, trans_mat, x_ant, y_ant, pixel_size, freq_MHz, max_loss)
    else:
        # default to FSPL if unknown
        return _backfill_fspl(out, cnt, x_ant, y_ant, pixel_size, freq_MHz, max_loss)

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

@njit(cache=False)
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
    fspl_lut, use_lut
):
    # stop if already worse than the budget
    if acc_loss >= max_loss:
        return

    px_hit, py_hit, px_prev, py_prev, travelled, last_val, cur_val = _step_until_wall(
        trans_mat, x0, y0, dx, dy, radial_step, max_dist
    )

    # ─ paint current segment (min-dB rule) ─
    steps = int(travelled / radial_step) + 1
    for s in range(1, steps):  # s=0 would double-paint
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

    # If the segment was too short to paint (travelled < radial_step), paint its endpoint pixel once
    if steps <= 1:
        ix = px_prev; iy = py_prev
        if 0 <= ix < out_img.shape[1] and 0 <= iy < out_img.shape[0]:
            if use_lut:
                k = int(global_r + travelled)
                fspl = _fspl_from_lut(fspl_lut, k)
            else:
                fspl = _fspl((global_r + travelled) * pixel_size, freq_MHz)
            tot = acc_loss + fspl
            if tot > max_loss:
                tot = max_loss
            if tot < out_img[iy, ix]:
                out_img[iy, ix] = tot
            counts[iy, ix] += 1.0

    # left the map?
    if px_hit < 0:
        return

    # advance position & distance
    new_x = x0 + dx * travelled
    new_y = y0 + dy * travelled
    new_r = global_r + travelled

    # Branch-specific handling at boundary
    is_transmit_exit = (last_val > 0. and cur_val == 0.)

    # ─ straight continuation (transmitted branch if crossing) ─
    acc_loss_trans = acc_loss + (last_val if is_transmit_exit else 0.0)
    trans_ct_trans = trans_ct + (1 if is_transmit_exit else 0)
    if not (acc_loss_trans >= max_loss or trans_ct_trans > max_trans):
        _trace_ray_recursive(
            refl_mat, trans_mat, nx_img, ny_img,
            out_img, counts,
            new_x, new_y, dx, dy,
            trans_ct_trans, refl_ct,
            acc_loss_trans, new_r,
            pixel_size, freq_MHz,
            radial_step, max_dist,
            max_trans, max_refl, max_loss,
            fspl_lut, use_lut
        )

    # ─ reflection branch ─
    if refl_ct < max_refl:
        # Reflect only on reflective walls (guard against normals from pure transmission)
        refl_val = refl_mat[py_hit, px_hit]
        if refl_val > 0.0:
            nx, ny = _estimate_normal(nx_img, ny_img, px_hit, py_hit)
            if nx != 0.0 or ny != 0.0:
                rdx, rdy = _reflect_dir(dx, dy, nx, ny)
                # Start the reflected ray on the air side to avoid charging a spurious transmission
                # when immediately exiting the wall after reflecting.
                if (last_val == 0.0 and cur_val > 0.0):
                    rx0 = float(px_prev)
                    ry0 = float(py_prev)
                else:
                    rx0 = new_x
                    ry0 = new_y
                _trace_ray_recursive(
                    refl_mat, trans_mat, nx_img, ny_img,
                    out_img, counts,
                    rx0, ry0, rdx, rdy,
                    trans_ct, refl_ct + 1,
                    acc_loss + refl_val,
                    new_r,
                    pixel_size, freq_MHz,
                    radial_step, max_dist,
                    max_trans, max_refl, max_loss,
                    fspl_lut, use_lut
                )


# Variant that uses provided normal maps (from precomputed data)
@njit(parallel=True, fastmath=True, nogil=True, boundscheck=False, cache=False)
def calculate_combined_loss_with_normals(
    reflectance_mat, transmittance_mat, nx_img, ny_img,
    x_ant, y_ant, freq_MHz,
    n_angles,
    max_refl=MAX_REFL, max_trans=MAX_TRANS,
    pixel_size=0.25,
    radial_step=1.0,
    max_loss=32000.0,
    use_fspl_lut=True
):
    h, w = reflectance_mat.shape
    out  = np.full((h, w), max_loss, np.float32)
    cnt  = np.zeros((h, w), np.float32)

    dtheta   = 2.0 * np.pi / n_angles
    max_dist = 100.0 * np.hypot(w, h)
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
            fspl_lut, use_lut
        )

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
        use_fspl_lut=True
    )
    _WARMED_UP = True

# ---------------------------------------------------------------------#
#  TRANSMISSION-ONLY TRACE                                             #
# ---------------------------------------------------------------------#
@njit(parallel=True, fastmath=True, nogil=True, boundscheck=False)
def calculate_transmission_loss_numpy(trans_mat, x_ant, y_ant, freq_MHz, n_angles=360*128, radial_step=1.0, max_walls=MAX_TRANS, max_loss=32000.0, pixel_size=0.25):

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

    return out, cnt

# ---------------------------------------------------------------------#
#  APPROX WRAPPER                                                      #
# ---------------------------------------------------------------------#
def _predict_worker(args):
    """Per-process worker to approximate a single sample.
    Sets numba threads before warmup to avoid oversubscription.
    """
    method, max_trans, max_refl, numba_threads, sample = args
    try:
        import numba as _nb
        if numba_threads and numba_threads > 0:
            _nb.set_num_threads(numba_threads)
    except Exception:
        pass
    # Construct per-process Approx (triggers JIT warmup once per process)
    model = Approx(method)
    return model.approximate(sample, max_trans=max_trans, max_refl=max_refl)

class Approx:
    def __init__(self, method='combined'):
        self.method = method
        _warmup_numba_once()

    def approximate(self, sample: RadarSample,
                    max_trans=MAX_TRANS, max_refl=MAX_REFL):
        ref, trans, _ = sample.input_img.cpu().numpy()
        x, y, f = sample.x_ant, sample.y_ant, sample.freq_MHz
        # Prefer in-memory normals if provided on sample; fallback to disk
        nx_img = None; ny_img = None
        try:
            if hasattr(sample, 'normals') and sample.normals is not None:
                n = sample.normals
                if isinstance(n, np.ndarray) and n.ndim == 3 and n.shape[2] == 2:
                    nx_img = np.ascontiguousarray(n[..., 0], dtype=np.float64)
                    ny_img = np.ascontiguousarray(n[..., 1], dtype=np.float64)
            elif hasattr(sample, 'nx_img') and hasattr(sample, 'ny_img') and sample.nx_img is not None and sample.ny_img is not None:
                nx_img = np.ascontiguousarray(sample.nx_img, dtype=np.float64)
                ny_img = np.ascontiguousarray(sample.ny_img, dtype=np.float64)
        except Exception:
            nx_img = None; ny_img = None
        if nx_img is None or ny_img is None:
            # Accept either a tuple (b, ant, f, sp) or a list/seq of such tuples
            building_id = 0
            if sample.ids:
                if isinstance(sample.ids, (tuple, list)):
                    first = sample.ids[0] if isinstance(sample.ids, list) else sample.ids
                    if isinstance(first, (tuple, list)) and len(first) >= 1:
                        building_id = int(first[0])
                    elif isinstance(first, (int, np.integer)):
                        building_id = int(first)
            try:
                nx_img, ny_img = load_precomputed_normals_for_building(building_id, ref, trans)
            except FileNotFoundError:
                # Fallback: compute normals on-the-fly from reflectance/transmittance mask
                building_mask = (ref + trans > 0).astype(np.uint8)
                angles = precompute_wall_angles_pca(building_mask)
                rad = np.deg2rad(angles + 90.0)
                nx_img = np.cos(rad).astype(np.float64)
                ny_img = np.sin(rad).astype(np.float64)
                invalid = angles < 0
                if np.any(invalid):
                    nx_img = nx_img.copy(); ny_img = ny_img.copy()
                    nx_img[invalid] = 0.0; ny_img[invalid] = 0.0
        # Ensure contiguous dtypes for numba
        ref_c  = np.ascontiguousarray(ref, dtype=np.float64)
        trans_c = np.ascontiguousarray(trans, dtype=np.float64)

        if self.method == 'combined':
            feat, cnt = calculate_combined_loss_with_normals(
                ref_c, trans_c, nx_img, ny_img,
                x, y, f,
                n_angles=N_ANGLES,
                max_refl=max_refl,
                max_trans=max_trans,
                radial_step=1.0,
                use_fspl_lut=True
            )
            feat = apply_backfill(feat, cnt, x, y, 0.25, f, 32000.0, BACKFILL_METHOD, BACKFILL_PARAMS, trans_mat=trans_c)
        else:
            feat, cnt = calculate_transmission_loss_numpy(trans_c, x, y, f, n_angles=360*128, max_walls=max_trans)
            feat = feat.astype(np.float32)
            feat = apply_backfill(feat, cnt.astype(np.float32), x, y, 0.25, f, 32000.0, BACKFILL_METHOD, BACKFILL_PARAMS, trans_mat=trans_c)
        feat = np.minimum(feat, 32000.0)
        return torch.from_numpy(np.floor(feat))

    def predict(self, samples, max_trans=MAX_TRANS, max_refl=MAX_REFL, num_workers: int = 0, numba_threads: int = 0, backend: str = "threads", auto_convert: bool = True):
        """Predict over a batch of samples.
        - num_workers <= 1: run sequentially
        - backend=='threads': use ThreadPool, set a low global Numba thread count to avoid oversubscription
        - backend=='processes': use ProcessPool (spawn-safe), set per-worker Numba threads
        """
        # Optionally adapt input dictionaries into RadarSample with normals
        if auto_convert:
            adapted = []
            for obj in samples:
                if isinstance(obj, RadarSample):
                    adapted.append(obj)
                elif isinstance(obj, dict):
                    adapted.append(read_sample(obj))
                else:
                    adapted.append(obj)
            samples = adapted
        if num_workers is None or num_workers <= 1:
            return [self.approximate(s, max_trans, max_refl) for s in tqdm(samples, "predicting")]
        max_workers = num_workers if isinstance(num_workers, int) and num_workers > 0 else max(1, (_mp.cpu_count() or 2) - 1)
        if backend == "threads":
            try:
                import numba as _nb
                if numba_threads and numba_threads > 0:
                    _nb.set_num_threads(numba_threads)
                else:
                    # default: split cores across workers
                    per = max(1, (_mp.cpu_count() or 2) // max_workers)
                    _nb.set_num_threads(per)
            except Exception:
                pass
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(self.approximate, s, max_trans, max_refl) for s in samples]
                return [f.result() for f in tqdm(futures, total=len(futures), desc="predicting")]
        else:
            args_iter = [(self.method, max_trans, max_refl, (numba_threads or 0), s) for s in samples]
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                return list(tqdm(ex.map(_predict_worker, args_iter), total=len(samples), desc="predicting"))


# ---------------------------------------------------------------------#
#  MAIN                                                                #
# ---------------------------------------------------------------------#
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Approximation demo")
    parser.add_argument("-N", type=int, default=200, help="number of random samples to load")
    parser.add_argument("--seed", type=str, default=None, help="random seed; use 'None' for fully random each run")
    parser.add_argument("--compare0", action="store_true",
                        help="generate zero-reflection combined vs transmission comparison (val.png)")
    parser.add_argument("--hits", action="store_true",
                        help="generate hit-count plots (hit_counts.png)")
    parser.add_argument("--viz_k", type=int, default=12, help="number of samples to show in viz.png (top-|ΔRMSE|)")
    parser.add_argument("--workers", type=int, default=0, help="parallel workers for batch (0/1 = off)")
    parser.add_argument("--numba_threads", type=int, default=0, help="Numba threads (threads backend: global per process; processes backend: per worker)")
    parser.add_argument("--backend", type=str, default="threads", choices=["threads","processes"], help="parallel backend for batch")
    parser.add_argument("--untouched", action="store_true", help="compute and visualize untouched pixel masks (slow)")
    args = parser.parse_args()

    # Use spawn to avoid forking issues with threaded libraries (Numba/TBB)
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=False)
    except Exception:
        pass

    N = args.N
    seed = None if args.seed is None or str(args.seed).lower() == 'none' else int(args.seed)
    samples = load_samples(num_samples=N, seed=seed)

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
            use_fspl_lut=True)
        # Apply backfill to combined for fair comparison
        cmb_map = apply_backfill(cmb_map.astype(np.float32), cmb_cnt.astype(np.float32), x, y, 0.25, f, 32000.0, BACKFILL_METHOD, BACKFILL_PARAMS, trans_mat=trans)

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

    print("Predicting combined")
    preds_comb = comb_model.predict(samples, max_refl=MAX_REFL, num_workers=args.workers, numba_threads=args.numba_threads, backend=args.backend)
    print("Predicting tx-only")
    preds_tx   = txonly_model.predict(samples, max_refl=0, num_workers=args.workers, numba_threads=args.numba_threads, backend=args.backend)

    rms_c = [rmse(p, s.output_img) for p, s in zip(preds_comb, samples)]
    rms_t = [rmse(p, s.output_img) for p, s in zip(preds_tx, samples)]

    print(f"RMSE (combined) : {np.mean(rms_c):.3f}")
    print(f"RMSE (tx-only)  : {np.mean(rms_t):.3f}")

    # ────────────────────────────────────────────────────────────
    # Visual comparison (generic): GT + each approximator + all
    # unique pairwise diff images. Diff is shown as |A-B| in gray.
    # Select top-|ΔRMSE| samples to visualize.
    # ────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import itertools
    import numpy as _np

    n_show = min(args.viz_k, N)
    base_names = ["GT", "Tx-Only", "Combined"]
    top_idx = _np.arange(n_show)

    mask_c0 = {}
    mask_t0 = {}
    if args.untouched:
        # For selected samples, compute and print untouched pixel counts using the same params
        for idx in top_idx:
            ref, trans, _in = samples[idx].input_img.cpu().numpy()
            x, y, f = samples[idx].x_ant, samples[idx].y_ant, samples[idx].freq_MHz
            b_id = samples[idx].ids[0] if samples[idx].ids else 0
            nx_i, ny_i = load_precomputed_normals_for_building(b_id, ref, trans)
            _, cnt_comb = calculate_combined_loss_with_normals(
                ref, trans, nx_i, ny_i, x, y, f,
                n_angles=N_ANGLES,
                max_refl=MAX_REFL, max_trans=MAX_TRANS,
                radial_step=1.0,
                use_fspl_lut=True)
            _, cnt_tx = calculate_transmission_loss_numpy(
                trans, x, y, f, n_angles=360*128, max_walls=MAX_TRANS)
            total = cnt_comb.shape[0] * cnt_comb.shape[1]
            untouched_c = int((_np.sum(cnt_comb == 0)).item())
            untouched_t = int((_np.sum(cnt_tx == 0)).item())
            pct_c = 100.0 * untouched_c / total
            pct_t = 100.0 * untouched_t / total
            print(f"Untouched pixels [sample {idx}]: combined={untouched_c} ({pct_c:.2f}%), tx-only={untouched_t} ({pct_t:.2f}%)")
            mask_c0[idx] = (cnt_comb == 0)
            mask_t0[idx] = (cnt_tx == 0)

    for row, idx in enumerate(top_idx):
        # Get all 3 matrices: GT, Tx-Only, Combined
        gt = samples[idx].output_img
        tx_only = preds_tx[idx]
        comb_v1 = preds_comb[idx]

        # Convert to NumPy arrays before using NumPy ops to avoid __array_wrap__ warnings
        def _to_numpy(t):
            return t.detach().cpu().numpy() if hasattr(t, 'detach') else (_np.asarray(t.cpu()) if hasattr(t, 'cpu') else _np.asarray(t))

        gt_np = _to_numpy(gt)
        tx_np = _to_numpy(tx_only)
        c_np  = _to_numpy(comb_v1)

        mats = [gt_np, tx_np, c_np]

        # Layout: GT | T_ONLY | COMB | GT-T | GT-C | C-T
        row_mats = mats + [
            _np.abs(gt_np - tx_np),    # GT-T
            _np.abs(gt_np - c_np),     # GT-C
            _np.abs(c_np - tx_np)      # C-T
        ]

        row_titles = base_names.copy() + ["GT-T", "GT-C", "C-T"]

        # append RMSE for approximators in titles
        row_titles[1] = f"Tx-Only ({rmse(mats[1], mats[0]):.2f})"
        row_titles[2] = f"Combined ({rmse(mats[2], mats[0]):.2f})"

        if row == 0:
            # create figure with 6 columns
            fig, axes = plt.subplots(n_show, 6, figsize=(24, 4 * n_show))
            axes = _np.atleast_2d(axes)

        for col, (mat, title) in enumerate(zip(row_mats, row_titles)):
            ax = axes[row, col]
            if col < len(mats):
                # Original methods - use full color range
                im = ax.imshow(mat)
            else:
                # Difference images - use grayscale with consistent scaling
                vmax = max(_np.abs(gt_np - tx_np).max(),
                          _np.abs(gt_np - c_np).max(),
                          _np.abs(c_np - tx_np).max())
                im = ax.imshow(mat, cmap='gray', vmin=0, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    plt.tight_layout(); plt.savefig("viz.png", dpi=150); plt.close(fig)

    saved = ["viz.png"]
    if args.compare0:
        saved.append("val.png")
    if args.hits:
        saved.append("hit_counts.png")
    print("Saved:", "  ".join(saved))