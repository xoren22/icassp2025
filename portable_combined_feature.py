"""
Portable combined-feature module

Provides two main entry points:
- compute_and_save_normals(refl, trans, out_npz_path)
    Computes wall normals from a binary mask derived from reflectance+transmittance and
    saves an npz with keys: nx, ny (float32) and angles (float32, degrees).

- compute_combined_feature(refl, trans, x_ant, y_ant, freq_MHz, normals_npz_path=None,
                           nx=None, ny=None, n_angles=360*64, max_refl=5, max_trans=15,
                           radial_step=1.0, max_loss=160.0)
    Computes the approximate pathloss feature using precomputed wall normals.
    You can pass a path to the saved normals or provide nx, ny arrays directly.

Dependencies: numpy, numba (optional but recommended). The code will work without
Numba, but will be significantly slower.
"""
from __future__ import annotations
import os
import math
import numpy as np

# Cap threads BEFORE importing numba to avoid notebook/REPL crashes from oversubscription
if "NUMBA_THREADING_LAYER" not in os.environ:
    os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("TBB_NUM_THREADS", "1")

from numba import njit, prange
import argparse
from typing import Tuple
 

# --------------------------- PCA angle utilities --------------------------- #
@njit(cache=True, inline='always')
def _pca_angle(xs, ys):
    n = xs.size
    x_mean = xs.mean()
    y_mean = ys.mean()
    sxx = 0.0
    syy = 0.0
    sxy = 0.0
    for i in range(n):
        dx = xs[i] - x_mean
        dy = ys[i] - y_mean
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    sxx /= n
    syy /= n
    sxy /= n
    ang = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    return (math.degrees(ang)) % 180.0

@njit(cache=True, inline='always')
def _angle_vote_oriented_strip(xs, ys, px, py, win, band=1.25, along_margin=2):
    n = xs.size
    if n < 3:
        return -1.0
    best_score = -1.0
    best_angle = -1.0
    max_len = float(max(3, win - along_margin))
    for k in range(60):  # 3-degree steps
        ang = 3.0 * k
        rad = ang * math.pi / 180.0
        c = math.cos(rad)
        s = math.sin(rad)
        score = 0.0
        count = 0
        for i in range(n):
            dx = xs[i] - px
            dy = ys[i] - py
            perp = abs(-s * dx + c * dy)
            if perp <= band:
                t = c * dx + s * dy
                if abs(t) <= max_len:
                    score += 1.0 / (1.0 + 0.5 * perp + 0.25 * abs(t))
                    count += 1
        if count >= 3 and score > best_score:
            best_score = score
            best_angle = ang
    return best_angle if best_score >= 0.0 else -1.0

@njit(cache=True)
def _compute_wall_angle_pca(mask, px, py, win=5):
    h, w = mask.shape
    y0, y1 = max(py - win, 0), min(py + win + 1, h)
    x0, x1 = max(px - win, 0), min(px + win + 1, w)
    patch = mask[y0:y1, x0:x1]
    ys, xs = np.nonzero(patch)
    n = xs.size
    if n < 3:
        return -1.0
    xs = xs.astype(np.float32) + x0
    ys = ys.astype(np.float32) + y0
    angle1 = _angle_vote_oriented_strip(xs, ys, px, py, win, band=1.25, along_margin=2)
    if angle1 < 0.0:
        return _pca_angle(xs, ys)
    return angle1

@njit(cache=True)
def _compute_wall_angle_multiscale(mask, px, py):
    h, w = mask.shape
    for win in (5, 6, 7):
        ang = _compute_wall_angle_pca(mask, px, py, win)
        if ang >= 0:
            return ang
    # Fallback: estimate from nearest empty pixel direction
    for radius in range(1, max(h, w)):
        found = False
        best_dx = 0
        best_dy = 0
        min_d2 = 1e9
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if abs(dx) != radius and abs(dy) != radius:
                    continue
                ny, nx = py + dy, px + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] == 0:
                    d2 = dx * dx + dy * dy
                    if d2 < min_d2:
                        min_d2 = d2
                        best_dx, best_dy = dx, dy
                        found = True
        if found:
            normal_angle = math.degrees(math.atan2(best_dy, best_dx))
            return (normal_angle + 90.0) % 180.0
    return 90.0

# Public: compute angles for all wall pixels
def precompute_wall_angles_pca(building_mask: np.ndarray) -> np.ndarray:
    h, w = building_mask.shape
    out = np.zeros((h, w), dtype=np.float32)
    for py in range(h):
        for px in range(w):
            if building_mask[py, px] > 0:
                out[py, px] = _compute_wall_angle_multiscale(building_mask, px, py)
    return out

# -------------------------- Combined approximator ------------------------- #
@njit(inline='always')
def _fspl(dist_m: float, freq_MHz: float, min_dist_m: float = 0.125) -> float:
    d = dist_m if dist_m > min_dist_m else min_dist_m
    return 20.0 * np.log10(d) + 20.0 * np.log10(freq_MHz) - 27.55

@njit(cache=False)
def _build_fspl_lut(max_steps: int, pixel_size: float, freq_MHz: float, min_dist_m: float = 0.125):
    lut = np.empty(max_steps + 1, np.float64)
    for k in range(max_steps + 1):
        d = k * pixel_size
        if d < min_dist_m:
            d = min_dist_m
        lut[k] = 20.0 * np.log10(d) + 20.0 * np.log10(freq_MHz) - 27.55
    return lut

@njit(inline='always')
def _fspl_from_lut(lut: np.ndarray, step_index: int) -> float:
    if step_index < 0:
        step_index = 0
    n = lut.shape[0]
    if step_index >= n:
        step_index = n - 1
    return lut[step_index]

@njit(inline='always')
def _euclid_px(px, py, x0, y0):
    return np.hypot(px - x0, py - y0)

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

@njit(inline='always')
def _estimate_normal(nx_img, ny_img, px, py):
    return nx_img[py, px], ny_img[py, px]

@njit(inline='always')
def _reflect_dir(dx, dy, nx, ny):
    dot = dx * nx + dy * ny
    rx = dx - 2.0 * dot * nx
    ry = dy - 2.0 * dot * ny
    mag = np.hypot(rx, ry)
    return (-dx, -dy) if mag == 0 else (rx / mag, ry / mag)

@njit(cache=False)
def _trace_ray_recursive(refl_mat, trans_mat, nx_img, ny_img, out_img, counts,
                         x0, y0, dx, dy, trans_ct, refl_ct, acc_loss, global_r,
                         pixel_size, freq_MHz, radial_step, max_dist,
                         max_trans, max_refl, max_loss,
                         fspl_lut, use_lut):
    if acc_loss >= max_loss:
        return
    px_hit, py_hit, _, _, travelled, last_val, cur_val = _step_until_wall(
        trans_mat, x0, y0, dx, dy, radial_step, max_dist)
    # paint segment
    steps = int(travelled / radial_step) + 1
    for s in range(1, steps):
        xi = x0 + dx * radial_step * s
        yi = y0 + dy * radial_step * s
        ix = int(round(xi));  iy = int(round(yi))
        if ix < 0 or ix >= out_img.shape[1] or iy < 0 or iy >= out_img.shape[0]:
            break
        if use_lut:
            k = int(global_r + s)
            fspl = _fspl_from_lut(fspl_lut, k)
        else:
            fspl = _fspl(global_r * pixel_size + radial_step * s * pixel_size, freq_MHz)
        tot = acc_loss + fspl
        if tot > max_loss:
            tot = max_loss
        if tot < out_img[iy, ix]:
            out_img[iy, ix] = tot
        counts[iy, ix] += 1.0
    if px_hit < 0:
        return
    # wall exit
    if last_val > 0. and cur_val == 0.:
        acc_loss += last_val
        trans_ct += 1
        if acc_loss >= max_loss or trans_ct > max_trans:
            return
    new_x = x0 + dx * travelled
    new_y = y0 + dy * travelled
    new_r = global_r + travelled
    # straight
    _trace_ray_recursive(refl_mat, trans_mat, nx_img, ny_img, out_img, counts,
                         new_x, new_y, dx, dy, trans_ct, refl_ct, acc_loss, new_r,
                         pixel_size, freq_MHz, radial_step, max_dist,
                         max_trans, max_refl, max_loss,
                         fspl_lut, use_lut)
    # reflection (only if reflective wall)
    if refl_ct < max_refl:
        refl_val = refl_mat[py_hit, px_hit]
        if refl_val > 0.0:
            nx, ny = _estimate_normal(nx_img, ny_img, px_hit, py_hit)
            if nx != 0.0 or ny != 0.0:
                rdx, rdy = _reflect_dir(dx, dy, nx, ny)
                _trace_ray_recursive(refl_mat, trans_mat, nx_img, ny_img, out_img, counts,
                                     new_x, new_y, rdx, rdy, trans_ct, refl_ct + 1,
                                     acc_loss + refl_val, new_r,
                                     pixel_size, freq_MHz, radial_step, max_dist,
                                     max_trans, max_refl, max_loss,
                                     fspl_lut, use_lut)

@njit(parallel=True, fastmath=True, nogil=True, boundscheck=False, cache=False)
def calculate_combined_loss_with_normals(reflectance_mat, transmittance_mat, nx_img, ny_img,
                                         x_ant, y_ant, freq_MHz,
                                         max_refl=5, max_trans=15,
                                         pixel_size=0.25,
                                         n_angles=360*64, radial_step=1.0,
                                         max_loss=160.0):
    h, w = reflectance_mat.shape
    out = np.full((h, w), max_loss, np.float64)
    cnt = np.zeros((h, w), np.float32)
    dtheta = 2.0 * np.pi / n_angles
    max_dist = np.hypot(w, h)
    cos_v = np.cos(np.arange(n_angles) * dtheta)
    sin_v = np.sin(np.arange(n_angles) * dtheta)

    use_lut = (radial_step == 1.0)
    if use_lut:
        max_steps = int(max_dist) + 2
        fspl_lut = _build_fspl_lut(max_steps, pixel_size, freq_MHz)
    else:
        fspl_lut = np.zeros(1, np.float64)
    for i in prange(n_angles):
        _trace_ray_recursive(reflectance_mat, transmittance_mat, nx_img, ny_img,
                             out, cnt, x_ant, y_ant, cos_v[i], sin_v[i],
                             0, 0, 0.0, 0.0, pixel_size, freq_MHz,
                             radial_step, max_dist, max_trans, max_refl, max_loss,
                             fspl_lut, use_lut)
    # direct-FSPL fill for untouched pixels
    for py in prange(h):
        for px in range(w):
            if cnt[py, px] == 0:
                d = _euclid_px(px, py, x_ant, y_ant) * pixel_size
                fspl = _fspl(d, freq_MHz)
                out[py, px] = fspl if fspl < max_loss else max_loss
    return out

# ----------------------------- Public helpers ----------------------------- #
def compute_and_save_normals(refl: np.ndarray, trans: np.ndarray, out_npz_path: str) -> None:
    """Compute normals from (refl+trans)>0 mask and save as npz."""
    assert refl.shape == trans.shape
    mask = ((refl + trans) > 0).astype(np.uint8)
    angles = precompute_wall_angles_pca(mask)
    rad = np.deg2rad(angles + 90.0)
    nx = np.cos(rad).astype(np.float32)
    ny = np.sin(rad).astype(np.float32)
    invalid = angles < 0
    if np.any(invalid):
        nx = nx.copy(); ny = ny.copy()
        nx[invalid] = 0.0
        ny[invalid] = 0.0
    os.makedirs(os.path.dirname(out_npz_path) or '.', exist_ok=True)
    np.savez_compressed(out_npz_path, nx=nx, ny=ny, angles=angles.astype(np.float32))


def compute_combined_feature(refl: np.ndarray, trans: np.ndarray,
                             x_ant: float, y_ant: float, freq_MHz: float,
                             normals_npz_path: str | None = None,
                             nx: np.ndarray | None = None,
                             ny: np.ndarray | None = None,
                             n_angles: int = 360*64,
                             max_refl: int = 5,
                             max_trans: int = 15,
                             radial_step: float = 1.0,
                             pixel_size: float = 0.25,
                             max_loss: float = 160.0) -> np.ndarray:
    """Compute combined feature given reflectance, transmittance, and normals.

    Provide either normals_npz_path pointing to a file with keys nx, ny, or
    pass nx, ny arrays directly. Returns a float32 (H,W) map clipped to [0,160].
    """
    H, W = refl.shape
    if nx is None or ny is None:
        if normals_npz_path is None:
            raise ValueError("Provide normals_npz_path or nx,ny arrays")
        data = np.load(normals_npz_path)
        nx = data['nx'].astype(np.float64, copy=False)
        ny = data['ny'].astype(np.float64, copy=False)
    if nx.shape != (H, W) or ny.shape != (H, W):
        raise ValueError("Normal map shape mismatch")
    feat = calculate_combined_loss_with_normals(
        refl.astype(np.float64, copy=False),
        trans.astype(np.float64, copy=False),
        nx, ny,
        float(x_ant), float(y_ant), float(freq_MHz),
        max_refl=max_refl, max_trans=max_trans,
        pixel_size=pixel_size,
        n_angles=n_angles, radial_step=radial_step,
        max_loss=max_loss,
    )
    feat = np.nan_to_num(feat, nan=max_loss, posinf=max_loss, neginf=0.0)
    return feat.astype(np.float32)


def _load_array_from_path(path: str) -> np.ndarray:
    """Best-effort loader for reflectance/transmittance arrays.

    Supports .npy/.npz directly. If reading an image file, tries to use imageio if available.
    Returns a 2D float32 array.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        arr = np.load(path)
    elif ext == '.npz':
        data = np.load(path)
        keys = list(data.keys())
        if len(keys) == 1:
            arr = data[keys[0]]
        else:
            for candidate in ('refl', 'trans', 'image', 'arr_0'):
                if candidate in data:
                    arr = data[candidate]
                    break
            else:
                raise ValueError(".npz must contain a single array or a key among: 'refl', 'trans', 'image', 'arr_0'")
    else:
        try:
            import imageio.v3 as iio
            arr = iio.imread(path)
        except Exception as e:
            raise ValueError(f"Unsupported file type for {path}. Use .npy/.npz or install imageio to read images.") from e
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return arr.astype(np.float32, copy=False)


def _load_building_from_path(path: str) -> np.ndarray:
    """Load a 3-channel building image/array as float32 HxWx3.

    Supports .npy/.npz (expects HxWx3) or image files via imageio.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        arr = np.load(path)
    elif ext == '.npz':
        data = np.load(path)
        keys = list(data.keys())
        if len(keys) == 1:
            arr = data[keys[0]]
        else:
            for candidate in ('image', 'building', 'rgb', 'arr_0'):
                if candidate in data:
                    arr = data[candidate]
                    break
            else:
                raise ValueError(".npz must contain a single HxWx3 array or a key among: 'image', 'building', 'rgb', 'arr_0'")
    else:
        try:
            import imageio.v3 as iio
            arr = iio.imread(path)
        except Exception as e:
            raise ValueError(f"Unsupported file type for {path}. Use .npy/.npz or install imageio to read images.") from e
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Expected HxWx3 array for building image, got shape {arr.shape}")
    # Use only first 3 channels if more are present
    arr = arr[..., :3]
    return arr.astype(np.float32, copy=False)


def _parse_channel_selector(sel: str) -> int:
    """Parse channel selector like 'r','g','b' or '0','1','2' into index 0..2."""
    lookup = {'r': 0, 'g': 1, 'b': 2}
    s = str(sel).strip().lower()
    if s in lookup:
        return lookup[s]
    try:
        idx = int(s)
    except Exception:
        raise ValueError(f"Invalid channel selector '{sel}'. Use r/g/b or 0/1/2.")
    if idx not in (0, 1, 2):
        raise ValueError(f"Channel index must be 0,1,2; got {idx}")
    return idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute and save wall normals from reflectance/transmittance maps.')
    # Optional single-file modes; if none provided, default to batch mode using repo layout
    parser.add_argument('--building', type=str, help='Path to 3-channel building image/array (.npy/.npz or image).')
    parser.add_argument('--refl', type=str, help='Path to reflectance array/image (.npy/.npz or image file).')
    parser.add_argument('--trans', type=str, help='Path to transmittance array/image (.npy/.npz or image file).')
    parser.add_argument('--value-scale', type=float, default=1.0, help='Optional multiplicative scale to apply to extracted channels (default 1.0).')
    parser.add_argument('--out-normals', type=str, help='Output .npz path to store normals (keys: nx, ny, angles). Required for single-file modes.')
    parser.add_argument('--buildings', type=int, nargs='*', help='Batch mode: which building ids to process (default: 1..25).')
    parser.add_argument('--overwrite', action='store_true', help='Batch mode: recompute even if output exists.')
    parser.add_argument('--data-root', type=str, default=None, help='Path to data/train folder; defaults to <repo>/data/train')
    parser.add_argument('--viz-demo', action='store_true', default=True, help='After ensuring normals, run a standalone demo: compute feature for one sample and save a 3-panel visualization with RMSE.')
    args = parser.parse_args()

    # Determine input mode
    if not args.building and not (args.refl and args.trans):
        # Batch mode using repo layout: <repo_root>/data/train/Inputs/Task_2_ICASSP
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_root = args.data_root if args.data_root else os.path.join(base_dir, 'data', 'train')
        input_dir = os.path.join(data_root, 'Inputs', 'Task_2_ICASSP')
        outputs_dir = os.path.join(data_root, 'Outputs', 'Task_2_ICASSP')
        positions_dir = os.path.join(data_root, 'Positions')
        out_dir = os.path.join(base_dir, 'parsed_buildings')
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.isdir(input_dir):
            print(f"Error: expected folder not found: {input_dir}")
            raise SystemExit(2)

        b_ids = args.buildings if args.buildings else list(range(1, 26))

        def find_one_input_for_building(bid: int) -> str:
            for ant in (1, 2):
                for f_idx in (1, 2, 3):
                    for sp in range(80):
                        name = f"B{bid}_Ant{ant}_f{f_idx}_S{sp}.png"
                        p = os.path.join(input_dir, name)
                        if os.path.exists(p):
                            return p
            return ''

        n_done = 0
        for b in b_ids:
            out_path = os.path.join(out_dir, f"B{b}_normals.npz")
            if os.path.exists(out_path) and not args.overwrite:
                print(f"B{b}: exists → {out_path}")
                n_done += 1
                continue
            src = find_one_input_for_building(b)
            if not src:
                print(f"B{b}: no input found in {input_dir}")
                continue
            try:
                img = _load_building_from_path(src)
                # Reflectance: Red channel (0), Transmittance: Green channel (1)
                refl = img[..., 0] * float(args.value_scale)
                trans = img[..., 1] * float(args.value_scale)
                compute_and_save_normals(refl, trans, out_path)
                print(f"B{b}: saved → {out_path}")
                n_done += 1
            except Exception as e:
                print(f"B{b}: error → {e}")
        print(f"Finished: {n_done}/{len(b_ids)} saved")

        if not args.viz_demo:
            raise SystemExit(0)

        # ─────────────────────────────────────────────
        # Standalone demo: compute feature for one sample
        # ─────────────────────────────────────────────
        def _find_one_sample(inp_dir: str, out_dir: str) -> Tuple[str, int, int, int, int]:
            for b in range(1, 26):
                for ant in (1, 2):
                    for f_idx in (1, 2, 3):
                        for sp in range(80):
                            name = f"B{b}_Ant{ant}_f{f_idx}_S{sp}.png"
                            p_in = os.path.join(inp_dir, name)
                            p_out = os.path.join(out_dir, name)
                            if os.path.exists(p_in) and os.path.exists(p_out):
                                return p_in, b, ant, f_idx, sp
            return '', -1, -1, -1, -1

        sample_path, b, ant, f_idx, sp = _find_one_sample(input_dir, outputs_dir)
        if not sample_path:
            print(f"Error: no matching sample found under {input_dir} and {outputs_dir}")
            raise SystemExit(2)

        try:
            import imageio.v3 as iio
        except Exception as e:
            print("Error: imageio is required for demo visualization. Install with `pip install imageio`.\n", e)
            raise SystemExit(2)

        img = iio.imread(sample_path).astype(np.float32)
        if img.ndim != 3 or img.shape[2] < 2:
            print(f"Error: expected HxWx3 image at {sample_path}, got shape {img.shape}")
            raise SystemExit(2)
        img = img[..., :3]
        refl = img[..., 0] * float(args.value_scale)
        trans = img[..., 1] * float(args.value_scale)

        # Load GT
        gt_path = os.path.join(outputs_dir, os.path.basename(sample_path))
        gt_img = iio.imread(gt_path)
        if gt_img.ndim == 3:
            gt = gt_img[..., 0].astype(np.float32)
        else:
            gt = gt_img.astype(np.float32)

        # Load position CSV
        pos_csv = os.path.join(positions_dir, f"Positions_B{b}_Ant{ant}_f{f_idx}.csv")
        if not os.path.exists(pos_csv):
            print(f"Error: positions CSV not found: {pos_csv}")
            raise SystemExit(2)
        import csv
        x_ant = None; y_ant = None
        with open(pos_csv, 'r', newline='') as fh:
            reader = csv.DictReader(fh)
            idx = 0
            for row in reader:
                if idx == sp:
                    try:
                        # Match repo convention: image coords (x,y) ≡ (col,row) ≡ (Y,X) in CSV
                        x_ant = float(row.get('Y'))
                        y_ant = float(row.get('X'))
                    except Exception:
                        # fallback: try lowercase
                        x_ant = float(row.get('y') or 0)
                        y_ant = float(row.get('x') or 0)
                    break
                idx += 1
        if x_ant is None or y_ant is None:
            print(f"Error: could not read X/Y for sample index {sp} from {pos_csv}")
            raise SystemExit(2)

        # Map frequency index to MHz
        freqs = {1: 868.0, 2: 1800.0, 3: 3500.0}
        freq_MHz = float(freqs.get(int(f_idx), 1800.0))

        # Load normals
        npz_path = os.path.join(out_dir, f"B{b}_normals.npz")
        if not os.path.exists(npz_path):
            print(f"Error: normals not found: {npz_path}")
            raise SystemExit(2)
        data = np.load(npz_path)
        nx = data['nx'].astype(np.float64, copy=False)
        ny = data['ny'].astype(np.float64, copy=False)
        if nx.shape != refl.shape or ny.shape != refl.shape:
            print(f"Error: normal map shape mismatch: normals {nx.shape} vs image {refl.shape}")
            raise SystemExit(2)

        # Compute feature using standalone function
        feat = calculate_combined_loss_with_normals(
            refl.astype(np.float64, copy=False),
            trans.astype(np.float64, copy=False),
            nx, ny, float(x_ant), float(y_ant), float(freq_MHz),
            max_refl=5, max_trans=15, pixel_size=0.25, n_angles=360*64, radial_step=1.0, max_loss=160.0
        )
        feat = np.nan_to_num(feat, nan=160.0, posinf=160.0, neginf=0.0).astype(np.float32)

        # Save visualization with RMSE
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print("Error: matplotlib is required for demo visualization. Install with `pip install matplotlib`.\n", e)
            raise SystemExit(2)

        # Align GT range and shape
        gt = gt.astype(np.float32)
        gt = np.nan_to_num(gt, nan=160.0, posinf=160.0, neginf=0.0)
        if gt.max() > 0:
            gt = np.minimum(gt, 160.0)

        # Resize checks: assume same HxW as inputs
        if gt.shape != feat.shape:
            print(f"Warning: GT shape {gt.shape} != feature shape {feat.shape}; attempting to crop to min overlap")
            h = min(gt.shape[0], feat.shape[0])
            w = min(gt.shape[1], feat.shape[1])
            gt = gt[:h, :w]
            feat = feat[:h, :w]

        rmse = float(np.sqrt(np.mean((gt - feat) ** 2)))

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        im0 = axes[0].imshow(gt, vmin=0, vmax=160)
        axes[0].set_title('GT')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        im1 = axes[1].imshow(feat, vmin=0, vmax=160)
        axes[1].set_title('Standalone Feature')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        diff = np.abs(gt - feat)
        im2 = axes[2].imshow(diff, cmap='gray')
        axes[2].set_title('Diff |GT-Feat|')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
        for ax in axes:
            ax.axis('off')
        fig.suptitle(f"B{b} Ant{ant} f{f_idx} S{sp}  RMSE={rmse:.2f} dB")
        plt.tight_layout()
        viz_path = os.path.join(base_dir, 'portable_demo_viz.png')
        plt.savefig(viz_path, dpi=150)
        plt.close(fig)
        print(f'Saved visualization to: {viz_path}')
        raise SystemExit(0)

    elif args.building:
        try:
            img = _load_building_from_path(args.building)
            refl = img[..., 0] * float(args.value_scale)
            trans = img[..., 1] * float(args.value_scale)
        except Exception as e:
            print(f"Error: failed to load/parse building image - {e}")
            raise SystemExit(2)
        if not args.out_normals:
            print("Error: --out-normals must be provided when using --building mode")
            raise SystemExit(2)
        try:
            compute_and_save_normals(refl, trans, args.out_normals)
            print(f"Saved normals to: {args.out_normals}")
        except Exception as e:
            print(f"Error: failed to compute/save normals - {e}")
            raise SystemExit(2)
        raise SystemExit(0)
    else:
        if not args.refl or not args.trans or not args.out_normals:
            print("Error: provide both --refl and --trans and --out-normals.")
            raise SystemExit(2)
        try:
            refl = _load_array_from_path(args.refl)
            trans = _load_array_from_path(args.trans)
        except Exception as e:
            print(f"Error: failed to load input arrays - {e}")
            raise SystemExit(2)
        if refl.shape != trans.shape:
            print(f"Error: shape mismatch between refl {refl.shape} and trans {trans.shape}")
            raise SystemExit(2)
        try:
            compute_and_save_normals(refl, trans, args.out_normals)
            print(f"Saved normals to: {args.out_normals}")
        except Exception as e:
            print(f"Error: failed to compute/save normals - {e}")
            raise SystemExit(2)