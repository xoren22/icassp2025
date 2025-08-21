"""
Batch featurization runner (parallel, process-based)

Runs the standalone combined-feature on all available samples using 5 workers
by default, saves per-sample arrays (feat, gt, diff) and a metrics CSV.

Outputs:
- analysis_outputs/per_sample/B{b}_Ant{ant}_f{f}_S{sp}.npz  (feat, gt, diff, meta)
- analysis_outputs/featurizer_metrics.csv                     (CSV rows)

Safe for notebooks: each worker limits its own Numba threads to avoid crashes.
"""

from __future__ import annotations
import os
import csv
import math
import time
from typing import List, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Cap global threads a bit for the parent too
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("TBB_NUM_THREADS", "1")

import numpy as np
import imageio.v3 as iio

from portable_combined_feature import compute_combined_feature


def _repo_paths() -> Tuple[str, str, str, str, str, str]:
    repo_root = os.path.abspath(os.path.dirname(__file__))
    data_root = os.path.join(repo_root, "data", "train")
    inp_dir = os.path.join(data_root, "Inputs", "Task_2_ICASSP")
    out_dir = os.path.join(data_root, "Outputs", "Task_2_ICASSP")
    pos_dir = os.path.join(data_root, "Positions")
    parsed_dir = os.path.join(repo_root, "parsed_buildings")
    analysis_dir = os.path.join(repo_root, "analysis_outputs")
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(os.path.join(analysis_dir, "per_sample"), exist_ok=True)
    return repo_root, data_root, inp_dir, out_dir, pos_dir, parsed_dir


def _discover_catalog(inp_dir: str, out_dir: str, pos_dir: str) -> List[Tuple[int,int,int,int,str,str,str]]:
    items: List[Tuple[int,int,int,int,str,str,str]] = []
    for b in range(1, 26):
        for ant in (1, 2):
            for f_idx in (1, 2, 3):
                pos_csv = os.path.join(pos_dir, f"Positions_B{b}_Ant{ant}_f{f_idx}.csv")
                if not os.path.exists(pos_csv):
                    continue
                for sp in range(80):
                    name = f"B{b}_Ant{ant}_f{f_idx}_S{sp}.png"
                    p_in = os.path.join(inp_dir, name)
                    p_out = os.path.join(out_dir, name)
                    if os.path.exists(p_in) and os.path.exists(p_out):
                        items.append((b, ant, f_idx, sp, p_in, p_out, pos_csv))
    return items


def _read_xy_from_positions(csv_path: str, sample_index: int) -> Tuple[float, float]:
    import csv as _csv
    with open(csv_path, "r", newline="") as fh:
        reader = _csv.DictReader(fh)
        for idx, row in enumerate(reader):
            if idx == sample_index:
                # Map CSV (Y, X) -> image coords (x, y)
                x_ant = float(row.get("Y"))
                y_ant = float(row.get("X"))
                return x_ant, y_ant
    raise IndexError(f"Row {sample_index} not found in {csv_path}")


def _safe_cap(img: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(img, nan=160.0, posinf=160.0, neginf=0.0), 0.0, 160.0).astype(np.float32)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _set_worker_env(numba_threads: int) -> None:
    os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TBB_NUM_THREADS", "1")
    os.environ["NUMBA_NUM_THREADS"] = str(max(1, int(numba_threads)))
    try:
        import numba as _nb
        _nb.set_num_threads(max(1, int(numba_threads)))
    except Exception:
        pass


def _encode_array(arr: np.ndarray, mode: str, *, max_val: float = 160.0):
    """Return (stored, enc_meta) where stored is array with chosen dtype and enc_meta hints.
    mode: 'f32' | 'f16' | 'u16'
    """
    if mode == "f32":
        return arr.astype(np.float32, copy=False), {"enc": "f32"}
    if mode == "f16":
        return arr.astype(np.float16), {"enc": "f16"}
    if mode == "u16":
        mul = 65535.0 / max_val
        stored = np.rint(np.clip(arr, 0.0, max_val) * mul).astype(np.uint16)
        return stored, {"enc": "u16", "mul": mul, "max": max_val}
    raise ValueError(f"Unknown encode mode: {mode}")


def _worker(task: Tuple, angles: int, overwrite: bool, numba_threads: int, save_kind: str, enc_mode: str) -> Tuple[bool, List[str]]:
    try:
        _set_worker_env(numba_threads)

        b, ant, f_idx, sp, p_in, p_out, pos_csv, parsed_dir, per_sample_dir = task

        # Skip if exists and not overwriting
        base = f"B{b}_Ant{ant}_f{f_idx}_S{sp}"
        npz_path = os.path.join(per_sample_dir, base + ".npz")
        if (not overwrite) and os.path.exists(npz_path):
            # Still compute metrics from existing file without recompute
            try:
                data = np.load(npz_path)
                feat = data["feat"].astype(np.float32)
                gt = data["gt"].astype(np.float32)
                r = _rmse(gt, feat)
                m = float(np.mean(np.abs(gt - feat)))
                return True, [str(b), str(ant), str(f_idx), str({1:868.0,2:1800.0,3:3500.0}[f_idx]), str(sp), str(feat.shape[0]), str(feat.shape[1]), f"{r:.6f}", f"{m:.6f}", npz_path]
            except Exception:
                # fall-through to recompute
                pass

        # Load inputs
        img = iio.imread(p_in).astype(np.float32)
        refl = img[..., 0]
        trans = img[..., 1]

        # Antenna
        x_ant, y_ant = _read_xy_from_positions(pos_csv, sp)
        freq_MHz = float({1:868.0, 2:1800.0, 3:3500.0}[f_idx])

        # Normals
        normals_path = os.path.join(parsed_dir, f"B{b}_normals.npz")
        if not os.path.exists(normals_path):
            return False, [f"Missing normals: {normals_path}"]
        nd = np.load(normals_path)
        nx = nd["nx"].astype(np.float64, copy=False)
        ny = nd["ny"].astype(np.float64, copy=False)

        # Feature (FSPL LUT auto-enabled when radial_step==1.0 in compute_combined_feature path)
        feat = compute_combined_feature(
            refl=refl, trans=trans,
            x_ant=x_ant, y_ant=y_ant, freq_MHz=freq_MHz,
            normals_npz_path=None, nx=nx, ny=ny,
            n_angles=angles,
        )
        feat_c = _safe_cap(feat)

        # GT
        gt_raw = iio.imread(p_out)
        gt = gt_raw[..., 0] if gt_raw.ndim == 3 else gt_raw
        gt_c = _safe_cap(gt.astype(np.float32))

        # Save arrays
        meta = np.array([b, ant, f_idx, sp, freq_MHz], dtype=np.float64)
        # Encode per save_kind to control disk footprint
        if save_kind == "feat":
            feat_enc, enc_meta = _encode_array(feat_c, enc_mode)
            np.savez_compressed(npz_path, feat=feat_enc, meta=meta, enc_meta=np.array(list(enc_meta.items()), dtype=object))
        elif save_kind == "diff":
            diff = gt_c - feat_c
            diff_enc, enc_meta = _encode_array(diff, enc_mode)
            np.savez_compressed(npz_path, diff=diff_enc, meta=meta, enc_meta=np.array(list(enc_meta.items()), dtype=object))
            # compute r/m with reconstructed diff
        else:  # both
            diff = gt_c - feat_c
            feat_enc, enc_meta_f = _encode_array(feat_c, enc_mode)
            # For 'both', use the requested encoding for diff as well
            diff_enc, enc_meta_d = _encode_array(diff, enc_mode)
            np.savez_compressed(npz_path, feat=feat_enc, diff=diff_enc, meta=meta,
                                enc_meta_f=np.array(list(enc_meta_f.items()), dtype=object),
                                enc_meta_d=np.array(list(enc_meta_d.items()), dtype=object))

        # Metrics row
        r = _rmse(gt_c, feat_c)
        m = float(np.mean(np.abs(gt_c - feat_c)))
        return True, [str(b), str(ant), str(f_idx), str(freq_MHz), str(sp), str(feat_c.shape[0]), str(feat_c.shape[1]), f"{r:.6f}", f"{m:.6f}", npz_path]

    except Exception as e:
        return False, [str(e)]


def main():
    import argparse
    parser = argparse.ArgumentParser("Batch featurizer (parallel)")
    parser.add_argument("--workers", type=int, default=10, help="Number of processes")
    parser.add_argument("--angles", type=int, default=23040, help="Angular resolution (use 23040 for 360*64")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if per-sample NPZ exists")
    parser.add_argument("--save", type=str, default="feat", choices=["feat","diff","both"], help="What to store per sample to disk")
    parser.add_argument("--dtype", type=str, default="f32", choices=["f32","f16","u16"], help="Encoding for arrays on disk (default f32)")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of samples for quick runs")
    parser.add_argument("--numba_threads", type=int, default=1, help="Numba threads per worker (keep small)")
    args = parser.parse_args()

    repo_root, data_root, inp_dir, out_dir, pos_dir, parsed_dir = _repo_paths()
    per_sample_dir = os.path.join(repo_root, "analysis_outputs", "per_sample")

    catalog = _discover_catalog(inp_dir, out_dir, pos_dir)
    if args.limit and args.limit > 0:
        catalog = catalog[:args.limit]
    print(f"Samples: {len(catalog)}  |  workers={args.workers}  angles={args.angles}  numba_threads={args.numba_threads}")

    metrics_csv = os.path.join(repo_root, "analysis_outputs", "featurizer_metrics.csv")
    t0 = time.time()
    done = 0; ok = 0; bad = 0

    with open(metrics_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["building","antenna","freq_idx","freq_MHz","sample_pos","H","W","rmse","mae","npz_path"])

        tasks = [ (b,ant,f,sp,inp,out,pos_dir,parsed_dir,per_sample_dir) for (b,ant,f,sp,inp,out,pos_dir) in catalog ]
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = [ ex.submit(_worker, t, args.angles, args.overwrite, args.numba_threads, args.save, args.dtype) for t in tasks ]
            with tqdm(total=len(futs), desc="Featurizing", smoothing=0.1) as pbar:
                for fut in as_completed(futs):
                    success, payload = fut.result()
                    done += 1
                    if success:
                        writer.writerow(payload)
                        ok += 1
                    else:
                        bad += 1
                        print("Error:", payload[0], flush=True)
                    pbar.update(1)

    dt = time.time() - t0
    print(f"Done. ok={ok} bad={bad}  time={dt/60.0:.1f} min  -> {metrics_csv}")


if __name__ == "__main__":
    # Use spawn to avoid forking issues with threaded libs
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=False)
    except Exception:
        pass
    main()


