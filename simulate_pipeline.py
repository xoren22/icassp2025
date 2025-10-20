import os
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import secrets
import json

from approx import Approx
from helper import RadarSample
from room_generator import generate_floor_scene


def _save_normals_npz(normals: np.ndarray, building_id: int, expected_shape: tuple[int,int]):
	"""
	Save normals to parsed_buildings/B{building_id}_normals.npz in the format expected by approx.py.
	"""
	H, W = expected_shape
	if normals.ndim != 3 or normals.shape[:2] != (H, W) or normals.shape[2] != 2:
		raise ValueError(f"normals must be HxWx2 with shape {(H,W,2)}, got {normals.shape}")
	base_dir = os.path.dirname(os.path.abspath(__file__))
	parsed_dir = os.path.join(base_dir, "parsed_buildings")
	os.makedirs(parsed_dir, exist_ok=True)
	path = os.path.join(parsed_dir, f"B{building_id}_normals.npz")
	nx = normals[..., 0].astype(np.float32, copy=False)
	ny = normals[..., 1].astype(np.float32, copy=False)
	# Overwrite safely
	with open(path, "wb") as f:
		np.savez(f, nx=nx, ny=ny)
	return path


def _export_sample_npz_json(out_root: str, sample_name: str, arrays: dict, metadata: dict) -> tuple[str, str]:
	"""
	Save arrays to {out_root}/{sample_name}/{sample_name}.npz (compressed) and metadata JSON alongside.
	Returns (npz_path, json_path).
	"""
	sample_dir = os.path.join(out_root, sample_name)
	os.makedirs(sample_dir, exist_ok=True)
	npz_path = os.path.join(sample_dir, f"{sample_name}.npz")
	json_path = os.path.join(sample_dir, f"{sample_name}.json")

	# Normalize dtypes for compact, precise storage
	np_arrays = {}
	for key, val in arrays.items():
		if isinstance(val, torch.Tensor):
			val = val.detach().cpu().numpy()
		if isinstance(val, np.ndarray):
			if val.dtype.kind in ('f',):
				np_arrays[key] = val.astype(np.float32, copy=False)
			else:
				np_arrays[key] = val
		else:
			raise TypeError(f"Array value for key '{key}' must be a numpy array or tensor, got {type(val)}")

	with open(npz_path, "wb") as f:
		np.savez_compressed(f, **np_arrays)
	with open(json_path, "w") as f:
		json.dump(metadata, f, indent=2)
	return npz_path, json_path


def build_sample_from_generated(mask, normals, scene, reflectance, transmittance, dist_map, building_id: int = 0):
	H, W = mask.shape
	# Input image channels: reflectance, transmittance, distance
	inp = np.stack([reflectance, transmittance, dist_map], axis=0).astype(np.float32)
	input_img = torch.from_numpy(inp)
	# Output image unknown here; approximator will predict
	output_img = torch.zeros((H, W), dtype=torch.float32)

	ant = scene.get("antenna", {})
	x_ant = float(ant.get("x", 0))
	y_ant = float(ant.get("y", 0))
	freq_MHz = float(scene.get("frequency_MHz", 1800))

	# Radiation pattern placeholder: isotropic 360 values
	radiation_pattern = torch.ones(360, dtype=torch.float32)

	return RadarSample(
		H=H,
		W=W,
		x_ant=x_ant,
		y_ant=y_ant,
		azimuth=0.0,
		freq_MHz=freq_MHz,
		input_img=input_img,
		output_img=output_img,
		radiation_pattern=radiation_pattern,
		pixel_size=0.25,
		mask=torch.from_numpy(mask.astype(np.float32)),
		ids=(int(building_id), 0, 0, 0),
	)


def visualize_triplet(transmittance, reflectance, pred, x_ant, y_ant, width_m, height_m, freq_MHz, save_path=None):
	fig, axes = plt.subplots(1, 3, figsize=(18, 6))
	# Use a shared colormap without masking so background uses low-value color
	cmap_shared = 'viridis'
	# Transmittance
	im0 = axes[0].imshow(transmittance, cmap=cmap_shared)
	axes[0].set_title("Transmittance (1–15)")
	axes[0].plot(x_ant, y_ant, marker='*', color='red', markersize=10)
	axes[0].axis('off')
	plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
	# Reflectance
	im1 = axes[1].imshow(reflectance, cmap=cmap_shared)
	axes[1].set_title("Reflectance (1–15)")
	axes[1].plot(x_ant, y_ant, marker='*', color='red', markersize=10)
	axes[1].axis('off')
	plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
	# Predicted pathloss
	im2 = axes[2].imshow(pred, cmap='viridis')
	axes[2].set_title("Predicted Pathloss (dB)")
	axes[2].plot(x_ant, y_ant, marker='*', color='red', markersize=10)
	axes[2].axis('off')
	plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
	fig.suptitle(f"{int(width_m)}m x {int(height_m)}m @ {int(freq_MHz)} MHz", fontsize=14)
	plt.tight_layout()
	if save_path:
		plt.savefig(save_path, dpi=150)
		logging.info(f"Saved visualization to {save_path}")
	else:
		plt.show()


def _build_synthetic_dataset(args):
	"""
	Generate a synthetic dataset as precise per-sample bundles (no PNG/CSV):
	- <out_dir>/samples/<name>/<name>.npz arrays: normals(HxWx2), reflectance, transmittance, distance_m, mask, pathloss
	- <out_dir>/samples/<name>/<name>.json metadata: antenna(px), frequency_MHz, canvas(m), ids, pixel_size_m
	"""
	base_dir = os.path.dirname(os.path.abspath(__file__))
	# Resolve output dir
	out_dir = args.data_out
	if not os.path.isabs(out_dir):
		out_dir = os.path.join(base_dir, out_dir)

	samples_dir = os.path.join(out_dir, 'samples')
	os.makedirs(samples_dir, exist_ok=True)

	N = int(max(1, args.num))
	approx_model = Approx()
	logging.info(f"Building synthetic dataset (NPZ+JSON) with {N} samples at {out_dir}")

	for i in tqdm(range(N), desc='SynthSamples'):
		mask, normals, scene, refl, trans, dist = generate_floor_scene()
		H, W = mask.shape
		ant_xy = scene.get('antenna', {})
		x_ant = int(ant_xy.get('x', 0))
		y_ant = int(ant_xy.get('y', 0))
		freq_MHz = int(scene.get('frequency_MHz', 1800))

		freqs = [868, 1800, 3500]
		try:
			f_idx = 1 + freqs.index(freq_MHz)
		except ValueError:
			f_idx = 1 + int(np.argmin([abs(freq_MHz - v) for v in freqs]))

		# Simple synthetic ids for traceability
		b = (i % 10) + 1
		ant_id = (i % 2) + 1
		sp = i

		# Persist normals for this synthetic building id (required by approx.py)
		_save_normals_npz(normals, b, mask.shape)

		sample = build_sample_from_generated(mask, normals, scene, refl, trans, dist, building_id=b)
		# Override ids to list containing one tuple as expected by approx.py
		sample.ids = [(int(b), int(ant_id), int(f_idx), int(sp))]

		# Predict pathloss map (keep full precision floats)
		pred = approx_model.approximate(sample).cpu().numpy().astype(np.float32)

		# Save per-sample arrays and metadata
		distance_m = (dist.astype(np.float32, copy=False)) * float(sample.pixel_size)
		sample_name = f'B{b}_Ant{ant_id}_f{f_idx}_S{sp}'
		arrays = {
			'normals': normals.astype(np.float32, copy=False),
			'reflectance': refl.astype(np.float32, copy=False),
			'transmittance': trans.astype(np.float32, copy=False),
			'distance_m': distance_m,
			'mask': mask.astype(np.uint8, copy=False),
			'pathloss': pred,
		}
		canvas = scene.get('canvas', {})
		metadata = {
			'sample_name': sample_name,
			'shape_hw': [int(H), int(W)],
			'pixel_size_m': float(sample.pixel_size),
			'antenna': {'x_px': int(x_ant), 'y_px': int(y_ant)},
			'frequency_MHz': int(freq_MHz),
			'canvas': {'width_m': float(canvas.get('width_m', 0.0)), 'height_m': float(canvas.get('height_m', 0.0))},
			'ids': {'building': int(b), 'antenna': int(ant_id), 'frequency_index': int(f_idx), 'sample_index': int(sp)},
			'created_at_unix_s': float(time.time()),
		}
		_export_sample_npz_json(samples_dir, sample_name, arrays, metadata)

	logging.info("Synthetic dataset export complete.")


def main():
	import argparse
	logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

	parser = argparse.ArgumentParser("Room generation + approximation pipeline")
	parser.add_argument('--num', type=int, default=5, help='Number of rooms to generate and approximate')
	parser.add_argument('--batch_size', type=int, default=10, help='Batch size for generate->predict->save streaming')
	parser.add_argument('--backend', type=str, default='processes', choices=['threads','processes'], help='Parallel backend for prediction')
	parser.add_argument('--numba_threads', type=int, default=0, help='Numba threads per worker (0 = auto)')
	parser.add_argument('--make_dataset', action='store_true', help='Export synthetic dataset as NPZ+JSON per sample (precise arrays + metadata)')
	parser.add_argument('--data_out', type=str, default='data/synthetic', help='Output dataset directory')
	parser.add_argument('--run_id', type=str, default=None, help='Unique run identifier; auto-generated if omitted')
	parser.add_argument('--viz', action='store_true', help='Additionally save PNG visualizations for streamed pipeline')
	args = parser.parse_args()

	# If requested, build synthetic dataset and exit
	if args.make_dataset:
		_build_synthetic_dataset(args)
		return

	N = int(max(1, args.num))
	B = int(max(1, args.batch_size))

	# Choose backend and threads; always use a single worker to avoid deadlocks
	chosen_backend = args.backend
	chosen_numba_threads = args.numba_threads if (args.numba_threads and args.numba_threads > 0) else 1

	base_out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated_rooms')
	os.makedirs(base_out_root, exist_ok=True)

	# Unique run identifier and output dir (safe across concurrent runs)
	if args.run_id and len(str(args.run_id)) > 0:
		run_id = str(args.run_id)
	else:
		# timestamp (µs) + pid + random 16-bit suffix
		us = int(time.time() * 1_000_000)
		pid = os.getpid() & 0xFFFF
		rnd = secrets.randbits(16)
		run_id = f"{us}_{pid:04x}_{rnd:04x}"
	out_dir = os.path.join(base_out_root, run_id)
	os.makedirs(out_dir, exist_ok=True)
	logging.info(f"Run ID: {run_id}; outputs -> {out_dir}")

	# Building IDs: integer seconds timestamp + random offset [0, 1_000_000]

	# Streamed Stage 1+2: generate->predict->save in batches (save NPZ+JSON per sample)
	logging.info(f"Processing {N} samples in batches of {B} (backend={chosen_backend})...")
	model = Approx()
	global_idx = 0
	samples_dir = os.path.join(out_dir, 'samples')
	os.makedirs(samples_dir, exist_ok=True)
	for start in tqdm(range(0, N, B), desc='Batches'):
		end = min(start + B, N)
		batch = []  # (sample, mask, normals, refl, trans, dist, scene, gidx)
		for _ in range(start, end):
			mask, normals, scene, refl, trans, dist = generate_floor_scene()
			# Stats (debug)
			nz_refl = int(np.count_nonzero(refl)); nz_trans = int(np.count_nonzero(trans))
			min_refl = float(np.min(refl)) if nz_refl > 0 else 0.0
			max_refl = float(np.max(refl)) if nz_refl > 0 else 0.0
			min_trans = float(np.min(trans)) if nz_trans > 0 else 0.0
			max_trans = float(np.max(trans)) if nz_trans > 0 else 0.0
			logging.debug(f"[{global_idx}] refl nz={nz_refl} [{min_refl:.2f},{max_refl:.2f}], trans nz={nz_trans} [{min_trans:.2f},{max_trans:.2f}]")

			# Persist normals with requested building_id scheme
			building_id = int(time.time()) + np.random.randint(0, 1_000_000)
			_save_normals_npz(normals, building_id, mask.shape)

			# Build sample aligned with approx.py expectations
			sample = build_sample_from_generated(mask, normals, scene, refl, trans, dist, building_id=building_id)
			# Ensure ids is list-of-tuple
			if not sample.ids or not isinstance(sample.ids, list):
				sample.ids = [(int(building_id), 0, 0, 0)]
			elif isinstance(sample.ids, tuple):
				sample.ids = [tuple(int(v) for v in sample.ids)]
			batch.append((sample, mask, normals, refl, trans, dist, scene, global_idx))
			global_idx += 1

		# Predict for this batch
		samples = [t[0] for t in batch]
		preds = model.predict(samples, num_workers=1, numba_threads=chosen_numba_threads, backend=chosen_backend)
		for (sample, mask, normals, refl, trans, dist, scene, gidx), pred_t in zip(batch, preds):
			pred = pred_t.cpu().numpy() if hasattr(pred_t, 'cpu') else np.array(pred_t)
			# Save precise per-sample bundle
			distance_m = (dist.astype(np.float32, copy=False)) * float(sample.pixel_size)
			freq_MHz = int(scene.get('frequency_MHz', 1800))
			sample_name = f's{gidx:06d}'
			arrays = {
				'normals': normals.astype(np.float32, copy=False),
				'reflectance': refl.astype(np.float32, copy=False),
				'transmittance': trans.astype(np.float32, copy=False),
				'distance_m': distance_m,
				'mask': mask.astype(np.uint8, copy=False),
				'pathloss': pred.astype(np.float32, copy=False),
			}
			canvas = scene.get('canvas', {})
			metadata = {
				'sample_name': sample_name,
				'shape_hw': [int(sample.H), int(sample.W)],
				'pixel_size_m': float(sample.pixel_size),
				'antenna': {'x_px': int(sample.x_ant), 'y_px': int(sample.y_ant)},
				'frequency_MHz': int(freq_MHz),
				'canvas': {'width_m': float(canvas.get('width_m', 0.0)), 'height_m': float(canvas.get('height_m', 0.0))},
				'ids': {'building': int(sample.ids[0][0]), 'antenna': int(sample.ids[0][1]), 'frequency_index': int(sample.ids[0][2]), 'sample_index': int(sample.ids[0][3])},
				'created_at_unix_s': float(time.time()),
			}
			_export_sample_npz_json(samples_dir, sample_name, arrays, metadata)

			# Optional visualization if requested via CLI flag
			if getattr(args, 'viz', False):
				width_m = canvas.get('width_m', 0.0); height_m = canvas.get('height_m', 0.0)
				viz_name = f'viz_{gidx:06d}_{run_id}.png'
				viz_path = os.path.join(out_dir, viz_name)
				visualize_triplet(trans, refl, pred, sample.x_ant, sample.y_ant, width_m, height_m, freq_MHz, save_path=viz_path)

	logging.info("All batches processed. Done.")


if __name__ == "__main__":
	main()
