import os
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

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


def main():
	import argparse
	logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

	parser = argparse.ArgumentParser("Room generation + approximation pipeline")
	parser.add_argument('--num', type=int, default=5, help='Number of rooms to generate and approximate')
	parser.add_argument('--workers', type=int, default=0, help='Parallel workers for prediction (0/1 = auto)')
	parser.add_argument('--backend', type=str, default='processes', choices=['threads','processes'], help='Parallel backend for prediction')
	parser.add_argument('--numba_threads', type=int, default=0, help='Numba threads per worker (0 = auto)')
	args = parser.parse_args()

	N = int(max(1, args.num))

	# Choose fast, safe defaults if not provided: processes backend, 1 Numba thread per worker
	import multiprocessing as _mp
	cpu = max(1, (_mp.cpu_count() or 2))
	chosen_backend = args.backend
	chosen_workers = args.workers if (args.workers and args.workers > 0) else max(1, cpu - 1)
	chosen_numba_threads = args.numba_threads if (args.numba_threads and args.numba_threads > 0) else 1
	# If user left backend at default or gave threads with ambiguous settings, prefer processes to avoid nested threading issues
	if args.backend not in ('threads','processes') or args.backend == 'threads' and (args.workers is None or args.workers <= 0):
		chosen_backend = 'processes'

	out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generated_rooms')
	os.makedirs(out_dir, exist_ok=True)

	# Stage 1: generate N rooms and persist normals for approx.py
	logging.info(f"Generating {N} rooms...")
	generated = []
	for i in tqdm(range(N), desc='Generating'):
		mask, normals, scene, refl, trans, dist = generate_floor_scene()
		# Stats
		nz_refl = int(np.count_nonzero(refl)); nz_trans = int(np.count_nonzero(trans))
		min_refl = float(np.min(refl)) if nz_refl > 0 else 0.0
		max_refl = float(np.max(refl)) if nz_refl > 0 else 0.0
		min_trans = float(np.min(trans)) if nz_trans > 0 else 0.0
		max_trans = float(np.max(trans)) if nz_trans > 0 else 0.0
		logging.debug(f"[{i}] refl nz={nz_refl} [{min_refl:.2f},{max_refl:.2f}], trans nz={nz_trans} [{min_trans:.2f},{max_trans:.2f}]")

		# Persist normals with unique building_id per sample
		building_id = i
		_save_normals_npz(normals, building_id, mask.shape)

		# Build sample aligned with approx.py expectations
		sample = build_sample_from_generated(mask, normals, scene, refl, trans, dist, building_id=building_id)
		generated.append((sample, refl, trans, scene))

	# Stage 2: approximate sequentially with tqdm
	logging.info("Approximating rooms...")
	model = Approx('combined')
	samples = [t[0] for t in generated]
	# Use built-in parallel prediction with selected defaults
	preds = model.predict(samples, num_workers=chosen_workers, numba_threads=chosen_numba_threads, backend=chosen_backend)
	for i, ((sample, refl, trans, scene), pred_t) in enumerate(zip(generated, preds)):
		pred = pred_t.cpu().numpy() if hasattr(pred_t, 'cpu') else np.array(pred_t)
		width_m = scene['canvas']['width_m']; height_m = scene['canvas']['height_m']
		freq = scene.get('frequency_MHz', 1800)
		viz_path = os.path.join(out_dir, f'pipeline_demo_{i:03d}.png')
		visualize_triplet(trans, refl, pred, sample.x_ant, sample.y_ant, width_m, height_m, freq, save_path=viz_path)
	logging.info("Done.")


if __name__ == "__main__":
	main()
