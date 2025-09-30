import os
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import pandas as pd
import shutil
import time
import secrets

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


def _build_synthetic_dataset(args):
	"""
	Generate a synthetic dataset compatible with main.py expectations (Task_2_ICASSP):
	  - Inputs/Task_2_ICASSP/B{b}_Ant{ant}_f{f}_S{sp}.png (RGB: refl, trans, distance[m])
	  - Outputs/Task_2_ICASSP/B{b}_Ant{ant}_f{f}_S{sp}.png (L pathloss dB)
	  - Positions/Positions_B{b}_Ant{ant}_f{f}.csv (80 rows; X,Y,Azimuth)
	  - Radiation_Patterns/Ant{1,2}_Pattern.csv (copied or isotropic fallback)
	"""
	base_dir = os.path.dirname(os.path.abspath(__file__))
	# Resolve output dir
	out_dir = args.data_out
	if not os.path.isabs(out_dir):
		out_dir = os.path.join(base_dir, out_dir)

	inputs_dir = os.path.join(out_dir, 'Inputs', 'Task_2_ICASSP')
	outputs_dir = os.path.join(out_dir, 'Outputs', 'Task_2_ICASSP')
	positions_dir = os.path.join(out_dir, 'Positions')
	rp_dir = os.path.join(out_dir, 'Radiation_Patterns')
	for d in (inputs_dir, outputs_dir, positions_dir, rp_dir):
		os.makedirs(d, exist_ok=True)

	# Copy radiation patterns for Ant1, Ant2 if present; else create isotropic zeros
	src_rp_dir = os.path.join(base_dir, 'data', 'train', 'Radiation_Patterns')
	for ant in (1, 2):
		dst = os.path.join(rp_dir, f'Ant{ant}_Pattern.csv')
		if not os.path.exists(dst):
			src = os.path.join(src_rp_dir, f'Ant{ant}_Pattern.csv')
			if os.path.exists(src):
				shutil.copyfile(src, dst)
			else:
				pd.Series([0.0]*360).to_csv(dst, index=False, header=False)

	# Ensure a positions CSV exists with 80 rows and index 0..79
	def ensure_positions_csv(b: int, ant: int, f: int, x_ant: int, y_ant: int, azimuth: int = 0):
		path = os.path.join(positions_dir, f'Positions_B{b}_Ant{ant}_f{f}.csv')
		if os.path.exists(path):
			return path
		df = pd.DataFrame({
			'X': [x_ant]*80,
			'Y': [y_ant]*80,
			'Azimuth': [azimuth]*80,
		})
		df.to_csv(path, index=True)
		return path

	N = int(max(1, args.num))
	approx_model = Approx('combined')
	sp_counters = {}
	logging.info(f"Building synthetic dataset with {N} samples at {out_dir}")

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

		b = (i % 10) + 1
		ant_id = (i % 2) + 1
		key = (b, ant_id, f_idx)
		sp = sp_counters.get(key, 0)
		if sp >= 80:
			# find another slot
			found = False
			for bb in range(1, 11):
				for aa in (1, 2):
					for ff in (1, 2, 3):
						k2 = (bb, aa, ff)
						if sp_counters.get(k2, 0) < 80:
							key = k2; b, ant_id, f_idx = k2; sp = sp_counters.get(k2, 0); found = True; break
						# end for ff
					if found: break
				# end for aa
				if found: break
			# end for bb
			if not found:
				break

		# Persist normals for this synthetic building id
		_save_normals_npz(normals, b, mask.shape)

		sample = build_sample_from_generated(mask, normals, scene, refl, trans, dist, building_id=b)
		# Override ids to (b, ant, f, sp)
		sample.ids = (int(b), int(ant_id), int(f_idx), int(sp))

		# Predict pathloss map
		pred = approx_model.approximate(sample).cpu().numpy().astype(np.float32)
		pred = np.clip(pred, 0.0, 255.0)

		# Prepare input RGB image
		refl_u8 = np.clip(refl * 10.0, 0.0, 255.0).astype(np.uint8)
		trans_u8 = np.clip(trans * 10.0, 0.0, 255.0).astype(np.uint8)
		dist_m = dist * 0.25
		dist_u8 = np.clip(dist_m, 0.0, 255.0).astype(np.uint8)
		rgb = np.stack([refl_u8, trans_u8, dist_u8], axis=-1)

		in_name = f'B{b}_Ant{ant_id}_f{f_idx}_S{sp}.png'
		out_name = in_name
		Image.fromarray(rgb, mode='RGB').save(os.path.join(inputs_dir, in_name))
		Image.fromarray(pred.astype(np.uint8), mode='L').save(os.path.join(outputs_dir, out_name))

		# Ensure positions CSV
		ensure_positions_csv(b, ant_id, f_idx, x_ant=x_ant, y_ant=y_ant, azimuth=0)

		sp_counters[key] = sp + 1

	logging.info("Synthetic dataset export complete.")


def main():
	import argparse
	logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

	parser = argparse.ArgumentParser("Room generation + approximation pipeline")
	parser.add_argument('--num', type=int, default=5, help='Number of rooms to generate and approximate')
	parser.add_argument('--batch_size', type=int, default=10, help='Batch size for generate->predict->save streaming')
	parser.add_argument('--workers', type=int, default=0, help='Parallel workers for prediction (0/1 = auto)')
	parser.add_argument('--backend', type=str, default='processes', choices=['threads','processes'], help='Parallel backend for prediction')
	parser.add_argument('--numba_threads', type=int, default=0, help='Numba threads per worker (0 = auto)')
	parser.add_argument('--make_dataset', action='store_true', help='Export synthetic dataset matching data/train structure (Task_2_ICASSP)')
	parser.add_argument('--data_out', type=str, default='data/synthetic', help='Output dataset directory')
	parser.add_argument('--run_id', type=str, default=None, help='Unique run identifier; auto-generated if omitted')
	args = parser.parse_args()

	# If requested, build synthetic dataset and exit
	if args.make_dataset:
		_build_synthetic_dataset(args)
		return

	N = int(max(1, args.num))
	B = int(max(1, args.batch_size))

	# Choose fast, safe defaults if not provided: processes backend, 1 Numba thread per worker
	import multiprocessing as _mp
	cpu = max(1, (_mp.cpu_count() or 2))
	chosen_backend = args.backend
	chosen_workers = args.workers if (args.workers and args.workers > 0) else max(1, cpu - 1)
	chosen_numba_threads = args.numba_threads if (args.numba_threads and args.numba_threads > 0) else 1
	# If user left backend at default or gave threads with ambiguous settings, prefer processes to avoid nested threading issues
	if args.backend not in ('threads','processes') or args.backend == 'threads' and (args.workers is None or args.workers <= 0):
		chosen_backend = 'processes'

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

	# Streamed Stage 1+2: generate->predict->save in batches
	logging.info(f"Processing {N} samples in batches of {B} (backend={chosen_backend}, workers={chosen_workers})...")
	model = Approx('combined')
	global_idx = 0
	for start in tqdm(range(0, N, B), desc='Batches'):
		end = min(start + B, N)
		batch = []  # (sample, refl, trans, scene, gidx)
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
			batch.append((sample, refl, trans, scene, global_idx))
			global_idx += 1

		# Predict for this batch
		samples = [t[0] for t in batch]
		preds = model.predict(samples, num_workers=chosen_workers, numba_threads=chosen_numba_threads, backend=chosen_backend)
		for (sample, refl, trans, scene, gidx), pred_t in zip(batch, preds):
			pred = pred_t.cpu().numpy() if hasattr(pred_t, 'cpu') else np.array(pred_t)
			width_m = scene['canvas']['width_m']; height_m = scene['canvas']['height_m']
			freq = scene.get('frequency_MHz', 1800)
			viz_name = f'pipeline_demo_{gidx:06d}_{run_id}.png'
			viz_path = os.path.join(out_dir, viz_name)
			visualize_triplet(trans, refl, pred, sample.x_ant, sample.y_ant, width_m, height_m, freq, save_path=viz_path)

	logging.info("All batches processed. Done.")


if __name__ == "__main__":
	main()
