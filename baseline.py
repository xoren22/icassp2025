import os
import numpy as np
import torch
from tqdm import tqdm
from typing import List

from _types import RadarSampleInputs
from data_module import read_sample
from approx import _compute_fspl_field, calculate_transmission_loss_numpy, MAX_TRANS

def load_task2_data(data_dir=None, max_samples=None):
    """
    Load all Task 2 data samples.

    Args:
        data_dir: Base data directory (defaults to data/train/)
        max_samples: Maximum number of samples to load (None = all)

    Returns:
        List of RadarSampleInputs objects
    """
    if data_dir is None:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(BASE_DIR, "data/train/")

    INPUT_DIR = os.path.join(data_dir, "Inputs/Task_2_ICASSP/")
    OUTPUT_DIR = os.path.join(data_dir, "Outputs/Task_2_ICASSP/")
    POS_DIR = os.path.join(data_dir, "Positions/")
    PAT_DIR = os.path.join(data_dir, "Radiation_Patterns/")

    freqs_MHz = [868, 1800, 3500]
    inputs_list: List[RadarSampleInputs] = []

    print("Loading Task 2 data samples...")

    for b in tqdm(range(1, 26), desc="Buildings"):
        for f_idx, freq_MHz in enumerate(freqs_MHz, start=1):
            for sp in range(80):
                ant = 1
                name = f"B{b}_Ant{ant}_f{f_idx}_S{sp}.png"
                inp = os.path.join(INPUT_DIR, name)
                out = os.path.join(OUTPUT_DIR, name)
                if not (os.path.exists(inp) and os.path.exists(out)):
                    continue
                radiation_file = os.path.join(PAT_DIR, f"Ant{ant}_Pattern.csv")
                position_file = os.path.join(POS_DIR, f"Positions_B{b}_Ant{ant}_f{f_idx}.csv")
                inputs_list.append(
                    RadarSampleInputs(
                        freq_MHz=freq_MHz,
                        input_file=inp,
                        output_file=out,
                        position_file=position_file,
                        radiation_pattern_file=radiation_file,
                        sampling_position=sp,
                        ids=(b, ant, f_idx, sp),
                    )
                )

    print(f"Loaded {len(inputs_list)} samples")

    if max_samples and len(inputs_list) > max_samples:
        inputs_list = inputs_list[:max_samples]
        print(f"Limited to {max_samples} samples")

    return inputs_list

def compute_fspl_field(sample):
    H, W = sample.H, sample.W
    x_ant, y_ant = sample.x_ant, sample.y_ant
    freq_MHz = sample.freq_MHz
    pixel_size = sample.pixel_size
    fspl_field = _compute_fspl_field(
        h=H, w=W,
        x_ant=x_ant, y_ant=y_ant,
        pixel_size=pixel_size,
        freq_MHz=freq_MHz,
        max_loss=160.0
    )

    return fspl_field

def calculate_transmission_loss(sample):
    trans, _ = calculate_transmission_loss_numpy(
        sample.input_img[1].cpu().numpy(), sample.x_ant, sample.y_ant, sample.freq_MHz,
        n_angles=360*128, max_walls=MAX_TRANS
    )
    return trans

def evaluate_baselines_on_dataset(inputs_list, max_samples=None):
    baseline_names = ['fspl', 'tx']
    baselines = [compute_fspl_field, calculate_transmission_loss]
    baseline_results = {name: {'se': 0.0, 'total_pixels': 0} for name in baseline_names}

    if max_samples:
        inputs_list = inputs_list[:max_samples]

    print(f"Evaluating {len(baseline_names)} baselines on {len(inputs_list)} samples...")

    for sample_input in tqdm(inputs_list, desc="Evaluating"):
        sample = read_sample(sample_input)
        for baseline_func, name in zip(baselines, baseline_names):
            field = baseline_func(sample)
            baseline_results[name]['se'] += np.sum((sample.output_img.numpy() - field) ** 2)
            baseline_results[name]['total_pixels'] += sample.output_img.numel()

    for name, result in baseline_results.items():
        result['rmse'] = np.sqrt(result['se'] / result['total_pixels'])
    
    return baseline_results


def main():
    """Main function to run FSPL and Tx-only evaluation on Task 2 data."""
    print("FSPL and Tx-only RMSE Evaluation on Task 2 Dataset")
    print("=" * 60)

    inputs_list = load_task2_data()
    results = evaluate_baselines_on_dataset(inputs_list)

    print("\nFinal Results:")
    for name, result in results.items():
        print(f"{name.capitalize()} RMSE:   {result['rmse']:.4f}")
        print(f"Total pixels: {result['total_pixels']:,}")

if __name__ == "__main__":
    main()
