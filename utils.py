import os
import re
import time
import torch
import numpy as np
import pickle as pkl
from typing import List
import matplotlib.pyplot as plt
from contextlib import contextmanager

from model import UNetModel
from _types import RadarSampleInputs


@contextmanager
def measure_time(label="Execution"):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        elapsed = end - start
        print(f"{label} took {elapsed:.4f} seconds.")

def evaluate_fspl(output_img, fspl):
    """
    Compare how well FSPL predicts the ground truth (output_img).
    
    Args:
        output_img: 2D (H,W) array of ground-truth pathloss values
        fspl: 2D (H,W) array of FSPL-based pathloss predictions
        
    Prints:
        - Baseline RMSE: If we always predict mean(output_img)
        - FSPL RMSE
        - FSPL R^2
    """
    # Flatten to 1D for convenience
    y = output_img.flatten().astype(np.float32)
    y_pred = fspl.flatten().astype(np.float32)
    
    # Baseline: always predict the average ground truth
    y_mean = y.mean()
    baseline_rmse = np.sqrt(np.mean((y - y_mean)**2))
    
    # FSPL RMSE
    diff = (y - y_pred)
    fspl_rmse = np.sqrt(np.mean((diff - diff.mean())**2))
    
    # R^2 = 1 - SS_res / SS_tot
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y_mean)**2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float('nan')
    
    print(f"Baseline RMSE (predict mean): {baseline_rmse:.4f}")
    print(f"FSPL RMSE:                   {fspl_rmse:.4f}")
    print(f"FSPL R^2:                    {r2:.4f}")
    print("-"*100, "\n\n")


def matrix_to_image(*matrices, titles=None, save_path=None):
    n = abs(int(matrices[1].sum()))
    save_path = save_path or f"foo/1_{n}.png"
    if len(matrices) < 2:
        raise ValueError("At least two matrices are required: 1 ground truth + 1 comparison.")

    ground_truth = matrices[0]
    others = matrices[1:]
    n = len(others)
    if not titles or len(titles) != n:
        titles = [f"Matrix {i+1}" for i in range(n)]

    diffs = []
    for mat in others:
        diffs.append(np.abs(ground_truth - mat))

    all_mats = [ground_truth] + list(others) + diffs
    vmin = min(m.min() for m in all_mats)
    vmax = max(m.max() for m in all_mats)

    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(15, 5*n))
    if n == 1:
        axes = np.array([axes])

    for i in range(n):
        im_gt = axes[i, 0].imshow(ground_truth, cmap='coolwarm', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title("Ground Truth")
        fig.colorbar(im_gt, ax=axes[i, 0])

        im_cmp = axes[i, 1].imshow(others[i], cmap='coolwarm', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(titles[i])
        fig.colorbar(im_cmp, ax=axes[i, 1])

        im_diff = axes[i, 2].imshow(diffs[i], cmap='coolwarm', vmin=vmin, vmax=vmax)
        axes[i, 2].set_title("Absolute Diff")
        fig.colorbar(im_diff, ax=axes[i, 2])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved image at {save_path}")
    return diffs

def load_model(weights_path=None, device=None):    
    model = UNetModel()
    if not os.path.isfile(weights_path):
        raise ValueError(f"Model weights_path {weights_path} doesn't exist.")
    
    model.load_state_dict(torch.load(weights_path))
    if device:
        model = model.to(device)
    print(f"Loaded model params from '{weights_path}' and moved to device '{device}")
    return model

def fname_to_ids(fnames, list):
    if not isinstance(fnames):
        fnames = [fnames]
    ids = [list(map(int, re.findall(r"\d+", fname.split(".")[0]))) for fname in fnames]
    if len(fnames) == 1:
        return ids[0]
    return ids


def check_tensor_device(tensor, name, expected_device='cpu'):
    if tensor.device.type != expected_device:
        print(f"WARNING: {name} is on {tensor.device} but should be on {expected_device}")
        return False
    return True


def split_data_task1(inputs_list: List[RadarSampleInputs], val_ratio=0.25, split_save_path=None, seed=None):
    building_ids = list(set([f.ids[0] for f in inputs_list]))
    np.random.seed(seed=seed)
    np.random.shuffle(building_ids)

    n_buildings_total = len(building_ids)  
    n_buildings_valid = int(n_buildings_total*val_ratio)

    if n_buildings_total == 0 or n_buildings_valid == 0:
        raise ValueError(f"Invalid split, total number of buildings: {n_buildings_total}, ratio of validation set: {val_ratio}. Number of validation buildings {n_buildings_valid}")
    
    val_buildings = building_ids[:n_buildings_valid]
    train_buildings = building_ids[n_buildings_valid:]

    val_inputs, train_inputs = [], []
    for f in inputs_list:
        if f.ids[0] in val_buildings:
            val_inputs.append(f)
        else:
            train_inputs.append(f)
    if split_save_path:
        with open(split_save_path, "wb") as f:
            split_dict = {
                "val_inputs": val_inputs,
                "train_inputs": train_inputs,
            }
            pkl.dump(split_dict, f)
    return train_inputs, val_inputs



def split_data_task2(inputs_list: List[RadarSampleInputs], val_freqs, split_save_path=None):
    train_inputs, val_inputs = split_data_task1(inputs_list)
    val_freqs = val_freqs if isinstance(val_freqs, list) else [val_freqs]
    val_inputs = [f for f in val_inputs if f.ids[2] in val_freqs]
    train_inputs = [f for f in train_inputs if f.ids[2] not in val_freqs]

    if split_save_path:
        with open(split_save_path, "wb") as fp:
            pkl.dump({
                "train_inputs": train_inputs, 
                "val_inputs": val_inputs,
                "val_freqs": val_freqs}, fp
            )
    return train_inputs, val_inputs

