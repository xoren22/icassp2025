import os
import argparse
import numpy as np
from typing import List, Tuple
from torchvision.io import read_image


def build_paths() -> Tuple[str, str, str]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data/train/")
    input_dir = os.path.join(data_dir, "Inputs/Task_2_ICASSP/")
    return base_dir, data_dir, input_dir


def list_available_samples_for_building(building_id: int) -> List[str]:
    """Return list of input image paths for a given building id."""
    _, _, input_dir = build_paths()
    paths: List[str] = []
    for ant in (1, 2):
        for f_idx in (1, 2, 3):
            for sp in range(80):
                name = f"B{building_id}_Ant{ant}_f{f_idx}_S{sp}.png"
                inp = os.path.join(input_dir, name)
                if os.path.exists(inp):
                    paths.append(inp)
    return paths


def load_mask_from_input_png(input_path: str) -> np.ndarray:
    """Load a single input image and return binary wall mask (H,W) as bool."""
    img = read_image(input_path).numpy()  # (C,H,W), uint8
    refl = img[0]
    trans = img[1]
    mask = (refl + trans) > 0
    return mask


def compare_masks_for_building(building_id: int, num_samples: int = 5, seed: int = 0) -> bool:
    paths = list_available_samples_for_building(building_id)
    if not paths:
        print(f"Building {building_id}: no inputs found.")
        return False

    rng = np.random.default_rng(seed + building_id)
    sel = rng.choice(len(paths), size=min(num_samples, len(paths)), replace=False)
    sel_paths = [paths[i] for i in sel]

    ref_mask = None
    all_match = True
    for idx, p in enumerate(sel_paths):
        m = load_mask_from_input_png(p)
        if ref_mask is None:
            ref_mask = m
            ref_name = os.path.basename(p)
            continue
        if ref_mask.shape != m.shape:
            print(f"Building {building_id}: shape mismatch {ref_mask.shape} vs {m.shape} for {os.path.basename(p)}")
            all_match = False
            continue
        diff = np.logical_xor(ref_mask, m)
        diff_count = int(diff.sum())
        if diff_count > 0:
            all_match = False
            print(f"Building {building_id}: mismatch vs ref ({ref_name}) in {diff_count} pixels for {os.path.basename(p)}")
    if all_match:
        print(f"Building {building_id}: all {len(sel_paths)} sampled masks match.")
    return all_match


def main():
    parser = argparse.ArgumentParser("Check mask consistency within buildings")
    parser.add_argument("--buildings", type=int, nargs="*", default=None,
                        help="Building ids to check (default: all 1..25)")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of random samples per building to compare")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    buildings = args.buildings if args.buildings else list(range(1, 26))

    ok = True
    for b in buildings:
        same = compare_masks_for_building(b, num_samples=args.num_samples, seed=args.seed)
        ok = ok and same

    if not ok:
        print("Some buildings have inconsistent masks.")
    else:
        print("All checked buildings have consistent masks.")


if __name__ == "__main__":
    main() 