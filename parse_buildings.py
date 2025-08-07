import os
import argparse
import numpy as np
from typing import List
from torchvision.io import read_image
from normal_parser import precompute_wall_angles_pca


def build_paths():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data/train/")
    input_dir = os.path.join(data_dir, "Inputs/Task_2_ICASSP/")
    out_dir = os.path.join(base_dir, "parsed_buildings")
    os.makedirs(out_dir, exist_ok=True)
    return input_dir, out_dir


def find_one_input_for_building(building_id: int) -> str:
    input_dir, _ = build_paths()
    for ant in (1, 2):
        for f_idx in (1, 2, 3):
            for sp in range(80):
                name = f"B{building_id}_Ant{ant}_f{f_idx}_S{sp}.png"
                p = os.path.join(input_dir, name)
                if os.path.exists(p):
                    return p
    return ""


def load_building_mask(input_path: str) -> np.ndarray:
    img = read_image(input_path).numpy()  # (C,H,W)
    refl = img[0]
    trans = img[1]
    mask = (refl + trans) > 0
    return mask.astype(np.uint8)


def angles_to_normals(angles_deg: np.ndarray) -> tuple:
    # Normal is wall angle + 90 degrees
    rad = np.deg2rad(angles_deg + 90.0)
    nx = np.cos(rad)
    ny = np.sin(rad)
    # For pixels with invalid angles (<0), set zeros
    invalid = angles_deg < 0
    if invalid.any():
        nx = nx.copy(); ny = ny.copy()
        nx[invalid] = 0.0
        ny[invalid] = 0.0
    return nx.astype(np.float32), ny.astype(np.float32)


def process_building(building_id: int, overwrite: bool = True) -> str:
    input_dir, out_dir = build_paths()
    out_path = os.path.join(out_dir, f"B{building_id}_normals.npz")
    if os.path.exists(out_path) and not overwrite:
        print(f"B{building_id}: already exists → {out_path}")
        return out_path

    inp = find_one_input_for_building(building_id)
    if not inp:
        print(f"B{building_id}: no input found in {input_dir}")
        return ""

    mask = load_building_mask(inp)
    print(f"B{building_id}: computing wall angles on {mask.shape[1]}x{mask.shape[0]} map…")
    angles = precompute_wall_angles_pca(mask)
    nx, ny = angles_to_normals(angles)

    np.savez_compressed(out_path, nx=nx, ny=ny, angles=angles.astype(np.float32))
    print(f"B{building_id}: saved → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser("Parse wall normals for buildings")
    parser.add_argument("--buildings", type=int, nargs="*", default=None,
                        help="Building ids (default: 1..25)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if output exists")
    args = parser.parse_args()

    buildings: List[int] = args.buildings if args.buildings else list(range(1, 26))

    done = 0
    for b in buildings:
        p = process_building(b, overwrite=args.overwrite)
        if p:
            done += 1
    print(f"Finished: {done}/{len(buildings)} saved")


if __name__ == "__main__":
    main() 