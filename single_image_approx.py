import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torchvision.io import read_image

from helper import RadarSample
from approx import calculate_combined_loss_with_normals, apply_backfill, N_ANGLES
from normal_parser import precompute_wall_angles_pca

# -----------------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------------
IMAGE_PATH = "/Users/xoren/Downloads/rt_dist.npy"
ANTENNA_X = 300.0
ANTENNA_Y = 500.0
FREQ_MHZ = 1800.0
OUTPUT_PATH = "/Users/xoren/icassp2025/foo/single_pred.png"


def load_input_image(image_path: str) -> torch.Tensor:
    """Load input as 3xHxW tensor from .npy or image file.
    - For .npy: accept (3,H,W) or (H,W,3); convert to float32 torch tensor
    - For images: use read_image (C,H,W), take first 3 channels
    """
    if image_path.lower().endswith('.npy'):
        arr = np.load(image_path)
        if arr.ndim == 3 and arr.shape[0] == 3:
            pass  # already (3,H,W)
        elif arr.ndim == 3 and arr.shape[-1] == 3:
            arr = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError(f"Expected .npy shape (3,H,W) or (H,W,3), got {arr.shape}")
        return torch.from_numpy(arr).to(torch.float32)
    else:
        img = read_image(image_path).to(torch.float32)
        if img.ndim != 3 or img.size(0) < 3:
            raise ValueError("Input image must have at least 3 channels (reflectance, transmittance, distance)")
        return img[:3]


def _nonzero_min_max(arr: np.ndarray) -> tuple:
    nz = arr[arr != 0]
    if nz.size == 0:
        return float('nan'), float('nan')
    return float(np.min(nz)), float(np.max(nz))


def build_sample(img: torch.Tensor, x_ant: float, y_ant: float, freq_mhz: float) -> RadarSample:
    _, H, W = img.shape
    radiation = torch.zeros(360, dtype=torch.float32)
    sample = RadarSample(
        H=H,
        W=W,
        x_ant=float(x_ant),
        y_ant=float(y_ant),
        azimuth=0.0,
        freq_MHz=float(freq_mhz),
        input_img=img,
        output_img=torch.zeros((H, W), dtype=torch.float32),
        radiation_pattern=radiation,
        pixel_size=0.25,
        mask=torch.ones((H, W), dtype=torch.float32),
        ids=(0, 1, 1, 0),
    )
    return sample


def save_prediction(pred: torch.Tensor, save_path: str, x_ant: float, y_ant: float) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mat = pred.detach().cpu().numpy()
    plt.figure(figsize=(8, 6))
    im = plt.imshow(mat)  # default colormap (viridis), not grayscale
    plt.plot([x_ant], [y_ant], 'r*', markersize=10)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    img = load_input_image(IMAGE_PATH)

    # Stats for channels
    refl = img[0].detach().cpu().numpy()
    trans = img[1].detach().cpu().numpy()
    refl_min, refl_max = _nonzero_min_max(refl)
    trans_min, trans_max = _nonzero_min_max(trans)

    # Keep reflectance/transmittance values as-is (no normalization)
    img_proc = img

    sample = build_sample(img_proc, ANTENNA_X, ANTENNA_Y, FREQ_MHZ)

    # Compute wall normals on-the-fly from binary building mask
    refl_np = img_proc[0].detach().cpu().numpy()
    trans_np = img_proc[1].detach().cpu().numpy()
    building_mask = (refl_np + trans_np > 0).astype(np.uint8)
    angles = precompute_wall_angles_pca(building_mask)
    rad = np.deg2rad(angles + 90.0)
    nx = np.cos(rad).astype(np.float64)
    ny = np.sin(rad).astype(np.float64)
    invalid = angles < 0
    if np.any(invalid):
        nx = nx.copy(); ny = ny.copy()
        nx[invalid] = 0.0
        ny[invalid] = 0.0

    # Combined approximation with normals
    out_np, cnt_np = calculate_combined_loss_with_normals(
        refl_np.astype(np.float64),
        trans_np.astype(np.float64),
        nx, ny,
        float(ANTENNA_X),
        float(ANTENNA_Y),
        float(FREQ_MHZ),
        n_angles=N_ANGLES,
    )
    out_np = out_np.astype(np.float32)
    cnt_np = cnt_np.astype(np.float32)
    out_np = apply_backfill(
        out_np,
        cnt_np,
        float(ANTENNA_X),
        float(ANTENNA_Y),
        0.25,
        float(FREQ_MHZ),
        32000.0,
        trans_mat=trans_np.astype(np.float32),
    )
    pred = torch.from_numpy(out_np)

    # Save prediction image with default colormap
    save_prediction(pred.to(torch.float32), OUTPUT_PATH, float(ANTENNA_X), float(ANTENNA_Y))

    pred_np = pred.detach().cpu().numpy()
    pred_min, pred_max = _nonzero_min_max(pred_np)

    print(f"Saved prediction to: {OUTPUT_PATH}")
    print(f"Prediction min/max: {pred_min:.4f} / {pred_max:.4f}")
    print(f"Reflectance min/max: {refl_min:.4f} / {refl_max:.4f}")
    print(f"Transmittance min/max: {trans_min:.4f} / {trans_max:.4f}")


if __name__ == "__main__":
    main()


