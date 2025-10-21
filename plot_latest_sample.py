import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def find_latest_sample(base_runs_dir: str) -> str:
    runs = [d for d in glob.glob(os.path.join(base_runs_dir, "*")) if os.path.isdir(d)]
    if not runs:
        raise FileNotFoundError(f"No run directories found in {base_runs_dir}")
    latest_run = max(runs, key=os.path.getmtime)
    sample_dirs = [d for d in glob.glob(os.path.join(latest_run, "s*")) if os.path.isdir(d)]
    if not sample_dirs:
        raise FileNotFoundError(f"No sample directories found in {latest_run}")
    latest_sample_dir = max(sample_dirs, key=os.path.getmtime)
    npz_files = glob.glob(os.path.join(latest_sample_dir, "*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No NPZ file found in {latest_sample_dir}")
    return npz_files[0]


def main():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "synthetic")
    npz_path = find_latest_sample(base)

    data = np.load(npz_path)
    trans = np.asarray(data["transmittance"], dtype=np.float32)
    refl = np.asarray(data["reflectance"], dtype=np.float32)
    pl = np.asarray(data["pathloss"], dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im0 = axes[0].imshow(trans)
    axes[0].set_title("Transmittance")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(refl)
    axes[1].set_title("Reflectance")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(pl)
    axes[2].set_title("Pathloss")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(base), "preview")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "latest_sample.png")
    plt.savefig(out_path, dpi=150)
    print(out_path)


if __name__ == "__main__":
    main()


