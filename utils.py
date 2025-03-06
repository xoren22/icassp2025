import time
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager


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


def matrix_to_image(*matrices, titles=None):
    if len(matrices) < 2:
        raise ValueError("At least two matrices are required: 1 ground truth + 1 comparison.")

    ground_truth = matrices[0]
    others = matrices[1:]   # comparison matrices
    n = len(others)

    # If no custom titles are provided or the number of titles is off, create default titles
    if not titles or len(titles) != n:
        titles = [f"Matrix {i+1}" for i in range(n)]

    # Create n x 3 subplots
    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(15, 5*n))

    # If there's only one row, 'axes' will be a 1D array; force it to 2D
    if n == 1:
        axes = np.array([axes])

    diffs = []  # Collect difference matrices

    for i in range(n):
        comparison_matrix = others[i]
        diff = np.abs(ground_truth - comparison_matrix)
        diffs.append(diff)

        # Ground truth
        im_gt = axes[i, 0].imshow(ground_truth, cmap='coolwarm')
        axes[i, 0].set_title("Ground Truth")
        fig.colorbar(im_gt, ax=axes[i, 0])

        # Comparison
        im_comp = axes[i, 1].imshow(comparison_matrix, cmap='coolwarm')
        axes[i, 1].set_title(titles[i])
        fig.colorbar(im_comp, ax=axes[i, 1])

        # Absolute difference
        im_diff = axes[i, 2].imshow(diff, cmap='coolwarm')
        axes[i, 2].set_title("Absolute Diff")
        fig.colorbar(im_diff, ax=axes[i, 2])

    plt.tight_layout()
    plt.savefig(f"foo/{time.time()}.png")
    plt.close()

    return diffs

