import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Union, Tuple, Optional, List
from dataclasses import dataclass, asdict
from torchvision.io import read_image


@dataclass
class RadarSample:
    H: int
    W: int
    x_ant: float
    y_ant: float
    azimuth: float
    freq_MHz: float
    input_img: torch.Tensor  # In format (C, H, W)
    output_img: torch.Tensor  # In format (H, W) or (1, H, W)
    radiation_pattern: torch.Tensor
    pixel_size: float = 0.25
    mask: Union[torch.Tensor, None] = None
    ids: Optional[List[Tuple[int, int, int, int]]] = None

    def copy(self):
        return RadarSample(
            self.H,
            self.W,
            self.x_ant,
            self.y_ant,
            self.azimuth,
            self.freq_MHz,
            self.input_img,  
            self.output_img, 
            self.radiation_pattern,
            self.pixel_size,
            self.mask,
            self.ids,
        )


@dataclass
class RadarSampleInputs:
    freq_MHz: float
    input_file: str
    output_file: Union[str, None]
    position_file: str
    radiation_pattern_file: str
    sampling_position : int
    ids: Optional[Tuple[int, int, int, int]] = None

    def asdict(self):
        return asdict(self)
    
    def __post_init__(self):
        if self.ids and not all(isinstance(i, int) for i in self.ids):
            raise ValueError("All IDs must be integers")
        
        if not isinstance(self.freq_MHz, (int, float)):
            raise ValueError("freq_MHz must be a number")
        
        for path_attr in ['input_file', 'position_file', 'radiation_pattern_file']:
            path = getattr(self, path_attr)
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")


def read_sample(inputs: Union[RadarSampleInputs, dict]) -> RadarSample:
    """Read a single radar sample from file paths."""
    if isinstance(inputs, RadarSampleInputs):
        inputs = inputs.asdict()

    freq_MHz = inputs["freq_MHz"]
    input_file = inputs["input_file"]
    output_file = inputs.get("output_file")
    position_file = inputs["position_file"]
    sampling_position = inputs["sampling_position"]
    radiation_pattern_file = inputs["radiation_pattern_file"]
    
    input_img = read_image(input_file).float()
    C, H, W = input_img.shape
    
    output_img = None
    if output_file:
        output_img = read_image(output_file).float()
        if output_img.size(0) == 1:  # If single channel, remove channel dimension
            output_img = output_img.squeeze(0)
        
    sampling_positions = pd.read_csv(position_file)
    x_ant, y_ant, azimuth = sampling_positions.loc[int(sampling_position), ["Y", "X", "Azimuth"]]
    
    radiation_pattern_np = np.genfromtxt(radiation_pattern_file, delimiter=',')
    radiation_pattern = torch.from_numpy(radiation_pattern_np).float()

    sample = RadarSample(
        H=H,
        W=W,
        x_ant=x_ant,
        y_ant=y_ant,
        azimuth=azimuth,
        freq_MHz=freq_MHz,
        input_img=input_img,
        output_img=output_img,
        pixel_size=0.25,  # INITIAL_PIXEL_SIZE
        mask=torch.ones((H, W)),
        radiation_pattern=radiation_pattern,
    )
    
    return sample


def load_samples(num_samples: int = 5) -> List[RadarSample]:
    """Load a specified number of radar samples from the default data directory."""
    # Define paths (matching main.py structure)
    freqs_MHz = [868, 1800, 3500]
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data/train/")
    INPUT_PATH = os.path.join(DATA_DIR, "Inputs/Task_2_ICASSP/")
    OUTPUT_PATH = os.path.join(DATA_DIR, "Outputs/Task_2_ICASSP/")
    POSITIONS_PATH = os.path.join(DATA_DIR, "Positions/")
    RADIATION_PATTERNS_PATH = os.path.join(DATA_DIR, "Radiation_Patterns/")
    
    # Build list of available samples
    inputs_list = []
    for b in range(1, 26):  # 25 buildings
        for ant in range(1, 3):  # 2 antenna types
            for f in range(1, 4):  # 3 frequencies
                for sp in range(80):  # 80 sampling positions
                    # Check if files exist
                    input_file = f"B{b}_Ant{ant}_f{f}_S{sp}.png"
                    output_file = f"B{b}_Ant{ant}_f{f}_S{sp}.png"
                    
                    input_img_path = os.path.join(INPUT_PATH, input_file)
                    output_img_path = os.path.join(OUTPUT_PATH, output_file)
                    
                    if os.path.exists(input_img_path) and os.path.exists(output_img_path):
                        radiation_file = f"Ant{ant}_Pattern.csv"
                        position_file = f"Positions_B{b}_Ant{ant}_f{f}.csv"

                        freq_MHz = freqs_MHz[f-1]
                        positions_path = os.path.join(POSITIONS_PATH, position_file)
                        radiation_pattern_file = os.path.join(RADIATION_PATTERNS_PATH, radiation_file)

                        radar_sample_inputs = RadarSampleInputs(
                            freq_MHz=freq_MHz,
                            input_file=input_img_path,
                            output_file=output_img_path,
                            position_file=positions_path,
                            radiation_pattern_file=radiation_pattern_file,
                            sampling_position=sp,
                            ids=(b, ant, f, sp),
                        )
                        inputs_list.append(radar_sample_inputs)
                        
                        # Stop if we have enough samples
                        if len(inputs_list) >= num_samples:
                            break
                if len(inputs_list) >= num_samples:
                    break
            if len(inputs_list) >= num_samples:
                break
        if len(inputs_list) >= num_samples:
            break
    
    # Convert to RadarSample objects
    samples = []
    np.random.shuffle(inputs_list)
    for inputs in tqdm(inputs_list[-num_samples:], desc="Loading samples"):
        sample = read_sample(inputs)
        samples.append(sample)
    
    return samples


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate RMSE between prediction and target tensors."""
    pred_np = pred.cpu().numpy() if hasattr(pred, 'cpu') else pred
    target_np = target.cpu().numpy() if hasattr(target, 'cpu') else target
    
    # Flatten arrays
    pred_flat = pred_np.flatten()
    target_flat = target_np.flatten()
    
    # Calculate RMSE
    mse = np.mean((pred_flat - target_flat) ** 2)
    return np.sqrt(mse)


def compare_two_matrices(matrix1: np.ndarray, matrix2: np.ndarray, 
                        title1: str = "Matrix 1", title2: str = "Matrix 2",
                        save_path: str = None) -> None:
    """Compare two matrices side by side with their difference."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Calculate value range for consistent coloring
    vmin = min(matrix1.min(), matrix2.min())
    vmax = max(matrix1.max(), matrix2.max())
    
    # Plot first matrix
    im1 = axes[0].imshow(matrix1, vmin=vmin, vmax=vmax)
    axes[0].set_title(f'{title1}\n(Mean: {matrix1.mean():.2f})')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.colorbar(im1, ax=axes[0])
    
    # Plot second matrix
    im2 = axes[1].imshow(matrix2, vmin=vmin, vmax=vmax)
    axes[1].set_title(f'{title2}\n(Mean: {matrix2.mean():.2f})')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(im2, ax=axes[1])
    
    # Plot difference
    diff = matrix1 - matrix2
    im3 = axes[2].imshow(diff, cmap='gray')
    axes[2].set_title(f'Difference\n(Mean: {diff.mean():.2f}, RMSE: {np.sqrt(np.mean(diff**2)):.2f})')
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to: {save_path}")
    else:
        plt.show()
    
    plt.close() 


def visualize_predictions(samples, gts, preds_c, preds_t, n=3, save_path=None, trans_mask=None):
    n = min(n, len(samples))
    fig, axs = plt.subplots(n, 6, figsize=(24,4*n))
    for i in range(n):
        s=samples[i]
        gt, pc, pt = gts[i], preds_c[i], preds_t[i]
        diff = pc-pt
        err = gt-pc
        
        # Calculate RMSE values
        rmse_combined = rmse(pc, gt)
        rmse_transmission = rmse(pt, gt)
        
        for j,(mat,title) in enumerate([
            (gt,       "GT"),
            (pc,       f"Combined (RMSE: {rmse_combined:.2f})"),
            (pt,       f"Transmission (RMSE: {rmse_transmission:.2f})"),
            (err,      "GT-Comb"),
            (diff,     "Comb-Trans"),
            (gt-pt,    "GT-Trans"),
        ]):
            if j > 2:
                im=axs[i,j].imshow(np.abs(mat), cmap='gray')
            else:
                tm = trans_mask[i]
                im=axs[i,j].imshow(mat+tm.cpu().numpy(), vmax=160)
            axs[i,j].set_title(f"{title}")
            axs[i,j].plot(s.x_ant, s.y_ant, 'r*')
            axs[i,j].axis('off')
            fig.colorbar(im, ax=axs[i,j])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


def _show_hit_stats(counts, title, ax_hm, ax_hist):
    vmax = np.percentile(counts, 99)
    im   = ax_hm.imshow(counts, cmap="magma", vmin=0, vmax=vmax)
    ax_hm.set_title(f"{title}\nmax={counts.max():.0f}", fontsize=10)
    ax_hm.axis("off"); plt.colorbar(im, ax=ax_hm, fraction=0.05)

    flat = counts.ravel()
    ax_hist.hist(flat, bins=np.arange(flat.max()+2)-0.5, log=True, edgecolor="k", linewidth=0.3)
    ax_hist.set_xlabel("hits / pixel"); ax_hist.set_ylabel("#pixels (log)")
    ax_hist.set_title("Histogram")

def compare_hit_counts(tx_cnt, cmb_cnt, save="hit_counts.png"):
    fig, axes = plt.subplots(3,2, figsize=(10,15))
    _show_hit_stats(tx_cnt,  "Transmission",                                    axes[0,0], axes[0,1])
    _show_hit_stats(cmb_cnt, "Combined",                                        axes[1,0], axes[1,1])
    _show_hit_stats(np.float32(np.bool_(cmb_cnt)), "Boolean of Combined",       axes[2,0], axes[2,1])
    plt.tight_layout(); plt.savefig(save, dpi=150); plt.close(fig)
    print(f"non hit counts for combined: {np.sum(cmb_cnt==0)}")
    print(f"non hit counts for transmission: {np.sum(tx_cnt==0)}")