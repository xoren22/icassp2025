from numba import njit, prange
import numpy as np
import matplotlib.pyplot as plt
import math

from helper import load_samples


@njit(cache=True, inline='always')
def _pca_angle(xs, ys):
    """
    Return principal-axis angle (deg ∈ [0,180)) of the 2-D point cloud.
    Closed-form 2×2 eigen decomposition:
        θ = ½·atan2(2·cov_xy, cov_xx – cov_yy)
    """
    n = xs.size
    x_mean = xs.mean()
    y_mean = ys.mean()

    sxx = 0.0
    syy = 0.0
    sxy = 0.0
    for i in range(n):
        dx = xs[i] - x_mean
        dy = ys[i] - y_mean
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    sxx /= n
    syy /= n
    sxy /= n

    angle = 0.5 * math.atan2(2.0 * sxy, sxx - syy)   # radians
    angle_deg = (math.degrees(angle)) % 180.0
    return angle_deg


@njit(cache=True)
def compute_wall_angle_pca(img, px, py, win=5):
    """
    PCA-based wall angle estimator for a single window size.
    
    Returns: wall angle in degrees (0-180°), or -1 if insufficient data
    """
    h, w = img.shape
    y0, y1 = max(py - win, 0), min(py + win + 1, h)
    x0, x1 = max(px - win, 0), min(px + win + 1, w)
    patch = img[y0:y1, x0:x1]

    ys, xs = np.nonzero(patch)          # local wall pixels
    if xs.size < 3:                     # too sparse
        return -1.0
    else:
        # Convert to global coordinates
        xs = xs.astype(np.float32) + x0
        ys = ys.astype(np.float32) + y0
        
        # Get wall direction from PCA
        wall_angle_deg = _pca_angle(xs, ys)
        return wall_angle_deg


@njit(cache=True)
def compute_wall_angle_multiscale_pca(img, px, py):
    """
    Multi-scale PCA-based wall angle estimator.
    
    Tries multiple window sizes and picks the most consistent result.
    This helps avoid corner contamination issues.
    
    Returns: wall angle in degrees (0-180°)
    """
    h, w = img.shape
    
    # Try multiple window sizes (small to large)
    window_sizes = np.array([2, 3, 4, 5, 6])
    angles = np.zeros(len(window_sizes), dtype=np.float32)
    valid_count = 0
    
    # Compute angle for each window size
    for i, win_size in enumerate(window_sizes):
        angle = compute_wall_angle_pca(img, px, py, win_size)
        if angle >= 0:  # Valid result (note: 0° is valid horizontal wall!)
            angles[valid_count] = angle
            valid_count += 1
    
    if valid_count == 0:
        # Fallback: find nearest empty pixel
        for radius in range(1, min(h, w)):
            found = False
            best_dx, best_dy = 0, 0
            min_dist_sq = float('inf')
            
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    ny_check, nx_check = py + dy, px + dx
                    if (0 <= ny_check < h and 0 <= nx_check < w and 
                        img[ny_check, nx_check] == 0):
                        dist_sq = dx*dx + dy*dy
                        if dist_sq < min_dist_sq:
                            min_dist_sq = dist_sq
                            best_dx, best_dy = dx, dy
                            found = True
            
            if found:
                normal_angle = math.degrees(math.atan2(best_dy, best_dx))
                wall_angle = (normal_angle + 90) % 180
                return wall_angle
        
        return 90.0  # Default horizontal wall
    
    else:
        # Use smallest window that works (most local, least contaminated)
        return angles[0]


def precompute_wall_angles_pca(building_mask: np.ndarray) -> np.ndarray:
    """
    Computes wall angles for ALL non-zero pixels using multi-scale PCA method.
    
    This method tries multiple window sizes and picks the most consistent result
    to be robust against corners and mixed orientations.
    Returns wall angles (0-180°) which are side-agnostic for reflection calculations.
    
    Parameters
    ----------
    building_mask : np.ndarray
        A 2D array where non-zero values indicate wall locations.
    
    Returns
    -------
    np.ndarray
        Array with wall angles (0-180°) for every wall pixel.
    """
    h, w = building_mask.shape
    
    # Initialize output array
    angles_img = np.zeros((h, w), dtype=np.float32)
    
    # Process every wall pixel
    for py in range(h):
        for px in range(w):
            if building_mask[py, px] > 0:
                angle = compute_wall_angle_multiscale_pca(building_mask, px, py)
                angles_img[py, px] = angle
    
    return angles_img





def visualize_wall_angles(
    building_mask: np.ndarray,
    angles_img: np.ndarray,
    title: str = "Wall Angles Visualization",
    sample_prob: float = 0.1,
    save_path: str = None
):
    """
    Visualizes wall angles by drawing normal arrows for randomly sampled wall pixels.
    """
    # Get all wall pixel coordinates
    wall_y, wall_x = np.where(building_mask > 0)
    
    # Randomly sample wall pixels with given probability
    num_pixels = len(wall_x)
    keep_mask = np.random.random(num_pixels) < sample_prob
    
    sampled_x = wall_x[keep_mask]
    sampled_y = wall_y[keep_mask]
    sampled_angles = angles_img[sampled_y, sampled_x]
    
    # Convert wall angles to normal vectors for visualization
    # Normal is perpendicular to wall (wall_angle + 90°)
    normal_angles_rad = np.radians(sampled_angles + 90)
    sampled_nx = np.cos(normal_angles_rad)
    sampled_ny = np.sin(normal_angles_rad)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(building_mask, cmap='gray', origin='upper')

    # Dynamically scale arrows to be proportional to the building size
    h, w = building_mask.shape
    # Set arrow length to be ~2.5% of the image's largest dimension.
    # A smaller scale value makes arrows longer.
    dynamic_scale = 1 / (0.025 * max(h, w))

    # Use quiver to draw all arrows at once
    ax.quiver(
        sampled_x, sampled_y, sampled_nx, sampled_ny,
        color='cyan',
        angles='xy',
        scale_units='xy',
        scale=dynamic_scale,
        width=0.002,
        headwidth=4,
    )

    ax.set_title(title)
    ax.set_aspect('equal')
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        print(f"Saved visualization to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Load a single sample
    sample = load_samples(num_samples=1, seed=1)[0]
    building_id = sample.ids[0]

    # Create the binary layout mask from reflectance and transmittance channels
    refl, trans, _ = sample.input_img.cpu().numpy()
    building_mask = (refl + trans > 0).astype(np.uint8)

    print(f"Computing multi-scale PCA wall angles for Building {building_id}...")
    
    # Multi-scale PCA method
    wall_angles = precompute_wall_angles_pca(building_mask)

    # Visualize results
    visualize_wall_angles(
        building_mask, wall_angles,
        title=f"Multi-Scale PCA Wall Angles (Building {building_id})",
        sample_prob=0.9,
        save_path="wall_angles_multiscale.png"
    )
    
    # Print stats (fix coverage calculation - 0° is valid!)
    total_walls = np.sum(building_mask)
    wall_pixels_mask = building_mask > 0
    wall_angles_at_walls = wall_angles[wall_pixels_mask]
    
    # Count angles < 0 as invalid (since valid range is [0, 180))
    invalid_angles = np.sum(wall_angles_at_walls < 0)
    valid_angles = total_walls - invalid_angles
    
    print(f"\nResults:")
    print(f"Total wall pixels: {total_walls}")
    print(f"Multi-scale PCA coverage: {valid_angles}/{total_walls} ({valid_angles/total_walls*100:.1f}%)")
    
    if valid_angles > 0:
        valid_wall_angles = wall_angles_at_walls[wall_angles_at_walls >= 0]
        print(f"Angle range: {valid_wall_angles.min():.1f}° to {valid_wall_angles.max():.1f}°")
    
    print(f"\nSaved: wall_angles_multiscale.png") 