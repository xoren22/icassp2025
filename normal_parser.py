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


@njit(cache=True, inline='always')
def _pca_angle_trimmed(xs, ys):
    """
    Robust PCA angle using trimmed fitting.
    1. Initial PCA fit
    2. Remove worst 20% of points (furthest from line)
    3. Refit PCA on remaining 80%
    """
    n = xs.size
    if n < 3:
        return _pca_angle(xs, ys)  # Too few points for trimming
    
    # Step 1: Initial PCA fit
    initial_angle = _pca_angle(xs, ys)
    
    # Step 2: Compute distances to initial line
    # Line equation: y = mx + b where m = tan(angle)
    angle_rad = math.radians(initial_angle)
    
    # For vertical lines (angle near 90°), use x = c form
    if abs(initial_angle - 90.0) < 1.0:
        # Vertical line: x = constant
        line_x = xs.mean()
        distances = np.abs(xs - line_x)
    else:
        # Non-vertical line: y = mx + b
        m = math.tan(angle_rad)
        # Find b using centroid: y_mean = m * x_mean + b
        b = ys.mean() - m * xs.mean()
        
        # Distance from point (x,y) to line mx - y + b = 0
        # is |mx - y + b| / sqrt(m² + 1)
        norm_factor = math.sqrt(m * m + 1.0)
        distances = np.abs(m * xs - ys + b) / norm_factor
    
    # Step 3: Keep best 80% (smallest distances)
    keep_count = max(3, int(0.8 * n))  # Keep at least 3 points
    if keep_count >= n:
        return initial_angle  # No trimming needed
    
    # Find indices of points with smallest distances
    indices = np.argsort(distances)[:keep_count]
    
    # Step 4: Refit PCA on trimmed points
    xs_trimmed = xs[indices]
    ys_trimmed = ys[indices]
    
    return _pca_angle(xs_trimmed, ys_trimmed)


# distance-weighted PCA centered at source pixel
@njit(cache=True, inline='always')
def _pca_angle_weighted_centered(xs, ys, px, py, distance_power=1.0):
    n = xs.size
    if n < 3:
        return _pca_angle(xs, ys)

    dx = xs - px
    dy = ys - py

    distances = np.sqrt(dx * dx + dy * dy)
    weights = np.empty(n, dtype=np.float32)
    for i in range(n):
        d = distances[i]
        if d < 1e-6:
            weights[i] = 10.0
        else:
            weights[i] = 1.0 / (d ** distance_power)

    wsum = weights.sum()
    sxx = 0.0
    syy = 0.0
    sxy = 0.0
    for i in range(n):
        w = weights[i] / wsum
        sxx += w * dx[i] * dx[i]
        syy += w * dy[i] * dy[i]
        sxy += w * dx[i] * dy[i]

    ang = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    return math.degrees(ang) % 180.0


@njit(cache=True)
def compute_wall_angle_pca(img, px, py, win=5):
    h, w = img.shape
    y0, y1 = max(py - win, 0), min(py + win + 1, h)
    x0, x1 = max(px - win, 0), min(px + win + 1, w)

    patch = img[y0:y1, x0:x1]
    ys, xs = np.nonzero(patch)
    n = xs.size
    if n < 3:
        return -1.0

    xs = xs.astype(np.float32) + x0
    ys = ys.astype(np.float32) + y0

    # First line using weighted PCA centred at (px,py)
    angle1 = _pca_angle_weighted_centered(xs, ys, px, py, 1.0)

    rad1 = math.radians(angle1)
    c1 = math.cos(rad1)
    s1 = math.sin(rad1)

    # Split points into two sets based on distance to line1
    dt = 1.0
    xs1 = np.empty(n, dtype=np.float32)
    ys1 = np.empty(n, dtype=np.float32)
    xs2 = np.empty(n, dtype=np.float32)
    ys2 = np.empty(n, dtype=np.float32)
    c1_idx = 0
    c2_idx = 0

    for i in range(n):
        dx = xs[i] - px
        dy = ys[i] - py
        dist = abs(-s1 * dx + c1 * dy)
        if dist <= dt:
            xs1[c1_idx] = xs[i]
            ys1[c1_idx] = ys[i]
            c1_idx += 1
        else:
            xs2[c2_idx] = xs[i]
            ys2[c2_idx] = ys[i]
            c2_idx += 1

    if c2_idx < 3:
        # Only one wall present
        return angle1

    # Compute second line on remaining points
    xs2 = xs2[:c2_idx]
    ys2 = ys2[:c2_idx]
    angle2 = _pca_angle(xs2, ys2)

    # Angle difference
    diff = abs(((angle1 - angle2 + 90.0) % 180.0) - 90.0)

    merge_th = 20.0
    if diff < merge_th:
        # Treat as same wall
        return angle1

    # Keep only cluster that contains source pixel (cluster1)
    if c1_idx < 3:
        return angle1

    xs1 = xs1[:c1_idx]
    ys1 = ys1[:c1_idx]
    final_angle = _pca_angle_weighted_centered(xs1, ys1, px, py, 1.0)
    return final_angle


# debug
def _debug_plot_window(building_mask, px, py, win=5, angle1=None, angle2=None, save_path=None):
    import matplotlib.pyplot as plt
    y0, y1 = max(py - win, 0), min(py + win + 1, building_mask.shape[0])
    x0, x1 = max(px - win, 0), min(px + win + 1, building_mask.shape[1])
    patch = building_mask[y0:y1, x0:x1]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(patch, cmap='gray', origin='upper')
    ax.plot(px - x0, py - y0, 'ro')

    if angle1 is not None:
        rad = math.radians(angle1)
        dx = math.cos(rad)
        dy = math.sin(rad)
        ax.quiver(px - x0, py - y0, dx, dy, color='cyan', scale_units='xy', scale=1, width=0.005)
    if angle2 is not None:
        rad = math.radians(angle2)
        dx = math.cos(rad)
        dy = math.sin(rad)
        ax.quiver(px - x0, py - y0, dx, dy, color='yellow', scale_units='xy', scale=1, width=0.005)
    ax.set_title(f"({px},{py})")
    ax.set_axis_off()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


# simple python helper (non-numba) to test if two-line split exists
def _detect_two_lines(building_mask, px, py, win=5, dist_th=1.0, merge_th=20.0):
    h, w = building_mask.shape
    y0, y1 = max(py - win, 0), min(py + win + 1, h)
    x0, x1 = max(px - win, 0), min(px + win + 1, w)
    patch = building_mask[y0:y1, x0:x1]
    ys, xs = np.nonzero(patch)
    n = xs.size
    if n < 6:
        return None
    xs = xs.astype(np.float32) + x0
    ys = ys.astype(np.float32) + y0

    angle1 = _pca_angle_weighted_centered(xs, ys, px, py, 1.0)
    rad1 = math.radians(angle1)
    c1 = math.cos(rad1)
    s1 = math.sin(rad1)

    xs2 = []
    ys2 = []
    for i in range(n):
        dx = xs[i] - px
        dy = ys[i] - py
        dist = abs(-s1 * dx + c1 * dy)
        if dist > dist_th:
            xs2.append(xs[i])
            ys2.append(ys[i])

    if len(xs2) < 3:
        return None

    xs2 = np.array(xs2, dtype=np.float32)
    ys2 = np.array(ys2, dtype=np.float32)
    angle2 = _pca_angle(xs2, ys2)
    diff = abs(((angle1 - angle2 + 90.0) % 180.0) - 90.0)
    if diff < merge_th:
        return None
    return angle1, angle2


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
    window_sizes = np.array([11, 12])
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
    nx_raw = np.cos(normal_angles_rad)
    ny_raw = np.sin(normal_angles_rad)
    
    # Find center of building floor (centroid of empty space)
    empty_y, empty_x = np.where(building_mask == 0)
    if len(empty_x) > 0:
        center_x = empty_x.mean()
        center_y = empty_y.mean()
    else:
        # Fallback if no empty space found
        h, w = building_mask.shape
        center_x, center_y = w/2, h/2
    
    # For each arrow, pick direction pointing toward building center
    sampled_nx = np.zeros_like(nx_raw)
    sampled_ny = np.zeros_like(ny_raw)
    
    for i in range(len(sampled_x)):
        # Vector from wall pixel to building center
        to_center_x = center_x - sampled_x[i]
        to_center_y = center_y - sampled_y[i]
        
        # Test both normal directions
        normal_1 = (nx_raw[i], ny_raw[i])
        normal_2 = (-nx_raw[i], -ny_raw[i])
        
        # Pick direction with positive dot product toward building center
        dot_1 = normal_1[0] * to_center_x + normal_1[1] * to_center_y
        dot_2 = normal_2[0] * to_center_x + normal_2[1] * to_center_y
        
        if dot_1 > dot_2:
            sampled_nx[i] = normal_1[0]
            sampled_ny[i] = normal_1[1]
        else:
            sampled_nx[i] = normal_2[0]
            sampled_ny[i] = normal_2[1]

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

    # Draw the building center point
    ax.plot(center_x, center_y, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2, label='Building Center')
    
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

    print(f"Computing split-line multi-scale PCA wall angles for Building {building_id}...")
    
    # Multi-scale PCA method
    wall_angles = precompute_wall_angles_pca(building_mask)

    # Visualize results
    visualize_wall_angles(
        building_mask, wall_angles,
        title=f"Trimmed Multi-Scale PCA Wall Angles (Building {building_id})",
        sample_prob=0.5,
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
    print(f"Split-line multi-scale PCA coverage: {valid_angles}/{total_walls} ({valid_angles/total_walls*100:.1f}%)")
    
    if valid_angles > 0:
        valid_wall_angles = wall_angles_at_walls[wall_angles_at_walls >= 0]
        print(f"Angle range: {valid_wall_angles.min():.1f}° to {valid_wall_angles.max():.1f}°")
    
    print(f"\nSaved: wall_angles_multiscale.png") 

    # try to show up to 6 double-line windows
    wy, wx = np.where(building_mask > 0)
    shown = 0
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(wx))
    for idx in perm:
        px = int(wx[idx])
        py = int(wy[idx])
        res = _detect_two_lines(building_mask, px, py, 5)
        if res is not None:
            a1, a2 = res
            _debug_plot_window(building_mask, px, py, 5, angle1=a1, angle2=a2)
            shown += 1
            if shown >= 6:
                break 