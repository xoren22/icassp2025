# Floor plan generator with one-click randomization.
# - Produces a binary wall mask (1=wall, 0=no wall)
# - Includes straight and curved walls, realistic thickness variation, and corridors
# - A "New floor" button regenerates a different plan on each click
#
# Dependencies: numpy, matplotlib, ipywidgets (all standard in Jupyter). No internet access required.

import numpy as np
import matplotlib.pyplot as plt

# Try to import ipywidgets; fall back to a Matplotlib button if unavailable.
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    _HAS_WIDGETS = True
except Exception:
    from matplotlib.widgets import Button
    _HAS_WIDGETS = False

# Detect if we're running inside a Jupyter environment (with an active kernel).
def _in_jupyter():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and hasattr(ip, "kernel")
    except Exception:
        return False

# ---------- Geometry helpers (raster) ----------

def draw_thick_segment(mask, p0, p1, width_px):
    """Draw a thick line segment (inclusive) into a binary mask."""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    w = max(1.0, float(width_px))
    r = w / 2.0

    xmin = int(np.floor(min(x0, x1) - r - 2))
    xmax = int(np.ceil (max(x0, x1) + r + 2))
    ymin = int(np.floor(min(y0, y1) - r - 2))
    ymax = int(np.ceil (max(y0, y1) + r + 2))

    # clamp to mask bounds
    H, W = mask.shape
    xmin = max(0, xmin); ymin = max(0, ymin)
    xmax = min(W - 1, xmax); ymax = min(H - 1, ymax)
    if xmin > xmax or ymin > ymax:
        return

    xs = np.arange(xmin, xmax + 1)
    ys = np.arange(ymin, ymax + 1)
    gx, gy = np.meshgrid(xs, ys)

    dx = x1 - x0
    dy = y1 - y0
    L2 = dx * dx + dy * dy
    if L2 == 0:
        # draw a disk
        d2 = (gx - x0) ** 2 + (gy - y0) ** 2
        mask[ymin:ymax + 1, xmin:xmax + 1] |= (d2 <= r * r)
        return

    t = ((gx - x0) * dx + (gy - y0) * dy) / L2
    t = np.clip(t, 0.0, 1.0)
    cx = x0 + t * dx
    cy = y0 + t * dy
    d2 = (gx - cx) ** 2 + (gy - cy) ** 2
    mask[ymin:ymax + 1, xmin:xmax + 1] |= (d2 <= r * r)

def draw_polyline(mask, points, width_px, closed=False):
    if closed and len(points) >= 2:
        pts = points + [points[0]]
    else:
        pts = points
    for i in range(len(pts) - 1):
        draw_thick_segment(mask, pts[i], pts[i + 1], width_px)

def draw_rect_border(mask, x0, y0, x1, y1, width_px, rounded_r_px=0):
    """Draw a rectangle border (with optional rounded corners)."""
    x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
    if rounded_r_px <= 1:
        poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        draw_polyline(mask, poly, width_px, closed=True)
    else:
        r = float(rounded_r_px)
        r = min(r, (x1 - x0) / 2 - 1, (y1 - y0) / 2 - 1)
        # Build a rounded rectangle as poly with arc samples at corners
        def arc(cx, cy, start_ang, end_ang, rad, n=16):
            ts = np.linspace(start_ang, end_ang, n)
            return [(cx + rad * np.cos(t), cy + rad * np.sin(t)) for t in ts]

        pts = []
        # top-right corner (clockwise path)
        pts += arc(x1 - r, y0 + r, -np.pi/2, 0, r)
        # right edge to bottom-right
        pts += arc(x1 - r, y1 - r, 0, np.pi/2, r)
        # bottom-right to bottom-left
        pts += arc(x0 + r, y1 - r, np.pi/2, np.pi, r)
        # bottom-left to top-left
        pts += arc(x0 + r, y0 + r, np.pi, 3*np.pi/2, r)
        draw_polyline(mask, pts, width_px, closed=True)

def draw_quadratic_bezier(mask, p0, p1, pc, width_px, samples=40):
    """Quadratic Bezier curve with thick stroke."""
    t = np.linspace(0.0, 1.0, samples)
    x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * pc[0] + t ** 2 * p1[0]
    y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * pc[1] + t ** 2 * p1[1]
    pts = list(zip(x, y))
    draw_polyline(mask, pts, width_px, closed=False)

def dilate(binmask, r=1):
    """Binary dilation with a square structuring element of radius r."""
    if r <= 0:
        return binmask.copy()
    H, W = binmask.shape
    out = np.zeros_like(binmask, dtype=bool)
    # Sliding window via integral trick: use convolution by summing shifted versions
    for dy in range(-r, r + 1):
        ys = slice(max(0, dy), H + min(0, dy))
        yd = slice(max(0, -dy), H + min(0, -dy))
        for dx in range(-r, r + 1):
            xs = slice(max(0, dx), W + min(0, dx))
            xd = slice(max(0, -dx), W + min(0, -dx))
            out[yd, xd] |= binmask[ys, xs]
    return out

def erode(binmask, r=1):
    """Binary erosion with a square structuring element of radius r."""
    if r <= 0:
        return binmask.copy()
    H, W = binmask.shape
    out = np.ones_like(binmask, dtype=bool)
    for dy in range(-r, r + 1):
        ys = slice(max(0, dy), H + min(0, dy))
        yd = slice(max(0, -dy), H + min(0, -dy))
        for dx in range(-r, r + 1):
            xs = slice(max(0, dx), W + min(0, dx))
            xd = slice(max(0, -dx), W + min(0, -dx))
            out[yd, xd] &= binmask[ys, xs]
    return out

def closing(binmask, r=1):
    return erode(dilate(binmask, r), r)

# ---------- Generator ----------

def generate_floor_mask(
    width_m=200, height_m=100, px_per_m=4, seed=None,
    style="office_hospital_mix"
):
    """
    Returns (mask: HxW bool array).
    1 = wall, 0 = no wall.
    """
    rng = np.random.default_rng(seed)

    W = int(round(width_m * px_per_m))
    H = int(round(height_m * px_per_m))

    walls = np.zeros((H, W), dtype=bool)
    carve = np.zeros((H, W), dtype=bool)  # areas to clear (corridors)

    # ---- Parameters sampled with realistic ranges ----
    ext_th = rng.uniform(0.28, 0.45) * px_per_m  # exterior wall thickness (m -> px)
    part_th = rng.uniform(0.09, 0.14) * px_per_m # interior partitions
    core_th = rng.uniform(0.20, 0.30) * px_per_m # core walls
    curve_prob = 0.18
    diag_prob  = 0.06

    corridor_w = rng.uniform(1.9, 2.6) * px_per_m  # clear width
    corridor_wall_th = rng.uniform(0.10, 0.18) * px_per_m
    belt_offset = rng.uniform(7.0, 12.0) * px_per_m  # ring distance from exterior

    # grid module for BSP
    module_m = rng.choice([3.0, 3.6, 4.2, 4.8, 5.4])
    module_px = module_m * px_per_m
    min_room_w_m = rng.uniform(2.4, 2.9)
    min_room_w = min_room_w_m * px_per_m
    max_aspect = 3.5

    # ----- Outer footprint with optional rounded corners -----
    round_r = rng.uniform(0.0, 6.0) * px_per_m if rng.random() < 0.65 else 0.0
    draw_rect_border(walls, 2, 2, W - 3, H - 3, ext_th, rounded_r_px=round_r)

    # ----- Cores (stairs/elevators) -----
    n_cores = rng.integers(1, 4)
    core_rects = []
    for _ in range(n_cores):
        side_m = rng.uniform(6.5, 12.0)
        cw = int(side_m * px_per_m)
        ch = int((side_m * rng.uniform(0.85, 1.2)) * px_per_m)
        cx = rng.integers(W // 5, 4 * W // 5 - cw)
        cy = rng.integers(H // 5, 4 * H // 5 - ch)
        rect = (cx, cy, cx + cw, cy + ch)
        core_rects.append(rect)
        draw_rect_border(walls, *rect, core_th, rounded_r_px=int(rng.uniform(0, 2.0) * px_per_m))

    # ----- Corridor ring (belt) -----
    # Define inner rectangle for the ring centerline
    rx0 = int(2 + belt_offset)
    ry0 = int(2 + belt_offset)
    rx1 = int(W - 3 - belt_offset)
    ry1 = int(H - 3 - belt_offset)
    if rx1 - rx0 > 40 and ry1 - ry0 > 40:
        # Ring walls (draw thicker, then carve the corridor void)
        draw_rect_border(walls, rx0, ry0, rx1, ry1, corridor_w + 2 * corridor_wall_th, rounded_r_px=int(round_r * 0.6))
        draw_rect_border(carve, rx0, ry0, rx1, ry1, corridor_w, rounded_r_px=int(round_r * 0.6))

    # ----- Corridor spines -----
    n_spines = rng.integers(1, 4)
    for _ in range(n_spines):
        if rng.random() < 0.6:
            # vertical spine
            x = rng.integers(int(W * 0.25), int(W * 0.75))
            draw_thick_segment(walls, (x, 3), (x, H - 4), corridor_w + 2 * corridor_wall_th)
            draw_thick_segment(carve, (x, 3), (x, H - 4), corridor_w)
        else:
            # horizontal spine
            y = rng.integers(int(H * 0.25), int(H * 0.75))
            draw_thick_segment(walls, (3, y), (W - 4, y), corridor_w + 2 * corridor_wall_th)
            draw_thick_segment(carve, (3, y), (W - 4, y), corridor_w)

    # Connect cores to nearest spine/ring with short corridors
    def nearest_point_on_ring(cx, cy):
        # project to the ring rectangle edges
        if rx1 - rx0 <= 0 or ry1 - ry0 <= 0:
            return (cx, cy)
        # choose nearest edge center
        cands = [
            (cx, ry0), (cx, ry1), (rx0, cy), (rx1, cy)
        ]
        d2 = [(cx - px) ** 2 + (cy - py) ** 2 for (px, py) in cands]
        return cands[int(np.argmin(d2))]

    for (x0, y0, x1, y1) in core_rects:
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        tx, ty = nearest_point_on_ring(cx, cy)
        draw_thick_segment(walls, (cx, cy), (tx, ty), corridor_w + 2 * corridor_wall_th)
        draw_thick_segment(carve, (cx, cy), (tx, ty), corridor_w)

    # ----- Partitioning via stochastic BSP (axis-aligned) -----
    # We place splits; later corridors will carve through to create connectivity.
    # Start with one big rectangle (keep a small inset from exterior walls)
    inset = int(ext_th + px_per_m * 0.6)
    stack = [(inset, inset, W - inset - 1, H - inset - 1)]
    splits_drawn = 0
    max_splits = 180  # keep reasonable for speed

    def draw_split(rect, vertical, s):
        x0, y0, x1, y1 = rect
        if vertical:
            p0, p1 = (s, y0), (s, y1)
        else:
            p0, p1 = (x0, s), (x1, s)
        draw_thick_segment(walls, p0, p1, part_th)

    while stack and splits_drawn < max_splits:
        # pick the largest rect
        areas = [ (r[2]-r[0])*(r[3]-r[1]) for r in stack ]
        idx = int(np.argmax(areas))
        x0, y0, x1, y1 = stack.pop(idx)
        w = x1 - x0
        h = y1 - y0
        if w < 2 * min_room_w or h < 2 * min_room_w:
            continue

        # decide orientation by aspect + randomness
        if (w > h * 1.1 and rng.random() < 0.8) or (w > h and rng.random() < 0.6):
            vertical = True
        elif (h > w * 1.1 and rng.random() < 0.8) or (h > w and rng.random() < 0.6):
            vertical = False
        else:
            vertical = rng.random() < 0.5

        if vertical:
            smin = x0 + int(min_room_w)
            smax = x1 - int(min_room_w)
            if smax - smin < module_px * 0.5:
                continue
            # snap to module with jitter
            s = int(rng.normal(rng.integers(smin, smax), module_px * 0.12))
            s = int(np.clip(s, smin, smax))
            if rng.random() < curve_prob:
                # Draw a subtle curved wall instead of straight
                p0 = (s, y0)
                p1 = (s, y1)
                ctrl = (s + rng.normal(0, module_px * 0.5), (y0 + y1) / 2 + rng.normal(0, module_px * 0.6))
                draw_quadratic_bezier(walls, p0, p1, ctrl, part_th, samples=48)
            elif rng.random() < diag_prob:
                # 45-ish diagonal split
                dy = rng.integers(-int(h*0.25), int(h*0.25))
                draw_thick_segment(walls, (s, y0), (s + dy, y1), part_th)
            else:
                draw_split((x0, y0, x1, y1), True, s)
            # push children
            stack.append((x0, y0, s, y1))
            stack.append((s, y0, x1, y1))
        else:
            smin = y0 + int(min_room_w)
            smax = y1 - int(min_room_w)
            if smax - smin < module_px * 0.5:
                continue
            s = int(rng.normal(rng.integers(smin, smax), module_px * 0.12))
            s = int(np.clip(s, smin, smax))
            if rng.random() < curve_prob:
                p0 = (x0, s)
                p1 = (x1, s)
                ctrl = ((x0 + x1) / 2 + rng.normal(0, module_px * 0.6), s + rng.normal(0, module_px * 0.5))
                draw_quadratic_bezier(walls, p0, p1, ctrl, part_th, samples=48)
            elif rng.random() < diag_prob:
                dx = rng.integers(-int(w*0.25), int(w*0.25))
                draw_thick_segment(walls, (x0, s), (x1, s + dx), part_th)
            else:
                draw_split((x0, y0, x1, y1), False, s)
            stack.append((x0, y0, x1, s))
            stack.append((x0, s, x1, y1))

        splits_drawn += 1

    # ----- Add a rotated "secondary frame" (mixture-of-Manhattan) -----
    if rng.random() < 0.45:
        theta = np.deg2rad(rng.uniform(10, 35))
        ct, st = np.cos(theta), np.sin(theta)

        # a few long oblique walls
        n_oblique = rng.integers(3, 7)
        for _ in range(n_oblique):
            # pick a line by sampling intercept along one bounding axis
            d = rng.uniform(-W*0.2, W*1.2)  # extended range to ensure intersection
            # parametric form: x*ct + y*st = d
            # intersect with rectangle bounds to find segment endpoints
            # test intersections with the 4 borders; collect those inside
            candidates = []
            # y=0..H-1
            for y in (inset, H - inset):
                x = (d - y * st) / ct
                if inset <= x <= W - inset:
                    candidates.append((x, y))
            for x in (inset, W - inset):
                y = (d - x * ct) / st if abs(st) > 1e-6 else None
                if y is not None and inset <= y <= H - inset:
                    candidates.append((x, y))
            if len(candidates) >= 2:
                p0, p1 = candidates[0], candidates[1]
                # Some curves along the oblique frame
                if rng.random() < 0.4:
                    mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
                    # bend normal to the line
                    nx, ny = -st, ct
                    ctrl = (mid[0] + nx * rng.uniform(20, 60), mid[1] + ny * rng.uniform(20, 60))
                    draw_quadratic_bezier(walls, p0, p1, ctrl, part_th, samples=64)
                else:
                    draw_thick_segment(walls, p0, p1, part_th)

    # ----- Apply corridor carving (clear the inside of corridors) -----
    walls = walls & (~carve)

    # ----- Clean-up: closing to fix tiny gaps; trim singletons -----
    walls = closing(walls, r=1)
    # remove tiny isolated pixels (degree < 2 in 3x3 neighborhood)
    H, W = walls.shape
    neighbors = np.zeros_like(walls, dtype=int)
    for dy in (-1, 0, 1):
        ys = slice(max(0, dy), H + min(0, dy))
        yd = slice(max(0, -dy), H + min(0, -dy))
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0: 
                continue
            xs = slice(max(0, dx), W + min(0, dx))
            xd = slice(max(0, -dx), W + min(0, -dx))
            neighbors[yd, xd] += walls[ys, xs]
    walls[(walls == 1) & (neighbors < 2)] = 0

    return walls

# ---------- UI ----------

def show_generator():
    width_m, height_m, ppm = 200, 100, 4  # 0.25 m/px -> 800x400
    # Only use ipywidgets when we're actually in a Jupyter notebook
    if _HAS_WIDGETS and _in_jupyter():
        btn = widgets.Button(description="New floor", button_style="primary", tooltip="Generate a new random wall mask")
        out = widgets.Output()

        def on_click(_):
            with out:
                out.clear_output(wait=True)
                mask = generate_floor_mask(width_m, height_m, ppm)
                fig = plt.figure(figsize=(10, 5))
                ax = plt.gca()
                ax.imshow(mask, cmap="gray_r", interpolation="nearest")
                ax.set_title("Wall mask (1=wall)")
                ax.set_axis_off()
                plt.show()

        btn.on_click(on_click)

        display(btn, out)
        # initial render
        on_click(None)
    else:
        # Fallback: Matplotlib button inside the figure (less reliable in some Jupyter setups)
        # Ensure Button is available even if ipywidgets import succeeded.
        try:
            from matplotlib.widgets import Button  # type: ignore
        except Exception:
            pass
        mask = generate_floor_mask(width_m, height_m, ppm)
        fig, ax = plt.subplots(figsize=(10,5))
        # Place controls at the top so they're not obscured by the toolbar
        plt.subplots_adjust(top=0.88, bottom=0.06)
        im = ax.imshow(mask, cmap="gray_r", interpolation="nearest")
        ax.set_axis_off()
        ax.set_title("Wall mask (1=wall)")

        # Button at the top center
        ax_btn = plt.axes([0.40, 0.91, 0.20, 0.06])
        button = Button(ax_btn, "New floor")

        def regenerate(_=None):
            im.set_data(generate_floor_mask(width_m, height_m, ppm))
            fig.canvas.draw_idle()

        button.on_clicked(regenerate)
        # Convenience: keyboard/mouse shortcuts to regenerate
        fig.canvas.mpl_connect("key_press_event", lambda e: regenerate() if e.key in ("r", " ", "enter") else None)
        fig.canvas.mpl_connect("button_press_event", lambda e: regenerate() if e.button == 1 else None)
        plt.show()

show_generator()
