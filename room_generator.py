# Upgraded floor plan generator:
# - Adds doors with variable widths conditioned on local wall length
# - Uses heavy-tailed splitting to diversify room sizes (many small, occasional very large)
# - Enforces realistic minimum room width (rare very small rooms)
# - Records full metadata for reproducibility (strokes + carve ops + doors) to JSON
# - Computes per-pixel wall normals (H x W x 2 float32)
# - UI: one "New floor" button regenerates; the current plan is saved to /mnt/data/*

import json
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    _HAS_WIDGETS = True
except Exception:
    _HAS_WIDGETS = False

# ---------------- Raster canvas with normals ----------------

@dataclass
class RasterCanvas:
    H: int
    W: int
    wall: np.ndarray = field(init=False)
    nx: np.ndarray = field(init=False)   # normal x
    ny: np.ndarray = field(init=False)   # normal y
    d: np.ndarray  = field(init=False)   # distance to stroke centerline
    corridor_mask: np.ndarray = field(init=False)

    def __post_init__(self):
        self.wall = np.zeros((self.H, self.W), dtype=bool)
        self.nx   = np.zeros((self.H, self.W), dtype=np.float32)
        self.ny   = np.zeros((self.H, self.W), dtype=np.float32)
        self.d    = np.full((self.H, self.W), np.inf, dtype=np.float32)
        self.corridor_mask = np.zeros((self.H, self.W), dtype=bool)

    # --- Low-level paint/clear ---
    def _paint_segment(self, p0, p1, width_px, set_wall=True, mark_corridor=False):
        """Draw a thick segment. Updates normals (pointing outward from centerline).
        If set_wall=False -> clear region (used for carving corridors or doors)."""
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        w = max(1.0, float(width_px))
        r = w / 2.0

        xmin = int(np.floor(min(x0, x1) - r - 2))
        xmax = int(np.ceil (max(x0, x1) + r + 2))
        ymin = int(np.floor(min(y0, y1) - r - 2))
        ymax = int(np.ceil (max(y0, y1) + r + 2))

        xmin = max(0, xmin); ymin = max(0, ymin)
        xmax = min(self.W - 1, xmax); ymax = min(self.H - 1, ymax)
        if xmin > xmax or ymin > ymax:
            return

        xs = np.arange(xmin, xmax + 1)
        ys = np.arange(ymin, ymax + 1)
        gx, gy = np.meshgrid(xs, ys)

        dx = x1 - x0
        dy = y1 - y0
        L2 = dx * dx + dy * dy
        if L2 == 0:
            cx = np.full_like(gx, x0, dtype=np.float32)
            cy = np.full_like(gy, y0, dtype=np.float32)
        else:
            t = ((gx - x0) * dx + (gy - y0) * dy) / L2
            t = np.clip(t, 0.0, 1.0)
            cx = x0 + t * dx
            cy = y0 + t * dy

        vx = gx - cx
        vy = gy - cy
        d2 = vx * vx + vy * vy
        inside = d2 <= (r * r)

        if set_wall:
            # Only update normals where this stroke is closer than previous
            closer = inside & (np.sqrt(d2, dtype=np.float32) < self.d[ymin:ymax+1, xmin:xmax+1])
            if np.any(closer):
                self.wall[ymin:ymax+1, xmin:xmax+1][inside] = True
                self.d[ymin:ymax+1, xmin:xmax+1][closer] = np.sqrt(d2[closer]).astype(np.float32)

                # Normalize vectors; handle zero-length safely
                nrm = np.sqrt(vx[closer] * vx[closer] + vy[closer] * vy[closer]).astype(np.float32)
                nz = nrm > 1e-6
                nx_new = np.zeros_like(nrm, dtype=np.float32)
                ny_new = np.zeros_like(nrm, dtype=np.float32)
                nx_new[nz] = (vx[closer][nz] / nrm[nz]).astype(np.float32)
                ny_new[nz] = (vy[closer][nz] / nrm[nz]).astype(np.float32)
                self.nx[ymin:ymax+1, xmin:xmax+1][closer] = nx_new
                self.ny[ymin:ymax+1, xmin:xmax+1][closer] = ny_new
        else:
            # Clear region: remove wall and normals
            if np.any(inside):
                sub = self.wall[ymin:ymax+1, xmin:xmax+1]
                will_clear = inside & sub
                self.wall[ymin:ymax+1, xmin:xmax+1][will_clear] = False
                self.nx[ymin:ymax+1, xmin:xmax+1][will_clear] = 0.0
                self.ny[ymin:ymax+1, xmin:xmax+1][will_clear] = 0.0
                self.d[ymin:ymax+1, xmin:xmax+1][will_clear]  = np.inf
        if mark_corridor and np.any(inside):
            self.corridor_mask[ymin:ymax+1, xmin:xmax+1] |= inside

    def paint_polyline(self, pts, width_px, closed=False, mark_corridor=False):
        if closed and len(pts) >= 2:
            seq = pts + [pts[0]]
        else:
            seq = pts
        for i in range(len(seq) - 1):
            self._paint_segment(seq[i], seq[i+1], width_px, set_wall=True, mark_corridor=mark_corridor)

    def carve_polyline(self, pts, width_px, closed=False, mark_corridor=False):
        if closed and len(pts) >= 2:
            seq = pts + [pts[0]]
        else:
            seq = pts
        for i in range(len(seq) - 1):
            self._paint_segment(seq[i], seq[i+1], width_px, set_wall=False, mark_corridor=mark_corridor)

    def paint_bezier_quadratic(self, p0, p1, pc, width_px, samples=64, mark_corridor=False):
        t = np.linspace(0.0, 1.0, samples)
        x = (1 - t)**2 * p0[0] + 2*(1 - t)*t * pc[0] + t**2 * p1[0]
        y = (1 - t)**2 * p0[1] + 2*(1 - t)*t * pc[1] + t**2 * p1[1]
        pts = list(zip(x, y))
        self.paint_polyline(pts, width_px, closed=False, mark_corridor=mark_corridor)

    def carve_bezier_quadratic(self, p0, p1, pc, width_px, samples=64, mark_corridor=False):
        t = np.linspace(0.0, 1.0, samples)
        x = (1 - t)**2 * p0[0] + 2*(1 - t)*t * pc[0] + t**2 * p1[0]
        y = (1 - t)**2 * p0[1] + 2*(1 - t)*t * pc[1] + t**2 * p1[1]
        pts = list(zip(x, y))
        self.carve_polyline(pts, width_px, closed=False, mark_corridor=mark_corridor)

    def paint_rect_border(self, x0, y0, x1, y1, width_px, rounded_r_px=0, mark_corridor=False):
        x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
        if rounded_r_px <= 1:
            poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            self.paint_polyline(poly, width_px, closed=True, mark_corridor=mark_corridor)
        else:
            r = float(rounded_r_px)
            r = max(1.0, min(r, (x1 - x0) / 2 - 1, (y1 - y0) / 2 - 1))

            def arc(cx, cy, start_ang, end_ang, rad, n=20):
                ts = np.linspace(start_ang, end_ang, n)
                return [(cx + rad*np.cos(t), cy + rad*np.sin(t)) for t in ts]

            pts = []
            pts += arc(x1 - r, y0 + r, -np.pi/2, 0,           r)
            pts += arc(x1 - r, y1 - r, 0,           np.pi/2,   r)
            pts += arc(x0 + r, y1 - r, np.pi/2,     np.pi,     r)
            pts += arc(x0 + r, y0 + r, np.pi,       3*np.pi/2, r)
            self.paint_polyline(pts, width_px, closed=True, mark_corridor=mark_corridor)

# ---------------- Utilities ----------------

def closing(binmask, r=1):
    if r <= 0:
        return binmask.copy()
    H, W = binmask.shape
    # dilation
    out = np.zeros_like(binmask, dtype=bool)
    for dy in range(-r, r + 1):
        ys = slice(max(0, dy), H + min(0, dy))
        yd = slice(max(0, -dy), H + min(0, -dy))
        for dx in range(-r, r + 1):
            xs = slice(max(0, dx), W + min(0, dx))
            xd = slice(max(0, -dx), W + min(0, -dx))
            out[yd, xd] |= binmask[ys, xs]
    dil = out
    # erosion
    out2 = np.ones_like(binmask, dtype=bool)
    for dy in range(-r, r + 1):
        ys = slice(max(0, dy), H + min(0, dy))
        yd = slice(max(0, -dy), H + min(0, -dy))
        for dx in range(-r, r + 1):
            xs = slice(max(0, dx), W + min(0, dx))
            xd = slice(max(0, -dx), W + min(0, -dx))
            out2[yd, xd] &= dil[ys, xs]
    return out2

def connected_components(binmask):
    """4-connected components labeling, returns labels (int32) and sizes dict."""
    H, W = binmask.shape
    labels = np.full((H, W), -1, dtype=np.int32)
    comp_id = 0
    sizes = {}
    for y in range(H):
        for x in range(W):
            if binmask[y, x] and labels[y, x] == -1:
                # BFS
                qx = [x]; qy = [y]
                labels[y, x] = comp_id
                size = 1
                head = 0
                while head < len(qx):
                    cx, cy = qx[head], qy[head]
                    head += 1
                    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx = cx + dx; ny = cy + dy
                        if 0 <= nx < W and 0 <= ny < H:
                            if binmask[ny, nx] and labels[ny, nx] == -1:
                                labels[ny, nx] = comp_id
                                qx.append(nx); qy.append(ny)
                                size += 1
                sizes[comp_id] = size
                comp_id += 1
    return labels, sizes

def line_length(p0, p1):
    return float(math.hypot(p1[0]-p0[0], p1[1]-p0[1]))

# Parametric helpers for quadratic Bezier
def bezier_point(p0, p1, pc, t):
    x = (1 - t)**2 * p0[0] + 2*(1 - t)*t * pc[0] + t**2 * p1[0]
    y = (1 - t)**2 * p0[1] + 2*(1 - t)*t * pc[1] + t**2 * p1[1]
    return (x, y)

def bezier_tangent(p0, p1, pc, t):
    # derivative of quadratic Bezier
    dx = 2*(1 - t)*(pc[0] - p0[0]) + 2*t*(p1[0] - pc[0])
    dy = 2*(1 - t)*(pc[1] - p0[1]) + 2*t*(p1[1] - pc[1])
    return (dx, dy)

def unit(vx, vy):
    n = math.hypot(vx, vy)
    if n < 1e-6: 
        return (0.0, 0.0)
    return (vx / n, vy / n)

# ---------------- Generator + metadata ----------------

def generate_floor_scene(width_m=200, height_m=100, px_per_m=4, seed=None):
    # Seed
    if seed is None:
        seed = int(np.random.SeedSequence().entropy)
    rng = np.random.default_rng(seed)

    W = int(round(width_m * px_per_m))
    H = int(round(height_m * px_per_m))

    canvas = RasterCanvas(H, W)
    ops: List[Dict[str, Any]] = []
    strokes: List[Dict[str, Any]] = []
    doors: List[Dict[str, Any]] = []

    # ---- Parameters ----
    params = {}
    params['px_per_m'] = float(px_per_m)

    ext_th      = float(rng.uniform(0.28, 0.45) * px_per_m)
    part_th     = float(rng.uniform(0.09, 0.14) * px_per_m)
    core_th     = float(rng.uniform(0.20, 0.30) * px_per_m)
    curve_prob  = float(0.20)
    diag_prob   = float(0.08)
    big_room_mode = bool(rng.random() < 0.35)

    corridor_w        = float(rng.uniform(1.9, 2.7) * px_per_m)  # clear
    corridor_wall_th  = float(rng.uniform(0.10, 0.18) * px_per_m)
    belt_offset       = float(rng.uniform(7.0, 12.0) * px_per_m)

    module_m = float(rng.choice([3.0, 3.6, 4.2, 4.8, 5.4]))
    module_px = module_m * px_per_m
    # Minimum room width (rarely allow a small one)
    if rng.random() < 0.08:
        min_room_w_m = rng.uniform(1.3, 2.0)  # rare: allow a small room
    else:
        min_room_w_m = rng.uniform(2.4, 2.9)
    min_room_w = float(min_room_w_m * px_per_m)
    # Heavy-tailed target areas (m^2)
    pareto_alpha = 1.6
    base_area_m2 = rng.uniform(18.0, 35.0)  # typical small offices
    params.update(dict(ext_th=ext_th, part_th=part_th, core_th=core_th, curve_prob=curve_prob,
                       diag_prob=diag_prob, corridor_w=corridor_w, corridor_wall_th=corridor_wall_th,
                       belt_offset=belt_offset, module_m=module_m, min_room_w_m=min_room_w_m,
                       pareto_alpha=pareto_alpha, base_area_m2=base_area_m2, big_room_mode=big_room_mode))

    # ---- Footprint ----
    round_r = float(rng.uniform(0.0, 6.0) * px_per_m if rng.random() < 0.65 else 0.0)
    canvas.paint_rect_border(2, 2, W-3, H-3, ext_th, rounded_r_px=int(round_r))
    strokes.append(dict(kind="rect_border", layer="exterior", x0=2, y0=2, x1=W-3, y1=H-3, width_px=ext_th, rounded_r_px=int(round_r)))

    inset = int(ext_th + px_per_m * 0.6)

    # ---- Cores ----
    n_cores = int(rng.integers(1, 4))
    core_rects = []
    for _ in range(n_cores):
        side_m = rng.uniform(6.5, 12.0)
        cw = int(side_m * px_per_m)
        ch = int((side_m * rng.uniform(0.85, 1.2)) * px_per_m)
        cx = int(rng.integers(W // 5, 4 * W // 5 - cw))
        cy = int(rng.integers(H // 5, 4 * H // 5 - ch))
        rect = (cx, cy, cx + cw, cy + ch)
        core_rects.append(rect)
        canvas.paint_rect_border(*rect, core_th, rounded_r_px=int(rng.uniform(0, 2.0) * px_per_m))
        strokes.append(dict(kind="rect_border", layer="core", x0=int(rect[0]), y0=int(rect[1]), x1=int(rect[2]), y1=int(rect[3]), width_px=float(core_th)))

    # ---- Corridor ring ----
    rx0 = int(2 + belt_offset)
    ry0 = int(2 + belt_offset)
    rx1 = int(W - 3 - belt_offset)
    ry1 = int(H - 3 - belt_offset)

    if rx1 - rx0 > 40 and ry1 - ry0 > 40:
        # Draw ring walls (thicker), then carve corridor void along centerline
        canvas.paint_rect_border(rx0, ry0, rx1, ry1, corridor_w + 2*corridor_wall_th, rounded_r_px=int(round_r * 0.6))
        strokes.append(dict(kind="rect_border", layer="corridor_wall", x0=rx0, y0=ry0, x1=rx1, y1=ry1,
                            width_px=float(corridor_w + 2*corridor_wall_th), rounded_r_px=int(round_r * 0.6)))

        # Carve the corridor clear space
        canvas.paint_rect_border(rx0, ry0, rx1, ry1, 0.0, rounded_r_px=0)  # metadata only (no-op wall)
        canvas.carve_polyline([(rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1)], corridor_w, closed=True, mark_corridor=True)
        ops.append(dict(op="carve_corridor_ring", x0=rx0, y0=ry0, x1=rx1, y1=ry1, width_px=float(corridor_w), rounded_r_px=int(round_r * 0.6)))

    # ---- Corridor spines ----
    n_spines = int(rng.integers(1, 4))
    spine_lines = []
    for _ in range(n_spines):
        if rng.random() < 0.6:
            x = int(rng.integers(int(W*0.25), int(W*0.75)))
            # paint walls around the spine: (corridor_w + 2*wall_th)
            canvas._paint_segment((x, 3), (x, H-4), corridor_w + 2*corridor_wall_th, set_wall=True)
            strokes.append(dict(kind="segment", layer="corridor_wall", p0=(x, 3), p1=(x, H-4), width_px=float(corridor_w + 2*corridor_wall_th)))
            # carve clear corridor
            canvas._paint_segment((x, 3), (x, H-4), corridor_w, set_wall=False, mark_corridor=True)
            ops.append(dict(op="carve_corridor_spine", p0=(x, 3), p1=(x, H-4), width_px=float(corridor_w)))
            spine_lines.append(('segment', (x, 3), (x, H-4)))
        else:
            y = int(rng.integers(int(H*0.25), int(H*0.75)))
            canvas._paint_segment((3, y), (W-4, y), corridor_w + 2*corridor_wall_th, set_wall=True)
            strokes.append(dict(kind="segment", layer="corridor_wall", p0=(3, y), p1=(W-4, y), width_px=float(corridor_w + 2*corridor_wall_th)))
            canvas._paint_segment((3, y), (W-4, y), corridor_w, set_wall=False, mark_corridor=True)
            ops.append(dict(op="carve_corridor_spine", p0=(3, y), p1=(W-4, y), width_px=float(corridor_w)))
            spine_lines.append(('segment', (3, y), (W-4, y)))

    # Connect cores to ring
    def nearest_point_on_ring(cx, cy):
        cands = [(cx, ry0), (cx, ry1), (rx0, cy), (rx1, cy)]
        d2 = [(cx - px)**2 + (cy - py)**2 for (px, py) in cands]
        return cands[int(np.argmin(d2))]

    for (x0, y0, x1, y1) in core_rects:
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        tx, ty = nearest_point_on_ring(cx, cy)
        canvas._paint_segment((cx, cy), (tx, ty), corridor_w + 2*corridor_wall_th, set_wall=True)
        strokes.append(dict(kind="segment", layer="corridor_wall", p0=(int(cx), int(cy)), p1=(int(tx), int(ty)), width_px=float(corridor_w + 2*corridor_wall_th)))
        canvas._paint_segment((cx, cy), (tx, ty), corridor_w, set_wall=False, mark_corridor=True)
        ops.append(dict(op="carve_corridor_link", p0=(int(cx), int(cy)), p1=(int(tx), int(ty)), width_px=float(corridor_w)))

    # ---- Partitioning with heavy-tailed target areas ----
    inner_rect = (inset, inset, W - inset - 1, H - inset - 1)
    stack = [inner_rect]
    splits_drawn = 0
    max_splits = int(rng.integers(120, 220) if not big_room_mode else rng.integers(20, 90))

    def draw_split(rect, vertical, s, curve=False, diag=False):
        x0, y0, x1, y1 = rect
        if vertical:
            if curve:
                p0 = (s, y0); p1 = (s, y1)
                ctrl = (s + rng.normal(0, module_px * 0.6), (y0 + y1) / 2 + rng.normal(0, module_px * 0.6))
                canvas.paint_bezier_quadratic(p0, p1, ctrl, part_th, samples=48)
                strokes.append(dict(kind="quad_bezier", layer="partition", p0=(float(p0[0]), float(p0[1])),
                                    p1=(float(p1[0]), float(p1[1])), pc=(float(ctrl[0]), float(ctrl[1])), width_px=float(part_th)))
            elif diag:
                dy = rng.integers(-int((y1 - y0)*0.25), int((y1 - y0)*0.25))
                p0 = (s, y0); p1 = (s + dy, y1)
                canvas._paint_segment(p0, p1, part_th, set_wall=True)
                strokes.append(dict(kind="segment", layer="partition", p0=(int(p0[0]), int(p0[1])), p1=(int(p1[0]), int(p1[1])), width_px=float(part_th)))
            else:
                p0, p1 = (s, y0), (s, y1)
                canvas._paint_segment(p0, p1, part_th, set_wall=True)
                strokes.append(dict(kind="segment", layer="partition", p0=(int(p0[0]), int(p0[1])), p1=(int(p1[0]), int(p1[1])), width_px=float(part_th)))
        else:
            if curve:
                p0 = (x0, s); p1 = (x1, s)
                ctrl = ((x0 + x1) / 2 + rng.normal(0, module_px * 0.6), s + rng.normal(0, module_px * 0.6))
                canvas.paint_bezier_quadratic(p0, p1, ctrl, part_th, samples=48)
                strokes.append(dict(kind="quad_bezier", layer="partition", p0=(float(p0[0]), float(p0[1])),
                                    p1=(float(p1[0]), float(p1[1])), pc=(float(ctrl[0]), float(ctrl[1])), width_px=float(part_th)))
            elif diag:
                dx = rng.integers(-int((x1 - x0)*0.25), int((x1 - x0)*0.25))
                p0 = (x0, s); p1 = (x1, s + dx)
                canvas._paint_segment(p0, p1, part_th, set_wall=True)
                strokes.append(dict(kind="segment", layer="partition", p0=(int(p0[0]), int(p0[1])), p1=(int(p1[0]), int(p1[1])), width_px=float(part_th)))
            else:
                p0, p1 = (x0, s), (x1, s)
                canvas._paint_segment(p0, p1, part_th, set_wall=True)
                strokes.append(dict(kind="segment", layer="partition", p0=(int(p0[0]), int(p0[1])), p1=(int(p1[0]), int(p1[1])), width_px=float(part_th)))

    def area_stop_threshold():
        # Pareto tail: A = base * (1 + X), X~Pareto(alpha); in px^2
        X = rng.pareto(pareto_alpha)
        A_m2 = base_area_m2 * (1.0 + X)
        return A_m2 * (px_per_m ** 2)

    while stack and splits_drawn < max_splits:
        # pick largest region first (like best-first BSP)
        areas = [ (r[2]-r[0])*(r[3]-r[1]) for r in stack ]
        idx = int(np.argmax(areas))
        x0, y0, x1, y1 = stack.pop(idx)
        w = x1 - x0
        h = y1 - y0
        if w < 2 * min_room_w or h < 2 * min_room_w:
            continue

        # Heavy-tailed stopping: if current area below a sampled threshold, stop (leave big room)
        A = w * h
        if A < area_stop_threshold():
            continue

        # choose split orientation
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
            # skewed split: bias toward edges (more small rooms)
            u = rng.beta(0.9, 2.2)
            s = int(smin + u * (smax - smin))
            s = int(rng.normal(s, module_px * 0.12))
            s = int(np.clip(s, smin, smax))
            draw_split((x0, y0, x1, y1), True, s, curve=(rng.random() < curve_prob), diag=(rng.random() < diag_prob))
            stack.append((x0, y0, s, y1))
            stack.append((s, y0, x1, y1))
        else:
            smin = y0 + int(min_room_w)
            smax = y1 - int(min_room_w)
            if smax - smin < module_px * 0.5:
                continue
            u = rng.beta(0.9, 2.2)
            s = int(smin + u * (smax - smin))
            s = int(rng.normal(s, module_px * 0.12))
            s = int(np.clip(s, smin, smax))
            draw_split((x0, y0, x1, y1), False, s, curve=(rng.random() < curve_prob), diag=(rng.random() < diag_prob))
            stack.append((x0, y0, x1, s))
            stack.append((x0, s, x1, y1))
        splits_drawn += 1

    # Clean small artifacts
    canvas.wall = closing(canvas.wall, r=1)

    # ---- Place doors ----
    # Base door width distribution (m): lognormal, then truncated, then limited by local wall length
    def sample_door_width_m(local_wall_len_m):
        # mean around 0.95m, sigma 0.18m, clamp to [0.7, min(2.0, 0.33*local_len)]
        w = float(np.exp(rng.normal(np.log(0.95), 0.18)))
        w = max(0.70, min(w, 2.00, 0.33 * local_wall_len_m))
        return w

    # Helper: carve a "capsule" door opening centered at c, oriented by tangent t_hat, with clear width L, and radial depth R
    def carve_capsule(center_xy, t_hat, L_px, R_px):
        tx, ty = unit(t_hat[0], t_hat[1])
        if tx == 0 and ty == 0:
            return
        half = max(1.0, L_px / 2.0)
        p0 = (center_xy[0] - tx * half, center_xy[1] - ty * half)
        p1 = (center_xy[0] + tx * half, center_xy[1] + ty * half)
        canvas._paint_segment(p0, p1, 2*R_px, set_wall=False)
        ops.append(dict(op="carve_door_capsule", p0=(float(p0[0]), float(p0[1])), p1=(float(p1[0]), float(p1[1])), radius_px=float(R_px)))

    # Strategy:
    # 1) Seed random doors along corridor centerlines (more doors on longer segments).
    # 2) Ensure almost every room is connected to a corridor by adding corrective doors.
    # Corridor center approx: use canvas.corridor_mask as guide for positions; but we need tangents.
    # We'll sample positions on corridor spines and ring using the ops we recorded.
    def add_corridor_seed_doors():
        # Build a list of centerline primitives from ops
        center_prims = []
        for o in ops:
            if o['op'] in ('carve_corridor_spine', 'carve_corridor_link'):
                center_prims.append(('segment', o['p0'], o['p1']))
            elif o['op'] == 'carve_corridor_ring':
                # ring -> 4 segments rectangle
                center_prims.append(('segment', (o['x0'], o['y0']), (o['x1'], o['y0'])))
                center_prims.append(('segment', (o['x1'], o['y0']), (o['x1'], o['y1'])))
                center_prims.append(('segment', (o['x1'], o['y1']), (o['x0'], o['y1'])))
                center_prims.append(('segment', (o['x0'], o['y1']), (o['x0'], o['y0'])))

        for prim in center_prims:
            kind = prim[0]
            if kind == 'segment':
                p0, p1 = prim[1], prim[2]
                L = line_length(p0, p1)
                # expected spacing ~ 8-15 m
                exp_spacing_px = rng.uniform(8.0, 15.0) * px_per_m
                n_doors = max(1, int(L / exp_spacing_px))
                for _ in range(n_doors):
                    t = rng.random()
                    cx = p0[0] + t * (p1[0] - p0[0])
                    cy = p0[1] + t * (p1[1] - p0[1])
                    # tangent and normal
                    tx = (p1[0] - p0[0]); ty = (p1[1] - p0[1])
                    txu, tyu = unit(tx, ty)
                    nx, ny = -tyu, txu
                    side = 1.0 if rng.random() < 0.5 else -1.0
                    # move to wall center (one side)
                    door_center = (cx + side * nx * (corridor_w/2 + corridor_wall_th*0.75),
                                   cy + side * ny * (corridor_w/2 + corridor_wall_th*0.75))
                    local_len_m = (L / px_per_m)
                    clear_m = sample_door_width_m(local_len_m)
                    carve_capsule(door_center, (txu, tyu), L_px=clear_m*px_per_m, R_px=corridor_wall_th*0.7 + 1.5)
                    doors.append(dict(center=(float(door_center[0]), float(door_center[1])),
                                      tangent=(float(txu), float(tyu)), normal=(float(nx), float(ny)),
                                      clear_width_m=float(clear_m), side=int(np.sign(side)),
                                      source="seed_corridor"))
            # (Bezier corridors aren't used here; ring+spines are segments.)

    add_corridor_seed_doors()

    # After seeding, ensure connectivity: add a door for almost every room not connected to a corridor.
    def ensure_room_doors():
        space = ~canvas.wall
        labels, sizes = connected_components(space)
        corridor_ids = set(np.unique(labels[canvas.corridor_mask])); corridor_ids.discard(-1)
        if not corridor_ids:
            return
        corridor_ids = set(list(corridor_ids))
        H, W = space.shape
        # For each component not in corridor_ids, try to place a door on boundary touching a corridor across a wall.
        all_ids = set(np.unique(labels)); all_ids.discard(-1)
        room_ids = [cid for cid in all_ids if cid not in corridor_ids]
        rng.shuffle(room_ids)

        for cid in room_ids:
            # probability to allow a rare doorless room
            if rng.random() < 0.03:
                continue

            # Find perimeter of this room
            mask_room = labels == cid
            # boundary pixels of room (dilation minus mask)
            bd = np.zeros_like(mask_room, dtype=bool)
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if dx==0 and dy==0: continue
                    ys = slice(max(0, dy), H + min(0, dy))
                    yd = slice(max(0, -dy), H + min(0, -dy))
                    xs = slice(max(0, dx), W + min(0, dx))
                    xd = slice(max(0, -dx), W + min(0, -dx))
                    bd[yd, xd] |= mask_room[ys, xs]
            bd &= ~mask_room  # strip interior

            # candidate wall pixels between room and corridor: wall AND adjacent to room and corridor
            near_room = bd
            near_corr = np.zeros_like(mask_room, dtype=bool)
            for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                ys = slice(max(0, dy), H + min(0, dy))
                yd = slice(max(0, -dy), H + min(0, -dy))
                xs = slice(max(0, dx), W + min(0, dx))
                xd = slice(max(0, -dx), W + min(0, -dx))
                near_corr[yd, xd] |= canvas.corridor_mask[ys, xs]
            candidate = canvas.wall & near_room & near_corr
            ys, xs = np.where(candidate)
            if len(xs) == 0:
                # fallback: pick any wall pixel adjacent to the room
                candidate = canvas.wall & near_room
                ys, xs = np.where(candidate)
                if len(xs) == 0:
                    continue

            k = int(rng.integers(0, len(xs)))
            x = xs[k]; y = ys[k]
            nx = float(canvas.nx[y, x]); ny = float(canvas.ny[y, x])
            # tangent is perpendicular to normal
            tx, ty = -ny, nx
            # estimate local wall length by walking along tangent until normal changes a lot
            L_count = 0
            max_walk = int(12 * px_per_m)  # up to ~12 m
            for s in range(1, max_walk):
                xi = int(round(x + tx*s)); yi = int(round(y + ty*s))
                if not (0 <= xi < W and 0 <= yi < H): break
                if not canvas.wall[yi, xi]: break
                nxd = canvas.nx[yi, xi]; nyd = canvas.ny[yi, xi]
                if nxd*nx + nyd*ny < 0.94: break  # angle dev > ~20 deg
                L_count += 1
            for s in range(1, max_walk):
                xi = int(round(x - tx*s)); yi = int(round(y - ty*s))
                if not (0 <= xi < W and 0 <= yi < H): break
                if not canvas.wall[yi, xi]: break
                nxd = canvas.nx[yi, xi]; nyd = canvas.ny[yi, xi]
                if nxd*nx + nyd*ny < 0.94: break
                L_count += 1
            local_len_px = max(4, L_count)
            clear_m = sample_door_width_m(local_len_px / px_per_m)
            carve_capsule((x, y), (tx, ty), L_px=clear_m*px_per_m, R_px=corridor_wall_th*0.7 + 1.5)
            doors.append(dict(center=(float(x), float(y)),
                              tangent=(float(tx), float(ty)), normal=(float(nx), float(ny)),
                              clear_width_m=float(clear_m), side=0, source="connectivity_fix"))

    ensure_room_doors()

    # Final morphological clean to remove tiny artifacts after carving
    canvas.wall = closing(canvas.wall, r=1)
    # Zero normals where there is no wall
    canvas.nx[~canvas.wall] = 0.0
    canvas.ny[~canvas.wall] = 0.0

    # ---- Pack metadata ----
    scene = dict(
        version=2,
        seed=int(seed),
        canvas=dict(width_m=float(width_m), height_m=float(height_m), px_per_m=float(px_per_m),
                    W=int(W), H=int(H)),
        params=params,
        strokes=strokes,  # paint ops (walls)
        ops=ops,          # carve ops (corridors, doors)
        doors=doors       # explicit door list (redundant info but convenient)
    )

    return canvas.wall, np.stack([canvas.nx, canvas.ny], axis=-1).astype(np.float32), scene

# ---------------- UI + save ----------------

def save_artifacts(mask, normals, scene, prefix="/Users/xoren/icassp2025/generated_rooms/"):
    np.save(prefix + "_mask.npy", mask.astype(np.uint8))
    np.save(prefix + "_normals.npy", normals.astype(np.float32))
    with open(prefix + ".json", "w") as f:
        json.dump(scene, f, indent=2)

def visualize(mask):
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.imshow(mask, cmap="gray_r", interpolation="nearest")
    ax.set_title("Wall mask (1=wall)")
    ax.set_axis_off()
    plt.show()

def run_once_and_show():
    mask, normals, scene = generate_floor_scene()
    save_artifacts(mask, normals, scene)
    visualize(mask)

def show_generator():
    if _HAS_WIDGETS:
        btn = widgets.Button(description="New floor", button_style="primary")
        out = widgets.Output()

        def on_click(_):
            with out:
                out.clear_output(wait=True)
                mask, normals, scene = generate_floor_scene()
                save_artifacts(mask, normals, scene)
                visualize(mask)

        btn.on_click(on_click)
        display(btn, out)
        # initial render & save
        on_click(None)
    else:
        # Fallback single render
        run_once_and_show()

show_generator()
