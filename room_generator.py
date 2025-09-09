import json
import math
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any

import numpy as np


# ---------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------

@dataclass
class WallSpec:
    wall_id: int
    start_m: Tuple[float, float]
    end_m: Tuple[float, float]
    start_px: Tuple[int, int]
    end_px: Tuple[int, int]
    thickness_m: float
    thickness_px: float
    orientation_rad: float
    length_m: float
    transmittance_db: float
    reflectance_db: float
    normal_xy: Tuple[float, float]


@dataclass
class RoomMeta:
    width_m: float
    height_m: float
    pixel_size_m: float
    grid_h: int
    grid_w: int
    wall_density_per_100sqm: float
    outer_wall_thickness_m: float
    seed: Optional[int]
    walls: List[WallSpec]

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)


# ---------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------

def meters_to_pixels(x_m: float, pixel_size: float) -> float:
    return x_m / pixel_size


def pixels_to_meters(x_px: float, pixel_size: float) -> float:
    return x_px * pixel_size


def segment_bounding_box(x0: float, y0: float, x1: float, y1: float, half_width: float) -> Tuple[int, int, int, int]:
    xmin = math.floor(min(x0, x1) - half_width - 1)
    xmax = math.ceil(max(x0, x1) + half_width + 1)
    ymin = math.floor(min(y0, y1) - half_width - 1)
    ymax = math.ceil(max(y0, y1) + half_width + 1)
    return xmin, ymin, xmax, ymax


def rasterize_thick_segment(
    canvas: np.ndarray,
    nx_canvas: np.ndarray,
    ny_canvas: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    thickness_px: float,
    value: float,
    nx: float,
    ny: float,
    *,
    op: str = "max",
) -> None:
    """
    Rasterize a thick line segment onto canvas using a distance-to-segment test.
    - canvas: 2D array to write dB values into
    - nx_canvas/ny_canvas: 2D arrays to store normals per pixel
    - (x0,y0)-(x1,y1): segment endpoints in pixel coordinates (float)
    - thickness_px: total thickness in pixels
    - value: dB value to write into canvas
    - nx, ny: unit normal components to assign where drawn
    - op: 'max' (recommended) or 'add' for combining with existing values
    """
    h, w = canvas.shape
    half_w = 0.5 * max(1.0, float(thickness_px))

    xmin, ymin, xmax, ymax = segment_bounding_box(x0, y0, x1, y1, half_w)
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w - 1, xmax)
    ymax = min(h - 1, ymax)

    seg_dx = x1 - x0
    seg_dy = y1 - y0
    seg_len2 = seg_dx * seg_dx + seg_dy * seg_dy
    if seg_len2 <= 1e-9:
        # Degenerate => draw a disk
        rr = int(math.ceil(half_w))
        cx = int(round(x0))
        cy = int(round(y0))
        for py in range(max(0, cy - rr), min(h, cy + rr + 1)):
            for px in range(max(0, cx - rr), min(w, cx + rr + 1)):
                if (px - cx) * (px - cx) + (py - cy) * (py - cy) <= rr * rr:
                    if op == "max":
                        if value > canvas[py, px]:
                            canvas[py, px] = value
                            nx_canvas[py, px] = nx
                            ny_canvas[py, px] = ny
                    elif op == "add":
                        canvas[py, px] += value
                        nx_canvas[py, px] = nx
                        ny_canvas[py, px] = ny
        return

    inv_len2 = 1.0 / seg_len2
    half_w2 = half_w * half_w

    for py in range(ymin, ymax + 1):
        for px in range(xmin, xmax + 1):
            # Point projection parameter t in [0,1]
            vx = px - x0
            vy = py - y0
            t = (vx * seg_dx + vy * seg_dy) * inv_len2
            if t < 0.0:
                dxp = px - x0
                dyp = py - y0
            elif t > 1.0:
                dxp = px - x1
                dyp = py - y1
            else:
                projx = x0 + t * seg_dx
                projy = y0 + t * seg_dy
                dxp = px - projx
                dyp = py - projy
            if dxp * dxp + dyp * dyp <= half_w2:
                if op == "max":
                    if value > canvas[py, px]:
                        canvas[py, px] = value
                        nx_canvas[py, px] = nx
                        ny_canvas[py, px] = ny
                elif op == "add":
                    canvas[py, px] += value
                    nx_canvas[py, px] = nx
                    ny_canvas[py, px] = ny


# ---------------------------------------------------------------
# Room generation
# ---------------------------------------------------------------

def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed if seed is not None else None)


def _sample_num_walls(area_m2: float, density_per_100sqm: float, rng: np.random.Generator) -> int:
    lam = max(0.0, density_per_100sqm) * (area_m2 / 100.0)
    return int(rng.poisson(lam=lam))


def _clip_to_room(px: float, py: float, w: int, h: int, margin_px: float) -> Tuple[float, float]:
    return (
        float(np.clip(px, margin_px, w - 1 - margin_px)),
        float(np.clip(py, margin_px, h - 1 - margin_px)),
    )


def _random_segment_within_room(
    w: int,
    h: int,
    length_px: float,
    rng: np.random.Generator,
    margin_px: float,
) -> Tuple[float, float, float, float, float]:
    angle = rng.uniform(0.0, 2.0 * math.pi)
    dx = math.cos(angle)
    dy = math.sin(angle)
    cx = rng.uniform(margin_px, w - 1 - margin_px)
    cy = rng.uniform(margin_px, h - 1 - margin_px)
    half_len = 0.5 * length_px
    x0 = cx - dx * half_len
    y0 = cy - dy * half_len
    x1 = cx + dx * half_len
    y1 = cy + dy * half_len
    x0, y0 = _clip_to_room(x0, y0, w, h, margin_px)
    x1, y1 = _clip_to_room(x1, y1, w, h, margin_px)
    return x0, y0, x1, y1, angle


# ------------------ BSP partitioning (axis-aligned layout) ------------------

@dataclass
class _Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0


def _bsp_partition(
    root: _Rect,
    min_w: float,
    min_h: float,
    rng: np.random.Generator,
) -> Tuple[List[_Rect], List[Tuple[str, float, float, float]]]:
    """
    Partition root into axis-aligned sub-rectangles using BSP.
    Returns (leaf_rects, splits) where each split is:
      - ('V', x, y0, y1) for a vertical wall at x spanning y0..y1
      - ('H', y, x0, x1) for a horizontal wall at y spanning x0..x1
    All coordinates are in pixels (float).
    """
    leaves: List[_Rect] = []
    splits: List[Tuple[str, float, float, float]] = []

    stack: List[_Rect] = [root]
    while stack:
        r = stack.pop()
        can_split_v = r.w >= 2 * min_w
        can_split_h = r.h >= 2 * min_h
        if not can_split_v and not can_split_h:
            leaves.append(r)
            continue
        # Choose orientation
        if can_split_v and (not can_split_h or r.w > r.h or rng.random() < 0.5):
            x = float(rng.uniform(r.x0 + min_w, r.x1 - min_w))
            splits.append(('V', x, r.y0, r.y1))
            left = _Rect(r.x0, r.y0, x, r.y1)
            right = _Rect(x, r.y0, r.x1, r.y1)
            # Recurse
            stack.append(left)
            stack.append(right)
        else:
            y = float(rng.uniform(r.y0 + min_h, r.y1 - min_h))
            splits.append(('H', y, r.x0, r.x1))
            bottom = _Rect(r.x0, r.y0, r.x1, y)
            top = _Rect(r.x0, y, r.x1, r.y1)
            stack.append(bottom)
            stack.append(top)

    return leaves, splits


def _draw_bsp_splits_as_walls(
    trans: np.ndarray,
    refl: np.ndarray,
    nx_img: np.ndarray,
    ny_img: np.ndarray,
    splits: List[Tuple[str, float, float, float]],
    *,
    thickness_px: float,
    trans_db_range: Tuple[float, float],
    refl_db_range: Tuple[float, float],
    door_prob: float,
    door_width_px_range: Tuple[float, float],
    pixel_size_m: float,
    walls_out: List[WallSpec],
    rng: np.random.Generator,
) -> None:
    """
    Convert partition splits into drawable wall segments with optional door gaps.
    For each split, we sample a dB pair and optionally carve a gap of given width.
    """
    for split in splits:
        orient, c, a0, a1 = split
        # Sample properties per wall
        t_db = float(rng.uniform(*trans_db_range))
        r_db = float(rng.uniform(*refl_db_range))
        half_th = 0.5 * max(1.0, thickness_px)

        length = abs(a1 - a0)
        has_door = (rng.random() < max(0.0, min(1.0, door_prob))) and (length > 4 * half_th)
        door_lo = door_hi = None
        if has_door:
            w_px = float(rng.uniform(*door_width_px_range))
            w_px = max(2.0, min(w_px, length - 2 * half_th))
            center = float(rng.uniform(a0 + half_th + 0.5 * w_px, a1 - half_th - 0.5 * w_px))
            door_lo = center - 0.5 * w_px
            door_hi = center + 0.5 * w_px

        # Build up to two segments excluding the door gap
        segments: List[Tuple[float, float, float, float, float, float]] = []  # x0,y0,x1,y1,nx,ny
        if orient == 'V':
            # vertical line x=c, spanning y in [a0,a1]
            nx, ny = 1.0, 0.0
            if has_door:
                if door_lo > a0:
                    segments.append((c, a0, c, door_lo, nx, ny))
                if door_hi < a1:
                    segments.append((c, door_hi, c, a1, nx, ny))
            else:
                segments.append((c, a0, c, a1, nx, ny))
        else:
            # horizontal line y=c, spanning x in [a0,a1]
            nx, ny = 0.0, 1.0
            if has_door:
                if door_lo > a0:
                    segments.append((a0, c, door_lo, c, nx, ny))
                if door_hi < a1:
                    segments.append((door_hi, c, a1, c, nx, ny))
            else:
                segments.append((a0, c, a1, c, nx, ny))

        for (x0, y0, x1, y1, nxw, nyw) in segments:
            rasterize_thick_segment(trans, nx_img, ny_img, x0, y0, x1, y1, thickness_px, t_db, nxw, nyw, op="max")
            rasterize_thick_segment(refl, nx_img, ny_img, x0, y0, x1, y1, thickness_px, r_db, nxw, nyw, op="max")
            # Record metadata
            walls_out.append(
                WallSpec(
                    wall_id=len(walls_out),
                    start_m=(pixels_to_meters(x0, pixel_size_m), pixels_to_meters(y0, pixel_size_m)),
                    end_m=(pixels_to_meters(x1, pixel_size_m), pixels_to_meters(y1, pixel_size_m)),
                    start_px=(int(round(x0)), int(round(y0))),
                    end_px=(int(round(x1)), int(round(y1))),
                    thickness_m=pixels_to_meters(thickness_px, pixel_size_m),
                    thickness_px=float(thickness_px),
                    orientation_rad=(0.0 if orient == 'H' else math.pi * 0.5),
                    length_m=pixels_to_meters(math.hypot(x1 - x0, y1 - y0), pixel_size_m),
                    transmittance_db=t_db,
                    reflectance_db=r_db,
                    normal_xy=(nxw, nyw),
                )
            )


def generate_room(
    *,
    width_m: float,
    height_m: float,
    wall_density_per_100sqm: float = 2.0,
    pixel_size_m: float = 0.25,
    outer_wall_thickness_m: float = 0.3,
    transmittance_db_range: Tuple[float, float] = (6.0, 18.0),
    reflectance_db_range: Tuple[float, float] = (2.0, 10.0),
    wall_length_m_range: Tuple[float, float] = (2.0, 10.0),
    wall_thickness_m_range: Tuple[float, float] = (0.15, 0.6),
    max_overlap_fraction: float = 0.15,
    layout: str = "bsp",
    min_room_w_m: float = 2.0,
    min_room_h_m: float = 2.0,
    door_prob: float = 0.7,
    door_width_m_range: Tuple[float, float] = (0.7, 1.2),
    seed: Optional[int] = None,
    save_dir: Optional[str] = None,
    basename: str = "room",
) -> Dict[str, Any]:
    """
    Generate a synthetic room with:
      - transmittance map (dB per wall crossing)
      - reflectance map (dB per reflection on wall)
      - normals map (nx, ny per pixel; 0 outside walls)

    Returns a dict with: trans (H,W), refl (H,W), nx (H,W), ny (H,W), meta (RoomMeta).
    Optionally saves arrays to '<save_dir>/<basename>_maps.npz' and metadata to JSON.
    """
    assert width_m > 0 and height_m > 0, "Room dimensions must be positive"
    assert pixel_size_m > 0, "pixel_size_m must be positive"

    rng = _rng(seed)

    w_px = int(round(width_m / pixel_size_m))
    h_px = int(round(height_m / pixel_size_m))
    w_px = max(8, w_px)
    h_px = max(8, h_px)

    # Initialize maps
    trans = np.zeros((h_px, w_px), dtype=np.float32)
    refl = np.zeros((h_px, w_px), dtype=np.float32)
    nx_img = np.zeros((h_px, w_px), dtype=np.float32)
    ny_img = np.zeros((h_px, w_px), dtype=np.float32)

    walls: List[WallSpec] = []

    # Helper canvas for overlap control
    occupancy = np.zeros((h_px, w_px), dtype=np.uint8)

    def _estimate_overlap_fraction(mask_new: np.ndarray) -> float:
        inter = (occupancy & mask_new).sum()
        area = int(mask_new.sum())
        if area == 0:
            return 0.0
        return float(inter) / float(area)

    # Draw outer boundary walls as a rectangle inset by half thickness
    ow_th_m = float(max(0.05, outer_wall_thickness_m))
    ow_th_px = float(max(1.0, meters_to_pixels(ow_th_m, pixel_size_m)))
    half_ow = 0.5 * ow_th_px
    # Choose a mid-strength dB for outer walls
    ow_trans_db = 0.5 * (transmittance_db_range[0] + transmittance_db_range[1])
    ow_refl_db = 0.5 * (reflectance_db_range[0] + reflectance_db_range[1])

    # Top and bottom segments (horizontal)
    for y in [half_ow, h_px - 1 - half_ow]:
        x0, y0 = 1 + half_ow, y
        x1, y1 = w_px - 2 - half_ow, y
        angle = 0.0  # along +x
        nxw, nyw = 0.0, 1.0 if y < (h_px / 2.0) else -1.0
        rasterize_thick_segment(trans, nx_img, ny_img, x0, y0, x1, y1, ow_th_px, ow_trans_db, nxw, nyw, op="max")
        rasterize_thick_segment(refl, nx_img, ny_img, x0, y0, x1, y1, ow_th_px, ow_refl_db, nxw, nyw, op="max")
        walls.append(
            WallSpec(
                wall_id=len(walls),
                start_m=(pixels_to_meters(x0, pixel_size_m), pixels_to_meters(y0, pixel_size_m)),
                end_m=(pixels_to_meters(x1, pixel_size_m), pixels_to_meters(y1, pixel_size_m)),
                start_px=(int(round(x0)), int(round(y0))),
                end_px=(int(round(x1)), int(round(y1))),
                thickness_m=ow_th_m,
                thickness_px=ow_th_px,
                orientation_rad=0.0,
                length_m=pixels_to_meters(abs(x1 - x0), pixel_size_m),
                transmittance_db=ow_trans_db,
                reflectance_db=ow_refl_db,
                normal_xy=(nxw, nyw),
            )
        )
    # Left and right segments (vertical)
    for x in [half_ow, w_px - 1 - half_ow]:
        x0, y0 = x, 1 + half_ow
        x1, y1 = x, h_px - 2 - half_ow
        angle = math.pi * 0.5  # along +y
        nxw, nyw = 1.0 if x < (w_px / 2.0) else -1.0, 0.0
        rasterize_thick_segment(trans, nx_img, ny_img, x0, y0, x1, y1, ow_th_px, ow_trans_db, nxw, nyw, op="max")
        rasterize_thick_segment(refl, nx_img, ny_img, x0, y0, x1, y1, ow_th_px, ow_refl_db, nxw, nyw, op="max")
        walls.append(
            WallSpec(
                wall_id=len(walls),
                start_m=(pixels_to_meters(x0, pixel_size_m), pixels_to_meters(y0, pixel_size_m)),
                end_m=(pixels_to_meters(x1, pixel_size_m), pixels_to_meters(y1, pixel_size_m)),
                start_px=(int(round(x0)), int(round(y0))),
                end_px=(int(round(x1)), int(round(y1))),
                thickness_m=ow_th_m,
                thickness_px=ow_th_px,
                orientation_rad=angle,
                length_m=pixels_to_meters(abs(y1 - y0), pixel_size_m),
                transmittance_db=ow_trans_db,
                reflectance_db=ow_refl_db,
                normal_xy=(nxw, nyw),
            )
        )

    # Update occupancy for the boundary
    occupancy[...] = (trans > 0).astype(np.uint8)

    if layout.lower() == "bsp":
        # Build axis-aligned internal walls using BSP with door gaps
        min_w_px = meters_to_pixels(max(0.5, min_room_w_m), pixel_size_m)
        min_h_px = meters_to_pixels(max(0.5, min_room_h_m), pixel_size_m)
        root = _Rect(half_ow + 2.0, half_ow + 2.0, w_px - 1 - half_ow - 2.0, h_px - 1 - half_ow - 2.0)
        _, splits = _bsp_partition(root, min_w_px, min_h_px, rng)
        wall_th_px = float(max(1.0, meters_to_pixels(float(rng.uniform(*wall_thickness_m_range)), pixel_size_m)))
        door_w_px_range = (
            float(max(0.5, meters_to_pixels(door_width_m_range[0], pixel_size_m))),
            float(max(0.5, meters_to_pixels(door_width_m_range[1], pixel_size_m))),
        )
        _draw_bsp_splits_as_walls(
            trans,
            refl,
            nx_img,
            ny_img,
            splits,
            thickness_px=wall_th_px,
            trans_db_range=transmittance_db_range,
            refl_db_range=reflectance_db_range,
            door_prob=door_prob,
            door_width_px_range=door_w_px_range,
            pixel_size_m=pixel_size_m,
            walls_out=walls,
            rng=rng,
        )
        occupancy[...] = (trans > 0).astype(np.uint8)
    else:
        # Random angled walls (previous behavior)
        area_m2 = width_m * height_m
        n_internal = _sample_num_walls(area_m2, wall_density_per_100sqm, rng)

        for _ in range(n_internal):
            length_m = float(rng.uniform(*wall_length_m_range))
            thickness_m = float(rng.uniform(*wall_thickness_m_range))
            trans_db = float(rng.uniform(*transmittance_db_range))
            refl_db = float(rng.uniform(*reflectance_db_range))

            length_px = meters_to_pixels(length_m, pixel_size_m)
            thickness_px = max(1.0, meters_to_pixels(thickness_m, pixel_size_m))
            margin_px = max(2.0, thickness_px)

            x0, y0, x1, y1, ang = _random_segment_within_room(w_px, h_px, length_px, rng, margin_px=margin_px)
            nxw = -math.sin(ang)
            nyw = math.cos(ang)

            mask_new = np.zeros_like(occupancy)
            rasterize_thick_segment(mask_new, mask_new, mask_new, x0, y0, x1, y1, thickness_px, 1.0, 0.0, 0.0, op="max")
            overlap = _estimate_overlap_fraction(mask_new)
            if overlap > max_overlap_fraction:
                continue
            rasterize_thick_segment(trans, nx_img, ny_img, x0, y0, x1, y1, thickness_px, trans_db, nxw, nyw, op="max")
            rasterize_thick_segment(refl, nx_img, ny_img, x0, y0, x1, y1, thickness_px, refl_db, nxw, nyw, op="max")
            occupancy |= (mask_new > 0).astype(np.uint8)
            walls.append(
                WallSpec(
                    wall_id=len(walls),
                    start_m=(pixels_to_meters(x0, pixel_size_m), pixels_to_meters(y0, pixel_size_m)),
                    end_m=(pixels_to_meters(x1, pixel_size_m), pixels_to_meters(y1, pixel_size_m)),
                    start_px=(int(round(x0)), int(round(y0))),
                    end_px=(int(round(x1)), int(round(y1))),
                    thickness_m=thickness_m,
                    thickness_px=thickness_px,
                    orientation_rad=ang,
                    length_m=length_m,
                    transmittance_db=trans_db,
                    reflectance_db=refl_db,
                    normal_xy=(nxw, nyw),
                )
            )

    meta = RoomMeta(
        width_m=width_m,
        height_m=height_m,
        pixel_size_m=pixel_size_m,
        grid_h=h_px,
        grid_w=w_px,
        wall_density_per_100sqm=wall_density_per_100sqm,
        outer_wall_thickness_m=outer_wall_thickness_m,
        seed=seed,
        walls=walls,
    )

    result = {"trans": trans, "refl": refl, "nx": nx_img, "ny": ny_img, "meta": meta}

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        npz_path = os.path.join(save_dir, f"{basename}_maps.npz")
        json_path = os.path.join(save_dir, f"{basename}_meta.json")
        np.savez_compressed(npz_path, trans=trans, refl=refl, nx=nx_img, ny=ny_img)
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(meta.to_json())
        result["paths"] = {"npz": npz_path, "json": json_path}

    return result


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def _parse_args():
    import argparse

    p = argparse.ArgumentParser("Synthetic room generator")
    p.add_argument("--width_m", type=float, default=20.0)
    p.add_argument("--height_m", type=float, default=20.0)
    p.add_argument("--pixel_size_m", type=float, default=0.25)
    p.add_argument("--density", type=float, default=2.0, help="walls per 100 m^2")
    p.add_argument("--outer_thickness_m", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save_dir", type=str, default="generated_rooms")
    p.add_argument("--basename", type=str, default="room")
    p.add_argument("--layout", type=str, default="bsp", choices=["bsp", "random"], help="layout algorithm")
    p.add_argument("--min_room_w_m", type=float, default=2.0)
    p.add_argument("--min_room_h_m", type=float, default=2.0)
    p.add_argument("--door_prob", type=float, default=0.7)
    p.add_argument("--door_width_m_min", type=float, default=0.7)
    p.add_argument("--door_width_m_max", type=float, default=1.2)
    p.add_argument("--viz", action="store_true", help="visualize a generated room instead of saving maps")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.viz:
        out = generate_room(
            width_m=args.width_m,
            height_m=args.height_m,
            pixel_size_m=args.pixel_size_m,
            wall_density_per_100sqm=args.density,
            outer_wall_thickness_m=args.outer_thickness_m,
            layout=args.layout,
            min_room_w_m=args.min_room_w_m,
            min_room_h_m=args.min_room_h_m,
            door_prob=args.door_prob,
            door_width_m_range=(args.door_width_m_min, args.door_width_m_max),
            seed=args.seed,
            save_dir=None,
        )

        trans = out["trans"]; refl = out["refl"]; nx = out["nx"]; ny = out["ny"]; meta = out["meta"]
        print("Room:")
        print(f"  size_m: {meta.width_m} x {meta.height_m}  grid_px: {meta.grid_w} x {meta.grid_h}  pixel_size: {meta.pixel_size_m} m")
        print(f"  seed: {meta.seed}  walls: {len(meta.walls)}  density/100m^2: {meta.wall_density_per_100sqm}")
        show_n = min(8, len(meta.walls))
        for ws in meta.walls[:show_n]:
            print(
                f"  wall#{ws.wall_id}: start_m={ws.start_m} end_m={ws.end_m} thick_m={ws.thickness_m:.2f} len_m={ws.length_m:.2f} "
                f"trans_db={ws.transmittance_db:.1f} refl_db={ws.reflectance_db:.1f} n=({ws.normal_xy[0]:.2f},{ws.normal_xy[1]:.2f})"
            )
        if len(meta.walls) > show_n:
            print(f"  ... (+{len(meta.walls)-show_n} more walls)")

        import matplotlib.pyplot as plt
        import numpy as _np

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ax0, ax1, ax2 = axes

        im0 = ax0.imshow(trans, origin='lower', cmap='inferno')
        ax0.set_title('Transmittance (dB)')
        fig.colorbar(im0, ax=ax0, fraction=0.046)

        # Overlay wall segments (pixel coordinates)
        for ws in meta.walls:
            x0, y0 = ws.start_px
            x1, y1 = ws.end_px
            lw = max(1.0, ws.thickness_px * 0.5)
            ax0.plot([x0, x1], [y0, y1], color='cyan', linewidth=lw, alpha=0.7)

        im1 = ax1.imshow(refl, origin='lower', cmap='magma')
        ax1.set_title('Reflectance (dB)')
        fig.colorbar(im1, ax=ax1, fraction=0.046)

        # Normals magnitude + quiver overlay
        mag = _np.hypot(nx, ny)
        im2 = ax2.imshow(mag, origin='lower', cmap='gray', vmin=0.0, vmax=1.0)
        ax2.set_title('Normals (magnitude + quiver)')
        fig.colorbar(im2, ax=ax2, fraction=0.046)
        h, w = nx.shape
        stride = max(1, min(h, w) // 40)
        yy, xx = _np.mgrid[0:h:stride, 0:w:stride]
        ax2.quiver(xx, yy, nx[::stride, ::stride], ny[::stride, ::stride], color='lime', scale=25)

        for ax in axes:
            ax.set_xlim(0, meta.grid_w - 1)
            ax.set_ylim(0, meta.grid_h - 1)
            ax.set_aspect('equal')
            ax.set_xlabel('x (px)')
            ax.set_ylabel('y (px)')

        plt.tight_layout()
        os.makedirs(args.save_dir, exist_ok=True)
        viz_path = os.path.join(args.save_dir, f"{args.basename}_viz.png")
        plt.savefig(viz_path, dpi=150)
        print(f"Saved visualization: {viz_path}")
        try:
            plt.show()
        except Exception:
            pass
        raise SystemExit(0)
    out = generate_room(
        width_m=args.width_m,
        height_m=args.height_m,
        pixel_size_m=args.pixel_size_m,
        wall_density_per_100sqm=args.density,
        outer_wall_thickness_m=args.outer_thickness_m,
        layout=args.layout,
        min_room_w_m=args.min_room_w_m,
        min_room_h_m=args.min_room_h_m,
        door_prob=args.door_prob,
        door_width_m_range=(args.door_width_m_min, args.door_width_m_max),
        seed=args.seed,
        save_dir=args.save_dir,
        basename=args.basename,
    )
    paths = out.get("paths", {})
    if paths:
        print(f"Saved: {paths.get('npz')} and {paths.get('json')}")
    else:
        print("Generation complete (not saved)")


