import numpy as np
from numba import njit, prange

# Re-use helpers from approx.py so we avoid duplication
from approx import (
    MAX_REFL,
    MAX_TRANS,
    _fspl,
    _euclidean_distance,
    _step_until_wall,
    _estimate_normal,
    _reflect_dir,
)


@njit
def _perp_dir(dx: float, dy: float):
    """Return an integer (px, py) offset that is perpendicular to (dx, dy).

    We pick the dominant axis to guarantee offsets land on neighbouring
    grid cells; the exact sign is irrelevant because we later sweep both
    +/- directions.
    """
    if abs(dx) > abs(dy):
        return 0, 1  # horizontal ray – use vertical neighbours
    else:
        return 1, 0  # vertical / diagonal ray – use horizontal neighbours


@njit
def _update_beam_half_width(global_r_px: float, max_side: int):
    """Quadratic-like growth: half-width ≈ sqrt(r) until saturation.

    This is a cheap surrogate for the circular wavefront.  For small
    `max_side` (≤3) it matches intuitive behaviour: width reaches the
    target after a few pixels.
    """
    side = int(np.sqrt(global_r_px))
    return side if side < max_side else max_side


@njit
def _trace_ray_recursive_beam(
    refl_mat, trans_mat, nx_img, ny_img,
    out_img, counts,
    x0, y0, dx, dy,
    trans_ct, refl_ct,
    acc_loss,
    global_r,
    pixel_size, freq_MHz,
    radial_step, max_dist,
    max_trans, max_refl, max_loss,
    beam_side, cur_side,
):
    # terminate if already too weak
    if acc_loss >= max_loss:
        return

    px_hit, py_hit, _, _, travelled, last_val, cur_val = _step_until_wall(
        trans_mat, x0, y0, dx, dy, radial_step, max_dist
    )

    # paint current segment ---------------------------------------------------
    steps = int(travelled / radial_step) + 1
    perp_x, perp_y = _perp_dir(dx, dy)

    for s in range(1, steps):  # skip s=0 – starting cell was painted earlier
        xi = x0 + dx * radial_step * s
        yi = y0 + dy * radial_step * s
        ix = int(round(xi));  iy = int(round(yi))
        if ix < 0 or ix >= out_img.shape[1] or iy < 0 or iy >= out_img.shape[0]:
            break

        # --- beam half-width (may expand) ---
        dist_px = global_r + radial_step * s
        half = _update_beam_half_width(dist_px, beam_side)
        if half < cur_side:
            half = cur_side  # never shrink within a straight segment

        fspl = _fspl(dist_px * pixel_size, freq_MHz)
        tot  = acc_loss + fspl
        if tot > max_loss:
            tot = max_loss

        for off in range(-half, half + 1):
            px = ix + off * perp_x
            py = iy + off * perp_y
            if px < 0 or px >= out_img.shape[1] or py < 0 or py >= out_img.shape[0]:
                continue
            if tot < out_img[py, px]:
                out_img[py, px] = tot
            counts[py, px] += 1.0

        cur_side = half  # remember for the next step

    # left the map?
    if px_hit < 0:
        return

    # crossing a wall (air → wall exit)
    if last_val > 0. and cur_val == 0.:
        acc_loss += last_val
        trans_ct += 1
        if acc_loss >= max_loss or trans_ct > max_trans:
            return

    # advance position & distance
    new_x = x0 + dx * travelled
    new_y = y0 + dy * travelled
    new_r = global_r + travelled

    # --------------- straight continuation ------------------
    _trace_ray_recursive_beam(
        refl_mat, trans_mat, nx_img, ny_img,
        out_img, counts,
        new_x, new_y, dx, dy,
        trans_ct, refl_ct,
        acc_loss, new_r,
        pixel_size, freq_MHz,
        radial_step, max_dist,
        max_trans, max_refl, max_loss,
        beam_side, cur_side,
    )

    # ---------------- reflection branch --------------------
    if refl_ct < max_refl:
        # reflect only at reflective walls
        refl_val = refl_mat[py_hit, px_hit]
        if refl_val > 0.0:
            nx, ny = _estimate_normal(nx_img, ny_img, px_hit, py_hit)
            if nx != 0.0 or ny != 0.0:
                rdx, rdy = _reflect_dir(dx, dy, nx, ny)
                _trace_ray_recursive_beam(
                    refl_mat, trans_mat, nx_img, ny_img,
                    out_img, counts,
                    new_x, new_y, rdx, rdy,
                    trans_ct, refl_ct + 1,
                    acc_loss + refl_val,
                    new_r,
                    pixel_size, freq_MHz,
                    radial_step, max_dist,
                    max_trans, max_refl, max_loss,
                    beam_side, 0,      # collapse beam on reflection
                )


@njit(parallel=True, fastmath=True, nogil=True, boundscheck=False)
def calculate_beamtrace_loss_with_normals(
    reflectance_mat, transmittance_mat, nx_img, ny_img,
    x_ant, y_ant, freq_MHz,
    beam_side=3,                     # neighbours per side → width = 2*beam_side+1
    max_refl=MAX_REFL, max_trans=MAX_TRANS,
    pixel_size=0.25,
    n_angles=360*128, radial_step=1.0,
    max_loss=160.0,
):
    """Ray tracing with a finite-width beam (beamtrace) using provided normals."""
    h, w = reflectance_mat.shape
    out  = np.full((h, w), max_loss, np.float64)
    cnt  = np.zeros((h, w), np.float32)

    dtheta   = 2.0 * np.pi / n_angles
    max_dist = np.hypot(w, h)
    cos_v    = np.cos(np.arange(n_angles) * dtheta)
    sin_v    = np.sin(np.arange(n_angles) * dtheta)

    for i in prange(n_angles):
        _trace_ray_recursive_beam(
            reflectance_mat, transmittance_mat,
            nx_img, ny_img,
            out, cnt,
            x_ant, y_ant,
            cos_v[i], sin_v[i],
            0, 0,
            0.0, 0.0,
            pixel_size, freq_MHz,
            radial_step, max_dist,
            max_trans, max_refl, max_loss,
            beam_side, 0,
        )

    # untouched pixels: direct FSPL
    for py in prange(h):
        for px in range(w):
            if cnt[py, px] == 0:
                d    = _euclidean_distance(px, py, x_ant, y_ant, pixel_size)
                fspl = _fspl(d, freq_MHz)
                out[py, px] = fspl if fspl < max_loss else max_loss

    return out, cnt 