#!/usr/bin/env python3
"""
interactive_ray_debug.py
One‐wall, one‐bounce ray visualiser with live knobs for

• Launch angle        (0–359°)
• Wall thickness      (1–10 px)
• Wall orientation    (0–179°)     ← NEW
───────────────────────────────────────────────────────────────
Resolution is now a CLI flag:

    --width  <pixels>   default 120
    --height <pixels>   default 100

Example:
    python interactive_ray_debug.py --width 180 --height 140
"""
import numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import argparse, math
from numba import njit
import numpy as np, math

# ───────────────────────────────── wall-angle calculator via PCA
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
def wall_angle(img, px, py, inc_deg, win=10, default_wall_deg=90.0):
    """
    PCA-based wall orientation estimator (Numba-accelerated).

    • win: half-window size (patch = (2·win+1)²).  Must encompass the wall.
    • Uses first principal component ⇒ variance-maximising line (unbiased w.r.t x/y).
    """
    h, w = img.shape
    y0, y1 = max(py - win, 0), min(py + win + 1, h)
    x0, x1 = max(px - win, 0), min(px + win + 1, w)
    patch = img[y0:y1, x0:x1]

    ys, xs = np.nonzero(patch)          # local wall pixels
    if xs.size < 2:                     # too sparse → fall back
        wall_deg = default_wall_deg
    else:
        xs = xs.astype(np.float32) + x0
        ys = ys.astype(np.float32) + y0
        wall_deg = _pca_angle(xs, ys)

    # specular reflection (+180° flip requested earlier)
    return (2.0 * wall_deg - inc_deg) % 360.0



# ─────────── scene generator ─ configurable thickness & angle ───────────
def build_scene(w, h, wall_px=1, wall_deg=90):
    """
    Returns a binary image with a straight wall of given thickness (pixels)
    and orientation `wall_deg` (deg from +x axis, 0–179).
    The wall passes through the centre of the room and spans the full diagonal.
    """
    img = np.zeros((h, w), np.uint8)
    cx, cy = w / 2, h / 2
    theta  = math.radians(wall_deg)
    vx, vy = math.cos(theta), math.sin(theta)              # wall direction (unit)
    half_len = math.hypot(w, h) / 2                        # long enough
    half_th  = wall_px / 2.0
    # scan-convert: mark pixels whose perpendicular distance < half_th
    yy, xx = np.indices(img.shape)
    dx, dy = xx - cx, yy - cy
    perp   = np.abs(-vy * dx + vx * dy) / math.hypot(vx, vy)
    proj   =  dx * vx + dy * vy
    mask   = (perp <= half_th) & (np.abs(proj) <= half_len)
    img[mask] = 1
    return img


# ───────── trace: return 6 items (path + first-hit data) ──────────
def trace(img, x0, y0, ang_deg, wall_deg, step=1.0):
    """
    Returns
        xs, ys               – full ray path
        hx, hy               – coords of first wall hit   (None if no hit)
        ang_pred, ang_true   – estimator vs. ground-truth reflection angles
    """
    h, w = img.shape
    dx, dy = math.cos(math.radians(ang_deg)), math.sin(math.radians(ang_deg))
    xs, ys = [x0], [y0]
    x, y = x0, y0
    bounced = False
    hx = hy = ang_pred = ang_true = None

    while 0 <= x < w and 0 <= y < h:
        x += dx * step; y += dy * step
        xs.append(x);  ys.append(y)
        px, py = int(round(x)), int(round(y))
        if not (0 <= px < w and 0 <= py < h):
            break
        if img[py, px] and not bounced:       # first wall hit
            inc = math.degrees(math.atan2(dy, dx)) % 360
            ang_pred = wall_angle(img, px, py, inc)          # 3×3 patch
            ang_true = (2 * wall_deg - inc) % 360            # slider truth
            hx, hy = x, y
            dx, dy = math.cos(math.radians(ang_pred)), math.sin(math.radians(ang_pred))
            bounced = True
            x += dx * 1e-3; y += dy * 1e-3                   # leave pixel
    return np.array(xs), np.array(ys), hx, hy, ang_pred, ang_true


# ─────────── interactive visualiser ─ three sliders ───────────
def interactive(xa, ya, init_ang, init_wp, init_wd, W, H):
    fig, ax = plt.subplots(figsize=(6,5)); plt.subplots_adjust(bottom=0.35)
    ln, = ax.plot([],[],'b-',lw=2); ax.plot(xa,ya,'go')

    s_ang = Slider(plt.axes([0.15,0.22,0.7,0.03]),"Ray °",0,359,
                   valinit=init_ang,valstep=1)
    s_thk = Slider(plt.axes([0.15,0.15,0.7,0.03]),"Wall px",1,10,
                   valinit=init_wp,valstep=1)
    s_wdg = Slider(plt.axes([0.15,0.08,0.7,0.03]),"Wall °",0,179,
                   valinit=init_wd,valstep=1)

    ani=[None]
    # ───────── restart: rebuild scene, trace, draw extras ──────────
    def restart(*_):
        # scene from current slider values
        wall_px  = int(s_thk.val)
        wall_deg = int(s_wdg.val)
        refl = build_scene(W, H, wall_px=wall_px, wall_deg=wall_deg)

        # clear axes and draw background
        ax.clear()
        ax.imshow(refl, cmap='gray_r', origin='upper')
        ax.plot(xa, ya, 'go')
        ax.set_xlim(0, W); ax.set_ylim(H, 0)

        # trace ray → get path and first-hit diagnostics
        xs, ys, hx, hy, ang_pred, ang_true = trace(
            refl, xa, ya, s_ang.val, wall_deg)

        # animated blue path
        ln, = ax.plot([], [], 'b-', lw=2)

        # draw reflection arrows if a wall was hit
        if hx is not None:
            L = 15  # arrow length
            ax.arrow(hx, hy, L*math.cos(math.radians(ang_true)),
                    L*math.sin(math.radians(ang_true)),
                    color='green', width=0.5, head_width=3)
            ax.arrow(hx, hy, L*math.cos(math.radians(ang_pred)),
                    L*math.sin(math.radians(ang_pred)),
                    color='red',   width=0.5, head_width=3)
            err = abs((ang_pred - ang_true + 180) % 360 - 180)
            ax.text(hx, hy-8, f"err={err:.1f}°",
                    color='blue', fontsize=8, ha='center')

        # (re)start animation
        if ani[0]: ani[0].event_source.stop()
        def update(i):
            ln.set_data(xs[:i+1], ys[:i+1])
            return ln,
        ani[0] = FuncAnimation(fig, update, frames=len(xs), interval=25, blit=True)


    s_ang.on_changed(restart); s_thk.on_changed(restart); s_wdg.on_changed(restart)
    restart(); plt.show()

# ─────────── CLI ───────────
if __name__=="__main__":
    import warnings

    p=argparse.ArgumentParser()
    p.add_argument('--x_ant',type=float,default=20)
    p.add_argument('--y_ant',type=float,default=50)
    p.add_argument('--angle',type=int,default=30)
    p.add_argument('--wall_px',type=int,default=1)
    p.add_argument('--wall_deg',type=int,default=90)
    p.add_argument('--width', type=int, default=120)
    p.add_argument('--height',type=int, default=100)
    a=p.parse_args()

    try:
        from sklearn.decomposition import PCA
        import numpy.random as npr
        rng = npr.default_rng(0)
        xs = rng.uniform(-5, 5, 80)
        true_angle = 25.0                # ground-truth wall angle
        rad = math.radians(true_angle)
        ys = math.tan(rad) * xs + rng.normal(0, 0.3, xs.size)

        pca = PCA(n_components=1)
        pca.fit(np.vstack([xs, ys]).T)
        comp = pca.components_[0]
        skl_ang = (math.degrees(math.atan2(comp[1], comp[0]))) % 180

        ours = _pca_angle(xs.astype(np.float32), ys.astype(np.float32))

        print(f"[TEST] PCA angle  ours={ours:.2f}°  sklearn={skl_ang:.2f}°  Δ={abs(ours-skl_ang):.2e}")
    except ImportError:
        print("[TEST] sklearn not installed – PCA cross-check skipped")
        
    interactive(a.x_ant, a.y_ant, a.angle, a.wall_px, a.wall_deg, a.width, a.height)
