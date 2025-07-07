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

# ─────────── single-bounce tracer (unchanged core logic) ───────────
def wall_angle(img, px, py, inc_deg):
    patch = img[max(py-1,0):py+2, max(px-1,0):px+2].astype(float)
    gy, gx = np.gradient(patch); nx, ny = gx.mean(), gy.mean()
    wdeg = 90. if nx==ny==0 else (math.degrees(math.atan2(ny,nx))+90)%180
    return (2*wdeg - inc_deg) % 360

def trace(img, x0, y0, ang_deg, step=1.0):
    h,w = img.shape
    dx,dy = math.cos(math.radians(ang_deg)), math.sin(math.radians(ang_deg))
    xs,ys=[x0],[y0]; x,y=x0,y0; bounced=False
    while 0<=x<w and 0<=y<h:
        x+=dx*step; y+=dy*step; xs.append(x); ys.append(y)
        px,py=int(round(x)),int(round(y))
        if not(0<=px<w and 0<=py<h): break
        if img[py,px] and not bounced:
            inc = math.degrees(math.atan2(dy,dx))%360
            out = wall_angle(img,px,py,inc)
            dx,dy = math.cos(math.radians(out)), math.sin(math.radians(out))
            bounced=True; x+=dx*1e-3; y+=dy*1e-3
    return np.array(xs), np.array(ys)

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
    def restart(*_):
        refl = build_scene(W, H, wall_px=int(s_thk.val),
                           wall_deg=int(s_wdg.val))
        ax.clear(); ax.imshow(refl,cmap='gray_r',origin='upper')
        ax.plot(xa,ya,'go'); ax.set_xlim(0,W); ax.set_ylim(H,0)
        xs,ys = trace(refl, xa, ya, s_ang.val)
        if ani[0]: ani[0].event_source.stop()
        ln, = ax.plot([],[],'b-',lw=2)
        def upd(i): ln.set_data(xs[:i+1],ys[:i+1]); return ln,
        ani[0]=FuncAnimation(fig,upd,frames=len(xs),interval=25,blit=True)

    s_ang.on_changed(restart); s_thk.on_changed(restart); s_wdg.on_changed(restart)
    restart(); plt.show()

# ─────────── CLI ───────────
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument('--x_ant',type=float,default=20)
    p.add_argument('--y_ant',type=float,default=50)
    p.add_argument('--angle',type=int,default=30)
    p.add_argument('--wall_px',type=int,default=1)
    p.add_argument('--wall_deg',type=int,default=90)
    p.add_argument('--width', type=int, default=120)
    p.add_argument('--height',type=int, default=100)
    a=p.parse_args()
    interactive(a.x_ant,a.y_ant,a.angle,a.wall_px,a.wall_deg,
                a.width,a.height)
