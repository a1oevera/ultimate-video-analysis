"""Find a frame where the centre circle AND enough field lines are BOTH well resolved.

This is the blocker 3.56/3.57 ended on. The far endzone is pinned to ~1.4m and no
better, because nothing ties t=3625 to the trusted t=4200: the circle could tie
them, but t=4200 measures the circle at field x~54m while its own validity stops at
50.7m. What would fix it is a frame that sees the circle AND enough lines to
calibrate from, with BOTH inside the region where that frame resolves the ground.
3.51 said no frame carries the full circle and the near sideline, and guessed that
was structural. This tests it instead of guessing.

Scoring, per sampled frame, using the chained homography (good enough to RANK, not
to calibrate -- see run_pan_extent.py):

  resolution  metres of ground per image pixel, from the singular values of the
              field->pixel Jacobian. The same measure field_visibility uses, so
              "well resolved" means the same thing here as everywhere else.
  circle      fraction of the circle's 360 degrees that lands in frame and under
              the resolution cutoff
  lines       visible length in pixels of each field line, in frame and under the
              cutoff; both sidelines are required, per the measured conditioning
              rule (4 lines WITH both sidelines predict to 7-47px, with one,
              284-1769px)

The circle's field position is itself uncertain by ~4.5m between the two frames
that have measured it, so the nominal used here is deliberately a midpoint and the
cutoffs are loose. This ranks candidates; each still has to be clicked.

CHAIN DRIFT BOUNDS THIS. The chain now spans the whole video, but 840 steps from
the anchor it is worthless: at t=180 the predicted lines land in the top-left
corner of a frame whose centre circle fills the middle. So the search is windowed
to where the chain has been checked against the picture -- t=2795..4310, the span
the neighbour-only chain held on its own, inside which the t=3625 prediction of the
near sideline was measured at 21cm (sd 1.6cm). Widen it only with a frame-by-frame
check, not on trust.

Run:  python3 run_find_circle_frame.py [max_m_per_px] [t_lo] [t_hi]
"""
import os
import sys
import numpy as np

from frisbee_analysis import UltimatePitchConfiguration

cfg = UltimatePitchConfiguration()
MAX_M_PER_PX = float(sys.argv[1]) if len(sys.argv) > 1 else 0.20
T_LO = float(sys.argv[2]) if len(sys.argv) > 2 else 2795.0
T_HI = float(sys.argv[3]) if len(sys.argv) > 3 else 4310.0
W, H = 1920, 1080
# midpoint of the two disagreeing measurements (53.94,18.38,8.595) and
# (50.04,16.26,8.070) -- the disagreement is 3.56's open question, not settled here
CIRCLE = (5200.0, 1730.0, 830.0)
LINES = {(1, 7): "near sideline", (2, 8): "far sideline", (1, 2): "end line x=0",
         (3, 4): "goal x=18", (5, 6): "goal x=82", (7, 8): "end line x=100",
         (11, 12): "halfway"}


def resolved(Hf, fx, fy):
    """metres of ground per pixel, worst direction, at these field points"""
    def proj(x, y):
        p = Hf @ np.stack([x, y, np.ones_like(x)])
        return p[:2] / p[2], p[2]
    p0, w0 = proj(fx, fy)
    px, _ = proj(fx + 1.0, fy)
    py, _ = proj(fx, fy + 1.0)
    J = np.stack([(px - p0), (py - p0)], axis=-1).transpose(1, 0, 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        smin = np.linalg.svd(J, compute_uv=False)[:, -1]
        mpp = 1.0 / (smin * 100.0)
    return p0.T, w0, mpp


Z = np.load("outputs/pan_extent/chain_field_H.npz")
rows = []
for key in Z.files:
    if not (T_LO <= int(key) <= T_HI):
        continue
    M = Z[key]
    Hf = np.linalg.inv(M)
    ref = Hf @ np.array([cfg.length / 2, cfg.width / 2, 1.0])
    sgn = np.sign(ref[2])

    th = np.linspace(0, 2 * np.pi, 360)
    cx, cy, r = CIRCLE
    uv, w, mpp = resolved(Hf, cx + r * np.cos(th), cy + r * np.sin(th))
    ok = ((np.sign(w) == sgn) & np.isfinite(mpp) & (mpp <= MAX_M_PER_PX)
          & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H))
    circ_frac = ok.mean()

    lens = {}
    for e in LINES:
        A, B = cfg.vertices[e[0] - 1], cfg.vertices[e[1] - 1]
        t = np.linspace(0, 1, 400)
        uv, w, mpp = resolved(Hf, A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1]))
        ok = ((np.sign(w) == sgn) & np.isfinite(mpp) & (mpp <= MAX_M_PER_PX)
              & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H))
        lens[e] = (np.hypot(*(uv[ok][-1] - uv[ok][0])) if ok.sum() > 2 else 0.0)

    both = bool(lens[(1, 7)] > 150 and lens[(2, 8)] > 150)
    n_lines = sum(1 for v in lens.values() if v > 150)
    rows.append((int(key), circ_frac, both, n_lines, lens))

rows.sort(key=lambda r: (-(r[2]), -(min(r[1], 0.5) * 2 + r[3] / 7.0)))
print(f"{len(rows)} samples in t={T_LO:.0f}..{T_HI:.0f}, "
      f"resolution cutoff {MAX_M_PER_PX} m/px\n")
n_both = sum(1 for r in rows if r[2])
n_good = sum(1 for r in rows if r[2] and r[1] > 0.25)
print(f"frames with BOTH sidelines well resolved: {n_both}")
print(f"  ... and at least a quarter of the circle too: {n_good}\n")
print(f"{'t':>6} {'circle':>7} {'lines':>6}  best lines")
for t, cf, both, n, lens in rows[:20]:
    top = sorted(lens.items(), key=lambda kv: -kv[1])[:4]
    txt = ", ".join(f"{LINES[e]} {v:.0f}px" for e, v in top if v > 150)
    print(f"{t:6d} {cf*100:6.0f}% {n:6d}  {txt}")
