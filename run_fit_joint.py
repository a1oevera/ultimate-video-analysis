"""Stabilise the far-endzone fit by tying it to the soccer centre circle.

THE PROBLEM (3.54). t=3625 holds the only view of the goal line x=82 and end line
x=100 in this footage, but its far sideline is a 112px stroke at the horizon, so
2px of click error moves the far endzone corner 1.94m (median) and 8.70m (worst).
Lengthening that stroke does not help -- what is unpinned is the line's
perpendicular offset, not its direction, and at the horizon a pixel of offset is
worth metres of ground.

WHAT WAS TRIED FIRST AND REJECTED. The camera is a Veo fixed-centre pan, so one
homography G relates any two frames exactly, and a single field homography should
explain both frames at once: x_4200 = H X, x_3625 = G^-1 H X. That fit converges,
but to 3-16px residuals where t=4200 alone is sub-pixel, because the available G
is not good enough. MEASURED: carrying t=3625's sideline clicks into t=4200 with
the chained G lands them 16-61px off t=4200's own sidelines. The chain accumulates
over 115 steps of 480x270 thumbnails. Direct registration cannot replace it -- the
two frames share no background (3.55).

TYING THE FRAMES THROUGH THE CIRCLE ALSO FAILED, and it is worth saying why. The
soccer centre circle is one physical object both frames see, so requiring both to
describe the SAME circle should couple them without G. It does not work here: the
circle centre sits at field x~54m and t=4200's validity stops at 50.7m, so the
trusted frame measures the circle OUTSIDE its own trustworthy region. Fitted
separately the two frames disagree by 4.48m on where the circle is (though only
0.32m on its radius), and forcing them to share one circle drags t=4200's clicked
lines from sub-pixel out to 10px. A tie has to rest on something both frames can
actually see well, and neither can see the circle's POSITION well.

WHAT DOES WORK is weaker but self-contained: the circularity constraint alone.
t=3625's 16 circle clicks must map to a circle of SOME radius at SOME position --
no reference to t=4200, no assumed 8.5m or 9.15m. That is still 13 constraints on
the homography, anchored at x~50m where t=3625 resolves the ground well, half a
field away from the endzone lines. It is the long baseline the fit was short of.

MEASURED, far-endzone corner movement under 2px of click jitter (median / worst):

    lines only          1.94 m / 8.70 m     at the (100, 37) corner
    circularity free    1.37 m / 3.24 m
    circle FIXED to
      t=4200's          0.86 m / 2.34 m     but bakes in the 4.48m disagreement,
                                            and its line residuals go to 5-15px

The free version is the default here because it is unbiased: all four lines come
out sub-pixel (max 1.13px) and the circle to 4.1cm. It is a real improvement over
lines alone and it is still an order of magnitude short of t=4200's quality.

Run:  python3 run_fit_joint.py [clicked.json] [--fixed-circle]
"""
import json
import os
import sys
import numpy as np
import cv2
from scipy.optimize import least_squares

from frisbee_analysis import UltimatePitchConfiguration
from frisbee_analysis.calibration import Calibration

EDGE_FOR = {
    "near": {"near sideline": (1, 7), "far sideline": (2, 8),
             "goal line": (3, 4), "end line": (1, 2), "halfway": (11, 12)},
    "far":  {"near sideline": (1, 7), "far sideline": (2, 8),
             "goal line": (5, 6), "end line": (7, 8), "halfway": (11, 12)},
}
NAMES = {(1, 7): "near sideline y=0", (2, 8): "far sideline y=37",
         (1, 2): "end line x=0", (3, 4): "goal line x=18",
         (5, 6): "goal line x=82", (7, 8): "end line x=100",
         (11, 12): "halfway x=50"}
cfg = UltimatePitchConfiguration()
# WITHDRAWN as a measurement (3.59): t=4200 reaches x=34.5m and the circle is at
# x~50m, so this was taken 15m outside validity. Kept ONLY as a --fixed-circle
# option and a fallback seed; the default free fit never uses it as truth.
CX, CY, R = np.load("outputs/diagnostics/circle_field_t4200.npy")


def field_line(e):
    a = np.array([*cfg.vertices[e[0] - 1], 1.0])
    b = np.array([*cfg.vertices[e[1] - 1], 1.0])
    return np.cross(a, b)


def unpack(v):
    return np.append(v, 1.0).reshape(3, 3)


def scale_px_per_cm(Hf, x, y):
    """Local ground resolution, so a circle residual in cm can be compared with a
    line residual in px instead of being weighted by guesswork."""
    def proj(a, b):
        p = Hf @ np.stack([a, b, np.ones_like(a)])
        return p[:2] / p[2]
    p0 = proj(x, y)
    return np.hypot(*(proj(x + 1.0, y) - p0))


def make_residuals(lines, circle, free=True, w=1.0):
    """free=True: the circle's centre and radius are unknowns, so the constraint is
    only that the clicks lie on SOME circle. free=False pins them to t=4200's
    measurement, which is more stable and demonstrably biased -- see the docstring."""
    def f(v):
        Hf = unpack(v[:8])                   # field -> image(3625)
        cx, cy, r = (v[8], v[9], v[10]) if free else (CX, CY, R)
        out = []
        Hi = np.linalg.inv(Hf)
        for e, P in sorted(lines.items()):
            l = Hi.T @ field_line(e)
            out.append((P @ l[:2] + l[2]) / np.hypot(l[0], l[1]))
        if len(circle):
            q = Hi @ np.column_stack([circle, np.ones(len(circle))]).T
            fx, fy = q[0] / q[2], q[1] / q[2]
            out.append(w * (np.hypot(fx - cx, fy - cy) - r) * scale_px_per_cm(Hf, fx, fy))
        return np.concatenate(out)
    return f


args = [a for a in sys.argv[1:] if not a.startswith("--")]
src = args[0] if args else "outputs/diagnostics/clicked_3625.json"
d = json.load(open(src))
table = EDGE_FOR[d.get("end", "near")]
stem = os.path.splitext(os.path.basename(src))[0].replace("clicked_", "")
lines = {table[L["name"]]: np.array(L["points"], float)
         for L in d["lines"] if L["name"] in table and len(L["points"]) >= 2}
circle = np.array(next((L["points"] for L in d["lines"]
                        if L["name"] == "soccer centre circle"), []), float)

print(f"evidence: {len(lines)} lines, {len(circle)} circle points\n")

# BOUNDS AND MULTI-START, because an unbounded free circle runs away. MEASURED at
# t=2935 and t=3020: from the lines-only fit as a start, the optimiser settled on a
# 37.5m and a 17.0m "circle" with line residuals of 240px and 129px -- it bought a
# good circle fit by wrecking the lines. The circle is also outvoted or outvoting on
# raw point count (16-25 circle clicks against 8 line points), so it is weighted to
# parity. w is reported because the answer is sensitive to it: pushing w to 5 drags
# the radius from 7.4m to 9.6m AND the line residuals from 0.4px to 6.6px, so the
# circle and the lines genuinely disagree and the lines are the direct evidence.
W_CIRCLE = float(sys.argv[sys.argv.index("--w") + 1]) if "--w" in sys.argv else None
R_BOUNDS = (500.0, 1500.0)
FREE = "--fixed-circle" not in sys.argv
cands = [f"outputs/calibration/calibration_t{stem}_clicked_REJECTED.json",
         f"outputs/calibration/calibration_t{stem}_clicked.json"]
start_path = next((c for c in cands if os.path.exists(c)), None)
if start_path is None:
    sys.exit(f"no lines-only fit to start from; run run_fit_clicked.py on {src} first")
print(f"starting from {start_path}")
start = Calibration.load(start_path)
Hf0 = np.linalg.inv(np.asarray(start.H())); Hf0 /= Hf0[2, 2]
v0 = np.concatenate([Hf0.ravel()[:8], [CX, CY, R]])

w = W_CIRCLE if W_CIRCLE else (np.sqrt(len(lines) * 2 / len(circle)) if len(circle) else 1.0)
f = make_residuals(lines, circle, FREE, w)
if FREE and len(circle):
    lo = np.concatenate([np.full(8, -np.inf), [3000., -2000., R_BOUNDS[0]]])
    hi = np.concatenate([np.full(8, np.inf), [8000., 6000., R_BOUNDS[1]]])
    # SEEDING, in two stages, because neither a guess nor the data alone is enough.
    # MEASURED: hardcoded starts at centre y=18.4m could not reach t=2935's solution
    # at y=15.35m (it returned an 11.0m circle with 507px lines, cost 3.5e5, where
    # starting near the answer gives cost 41.3 with sub-pixel lines). Seeding from
    # the clicks instead works at t=3625 but not at t=2935/t=3020, whose lines-only
    # homography is unstable exactly where the arc sits -- it put the circle centre
    # at (69.1, -8.6)m, outside the field. So: use the data seed when it is
    # physically plausible, and in every case sweep a coarse grid cheaply and refine
    # only the best few. Cost separates these by four orders of magnitude, so picking
    # the winner is not a close call.
    seeds = []
    try:
        q = np.linalg.inv(unpack(Hf0.ravel()[:8])) @ np.column_stack(
            [circle, np.ones(len(circle))]).T
        fx, fy = q[0] / q[2], q[1] / q[2]
        A = np.column_stack([fx, fy, np.ones(len(fx))])
        abc = np.linalg.lstsq(A, fx ** 2 + fy ** 2, rcond=None)[0]
        cx0, cy0 = abc[0] / 2, abc[1] / 2
        r0 = np.sqrt(max(abc[2] + cx0 ** 2 + cy0 ** 2, 1.0))
        if 2000 < cx0 < 8000 and 0 < cy0 < 3700 and R_BOUNDS[0] < r0 < R_BOUNDS[1]:
            seeds.append([cx0, cy0, r0])
            print(f"  circle seeded from the clicks at ({cx0/100:.1f}, {cy0/100:.1f}) "
                  f"r={r0/100:.2f} m")
        else:
            print(f"  data seed ({cx0/100:.1f}, {cy0/100:.1f}) r={r0/100:.2f} m is off "
                  f"the field -- falling back to a grid")
    except Exception:
        pass
    for cx0 in (4000., 4500., 5000., 5500.):
        for cy0 in (1200., 1500., 1850.):
            for r0 in (750., 900.):
                seeds.append([cx0, cy0, r0])

    coarse = []
    for sd in seeds:
        try:
            c = least_squares(f, np.concatenate([Hf0.ravel()[:8], sd]),
                              bounds=(lo, hi), max_nfev=300)
            coarse.append((c.cost, sd))
        except Exception:
            continue
    coarse.sort(key=lambda z: z[0])
    sol = None
    for _, sd in coarse[:3]:
        try:
            cand = least_squares(f, np.concatenate([Hf0.ravel()[:8], sd]),
                                 bounds=(lo, hi), max_nfev=20000)
        except Exception:
            continue
        if sol is None or cand.cost < sol.cost:
            sol = cand
    if sol is None:
        sys.exit("every start failed to converge")
else:
    sol = least_squares(f, v0, method="lm", max_nfev=60000)
print(f"circle weighted at w={w:.2f} relative to the line points")
H = unpack(sol.x[:8])                        # field -> image(3625)
Hi = np.linalg.inv(H)
cx, cy, r = (sol.x[8], sol.x[9], sol.x[10]) if FREE else (CX, CY, R)
print(f"circle {'fitted freely' if FREE else 'held fixed'}: "
      f"centre ({cx/100:.2f}, {cy/100:.2f}) m, radius {r/100:.3f} m\n")

print("residuals of the joint fit, in image pixels:")
for e, P in sorted(lines.items()):
    l = Hi.T @ field_line(e)
    dd = np.abs(P @ l[:2] + l[2]) / np.hypot(l[0], l[1])
    print(f"  {NAMES[e]:22s} max={dd.max():6.2f}px mean={dd.mean():5.2f}px")
q = Hi @ np.column_stack([circle, np.ones(len(circle))]).T
fx, fy = q[0] / q[2], q[1] / q[2]
rad = np.hypot(fx - cx, fy - cy) - r
print(f"  {'centre circle':22s} max={np.abs(rad).max():6.1f}cm mean={np.abs(rad).mean():5.1f}cm "
      f"({len(circle)} pts)")

calib = Calibration(homography=Hi.tolist(),
                    keypoint_labels=[NAMES[e] for e in sorted(lines)] + ["centre circle"],
                    image_points=[], field_points=[],
                    image_path=f"{d.get('source', src)} (lines + centre circle)",
                    line_details=[{"edge": list(e), "points": lines[e].tolist(),
                                   "frame_indices": None} for e in sorted(lines)])
# REFUSE A FIT THAT BOUGHT THE CIRCLE BY WRECKING THE LINES. MEASURED at t=4200,
# whose arc is 94 degrees at the extreme left edge of the frame: the free circle ran
# to the 15m bound and fitted the arc to 0.6cm while the near sideline went to 223px.
# A tiny circle residual is NOT evidence the fit is good -- a large circle far away
# matches a shallow arc beautifully. The lines are the direct evidence and they are
# the test. Bounds and multi-start do not prevent this; only checking does.
worst = 0.0
for e, P in sorted(lines.items()):
    l = Hi.T @ field_line(e)
    worst = max(worst, (np.abs(P @ l[:2] + l[2]) / np.hypot(l[0], l[1])).max())
bad = worst > 5.0
if bad:
    print(f"\n  *** REFUSED: worst line residual {worst:.1f}px. The circle fitted to "
          f"{np.abs(rad).mean():.1f}cm\n"
          f"  *** by distorting the plane instead. This frame's arc is too short or too\n"
          f"  *** close to the frame edge to constrain a free circle -- it cannot tell a\n"
          f"  *** small near circle from a large far one. Use the lines alone here.")

out = (f"outputs/calibration/calibration_t{stem}_circletied_REJECTED.json" if bad
       else f"outputs/calibration/calibration_t{stem}_circletied.json")
calib.save(out)
print(f"\nsaved {out}")
