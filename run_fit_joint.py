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

Run:  python3 run_fit_joint.py
"""
import json
import sys
import numpy as np
import cv2
from scipy.optimize import least_squares

from frisbee_analysis import UltimatePitchConfiguration
from frisbee_analysis.calibration import Calibration

FAR = {"near sideline": (1, 7), "far sideline": (2, 8),
       "goal line": (5, 6), "end line": (7, 8)}
NAMES = {(1, 7): "near sideline y=0", (2, 8): "far sideline y=37",
         (5, 6): "goal line x=82", (7, 8): "end line x=100"}
cfg = UltimatePitchConfiguration()
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
    jx = proj(x + 1.0, y) - p0
    jy = proj(x, y + 1.0) - p0
    return np.hypot(*jx) * 0.5 + np.hypot(*jy) * 0.5


def make_residuals(lines, circle, free=True):
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
            out.append((np.hypot(fx - cx, fy - cy) - r) * scale_px_per_cm(Hf, fx, fy))
        return np.concatenate(out)
    return f


d = json.load(open("outputs/diagnostics/clicked_3625.json"))
lines = {FAR[L["name"]]: np.array(L["points"], float)
         for L in d["lines"] if L["name"] in FAR}
circle = np.array(next((L["points"] for L in d["lines"]
                        if L["name"] == "soccer centre circle"), []), float)

print(f"circle from t=4200: centre ({CX/100:.2f}, {CY/100:.2f}) m, radius {R/100:.3f} m")
print(f"t=3625 evidence: {len(lines)} lines, {len(circle)} circle points\n")

FREE = "--fixed-circle" not in sys.argv
start = Calibration.load("outputs/calibration/calibration_t3625_clicked_REJECTED.json")
Hf0 = np.linalg.inv(np.asarray(start.H())); Hf0 /= Hf0[2, 2]
v0 = np.concatenate([Hf0.ravel()[:8], [CX, CY, R]])

sol = least_squares(make_residuals(lines, circle, FREE), v0, method="lm", max_nfev=60000)
H = unpack(sol.x[:8])                        # field -> image(3625)
Hi = np.linalg.inv(H)
cx, cy, r = (sol.x[8], sol.x[9], sol.x[10]) if FREE else (CX, CY, R)
print(f"circle {'fitted freely' if FREE else 'held fixed'}: "
      f"centre ({cx/100:.2f}, {cy/100:.2f}) m, radius {r/100:.3f} m\n")

print("residuals of the joint fit, in t=3625 pixels:")
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
                    image_path="torovcanyon.mp4 t=3625s (4 lines + centre circle from t=4200)",
                    line_details=[{"edge": list(e), "points": lines[e].tolist(),
                                   "frame_indices": None} for e in sorted(lines)])
calib.save("outputs/calibration/calibration_t3625_circletied.json")
print("\nsaved outputs/calibration/calibration_t3625_circletied.json")
