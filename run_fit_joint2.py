"""Both frames and the circle in ONE system, with no inter-frame homography.

RESULT: THIS DOES NOT WORK, and the script is kept because the next person will
try it. Forcing both frames to share one circle drags t=4200's five clicked lines
from sub-pixel out to 10px (goal line x=18: max 10.13px, far sideline 6.35px).
Cause: the circle centre lands at field x~54m and t=4200's validity stops at
50.7m, so the trusted frame is measuring the circle OUTSIDE its own trustworthy
region -- field_visibility says NO at the centre and at three of the four rim
points. A tie between frames has to rest on something both can see WELL, and
neither can see the circle's position well. Fitted separately the two frames put
the centre 4.48m apart.

What survives is the weaker, self-contained constraint in run_fit_joint.py:
t=3625's circle clicks must lie on SOME circle, with no reference to t=4200.

The two frames cannot be tied through a registration G -- the chained G is off by
16-61px on lines visible in both, and the frames share no background so it cannot
be measured directly (3.55). But they can be tied through a physical object they
both see: the soccer centre circle is at ONE field position, so requiring both
frames' circle evidence to describe the SAME circle couples them without G.

Unknowns (19):  H4200 (8) + H3625 (8) + circle cx, cy, r (3)
Evidence:       t=4200's 5 clicked lines, its circle paint, t=3625's 4 clicked
                lines, its 16 circle clicks

Why this matters: t=4200's validity stops at 50.7m and the circle centre sits at
~54m, so t=4200 alone measures the circle OUTSIDE its own trustworthy region --
which is why the two frames disagreed by 4.48m on where the circle is when fitted
separately. Solving them together lets t=3625, which sees the circle from a
different angle, contribute to pinning it.

WEIGHTING is a judgement call and is reported, not hidden: residuals are in pixels
throughout (circle radial error in cm is multiplied by the local px/cm from the
Jacobian), every point counts the same, and t=4200's dense paint is subsampled so
it cannot swamp the 18 line points. --sweep varies that subsample to show whether
the answer depends on the choice.
"""
import json
import sys
import numpy as np
import cv2
from scipy.optimize import least_squares

from frisbee_analysis import UltimatePitchConfiguration
from frisbee_analysis.calibration import Calibration

cfg = UltimatePitchConfiguration()
NEAR_T = {(1, 7): "near sideline y=0", (2, 8): "far sideline y=37",
          (1, 2): "end line x=0", (3, 4): "goal line x=18", (11, 12): "halfway x=50"}
FARMAP = {"near sideline": (1, 7), "far sideline": (2, 8),
          "goal line": (5, 6), "end line": (7, 8)}
FAR_T = {(1, 7): "near sideline y=0", (2, 8): "far sideline y=37",
         (5, 6): "goal line x=82", (7, 8): "end line x=100"}


def fline(e):
    return np.cross(np.array([*cfg.vertices[e[0] - 1], 1.0]),
                    np.array([*cfg.vertices[e[1] - 1], 1.0]))


def circle_paint_4200(H, roi=(0, 540, 262, 820)):
    """The arc, restricted to where it is actually visible. Unrestricted, RANSAC
    locks onto spectators' clothing and returns a 9.32m circle running through the
    crowd -- checked by overlay, per 3.51."""
    frame = cv2.imread("outputs/diagnostics/frame_4200.jpg")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    w = (hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 170)
    g = ((hsv[:, :, 0] > 30) & (hsv[:, :, 0] < 95) & (hsv[:, :, 1] > 60)).astype(np.uint8)
    w &= cv2.dilate(g, np.ones((25, 25), np.uint8)) > 0
    k = np.zeros_like(w)
    k[roi[1]:roi[3], roi[0]:roi[2]] = True
    ys, xs = np.nonzero(w & k)
    return np.stack([xs, ys], 1).astype(float)


def px_per_cm(Hf, fx, fy):
    p0 = Hf @ np.stack([fx, fy, np.ones_like(fx)]); p0 = p0[:2] / p0[2]
    p1 = Hf @ np.stack([fx + 1, fy, np.ones_like(fx)]); p1 = p1[:2] / p1[2]
    return np.hypot(*(p1 - p0))


def build(n_sub):
    base = Calibration.load("outputs/calibration/calibration_t4200_clicked.json")
    near_lines = {tuple(d["edge"]): np.array(d["points"], float) for d in base.line_details}
    H4_0 = np.linalg.inv(np.asarray(base.H())); H4_0 /= H4_0[2, 2]
    d = json.load(open("outputs/diagnostics/clicked_3625.json"))
    far_lines = {FARMAP[L["name"]]: np.array(L["points"], float)
                 for L in d["lines"] if L["name"] in FARMAP}
    far_circ = np.array(next(L["points"] for L in d["lines"]
                             if L["name"] == "soccer centre circle"), float)
    paint = circle_paint_4200(np.asarray(base.H()))
    idx = np.linspace(0, len(paint) - 1, n_sub).astype(int)
    near_circ = paint[idx]
    s3 = Calibration.load("outputs/calibration/calibration_t3625_clicked_REJECTED.json")
    H3_0 = np.linalg.inv(np.asarray(s3.H())); H3_0 /= H3_0[2, 2]
    c0 = np.load("outputs/diagnostics/circle_field_t4200.npy")
    return (near_lines, near_circ, far_lines, far_circ,
            np.concatenate([H4_0.ravel()[:8], H3_0.ravel()[:8], c0]))


def solve(near_lines, near_circ, far_lines, far_circ, v0):
    def resid(v):
        H4 = np.append(v[:8], 1.0).reshape(3, 3)
        H3 = np.append(v[8:16], 1.0).reshape(3, 3)
        cx, cy, r = v[16:19]
        out = []
        for Hf, lines in ((H4, near_lines), (H3, far_lines)):
            Hi = np.linalg.inv(Hf)
            for e, P in sorted(lines.items()):
                l = Hi.T @ fline(e)
                out.append((P @ l[:2] + l[2]) / np.hypot(l[0], l[1]))
        for Hf, C in ((H4, near_circ), (H3, far_circ)):
            Hi = np.linalg.inv(Hf)
            q = Hi @ np.column_stack([C, np.ones(len(C))]).T
            fx, fy = q[0] / q[2], q[1] / q[2]
            out.append((np.hypot(fx - cx, fy - cy) - r) * px_per_cm(Hf, fx, fy))
        return np.concatenate(out)
    s = least_squares(resid, v0, method="lm", max_nfev=200000)
    return s


if "--sweep" in sys.argv:
    print("does the answer depend on how much of t=4200's paint is used?")
    print(f"{'n_sub':>6}  {'centre x':>9} {'centre y':>9} {'radius':>8}")
    for n in (16, 32, 64, 128, 256):
        nl, nc, fl, fc, v0 = build(n)
        s = solve(nl, nc, fl, fc, v0)
        cx, cy, r = s.x[16:19]
        print(f"{n:6d}  {cx/100:9.2f} {cy/100:9.2f} {r/100:8.3f}")
    sys.exit()

nl, nc, fl, fc, v0 = build(32)
s = solve(nl, nc, fl, fc, v0)
H4 = np.append(s.x[:8], 1.0).reshape(3, 3)
H3 = np.append(s.x[8:16], 1.0).reshape(3, 3)
cx, cy, r = s.x[16:19]
print(f"converged={s.success}\n")
print(f"shared circle: centre ({cx/100:.2f}, {cy/100:.2f}) m, radius {r/100:.3f} m")
print(f"  t=4200 alone said (53.94, 18.38) r=8.595 m")
print(f"  t=3625 alone said (49.77, 16.76) r=8.276 m\n")
for tag, Hf, lines, names, C in (("t=4200", H4, nl, NEAR_T, nc),
                                 ("t=3625", H3, fl, FAR_T, fc)):
    Hi = np.linalg.inv(Hf)
    for e, P in sorted(lines.items()):
        l = Hi.T @ fline(e)
        dd = np.abs(P @ l[:2] + l[2]) / np.hypot(l[0], l[1])
        print(f"  {tag}  {names[e]:22s} max={dd.max():6.2f}px mean={dd.mean():5.2f}px")
    q = Hi @ np.column_stack([C, np.ones(len(C))]).T
    fx, fy = q[0] / q[2], q[1] / q[2]
    e = np.abs(np.hypot(fx - cx, fy - cy) - r)
    print(f"  {tag}  {'centre circle':22s} max={e.max():6.1f}cm mean={e.mean():5.1f}cm "
          f"({len(C)} pts)")

for tag, Hf, path in (("4200", H4, "outputs/calibration/calibration_joint_4200.json"),
                      ("3625", H3, "outputs/calibration/calibration_joint_3625.json")):
    lines = nl if tag == "4200" else fl
    Calibration(homography=np.linalg.inv(Hf).tolist(),
                keypoint_labels=[], image_points=[], field_points=[],
                image_path=f"torovcanyon.mp4 t={tag}s (joint two-frame + shared circle)",
                line_details=[{"edge": list(e), "points": P.tolist(),
                               "frame_indices": None} for e, P in sorted(lines.items())]
                ).save(path)
    print(f"saved {path}")
np.save("outputs/calibration/circle_joint.npy", np.array([cx, cy, r]))
