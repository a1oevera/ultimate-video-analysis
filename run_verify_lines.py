"""Do a calibration's PREDICTED field lines land on real paint?

This is the milestone's actual test: a calibration is useful if the lines nobody
clicked come out on the markings. Reprojection error cannot answer it -- a 4-line
fit reproduces its own clicks exactly however wrong the labelling was.

A naive paint-coverage score was tried before and FAILED (3.x): predicted
touchlines scored 58% on paint, but sliding the line sideways made the score rise
monotonically to 94% at +8m, so it was measuring "is there turf and markings over
there", not "is the line right". The fix is to keep the offset sweep as a built-in
CONTROL: a metric that works must PEAK AT ZERO OFFSET. If it does not, the number
is reported as meaningless rather than quoted.

Scoring is local contrast perpendicular to the line, in two channels, because the
two kinds of paint here are not alike:
  white lines  bright in grayscale
  near sideline  BLUE paint, which has almost no luminance contrast against green
                 (measured +2.0 on grayscale, 3.62) but stands out strongly in
                 B - (R+G)/2
The score takes the better of the two, so a blue line is not judged by a test that
cannot see it.

Run:  python3 run_verify_lines.py <calibration.json> <frame_seconds>
"""
import os
import sys
import numpy as np
import cv2

from frisbee_analysis import UltimatePitchConfiguration
from frisbee_analysis.calibration import Calibration, field_visibility

cfg = UltimatePitchConfiguration()
NAMES = {(1, 7): "near sideline y=0", (2, 8): "far sideline y=37",
         (1, 2): "end line x=0", (3, 4): "goal line x=18",
         (5, 6): "goal line x=82", (7, 8): "end line x=100"}
OFFSETS = np.arange(-500, 501, 50)

cal_path, t = sys.argv[1], sys.argv[2]
calib = Calibration.load(cal_path)
Hf = np.linalg.inv(np.asarray(calib.H()))
clicked = {tuple(d["edge"]) for d in (calib.line_details or [])}
im = cv2.imread(f"outputs/diagnostics/frame_{t}.jpg")
if im is None:
    sys.exit(f"no frame at outputs/diagnostics/frame_{t}.jpg")
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
b, g, r = (im[:, :, i].astype(np.float32) for i in range(3))
blue = b - 0.5 * (r + g)
ref = Hf @ np.array([cfg.length / 2, cfg.width / 2, 1.0])


def resolution(pts):
    """metres of ground per pixel at each field point, worst direction"""
    def proj(P):
        q = Hf @ np.column_stack([P, np.ones(len(P))]).T
        return (q[:2] / q[2]).T
    p0 = proj(pts)
    jx = proj(pts + [1.0, 0.0]) - p0
    jy = proj(pts + [0.0, 1.0]) - p0
    J = np.stack([jx, jy], axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        smin = np.linalg.svd(J, compute_uv=False)[:, -1]
        return 1.0 / (smin * 100.0)


def score(edge, off):
    """median local contrast along the line, shifted `off` cm sideways"""
    A, B = np.array(cfg.vertices[edge[0] - 1], float), np.array(cfg.vertices[edge[1] - 1], float)
    d = (B - A) / np.linalg.norm(B - A)
    nrm = np.array([-d[1], d[0]])
    s = np.linspace(0.02, 0.98, 400)[:, None]
    pts = A + s * (B - A) + nrm * off
    Q = Hf @ np.column_stack([pts, np.ones(len(pts))]).T
    front = np.sign(Q[2]) == np.sign(ref[2])
    uv = (Q[:2] / Q[2]).T
    # NOT field_visibility: its third test re-fits from line_details and asks whether
    # the geometry is stable under click noise. These are 4-line exact fits, so they
    # fail it almost everywhere, and gating on it rejected lines that are plainly
    # drawn right across the picture (n=0..23 samples out of 400). Stability is a
    # question about the calibration; this function asks a different one -- is there
    # paint where the line is drawn -- so it needs only "in front" and "resolved".
    res = resolution(pts)
    ok = (front & (res < 0.5) & (uv[:, 0] > 8) & (uv[:, 0] < 1912)
          & (uv[:, 1] > 8) & (uv[:, 1] < 1072))
    if ok.sum() < 30:
        return None, int(ok.sum())
    P = uv[ok]
    tang = np.gradient(P, axis=0)
    n2 = np.stack([-tang[:, 1], tang[:, 0]], 1)
    n2 /= np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-9)
    out = []
    for img in (gray, blue):
        c = img[P[:, 1].astype(int), P[:, 0].astype(int)]
        a1 = P + n2 * 6, P - n2 * 6
        sides = [img[np.clip(q[:, 1], 0, 1079).astype(int),
                     np.clip(q[:, 0], 0, 1919).astype(int)] for q in a1]
        out.append(float(np.median(c - 0.5 * (sides[0] + sides[1]))))
    return max(out), int(ok.sum())


print(f"{os.path.basename(cal_path)} on t={t}\n")
print(f"{'line':20s} {'clicked':>8} {'score':>7} {'n':>5}  {'peak offset':>12}  verdict")
for e in NAMES:
    base, n = score(e, 0.0)
    if base is None:
        print(f"{NAMES[e]:20s} {'yes' if e in clicked else 'no':>8} "
              f"{'--':>7} {n:5d}  {'':>12}  not visible in this frame")
        continue
    sweep = [(o, score(e, float(o))[0]) for o in OFFSETS]
    sweep = [(o, v) for o, v in sweep if v is not None]
    peak = max(sweep, key=lambda z: z[1])[0]
    ok = abs(peak) <= 50
    print(f"{NAMES[e]:20s} {'yes' if e in clicked else 'no':>8} {base:7.1f} {n:5d}  "
          f"{peak/100:+10.2f} m  "
          + ("peaks at the line" if ok else "NO PEAK AT ZERO -- score meaningless"))
