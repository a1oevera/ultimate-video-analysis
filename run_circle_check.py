"""Independent metric check of a calibration using the SOCCER centre circle.

The ultimate field is coned; the soccer pitch under it is painted, and its centre
circle is 9.15m radius under the Laws of the Game for EVERY pitch size. Those
pixels are never used to fit an ultimate-field calibration, so measuring them
through the calibration is genuine verification, not a circular test.

Method: map paint pixels image->field with H (which maps IMAGE->FIELD), RANSAC a
circle in FIELD coordinates -- where a correct homography makes the circle a true
circle and the straight lines stay straight, so the two separate cleanly. Then
fit an ellipse to the inliers: axis ratio ~1.00 means the plane is metrically
right, not merely self-consistent.

Run:  python3 run_circle_check.py <calibration_stem> <anchor_seconds>
"""
import sys
import numpy as np
import cv2

from frisbee_analysis import UltimatePitchConfiguration
from frisbee_analysis.calibration import Calibration, field_validity_region

REGULATION_R_CM = 915.0

stem = sys.argv[1] if len(sys.argv) > 1 else "calibration_t3235_rowfit"
anchor_s = float(sys.argv[2]) if len(sys.argv) > 2 else 3235.0
video = sys.argv[3] if len(sys.argv) > 3 else "videos/torovcanyon.mp4"

cfg = UltimatePitchConfiguration()
calib = Calibration.load(f"outputs/calibration/{stem}.json")
H = np.asarray(calib.H())
xmax = field_validity_region(calib, cfg)["usable_x_max"]

cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
cap.set(cv2.CAP_PROP_POS_FRAMES, int(anchor_s * fps))
ok, frame = cap.read()
cap.release()
if not ok:
    sys.exit(f"could not read {video} at {anchor_s}s")

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
white = (hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 170)
grass = ((hsv[:, :, 0] > 30) & (hsv[:, :, 0] < 95) & (hsv[:, :, 1] > 60)).astype(np.uint8)
white = white & (cv2.dilate(grass, np.ones((25, 25), np.uint8)) > 0)

ys, xs = np.nonzero(white)
Q = H @ np.stack([xs, ys, np.ones_like(xs)]).astype(np.float64)
fx, fy = Q[0] / Q[2], Q[1] / Q[2]
keep = (fx > -2000) & (fx < xmax) & (fy > -3000) & (fy < 7000) \
    & np.isfinite(fx) & np.isfinite(fy)
pts = np.stack([fx[keep], fy[keep]], 1)
print(f"{stem} @ t={anchor_s:.0f}s: {white.sum()} paint px, {len(pts)} in field region")
if len(pts) < 50:
    sys.exit("  too little paint maps into the field region -- inconclusive")

# The test assumes the calibration's pixel frame matches the raw frame just read
# (true when the window's anchor IS that frame). When it is not, the paint maps
# to an absurd footprint and every number downstream is meaningless -- at t=1700
# the centre circle smeared across 179m of streaks, which reads as a broken
# calibration but is only a frame mismatch. Judge by the OUTCOME, not by whether
# the clicked points fit inside one frame: t=3235 has points outside the frame
# (they came from other frames of the pan) and maps correctly regardless.
spanx = np.percentile(pts[:, 0], 99) - np.percentile(pts[:, 0], 1)
spany = np.percentile(pts[:, 1], 99) - np.percentile(pts[:, 1], 1)
print(f"  mapped paint footprint {spanx/100:.0f}m x {spany/100:.0f}m")
if spanx > 15000 or spany > 12000:
    sys.exit(f"  INCONCLUSIVE: that footprint is larger than the field. Either this\n"
             f"  calibration is not expressed in the frame at t={anchor_s:.0f}s, or it is\n"
             f"  badly wrong -- this test cannot tell which. Verify the anchor first\n"
             f"  (overlay the saved line_details points and check they sit on features).")


def circle_from3(p):
    (x1, y1), (x2, y2), (x3, y3) = p
    A = np.array([[x2 - x1, y2 - y1], [x3 - x1, y3 - y1]], float) * 2
    b = np.array([x2 * x2 + y2 * y2 - x1 * x1 - y1 * y1,
                  x3 * x3 + y3 * y3 - x1 * x1 - y1 * y1], float)
    try:
        cen = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return cen, float(np.hypot(*(np.array([x1, y1]) - cen)))


rng = np.random.default_rng(0)
TOL = 40.0
best = None
for _ in range(4000):
    r = circle_from3(pts[rng.choice(len(pts), 3, replace=False)])
    if r is None:
        continue
    cen, rad = r
    if not (300 < rad < 3000):        # 3m..30m, wide around 9.15 so the answer is not assumed
        continue
    n = (np.abs(np.hypot(*(pts - cen).T) - rad) < TOL).sum()
    if best is None or n > best[0]:
        best = (n, cen, rad)
if best is None:
    sys.exit("  no circle found -- inconclusive")
n, cen, rad = best
inl = pts[np.abs(np.hypot(*(pts - cen).T) - rad) < TOL]

A = np.column_stack([inl[:, 0], inl[:, 1], np.ones(len(inl))])
sol, *_ = np.linalg.lstsq(A, (inl ** 2).sum(1), rcond=None)
cx, cy = sol[0] / 2, sol[1] / 2
R = float(np.sqrt(sol[2] + cx * cx + cy * cy))
res = np.hypot(*(inl - [cx, cy]).T) - R
angs = np.degrees(np.arctan2(inl[:, 1] - cy, inl[:, 0] - cx))
arc = (np.histogram(angs, bins=36, range=(-180, 180))[0] > 0).sum() * 10

# Always render what was selected. There is no reliable automatic test that the
# RANSAC circle IS the painted centre circle: at t=1700 a 25.45m "circle" with
# 0.4% sector spread and a plausible 76x17m footprint passed every numeric check
# and was, on inspection, a handful of specks near the halfway line. The overlay
# is the check. Do not quote the radius until the red pixels trace real paint.
_sel = np.nonzero(keep)[0][np.abs(np.hypot(*(pts - cen).T) - rad) < TOL]
_vis = frame.copy()
for _d in range(-3, 4):
    _vis[np.clip(ys[_sel] + _d, 0, _vis.shape[0] - 1), xs[_sel]] = (0, 0, 255)
for _c, _t in (((0, 0, 0), 6), ((255, 255, 255), 2)):
    cv2.putText(_vis, f"{stem}  r={rad/100:.2f}m  n={len(_sel)}  "
                      f"-- red must trace the painted circle", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, _c, _t)
_out = f"outputs/diagnostics/circle_check_{stem}.jpg"
cv2.imwrite(_out, _vis)

print(f"  inliers {len(inl)}, paint present over {arc} deg of 360")
print(f"  centre  ({cx/100:6.2f},{cy/100:6.2f}) m   (ultimate field centre is 50.00,18.50)")
print(f"  RADIUS  {R/100:5.2f} m  vs regulation 9.15 m   "
      f"-> {(R-REGULATION_R_CM)/100:+.2f} m ({(R/REGULATION_R_CM-1)*100:+.1f}%)")
print(f"  residual RMS {np.std(res):.1f} cm")

# Radius per angular sector is the honest shape test. Do NOT use cv2.fitEllipse
# here: the visible paint is typically half the circle, and on a half arc both
# cv2 and an algebraic conic fit report a ~1.18 axis ratio for what the per-sector
# radii show to be a circle constant to ~1.5%. That trap cost a wrong conclusion.
print("  radius by sector (flat = a real circle; an ellipse would swing widely):")
spread = []
for lo in range(-180, 180, 30):
    m = (angs >= lo) & (angs < lo + 30)
    if m.sum() < 20:
        continue
    rr = np.hypot(*(inl[m] - [cx, cy]).T).mean()
    spread.append(rr)
    print(f"    {lo:+4d}..{lo+30:+4d}  {rr/100:5.2f} m   n={m.sum()}")
if spread:
    sp = (max(spread) - min(spread)) / np.mean(spread) * 100
    print(f"  sector spread {sp:.1f}%  -> "
          + ("consistent with a circle" if sp < 5 else "NOT circular; plane is distorted"))
print(f"\n  CONFIRM BEFORE QUOTING: open {_out} and check the red pixels trace the")
print(f"  painted centre circle. Every number above is meaningless if they do not.")
