"""Visualize a calibration: the field model drawn on its pan canvas, plus the
canvas rectified into the field plane.

Two traps this encodes, both hit while writing it:
  * Calibration.H() maps IMAGE->FIELD, not field->image. For a WARP_INVERSE_MAP
    rectification you want  Hs @ inv(H) @ inv(F2C)  (dst -> field -> image ->
    canvas). Getting it backwards renders an almost-empty frame.
  * A line clipped at the validity limit must not reset the list you later take
    the label position from, or long lines silently report zero visible points.

Run:  python3 run_show_homography.py [calibration_stem] [canvas_glob]
"""
import sys
import glob
import json
import numpy as np
import cv2

from frisbee_analysis import UltimatePitchConfiguration
from frisbee_analysis.calibration import (Calibration, field_to_pixel,
                                          field_validity_region)

stem = sys.argv[1] if len(sys.argv) > 1 else "calibration_t3235_rowfit"
canvas_glob = sys.argv[2] if len(sys.argv) > 2 else "*3235*_pan.jpg"

cfg = UltimatePitchConfiguration()
calib = Calibration.load(f"outputs/calibration/{stem}.json")
xmax = field_validity_region(calib, cfg)["usable_x_max"]

pans = glob.glob(f"outputs/field_lines_cache/{canvas_glob}")
if not pans:
    sys.exit(f"no canvas matching {canvas_glob}")
pan = pans[0]
T = np.asarray(json.load(open(pan.replace("_pan.jpg", ".json")))["canvas_T"])
offx, offy = float(T[0, 2]), float(T[1, 2])
im = cv2.imread(pan)
H, W = im.shape[:2]
print(f"{stem}: valid to {xmax/100:.1f}m of {cfg.length/100:.0f}m   canvas {W}x{H}")

EDGES = [((1, 7), "near sideline", (255, 180, 0)), ((2, 8), "far sideline", (0, 220, 255)),
         ((3, 4), "near goal", (0, 0, 255)), ((5, 6), "far goal", (120, 120, 255)),
         ((1, 2), "near end", (0, 255, 0)), ((7, 8), "far end", (120, 255, 120)),
         ((11, 12), "halfway", (255, 0, 255))]
ov = im.copy()
for e, nm, col in EDGES:
    a = np.asarray(cfg.vertices[e[0] - 1], float)
    b = np.asarray(cfg.vertices[e[1] - 1], float)
    run, drawn, clipped = [], [], 0
    for t in np.linspace(0, 1, 400):
        fx, fy = a + (b - a) * t
        if fx > xmax:
            run = []           # break the polyline, but KEEP `drawn`
            clipped += 1
            continue
        x, y = field_to_pixel(fx, fy, calib)
        P = (int(x + offx), int(y + offy))
        if run:
            cv2.line(ov, run[-1], P, col, 3)
        run.append(P)
        drawn.append(P)
    vis = [p for p in drawn if 0 <= p[0] < W and 0 <= p[1] < H]
    if vis:
        q = vis[len(vis) // 2]
        for cc, th in (((0, 0, 0), 5), (col, 2)):
            cv2.putText(ov, nm, (q[0] + 8, q[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, cc, th)
    note = "ENTIRELY past validity" if not drawn else f"{len(vis)}/{len(drawn)} on canvas"
    print(f"  {nm:14s} {note}")
for cc, th in (((0, 0, 0), 6), ((255, 255, 255), 2)):
    cv2.putText(ov, f"{stem}  --  valid to {xmax/100:.0f}m of {cfg.length/100:.0f}m",
                (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, cc, th)
cv2.imwrite("outputs/diagnostics/homography_now_overlay.jpg", ov)

ppm, M = 8.0, 12.0
OW = int((cfg.length / 100. + 2 * M) * ppm)
OH = int((cfg.width / 100. + 2 * M) * ppm)
F2C = np.array([[ppm / 100., 0, M * ppm], [0, ppm / 100., M * ppm], [0, 0, 1.]])
Hs = np.array([[1, 0, offx], [0, 1, offy], [0, 0, 1.]], float)
Hc = Hs @ np.linalg.inv(np.asarray(calib.H())) @ np.linalg.inv(F2C)
rect = cv2.warpPerspective(im, Hc, (OW, OH), flags=cv2.WARP_INVERSE_MAP)
gx, _ = np.meshgrid(np.arange(OW), np.arange(OH))
rect[(gx - M * ppm) * 100. / ppm > xmax] = 45


def fp(x, y):
    p = F2C @ np.array([x, y, 1.])
    return int(p[0] / p[2]), int(p[1] / p[2])


for a, b in (((0, 0), (0, cfg.width)), ((cfg.length, 0), (cfg.length, cfg.width)),
             ((0, 0), (cfg.length, 0)), ((0, cfg.width), (cfg.length, cfg.width)),
             ((1800, 0), (1800, cfg.width)), ((5000, 0), (5000, cfg.width)),
             ((8200, 0), (8200, cfg.width))):
    cv2.line(rect, fp(*a), fp(*b), (0, 220, 255), 2)
# NB: the yellow box is the FRISBEE field, which is coned, not painted. Soccer
# paint sitting off the box is not evidence of a bad fit -- do not read it that way.
cv2.putText(rect, f"rectified {stem}   grey = past {xmax/100:.0f}m validity",
            (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.imwrite("outputs/diagnostics/homography_now_rectified.jpg", rect)
print("\n-> outputs/diagnostics/homography_now_overlay.jpg")
print("-> outputs/diagnostics/homography_now_rectified.jpg")
