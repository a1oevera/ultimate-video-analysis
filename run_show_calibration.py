"""Draw a calibration on its frame so it can be judged by eye.

Colours are chosen for one decision: is there real paint where the model says the
halfway line x=50m is? That line is drawn thick and magenta and labelled, because
a WFDF field has no painted halfway line -- the model's x=50 is derived, and
whatever was clicked as "halfway" at t=4200 has been retracted (3.59).

  green   lines that were actually clicked in this frame
  orange  lines the fit predicts but nobody clicked -- these landing on real paint
          is the check that matters
  magenta the halfway line x=50, the open question
  cyan    the fitted centre circle, with its radius printed
  red     the clicked points themselves

Run:  python3 run_show_calibration.py <calibration.json> <frame_seconds>
"""
import json
import os
import sys
import numpy as np
import cv2

from frisbee_analysis import UltimatePitchConfiguration
from frisbee_analysis.calibration import Calibration

cfg = UltimatePitchConfiguration()
NAMES = {(1, 7): "sideline y=0", (2, 8): "sideline y=37",
         (1, 2): "end line x=0", (3, 4): "goal line x=18",
         (5, 6): "goal line x=82", (7, 8): "end line x=100"}

cal_path, t = sys.argv[1], sys.argv[2]
calib = Calibration.load(cal_path)
Hf = np.linalg.inv(np.asarray(calib.H()))          # field -> image
clicked = {tuple(d["edge"]) for d in (calib.line_details or [])}

frame_path = f"outputs/diagnostics/frame_{t}.jpg"
if not os.path.exists(frame_path):
    cap = cv2.VideoCapture("videos/torovcanyon.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(float(t) * (cap.get(cv2.CAP_PROP_FPS) or 30)))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"could not read t={t}")
    cv2.imwrite(frame_path, fr)
im = cv2.imread(frame_path)
ref = Hf @ np.array([cfg.length / 2, cfg.width / 2, 1.0])


def draw(P, col, width, label=None):
    Q = Hf @ P
    uv = (Q[:2] / Q[2]).T[np.sign(Q[2]) == np.sign(ref[2])]
    keep = [p for p in uv if -4000 < p[0] < 6000 and -4000 < p[1] < 4000]
    for i in range(1, len(keep)):
        if np.hypot(*(keep[i] - keep[i - 1])) < 300:
            cv2.line(im, tuple(np.int32(keep[i - 1])), tuple(np.int32(keep[i])), col, width)
    vis = [p for p in keep if 0 <= p[0] < 1920 and 0 <= p[1] < 1080]
    if label and vis:
        p = vis[len(vis) // 2]
        for c, th in (((0, 0, 0), 6), (col, 2)):
            cv2.putText(im, label, (int(p[0]) + 8, int(p[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, th)


for e in list(NAMES):
    A, B = cfg.vertices[e[0] - 1], cfg.vertices[e[1] - 1]
    s = np.linspace(0, 1, 600)
    P = np.stack([A[0] + s * (B[0] - A[0]), A[1] + s * (B[1] - A[1]), np.ones(600)])
    on = e in clicked
    draw(P, (0, 255, 0) if on else (0, 165, 255), 3,
         NAMES[e] + ("" if on else "  (predicted)"))

# the halfway line: the thing being judged
s = np.linspace(0, 1, 600)
draw(np.stack([np.full(600, cfg.length / 2), s * cfg.width, np.ones(600)]),
     (255, 0, 255), 4, "halfway x=50  <- is there paint here?")

cj = cal_path.replace("_circletied.json", "_circle.npy")
circ = None
for guess in (cj, "outputs/calibration/circle_joint.npy"):
    if os.path.exists(guess):
        circ = np.load(guess)
        break
if circ is None and os.path.exists("outputs/diagnostics/circle_field_t4200.npy"):
    circ = None     # do not draw the withdrawn one
if circ is not None:
    th = np.linspace(0, 2 * np.pi, 720)
    draw(np.stack([circ[0] + circ[2] * np.cos(th), circ[1] + circ[2] * np.sin(th),
                   np.ones(720)]), (255, 255, 0), 3, f"circle r={circ[2]/100:.2f} m")

for d in (calib.line_details or []):
    for (x, y) in d["points"]:
        cv2.circle(im, (int(x), int(y)), 7, (0, 0, 255), -1)

out = f"outputs/diagnostics/view_{os.path.basename(cal_path).replace('.json','')}.jpg"
cv2.imwrite(out, im)
print(out)
