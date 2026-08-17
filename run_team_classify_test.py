"""
Team classification: split detected players into two teams by jersey color,
per-frame, with NO tracking/identity involved (team is a per-frame,
per-detection property -- doesn't need the cross-frame continuity that
measured badly fragmented under tracking, see results/tracking_finding.md).

Deliberately NOT using roboflow/sports' TeamClassifier (SigLIP-based) --
needs `transformers`' torch backend, measured unavailable on this platform
(torch capped at 2.2.2 on Intel macOS, transformers needs >=2.5; see
requirements.txt).

MEASURED (results/team_classify_finding.md): four variants of joint
hue/color k-means all failed -- same-team players kept splitting across
clusters. Root cause: a genuinely low-saturation team (e.g. white jerseys)
gets outvoted/distorted by clustering that's implicitly built around finding
HUE variation. Fix that actually worked: split on saturation FIRST (a
threshold, not a cluster) to separate white/grey from colored, then cluster
hue only within the colored group. For a white-vs-colored matchup (this
game) the saturation split alone IS the team split. For a colored-vs-colored
matchup this would need the hue-clustering step too, applied after removing
low-saturation outliers (bystanders), not instead of the saturation split.

Known remaining contamination: sideline/bench people (not on-field players)
still get classified into one team or the other. Fix: filter through
frisbee_analysis.calibration.is_in_field() once a calibration exists
(run_calibrate.py) -- not done here, this script works on raw detections.

Run:  python run_team_classify_test.py <video_path> <t_sec> [out.jpg]
Needs opencv-python + ultralytics (Track B deps).
"""
import sys
import cv2
import numpy as np
from ultralytics import YOLO

video_path = sys.argv[1] if len(sys.argv) > 1 else "videos/ojuc.mp4"
t_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 320.0
out_path = sys.argv[3] if len(sys.argv) > 3 else "outputs/team_classify_sample.jpg"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * fps))
ok, frame = cap.read()
cap.release()
if not ok:
    sys.exit(f"could not read frame at t={t_sec}s from {video_path}")

model = YOLO("yolo11n.pt")
results = model.predict(frame, classes=[0], conf=0.25, imgsz=1920, verbose=False)[0]
boxes = results.boxes.xyxy.cpu().numpy()
print(f"Detections: {len(boxes)}")

if len(boxes) < 2:
    sys.exit("not enough detections to classify")


def torso_saturation(frame, box):
    x1, y1, x2, y2 = box.astype(int)
    h, w = y2 - y1, x2 - x1
    # middle 50% width, middle third height -- avoid head/hair, legs/grass,
    # and background bleeding in at the box edges
    tx1, tx2 = x1 + w // 4, x2 - w // 4
    ty1, ty2 = y1 + h // 4, y1 + h // 2
    crop = frame[max(ty1, 0):max(ty2, ty1 + 1), max(tx1, 0):max(tx2, tx1 + 1)]
    if crop.size == 0:
        crop = frame[y1:y2, x1:x2]
    sat = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1]
    return float(np.median(sat))  # median: robust to a contaminated sub-region


med_sat = np.array([torso_saturation(frame, b) for b in boxes])
sat_thresh = np.median(med_sat)
is_low_sat = med_sat < sat_thresh  # white/grey candidate vs colored-jersey candidate

n_low, n_high = is_low_sat.sum(), (~is_low_sat).sum()
print(f"Low-saturation (white/grey candidate): {n_low}")
print(f"High-saturation (colored-jersey candidate): {n_high}")

vis = frame.copy()
for box, low_sat in zip(boxes, is_low_sat):
    x1, y1, x2, y2 = box.astype(int)
    color = (255, 100, 0) if low_sat else (0, 0, 255)  # blue = low-sat team, red = colored team
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
cv2.imwrite(out_path, vis)
print(f"\nSaved color-coded frame to {out_path}")
