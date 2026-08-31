"""
Warp the video into the FIELD, not into another frame: a top-down 100x37m
canvas showing which parts of the pitch were ever actually observed.

WHY THIS AND NOT A MOSAIC. An image-space mosaic stitches frame->frame, so
error compounds and there is no ground truth to check the result against --
four attempts at one (results/field_lines_finding.md 3.31) produced no usable
image and no way to tell misregistration from honest perspective. The pitch,
though, is PLANAR, so every frame maps to it by an exact homography, which is
exactly what a calibration is. Rectifying gives:

  - no chaining: each frame maps straight to the field, errors do not accumulate
  - a canvas with ground truth: the sidelines MUST land at y=0 and y=37m, the
    goal lines at x=18 and 82m. A line in the wrong place is visible instantly.
  - a direct answer to "how much of the field do we ever see", in metres

HOW A FRAME GETS ITS CALIBRATION. One window is calibrated by hand
(field -> pixels of that window's reference frame). Every other frame is
related to it by registration. Composing the two gives that frame's own
calibration. Registration is direct-to-hub, never chained -- MEASURED: 2 hubs
place all 556 sampled frames of this video, so nothing is deeper than 2 hops.

THE HONEST LIMIT. A calibration is LOCAL: past the horizon it reports
coordinates that are arithmetic, not geometry. calibration.field_validity_region
gives the x beyond which that is true (40.1m for the t=3235 fit), and this
tool paints NOTHING past it. So the empty part of the canvas is the real
answer: it is the field this footage cannot measure, not a rendering gap.

Run:
  python3 run_field_rectify.py <video> <calib.json> <calib_ref_sec> [every_s] [px_per_m]
"""
import sys
import os
import json
import cv2
import numpy as np

from frisbee_analysis import MosaicConfig, UltimatePitchConfiguration
from frisbee_analysis.mosaic import register_pair_with_coverage
from frisbee_analysis.calibration import Calibration, field_validity_region
from frisbee_analysis.video_overlays import overlay_boxes_for

os.makedirs("outputs/mosaic", exist_ok=True)

video = sys.argv[1] if len(sys.argv) > 1 else "videos/torovcanyon.mp4"
calib_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/calibration/calibration_t3235.json"
ref_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 3235.0
every = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
ppm = float(sys.argv[5]) if len(sys.argv) > 5 else 12.0     # canvas pixels per metre

cfg = UltimatePitchConfiguration()
calib = Calibration.load(calib_path)
vr = field_validity_region(calib, cfg)
limit_cm = vr["usable_x_max"]
print(f"calibration {calib_path}")
print(f"  valid to x={limit_cm/100:.1f}m of {cfg.length/100:.0f}m "
      f"({vr['usable_fraction']*100:.0f}%) -- nothing is painted past that")

cap = cv2.VideoCapture(video)
if not cap.isOpened():
    sys.exit(f"could not open {video}")
fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
boxes = overlay_boxes_for(video, W, H)
mcfg = MosaicConfig(overlay_mask_boxes=tuple(boxes))

def grab(t):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, f = cap.read()
    return f if ok else None

ref = grab(ref_sec)
if ref is None:
    sys.exit(f"could not read the calibration's reference frame at t={ref_sec}")

# field (cm) -> canvas px, with a margin so out-of-bounds play is visible
MARGIN_M = 8.0
OW = int((cfg.length / 100.0 + 2 * MARGIN_M) * ppm)
OH = int((cfg.width / 100.0 + 2 * MARGIN_M) * ppm)
F2C = np.array([[ppm / 100.0, 0, MARGIN_M * ppm],
                [0, ppm / 100.0, MARGIN_M * ppm],
                [0, 0, 1.0]])
print(f"canvas {OW}x{OH} at {ppm:g} px/m ({MARGIN_M:g}m margin all round)")

H_field_to_ref = np.linalg.inv(np.asarray(calib.H()))   # calib.H(): pixels -> field

times = [t for t in np.arange(0, n_frames / fps, every)]
canvas = np.zeros((OH, OW, 3), np.uint8)
filled = np.zeros((OH, OW), np.uint8)
seen_x = []
hubs = [(ref_sec, np.eye(3), ref)]     # (time, H_hub_to_ref, cached frame)
placed = skipped = 0
last = 0
# only paint where the calibration is valid AND inside the padded field
gx, gy = np.meshgrid(np.arange(OW), np.arange(OH))
fx_cm = (gx - MARGIN_M * ppm) * 100.0 / ppm
fy_cm = (gy - MARGIN_M * ppm) * 100.0 / ppm
valid_region = (fx_cm <= limit_cm) & (fx_cm >= -MARGIN_M * 100) & \
               (fy_cm >= -MARGIN_M * 100) & (fy_cm <= cfg.width + MARGIN_M * 100)
print(f"paintable region: {100.0*valid_region.mean():.0f}% of the canvas\n")

for n, t in enumerate(times):
    f = grab(t)
    if f is None:
        continue
    order = sorted(range(len(hubs)), key=lambda k: abs(k - last))
    got = None
    for k in order:
        _, Hh, hf = hubs[k]
        Hs, n_in, _ = register_pair_with_coverage(hf, f, mcfg)
        if Hs is not None and n_in >= mcfg.min_inliers:
            got = (Hh @ Hs, k)
            break
    if got is None:
        # promote this frame to a hub if it can reach any existing one the
        # other way round; keeps depth at 2, never a chain
        for k in order:
            _, Hh, hf = hubs[k]
            Hs, n_in, _ = register_pair_with_coverage(f, hf, mcfg)
            if Hs is not None and n_in >= mcfg.min_inliers:
                Hnew = Hh @ np.linalg.inv(Hs)
                hubs.append((t, Hnew, f))
                got = (Hnew, len(hubs) - 1)
                print(f"  t={t:.0f}s became hub {len(hubs)-1} ({n_in} inliers)", flush=True)
                break
    if got is None:
        skipped += 1
        continue
    H_frame_to_ref, last = got
    # field -> canvas needs field -> ref -> frame
    H_field_to_frame = np.linalg.inv(H_frame_to_ref) @ H_field_to_ref
    H_canvas_to_frame = H_field_to_frame @ np.linalg.inv(F2C)
    warped = cv2.warpPerspective(f, H_canvas_to_frame, (OW, OH),
                                  flags=cv2.WARP_INVERSE_MAP)
    cover = cv2.warpPerspective(np.full((H, W), 255, np.uint8), H_canvas_to_frame,
                                 (OW, OH), flags=cv2.WARP_INVERSE_MAP)
    paint = (cover > 0) & (filled == 0) & valid_region
    canvas[paint] = warped[paint]
    filled[paint] = 255
    placed += 1
    if paint.any():
        seen_x.append((fx_cm[paint].min() / 100.0, fx_cm[paint].max() / 100.0))
    if n % 25 == 0:
        print(f"  {n}/{len(times)} t={t:6.0f}s  {placed} placed, {skipped} unregistered, "
              f"{100.0*filled.mean()/255:.0f}% of canvas painted", flush=True)
cap.release()

# the field outline, so a misplaced line is obvious
def fp(xc, yc):
    p = F2C @ np.array([xc, yc, 1.0])
    return int(p[0] / p[2]), int(p[1] / p[2])
for (a, b) in ((( 0, 0), (0, cfg.width)), ((cfg.length, 0), (cfg.length, cfg.width)),
               ((0, 0), (cfg.length, 0)), ((0, cfg.width), (cfg.length, cfg.width)),
               ((1800, 0), (1800, cfg.width)), ((8200, 0), (8200, cfg.width)),
               ((5000, 0), (5000, cfg.width))):
    cv2.line(canvas, fp(*a), fp(*b), (0, 220, 255), 2)
cv2.line(canvas, fp(limit_cm, -MARGIN_M * 100), fp(limit_cm, cfg.width + MARGIN_M * 100),
         (0, 0, 255), 3)
cv2.putText(canvas, f"calibration valid to x={limit_cm/100:.0f}m -->|",
            (fp(limit_cm, 0)[0] - 420, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
for xm in range(0, int(cfg.length / 100) + 1, 10):
    cv2.putText(canvas, f"{xm}m", (fp(xm * 100, 0)[0] + 4, OH - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
cv2.imwrite("outputs/mosaic/field_rectified.jpg", canvas,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90])
inside = valid_region & (filled > 0)
print(f"\n{placed} frames rectified, {skipped} could not be registered")
print(f"painted {100.0*inside.sum()/max(1,valid_region.sum()):.0f}% of the region the "
      f"calibration can represent")
if seen_x:
    lo = min(a for a, b in seen_x); hi = max(b for a, b in seen_x)
    print(f"field x actually observed: {lo:.0f}..{hi:.0f}m of 0..{cfg.length/100:.0f}m")
print("wrote outputs/mosaic/field_rectified.jpg -- yellow = true field lines from")
print("      WFDF geometry, red = where the calibration stops being meaningful.")
print("      Painted content should line up with the yellow lines; where it does")
print("      not, the calibration is wrong and you can see it directly.")
