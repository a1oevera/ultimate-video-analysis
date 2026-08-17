"""
Interactive manual field calibration: click the visible calibration keypoints
(cones + brick marks, from frisbee_analysis/ultimate_config.py) on one image
-- typically a mosaic built by run_mosaic_visualize.py, since it aggregates
visibility across a whole segment (see frisbee_analysis/calibration.py's
module docstring for why off-camera keypoints don't block this).

NEEDS A REAL DISPLAY -- run this in your own terminal, not headless/over SSH
without X forwarding. Opens a GUI window.

Controls:
  left-click  mark the current keypoint at that pixel
  s           skip the current keypoint (it's off-camera in this image --
              fine, you only need >=4 total, not all 10)
  u           undo the last click
  q           finish once you have enough points (need >= 4, more is better,
              and prefer them spread across the field over clustered)

Run:  python run_calibrate.py <image_path> [out_calibration.json]
Needs opencv-python (Track B deps, see requirements.txt).
"""
import sys
import cv2
from frisbee_analysis import UltimatePitchConfiguration
from frisbee_analysis.calibration import fit_calibration, draw_field_overlay

image_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/mosaic_sample.jpg"
out_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/calibration.json"

img = cv2.imread(image_path)
if img is None:
    sys.exit(f"could not read {image_path}")

cfg = UltimatePitchConfiguration()
KEYPOINT_LABELS = {
    1: "cone: near-left back corner (closest to camera -- often off-camera)",
    2: "cone: near-right back corner (closest to camera -- often off-camera)",
    3: "cone: near-left goal corner",
    4: "cone: near-right goal corner",
    5: "cone: far-left goal corner",
    6: "cone: far-right goal corner",
    7: "cone: far-left back corner",
    8: "cone: far-right back corner",
    9: "brick mark: near",
    10: "brick mark: far",
}
order = list(cfg.CALIBRATION_INDICES)

# display at a manageable size if the source is large (e.g. a wide mosaic);
# clicks are mapped back to ORIGINAL image coordinates before saving
max_w = 1600
scale = min(1.0, max_w / img.shape[1])
disp_base = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1.0 else img.copy()

clicked = {}       # keypoint_index -> (x, y) in ORIGINAL image coords
current = [0]       # mutable index into `order`


def redraw():
    vis = disp_base.copy()
    for idx, (x, y) in clicked.items():
        cv2.circle(vis, (int(x * scale), int(y * scale)), 6, (0, 255, 0), -1)
        cv2.putText(vis, str(idx), (int(x * scale) + 8, int(y * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if current[0] < len(order):
        label = f"Click: {KEYPOINT_LABELS[order[current[0]]]}  ({len(clicked)} placed so far, need >=4)"
    else:
        label = f"All prompted -- {len(clicked)} placed. Press 'q' to finish."
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(vis, "click=mark  s=skip  u=undo  q=finish",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("calibrate", vis)


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and current[0] < len(order):
        clicked[order[current[0]]] = (x / scale, y / scale)
        current[0] += 1
        redraw()


cv2.namedWindow("calibrate")
cv2.setMouseCallback("calibrate", on_mouse)
redraw()
while True:
    key = cv2.waitKey(20) & 0xFF
    if key == ord('s') and current[0] < len(order):
        current[0] += 1
        redraw()
    elif key == ord('u'):
        if current[0] > 0:
            current[0] -= 1
            clicked.pop(order[current[0]], None)
            redraw()
    elif key == ord('q'):
        break
cv2.destroyAllWindows()

if len(clicked) < 4:
    sys.exit(f"only {len(clicked)} points placed -- need >= 4 for a homography. Re-run and click more.")

indices = list(clicked.keys())
points = [clicked[i] for i in indices]
calib = fit_calibration(points, indices, image_path, cfg)
calib.save(out_path)
print(f"Saved calibration ({len(points)} points: {indices}) to {out_path}")

overlay = draw_field_overlay(img, calib, cfg)
overlay_path = out_path.rsplit(".", 1)[0] + "_overlay.jpg"
cv2.imwrite(overlay_path, overlay)
print(f"Saved field-outline overlay (sanity check -- includes any off-camera / "
      f"estimated regions like a skipped endzone) to {overlay_path}")
