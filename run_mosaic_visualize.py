"""
Builds an actual stitched mosaic image from real footage, using
frisbee_analysis.mosaic's registration, so alignment can be checked visually
instead of just trusting inlier counts.

Run:  python run_mosaic_visualize.py <video_path> <out.jpg> [start_sec] [duration_sec] [sample_fps]
Needs opencv-python (Track B deps, see requirements.txt).
"""
import sys
import cv2
import numpy as np
from frisbee_analysis import MosaicConfig, register_sequence, interpolate_missing

video_path = sys.argv[1] if len(sys.argv) > 1 else "videos/ojuc.mp4"
out_path = sys.argv[2] if len(sys.argv) > 2 else "mosaic.jpg"
start_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 270.0
duration_sec = float(sys.argv[4]) if len(sys.argv) > 4 else 300.0
sample_fps = float(sys.argv[5]) if len(sys.argv) > 5 else 0.2

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    sys.exit(f"could not open {video_path}")

# Static broadcast overlay position depends on the source video -- MEASURED BUG
# (see results/calibration_finding.md): a hardcoded-for-one-video overlay box
# silently does the wrong thing (or nothing) against a different video's
# resolution, and can corrupt registration by leaving a real overlay unmasked.
# Full-width top band for ojuc.mp4: a second, larger semi-transparent
# "ROGERS tv" watermark shows up intermittently higher up and wider than the
# crisp corner logo -- a small box missed it. y=230 stays above the treeline
# in that footage's wide shots.
_frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
_frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if (_frame_w, _frame_h) == (1920, 1080):
    OVERLAY_BOXES = [(0, 0, 1920, 230), (0, 860, 1920, 950)]  # ojuc.mp4: top band, captions band
elif (_frame_w, _frame_h) == (640, 360):
    OVERLAY_BOXES = [(0, 295, 640, 360)]  # videoplayback.mp4: UFA score bug + ticker, bottom band
else:
    OVERLAY_BOXES = []
    print(f"WARNING: no known overlay mask for {_frame_w}x{_frame_h} -- add one if this "
          f"video has a static broadcast graphic, or registration may be corrupted by it.")

video_fps = cap.get(cv2.CAP_PROP_FPS)
step = int(round(video_fps / sample_fps))
start_frame = int(round(start_sec * video_fps))
n_samples = int(duration_sec * sample_fps)

frames = []
for k in range(n_samples):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + k * step)
    ok, frame = cap.read()
    if not ok:
        break
    frames.append(frame)
cap.release()
print(f"Loaded {len(frames)} frames (t={start_sec:.0f}-{start_sec + duration_sec:.0f}s, ~{sample_fps} fps)")

cfg = MosaicConfig(overlay_mask_boxes=tuple(OVERLAY_BOXES))
results = interpolate_missing(register_sequence(frames, cfg))

# use the largest segment -- frames only share a coordinate system within one
segment_sizes = {}
for r in results:
    if r.ok:
        segment_sizes[r.segment_id] = segment_sizes.get(r.segment_id, 0) + 1
best_seg = max(segment_sizes, key=segment_sizes.get)
seg_results = [r for r in results if r.homography is not None and r.segment_id == best_seg]
print(f"Using segment {best_seg}: {len(seg_results)} frames")

h, w = frames[0].shape[:2]
corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
all_pts = []
for r in seg_results:
    all_pts.append(cv2.perspectiveTransform(corners, r.homography))
all_pts = np.concatenate(all_pts, axis=0).reshape(-1, 2)
min_x, min_y = all_pts.min(axis=0)
max_x, max_y = all_pts.max(axis=0)
canvas_w, canvas_h = int(np.ceil(max_x - min_x)), int(np.ceil(max_y - min_y))
print(f"Canvas size: {canvas_w}x{canvas_h} (vs single frame {w}x{h})")

# same static-overlay regions as OVERLAY_BOXES, but this time excluded from
# the COMPOSITED pixels, not just from feature detection -- otherwise the
# scoreboard/watermark (camera-static, not world-static) gets warped into a
# different position in world coordinates per frame and shows up as ghosted
# duplicates in the mosaic (caught by eyeballing the output, not by the
# inlier-count metric, which only ever looked at registration quality).
source_mask = np.ones((h, w), dtype=np.float32)
for (x0, y0, x1, y1) in OVERLAY_BOXES:
    source_mask[y0:y1, x0:x1] = 0

T = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64)
canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

for r, frame in zip(seg_results, [frames[r.frame_idx] for r in seg_results]):
    H = T @ r.homography
    warped = cv2.warpPerspective(frame, H, (canvas_w, canvas_h))
    mask = cv2.warpPerspective(source_mask, H, (canvas_w, canvas_h))
    canvas += warped.astype(np.float32) * mask[..., None]
    weight += mask

weight[weight == 0] = 1
mosaic = (canvas / weight[..., None]).astype(np.uint8)
cv2.imwrite(out_path, mosaic)
print(f"Saved mosaic to {out_path}")
