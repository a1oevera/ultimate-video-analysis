"""
Mosaic-registration pipeline test: runs frisbee_analysis.mosaic on a real clip
and reports how many frames registered successfully vs. needed interpolation.

This is the reusable version of the ad hoc feasibility test that produced
results/mosaic_finding.md -- that answered "is this possible at all" (yes,
for wide frames); this script exercises the actual pipeline code
(register_sequence + interpolate_missing) end to end on a sustained stretch
of frames, the way it would really run.

The overlay mask boxes below are specific to THIS footage's broadcast bug
(ROGERS tv scoreboard top-left, watermark top-right) -- adjust or clear them
for footage without that overlay.

Run:  python run_mosaic_test.py <video_path> [start_sec] [duration_sec] [sample_fps]
Needs opencv-python (Track B deps, see requirements.txt).
"""
import sys
import cv2
import numpy as np
from frisbee_analysis import MosaicConfig, register_sequence, interpolate_missing

video_path = sys.argv[1] if len(sys.argv) > 1 else "videos/ojuc.mp4"
start_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
duration_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0
sample_fps = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    sys.exit(f"could not open {video_path}")

# Static broadcast overlay position depends on the source video -- MEASURED BUG
# (see results/calibration_finding.md): a hardcoded-for-one-video overlay box
# silently does the wrong thing against a different video's resolution, and
# can corrupt registration by leaving a real overlay unmasked.
_frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
_frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if (_frame_w, _frame_h) == (1920, 1080):
    OVERLAY_BOXES = [(0, 0, 700, 150), (1750, 0, 1920, 120), (0, 860, 1920, 950)]  # ojuc.mp4: scoreboard, watermark, captions
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
print(f"Loaded {len(frames)} frames from {video_path} "
      f"(t={start_sec:.0f}-{start_sec + duration_sec:.0f}s, ~{sample_fps} fps sampling)")

cfg = MosaicConfig(overlay_mask_boxes=tuple(OVERLAY_BOXES))
results = register_sequence(frames, cfg)
n_ok = sum(1 for r in results if r.ok)
n_fail = len(results) - n_ok
inlier_counts = [r.n_inliers for r in results[1:]]  # skip the reference frame's placeholder count

print(f"\nregistered:   {n_ok}/{len(results)} frames directly (>= {cfg.min_inliers} inliers)")
print(f"failed:       {n_fail}/{len(results)} frames (tight close-up, or no shared background at all)")
if inlier_counts:
    print(f"inlier count: mean={np.mean(inlier_counts):.1f}  "
          f"median={np.median(inlier_counts):.1f}  min={min(inlier_counts)}  max={max(inlier_counts)}")

segment_sizes = {}
for r in results:
    if r.ok:
        segment_sizes[r.segment_id] = segment_sizes.get(r.segment_id, 0) + 1
print(f"\nsegments: {len(segment_sizes)} (frames only share a coordinate system within a "
      f"segment -- a new one starts wherever the anchor goes stale but the frame still "
      f"matches its immediate predecessor, e.g. a broadcast cut into continuous gameplay)")
for sid, size in sorted(segment_sizes.items()):
    print(f"  segment {sid}: {size} frames")

filled = interpolate_missing(results)
still_missing = sum(1 for r in filled if r.homography is None)
print(f"\nafter interpolate_missing: {len(filled) - still_missing}/{len(filled)} frames have a usable homography")
if still_missing:
    print(f"  ({still_missing} frames have NO usable homography -- a failure run with no "
          f"nearby successfully-registered anchor on either side)")

# print the longest run of consecutive failures -- the case interpolation is weakest for
run, best_run, run_start, best_start = 0, 0, 0, 0
for i, r in enumerate(results):
    if not r.ok:
        if run == 0:
            run_start = i
        run += 1
        if run > best_run:
            best_run, best_start = run, run_start
    else:
        run = 0
if best_run:
    print(f"\nlongest consecutive-failure run: {best_run} frames "
          f"(indices {best_start}-{best_start + best_run - 1}, "
          f"~{best_run / sample_fps:.1f}s of tight-zoom coverage)")
