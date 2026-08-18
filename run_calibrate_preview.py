"""
Renders a video clip with the calibrated field outline overlaid on every
frame of the segment, not just the single reference frame
run_calibrate.py's static overlay uses -- lets you actually watch whether
the calibration holds up as the camera pans/zooms across the segment,
instead of trusting one static snapshot.

For each frame, composes that frame's own homography (frame pixels ->
segment reference, from register_sequence) with the calibration's homography
(segment reference -> field) to get a frame-specific field-to-pixel
transform, then reuses draw_field_overlay's logic via a temporary
Calibration standing in for that composed transform.

Needs the SAME video/window (start_sec, duration_sec, sample_fps) used for
the calibration, so the same segment gets rebuilt -- the script finds which
segment actually contains the calibration's own source frames and uses that
one automatically, so a slightly different duration_sec is fine as long as
the window still covers those frames.

Run:  python run_calibrate_preview.py <video_path> <calibration.json> <start_sec> <duration_sec> [sample_fps] [out.mp4]
Needs opencv-python (Track B deps, see requirements.txt).
"""
import sys
import cv2
import numpy as np
from frisbee_analysis import UltimatePitchConfiguration, MosaicConfig, register_sequence
from frisbee_analysis.calibration import Calibration, draw_field_overlay

video_path = sys.argv[1] if len(sys.argv) > 1 else "videos/videoplayback.mp4"
calib_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/calibration.json"
start_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 1158.0
duration_sec = float(sys.argv[4]) if len(sys.argv) > 4 else 12.0
sample_fps = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
out_path = sys.argv[6] if len(sys.argv) > 6 else "outputs/calibration_preview.mp4"

calib = Calibration.load(calib_path)
cfg = UltimatePitchConfiguration()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    sys.exit(f"could not open {video_path}")
_frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
_frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if (_frame_w, _frame_h) == (1920, 1080):
    OVERLAY_BOXES = [(0, 0, 1920, 230), (0, 860, 1920, 950)]
elif (_frame_w, _frame_h) == (640, 360):
    OVERLAY_BOXES = [(0, 295, 640, 360)]
else:
    OVERLAY_BOXES = []

video_fps = cap.get(cv2.CAP_PROP_FPS)
step = max(1, int(round(video_fps / sample_fps)))
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

mosaic_cfg = MosaicConfig(overlay_mask_boxes=tuple(OVERLAY_BOXES))
regs = register_sequence(frames, mosaic_cfg)
homography_by_idx = {r.frame_idx: r.homography for r in regs if r.ok}
segment_of_frame = {r.frame_idx: r.segment_id for r in regs if r.ok}

# find which segment actually contains the calibration's own source frames --
# so a slightly different duration_sec than the original calibration run
# still works, as long as the window covers those frames
calib_frames = set(calib.source_frame_indices or [])
if not calib_frames and calib.line_details:
    calib_frames = {fi for d in calib.line_details for fi in (d["frame_indices"] or [])}
matching_segs = {segment_of_frame[fi] for fi in calib_frames if fi in segment_of_frame}
if not matching_segs:
    sys.exit("none of the calibration's source frames registered in this window -- "
              "use the same start_sec/duration_sec/sample_fps as the calibration run")
seg_id = sorted(matching_segs)[0]
seg_idx = [fi for fi, sid in segment_of_frame.items() if sid == seg_id]
seg_idx.sort()
print(f"Using segment {seg_id}: {len(seg_idx)} frames (contains {len(matching_segs & {seg_id})} "
      f"of the calibration's own source frames)")

h, w = frames[0].shape[:2]
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), sample_fps, (w, h))
for fi in seg_idx:
    # compose: this frame's pixels -> segment reference -> field, i.e. a
    # frame-specific stand-in Calibration whose H is (calib.H() @ frame_H)
    frame_calib = Calibration(homography=(calib.H() @ homography_by_idx[fi]).tolist(),
                               keypoint_labels=[], image_points=[], field_points=[],
                               image_path=calib.image_path)
    overlay = draw_field_overlay(frames[fi], frame_calib, cfg)
    writer.write(overlay)
writer.release()
print(f"Saved {len(seg_idx)}-frame preview to {out_path}")
