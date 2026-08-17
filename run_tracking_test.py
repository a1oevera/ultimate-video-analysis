"""
B3 "next" step: track detections across frames instead of just per-frame
boxes. Runs YOLO (same imgsz=1920 fix as run_player_detection_test.py --
default settings miss the whole field of play, see results/detection_finding.md)
+ ByteTrack (via `supervision`) over a real window of footage, and reports
track persistence/fragmentation, not just a per-frame detection count.

Run:  python run_tracking_test.py <video_path> <start_sec> <duration_sec> [sample_fps] [out.jpg]
Needs opencv-python + ultralytics + supervision (Track B deps).
"""
import sys
import warnings
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

video_path = sys.argv[1] if len(sys.argv) > 1 else "videos/ojuc.mp4"
start_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 320.0
duration_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
sample_fps = float(sys.argv[4]) if len(sys.argv) > 4 else 8.0
out_path = sys.argv[5] if len(sys.argv) > 5 else "outputs/tracking_sample.jpg"

cap = cv2.VideoCapture(video_path)
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

model = YOLO("yolo11n.pt")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)  # sv.ByteTrack deprecated as of supervision 0.30.0, still functional
    tracker = sv.ByteTrack()

track_frame_counts = {}   # track_id -> number of frames it appeared in
track_first_seen = {}
track_last_seen = {}
per_frame_counts = []
annotated_last = None

for i, frame in enumerate(frames):
    results = model.predict(frame, classes=[0], conf=0.25, imgsz=1920, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    per_frame_counts.append(len(detections))
    for tid in detections.tracker_id:
        track_frame_counts[tid] = track_frame_counts.get(tid, 0) + 1
        track_first_seen.setdefault(tid, i)
        track_last_seen[tid] = i
    if i == len(frames) - 1:
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        labels = [f"#{tid}" for tid in detections.tracker_id]
        annotated_last = box_annotator.annotate(frame.copy(), detections)
        annotated_last = label_annotator.annotate(annotated_last, detections, labels=labels)

print(f"\nPer-frame detection count: mean={np.mean(per_frame_counts):.1f} "
      f"min={min(per_frame_counts)} max={max(per_frame_counts)}")

n_tracks = len(track_frame_counts)
lifespans = list(track_frame_counts.values())
span_frames = [track_last_seen[t] - track_first_seen[t] + 1 for t in track_frame_counts]
print(f"\nUnique track IDs assigned: {n_tracks}")
print(f"Frames each track was actually detected in: mean={np.mean(lifespans):.1f} "
      f"median={np.median(lifespans):.0f} max={max(lifespans)}")
print(f"Track ID span (last-first+1, includes gaps): mean={np.mean(span_frames):.1f} "
      f"median={np.median(span_frames):.0f} max={max(span_frames)}")

single_frame_tracks = sum(1 for v in lifespans if v == 1)
print(f"\nSingle-frame tracks (never re-matched -- fragmentation signal): "
      f"{single_frame_tracks}/{n_tracks} ({100*single_frame_tracks/n_tracks:.0f}%)")

full_span_tracks = sum(1 for v in span_frames if v >= len(frames) * 0.8)
print(f"Tracks spanning >=80% of the window (stable): {full_span_tracks}/{n_tracks}")

if annotated_last is not None:
    cv2.imwrite(out_path, annotated_last)
    print(f"\nSaved last frame with track ID labels to {out_path}")
