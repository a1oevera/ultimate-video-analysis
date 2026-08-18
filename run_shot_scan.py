"""
Scan the whole video for camera cuts, then cluster those cuts into DISTINCT
recurring framings -- answers "how many shots actually need their own
calibration", not just "how many cuts" (which would also count every quick
replay/interstitial the broadcast throws in).

Two passes, both reusing mosaic.py's register_pair (no new registration
logic):
  1. Segment (cut) detection: same incremental logic as register_sequence
     (last-good-anchor, fall back to previous-frame), coarsely sampled
     (default every 5s) since we only care about SUSTAINED gameplay shots,
     not fast production cuts between them.
  2. Clustering: for each detected segment's anchor frame, try matching it
     against every cluster's representative frame found SO FAR (same
     match-or-new-cluster logic as calibration_bank.py's try_auto_calibrate,
     just tracking membership instead of producing a calibration). A segment
     that matches an existing cluster is the SAME recurring camera framing
     seen before; one that matches nothing starts a new cluster.

Uses sequential grab()-skip instead of repeated cv2.set() seeks -- much
faster for scanning a long video than the seek-per-sample pattern used
elsewhere in this project for short windows.

Run:  python run_shot_scan.py <video_path> [sample_every_sec] [max_minutes] [start_min]
Needs opencv-python.
"""
import sys
import time
import cv2
from frisbee_analysis import MosaicConfig, register_pair, looks_like_field_view

video_path = sys.argv[1] if len(sys.argv) > 1 else "videos/videoplayback.mp4"
sample_every_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
max_minutes = float(sys.argv[3]) if len(sys.argv) > 3 else None
start_min = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    sys.exit(f"could not open {video_path}")
fps = cap.get(cv2.CAP_PROP_FPS)
n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
start_s = start_min * 60
duration_s = n_frames_total / fps - start_s
if max_minutes:
    duration_s = min(duration_s, max_minutes * 60)
if start_s > 0:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))

_w, _h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if (_w, _h) == (1920, 1080):
    OVERLAY_BOXES = [(0, 0, 1920, 230), (0, 860, 1920, 950)]
elif (_w, _h) == (640, 360):
    OVERLAY_BOXES = [(0, 295, 640, 360)]
else:
    OVERLAY_BOXES = []
cfg = MosaicConfig(overlay_mask_boxes=tuple(OVERLAY_BOXES))

step_frames = max(1, int(round(sample_every_sec * fps)))
print(f"{video_path}: sampling every {sample_every_sec}s over {duration_s/60:.1f} min "
      f"(~{int(duration_s/sample_every_sec)} samples)")

samples = []  # (t_sec, frame)
frame_idx = 0
next_sample_frame = 0
last_frame_idx = int(duration_s * fps)
t0 = time.time()
while frame_idx < last_frame_idx:
    ok = cap.grab()
    if not ok:
        break
    if frame_idx >= next_sample_frame:
        ok2, frame = cap.retrieve()
        if ok2:
            samples.append((start_s + frame_idx / fps, frame))
        next_sample_frame += step_frames
    frame_idx += 1
cap.release()
print(f"Collected {len(samples)} sample frames in {time.time()-t0:.0f}s")

# --- pass 1: segment (cut) detection ---
t1 = time.time()
segments = [(0, samples[0][1], samples[0][0])]
last_good = 0
for i in range(1, len(samples)):
    _, n_inl = register_pair(samples[last_good][1], samples[i][1], cfg)
    if n_inl >= cfg.min_inliers:
        last_good = i
        continue
    _, n_local = register_pair(samples[i - 1][1], samples[i][1], cfg)
    if n_local < cfg.min_inliers:
        segments.append((i, samples[i][1], samples[i][0]))
    last_good = i
print(f"\nRaw camera cuts (distinct continuous shots): {len(segments)}  ({time.time()-t1:.0f}s)")

# MEASURED: a big chunk of raw cuts are ground-level/close-up/crowd shots
# that can never be usefully calibrated anyway (see frisbee_analysis/
# shot_filter.py, results/shot_scan_finding.md) -- filter those out before
# clustering, both to cut wasted registration work AND because this IS the
# "skip non-gameplay shots" filter discussed earlier: not a separate feature,
# the same signal serves both purposes.
field_segments = [(i, f, t) for i, f, t in segments if looks_like_field_view(f)]
print(f"Field-view candidates (grass-heuristic filter): {len(field_segments)}/{len(segments)} "
      f"-- the rest are presumed ground-level/close-up/crowd cuts, not worth calibrating")

# --- pass 2: cluster into distinct recurring framings ---
t2 = time.time()
clusters = []
for start_i, frame, t_sec in field_segments:
    matched = None
    for c in clusters:
        _, n_inl = register_pair(c["frame"], frame, cfg)
        if n_inl >= cfg.min_inliers:
            matched = c
            break
    if matched:
        matched["members"].append(t_sec)
    else:
        clusters.append({"frame": frame, "members": [t_sec]})
print(f"Distinct recurring camera framings (clusters): {len(clusters)}  ({time.time()-t2:.0f}s)")

for i, c in enumerate(clusters):
    times = ", ".join(f"{int(t//60)}:{t%60:04.1f}" for t in c["members"][:6])
    more = f" (+{len(c['members'])-6} more)" if len(c["members"]) > 6 else ""
    print(f"  cluster {i}: {len(c['members'])} segment(s) -- first seen at {times}{more}")
    cv2.imwrite(f"outputs/shot_cluster_{i}.jpg", c["frame"])

print(f"\nTotal: {time.time()-t0:.0f}s. Saved one representative frame per cluster to "
      f"outputs/shot_cluster_*.jpg -- worth eyeballing before assuming they're all real "
      f"gameplay shots (some could be ads/crowd/replays, same caveat as everything else "
      f"in this project).")
