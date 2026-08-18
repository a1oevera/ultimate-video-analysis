"""
B5 first pass: combine everything built so far -- registration (mosaic.py),
calibration (calibration.py), detection+short-term tracking (YOLO+ByteTrack),
and team classification (saturation split) -- into actual team-level numbers:
on-field player counts per frame, and speed estimates in real units (m/s).

Per-player identity across the WHOLE window is NOT attempted (measured badly
fragmented, see results/tracking_finding.md) -- but ByteTrack's short-term
correspondence between ADJACENT sampled frames is exactly what's needed for
instantaneous speed: only consecutive-frame track matches are used, never a
track's full lifespan. This sidesteps the fragmentation problem instead of
solving it, per the B2/B4 descope decision (see NEXT_STEPS.md).

Per frame:
  1. this frame's own homography into the segment reference (mosaic.py,
     already computed for calibration) composed with the calibration's own
     homography (segment reference -> field cm) -- same composition
     run_calibrate_preview.py uses for the overlay video.
  2. YOLO person detection (imgsz=1920 -- see results/detection_finding.md,
     the 640px default misses the whole field) + ByteTrack, minus a size-
     sanity filter that drops cone/marker false positives (see below).
  3. each detection's foot point (bottom-center of the box -- approximates
     ground contact, unlike the box centroid) projected to field cm via the
     composed homography, then filtered with is_in_field(margin_cm=0) --
     drops sideline/bench people (see Addendum 3 in results/metrics_finding.md
     on why the margin was tightened to 0) and, per calibration_finding.md's
     wrong-field warning, anyone who ends up outside THIS field's boundary.
  4. team classified via GLOBAL (whole-window, not per-frame) torso
     BRIGHTNESS (HSV value, not saturation -- see torso_brightness()'s
     docstring for why saturation doesn't reliably separate white from dark
     jerseys here) threshold, found via Otsu's method over every in-field
     detection's brightness pooled across the whole window -- a per-frame
     median (the original approach) silently assumes each frame shows a
     ~50/50 split of both teams, which measurably misclassified the
     minority-represented team in imbalanced frames (see results/
     metrics_finding.md). The torso crop itself also excludes any columns
     covered by ANOTHER nearby detection's box, to reduce an occluding/
     overlapping player's jersey color bleeding into the sample.

Then: for each ByteTrack id seen in TWO CONSECUTIVE sampled frames, the
field-space displacement / dt gives one instantaneous speed sample: bucket
by team, report mean/median speed and total on-field player-frames per team
-- explicitly NOT per-player distance (would need real identity).

Run:  python run_metrics_test.py <video_path> <calib.json> <start_sec> <duration_sec> [sample_fps] [out.jpg]
Needs opencv-python + ultralytics + supervision (Track B deps).
"""
import sys
import warnings
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from frisbee_analysis import UltimatePitchConfiguration, MosaicConfig, register_sequence
from frisbee_analysis.calibration import Calibration, pixel_to_field, is_in_field

video_path = sys.argv[1] if len(sys.argv) > 1 else "videos/videoplayback.mp4"
calib_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/calibration.json"
start_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 1158.0
duration_sec = float(sys.argv[4]) if len(sys.argv) > 4 else 12.0
sample_fps = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
out_path = sys.argv[6] if len(sys.argv) > 6 else "outputs/metrics_sample.jpg"

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
frame_times = []
for k in range(n_samples):
    target_frame = start_frame + k * step
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame = cap.read()
    if not ok:
        break
    frames.append(frame)
    actual = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
    frame_times.append((actual if actual > 0 else target_frame) / video_fps)
cap.release()
print(f"Loaded {len(frames)} frames (t={start_sec:.0f}-{start_sec + duration_sec:.0f}s, ~{sample_fps} fps)")

mosaic_cfg = MosaicConfig(overlay_mask_boxes=tuple(OVERLAY_BOXES))
regs = register_sequence(frames, mosaic_cfg)
homography_by_idx = {r.frame_idx: r.homography for r in regs if r.ok}
segment_of_frame = {r.frame_idx: r.segment_id for r in regs if r.ok}

calib_frames = set(calib.source_frame_indices or [])
if not calib_frames and calib.line_details:
    calib_frames = {fi for d in calib.line_details for fi in (d["frame_indices"] or [])}
matching_segs = {segment_of_frame[fi] for fi in calib_frames if fi in segment_of_frame}
if not matching_segs:
    sys.exit("none of the calibration's source frames registered in this window -- "
              "use the same start_sec/duration_sec/sample_fps as the calibration run")
seg_id = sorted(matching_segs)[0]
seg_idx = sorted(fi for fi, sid in segment_of_frame.items() if sid == seg_id)
print(f"Using segment {seg_id}: {len(seg_idx)} frames")


def torso_brightness(frame, box, other_boxes):
    """Median HSV VALUE (brightness) of a box's torso region (middle 50%
    width, middle third height -- avoids head/hair and legs/grass at the box
    edges).

    MEASURED (person report): saturation (the original signal) does NOT
    reliably separate white from dark jerseys on this footage -- a dark/
    near-black jersey reads as LOW saturation too (a deep, nearly-neutral
    color isn't necessarily vivid), so on real frames its saturation
    overlapped white's enough to get misclassified as the white team.
    Direct measurement on one frame: saturation ranged 1-56 with heavy
    overlap between the two teams, but VALUE cleanly split 55-131 (dark
    jerseys) vs 219-255 (white), a real gap. Switched the whole
    classification to brightness for this reason -- true for a white-vs-dark
    matchup like this one; a colored-vs-colored matchup would need hue
    clustering instead (see results/team_classify_finding.md), not this.

    MEASURED (person report): when players overlap, the naive crop can also
    sample a NEIGHBORING player's jersey instead of this one's. Mitigation:
    exclude any torso columns that fall inside another nearby detection's box
    (in the same vertical band) before taking the median -- best-effort, not
    a full occlusion solve. Requires a real amount of unoccluded width left
    (60%, at least 6px) before trusting the filtered crop; a first version of
    this used a looser 30%-of-any-size floor, which let two heavily-
    overlapping boxes (near-duplicate detections a few px apart) reduce each
    other to unreliable few-pixel slivers -- caught by comparing two supposed
    detections of the same area giving wildly different values (median
    saturation 1 vs 56) before this fix."""
    x1, y1, x2, y2 = box.astype(int)
    h, w = y2 - y1, x2 - x1
    tx1, tx2 = x1 + w // 4, x2 - w // 4
    ty1, ty2 = y1 + h // 4, y1 + h // 2
    tx1, tx2 = max(tx1, 0), max(tx2, tx1 + 1)
    ty1, ty2 = max(ty1, 0), max(ty2, ty1 + 1)

    col_excluded = np.zeros(tx2 - tx1, dtype=bool)
    for ob in other_boxes:
        ox1, oy1, ox2, oy2 = ob
        if oy2 <= ty1 or oy1 >= ty2:
            continue  # other box doesn't vertically overlap the torso band
        lo, hi = max(int(ox1), tx1), min(int(ox2), tx2)
        if lo < hi:
            col_excluded[lo - tx1:hi - tx1] = True
    keep_cols = np.where(~col_excluded)[0]
    width = tx2 - tx1

    if len(keep_cols) >= max(0.6 * width, 6):
        crop = frame[ty1:ty2, tx1:tx2][:, keep_cols]
    else:
        crop = frame[ty1:ty2, tx1:tx2]  # too occluded to filter -- best effort
    if crop.size == 0:
        crop = frame[y1:y2, x1:x2]
    val = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 2]
    return float(np.median(val))


model = YOLO("yolo11n.pt")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    tracker = sv.ByteTrack()

# Pass 1: detect + track + geometry, but DON'T assign team yet -- team needs
# a threshold pooled across the whole window (see module docstring).
# per-frame list of dicts: box, tid (or None), fx/fy (field cm), sat
frame_records = {}

for fi in seg_idx:
    frame = frames[fi]
    results = model.predict(frame, classes=[0], conf=0.25, imgsz=1920, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    # MEASURED BUG: sv.ByteTrack.update_with_detections DROPS any detection it
    # can't confidently match to an already-existing track -- a first-seen (or
    # re-appearing after being lost) player with score in [0.25, det_thresh)
    # (det_thresh = track_activation_threshold + 0.1 = 0.35 by default) is
    # discarded entirely, never even getting a track id (supervision 0.30.0
    # core.py "Step 4: Init new stracks", `if track.score < self.det_thresh:
    # continue`). Only the SPEED estimate actually needs a track id -- on-field
    # counting and team classification don't. So: keep the RAW detections for
    # counting/classification, and only consult `tracked` (a same-order
    # SUBSET of the raw boxes) to look up a track id per box where one exists,
    # for speed matching. This is what fixed a real undercount (was reporting
    # 3 on-field players in a frame with 9 visible).
    tracked = tracker.update_with_detections(detections)
    track_id_by_box = {tuple(np.round(box, 3)): int(tid)
                        for box, tid in zip(tracked.xyxy, tracked.tracker_id)}

    frame_to_field = calib.H() @ homography_by_idx[fi]
    frame_calib = Calibration(homography=frame_to_field.tolist(), keypoint_labels=[],
                               image_points=[], field_points=[], image_path=calib.image_path)

    boxes_raw = detections.xyxy
    # MEASURED: fixing the ByteTrack drop above (see comment) un-hid a real
    # false positive it had been coincidentally filtering out -- a 7x13px box
    # (conf 0.34, squarely in the [0.25, 0.35) band ByteTrack used to eat) on
    # what a visual check confirmed is a field marker/cone, not a person, in
    # a frame where every real player's box was >=32px tall. Cheap size-
    # sanity filter: drop boxes under 40% of the frame's own median detection
    # height (only when there are enough detections, >=3, for that median to
    # be meaningful) -- a real player at any depth on this footage is still
    # roughly proportionate to the frame's other players, a cone isn't.
    if len(boxes_raw) >= 3:
        heights = boxes_raw[:, 3] - boxes_raw[:, 1]
        boxes = boxes_raw[heights >= 0.4 * np.median(heights)]
    else:
        boxes = boxes_raw

    records = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        foot_x, foot_y = (x1 + x2) / 2.0, y2
        # MEASURED (person report): a 100cm outward margin let sideline/bench
        # people (who stand close to the boundary, not far off it) get
        # counted as on-field. Tightened to 0 -- no tolerance past the fitted
        # boundary. Trade-off: a real player exactly on the line could drop
        # out on foot-point/calibration noise; that's the safer direction
        # than counting bystanders as players.
        if not is_in_field(foot_x, foot_y, frame_calib, cfg, margin_cm=0.0):
            continue
        fx, fy = pixel_to_field(foot_x, foot_y, frame_calib)
        other_boxes = np.delete(boxes, i, axis=0)
        val = torso_brightness(frame, box, other_boxes)
        tid = track_id_by_box.get(tuple(np.round(box, 3)))
        records.append({"box": box, "tid": tid, "fx": fx, "fy": fy, "val": val})
    frame_records[fi] = records

# GLOBAL team threshold: pool every in-field detection's brightness across
# the whole window and split with Otsu's method (finds the threshold that
# best separates a bimodal distribution) instead of each frame's own median
# (a per-frame threshold silently assumes each frame shows a ~50/50 split of
# both teams, which measurably misclassified the minority-represented team
# in imbalanced frames).
all_vals = np.array([r["val"] for recs in frame_records.values() for r in recs])
if len(all_vals) >= 4:
    val_u8 = np.clip(all_vals, 0, 255).astype(np.uint8).reshape(-1, 1)
    val_thresh, _ = cv2.threshold(val_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
else:
    val_thresh = float(np.median(all_vals)) if len(all_vals) else 0.0
print(f"\nGlobal team brightness threshold (Otsu, over {len(all_vals)} pooled in-field "
      f"detections): {val_thresh:.1f}")

# Pass 2: assign team with the global threshold, build outputs.
per_frame_tracks = []
per_frame_time = []
on_field_counts = {"A": [], "B": []}
annotated_last = None

for fi in seg_idx:
    frame_track_map = {}
    n_a = n_b = 0
    vis = frames[fi].copy() if fi == seg_idx[-1] else None
    for r in frame_records[fi]:
        team = "A" if r["val"] >= val_thresh else "B"  # A = brighter/white jersey, B = darker
        if r["tid"] is not None:
            frame_track_map[r["tid"]] = (r["fx"], r["fy"], team)
        if team == "A":
            n_a += 1
        else:
            n_b += 1
        if vis is not None:
            x1, y1, x2, y2 = r["box"]
            color = (255, 100, 0) if team == "A" else (0, 0, 255)
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"#{r['tid']}" if r["tid"] is not None else "?"
            cv2.putText(vis, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    on_field_counts["A"].append(n_a)
    on_field_counts["B"].append(n_b)
    per_frame_tracks.append(frame_track_map)
    per_frame_time.append(frame_times[fi])
    if vis is not None:
        annotated_last = vis

print(f"\nOn-field detections per frame -- team A: mean={np.mean(on_field_counts['A']):.1f} "
      f"min={min(on_field_counts['A'])} max={max(on_field_counts['A'])}")
print(f"On-field detections per frame -- team B: mean={np.mean(on_field_counts['B']):.1f} "
      f"min={min(on_field_counts['B'])} max={max(on_field_counts['B'])}")

# instantaneous speed samples: only consecutive SAMPLED frames, same track id
speeds_cm_s = {"A": [], "B": []}
for i in range(len(per_frame_tracks) - 1):
    dt = per_frame_time[i + 1] - per_frame_time[i]
    if dt <= 0:
        continue
    cur, nxt = per_frame_tracks[i], per_frame_tracks[i + 1]
    for tid, (fx0, fy0, team0) in cur.items():
        if tid in nxt:
            fx1, fy1, team1 = nxt[tid]
            dist_cm = float(np.hypot(fx1 - fx0, fy1 - fy0))
            speeds_cm_s[team0].append(dist_cm / dt)

n_matched = len(speeds_cm_s["A"]) + len(speeds_cm_s["B"])
n_total_dets = sum(len(m) for m in per_frame_tracks)
print(f"\nConsecutive-frame track matches usable for speed: {n_matched} "
      f"(out of {n_total_dets} total on-field detections across all frames)")
for team in ("A", "B"):
    v = np.array(speeds_cm_s[team]) / 100.0  # cm/s -> m/s
    if len(v):
        print(f"Team {team} speed samples: n={len(v)} mean={v.mean():.2f} m/s "
              f"median={np.median(v):.2f} m/s max={v.max():.2f} m/s")
    else:
        print(f"Team {team}: no consecutive-frame matches -- can't estimate speed "
              f"(fragmentation too high at this sample rate, try a higher sample_fps)")

if annotated_last is not None:
    cv2.imwrite(out_path, annotated_last)
    print(f"\nSaved last frame (team-colored, in-field only) to {out_path}")
