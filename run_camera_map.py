"""
Whole-video camera map: where was the camera looking, at every moment, in ONE
coordinate system.

WHY THIS EXISTS (the person, after four windows of guesswork):
  *"this camera view only shows around half of the field and the far end line
  detected is actually the halfway point, i want to be able to already have
  known this based on running the entire video through in a way that we are
  aware of how the camera moves and around where the field we are looking at
  based on total camera movement"*

Every calibration attempt so far has been made blind: a 60s window, no idea
which part of the field it framed, and so no way to tell "the far end line"
from "the halfway line" -- the two look identical inside one shot and differ
only in where they sit along a 100m field. The total sweep of the camera over
92 minutes settles that, because the union of everything the camera ever
covered is close to the whole field, and any single shot is a known fraction
of it.

WHAT IT DOES
  1. Samples the video and registers consecutive samples (mosaic.register_sequence).
     A cut ends a segment; homographies are only comparable within one.
  2. LINKS segments to each other. A fixed broadcast camera returns to similar
     framings, so two segments minutes apart usually share background (the
     mosaic finding measured 16+ ORB inliers at 63 min apart). Linking makes
     one GLOBAL frame spanning the whole video, not N unrelated ones.
  3. Reports, per sample: which segment, and the footprint of that frame in
     global coordinates -- so camera motion is a trajectory, not a guess.
  4. If a calibration is supplied (or found in the bank), converts that
     trajectory into FIELD METRES: "t=3600s covers x=24..44m of 100m".

Run:
  python3 run_camera_map.py videos/torovcanyon.mp4 [sample_every_sec] [max_minutes]
"""
import sys
import os
import json
import cv2
import numpy as np

from frisbee_analysis import MosaicConfig, register_sequence, UltimatePitchConfiguration
from frisbee_analysis.mosaic import register_pair_with_coverage
from frisbee_analysis.calibration import Calibration, pixel_to_field
from frisbee_analysis.video_overlays import overlay_boxes_for

video_path = sys.argv[1] if len(sys.argv) > 1 else "videos/torovcanyon.mp4"
sample_every = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
max_minutes = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
calib_path = sys.argv[4] if len(sys.argv) > 4 else None

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    sys.exit(f"could not open {video_path}")
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = total / fps
if max_minutes > 0:
    duration = min(duration, max_minutes * 60.0)
step = max(1, int(round(sample_every * fps)))
n_samples = int(duration * fps / step)
print(f"{video_path}: {duration/60:.1f} min, sampling every {sample_every:g}s "
      f"-> {n_samples} frames at {frame_w}x{frame_h}")

frames, times = [], []
# sequential read is far cheaper than seeking per sample on a long file
idx = 0
want = 0
while True:
    ok = cap.grab()
    if not ok or idx >= duration * fps:
        break
    if idx == want:
        ok, f = cap.retrieve()
        if ok:
            frames.append(f)
            times.append(idx / fps)
        want += step
    idx += 1
cap.release()
print(f"read {len(frames)} frames")

boxes = overlay_boxes_for(video_path, frame_w, frame_h)
cfg_m = MosaicConfig(overlay_mask_boxes=tuple(boxes))
print("registering...")
regs = register_sequence(frames, cfg_m)
by_seg = {}
for r in regs:
    if r.ok and r.homography is not None:
        by_seg.setdefault(r.segment_id, []).append(r)
sizes = {k: len(v) for k, v in sorted(by_seg.items(), key=lambda kv: -len(kv[1]))}
print(f"{len(by_seg)} segments; largest: {list(sizes.items())[:8]}")

# ---- link segments into one global frame -------------------------------
def anchor_of(seg):
    members = by_seg[seg]
    for r in members:
        if np.allclose(r.homography, np.eye(3)):
            return r
    return members[0]

segs = sorted(by_seg, key=lambda s: -len(by_seg[s]))
root = segs[0]
to_global = {root: np.eye(3)}          # segment coords -> global coords
print(f"linking {len(segs)} segments to root segment {root} "
      f"({len(by_seg[root])} frames)...")
pending = [s for s in segs if s != root]
for _ in range(len(segs)):
    progressed = False
    for s in list(pending):
        best = None
        for t in list(to_global):
            H, n_in, cov = register_pair_with_coverage(
                frames[anchor_of(t).frame_idx], frames[anchor_of(s).frame_idx], cfg_m)
            if H is None:
                continue
            if best is None or n_in > best[1]:
                best = (to_global[t] @ H, n_in, t)
        if best is not None and best[1] >= cfg_m.min_inliers:
            to_global[s] = best[0]
            pending.remove(s)
            progressed = True
            print(f"  segment {s:>3} -> via {best[2]:>3} ({best[1]} inliers)")
    if not progressed:
        break
print(f"linked {len(to_global)}/{len(segs)} segments; {len(pending)} isolated")

# ---- footprint of every sample in global coordinates -------------------
box = np.array([[[0.0, 0.0]], [[frame_w, 0.0]], [[frame_w, frame_h]], [[0.0, frame_h]]])


def _quad_area(c):
    x, y = c[:, 0], c[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# MEASURED: chaining a homography through a segment link occasionally produces
# a wildly distorted warp -- on the first 8 minutes, 90% of footprints came out
# at 0.99-1.33x the frame's own area and the remaining 10% ranged up to 579x,
# which alone inflated the reported sweep from 2.8 frame-widths to 93. A camera
# that pans and zooms modestly cannot change its footprint area by 500x, so
# area is a cheap, assumption-light way to throw out a broken chain.
AREA_LO, AREA_HI = 0.25, 4.0
frame_area = float(frame_w * frame_h)
rows, rejected = [], []
for r in regs:
    if not (r.ok and r.homography is not None) or r.segment_id not in to_global:
        continue
    H = to_global[r.segment_id] @ r.homography
    corners = cv2.perspectiveTransform(box, H).reshape(-1, 2)
    ratio = _quad_area(corners) / frame_area
    entry = {"t": times[r.frame_idx], "frame_idx": r.frame_idx,
             "segment": r.segment_id, "corners": corners.tolist(),
             "centre": corners.mean(axis=0).tolist(), "area_ratio": float(ratio)}
    if not (AREA_LO <= ratio <= AREA_HI) or not np.all(np.isfinite(corners)):
        rejected.append(entry)
        continue
    rows.append(entry)
if rejected:
    print(f"rejected {len(rejected)} sample(s) whose footprint area was outside "
          f"{AREA_LO}x..{AREA_HI}x the frame (broken registration chain):")
    for e in rejected[:10]:
        print(f"    t={e['t']:7.0f}s segment {e['segment']:>3} area={e['area_ratio']:8.1f}x")
print(f"{len(rows)} samples placed in the global frame "
      f"({100.0*len(rows)/max(1,len(frames)):.0f}% of those read)")

calib = None
if calib_path and os.path.exists(calib_path):
    calib = Calibration.load(calib_path)
    print(f"using calibration {calib_path} to express the map in FIELD METRES")
    print("  NOTE this only makes the numbers meaningful if that calibration's own")
    print("  window is one of the linked segments -- otherwise it is being applied")
    print("  to a frame it was never fitted to.")
    cfgp = UltimatePitchConfiguration()
    for row in rows:
        try:
            fp = [pixel_to_field(x, y, calib) for x, y in row["corners"]]
            row["field_x_m"] = [p[0] / 100.0 for p in fp]
            row["field_y_m"] = [p[1] / 100.0 for p in fp]
        except Exception:
            pass

out = {"video": video_path, "sample_every_sec": sample_every,
        "n_rejected": len(rejected),
        "frame_w": frame_w, "frame_h": frame_h,
        "n_segments": len(by_seg), "n_linked": len(to_global),
        "root_segment": root, "samples": rows}
os.makedirs("outputs", exist_ok=True)
with open("outputs/camera_map.json", "w") as f:
    json.dump(out, f)
print("wrote outputs/camera_map.json")

# ---- picture of the sweep ----------------------------------------------
if rows:
    allc = np.array([c for r in rows for c in r["corners"]])
    lo, hi = allc.min(0), allc.max(0)
    scale = min(1.0, 1600.0 / max(hi[0] - lo[0], hi[1] - lo[1]))
    W = int((hi[0] - lo[0]) * scale) + 20
    H = int((hi[1] - lo[1]) * scale) + 20
    canvas = np.zeros((H, W, 3), np.uint8)
    for n, r in enumerate(rows):
        pts = ((np.array(r["corners"]) - lo) * scale + 10).astype(np.int32)
        shade = int(40 + 200 * n / max(1, len(rows) - 1))
        cv2.polylines(canvas, [pts], True, (shade, 120, 255 - shade), 1)
    cv2.imwrite("outputs/camera_map.jpg", canvas)
    print(f"wrote outputs/camera_map.jpg ({W}x{H}) -- every sampled framing, "
          f"blue=start of video, red=end")
    print(f"total swept extent in global px: {hi[0]-lo[0]:.0f} x {hi[1]-lo[1]:.0f} "
          f"(one frame is {frame_w}x{frame_h})")
