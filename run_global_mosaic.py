"""
One picture of everything the camera ever saw, for the whole video.

WHY: every window so far has been judged on its own, and that is how a
halfway line got labelled a back line and how "the other endzone is not in
this footage" survived three rounds of my own checking. A single image in one
coordinate system makes the question "which part of the field is this?"
answerable by looking, for any moment in the match.

Unlike run_camera_map.py (which reports where each frame sits) this one
COMPOSITES the frames, and it saves the per-sample homographies -- the gap
that stopped the camera map being usable for transferring lines.

Registration is frame-to-frame within a segment and segment-to-segment across
cuts, exactly as run_camera_map does. Frames whose warp is implausible are
rejected on footprint area (MEASURED: 90% of footprints land within
0.99-1.33x the frame's own area; the rest reach 579x and are broken chains).

Painted as a COLLAGE, nearest-to-anchor first, so each output pixel comes
from the frame that reached it with the least warping.

Run:  python3 run_global_mosaic.py <video> [sample_every_sec] [max_megapixels]
"""
import sys
import os
import json
import cv2
import numpy as np

from frisbee_analysis import MosaicConfig, register_sequence
from frisbee_analysis.mosaic import register_pair_with_coverage
from frisbee_analysis.video_overlays import overlay_boxes_for

# outputs are filed into folders; make sure ours exist
os.makedirs("outputs/mosaic", exist_ok=True)

video = sys.argv[1] if len(sys.argv) > 1 else "videos/torovcanyon.mp4"
every = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
max_mp = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0

cap = cv2.VideoCapture(video)
if not cap.isOpened():
    sys.exit(f"could not open {video}")
fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
step = max(1, int(round(every * fps)))
print(f"{video}: {n_frames/fps/60:.1f} min, sampling every {every:g}s")

frames, times = [], []
idx = want = 0
while True:
    if not cap.grab():
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

boxes = overlay_boxes_for(video, W, H)
cfg = MosaicConfig(overlay_mask_boxes=tuple(boxes))
print("registering...")
regs = register_sequence(frames, cfg)
by_seg = {}
for r in regs:
    if r.ok and r.homography is not None:
        by_seg.setdefault(r.segment_id, []).append(r)
print(f"{len(by_seg)} segments")


def anchor_of(seg):
    for r in by_seg[seg]:
        if np.allclose(r.homography, np.eye(3)):
            return r
    return by_seg[seg][0]


segs = sorted(by_seg, key=lambda s: -len(by_seg[s]))
root = segs[0]
to_global = {root: np.eye(3)}
pending = [s for s in segs if s != root]
print(f"linking {len(segs)} segments (root {root}, {len(by_seg[root])} frames)...")
for _ in range(len(segs)):
    moved = False
    for s in list(pending):
        best = None
        for t in list(to_global):
            Hs, n_in, _ = register_pair_with_coverage(
                frames[anchor_of(t).frame_idx], frames[anchor_of(s).frame_idx], cfg)
            if Hs is None:
                continue
            if best is None or n_in > best[1]:
                best = (to_global[t] @ Hs, n_in)
        if best and best[1] >= cfg.min_inliers:
            to_global[s] = best[0]
            pending.remove(s)
            moved = True
    if not moved:
        break
print(f"linked {len(to_global)}/{len(segs)} segments; {len(pending)} isolated")

box = np.array([[[0.0, 0.0]], [[W, 0.0]], [[W, H]], [[0.0, H]]])
frame_area = float(W * H)


def _warp_is_plausible(c, frame_area):
    """Is this warped frame outline something a real camera could produce?

    MEASURED: an area-only test (and one taking abs()) passed frames that had
    been MIRRORED by a bad chained homography -- the first whole-video mosaic
    came out with a third of it upside down, sky at the bottom, because a
    reflected quad has a perfectly ordinary area. Signed area catches the
    reflection; convexity and a bounded side ratio catch the shears that
    survive it."""
    x, y = c[:, 0], c[:, 1]
    signed = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if signed <= 0:                       # mirrored: corner order reversed
        return False
    if not (0.25 <= abs(signed) / frame_area <= 4.0):
        return False
    sides = np.linalg.norm(np.roll(c, -1, axis=0) - c, axis=1)
    if sides.min() < 1e-6 or sides.max() / sides.min() > 6.0:
        return False                      # extreme shear
    cross = []                            # every turn the same way = convex
    for i in range(4):
        a, b, d = c[i], c[(i + 1) % 4], c[(i + 2) % 4]
        cross.append(np.cross(b - a, d - b))
    return all(v > 0 for v in cross) or all(v < 0 for v in cross)
placed = []
for r in regs:
    if not (r.ok and r.homography is not None) or r.segment_id not in to_global:
        continue
    Hg = to_global[r.segment_id] @ r.homography
    c = cv2.perspectiveTransform(box, Hg).reshape(-1, 2)
    if not np.all(np.isfinite(c)):
        continue
    if not _warp_is_plausible(c, frame_area):
        continue
    x, y = c[:, 0], c[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / frame_area
    placed.append((r.frame_idx, r.segment_id, Hg, c, area))
print(f"{len(placed)} of {len(frames)} frames placed "
      f"({100.0*len(placed)/len(frames):.0f}%)")
if not placed:
    sys.exit("nothing placed")

allc = np.concatenate([p[3] for p in placed], axis=0)
lo, hi = allc.min(0), allc.max(0)
w_px, h_px = hi[0] - lo[0], hi[1] - lo[1]
scale = min(1.0, np.sqrt(max_mp * 1e6 / (w_px * h_px)))
OW, OH = int(w_px * scale) + 2, int(h_px * scale) + 2
S = np.array([[scale, 0, -lo[0] * scale], [0, scale, -lo[1] * scale], [0, 0, 1]])
print(f"canvas {OW}x{OH} ({OW*OH/1e6:.1f} MP), scale {scale:.3f}, "
      f"sweep {w_px:.0f}x{h_px:.0f}px = {w_px/W:.1f} frame-widths")

canvas = np.zeros((OH, OW, 3), np.uint8)
filled = np.zeros((OH, OW), np.uint8)
ones = np.full((H, W), 255, np.uint8)
# least-warped first: closest footprint area to 1.0
placed.sort(key=lambda p: abs(p[4] - 1.0))
for n, (fi, seg, Hg, c, area) in enumerate(placed):
    Hs = S @ Hg
    warped = cv2.warpPerspective(frames[fi], Hs, (OW, OH))
    cover = cv2.warpPerspective(ones, Hs, (OW, OH))
    paint = (cover > 0) & (filled == 0)
    canvas[paint] = warped[paint]
    filled[paint] = 255
    if n % 50 == 0:
        print(f"  {n}/{len(placed)} composited, {100.0*filled.mean()/255:.0f}% filled",
              flush=True)
os.makedirs("outputs", exist_ok=True)
cv2.imwrite("outputs/mosaic/global_mosaic.jpg", canvas,
            [int(cv2.IMWRITE_JPEG_QUALITY), 88])
grid = canvas.copy()
for gx in range(0, OW, 500):
    cv2.line(grid, (gx, 0), (gx, OH), (0, 200, 255), 1)
    cv2.putText(grid, str(gx), (gx + 4, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(grid, str(gx), (gx + 4, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 1)
for gy in range(0, OH, 500):
    cv2.line(grid, (0, gy), (OW, gy), (0, 200, 255), 1)
    cv2.putText(grid, str(gy), (4, gy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(grid, str(gy), (4, gy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 1)
cv2.imwrite("outputs/mosaic/global_mosaic_grid.jpg", grid,
            [int(cv2.IMWRITE_JPEG_QUALITY), 88])
with open("outputs/mosaic/global_mosaic_index.json", "w") as f:
    json.dump({"video": video, "sample_every_sec": every, "frame_w": W, "frame_h": H,
                "canvas_w": OW, "canvas_h": OH, "scale": scale,
                "S": S.tolist(),
                "samples": [{"t": times[fi], "frame_idx": fi, "segment": seg,
                              "H_to_global": Hg.tolist(),
                              "H_to_canvas": (S @ Hg).tolist(),
                              "area_ratio": float(area)} for fi, seg, Hg, c, area in placed]},
              f)
print(f"\nfilled {100.0*filled.mean()/255:.0f}% of the canvas")
print("wrote outputs/mosaic/global_mosaic.jpg, outputs/mosaic/global_mosaic_grid.jpg (500px grid),")
print("      outputs/mosaic/global_mosaic_index.json (per-sample homographies -- what the")
print("      camera map was missing, so lines can now be pushed to ANY sampled time)")
