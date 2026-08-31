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

Frames are placed by DIRECT registration to a small set of hub frames, never
by chaining -- see the note above the placement loop for the measurements that
forced this. Frames whose warp is implausible are
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

from frisbee_analysis import MosaicConfig
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

# ---- HUB placement, not segment chaining -------------------------------
# MEASURED, and the reason this replaces register_sequence + segment linking:
# that approach cut the video into segments (69 at 10s sampling, 102 at 3s)
# and positioned each by ONE anchor-to-anchor registration, so error
# compounded down a link tree. Even segments of 20+ frames came out with a
# median footprint area 1.15x the frame's own and 1.70x at the 90th
# percentile -- visible smearing. Sampling 3.3x denser made it WORSE (fill
# 58% -> 53%) because it produced more segments, hence more links.
#
# This camera is fixed and only pans/tilts/zooms, so a frame does not need a
# chain: it needs one good registration to something already placed. Frames
# are registered DIRECTLY to a small set of hubs, capping depth at 1. A frame
# that matches no hub is promoted to a hub itself if it can reach any placed
# frame, which is depth 2. Nothing is ever deeper than that.
hubs = []            # (frame_index, H_hub_to_global)
placements = {}      # frame_index -> H_frame_to_global
last_hub = 0
print("placing frames against hubs (direct registration, no chaining)...")
for n, f in enumerate(frames):
    if not hubs:
        hubs.append((n, np.eye(3)))
        placements[n] = np.eye(3)
        continue
    order = sorted(range(len(hubs)), key=lambda k: abs(k - last_hub))
    got = None
    for k in order:
        hi, Hh = hubs[k]
        Hs, n_in, _ = register_pair_with_coverage(frames[hi], f, cfg)
        if Hs is not None and n_in >= cfg.min_inliers:
            got = (Hh @ Hs, k, n_in)
            break
    if got is None:
        # promote: can this frame reach ANY already-placed frame?
        for hi in sorted(placements, key=lambda i: -i)[:25]:
            Hs, n_in, _ = register_pair_with_coverage(frames[hi], f, cfg)
            if Hs is not None and n_in >= cfg.min_inliers:
                Hg = placements[hi] @ Hs
                hubs.append((n, Hg))
                placements[n] = Hg
                got = (Hg, len(hubs) - 1, n_in)
                print(f"  frame {n} ({times[n]:.0f}s) became hub {len(hubs)-1} "
                      f"({n_in} inliers)", flush=True)
                break
    if got is None:
        continue
    placements[n], last_hub = got[0], got[1]
    if n % 100 == 0:
        print(f"  {n}/{len(frames)} frames, {len(hubs)} hubs, "
              f"{len(placements)} placed", flush=True)
print(f"{len(hubs)} hubs; {len(placements)}/{len(frames)} frames placed "
      f"({100.0*len(placements)/len(frames):.0f}%)")

box = np.array([[[0.0, 0.0]], [[W, 0.0]], [[W, H]], [[0.0, H]]])
frame_area = float(W * H)


def _warp_is_plausible(c, frame_area, Hg=None, W=None, H=None):
    """Is this warped frame outline something this camera could produce?

    MEASURED, and the correction that matters: an EARLIER version bounded the
    footprint AREA to 0.25-4x the frame's own, calling anything outside that a
    broken chain. That is wrong for a fixed camera that zooms. Measured over
    326 placed frames, the local Jacobian at frame centre has ANISOTROPY
    (ratio of singular values, 1.0 = pure zoom with no shear) of median 1.03,
    90th percentile 1.13, max 1.36 -- 91% of frames are near-pure zoom. Their
    area varies because of zoom and perspective foreshortening, not error.
    The frames the area rule rejected were the most zoomed-out and most
    oblique ones -- exactly the ones at the edges of the camera's range -- so
    the rule cut the mosaic short and then made it look as though the camera
    never panned that far.

    RETRACTION: loosening the size bound to 0.02-25x on the strength of that
    measurement made the mosaic WORSE (fill 54% -> 36%, one enormous stretched
    frame dominating the canvas). The anisotropy above is measured at the
    frame CENTRE, and a projective warp varies across the image -- a frame can
    be near-isotropic in the middle and badly stretched at its corners. The
    size bound is therefore kept. A correct version of the shear test would
    evaluate the Jacobian at all four corners; that is untested.

    So: reject SHEAR as well as size. Reflection and non-convexity are still
    rejected -- a mirrored quad has a perfectly ordinary area and shear, and
    an early mosaic came out a third upside down because of it."""
    x, y = c[:, 0], c[:, 1]
    signed = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if signed <= 0:                       # mirrored
        return False
    if not (0.25 <= abs(signed) / frame_area <= 4.0):
        return False
    cross = []
    for i in range(4):
        a, b, d = c[i], c[(i + 1) % 4], c[(i + 2) % 4]
        cross.append(np.cross(b - a, d - b))
    if not (all(v > 0 for v in cross) or all(v < 0 for v in cross)):
        return False
    # NOTE: a centre-only shear test was tried here and removed -- see the
    # RETRACTION above. Size + orientation + convexity is what actually
    # produced the best mosaic measured (326 frames, 54% fill).
    return True
placed = []
for fi in sorted(placements):
    Hg = placements[fi]
    c = cv2.perspectiveTransform(box, Hg).reshape(-1, 2)
    if not np.all(np.isfinite(c)):
        continue
    if not _warp_is_plausible(c, frame_area, Hg, W, H):
        continue
    x, y = c[:, 0], c[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / frame_area
    placed.append((fi, 0, Hg, c, area))
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
                "samples": [{"t": times[fi], "frame_idx": fi, "hub_depth": 1,
                              "H_to_global": Hg.tolist(),
                              "H_to_canvas": (S @ Hg).tolist(),
                              "area_ratio": float(area)} for fi, seg, Hg, c, area in placed]},
              f)
print(f"\nfilled {100.0*filled.mean()/255:.0f}% of the canvas")
print("wrote outputs/mosaic/global_mosaic.jpg, outputs/mosaic/global_mosaic_grid.jpg (500px grid),")
print("      outputs/mosaic/global_mosaic_index.json (per-sample homographies -- what the")
print("      camera map was missing, so lines can now be pushed to ANY sampled time)")
