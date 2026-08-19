"""
Try to calibrate a new camera segment WITHOUT manual line-clicking, by
matching it against the calibration bank (frisbee_analysis/calibration_bank.py)
-- camera framings you've already calibrated before, via run_calibrate.py.

MOTIVATION: this footage cuts/pans often enough that a full game could need
one manual run_calibrate.py session per camera segment -- dozens across a
game. But it's a fixed mount (pure rotation/zoom), so the broadcast almost
certainly reuses a handful of recurring framings (the standard wide shot,
close-ups near each endzone) far more than it invents new ones. This script
checks "have I already calibrated a shot like this one?" before falling back
to asking for manual clicks.

Mechanism: same ORB+RANSAC registration already used within a segment
(mosaic.py), applied BETWEEN this segment's reference frame and every bank
entry's reference frame. A good match composes the calibration for free --
no clicking. NOT a replacement for run_calibrate.py: if nothing in the bank
matches, this tells you to run that instead (which then grows the bank for
next time automatically).

CAVEAT (same as any calibration, see calibration_finding.md's wrong-field
warning): an automatic match is still just a homography composition -- check
the saved overlay image before trusting it, same as a manual calibration.

Run:  python run_calibrate_auto.py <video_path> [start_sec] [duration_sec] [sample_fps] [out.json] [segment_id] [bank_dir]
Needs opencv-python (Track B deps). No GUI/display needed -- unlike
run_calibrate.py, this can run headlessly.
"""
import sys
import cv2
import numpy as np
from frisbee_analysis import UltimatePitchConfiguration, MosaicConfig, register_sequence
from frisbee_analysis.calibration import draw_field_overlay
from frisbee_analysis.calibration_bank import try_auto_calibrate, add_entry as add_bank_entry

video_path = sys.argv[1] if len(sys.argv) > 1 else "videos/videoplayback.mp4"
start_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 1158.0
duration_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 180.0
sample_fps = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
out_path = sys.argv[5] if len(sys.argv) > 5 else "outputs/calibration_auto.json"
segment_choice = sys.argv[6] if len(sys.argv) > 6 else None
bank_dir = sys.argv[7] if len(sys.argv) > 7 else "outputs/calibration_bank"

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
segment_frame_idx = {}
for r in regs:
    if r.ok:
        segment_frame_idx.setdefault(r.segment_id, []).append(r.frame_idx)
if not segment_frame_idx:
    sys.exit("no frames registered at all -- try a different window (start_sec/duration_sec)")

print("\nSegments found in this window:")
for sid, idxs in sorted(segment_frame_idx.items()):
    print(f"  segment {sid}: {len(idxs)} frames")

if segment_choice is not None:
    seg_id = int(segment_choice)
    if seg_id not in segment_frame_idx:
        sys.exit(f"segment {seg_id} not found -- pick one of {sorted(segment_frame_idx.keys())}")
else:
    seg_id = max(segment_frame_idx, key=lambda s: len(segment_frame_idx[s]))
    print(f"(defaulting to the largest -- pass a segment id as a 6th argument to pick a different one)")
seg_idx = sorted(segment_frame_idx[seg_id])

anchor_fi = next((fi for fi in seg_idx if np.allclose(homography_by_idx[fi], np.eye(3))), seg_idx[0])
ref_frame = frames[anchor_fi]

calib, matched_name, n_inliers, coverage = try_auto_calibrate(ref_frame, bank_dir, mosaic_cfg)

if calib is None:
    print(f"\nNo match found in the calibration bank ({bank_dir}) -- this camera framing "
          f"hasn't been calibrated before (or the bank is empty).")
    print(f"Fall back to manual calibration:")
    print(f"  python run_calibrate.py {video_path} {start_sec:.0f} {duration_sec:.0f} "
          f"{sample_fps} outputs/calibration.json {seg_id}")
    print(f"That will automatically add this framing to the bank, so future segments like "
          f"it won't need manual clicks again.")
    sys.exit(0)

print(f"\nMatched bank entry '{matched_name}' with {n_inliers} RANSAC inliers, "
      f"{coverage*100:.0f}% frame coverage -- calibration composed automatically, "
      f"NO manual clicking needed.")
if coverage < 0.4:
    # MEASURED FALSE POSITIVE (person caught a match 26 minutes away from its
    # canvas, visibly wrong, that had barely cleared the OLD threshold --
    # calibration_bank.py's BANK_MIN_INLIERS/BANK_MIN_COVERAGE now hard-reject
    # anything that weak before it even reaches here. This is just a softer
    # heads-up band above that floor, not the last line of defense anymore.
    print(f"  NOTE: coverage ({coverage*100:.0f}%) is only just above the bank's minimum -- "
          f"the matched keypoints were somewhat clustered, which risks drift when the fit is "
          f"extrapolated to project something far from that cluster (e.g. a sideline). Check "
          f"the overlay extra carefully.")
# MEASURED (person caught visible drift in an auto-matched overlay -- see
# calibration_bank.py's try_auto_calibrate docstring): the underlying manual
# calibration this bank canvas was built from can itself have real
# reprojection error that a healthy inlier count/coverage won't reveal,
# since it isn't introduced by THIS match step at all. Surface it instead of
# letting it disappear once composed.
if calib.source_max_reprojection_px is not None:
    print(f"  NOTE: the calibration this canvas was built from had up to "
          f"{calib.source_max_reprojection_px:.1f}px reprojection error of its own -- "
          f"that carries through to this auto-match too, on top of anything from this match step.")
calib.save(out_path)
print(f"Saved calibration to {out_path}")

cfg = UltimatePitchConfiguration()
overlay = draw_field_overlay(ref_frame, calib, cfg)
overlay_path = out_path.rsplit(".", 1)[0] + "_overlay.jpg"
cv2.imwrite(overlay_path, overlay)
print(f"Saved sanity-check overlay to {overlay_path} -- CHECK THIS BY EYE before trusting "
      f"the auto-match, same as any other calibration (see calibration_finding.md's "
      f"wrong-field warning -- a coincidental match is still possible in principle).")

# Growing the bank with this newly (auto-)derived calibration too -- makes
# future auto-matches easier still, since there's now one more reference
# point that's already proven to compose correctly.
entry_dir = add_bank_entry(bank_dir, ref_frame, calib)
print(f"Also added this segment to the bank: {entry_dir}")
