"""
Interactive manual field calibration: click calibration keypoints (cones +
brick marks, from frisbee_analysis/ultimate_config.py) on RAW frames within a
segment, not on a blended mosaic image.

MEASURED: cones are small enough that mosaic blending (weighted-averaging
overlapping warped frames, see run_mosaic_visualize.py) washes them out or
distorts their apparent position -- a cone that looked present under
pixel-level HSV analysis of a built mosaic turned out NOT actually visible in
practice when checked in a real viewer. Real objects are much clearer in a
single raw (unblended) frame at full quality. See results/calibration_finding.md.

Fix: this script rebuilds the SAME segment registration
frisbee_analysis.mosaic already computes (register_sequence) and lets you
browse the RAW frames within that segment, clicking each keypoint on
whichever frame shows it most clearly -- different keypoints can come from
different frames, since they only need to be visible SOMEWHERE across the
whole segment, not all at once in one image (see
frisbee_analysis/calibration.py's module docstring). Each click is
transformed through THAT frame's own homography (already computed by
register_sequence) into the segment's shared reference coordinate system
(the segment's first frame's own pixel coords) before fitting the final
calibration, so points clicked on different frames still end up comparable.

*** WRONG-FIELD WARNING (multi-field tournament venues): *** wide shots on
this footage repeatedly show SEVERAL simultaneous games on adjacent fields in
the background. A cone-like object far from the active play is not
necessarily on YOUR field -- only click cones you can confidently place on
the SAME field boundary as the game actually being broadcast (e.g. clearly
in line with that field's own sideline/players), not just any orange dot
visible in the wide shot. A point from the wrong field's geometry will fit a
homography that looks superficially valid but is nonsense -- this is at least
as likely a cause of an earlier failed calibration attempt as the mosaic-
blending issue this script already fixes. See results/calibration_finding.md.

NEEDS A REAL DISPLAY -- run this in your own terminal, not headless. Opens a
GUI window. Recomputes registration for the given window on startup (same
cost as run_mosaic_test.py) before you can start clicking.

Controls:
  n / p       next / previous frame within the segment (browse to find
              whichever frame shows the current keypoint clearest)
  left-click  mark the current keypoint at that pixel, in the CURRENTLY
              SHOWN frame
  s           skip the current keypoint (checked several frames, still not
              visible -- fine, you only need >=4 total, not all 10)
  u           undo the last placed keypoint, go back to it
  q           finish once you have enough points (need >= 4, more is better,
              spread across the field rather than clustered)

Run:  python run_calibrate.py <video_path> [start_sec] [duration_sec] [sample_fps] [out.json]
Needs opencv-python (Track B deps, see requirements.txt).
"""
import sys
import cv2
import numpy as np
from frisbee_analysis import UltimatePitchConfiguration, MosaicConfig, register_sequence
from frisbee_analysis.calibration import fit_calibration, draw_field_overlay, field_to_pixel

video_path = sys.argv[1] if len(sys.argv) > 1 else "videos/ojuc.mp4"
start_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 1580.0
duration_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 180.0
sample_fps = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
out_path = sys.argv[5] if len(sys.argv) > 5 else "outputs/calibration.json"

# This broadcast's static overlay (scoreboard/watermark top band, captions
# bottom band) -- same as run_mosaic_test.py. Adjust or clear for other footage.
OVERLAY_BOXES = [(0, 0, 1920, 230), (0, 860, 1920, 950)]

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    sys.exit(f"could not open {video_path}")
video_fps = cap.get(cv2.CAP_PROP_FPS)
step = max(1, int(round(video_fps / sample_fps)))
start_frame = int(round(start_sec * video_fps))
n_samples = int(duration_sec * sample_fps)

frames = []
frame_abs_sec = []  # absolute video timestamp (seconds) per loaded frame -- so
# the on-screen display can show exactly what point in the video you're
# looking at, not just a relative index (see results/calibration_finding.md
# on why "what time is this actually showing" needs to be directly verifiable).
for k in range(n_samples):
    target_frame = start_frame + k * step
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame = cap.read()
    if not ok:
        break
    frames.append(frame)
    # use POS_FRAMES *after* the seek, not the requested target -- if the
    # seek snapped to a different frame than requested, this reflects reality
    actual_frame = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1  # -1: read() advances past it
    frame_abs_sec.append(actual_frame / video_fps if actual_frame > 0 else target_frame / video_fps)
cap.release()
print(f"Loaded {len(frames)} frames (t={start_sec:.0f}-{start_sec + duration_sec:.0f}s, ~{sample_fps} fps)")

print("Registering frames (same cost as run_mosaic_test.py)...")
mosaic_cfg = MosaicConfig(overlay_mask_boxes=tuple(OVERLAY_BOXES))
regs = register_sequence(frames, mosaic_cfg)
homography_by_idx = {r.frame_idx: r.homography for r in regs if r.ok}

segment_sizes = {}
for r in regs:
    if r.ok:
        segment_sizes[r.segment_id] = segment_sizes.get(r.segment_id, 0) + 1
if not segment_sizes:
    sys.exit("no frames registered at all -- try a different window (start_sec/duration_sec)")
best_seg = max(segment_sizes, key=segment_sizes.get)
seg_idx = [r.frame_idx for r in regs if r.ok and r.segment_id == best_seg]
print(f"Using segment {best_seg}: {len(seg_idx)} frames to browse "
      f"(t={start_sec + seg_idx[0] / sample_fps:.0f}s - {start_sec + seg_idx[-1] / sample_fps:.0f}s)")

cfg = UltimatePitchConfiguration()
KEYPOINT_LABELS = {
    1: "cone: near-left back corner (closest to camera -- often off-camera)",
    2: "cone: near-right back corner (closest to camera -- often off-camera)",
    3: "cone: near-left goal corner",
    4: "cone: near-right goal corner",
    5: "cone: far-left goal corner",
    6: "cone: far-right goal corner",
    7: "cone: far-left back corner",
    8: "cone: far-right back corner",
    9: "brick mark: near (often not marked on recreational/coned fields -- skip freely)",
    10: "brick mark: far (often not marked on recreational/coned fields -- skip freely)",
}
# Reliable goal-line/back corners first, then the commonly-unreliable near-back
# corners and brick marks last -- see results/calibration_finding.md.
order = [3, 4, 5, 6, 7, 8, 1, 2, 9, 10]

browse_pos = [0]    # index into seg_idx
clicked = {}         # keypoint_index -> (x, y) in the SEGMENT's reference coords
clicked_frame = {}   # keypoint_index -> raw frame index it was clicked on (diagnostic)
current = [0]        # index into `order`


def current_frame_idx():
    return seg_idx[browse_pos[0]]


def redraw():
    fi = current_frame_idx()
    vis = frames[fi].copy()
    if current[0] < len(order):
        label = f"Click: {KEYPOINT_LABELS[order[current[0]]]}"
    else:
        label = f"All prompted -- {len(clicked)} placed. Press 'q' to finish."
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    abs_t = frame_abs_sec[fi]
    cv2.putText(vis, f"VIDEO TIME: {int(abs_t // 60)}:{abs_t % 60:05.2f}  "
                      f"(frame {browse_pos[0] + 1}/{len(seg_idx)} in segment)  "
                      f"n/p=browse click=mark s=skip u=undo q=finish",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(vis, f"placed so far: {len(clicked)}  (need >= 4)", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    cv2.putText(vis, "only click cones on THIS field -- other fields' cones may be visible in the background",
                (10, vis.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
    cv2.imshow("calibrate", vis)


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and current[0] < len(order):
        fi = current_frame_idx()
        H = homography_by_idx[fi]
        pt = cv2.perspectiveTransform(np.array([[[float(x), float(y)]]]), H)
        clicked[order[current[0]]] = (float(pt[0, 0, 0]), float(pt[0, 0, 1]))
        clicked_frame[order[current[0]]] = fi
        current[0] += 1
        redraw()


cv2.namedWindow("calibrate")
cv2.setMouseCallback("calibrate", on_mouse)
redraw()
while True:
    key = cv2.waitKey(20) & 0xFF
    if key == ord('n'):
        browse_pos[0] = min(len(seg_idx) - 1, browse_pos[0] + 1)
        redraw()
    elif key == ord('p'):
        browse_pos[0] = max(0, browse_pos[0] - 1)
        redraw()
    elif key == ord('s') and current[0] < len(order):
        current[0] += 1
        redraw()
    elif key == ord('u'):
        if current[0] > 0:
            current[0] -= 1
            clicked.pop(order[current[0]], None)
            clicked_frame.pop(order[current[0]], None)
            redraw()
    elif key == ord('q'):
        break
cv2.destroyAllWindows()

if len(clicked) < 4:
    sys.exit(f"only {len(clicked)} points placed -- need >= 4 for a homography. "
              f"Re-run and click more (browse more frames with n/p).")

indices = list(clicked.keys())
points = [clicked[i] for i in indices]
frame_indices = [clicked_frame[i] for i in indices]
times = [frame_abs_sec[fi] for fi in frame_indices]
print("Points clicked at video times: " +
      ", ".join(f"kp{idx}={int(t // 60)}:{t % 60:05.2f}" for idx, t in zip(indices, times)))
print(f"  (spread: {max(times) - min(times):.1f}s apart -- wider spread means more "
      f"homography-chaining drift risk, see results/calibration_finding.md)")
calib = fit_calibration(points, indices, f"{video_path} t={start_sec:.0f}-{start_sec + duration_sec:.0f}s",
                         cfg, source_frame_indices=frame_indices)
calib.save(out_path)
print(f"Saved calibration ({len(points)} points: {indices}) to {out_path}")

# MEASURED: a calibration can look plausible in the overlay image while still
# being numerically unreliable -- verify it, don't just eyeball it. Reprojection
# error re-derives each point's pixel position from its known field coordinate
# through the fitted homography and compares to where it was actually clicked;
# large error means the points don't actually agree on one consistent
# homography (bad click, bad frame registration, or points too close to
# collinear to constrain the fit).
print("\nReprojection error per point (self-fit residual -- want this small, a few px):")
errs = []
for label, img_pt, field_pt, fi in zip(calib.keypoint_labels, calib.image_points, calib.field_points, frame_indices):
    proj_x, proj_y = field_to_pixel(field_pt[0], field_pt[1], calib)
    err = ((proj_x - img_pt[0]) ** 2 + (proj_y - img_pt[1]) ** 2) ** 0.5
    errs.append(err)
    frames_from_ref = fi - seg_idx[0]
    flag = "  <-- HIGH, this point may be bad" if err > 20 else ""
    print(f"  keypoint {label}: error={err:.1f}px  (frame {frames_from_ref} from segment start){flag}")
if max(errs) > 20:
    print("\nWARNING: high reprojection error means this calibration is NOT reliable yet.")
    print("Likely causes: points too close to collinear (need more spread, not just more")
    print("points), or drift in frame registration for points clicked far from the")
    print("segment's reference frame (try a shorter duration_sec, or click keypoints on")
    print("frames earlier in the browsing order). Re-run and add/replace points.")

# sanity-check overlay on the segment's own reference frame (identity homography)
ref_frame = frames[seg_idx[0]]
overlay = draw_field_overlay(ref_frame, calib, cfg)
overlay_path = out_path.rsplit(".", 1)[0] + "_overlay.jpg"
cv2.imwrite(overlay_path, overlay)
print(f"\nSaved field-outline overlay (on the segment's reference frame) to {overlay_path}")
print("Note: a plausible-looking overlay is NOT enough on its own -- check the")
print("reprojection error above too (see results/calibration_finding.md).")
