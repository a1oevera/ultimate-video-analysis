"""
Interactive manual field calibration: click LINE correspondences (>=2 pixels
you're confident lie somewhere along a known field line -- NOT exact corner
points) on RAW frames within a segment, not on a blended mosaic image.

WHY LINES, NOT POINTS: real footage kept producing too few reliable exact
corner points (see results/calibration_finding.md) -- registration breaks
routinely split "left side visible" from "right side visible" into different,
non-bridgeable segments, so a session often only has clean access to 2-3
corners at best. A line constraint is a much lower bar: you don't need the
precise intersection, just confidence about which painted line you're
looking at. `fit_calibration_with_lines` (lines-only) is synthetically
validated to recover a homography exactly; MIXING points and lines has a
real, unresolved bug (see the same doc) -- this tool deliberately only
collects lines, never points, to stay on the validated path.

MEASURED: cones/mosaics have their own problems (blending washout, wrong-
field contamination) that lines mostly sidestep -- a sideline is long enough
to still be identifiable even when heavily blurred or partially out of frame,
and you're not trying to pin down one small object.

Fix carried over from the earlier points-only version: this script rebuilds
the SAME segment registration frisbee_analysis.mosaic already computes
(register_sequence) and lets you browse RAW frames within that segment,
clicking each line's points on whichever frame shows it most clearly --
different lines (and different points on the same line) can come from
different frames. Each click is transformed through THAT frame's own
homography into the segment's shared reference coordinate system before
fitting, so clicks from different frames still end up comparable.

*** WRONG-FIELD WARNING (multi-field tournament venues): *** wide shots on
this footage repeatedly show SEVERAL simultaneous games on adjacent fields in
the background. Only click points you can confidently place on the SAME
field boundary as the game actually being broadcast -- a line from the wrong
field's geometry will fit a homography that looks superficially valid but is
nonsense. See results/calibration_finding.md.

NEEDS A REAL DISPLAY -- run this in your own terminal, not headless. Opens a
GUI window. Recomputes registration for the given window on startup (same
cost as run_mosaic_test.py) before you can start clicking.

Controls:
  n / p       next / previous frame within the segment (browse to find
              whichever frame shows the current line clearest)
  left-click  add a point to the CURRENT line, at that pixel in the
              CURRENTLY SHOWN frame (add as many as you want -- more points
              on the same line = a more robust fit via total-least-squares)
  SPACE       finish the current line (needs >= 2 points) and move to the
              next one
  s           skip the current line entirely (checked several frames, still
              not visible -- fine, you only need >= 4 lines total, not all 6)
  u           undo: removes the last point on the current line, or if there
              are none, re-opens the previous line for editing
  q           finish once you have enough lines (need >= 4, more is better)

Run:  python run_calibrate.py <video_path> [start_sec] [duration_sec] [sample_fps] [out.json] [segment_id]
Prints every segment found in the window (id, frame count, time range)
before opening the GUI. Omit segment_id to use the largest one (default);
pass a specific id from that printout to browse a different one -- e.g. if
what you want to click is in a smaller segment next to a bigger one.
Needs opencv-python (Track B deps, see requirements.txt).
"""
import sys
import cv2
import numpy as np
from frisbee_analysis import UltimatePitchConfiguration, MosaicConfig, register_sequence
from frisbee_analysis.calibration import (fit_calibration_with_lines, draw_field_overlay,
                                          field_to_pixel, check_line_reprojection_error)
from frisbee_analysis.calibration_bank import add_entry as add_bank_entry

video_path = sys.argv[1] if len(sys.argv) > 1 else "videos/ojuc.mp4"
start_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 1580.0
duration_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 180.0
sample_fps = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
out_path = sys.argv[5] if len(sys.argv) > 5 else "outputs/calibration.json"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    sys.exit(f"could not open {video_path}")

# Static broadcast overlay position depends on the source video -- MEASURED BUG
# (see results/calibration_finding.md): this used to be hardcoded for the
# 1920x1080 "ojuc.mp4" video's overlay position regardless of which video was
# actually loaded. Against the 640x360 UFA video that masked the top 64% of
# every frame (real field content, wasted) while completely missing the
# actual score bug at the bottom (never masked, corrupting registration --
# most likely cause of the segment-selection weirdness this was chasing).
_frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
_frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if (_frame_w, _frame_h) == (1920, 1080):
    OVERLAY_BOXES = [(0, 0, 1920, 230), (0, 860, 1920, 950)]  # ojuc.mp4: top scoreboard band, captions band
elif (_frame_w, _frame_h) == (640, 360):
    OVERLAY_BOXES = [(0, 295, 640, 360)]  # videoplayback.mp4: UFA score bug + ticker, bottom band
else:
    OVERLAY_BOXES = []
    print(f"WARNING: no known overlay mask for {_frame_w}x{_frame_h} -- add one to run_calibrate.py "
          f"if this video has a static broadcast graphic, or registration may be corrupted by it.")

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

segment_frame_idx = {}
for r in regs:
    if r.ok:
        segment_frame_idx.setdefault(r.segment_id, []).append(r.frame_idx)
if not segment_frame_idx:
    sys.exit("no frames registered at all -- try a different window (start_sec/duration_sec)")

# MEASURED (results/calibration_finding.md): always auto-picking the LARGEST
# segment silently excludes a smaller segment even when it's the one
# containing what you actually want to click -- confirmed against real
# footage where a genuine registration failure at one frame (camera
# motion/blur, not a bug) split a 2-frame segment containing the target
# moment from a 9-frame segment right after it, and the tool kept forcing
# the 9-frame one. Print every segment so this is visible and choosable
# instead of silently deciding for you.
print("\nSegments found in this window:")
for sid, idxs in sorted(segment_frame_idx.items()):
    t0, t1 = frame_abs_sec[idxs[0]], frame_abs_sec[idxs[-1]]
    print(f"  segment {sid}: {len(idxs)} frames, t={int(t0//60)}:{t0%60:05.2f} - {int(t1//60)}:{t1%60:05.2f}")

segment_choice = sys.argv[6] if len(sys.argv) > 6 else None
if segment_choice is not None:
    seg_id = int(segment_choice)
    if seg_id not in segment_frame_idx:
        sys.exit(f"segment {seg_id} not found -- pick one of {sorted(segment_frame_idx.keys())}")
    best_seg = seg_id
else:
    best_seg = max(segment_frame_idx, key=lambda s: len(segment_frame_idx[s]))
    print(f"(defaulting to the largest -- pass a segment id as a 6th argument to pick a different one)")
seg_idx = segment_frame_idx[best_seg]
t0, t1 = frame_abs_sec[seg_idx[0]], frame_abs_sec[seg_idx[-1]]
print(f"Using segment {best_seg}: {len(seg_idx)} frames to browse "
      f"(t={int(t0//60)}:{t0%60:05.2f} - {int(t1//60)}:{t1%60:05.2f})")

cfg = UltimatePitchConfiguration()
LINE_LABELS = {
    (1, 7): "near sideline (left side of field, closest to camera)",
    (2, 8): "far sideline (right side of field)",
    (3, 4): "near goal line",
    (5, 6): "far goal line",
    (1, 2): "near back line (very close to camera -- often off-camera)",
    (7, 8): "far back line",
}
# Sidelines + goal lines first: long lines, measured most likely to actually
# be visible/painted. Back lines last -- short, near-camera ones are often
# out of frame. See results/calibration_finding.md.
order = [(1, 7), (2, 8), (3, 4), (5, 6), (1, 2), (7, 8)]

browse_pos = [0]     # index into seg_idx
lines = {}            # field_edge -> list of (x, y) in SEGMENT reference coords
lines_frames = {}     # field_edge -> list of raw frame indices (diagnostic, parallel to lines[edge])
current_pts = []      # points collected so far for the line CURRENTLY being clicked
current_frames = []   # frame each of current_pts was clicked on
current = [0]          # index into `order`


def current_frame_idx():
    return seg_idx[browse_pos[0]]


def redraw():
    fi = current_frame_idx()
    vis = frames[fi].copy()
    if current[0] < len(order):
        edge = order[current[0]]
        label = f"Click 2+ points on: {LINE_LABELS[edge]}  ({len(current_pts)} placed on this line)"
    else:
        label = f"All prompted -- {len(lines)} lines done. Press 'q' to finish."
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    abs_t = frame_abs_sec[fi]
    cv2.putText(vis, f"VIDEO TIME: {int(abs_t // 60)}:{abs_t % 60:05.2f}  "
                      f"(frame {browse_pos[0] + 1}/{len(seg_idx)} in segment)",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(vis, "n/p=browse  click=add point  SPACE=finish line  s=skip line  u=undo  q=finish",
                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(vis, f"lines done: {len(lines)}  (need >= 4)", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    # draw points already placed on the line in progress, if they're on THIS frame
    for (px, py), pfi in zip(current_pts, current_frames):
        if pfi == fi:
            # current_pts are stored in segment-reference coords -- back-project
            # through this frame's own homography to draw them in the right spot
            H_inv = np.linalg.inv(homography_by_idx[fi])
            proj = cv2.perspectiveTransform(np.array([[[px, py]]]), H_inv)
            cx, cy = proj[0, 0]
            cv2.circle(vis, (int(cx), int(cy)), 5, (0, 255, 0), -1)
    cv2.putText(vis, "only click points on THIS field -- other fields may be visible in the background",
                (10, vis.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
    cv2.imshow("calibrate", vis)


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and current[0] < len(order):
        fi = current_frame_idx()
        H = homography_by_idx[fi]
        pt = cv2.perspectiveTransform(np.array([[[float(x), float(y)]]]), H)
        current_pts.append((float(pt[0, 0, 0]), float(pt[0, 0, 1])))
        current_frames.append(fi)
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
    elif key == ord(' ') and current[0] < len(order):
        if len(current_pts) >= 2:
            edge = order[current[0]]
            lines[edge] = list(current_pts)
            lines_frames[edge] = list(current_frames)
            current_pts, current_frames = [], []
            current[0] += 1
            redraw()
    elif key == ord('s') and current[0] < len(order):
        current_pts, current_frames = [], []
        current[0] += 1
        redraw()
    elif key == ord('u'):
        if current_pts:
            current_pts.pop()
            current_frames.pop()
        elif current[0] > 0:
            current[0] -= 1
            edge = order[current[0]]
            current_pts = lines.pop(edge, [])
            current_frames = lines_frames.pop(edge, [])
        redraw()
    elif key == ord('q'):
        break
cv2.destroyAllWindows()

if len(lines) < 4:
    sys.exit(f"only {len(lines)} lines placed -- need >= 4 for a homography. "
              f"Re-run and click more (browse more frames with n/p, SPACE to finish each line).")

edges = list(lines.keys())
line_image_points = [lines[e] for e in edges]
line_frame_indices = [lines_frames[e] for e in edges]
all_frame_indices = [fi for e in edges for fi in lines_frames[e]]
times = [frame_abs_sec[fi] for fi in all_frame_indices]
print("Lines clicked: " + ", ".join(f"{LINE_LABELS[e]} ({len(lines[e])} pts)" for e in edges))
print(f"  (time spread across all clicks: {max(times) - min(times):.1f}s -- wider spread means more "
      f"homography-chaining drift risk, see results/calibration_finding.md)")
calib = fit_calibration_with_lines([], [], f"{video_path} t={start_sec:.0f}-{start_sec + duration_sec:.0f}s",
                                    line_image_points=line_image_points, line_field_edges=edges,
                                    cfg=cfg, source_frame_indices=all_frame_indices,
                                    line_frame_indices=line_frame_indices)
calib.save(out_path)
print(f"Saved calibration ({len(edges)} lines) to {out_path}")

# MEASURED: a calibration can look plausible in the overlay image while still
# being numerically unreliable -- verify it, don't just eyeball it. For lines,
# the right check isn't point reprojection -- it's PERPENDICULAR DISTANCE of
# each clicked point from the fitted line. Uses the reusable checker (not
# inline math) so this is the SAME check whether run live here or re-run
# later from the saved file via check_line_reprojection_error(Calibration.load(...))
# -- a real gap this closes, see results/calibration_finding.md.
print("\nReprojection error per line (perpendicular distance of each clicked "
      "point from the fitted line -- want this small, a few px):")
line_results = check_line_reprojection_error(calib, cfg)
all_errs = []
for res in line_results:
    edge = tuple(res["edge"])
    all_errs.extend(res["errors"])
    flag = "  <-- HIGH" if res["max"] > 20 else ""
    print(f"  {LINE_LABELS[edge]}: max={res['max']:.1f}px mean={res['mean']:.1f}px "
          f"(over {len(res['errors'])} points){flag}")
if all_errs and max(all_errs) > 20:
    print("\nWARNING: high reprojection error means this calibration is NOT reliable yet.")
    print("Likely causes: lines too close to parallel/concurrent with each other (need more")
    print("geometric diversity, not just more lines), or drift in frame registration for points")
    print("clicked far from the segment's reference frame (try a shorter duration_sec, or click")
    print("on frames earlier in the browsing order). Re-run and add/replace lines.")

# sanity-check overlay on the segment's own REFERENCE frame -- the frame whose
# own homography is identity, i.e. its raw pixels ARE the segment's shared
# reference coordinate system that calib.H() operates in (NOT necessarily
# seg_idx[0] -- _reclaim_failed_frames in mosaic.py can fold an earlier-index
# frame into a segment whose anchor came later, so find the true identity
# frame rather than assuming the first index is it).
anchor_fi = next((fi for fi in seg_idx if np.allclose(homography_by_idx[fi], np.eye(3))), seg_idx[0])
ref_frame = frames[anchor_fi]
overlay = draw_field_overlay(ref_frame, calib, cfg)
overlay_path = out_path.rsplit(".", 1)[0] + "_overlay.jpg"
cv2.imwrite(overlay_path, overlay)
print(f"\nSaved field-outline overlay (on the segment's reference frame) to {overlay_path}")
print("Note: a plausible-looking overlay is NOT enough on its own -- check the")
print("reprojection error above too (see results/calibration_finding.md).")

# Add to the calibration bank so future segments with a similar camera
# framing can auto-match against this one instead of needing manual clicks
# again (see frisbee_analysis/calibration_bank.py). Skip if the reprojection
# error was flagged HIGH above -- don't propagate a bad calibration to future
# auto-matches just because it was already computed.
bank_dir = "outputs/calibration_bank"
if all_errs and max(all_errs) > 20:
    print(f"\nNOT added to the calibration bank ({bank_dir}) -- reprojection error was HIGH, "
          f"see the warning above. Fix the calibration first, or add it manually later via "
          f"frisbee_analysis.calibration_bank.add_entry once you trust it.")
else:
    entry_dir = add_bank_entry(bank_dir, ref_frame, calib)
    print(f"\nAdded to the calibration bank: {entry_dir}")
    print(f"Future segments with a similar camera framing can auto-match against this via "
          f"run_calibrate_auto.py, skipping manual clicking entirely.")
