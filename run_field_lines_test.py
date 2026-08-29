"""
Measure frisbee_analysis/field_lines.py against GROUND TRUTH, on real footage.

The claim this script exists to test, from results/calibration_finding.md:
"naive automated detection is genuinely hard on this footage" -- 89 Hough
segments on a mosaic that were nearly all stitching-seam artifacts, 80 on a
raw frame dominated by treeline/scoreboard/caption confounds, and a cone
colour filter that produced 81 false candidates on one crowd shot. That
finding also named what a working version would need: "overlay + non-grass
masking before any edge detection, orientation filtering, and probably a
dedicated cone detector". field_lines.py is that; this measures whether it
actually helped, or just moved the problem.

GROUND TRUTH: the calibration bank already holds human-verified calibrations
(outputs/calibration_bank/*, 2.1-2.7px reprojection error). Matching a
window's reference frame against the bank gives that frame a trusted
homography, which projects the TRUE position of every WFDF field line into
the exact coordinate system the tracker reports in. So "did we find the
sideline" is a measured pixel distance, not an eyeball.

Reported, in order:
  1. confound counts -- naive whole-frame Hough vs scene-aware, same frame
  2. per-frame vs tracked-across-the-pan line counts
  3. for each of the 6 WFDF edges: distance from the true line to the
     nearest tracked line (the number that says whether this works)
  4. what rank_line_labelings proposes, and how far its homography lands
     from the ground-truth one, in pixels, at every field vertex

Run:  python3 run_field_lines_test.py [video] [start_sec] [duration_sec] [sample_fps]
Defaults to videos/torovcanyon.mp4 t=3945s, the window whose calibration is
in the bank. Writes overlay images to outputs/.
"""
import sys
import os
import cv2
import numpy as np

from frisbee_analysis import (UltimatePitchConfiguration, MosaicConfig, register_sequence,
                              playing_surface_mask, line_pixel_mask, detect_line_segments,
                              track_lines, track_cone_hints, rank_line_labelings,
                              group_by_vanishing_point, draw_detections)
from frisbee_analysis.calibration import field_to_pixel, draw_field_overlay
from frisbee_analysis.calibration_bank import try_auto_calibrate
from frisbee_analysis.video_overlays import overlay_boxes_for
from frisbee_analysis.field_lines import SIDELINE_EDGES, CROSS_EDGES, projected_field_is_sane

video_path = sys.argv[1] if len(sys.argv) > 1 else "videos/torovcanyon.mp4"
start_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 3945.0
duration_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
sample_fps = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
BANK_DIR = "outputs/calibration_bank"
cfg = UltimatePitchConfiguration()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    sys.exit(f"could not open {video_path}")
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS)
overlay_boxes = overlay_boxes_for(video_path, frame_w, frame_h)

step = max(1, int(round(video_fps / sample_fps)))
start_frame = int(round(start_sec * video_fps))
frames = []
for k in range(int(duration_sec * sample_fps)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + k * step)
    ok, frame = cap.read()
    if not ok:
        break
    frames.append(frame)
cap.release()
if not frames:
    sys.exit("no frames read")
print(f"{video_path}  t={start_sec:.0f}-{start_sec + duration_sec:.0f}s  "
      f"{len(frames)} frames @ {sample_fps}fps  ({frame_w}x{frame_h})")

# --------------------------------------------------------------------------
print("\n=== 1. confounds: naive whole-frame Hough vs scene-aware ===")
# Same detector settings both ways; the ONLY difference is whether the search
# is restricted to the playing surface and to a white-paint ridge response.
probe = frames[0]
gray = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)
naive = cv2.HoughLinesP(cv2.Canny(gray, 50, 150), 1, np.pi / 360, threshold=55,
                         minLineLength=int(0.045 * frame_w), maxLineGap=int(0.012 * frame_w))
n_naive = 0 if naive is None else len(naive)

surface = playing_surface_mask(probe, overlay_boxes)
surface_frac = float((surface > 0).mean())
tops = []
for x in range(0, frame_w, 40):
    column = np.nonzero(surface[:, x])[0]
    tops.append(int(column[0]) if len(column) else frame_h)
ridge = line_pixel_mask(probe, surface)
scene_aware = detect_line_segments(probe, surface, overlay_boxes)
print(f"  naive Canny+Hough, whole frame : {n_naive:5d} raw segments")
print(f"  playing surface                : {surface_frac * 100:5.1f}% of frame, "
      f"top edge y median={int(np.median(tops))} min={min(tops)}")
print(f"  white-paint ridge pixels        : {float((ridge > 0).mean()) * 100:5.2f}% of frame")
print(f"  scene-aware, merged             : {len(scene_aware):5d} lines "
      f"({100 * (1 - len(scene_aware) / max(n_naive, 1)):.0f}% fewer things to reason about)")
cv2.imwrite("outputs/field_lines_surface.jpg",
            draw_detections(probe, surface, scene_aware))

# --------------------------------------------------------------------------
print("\n=== 2. tracking across the pan ===")
mosaic_cfg = MosaicConfig(overlay_mask_boxes=tuple(overlay_boxes))
print("  registering frames...")
regs = register_sequence(frames, mosaic_cfg)
by_segment = {}
for r in regs:
    if r.ok and r.homography is not None:
        by_segment.setdefault(r.segment_id, []).append(r.frame_idx)
if not by_segment:
    sys.exit("no frames registered -- pick another window")
segment_id = max(by_segment, key=lambda s: len(by_segment[s]))
member_idx = by_segment[segment_id]
print(f"  segments found: { {k: len(v) for k, v in by_segment.items()} }; "
      f"using segment {segment_id} ({len(member_idx)} frames)")

per_frame_counts = []
for idx in member_idx:
    per_frame_counts.append(len(detect_line_segments(frames[idx], None, overlay_boxes)))
tracked = track_lines(frames, regs, segment_id, overlay_boxes, min_frames=3)
print(f"  per-frame line counts          : min={min(per_frame_counts)} "
      f"median={int(np.median(per_frame_counts))} max={max(per_frame_counts)}")
print(f"  tracked >=3 frames             : {len(tracked)} lines")
for i, t in enumerate(tracked):
    print(f"    [{i}] {t.colour:>6} paint, seen in {t.n_frames:2d}/{len(member_idx)} frames, "
          f"len={t.length if hasattr(t, 'length') else np.hypot(*(np.subtract(t.p2, t.p1))):7.0f}px, "
          f"support={t.total_support_px:5d}, rms_fit={t.rms_fit_px:5.2f}px")

family_a, family_b, vp = group_by_vanishing_point(tracked)
print(f"  vanishing-point grouping       : {len(family_a)} + {len(family_b)} lines"
      + (f", VP at ({vp[0]:.0f},{vp[1]:.0f})" if vp else ", no VP found"))

# --------------------------------------------------------------------------
print("\n=== 3. ground truth: how close are tracked lines to the REAL field lines? ===")
reference_idx = member_idx[0]
gt_calib, canvas_name, n_inliers, coverage = try_auto_calibrate(
    frames[reference_idx], BANK_DIR, mosaic_cfg)
if gt_calib is None:
    print(f"  no bank match for the segment reference frame (idx {reference_idx}) -- "
          f"cannot ground-truth this window. Sections 1/2 above still stand; "
          f"re-run on a window whose framing is in the bank.")
else:
    print(f"  ground truth from bank {canvas_name}: {n_inliers} inliers, "
          f"{coverage * 100:.0f}% coverage, source error "
          f"{gt_calib.source_max_reprojection_px:.2f}px")
    gt_sane = projected_field_is_sane(gt_calib, cfg)
    print(f"  ground truth passes projected_field_is_sane: {gt_sane}")
    if not gt_sane:
        print("  *** THIS GROUND TRUTH IS ITSELF BROKEN -- it projects the field as a bowtie. ***")
        print("  Distances below are only meaningful for lines the human actually clicked on")
        print("  real, visible paint; anything far from those clicks is comparing against")
        print("  nonsense. Re-do that bank canvas's calibration before reading the far-field")
        print("  numbers as a verdict on the detector. (This is how the bug was found: see")
        print("  results/field_lines_finding.md.)")
    # GROUND TRUTH, THE VERSION THAT SURVIVES A BROKEN HOMOGRAPHY: compare
    # against the human's actual CLICKED POINTS, not against lines projected
    # through the fitted calibration. A misidentified line poisons the whole
    # homography (see the bowtie above) but says nothing about the OTHER
    # lines, whose clicks still sit on real, visible paint -- verified by
    # their own per-line reprojection error. So "did the detector find the
    # same physical line the human clicked?" is answerable even when "is the
    # calibration right?" is a firm no.
    from frisbee_analysis.mosaic import register_pair_with_coverage
    from frisbee_analysis.calibration import Calibration
    canvas_calib = Calibration.load(os.path.join(BANK_DIR, canvas_name, "calibration.json"))
    canvas_image = cv2.imread(os.path.join(BANK_DIR, canvas_name, "canvas.jpg"))
    H_ref_to_canvas, _, _ = register_pair_with_coverage(
        canvas_image, frames[reference_idx], mosaic_cfg)
    if canvas_calib.line_details and H_ref_to_canvas is not None:
        H_canvas_to_ref = np.linalg.inv(H_ref_to_canvas)
        print("\n  --- vs the human's own clicked points (homography not involved) ---")
        for entry in canvas_calib.line_details:
            clicked = cv2.perspectiveTransform(
                np.asarray(entry["points"], np.float64).reshape(-1, 1, 2),
                H_canvas_to_ref).reshape(-1, 2)
            fitted = np.cross([clicked[0][0], clicked[0][1], 1.0],
                               [clicked[-1][0], clicked[-1][1], 1.0])
            fitted = fitted / np.hypot(fitted[0], fitted[1])
            human_angle = np.degrees(np.arctan2(clicked[-1][1] - clicked[0][1],
                                                 clicked[-1][0] - clicked[0][0])) % 180
            best = None
            for i, t in enumerate(tracked):
                pts = np.asarray(t.points)
                d = float(np.mean(np.abs(pts @ fitted[:2] + fitted[2])))
                # only compare with a line pointing the same way -- a
                # perpendicular line passing nearby is not "the same line"
                ta = np.degrees(np.arctan2(t.p2[1] - t.p1[1], t.p2[0] - t.p1[0])) % 180
                da = abs(ta - human_angle)
                da = min(da, 180 - da)
                if best is None or d < best[0]:
                    best = (d, i, da)
            print(f"  human line {str(entry['edge']):8s} ({len(clicked)} clicks): "
                  f"nearest tracked [{best[1]}] at {best[0]:6.1f}px, angle diff {best[2]:5.1f} deg")

    print("\n  --- vs lines projected through the ground-truth homography ---")
    named = {(1, 7): "left sideline", (2, 8): "right sideline", (1, 2): "near back line",
             (3, 4): "NEAR GOAL LINE", (5, 6): "FAR GOAL LINE", (7, 8): "far back line"}
    for edge in list(SIDELINE_EDGES) + list(CROSS_EDGES):
        vi, vj = edge
        p1 = np.array(field_to_pixel(*cfg.vertices[vi - 1], gt_calib))
        p2 = np.array(field_to_pixel(*cfg.vertices[vj - 1], gt_calib))
        true_line = np.cross([p1[0], p1[1], 1.0], [p2[0], p2[1], 1.0])
        true_line = true_line / np.hypot(true_line[0], true_line[1])
        on_screen = any(-frame_w <= p[0] <= 2 * frame_w and -frame_h <= p[1] <= 2 * frame_h
                        for p in (p1, p2))
        best_d, best_i, best_angle = None, None, None
        for i, t in enumerate(tracked):
            pts = np.asarray(t.points)
            d = float(np.mean(np.abs(pts @ true_line[:2] + true_line[2])))
            ta = np.degrees(np.arctan2(t.p2[1] - t.p1[1], t.p2[0] - t.p1[0])) % 180
            ga = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])) % 180
            da = abs(ta - ga)
            da = min(da, 180 - da)
            if best_d is None or d < best_d:
                best_d, best_i, best_angle = d, i, da
        flag = "" if on_screen else "   (projects off-frame -- not visible here)"
        if best_d is None:
            print(f"  {named[edge]:16s} no tracked lines at all{flag}")
        else:
            print(f"  {named[edge]:16s} nearest tracked line [{best_i}] at "
                  f"{best_d:8.1f}px, angle diff {best_angle:5.1f} deg{flag}")
    # unclipped on purpose: this image exists to SHOW the fold, so it must
    # not be the honest view that hides everything past the horizon
    cv2.imwrite("outputs/field_lines_groundtruth.jpg",
                draw_field_overlay(frames[reference_idx], gt_calib,
                                   clip_to_validity=False))

# --------------------------------------------------------------------------
print("\n=== 4. cone hints ===")
yolo = None
try:
    from ultralytics import YOLO
    yolo = YOLO("yolo11n.pt")
except Exception as exc:  # ultralytics is a Track B dep; keep the rest usable
    print(f"  (no YOLO available: {exc}) -- cone hints WILL be dominated by skin tone")
cone_hints = track_cone_hints(frames, regs, segment_id, overlay_boxes, yolo_model=yolo)
print(f"  cone hints persisting >=2 frames: {len(cone_hints)}")
for h in cone_hints:
    print(f"    ({h.x:8.1f},{h.y:8.1f}) area={h.area}")

# --------------------------------------------------------------------------
print("\n=== 5. what does the geometry say these lines ARE? ===")
proposals = rank_line_labelings(tracked, frame_w, frame_h, cone_hints, cfg,
                                 image_path=f"{video_path} t={start_sec:.0f}s (auto)")
if not proposals:
    print("  no labelling survived the geometry checks "
          "(rank / all-vertices-in-bounds / reprojection). That is the honest "
          "answer, not a crash -- see field_lines.rank_line_labelings.")
else:
    for rank, p in enumerate(proposals):
        edges = ", ".join(f"{e}" for e, _ in p["assignment"])
        determined = "over-determined" if p["over_determined"] else \
            "EXACTLY determined -- its reprojection error says nothing about the LABELLING"
        print(f"  #{rank + 1}: {p['n_lines']} lines -> edges {edges}; "
              f"max reproj {p['max_reproj_px']:.2f}px, mean {p['mean_reproj_px']:.2f}px, "
              f"cone hits {p['cone_hits']}/8  [{determined}]")
    best = proposals[0]["calibration"]
    cv2.imwrite("outputs/field_lines_proposal.jpg",
                draw_field_overlay(frames[reference_idx], best))
    if gt_calib is not None and not projected_field_is_sane(gt_calib, cfg):
        print("  (skipping the vs-ground-truth comparison: that ground truth is a bowtie, "
              "so any number here would be measuring one broken thing against another)")
    elif gt_calib is not None:
        errs = []
        for fx, fy in cfg.vertices:
            a = np.array(field_to_pixel(fx, fy, gt_calib))
            b = np.array(field_to_pixel(fx, fy, best))
            errs.append(float(np.hypot(*(a - b))))
        print(f"  top proposal vs ground truth, per-vertex pixel error: "
              f"median={np.median(errs):.0f}px  max={max(errs):.0f}px")
        print("  (both projected into the same reference frame; a correct "
              "labelling should be tens of px, a wrong one thousands)")
print("\nOverlays written to outputs/field_lines_*.jpg -- LOOK AT THEM. "
      "Every real fix in this project came from someone eyeballing an overlay.")
