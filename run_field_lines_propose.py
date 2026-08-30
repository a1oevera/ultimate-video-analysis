"""
Propose field lines automatically, let a human say what they ARE, fit the
calibration -- the working middle ground between run_calibrate.py (click
every line by hand) and full automation (which MEASURED as not achievable on
this footage; see results/field_lines_finding.md).

WHY A HUMAN STEP REMAINS. Detection and tracking work: on torovcanyon.mp4
t=3945-4005s the tracker recovers 3 of the 4 lines a human clicked to within
1.4-6.3px and 1.3 degrees, across a 60-frame pan, with no clicking at all.
LABELLING them does not work: four line correspondences determine a
homography exactly, so every candidate labelling reproduces its own lines
perfectly and the fit cannot tell you which one is right. Worse, a broadcast
framing often shows only ONE of the two sidelines, so a labelling has to
invent the other. `rank_line_labelings` proposes and ranks, but its top
answer was measured WRONG on real footage -- so it is offered here as a
starting suggestion, never as the default.

What this actually saves: the person no longer clicks points along lines
they have to find by eye in a foreshortened pan (which is how the bowtie in
results/field_lines_finding.md happened -- a sideline "clicked" on bare
turf where no line was painted). They look at numbered lines the detector
found in real paint, and say which is which.

USAGE -- two steps.

  1. Detect (slow: registers the window, same cost as run_mosaic_test.py):
       python3 run_field_lines_propose.py videos/torovcanyon.mp4 3945 60

     Writes outputs/line_detection/field_lines_candidates.jpg with every candidate line
     drawn and numbered, prints a table, and caches the result so step 2 is
     instant. LOOK AT THE IMAGE. Only label a line you can see paint under.

  2. Assign and fit (fast, reads the cache):
       python3 run_field_lines_propose.py videos/torovcanyon.mp4 3945 60 \\
           --assign 0=3,4 5=1,2 6=2,8 2=7,8

     Each `INDEX=vi,vj` names a line by its two field vertices, exactly the
     edges frisbee_analysis/ultimate_config.py defines:
       1,7 = left sideline     2,8 = right sideline
       1,2 = near back line    3,4 = NEAR GOAL LINE (endzone)
       5,6 = FAR GOAL LINE     7,8 = far back line
       11,12 = HALFWAY line (x=50m) -- on a shared pitch this is usually the
             most clearly painted cross-field line there is, and mistaking it
             for a back line halves the length scale of everything.
     You need >= 4, and they must not all be parallel on the real field --
     at least two of each direction, or the homography is under-determined
     no matter how many lines you give it.

  2b. A line the detector MISSED -- draw it yourself:
       ... --assign 0=1,7 1=3,4 --line 2,8=1546,377,2100,880

     Coordinates are CANVAS pixels, read straight off the 100px grid in
     outputs/line_detection/field_lines_grid.jpg. MEASURED (results/field_lines_finding.md
     Result 3.25): on t=3600+60s the detector found the blue near sideline
     and the goal line but produced NO candidate on the far sideline, which
     is plainly visible to a person. Without --line the window is dead: you
     cannot name a line that was never found, and the closest automated
     substitute passed all four gates while drawing the field along the
     treeline. A manual line contributes zero support, so it never inflates
     the "explains N% of detected paint" score.

Fits through calibration.fit_calibration_with_lines in LINES-ONLY mode (the
path validated exactly correct in results/calibration_finding.md Addendum 3;
its point/line mixing bug is never touched), then applies all three
independent checks -- reprojection error, vertices_within_bounds, and
projected_field_is_sane -- before offering to save.
"""
import sys
import os
import json
import cv2
import numpy as np

from frisbee_analysis import (UltimatePitchConfiguration, MosaicConfig, register_sequence,
                              track_lines, track_cone_hints, rank_line_labelings,
                              group_by_vanishing_point, segment_pan_polygon, refine_to_anchor,
                              confusable_groups, draw_extended, segment_canvas, transform_lines,
                              detect_conics, arc_fractions, split_merged_lines)
from frisbee_analysis.watchers import (collect_static_clusters, find_rows, WatcherRow,
                                       watchers_outside_field)
from frisbee_analysis.field_lines import (TrackedLine, ConeHint, projected_field_is_sane,
                                          playing_surface_mask)
from frisbee_analysis.field_model import (fit_field_model, build_field_model,
                                          draw_field_model, FIELD_EDGES, EDGE_NAMES as MODEL_EDGE_NAMES)
from frisbee_analysis.calibration import (fit_calibration, fit_calibration_with_lines,
                                          check_line_reprojection_error,
                                          vertices_within_bounds, draw_field_overlay,
                                          field_in_front_of_camera, field_covers_points,
                                          field_to_pixel,
                                          field_validity_region)
from frisbee_analysis.video_overlays import overlay_boxes_for

# outputs are filed into folders; make sure ours exist
os.makedirs("outputs/calibration", exist_ok=True)
os.makedirs("outputs/field_model", exist_ok=True)
os.makedirs("outputs/line_detection", exist_ok=True)

EDGE_NAMES = {(1, 7): "left sideline", (2, 8): "right sideline",
              (1, 2): "near back line", (3, 4): "NEAR GOAL LINE (endzone)",
              (5, 6): "FAR GOAL LINE (endzone)", (7, 8): "far back line",
              # The halfway line is a real, labellable marking at x=50m
              # (ultimate_config vertices 11 and 12) and on a shared pitch it
              # is usually PAINTED, unlike the ultimate goal lines. MEASURED
              # (Result 3.28): without it the only cross-field edges on offer
              # were goal and back lines, so a detected halfway line got
              # labelled "far back line" -- which places a 50m line at 100m
              # and halves the length scale of the whole calibration. The
              # person caught it by eye: "the far end line detected is
              # actually the halfway point".
              (11, 12): "HALFWAY line (x=50m)"}
CACHE_DIR = "outputs/field_lines_cache"


def parse_args(argv):
    positional, assignments, out_path = [], {}, "outputs/calibration/calibration_from_lines.json"
    sideline_colour = None
    manual = []
    max_lines, min_frames = 12, 3
    corners = {}
    drop_arcs = False
    i = 1
    while i < len(argv):
        token = argv[i]
        if token == "--sideline-colour":
            sideline_colour = argv[i + 1]
            i += 2
            continue
        if token == "--assign":
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                index, _, edge = argv[i].partition("=")
                vi, _, vj = edge.partition(",")
                assignments[int(index)] = (int(vi), int(vj))
                i += 1
            continue
        if token == "--drop-arcs":
            drop_arcs = True; i += 1
            continue
        if token == "--corner":
            # A FIELD VERTEX given directly as a pixel, in canvas coords off
            # outputs/line_detection/field_lines_grid.jpg. Four of these fit a homography on
            # their own through the points-only path, which is the one
            # validated exactly correct in results/calibration_finding.md --
            # no line labelling, no gate stack, no ranking.
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                v, _, coords = argv[i].partition("=")
                cx, cy = (float(q) for q in coords.split(","))
                corners[int(v)] = (cx, cy)
                i += 1
            continue
        if token == "--max-lines":
            max_lines = int(argv[i + 1]); i += 2
            continue
        if token == "--min-frames":
            min_frames = int(argv[i + 1]); i += 2
            continue
        if token == "--line":
            # A line the DETECTOR MISSED, supplied by the person from the grid
            # image. Without this, --assign can only name lines that were
            # found -- so a sideline the person can plainly see but the
            # detector never fired on is simply unsayable, and the whole
            # window is unusable. Coordinates are CANVAS pixels, straight off
            # outputs/check_grid.jpg.
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                edge, _, coords = argv[i].partition("=")
                vi, _, vj = edge.partition(",")
                x1, y1, x2, y2 = (float(v) for v in coords.split(","))
                manual.append(((int(vi), int(vj)), (x1, y1), (x2, y2)))
                i += 1
            continue
        if token == "--out":
            out_path = argv[i + 1]
            i += 2
            continue
        positional.append(token)
        i += 1
    return positional, assignments, out_path, sideline_colour, manual, max_lines, min_frames, corners, drop_arcs


(positional, assignments, out_path, sideline_colour, manual_lines,
 max_lines, min_frames, corners, drop_arcs) = parse_args(sys.argv)
# The person knows their own field. On torovcanyon.mp4 the sidelines are the BLUE
# lines; saying so lets ranking prefer labellings that agree, without ever putting
# that fact into the geometry.
colour_prior = ({sideline_colour: [(1, 7), (2, 8)]} if sideline_colour else None)
video_path = positional[0] if positional else "videos/torovcanyon.mp4"
start_sec = float(positional[1]) if len(positional) > 1 else 3945.0
duration_sec = float(positional[2]) if len(positional) > 2 else 60.0
sample_fps = float(positional[3]) if len(positional) > 3 else 1.0
cfg = UltimatePitchConfiguration()

os.makedirs(CACHE_DIR, exist_ok=True)
cache_key = f"{os.path.basename(video_path)}_{start_sec:.0f}_{duration_sec:.0f}_{sample_fps:g}"
if (max_lines, min_frames) != (12, 3):
    cache_key += f"_m{max_lines}f{min_frames}"
cache_path = os.path.join(CACHE_DIR, cache_key + ".json")
reference_path = os.path.join(CACHE_DIR, cache_key + "_ref.jpg")
pan_path = os.path.join(CACHE_DIR, cache_key + "_pan.jpg")


def detect_and_cache():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"could not open {video_path}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
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
    overlay_boxes = overlay_boxes_for(video_path, frame_w, frame_h)
    mosaic_cfg = MosaicConfig(overlay_mask_boxes=tuple(overlay_boxes))
    print(f"Registering {len(frames)} frames (same cost as run_mosaic_test.py)...")
    regs = register_sequence(frames, mosaic_cfg)
    by_segment = {}
    for r in regs:
        if r.ok and r.homography is not None:
            by_segment.setdefault(r.segment_id, []).append(r.frame_idx)
    if not by_segment:
        sys.exit("no frames registered -- try another window")
    segment_id = max(by_segment, key=lambda s: len(by_segment[s]))
    print(f"Segments: { {k: len(v) for k, v in by_segment.items()} }; using {segment_id} "
          f"({len(by_segment[segment_id])} frames)")

    # MEASURED (results/field_lines_finding.md Result 3.22): register_sequence
    # chains frame-to-frame, so error compounds -- homographies diverge from a
    # direct anchor registration by a median 2.2px at chain depth <=20 and
    # 7.7px past depth 40. That is the whole gap between human-clicked lines
    # (0.5-2.7px) and tracked lines built on the chain (rms 4.0-8.4px).
    # refine_to_anchor re-registers what it can directly to the anchor and
    # routes the rest through one hub, cutting chain depth from "up to N" to
    # 1 or 2.
    print("Shortening registration chains (refine_to_anchor)...")
    regs = refine_to_anchor(frames, regs, mosaic_cfg, progress=True)
    print("Detecting and tracking lines across the pan...")
    tracked = track_lines(frames, regs, segment_id, overlay_boxes,
                           min_frames=min_frames, max_lines=max_lines)
    yolo = None
    try:
        from ultralytics import YOLO
        yolo = YOLO("yolo11n.pt")
    except Exception as exc:
        print(f"  (no YOLO: {exc}) -- cone hints will be dominated by skin tone, "
              f"see field_lines.detect_cone_hints")
    cones = track_cone_hints(frames, regs, segment_id, overlay_boxes, yolo_model=yolo)
    # what the camera actually swept, in the same coordinates the lines live in.
    # A broadcast shot never frames the whole field; without this the labelling
    # search has no way to know which parts of the field it could even have seen.
    pan_polygon = segment_pan_polygon(frames, regs, segment_id)

    # People as field evidence (see frisbee_analysis/watchers.py). A dense,
    # static, collinear row of people is a sideline that has no paint -- and
    # `watchers_outside_field` is the only signal measured so far that
    # separates a true sideline from a near-parallel neighbour.
    homography_by_idx = {r.frame_idx: r.homography for r in regs if r.ok}
    watcher_rows = []
    if yolo is not None:
        anchor_frame = frames[next((i for i in by_segment[segment_id]
                                    if np.allclose(homography_by_idx[i], np.eye(3))),
                                   by_segment[segment_id][0])]
        surface = playing_surface_mask(anchor_frame, overlay_boxes)
        print("Finding static rows of people (watchers mark unpainted sidelines)...")
        clusters = collect_static_clusters(frames, regs, segment_id, yolo,
                                            cluster_px=12.0, min_frames=20)
        watcher_rows = find_rows(clusters, surface_mask=surface, min_points=10)
        print(f"  {len(clusters)} static standing positions -> {len(watcher_rows)} row(s)")

    # the segment's reference frame is the one whose homography is identity --
    # its raw pixels ARE the coordinate system the tracked lines live in
    # (NOT necessarily the first index: mosaic._reclaim_failed_frames can fold
    # an earlier frame into a later segment's anchor)
    anchor = next((i for i in by_segment[segment_id]
                   if np.allclose(homography_by_idx[i], np.eye(3))), by_segment[segment_id][0])
    cv2.imwrite(reference_path, frames[anchor])

    # The labelling image must show everything the TRACKER knows about, not
    # everything one frame happened to contain. track_lines already fuses the
    # whole pan into these coordinates, so candidates routinely land outside
    # the anchor frame -- and a person cannot label a line they cannot see.
    print("Compositing the pan into one labelling canvas...")
    canvas, canvas_T = segment_canvas(frames, regs, segment_id)
    if canvas is not None:
        cv2.imwrite(pan_path, canvas)
        print(f"  pan canvas {canvas.shape[1]}x{canvas.shape[0]}px "
              f"(anchor frame is {frame_w}x{frame_h})")

    payload = {"video": video_path, "start_sec": start_sec, "duration_sec": duration_sec,
               "sample_fps": sample_fps, "frame_w": frame_w, "frame_h": frame_h,
               "segment_id": segment_id, "n_frames_in_segment": len(by_segment[segment_id]),
               "lines": [{"p1": list(t.p1), "p2": list(t.p2), "points": [list(p) for p in t.points],
                           "frame_indices": t.frame_indices, "n_frames": t.n_frames,
                           "support": t.total_support_px, "rms_fit_px": t.rms_fit_px,
                           "colour": t.colour}
                          for t in tracked],
               "cones": [{"x": c.x, "y": c.y, "area": c.area} for c in cones],
               "pan_polygon": [list(p) for p in pan_polygon],
               "canvas_T": (None if canvas is None else [list(map(float, r)) for r in canvas_T]),
               "watcher_rows": [{"line": list(map(float, r.line)),
                                  "points": [list(p) for p in r.points],
                                  "rms_px": r.rms_px, "n_frames": r.n_frames,
                                  "extent_px": r.extent_px,
                                  "median_box_h": r.median_box_h,
                                  "fraction_on_surface": r.fraction_on_surface}
                                 for r in watcher_rows]}
    with open(cache_path, "w") as f:
        json.dump(payload, f)
    return payload


if os.path.exists(cache_path) and (assignments or manual_lines or corners):
    with open(cache_path) as f:
        payload = json.load(f)
    print(f"Using cached detection: {cache_path}")
else:
    payload = detect_and_cache()

tracked = [TrackedLine(p1=tuple(d["p1"]), p2=tuple(d["p2"]),
                        points=[tuple(p) for p in d["points"]],
                        frame_indices=d["frame_indices"], n_frames=d["n_frames"],
                        total_support_px=d["support"], rms_fit_px=d["rms_fit_px"],
                        colour=d.get("colour", "unknown"))
           for d in payload["lines"]]
# Applied here too, not only in track_lines, so caches written before
# splitting existed still benefit. Idempotent: an already-split line fits
# under the rms threshold and is returned untouched.
_before = len(tracked)
tracked = split_merged_lines(tracked)
if len(tracked) != _before:
    print(f"split {len(tracked) - _before} extra candidate(s) out of merged clusters "
          f"({_before} -> {len(tracked)}); see field_lines.split_merged_lines")
tracked.sort(key=lambda t: (-t.n_frames, -t.total_support_px))
cones = [ConeHint(c["x"], c["y"], c["area"]) for c in payload["cones"]]
frame_w, frame_h = payload["frame_w"], payload["frame_h"]
pan_polygon = [tuple(p) for p in payload.get("pan_polygon", [])]
watcher_rows = [WatcherRow(line=np.asarray(w["line"], dtype=np.float64),
                            points=[tuple(p) for p in w["points"]],
                            rms_px=w["rms_px"], n_frames=w["n_frames"],
                            extent_px=w["extent_px"], median_box_h=w["median_box_h"],
                            fraction_on_surface=w["fraction_on_surface"])
                for w in payload.get("watcher_rows", [])]
reference_frame = cv2.imread(reference_path)

# The wide pan composite, and the transform that puts reference coordinates on
# it. Candidates are drawn HERE rather than on the anchor frame: the tracker
# pools every frame of the pan, so lines that were never simultaneously in
# shot are all in the list, and on a single frame most of them fall off the
# edge. Falls back to the anchor frame for caches written before this existed.
canvas_T = payload.get("canvas_T")
pan_canvas = cv2.imread(pan_path) if os.path.exists(pan_path) else None
if pan_canvas is not None and canvas_T is not None:
    canvas_T = np.asarray(canvas_T, dtype=np.float64)
    label_image, label_lines = pan_canvas, transform_lines(tracked, canvas_T)
    label_cones = [ConeHint(x, y, c.area) for c, (x, y) in
                   zip(cones, cv2.perspectiveTransform(
                       np.array([[[c.x, c.y]] for c in cones], dtype=np.float64),
                       canvas_T).reshape(-1, 2))] if cones else []
else:
    label_image, label_lines, label_cones = reference_frame, tracked, cones
    canvas_T = None

# Curved markings (soccer circles, penalty arcs) get shattered by the tracker
# into short chords that each look like a perfectly good straight line. Six of
# the twelve goal-line crossings in this window were chords of one circle.
arc_frac = [0.0] * len(tracked)
conics = []
if label_image is not None and len(tracked):
    conics = detect_conics(label_image)
    arc_frac = arc_fractions(transform_lines(tracked, canvas_T) if canvas_T is not None
                              else tracked, conics)
    if conics:
        k = conics[0]
        print(f"\nCurved marking detected: centre ({k['centre'][0]:.0f},{k['centre'][1]:.0f}) "
              f"semi-axes {k['semi_axes'][0]:.0f}/{k['semi_axes'][1]:.0f}px, "
              f"{k['coverage_deg']:.0f} deg of arc, {k['n_inliers']} pixels.")
        print("  Candidates with a high 'arc' fraction are chords of it, NOT field lines.")
if drop_arcs:
    keep = [i for i, f in enumerate(arc_frac)
            if not (f >= 0.25 and f * len(tracked[i].points) >= 5)]
    dropped = [i for i in range(len(tracked)) if i not in keep]
    if dropped:
        print(f"--drop-arcs: removing {dropped} (>=25% of points on the curve, >=5 points)")
    tracked = [tracked[i] for i in keep]
    arc_frac = [arc_frac[i] for i in keep]

def _label_at(i):
    """Canvas pixel where candidate i's NUMBER is drawn. Printed because
    cross-referencing the table against the image by reconstructing the
    ordering is exactly how an off-by-one sent an entire round of analysis
    onto the wrong line -- the person had to say "it is nowhere near the arc"
    before it was caught. Now the table names the same coordinates the eye
    reads off the picture."""
    t = (label_lines[i] if canvas_T is not None else tracked[i])
    return f"({(t.p1[0] + t.p2[0]) / 2:.0f},{(t.p1[1] + t.p2[1]) / 2:.0f})"


if not assignments and not manual_lines and not corners:
    family_a, family_b, vp = group_by_vanishing_point(tracked)
    family_of = {}
    for t in family_a:
        family_of[id(t)] = "A"
    for t in family_b:
        family_of[id(t)] = "B"
    print(f"\n{len(tracked)} candidate lines (tracked across "
          f"{payload['n_frames_in_segment']} frames of the pan):")
    print(f"  {'idx':>3} {'dir':>3} {'paint':>6} {'seen':>7} {'support':>8} {'rms':>6} {'arc':>5} "
          f"{'label at (canvas)':>18}  endpoints (segment reference px)")
    for i, t in enumerate(tracked):
        print(f"  {i:>3} {family_of.get(id(t), '?'):>3} {t.colour:>6} "
              f"{t.n_frames:>3}/{payload['n_frames_in_segment']:<3} {t.total_support_px:>8} "
              f"{t.rms_fit_px:>6.2f} {arc_frac[i]:>5.0%} "
              f"{_label_at(i):>18}  ({t.p1[0]:8.0f},{t.p1[1]:7.0f}) -> "
              f"({t.p2[0]:8.0f},{t.p2[1]:7.0f})")
    print("  'paint' is the marking's own colour -- on a multi-sport field this is often the")
    print("  fastest way to tell lines apart (e.g. a blue sideline vs white endzone paint).")
    print("  'dir' A/B are the two real-world directions (shared vanishing point). A valid")
    print("  assignment needs >= 2 lines from EACH -- lines that are all parallel on the")
    print("  field cannot pin down a homography, however many of them you have.")
    if vp is None:
        print("  WARNING: no two-direction split found. This framing probably shows only one")
        print("  field direction; widen the window so the pan covers more, or use run_calibrate.py.")
    # THE failure this project keeps hitting, and the one no automated score has
    # caught (see field_lines.confusable_groups): a true field line and a
    # near-parallel neighbour behind it. On this footage the far sideline's
    # endzone paint has the turf edge running behind the row of standing
    # players a few px away -- and support, length, persistence, flank-matching
    # and vanishing-point structure ALL prefer the wrong one. What resolves it
    # is seeing both extended across the frame, where an 11px gap at the paint
    # becomes 73px at the far edge. So surface the pairs and draw them.
    groups = confusable_groups(tracked)
    if groups:
        print(f"\n*** {len(groups)} group(s) of CONFUSABLE lines -- near-parallel candidates that")
        print("    could be mistaken for each other. No score in this module can tell them")
        print("    apart; a longer, straighter, higher-contrast line just off the real one")
        print("    (a turf edge, a neighbouring pitch's marking, the row your team stands on)")
        print("    wins every automated measure while being the wrong line. LOOK at these:")
        for n, group in enumerate(groups):
            print(f"    group {n}:")
            for i in group:
                t = tracked[i]
                print(f"      [{i:2d}] {t.colour:6s} seen={t.n_frames:2d}/"
                      f"{payload['n_frames_in_segment']:<3d} support={t.total_support_px:7d} "
                      f"rms={t.rms_fit_px:5.2f}")
            for a_i in range(len(group)):
                for b_i in range(a_i + 1, len(group)):
                    a, b = group[a_i], group[b_i]
                    la, lb = tracked[a].homogeneous(), tracked[b].homogeneous()
                    if abs(la[1]) > 1e-9 and abs(lb[1]) > 1e-9:
                        gap0 = abs((-(la[2]) / la[1]) - (-(lb[2]) / lb[1]))
                        gap1 = abs((-(la[0] * frame_w + la[2]) / la[1])
                                    - (-(lb[0] * frame_w + lb[2]) / lb[1]))
                        print(f"        [{a}] vs [{b}]: {gap0:5.0f}px apart at x=0, "
                              f"{gap1:5.0f}px at x={frame_w}")
            path = f"outputs/line_detection/field_lines_confusable_{n}.jpg"
            cv2.imwrite(path, draw_extended(label_image, [label_lines[i] for i in group],
                                             [f"line {i}" for i in group]))
            print(f"        -> {path}")

    print(f"\n{len(cones)} cone hints (corroboration only, never fitted):")
    for c in cones:
        print(f"    ({c.x:8.1f},{c.y:8.1f}) area={c.area}")

    annotated = label_image.copy()
    for i, t in enumerate(label_lines):
        p1 = tuple(int(round(v)) for v in t.p1)
        p2 = tuple(int(round(v)) for v in t.p2)
        colour_bgr = {"blue": (255, 120, 0), "white": (0, 0, 255)}.get(t.colour, (0, 165, 255))
        cv2.line(annotated, p1, p2, colour_bgr, 2)
        mid = (int((t.p1[0] + t.p2[0]) / 2), int((t.p1[1] + t.p2[1]) / 2))
        cv2.putText(annotated, str(i), mid, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
    for c in label_cones:
        cv2.circle(annotated, (int(round(c.x)), int(round(c.y))), 14, (255, 255, 0), 2)
    # A gridded copy, so a line the detector MISSED can still be given to the
    # tool: read two points off the grid and pass them to --line.
    grid = label_image.copy()
    gh, gw = grid.shape[:2]
    for gx in range(0, gw, 100):
        cv2.line(grid, (gx, 0), (gx, gh), (200, 200, 200), 1)
        for gc, gt in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(grid, str(gx), (gx + 3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gc, gt)
    for gy in range(0, gh, 100):
        cv2.line(grid, (0, gy), (gw, gy), (200, 200, 200), 1)
        for gc, gt in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(grid, str(gy), (3, gy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gc, gt)
    cv2.imwrite("outputs/line_detection/field_lines_grid.jpg", grid)

    candidates_path = "outputs/line_detection/field_lines_candidates.jpg"
    cv2.imwrite(candidates_path, annotated)
    print(f"\nWrote {candidates_path} -- OPEN IT.")
    if canvas_T is not None:
        print(f"  This is the WHOLE PAN composited ({label_image.shape[1]}x"
              f"{label_image.shape[0]}px), not one frame. You do NOT need a single frame")
        print("  showing all four lines -- the tracker fuses every frame of the pan into")
        print("  one coordinate system, so lines the camera saw at different moments are")
        print("  all here and all labellable together.")
    else:
        print("  (no pan canvas in this cache -- re-run step 1 to rebuild it.) Lines drawn")
        print("  off the edge were seen on other frames of the pan; still valid, just not")
        print("  visible here.")
    print("Also wrote outputs/line_detection/field_lines_grid.jpg -- the same image with a 100px grid.")
    print("  If a line you can SEE has no candidate on it, read two points off that grid")
    print("  and pass them directly, e.g.:  --line 2,8=1546,377,2100,880")

    print("\nFitting complete field models (six connected lines) ...")
    if watcher_rows:
        print(f"\n{len(watcher_rows)} static row(s) of people -- used as a labelling GATE")
        print("    (watchers stand OUTSIDE the boundary, so a labelling that puts one inside")
        print("     the field is wrong; this is the only signal measured so far that")
        print("     separates a true sideline from a near-parallel neighbour):")
        for n, r in enumerate(watcher_rows):
            print(f"      row {n}: {len(r.points):3d} standing positions, rms={r.rms_px:5.2f}px, "
                  f"extent={r.extent_px:6.0f}px, median box={r.median_box_h:5.1f}px, "
                  f"on_surface={r.fraction_on_surface * 100:3.0f}%")
        print("      NOTE a row is not automatically a sideline -- the fence/path beyond the")
        print("      field is STRAIGHTER than the players. Separate on box height, not rms.")
    models = fit_field_model(tracked, frame_w, frame_h, cones, cfg, top_k=3,
                             pan_polygon=pan_polygon, colour_prior=colour_prior,
                             watcher_rows=watcher_rows)
    if models:
        print("Auto-ranked field models -- STARTING POINTS ONLY. Ranked by evidence the fit")
        print("did not itself use (how much detected paint the six lines account for, cones,")
        print("how many of the six are directly observed). Confirm against the image: the")
        print("top-ranked model has measured WRONG on this footage before.")
        for rank, m in enumerate(models):
            pairs = " ".join(f"{tracked.index(line)}={e[0]},{e[1]}" for e, line in m.assignment)
            print(f"\n  MODEL #{rank + 1}: {m.n_observed}/6 observed, cones {m.cone_hits}/8, "
                  f"explains {m.explained_support_fraction * 100:.0f}% of detected paint, "
                  f"max reproj {m.max_reproj_px:.1f}px")
            if m.extent is not None:
                print(f"      view: {m.extent.description}")
            for l in m.lines:
                residual = "     -" if not np.isfinite(l.residual_px) else f"{l.residual_px:5.1f}px"
                print(f"      {l.name:16s} {l.source:9s} {l.visibility:8s} {l.colour:6s} "
                      f"residual={residual}")
            print(f"      --assign {pairs}")
        cv2.imwrite("outputs/field_model/field_model_auto.jpg",
                    draw_field_model(reference_frame, models[0], cfg))
        print("\n  Top model drawn to outputs/field_model/field_model_auto.jpg "
              "(solid = observed paint, dashed = estimated from WFDF geometry).")
    else:
        print("No field model survived the geometry checks. That is a real answer: either this")
        print("framing lacks both field directions, or the candidates are wrong. Six lines")
        print("cannot be estimated from evidence that doesn't determine a field.")
    print("\nThen run, with YOUR OWN reading of the image:")
    print(f"  python3 {sys.argv[0]} {video_path} {start_sec:.0f} {duration_sec:.0f} "
          f"{sample_fps:g} --assign 0=3,4 ...")
    sys.exit(0)

# ---------------------------------------------------------------- fit ------
# Four field VERTICES given directly. This bypasses line labelling entirely:
# the person names corners, which is what they have repeatedly turned out to
# be able to state precisely ("near x=1300, y=400 is the front right corner
# of the endzone") while no automated pick of the corresponding LINE has ever
# survived inspection. Vertices 1,2,3,4 are the near endzone's four corners.
if corners:
    inv = np.linalg.inv(canvas_T) if canvas_T is not None else np.eye(3)
    if len(corners) < 4:
        sys.exit(f"--corner needs >= 4 vertices to fit a homography, got {len(corners)}: "
                 f"{sorted(corners)}")
    ks = sorted(corners)
    pix = cv2.perspectiveTransform(
        np.array([[corners[k]] for k in ks], dtype=np.float64), inv).reshape(-1, 2)
    print("Fitting from CORNERS (points-only path):")
    for k, q in zip(ks, pix):
        print(f"  vertex {k:>2} {str(tuple(cfg.vertices[k - 1])):>14} cm  <-  "
              f"canvas {corners[k]} = ref ({q[0]:.0f},{q[1]:.0f})")
    calib = fit_calibration(image_points=[tuple(map(float, q)) for q in pix],
                             keypoint_indices=ks, image_path=
                             f"{video_path} t={start_sec:.0f}-{start_sec + duration_sec:.0f}s "
                             f"(corners, human-labelled)", cfg=cfg)
    lines, edges = [], []
    err = [np.hypot(*(np.array(field_to_pixel(*cfg.vertices[k - 1], calib)) - q))
           for k, q in zip(ks, pix)]
    print(f"\nreprojection: max {max(err):.2f}px mean {sum(err) / len(err):.2f}px "
          f"over {len(ks)} corners")
    print("  NOTE four corners determine a homography EXACTLY, so this error describes")
    print("  nothing but arithmetic. Give a 5th vertex for a number that means something.")
else:
    calib = None

# Lines the person supplied because the detector missed them. MEASURED on
# torovcanyon.mp4 t=3600+60s: the detector found the blue near sideline and
# the goal line, but produced no candidate on the far sideline the person
# could plainly see -- and every automated pick for it (line 6, closest to
# the cone-marked corner, 84% of paint explained, all four gates passing)
# drew the field along the treeline. A tool that can only name what it found
# has no move here, so let the person draw it.
if manual_lines:
    inv = np.linalg.inv(canvas_T) if canvas_T is not None else np.eye(3)
    if canvas_T is None:
        print("NOTE: no pan canvas for this window -- treating --line coordinates as")
        print("      REFERENCE-frame pixels, not canvas pixels.")
    for edge, c1, c2 in manual_lines:
        p1, p2 = (tuple(map(float, q)) for q in
                  cv2.perspectiveTransform(np.array([[c1], [c2]], dtype=np.float64),
                                            inv).reshape(-1, 2))
        # support 0 and n_frames 0 are deliberate and honest: no paint was
        # detected under this line, so it must not inflate the
        # explains-N%-of-paint score that ranking and the gates read.
        tracked.append(TrackedLine(p1=p1, p2=p2, points=[p1, p2], frame_indices=[],
                                    n_frames=0, total_support_px=0, rms_fit_px=0.0,
                                    colour="manual"))
        assignments[len(tracked) - 1] = edge
        print(f"  manual line {len(tracked) - 1} -> {edge} "
              f"{EDGE_NAMES.get(tuple(edge), '(not a known field edge!)')}: "
              f"canvas {c1} -> {c2}  = ref ({p1[0]:.0f},{p1[1]:.0f}) -> ({p2[0]:.0f},{p2[1]:.0f})")

if calib is None:
  for index in assignments:
    if not (0 <= index < len(tracked)):
        sys.exit(f"line index {index} doesn't exist -- there are {len(tracked)} candidates")
  if len(assignments) < 4:
    sys.exit(f"need >= 4 lines, got {len(assignments)}")
  edges = [assignments[i] for i in sorted(assignments)]
  lines = [tracked[i] for i in sorted(assignments)]
  print("Fitting from:")
  for i, edge in zip(sorted(assignments), edges):
    print(f"  line {i:>3} -> {edge}  {EDGE_NAMES.get(tuple(edge), '(not a known field edge!)')}")

  try:
    calib = fit_calibration_with_lines(
        image_points=[], keypoint_indices=[],
        image_path=f"{video_path} t={start_sec:.0f}-{start_sec + duration_sec:.0f}s "
                    f"(field_lines proposal, human-labelled)",
        line_image_points=[list(l.points) for l in lines],
        line_field_edges=edges, cfg=cfg)
  except ValueError as exc:
    sys.exit(f"\nFIT REFUSED: {exc}")

worst = 0.0
if calib.line_details:
    print("\nPer-line reprojection error (perpendicular distance of each detected point from "
          "its own fitted line):")
    for res in check_line_reprojection_error(calib, cfg):
        worst = max(worst, res["max"])
        print(f"  {str(tuple(res['edge'])):8s} {EDGE_NAMES.get(tuple(res['edge']), ''):26s} "
              f"max={res['max']:6.2f}px mean={res['mean']:6.2f}px over {len(res['errors'])} points")
    calib.source_max_reprojection_px = float(worst)
    if len(lines) == 4:
        print("  NOTE: exactly 4 lines determine a homography exactly, so these numbers describe how")
        print("  STRAIGHT the detected lines are, not whether your labelling is right. Add a 5th.")

in_bounds = vertices_within_bounds(calib, frame_w, frame_h, cfg)
shape_ok = projected_field_is_sane(calib, cfg)
in_front = field_in_front_of_camera(calib, cfg)
covers = field_covers_points(calib, [p for l in lines for p in l.points], cfg)
print(f"\nvertices_within_bounds  : {in_bounds}")
print(f"projected_field_is_sane : {shape_ok}")
_v = field_validity_region(calib, cfg)
print(f"field_in_front_of_camera: {in_front}"
      + ("" if in_front else f"   (LOCAL: usable to x={_v['usable_x_max']/100:.1f}m, "
                             f"{_v['usable_fraction']*100:.0f}% of the field -- not a defect)"))
print(f"field_covers_points     : {covers}")
if not in_front:
    print("  -> the field passes the horizon partway along, so this calibration is LOCAL:")
    print("     trust it out to the x above and not beyond. Normal for a shot that only")
    print("     framed one end; calibrate a second window for the far half.")
if not covers:
    print("  -> the fitted field does not contain the pixels it was fitted from, so it has")
    print("     explained nothing. Usually a labelling that collapses the field to a sliver.")
if not shape_ok:
    print("  -> the fitted field self-intersects or is inside-out. At least one line is")
    print("     labelled wrong. This is exactly the failure that produced bank canvas_002's")
    print("     bowtie (results/field_lines_finding.md) -- re-check the candidates image.")
if not in_bounds:
    print("  -> some field vertex projects absurdly far outside the frame; usually a line")
    print("     whose detected points are too close together to constrain the far field.")

model = build_field_model(calib, tracked, cones, list(zip(edges, lines)), cfg,
                          max_reproj_px=float(worst), frame_w=frame_w, frame_h=frame_h,
                          pan_polygon=pan_polygon)
if model.extent is not None:
    print(f"\nWHAT THIS SHOT ACTUALLY COVERS: {model.extent.description}")
    print(f"  one frame spans {model.extent.frame_length_fraction * 100:.0f}% of the field's "
          f"length; the whole pan spans {model.extent.pan_length_fraction * 100:.0f}%")
print("\nTHE SIX FIELD LINES (this is the output -- they connect, by construction):")
for l in model.lines:
    residual = "     -" if not np.isfinite(l.residual_px) else f"{l.residual_px:5.1f}px"
    print(f"  {l.name:16s} {l.source:9s} {l.visibility:8s} {l.colour:6s} residual={residual} "
          f"({l.p1[0]:8.0f},{l.p1[1]:7.0f}) -> ({l.p2[0]:8.0f},{l.p2[1]:7.0f})")
print(f"  {model.n_observed}/6 directly observed; the rest are computed from the fitted")
print(f"  homography and WFDF dimensions -- exact IF the labelling above is right.")
print(f"  cone hints on boundary corners: {model.cone_hits}/8 "
      f"(of {model.n_cone_hints} hints found)")
print(f"  this model accounts for {model.explained_support_fraction * 100:.0f}% of all "
      f"detected paint")

overlay_path = out_path.rsplit(".", 1)[0] + "_overlay.jpg"
cv2.imwrite(overlay_path, draw_field_model(reference_frame, model, cfg))
calib.save(out_path)
print(f"\nSaved calibration to {out_path}")
print(f"Saved the six-line field model to {overlay_path} -- CHECK IT BY EYE (solid = observed")
print("paint, dashed = estimated). All three checks passing is")
print("necessary, not sufficient; every real bug in this project was caught on an overlay.")
if in_bounds and covers and worst <= 20:
    print("\nAll checks passed. To put this in the calibration bank so future segments "
          "auto-match against it:")
    print("  python3 -c \"import cv2; from frisbee_analysis.calibration import Calibration; "
          "from frisbee_analysis.calibration_bank import add_entry; "
          f"add_entry('outputs/calibration_bank', cv2.imread('{reference_path}'), "
          f"Calibration.load('{out_path}'))\"")
else:
    reasons = []
    if not in_bounds:
        reasons.append("vertices_within_bounds failed")
    if not covers:
        reasons.append("field_covers_points failed")
    if worst > 20:
        reasons.append(f"worst line reprojection {worst:.1f}px > 20px")
    print(f"\nNOT bank-ready: {'; '.join(reasons)}.")
