"""
The field as ONE object: exactly six lines that connect.

WHY THIS REPLACES A LIST OF DETECTIONS. field_lines.py answers "what paint is
visible?", which on a real multi-sport pitch is a list of a dozen fragments --
some ultimate boundaries, some soccer/lacrosse markings, some the same line
seen twice through registration drift. That is the wrong shape of answer. An
ultimate field has SIX boundary lines and they are not independent: two
sidelines, two end lines, two goal lines, meeting at eight corners, with every
distance fixed by WFDF Appendix A1.2.2 (100m x 37m, 18m endzones). Once any
four independent lines are identified, the other two are DETERMINED -- there
is nothing left to detect.

So this module emits the six, always, each tagged `observed` or `estimated`:
  - observed  -- a tracked detection supports this line
  - estimated -- no paint was detectable there, so its position is computed
                 from the fitted homography and the known field dimensions
An estimated line is not a guess. It is where the line MUST be if the observed
ones are what they are called. That distinction is exactly why the tag exists:
`estimated` says "trusted only as far as the labelling and the homography are
trusted", and this footage has already produced one calibration where that
trust was misplaced (see results/field_lines_finding.md).

MEASURED NEED (person's own description of torovcanyon.mp4): "the close
sideline is a blue line and the far sideline can be detected by a white line
but the white line is only painted on the endzone... in the middle of the
field the far sideline does not exist". Half the boundary of that field is
not painted at all. Estimation is not a convenience here, it is the only way
the far sideline exists as a line across midfield.

HOW THE LABELLING IS CHOSEN, and why this scores better than reprojection
error. With exactly four line correspondences a homography is determined
exactly, so every candidate labelling reproduces its own four lines
perfectly -- reprojection error cannot rank them (measured: a deliberately
scrambled labelling scored 7.11px and passed every check). The evidence that
CAN rank them is the evidence the fit did not use:
  - the OTHER detected lines. A correct labelling predicts where the
    remaining field boundaries are, and real paint should be sitting there.
    Support-weighted, so a 400,000-support sideline counts for more than a
    100px fragment.
  - the cone hints. Correct geometry puts the eight boundary cones on top of
    actual orange blobs.
  - the field's own shape, via projected_field_is_sane.
None of these are inputs to the fit, so unlike reprojection error they are
not guaranteed to look good for a wrong answer.

ON DEPTH: monocular depth (DepthAnything) is deliberately NOT used -- it is
held for a later milestone. It would also be redundant for this particular
job: the two vanishing points of a rectangular field already fix the ground
plane's orientation, and WFDF's dimensions fix its scale. Depth becomes
interesting for things off the ground plane (player height, disc altitude),
not for where a line on the grass is.
"""
from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass
from itertools import permutations
from .ultimate_config import UltimatePitchConfiguration
# one dash helper for both drawings, so draw_field_model and
# calibration.draw_field_overlay mean the same thing by the same marks
from .calibration import draw_dashed_line as _dashed
from .watchers import watchers_outside_field
from .field_lines import (SIDELINE_EDGES, CROSS_EDGES, group_by_vanishing_point,
                          direction_hypotheses, sideline_family_candidates,
                          projected_field_is_sane, no_parallel_crossings)


# The six, and nothing else. Ordered so that drawing them in sequence walks
# the field: boundary first, then the two goal lines across it.
FIELD_EDGES = ((1, 7), (2, 8), (1, 2), (7, 8), (3, 4), (5, 6))
EDGE_NAMES = {
    (1, 7): "left sideline",
    (2, 8): "right sideline",
    (1, 2): "near end line",
    (7, 8): "far end line",
    (3, 4): "near goal line",
    (5, 6): "far goal line",
}
END_LINE_EDGES = ((1, 2), (7, 8))


@dataclass
class ModelLine:
    """One of the six, corner to corner -- never a fragment."""
    edge: tuple
    name: str
    p1: tuple            # projected position of the edge's first vertex
    p2: tuple            # ...and its second
    source: str          # "observed" | "estimated"
    support_px: int      # detection support behind it; 0 when estimated
    n_frames: int        # frames the supporting detection was seen in; 0 when estimated
    residual_px: float   # distance from the supporting detection to this line (inf if none)
    colour: str          # paint colour of the supporting detection ("" if estimated)
    visibility: str = "unknown"   # "in_frame" | "in_pan" | "unseen" -- whether the camera
    # was ever pointed at this line. An `estimated` line that is also `unseen` is the
    # weakest kind: nothing in this window constrains it except the WFDF dimensions.

    @property
    def is_goal_line(self) -> bool:
        return self.edge in ((3, 4), (5, 6))


@dataclass
class FieldModel:
    calibration: object          # Calibration -- maps reference px <-> field cm
    lines: list                  # exactly 6 ModelLine, in FIELD_EDGES order
    corners: list                # the 8 boundary vertices, projected, 1-based order 1..8
    n_observed: int
    cone_hits: int
    n_cone_hints: int
    explained_support_fraction: float   # how much detected paint the model accounts for
    max_reproj_px: float
    assignment: list             # [(edge, tracked_line), ...] actually fitted
    watchers_ok: bool = True     # no dense static row of people INSIDE the field
    extent: object = None        # VisibleExtent -- which PART of the field this shot covers
    colour_agreement: int = 0    # +1 per observed line whose paint colour is on an edge
    # the caller said that colour belongs to, -1 per contradiction, 0 with no prior

    def line(self, edge) -> ModelLine:
        for l in self.lines:
            if l.edge == tuple(edge):
                return l
        raise KeyError(edge)

    @property
    def goal_lines(self) -> list:
        return [l for l in self.lines if l.is_goal_line]


def _line_through(p1, p2) -> np.ndarray:
    l = np.cross([p1[0], p1[1], 1.0], [p2[0], p2[1], 1.0])
    n = np.hypot(l[0], l[1])
    return l / n if n > 1e-9 else l


def _distance_to(line: np.ndarray, points) -> float:
    pts = np.asarray(points, dtype=np.float64)
    return float(np.median(np.abs(pts @ line[:2] + line[2])))


def _angle_between(p1, p2, q1, q2) -> float:
    a = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])) % 180.0
    b = np.degrees(np.arctan2(q2[1] - q1[1], q2[0] - q1[0])) % 180.0
    d = abs(a - b)
    return min(d, 180.0 - d)


def _match_detections_to_edges(calib, tracked, cfg, match_px, match_deg):
    """Assign at most ONE detection to each field edge, and never the same
    detection to two edges.

    MEASURED BUG (torovcanyon.mp4 t=3945-4005s): without the exclusivity, a
    single detection with 21,047 support was counted as evidence for BOTH
    the far end line and the far goal line -- two lines 18m apart on the real
    field -- which inflated "5 of 6 observed" for a labelling that had really
    seen four. Greedy by residual: the closest (edge, detection) pair is
    committed first, then both are removed from consideration.

    Requires matching DIRECTION as well as proximity: a perpendicular line
    crossing near a field edge is not evidence for it."""
    from .calibration import field_to_pixel
    endpoints, candidates = {}, []
    for edge in FIELD_EDGES:
        vi, vj = edge
        p1 = field_to_pixel(*cfg.vertices[vi - 1], calib)
        p2 = field_to_pixel(*cfg.vertices[vj - 1], calib)
        endpoints[edge] = (p1, p2)
        if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
            continue
        line = _line_through(p1, p2)
        for t in tracked:
            if _angle_between(p1, p2, t.p1, t.p2) > match_deg:
                continue
            d = _distance_to(line, t.points)
            if d <= match_px:
                candidates.append((d, edge, t))
    candidates.sort(key=lambda c: c[0])
    matched, taken_edges, taken_lines = {}, set(), set()
    for d, edge, t in candidates:
        if edge in taken_edges or id(t) in taken_lines:
            continue
        matched[edge] = (t, d)
        taken_edges.add(edge)
        taken_lines.add(id(t))
    return matched, endpoints


def build_field_model(calib, tracked, cone_hints, assignment, cfg=None,
                       match_px: float = 45.0, match_deg: float = 6.0,
                       cone_tol_px: float = 40.0, max_reproj_px: float = float("nan"),
                       colour_prior=None, frame_w: int = None, frame_h: int = None,
                       pan_polygon=None) -> FieldModel:
    """Turn a fitted calibration into the six connected field lines.

    Every one of the six is emitted, corner to corner, whether or not any
    paint was ever visible along it -- that is the point. Each is then
    labelled `observed` if a tracked detection supports it, `estimated`
    otherwise.

    `colour_prior`, if given, says which EDGES a paint colour is allowed to
    belong to: {"blue": [(1,7), (2,8)]} reads "the blue line is one of the
    sidelines". Deliberately not "both sidelines are blue" -- MEASURED on
    torovcanyon.mp4, only the near sideline is blue and the far one is white,
    so a per-edge colour map states something false and drags ranking the
    wrong way. It never changes the geometry; it only lets ranking prefer
    labellings that agree with what the person knows about their own field."""
    from .calibration import field_to_pixel
    cfg = cfg or UltimatePitchConfiguration()
    matched, endpoints = _match_detections_to_edges(calib, tracked, cfg, match_px, match_deg)

    extent = None
    if frame_w and frame_h:
        extent = visible_extent(calib, frame_w, frame_h, pan_polygon, cfg)

    def visibility_of(edge):
        if extent is None:
            return "unknown"
        if not extent_supports_observation(edge, extent, cfg):
            return "unseen"
        frame_only = VisibleExtent(extent.frame_bbox, extent.frame_bbox,
                                    extent.frame_length_fraction, extent.frame_length_fraction,
                                    extent.pan_width_fraction, False, False, False, "",
                                    calibration=extent.calibration)
        return "in_frame" if extent_supports_observation(edge, frame_only, cfg) else "in_pan"

    lines = []
    total_support = sum(max(1, t.total_support_px) for t in tracked) or 1
    explained_support = 0
    for edge in FIELD_EDGES:
        p1, p2 = endpoints[edge]
        seen = visibility_of(edge)
        if edge in matched and seen != "unseen":
            detection, residual = matched[edge]
            explained_support += max(1, detection.total_support_px)
            lines.append(ModelLine(edge, EDGE_NAMES[edge], tuple(p1), tuple(p2),
                                    "observed", detection.total_support_px,
                                    detection.n_frames, float(residual),
                                    getattr(detection, "colour", ""), seen))
        else:
            # a detection matched to a line the camera was never pointed at is
            # not evidence -- it is a coincidence, and treating it as evidence is
            # how a fit talks itself into the wrong scale
            lines.append(ModelLine(edge, EDGE_NAMES[edge], tuple(p1), tuple(p2),
                                    "estimated", 0, 0, float("inf"), "", seen))

    corners = [tuple(field_to_pixel(*cfg.vertices[i - 1], calib)) for i in range(1, 9)]
    hits = 0
    for i in range(1, 9):
        cx, cy = corners[i - 1]
        if any(np.hypot(cx - h.x, cy - h.y) <= cone_tol_px for h in cone_hints):
            hits += 1

    agreements = 0
    if colour_prior:
        for l in lines:
            allowed = colour_prior.get(l.colour)
            if allowed and l.source == "observed":
                agreements += 1 if tuple(l.edge) in {tuple(e) for e in allowed} else -1
    model = FieldModel(calibration=calib, lines=lines, corners=corners,
                        n_observed=sum(1 for l in lines if l.source == "observed"),
                        cone_hits=hits, n_cone_hints=len(cone_hints),
                        explained_support_fraction=explained_support / total_support,
                        max_reproj_px=max_reproj_px, assignment=list(assignment),
                        extent=extent)
    model.colour_agreement = agreements
    return model


def _order_along(reference_line: np.ndarray, lines):
    """Order a family of lines by where they cross a reference line.

    A homography preserves incidence and order along a line, so the order in
    which the near end line, near goal line, far goal line and far end line
    cross a sideline is the SAME in the image as on the field. That fact
    collapses the labelling search enormously and, more importantly, is a
    real constraint rather than a heuristic: an assignment that puts the far
    goal line before the near end line is not merely unlikely, it is
    geometrically impossible.

    Returns [(t, line), ...] sorted by t, dropping lines that don't meet the
    reference (parallel in the image)."""
    out = []
    direction = np.array([-reference_line[1], reference_line[0]])
    for l in lines:
        point = np.cross(reference_line, l.homogeneous())
        if abs(point[2]) < 1e-9:
            continue
        xy = np.array([point[0] / point[2], point[1] / point[2]])
        out.append((float(xy @ direction), l))
    out.sort(key=lambda pair: pair[0])
    return out


def _ordered_edge_subsets(m: int):
    """The only ways m cross lines, in image order, can be labelled: an
    increasing subsequence of the field's own order, or its reverse (the
    camera can be on either end of the field). Everything else violates
    order preservation."""
    from itertools import combinations
    subsets = []
    for combo in combinations(CROSS_EDGES, m):
        subsets.append(list(combo))
        subsets.append(list(reversed(combo)))
    return subsets


def fit_field_model(tracked, frame_w: int, frame_h: int, cone_hints=(), cfg=None,
                    image_path: str = "field_model", max_fits: int = 30000,
                    top_k: int = 3, min_lines: int = 4,
                    max_reproj_reject_px: float = 30.0, colour_prior=None,
                    pan_polygon=None, require_fitted_lines_visible: bool = True,
                    watcher_rows=()) -> list:
    """Find the labelling that best explains ALL the evidence, and return
    ranked FieldModels. Empty list means "this framing does not determine a
    field", which is a real answer -- see the module docstring.

    Ranking uses the evidence the fit did NOT consume: how much of the
    detected paint the resulting six lines account for (support-weighted),
    how many of the eight boundary cones land on a cone hint, and how many
    of the six are directly observed.

    Reprojection error is a REJECT gate rather than a ranking term. With
    exactly four lines it is structurally uninformative (the fit is exact),
    but with five or six it is over-determined and a large value means the
    assigned lines genuinely disagree about where the field is -- measured,
    a labelling that put the blue sideline on an end line reached 215px and
    was still being ranked first when reprojection only contributed a capped
    5-point penalty. Anything past `max_reproj_reject_px` is thrown out
    instead of scored down.

    `pan_polygon` (frisbee_analysis.field_lines.segment_pan_polygon) is what
    makes the search aware that the camera never saw the whole field. With it,
    a labelling is rejected when it claims one of its FITTED lines is an edge
    the pan never covered -- e.g. calling a line the far end line when the
    shot only ever swept the near half. Without that constraint nothing stops
    a fit from shrinking the field until an unrelated painted line lands where
    a far boundary 'should' be, which is exactly how a soccer pitch's markings
    can masquerade as an ultimate field.

    `watcher_rows` (frisbee_analysis.watchers.find_rows) are dense, static,
    collinear rows of PEOPLE. Watchers stand outside the boundary, so a
    labelling that puts one inside the field is wrong -- and this is evidence
    the fit never consumed, so unlike reprojection error it can actually
    discriminate. Measured on torovcanyon.mp4: 29 standing positions behind
    1,435 detections, straight to 4.2px over 1,210px, sitting 4.0m outside a
    37m-wide field. That is far more evidence than the cone hints carry.

    `colour_prior` maps an edge to the paint colour it is known to be on
    this footage. Optional, never geometric: it only lets ranking prefer a
    labelling that agrees with what the person already knows about their own
    field (e.g. that the sidelines are the blue lines)."""
    from .calibration import (fit_calibration_with_lines, check_line_reprojection_error,
                              vertices_within_bounds, calibration_covers_evidence,
                              field_covers_points)
    cfg = cfg or UltimatePitchConfiguration()
    if len(tracked) < min_lines:
        return []

    # EVERY plausible two-direction split, not one. MEASURED (see
    # field_lines.direction_hypotheses): taking the largest vanishing-point
    # cluster as one family and everything else as the other makes the second
    # "family" a grab-bag rather than a direction, which on the test window
    # threw away all 672 candidate labellings that used the blue near sideline
    # as a sideline -- the data supported 11 of 11 possible partners for it.
    # Only the SIDELINE pair has to be vanishing-point concurrent -- that is
    # what conditions the fit. Requiring it of the cross lines too hid every
    # valid model on the test window (see field_lines.sideline_family_candidates).
    hypotheses = sideline_family_candidates(tracked)
    if not hypotheses:
        # Lines that share a vanishing point are parallel on the real field,
        # and parallel lines cannot constrain the perpendicular direction
        # however many there are. Refuse rather than invent a second
        # direction -- an earlier version guessed here and was confidently,
        # precisely wrong.
        return []

    from itertools import combinations
    results = []
    fits = 0

    class _Budget(Exception):
        pass

    try:
      for side_pool, cross_pool, _vp in hypotheses:
        if len(cross_pool) >= 2:
            for side_pair in combinations(range(len(side_pool)), 2):
                for side_order in ((0, 1), (1, 0)):
                    side_lines = [side_pool[side_pair[side_order[0]]],
                                  side_pool[side_pair[side_order[1]]]]
                    # order the cross lines by where they cross the first sideline;
                    # that order is preserved by any homography (see _order_along)
                    ordered = _order_along(side_lines[0].homogeneous(), cross_pool)
                    for n_cross in (4, 3, 2):
                        if n_cross > len(ordered):
                            continue
                        for picked in combinations(range(len(ordered)), n_cross):
                            cross_lines = [ordered[i][1] for i in picked]
                            if any(id(c) in (id(side_lines[0]), id(side_lines[1]))
                                   for c in cross_lines):
                                continue
                            for cross_edges in _ordered_edge_subsets(n_cross):
                                if fits >= max_fits:
                                    raise _Budget
                                chosen = side_lines + cross_lines
                                edges = list(SIDELINE_EDGES) + list(cross_edges)
                                # cheap, fit-free rejection first: two lines assigned to
                                # PARALLEL field edges cannot cross where they were seen
                                if not no_parallel_crossings(list(zip(edges, chosen))):
                                    continue
                                fits += 1
                                try:
                                    calib = fit_calibration_with_lines(
                                        image_points=[], keypoint_indices=[],
                                        image_path=image_path,
                                        line_image_points=[list(l.points) for l in chosen],
                                        line_field_edges=edges, cfg=cfg)
                                except (ValueError, np.linalg.LinAlgError):
                                    continue
                                if not vertices_within_bounds(calib, frame_w, frame_h, cfg):
                                    continue
                                if not projected_field_is_sane(calib, cfg):
                                    continue
                                # the horizon must not cut through the field, and the
                                # field must contain the pixels it was fitted from --
                                # see calibration.field_in_front_of_camera for why
                                # neither of the checks above can see these
                                # LOCAL is fine (a shot sees one end of the field);
                                # invalid where its own evidence lives is not.
                                if not calibration_covers_evidence(
                                        calib, [p for l in chosen for p in l.points], cfg):
                                    continue
                                if not field_covers_points(
                                        calib, [p for l in chosen for p in l.points], cfg):
                                    continue
                                try:
                                    per_line = check_line_reprojection_error(calib, cfg)
                                except ValueError:
                                    continue
                                worst = max(r["max"] for r in per_line)
                                if not np.isfinite(worst):
                                    continue
                                if len(chosen) > 4 and worst > max_reproj_reject_px:
                                    continue
                                calib.source_max_reprojection_px = float(worst)
                                if watcher_rows and not watchers_outside_field(
                                        calib, watcher_rows, cfg):
                                    continue      # people standing on the pitch: impossible
                                model = build_field_model(
                                    calib, tracked, cone_hints, list(zip(edges, chosen)),
                                    cfg, max_reproj_px=float(worst), colour_prior=colour_prior,
                                    frame_w=frame_w, frame_h=frame_h, pan_polygon=pan_polygon)
                                model.watchers_ok = bool(watcher_rows)
                                if model.extent is not None and not extent_is_plausible(model.extent, cfg):
                                    continue
                                if require_fitted_lines_visible and model.extent is not None:
                                    # every line we FITTED to must be one the camera could
                                    # actually have seen -- otherwise the correspondence
                                    # that produced this homography is fiction
                                    if any(not extent_supports_observation(e, model.extent, cfg)
                                           for e in edges):
                                        continue
                                results.append(model)
    except _Budget:
        pass

    def score(m: FieldModel):
        # higher is better; every term is evidence the FIT ITSELF did not use
        # MEASURED: explained-paint fraction cannot carry the ranking on its own.
        # A wrongly-SHRUNKEN field explains MORE paint, not less, because more of
        # the soccer/lacrosse markings the ultimate field is laid over fall inside
        # it -- on torovcanyon.mp4 the implausible models scored 80-87% while the
        # geometrically sensible one scored 15%. It stays in the score as a weak
        # term; the load is carried by evidence a wrong scale cannot fake.
        return (m.explained_support_fraction * 25.0
                + m.cone_hits * 8.0
                + m.n_observed * 6.0
                + m.colour_agreement * 15.0)

    # drop models whose implied field is absurd for a broadcast shot: if one
    # frame supposedly spans nearly the whole 100m, the fit has shrunk the
    # field to fit the paint it found rather than the other way round
    results = [m for m in results
               if m.extent is None or not np.isfinite(m.extent.frame_length_fraction)
               or m.extent.frame_length_fraction <= 0.98]

    results.sort(key=score, reverse=True)
    # collapse models that describe the same field (same observed set + close corners)
    unique = []
    for m in results:
        duplicate = False
        for kept in unique:
            if max(np.hypot(a[0] - b[0], a[1] - b[1])
                   for a, b in zip(m.corners, kept.corners)) < 30.0:
                duplicate = True
                break
        if not duplicate:
            unique.append(m)
        if len(unique) >= top_k:
            break
    return unique


def _valid_polyline(edge, calib, cfg, samples: int = 200):
    """Image polyline for a field edge, walked in FIELD space and cut where it
    passes the horizon.

    MEASURED BUG this fixes (person, from an overlay): *"I only see near end
    line and near goal line, and a third line going completely the wrong way
    ... NO sidelines, where did the sidelines go?"* The third line WAS a
    sideline. Drawing an edge as a straight segment from `field_to_pixel(v1)`
    to `field_to_pixel(v2)` is only valid when both endpoints are in front of
    the camera. For a LOCAL calibration (Result 3.14) they routinely are not:
    canvas_002's horizon crosses y=0 at x=64.4m, so vertex 7 at x=100m is
    behind it, and the drawn segment leaps across the vanishing line and off
    in an arbitrary direction.

    A straight line in the field is still a straight line in the image, but
    only the part in front of the camera exists. So walk the edge in field
    coordinates, keep the samples whose homogeneous w has the dominant sign,
    and draw those. What you get is the sideline running from the end line to
    where it vanishes -- what a person expects to see, and honest about where
    the calibration stops meaning anything."""
    from .calibration import field_to_pixel
    vi, vj = edge
    ax, ay = cfg.vertices[vi - 1]
    bx, by = cfg.vertices[vj - 1]
    H_inv = np.linalg.inv(calib.H())
    pts, ws = [], []
    for t in np.linspace(0.0, 1.0, samples):
        fx = ax + (bx - ax) * t
        fy = ay + (by - ay) * t
        ws.append(H_inv[2, 0] * fx + H_inv[2, 1] * fy + H_inv[2, 2])
        pts.append(field_to_pixel(fx, fy, calib))
    signs = [np.sign(w) for w in ws if abs(w) > 1e-12]
    if not signs:
        return []
    dominant = 1.0 if sum(1 for s in signs if s > 0) >= len(signs) / 2 else -1.0
    out = []
    for (px, py), w in zip(pts, ws):
        if abs(w) <= 1e-12 or np.sign(w) != dominant:
            continue
        if np.isfinite(px) and np.isfinite(py) and abs(px) < 1e5 and abs(py) < 1e5:
            out.append((px, py))
    return out


def draw_field_model(image: np.ndarray, model: FieldModel, cfg=None,
                     label: bool = True) -> np.ndarray:
    """Draw the six lines as one connected field. Solid = observed paint,
    dashed = estimated from geometry.

    Each edge is drawn as the part of it actually IN FRONT of the camera (see
    `_valid_polyline`), not as a segment between two projected vertices -- for
    a local calibration one of those vertices is often past the horizon, and
    joining them draws a line that goes nowhere real."""
    cfg = cfg or UltimatePitchConfiguration()
    vis = image.copy()
    observed_colour = (0, 230, 0)
    estimated_colour = (0, 200, 255)
    goal_colour = (255, 0, 255)
    for l in model.lines:
        polyline = _valid_polyline(l.edge, model.calibration, cfg)
        if len(polyline) < 2:
            continue
        colour = goal_colour if l.is_goal_line else (
            observed_colour if l.source == "observed" else estimated_colour)
        pts = [(int(round(x)), int(round(y))) for x, y in polyline]
        for k in range(len(pts) - 1):
            if l.source == "observed" or k % 2 == 0:
                cv2.line(vis, pts[k], pts[k + 1], colour, 3)
        if label:
            h, w = vis.shape[:2]
            on_screen = [p for p in pts if 0 <= p[0] < w and 0 <= p[1] < h]
            anchor = on_screen[len(on_screen) // 2] if on_screen else pts[len(pts) // 2]
            text = f"{l.name} [{l.source}]"
            for c, th in (((0, 0, 0), 5), (colour, 2)):
                cv2.putText(vis, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, th)
    for x, y in model.corners:
        if np.isfinite(x) and np.isfinite(y) and abs(x) < 1e5 and abs(y) < 1e5:
            cv2.circle(vis, (int(round(x)), int(round(y))), 7, (0, 0, 255), -1)
    return vis


# --------------------------------------------------------------------------
# which PART of the field is this shot even looking at?
# --------------------------------------------------------------------------

@dataclass
class VisibleExtent:
    """How much of the field a frame -- and the whole pan -- actually covers.

    MEASURED NEED (person's own observation on the torovcanyon model image):
    "the camera will almost never be looking at the whole field, only part of
    it... only both sidelines and one endzone including back line and goal
    line is visible". A calibration that silently assumes the whole field is
    in view will happily label a line it cannot possibly be seeing.

    Coordinates are field centimetres, so `x` runs 0..cfg.length along the
    field and `y` runs 0..cfg.width across it."""
    frame_bbox: tuple        # (x_min, x_max, y_min, y_max) for ONE frame, field cm
    pan_bbox: tuple          # ...and for everything the pan swept
    frame_length_fraction: float   # how much of the 100m a single frame spans
    pan_length_fraction: float
    pan_width_fraction: float
    sees_near_endzone: bool
    sees_far_endzone: bool
    sees_both_sidelines: bool
    description: str
    calibration: object = None   # so plausibility checks can clip to the validity region


def _bbox_in_field(points_px, calib, cfg, samples: int = 24):
    """Bounding box, in field cm, of an image region seen through `calib`.

    MEASURED BUG this fixes (torovcanyon.mp4 t=3945-4005s): the first version
    projected only the polygon's CORNERS and took a naive min/max. Through a
    LOCAL calibration -- one whose horizon crosses the frame, which Result
    3.14 establishes is normal rather than broken -- corners near or past the
    vanishing line project to enormous or sign-flipped field coordinates. The
    pan's extent came out as y = -163.9m .. 6.6m on a 37m-wide field, and
    `extent_supports_observation` then declared the very edges the homography
    had just been FITTED to "unseen", so `build_field_model` threw away its
    own evidence and reported 5 of 6 lines as estimated.

    Fixed by sampling the region rather than its corners, and keeping only
    samples genuinely in front of the camera (consistent sign of the
    homogeneous w). Points behind the horizon are not part of what the camera
    saw and must not contribute to a bounding box."""
    from .calibration import pixel_to_field, field_validity_region
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return (float("nan"),) * 4
    x_lo, x_hi = pts[:, 0].min(), pts[:, 0].max()
    y_lo, y_hi = pts[:, 1].min(), pts[:, 1].max()
    H = calib.H()
    grid = []
    for gx in np.linspace(x_lo, x_hi, samples):
        for gy in np.linspace(y_lo, y_hi, samples):
            w = H[2, 0] * gx + H[2, 1] * gy + H[2, 2]
            grid.append((gx, gy, w))
    signs = [np.sign(w) for _, _, w in grid if abs(w) > 1e-12]
    if not signs:
        return (float("nan"),) * 4
    dominant = 1.0 if sum(1 for s in signs if s > 0) >= len(signs) / 2 else -1.0
    # MEASURED: dropping only the samples BEHIND the camera is not enough. Points
    # merely NEAR the horizon still carry the right sign while projecting to
    # absurd field coordinates -- the pan bbox came out spanning y = -4.4..865.9m
    # on a 37m-wide field, which then made `extent_is_plausible` reject the
    # person's verified-correct labelling. Coordinates outside what the
    # calibration can represent are not observations; exclude them too.
    region = field_validity_region(calib, cfg)
    limit = max(region["usable_x_max"], 0.0)
    xs, ys = [], []
    for gx, gy, w in grid:
        if abs(w) <= 1e-12 or np.sign(w) != dominant:
            continue          # this pixel is behind the camera -- it saw nothing
        fx, fy = pixel_to_field(gx, gy, calib)
        if not (np.isfinite(fx) and np.isfinite(fy)):
            continue
        if fx > limit or fx < -cfg.length or abs(fy) > 5.0 * cfg.width:
            continue          # beyond where this calibration means anything
        xs.append(fx)
        ys.append(fy)
    if not xs:
        return (float("nan"),) * 4
    # ROBUST bounds, not min/max. MEASURED: near the horizon a handful of pixels
    # map to enormous field coordinates while still passing every validity test
    # -- the pan's y range came out -4.4..184.2m on a 37m field, set by a few
    # extreme samples, and `extent_is_plausible` then rejected the person's
    # verified-correct labelling. Percentiles describe where the camera was
    # actually looking; the extremes describe where the projection is breaking
    # down.
    xa, ya = np.asarray(xs), np.asarray(ys)
    return (float(np.percentile(xa, 2)), float(np.percentile(xa, 98)),
            float(np.percentile(ya, 2)), float(np.percentile(ya, 98)))


def visible_extent(calib, frame_w: int, frame_h: int, pan_polygon=None, cfg=None) -> VisibleExtent:
    """Push the frame's corners (and the pan's footprint) through the fitted
    homography to find out which part of the field is actually in view."""
    cfg = cfg or UltimatePitchConfiguration()
    frame_box = [(0.0, 0.0), (float(frame_w), 0.0),
                 (float(frame_w), float(frame_h)), (0.0, float(frame_h))]
    frame_bbox = _bbox_in_field(frame_box, calib, cfg)
    pan_bbox = _bbox_in_field(pan_polygon, calib, cfg) if pan_polygon else frame_bbox

    def clamped_fraction(lo, hi, total):
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return float("nan")
        return max(0.0, min(float(total), hi) - max(0.0, lo)) / float(total)

    frame_len = clamped_fraction(frame_bbox[0], frame_bbox[1], cfg.length)
    pan_len = clamped_fraction(pan_bbox[0], pan_bbox[1], cfg.length)
    pan_wid = clamped_fraction(pan_bbox[2], pan_bbox[3], cfg.width)

    # an endzone counts as seen when most of its depth is inside the pan
    near = clamped_fraction(max(pan_bbox[0], 0.0), min(pan_bbox[1], cfg.goal_line_near),
                            cfg.endzone_depth) if np.isfinite(pan_bbox[0]) else 0.0
    far_lo = max(pan_bbox[0], cfg.goal_line_far) if np.isfinite(pan_bbox[0]) else 0.0
    far_hi = min(pan_bbox[1], cfg.length) if np.isfinite(pan_bbox[1]) else 0.0
    far = max(0.0, far_hi - far_lo) / float(cfg.endzone_depth)
    both_sidelines = (np.isfinite(pan_bbox[2]) and pan_bbox[2] <= 0.05 * cfg.width
                      and pan_bbox[3] >= 0.95 * cfg.width)

    parts = []
    if np.isfinite(pan_bbox[0]):
        parts.append(f"pan covers x={pan_bbox[0] / 100:.0f}..{pan_bbox[1] / 100:.0f}m "
                     f"of 0..{cfg.length / 100:.0f}m ({pan_len * 100:.0f}% of the length), "
                     f"y={pan_bbox[2] / 100:.0f}..{pan_bbox[3] / 100:.0f}m "
                     f"of 0..{cfg.width / 100:.0f}m")
    parts.append("both sidelines in view" if both_sidelines else "NOT both sidelines in view")
    seen = [name for name, ok in (("near endzone", near >= 0.6), ("far endzone", far >= 0.6)) if ok]
    parts.append(("endzones in view: " + ", ".join(seen)) if seen else "neither endzone fully in view")
    return VisibleExtent(frame_bbox, pan_bbox, frame_len, pan_len, pan_wid,
                          near >= 0.6, far >= 0.6, both_sidelines, "; ".join(parts),
                          calibration=calib)


def extent_is_plausible(extent: VisibleExtent, cfg=None,
                         max_width_span: float = 3.0,
                         max_length_span: float = 2.0) -> bool:
    """Is the implied camera coverage physically sensible for a broadcast
    shot of THIS field?

    This is "knowledge of how big the field should be" used as a constraint on
    the fit rather than only as a way to draw lines. A homography fitted to
    the wrong lines does not fail quietly -- it rescales the world so that
    whatever paint it found lands where it wants. The tell is the implied
    coverage: MEASURED (torovcanyon.mp4 t=3945-4005s) a wrong labelling
    claimed its pan swept y = -110m..54m across a field that is 37m wide,
    i.e. 164m of width, and only 14% of the length. A camera filming a 37m
    wide field does see beyond the sidelines, but not four field-widths
    beyond, and a shot that wide cannot simultaneously be that narrow along
    the length.

    Bounds are deliberately loose -- they are meant to catch nonsense, not to
    police framing."""
    cfg = cfg or UltimatePitchConfiguration()
    x_min, x_max, y_min, y_max = extent.pan_bbox
    if not all(np.isfinite(v) for v in (x_min, x_max, y_min, y_max)):
        return True
    if (y_max - y_min) > max_width_span * cfg.width:
        return False
    if (x_max - x_min) > max_length_span * cfg.length:
        return False
    # the shot has to actually overlap the field it claims to be filming
    if x_max < 0 or x_min > cfg.length or y_max < 0 or y_min > cfg.width:
        return False
    return True


def extent_supports_observation(edge, extent: VisibleExtent, cfg=None, margin_cm: float = 500.0) -> bool:
    """Could this field edge plausibly have been SEEN, given what the pan
    covered? A line the camera never pointed at cannot have been observed,
    however good a detection appears to sit on it.

    This is the check the person's observation asks for: on the torovcanyon
    window both sidelines and ONE endzone (its end line and goal line) are in
    view, so a labelling that reports the far end line as `observed` is
    claiming to have seen 100m of field through a shot that swept far less.
    Without it, nothing stops a fit from scaling the field down until an
    unrelated painted line lands where the far end line 'should' be."""
    cfg = cfg or UltimatePitchConfiguration()
    x_min, x_max, y_min, y_max = extent.pan_bbox
    if not np.isfinite(x_min):
        return True   # no extent information -- don't reject on no evidence
    vi, vj = edge
    xs = [cfg.vertices[vi - 1][0], cfg.vertices[vj - 1][0]]
    ys = [cfg.vertices[vi - 1][1], cfg.vertices[vj - 1][1]]
    # a sideline runs the length of the field: it is "in view" if ANY of it is.
    # an end/goal line spans the width at one x: that x must be inside the pan.
    if xs[0] == xs[1]:
        return (x_min - margin_cm) <= xs[0] <= (x_max + margin_cm)
    return not (max(ys) < y_min - margin_cm or min(ys) > y_max + margin_cm)


def parallel_edges_meet_outside_field(calib, cfg=None, margin_cm: float = 500.0) -> bool:
    """FAILED EXPERIMENT, kept as a warning. Do not use as a gate.

    Motivated by the person's observation from an overlay -- *"the near end
    line runs parallel to near goal line no?? why is it diagonal across the
    field"* -- which is correct: lines parallel on the field meet only at a
    vanishing point, never on the playing surface. The idea was to intersect
    each same-group pair of projected edges and reject the fit if they met
    inside the field.

    MEASURED WRONG: it returns False for every calibration tested, INCLUDING
    canvas_000, the one verified good. The reason is fundamental rather than a
    tuning problem -- the intersection of two correctly-parallel field lines
    is a point at infinity, so `pixel_to_field` on it is ill-conditioned and
    returns an arbitrary finite coordinate that lands inside the field as
    often as not. The test cannot distinguish the case it was built for.

    What the person actually saw is a mis-SCALED field drawing its end line
    across the visible area, which is a symptom of bad placement, not a
    parallelism violation -- so this was the wrong check for that observation
    too. Any future version should work in homogeneous coordinates (a correct
    pair meets where w ~ 0) rather than mapping the meeting point back to the
    field."""
    raise NotImplementedError(
        "parallel_edges_meet_outside_field is a documented failed experiment -- "
        "it rejects known-good calibrations. See its docstring.")
