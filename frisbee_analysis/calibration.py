"""
Field calibration: manual point-correspondence homography from image pixels
to real-world field coordinates (cm, matching ultimate_config.py's convention).

MEASURED (results/calibration_finding.md): automated cone/line detection is
hard on real broadcast footage (mosaic stitching-seam artifacts, treeline/
overlay confounds dominating naive Hough). Manual calibration -- click a
handful of keypoints once per segment -- sidesteps that, and is cheap because
a segment can span minutes (frisbee_analysis/mosaic.py's register_sequence
already gives every frame in a segment a homography to one shared reference,
so calibrating that ONE reference calibrates the whole segment).

A cone being off-camera in the image you calibrate ON does not block using
it: homography extrapolates. Once >=4 well-spread points are correctly
located (anywhere in the segment's mosaic, not all in one raw frame), the
fitted transform maps EVERY pixel to field coordinates -- including pixels
showing players standing in a part of the field where no cone was ever
visible, and including the position of an endzone whose own corner cones
were off-camera the whole time. That position is COMPUTED from
ultimate_config.py's known WFDF geometry, not detected from pixels.
Accuracy still degrades with distance from the actually-clicked points (a
single-homography fit is a locally-good approximation, not a guarantee at
the far extreme of the field) -- prefer points spread across the field over
points clustered in one corner.
"""
from __future__ import annotations
import json
import numpy as np
import cv2
from dataclasses import dataclass, asdict
from .ultimate_config import UltimatePitchConfiguration


@dataclass
class Calibration:
    homography: list  # 3x3, image px -> field cm
    keypoint_labels: list
    image_points: list  # pixel coords used, same order as keypoint_labels
    field_points: list  # cm coords used, same order
    image_path: str
    source_frame_indices: list = None  # which raw frame (within the segment) each
    # point was clicked on, same order as keypoint_labels -- lets you check whether
    # error correlates with how many frames a point's homography was chained
    # through from the segment reference (see results/calibration_finding.md).
    # Optional/None for calibrations fit outside run_calibrate.py.
    line_details: list = None  # for line-based calibrations: [{"edge": [vi,vj],
    # "points": [[x,y],...], "frame_indices": [...]}, ...] -- the FULL point
    # list per line, not just the one representative point kept in
    # image_points/field_points. Without this, a proper per-line reprojection
    # check (perpendicular distance of each click from the fitted line) is
    # only possible live during run_calibrate.py's session, not after the
    # fact from the saved file -- learned that gap the hard way, keeping the
    # full data now so it's always re-checkable. None for point-only fits.
    source_max_reprojection_px: float = None  # MEASURED GAP (person caught a
    # visible sideline drift in an auto-matched calibration; traced it back
    # to the ORIGINAL manual calibration's own reprojection error -- 12.8px
    # on one line -- which was below the "reject" threshold but still
    # visible, and became UNCHECKABLE once composed into a bank match
    # because auto-matched calibrations have no line_details of their own to
    # re-verify). This field carries the original calibration's own
    # worst-line error FORWARD through every bank composition (see
    # calibration_bank.py), so that limitation stays visible instead of
    # silently disappearing. None if never checked (e.g. a point-only fit).

    def H(self) -> np.ndarray:
        return np.array(self.homography, dtype=np.float64)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path) -> "Calibration":
        with open(path) as f:
            return Calibration(**json.load(f))


def fit_calibration(image_points, keypoint_indices, image_path,
                     cfg: UltimatePitchConfiguration = None,
                     source_frame_indices=None) -> Calibration:
    """image_points: list of (x,y) pixel coords, one per entry in
    keypoint_indices (1-based indices into UltimatePitchConfiguration().vertices,
    matching CALIBRATION_INDICES' labelling). Points don't need to cover every
    keypoint -- only >=4, well-spread, not collinear."""
    cfg = cfg or UltimatePitchConfiguration()
    if len(image_points) < 4:
        raise ValueError(f"need >= 4 points for a homography, got {len(image_points)}")
    vertices = cfg.vertices
    field_points = [vertices[i - 1] for i in keypoint_indices]
    src = np.float32(image_points)
    dst = np.float32(field_points)
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise ValueError("homography fit failed -- points may be collinear or degenerate")
    return Calibration(homography=H.tolist(), keypoint_labels=[str(i) for i in keypoint_indices],
                        image_points=[list(map(float, p)) for p in image_points],
                        field_points=[list(map(float, p)) for p in field_points],
                        image_path=str(image_path),
                        source_frame_indices=list(source_frame_indices) if source_frame_indices else None)


def _fit_image_line(image_pts):
    """Best-fit homogeneous line [a,b,c] (ax+by+c=0) through >=2 points
    (already in whatever coordinate space they're passed in), via
    total-least-squares (SVD null vector) -- more robust to click noise than
    a 2-point cross product once you have 3+ points on the line."""
    pts = np.asarray(image_pts, dtype=np.float64)
    if len(pts) == 2:
        p1 = np.array([pts[0, 0], pts[0, 1], 1.0])
        p2 = np.array([pts[1, 0], pts[1, 1], 1.0])
        line = np.cross(p1, p2)
    else:
        A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
        _, _, Vt = np.linalg.svd(A)
        line = Vt[-1]
    return line


def _hartley_normalize(points):
    """Standard DLT preconditioning (Hartley 1997): a transform T such that
    T @ points has centroid at the origin and average distance sqrt(2) from
    it. Without this, point coordinates (pixels, ~100s) and field coordinates
    (cm, ~1000s-10000s) -- and especially line coefficients derived from them
    -- have wildly different natural scales, which measurably breaks the
    SVD-based solve (verified: a synthetic recovery test was off by
    thousands of pixels without normalization, ~0 with it)."""
    pts = np.asarray(points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    avg_dist = np.mean(np.hypot(centered[:, 0], centered[:, 1]))
    scale = np.sqrt(2) / avg_dist if avg_dist > 1e-9 else 1.0
    T = np.array([[scale, 0, -scale * centroid[0]],
                   [0, scale, -scale * centroid[1]],
                   [0, 0, 1]], dtype=np.float64)
    return T


def fit_calibration_with_lines(image_points, keypoint_indices, image_path,
                                line_image_points=None, line_field_edges=None,
                                cfg: UltimatePitchConfiguration = None,
                                source_frame_indices=None,
                                line_frame_indices=None) -> Calibration:
    """Like fit_calibration, but also accepts LINE correspondences -- useful
    when you can't confidently pin down enough exact corner points, but CAN
    identify >=2 pixels that lie somewhere along a known field line (e.g. the
    sideline) without knowing exactly where the corner is. A line constraint
    is more forgiving than a point: you don't need the precise intersection,
    just confidence about which line you're looking at.

    line_image_points: list of point-lists, each >=2 (x,y) pixels believed to
      lie on ONE field line (same segment reference coords as image_points).
    line_field_edges: parallel list of (vertex_i, vertex_j) 1-based indices
      into cfg.vertices defining which known field line each entry is.
    line_frame_indices: optional, parallel to line_image_points -- for each
      line, the raw frame index each of its points was clicked on. Stored in
      full on the returned Calibration (line_details) so reprojection error
      can be re-checked later from the saved file, not just live during the
      clicking session (a real gap this parameter exists to close).

    Combined-DLT in Hartley-normalized coordinates: point correspondences
    and line correspondences each contribute 2 rows to the same linear
    system (lines via l_image = H^T @ l_field, using projective duality); H
    is the right singular vector of the stacked, normalized design matrix
    for its smallest singular value, then denormalized back."""
    cfg = cfg or UltimatePitchConfiguration()
    line_image_points = line_image_points or []
    line_field_edges = line_field_edges or []
    if len(line_image_points) != len(line_field_edges):
        raise ValueError("line_image_points and line_field_edges must be the same length")

    n_constraints = len(image_points) + len(line_image_points)
    if n_constraints < 4:
        raise ValueError(f"need >= 4 total constraints (points + lines), got {n_constraints}")

    # MEASURED, UNRESOLVED BUG (see results/calibration_finding.md): mixing
    # point and line constraints in this DLT loses exactly one rank versus
    # what point-only or line-only give on their own -- confirmed with fully
    # random synthetic data, not a geometry-specific fluke. Root cause not
    # yet found despite validating the row formula, normalization, and
    # denormalization independently (all individually correct). Rather than
    # silently return a wrong homography from an under-determined system,
    # fail loudly if the constraint matrix doesn't reach full rank. Points-
    # only (fit_calibration) and lines-only (this function with
    # image_points=[]) are both independently verified correct -- only the
    # MIX is currently unsafe.
    if image_points and line_image_points:
        import warnings
        warnings.warn(
            "Mixing point and line constraints has a known, unresolved rank-deficiency "
            "bug (see results/calibration_finding.md) -- verify the rank check below passes, "
            "and treat the result with real suspicion even if it does. Prefer lines-only "
            "(image_points=[]) or points-only (fit_calibration) until this is fixed.",
            RuntimeWarning)

    vertices = cfg.vertices
    field_points = [vertices[i - 1] for i in keypoint_indices]

    # Normalize using ALL image-space points involved (explicit points +
    # every line-sample point) and ALL field-space points involved (explicit
    # field points + both endpoint vertices of each referenced line, used
    # only to get a sane normalization scale, not as hard constraints).
    all_img_for_norm = list(image_points) + [p for pts in line_image_points for p in pts]
    all_field_for_norm = list(field_points) + [
        vertices[v - 1] for edge in line_field_edges for v in edge]
    T_img = _hartley_normalize(all_img_for_norm)
    T_field = _hartley_normalize(all_field_for_norm)
    T_img_inv_T = np.linalg.inv(T_img).T
    T_field_inv_T = np.linalg.inv(T_field).T

    def to_norm(T, pt):
        v = T @ np.array([pt[0], pt[1], 1.0])
        return v[0] / v[2], v[1] / v[2]

    rows = []
    for (u, v), (x, y) in zip(image_points, field_points):
        un, vn = to_norm(T_img, (u, v))
        xn, yn = to_norm(T_field, (x, y))
        rows.append([-un, -vn, -1, 0, 0, 0, un * xn, vn * xn, xn])
        rows.append([0, 0, 0, -un, -vn, -1, un * yn, vn * yn, yn])

    for pts, (vi, vj) in zip(line_image_points, line_field_edges):
        if len(pts) < 2:
            raise ValueError("each line needs >= 2 image points")
        l_img_raw = _fit_image_line(pts)
        l_img = T_img_inv_T @ l_img_raw  # line transforms by inverse-transpose of the point transform
        p, q, r = l_img / np.linalg.norm(l_img[:2])

        fx1, fy1 = vertices[vi - 1]
        fx2, fy2 = vertices[vj - 1]
        l_field_raw = np.cross([fx1, fy1, 1.0], [fx2, fy2, 1.0])
        l_field = T_field_inv_T @ l_field_raw
        a, b, c = l_field / np.linalg.norm(l_field[:2])

        rows.append([0, -a * r, a * q, 0, -b * r, b * q, 0, -c * r, c * q])
        rows.append([a * r, 0, -a * p, b * r, 0, -b * p, c * r, 0, -c * p])

    A = np.array(rows, dtype=np.float64)
    singular_values = np.linalg.svd(A, compute_uv=False)
    # need rank 8 (9 unknowns, 1 DOF free for scale): the 8th singular value
    # should be comfortably larger than the 9th (~0). A small gap here means
    # the constraint set is under-determined -- the returned H would be an
    # arbitrary vector from a >1-dimensional null space, not a real answer.
    if len(singular_values) >= 8 and singular_values[7] < 1e-6:
        raise ValueError(
            f"constraint matrix is rank-deficient (8th singular value = {singular_values[7]:.2e}, "
            f"want >> the 9th = {singular_values[-1]:.2e}) -- these points/lines don't jointly "
            f"constrain the homography enough, even though the count looks sufficient. Add a "
            f"differently-oriented point or line, or see results/calibration_finding.md for the "
            f"known point+line mixing bug.")
    _, _, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T_field) @ H_norm @ T_img  # denormalize back to real pixel/cm coordinates
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]

    all_image_pts = list(image_points) + [list(pts[0]) for pts in line_image_points]
    all_field_pts = list(field_points) + [
        list(vertices[vi - 1]) for vi, vj in line_field_edges]  # representative point only, for logging

    line_details = None
    if line_image_points:
        frame_lists = line_frame_indices if line_frame_indices else [None] * len(line_image_points)
        line_details = [
            {"edge": list(edge),
             "points": [list(map(float, p)) for p in pts],
             "frame_indices": list(fis) if fis else None}
            for pts, edge, fis in zip(line_image_points, line_field_edges, frame_lists)
        ]

    return Calibration(homography=H.tolist(),
                        keypoint_labels=[str(i) for i in keypoint_indices] +
                                        [f"line{vi}-{vj}" for vi, vj in line_field_edges],
                        image_points=[list(map(float, p)) for p in all_image_pts],
                        field_points=[list(map(float, p)) for p in all_field_pts],
                        image_path=str(image_path),
                        source_frame_indices=list(source_frame_indices) if source_frame_indices else None,
                        line_details=line_details)


def check_line_reprojection_error(calib: Calibration, cfg: UltimatePitchConfiguration = None):
    """Re-derive each line's fitted image-space line (via the field edge's
    known endpoints projected through calib's homography) and measure the
    perpendicular distance of every ORIGINALLY CLICKED point from it. Needs
    calib.line_details (the full per-line point lists) -- unlike point
    reprojection error, this can't be reconstructed from just the one
    representative point kept in image_points/field_points. Works on any
    saved Calibration with line_details, not just live during
    run_calibrate.py's session -- that gap is exactly why line_details exists.

    Returns a list of dicts: [{"edge": [vi,vj], "errors": [...], "max": ..,
    "mean": ..}, ...], one per line."""
    if not calib.line_details:
        raise ValueError("this calibration has no line_details -- either it's a point-only "
                          "fit, or it predates line_details being saved (re-run the calibration)")
    cfg = cfg or UltimatePitchConfiguration()
    results = []
    for entry in calib.line_details:
        vi, vj = entry["edge"]
        fx1, fy1 = cfg.vertices[vi - 1]
        fx2, fy2 = cfg.vertices[vj - 1]
        p1 = field_to_pixel(fx1, fy1, calib)
        p2 = field_to_pixel(fx2, fy2, calib)
        l = np.cross([p1[0], p1[1], 1.0], [p2[0], p2[1], 1.0])
        l = l / np.hypot(l[0], l[1])
        errors = [abs(l[0] * px + l[1] * py + l[2]) for px, py in entry["points"]]
        results.append({"edge": entry["edge"], "errors": errors,
                         "max": max(errors), "mean": float(np.mean(errors))})
    return results


def pixel_to_field(x, y, calib: Calibration):
    pt = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float64), calib.H())
    return float(pt[0, 0, 0]), float(pt[0, 0, 1])


def field_to_pixel(fx, fy, calib: Calibration):
    H_inv = np.linalg.inv(calib.H())
    pt = cv2.perspectiveTransform(np.array([[[fx, fy]]], dtype=np.float64), H_inv)
    return float(pt[0, 0, 0]), float(pt[0, 0, 1])


def vertices_within_bounds(calib: Calibration, frame_w: float, frame_h: float,
                            cfg: UltimatePitchConfiguration = None, margin: float = 8.0) -> bool:
    """Sanity check a calibration can't get from reprojection error alone.

    MEASURED (torovcanyon.mp4, t=3945s): a calibration whose sideline click
    points were only 262px apart (spanning a field line that runs the WHOLE
    field length) had source_max_reprojection_px = 0.0036 -- near-perfect --
    while the far corner it never directly constrained projected to pixel
    (515660, 41790), over 500,000px outside any real frame. Reprojection
    error only checks that clicked points are self-consistent with the line
    already fit through them; it says nothing about whether that line had
    enough spread to constrain the REST of the field. This check catches
    that failure mode directly: every field vertex, including ones never
    directly clicked, must project somewhere within a generous margin around
    the actual frame/canvas -- a real calibration error might be off by tens
    of pixels, not hundreds of thousands.

    margin=8.0 (bound at 9x frame width/height) is set from the two real
    data points measured so far, not guessed: the ACTUALLY-GOOD recalibration
    above still had its farthest (never-clicked) vertex land at 4.6x frame
    width -- legitimately off-camera, not wrong -- while the broken one hit
    268.6x. 8x sits with wide margin above the good case and nowhere near the
    bad one; revisit if a real calibration ever legitimately needs more.

    This is what let calibration_bank.py's add_entry silently keep a broken
    calibration over a fixed one: it compared ONLY source_max_reprojection_px
    (0.0036 vs the fixed version's 2.74) and picked the "better" number,
    which was actually the worse calibration. Use this ALONGSIDE reprojection
    error, never as a replacement for it -- a calibration can fail this check
    while still having a fine line-only reprojection error, and vice versa."""
    cfg = cfg or UltimatePitchConfiguration()
    lo_x, hi_x = -margin * frame_w, (1 + margin) * frame_w
    lo_y, hi_y = -margin * frame_h, (1 + margin) * frame_h
    for fx, fy in cfg.vertices:
        px, py = field_to_pixel(fx, fy, calib)
        if not (lo_x <= px <= hi_x and lo_y <= py <= hi_y):
            return False
    return True


def is_in_field(x, y, calib: Calibration, cfg: UltimatePitchConfiguration = None,
                 margin_cm: float = 0.0) -> bool:
    """Is pixel (x,y) inside the field boundary (including endzones), per the
    fitted calibration? Works even for field regions (e.g. a far endzone)
    that were never directly visible/clicked -- see module docstring.

    WARNING: this asks only whether the point PROJECTS inside the field. It
    does not ask whether the calibration can be trusted there. Most
    calibrations from this footage are LOCAL (see `field_validity_region`) --
    canvas_002 is meaningful only out to x=54.7m of a 100m field -- and a
    pixel beyond that limit projects to an arbitrary coordinate that lands
    inside the boundary as often as not. For anything MEASURED from a
    position (speed, distance, which third of the field play is in), use
    `is_measurable` instead."""
    cfg = cfg or UltimatePitchConfiguration()
    fx, fy = pixel_to_field(x, y, calib)
    return (-margin_cm <= fx <= cfg.length + margin_cm) and (-margin_cm <= fy <= cfg.width + margin_cm)


GOAL_EDGES = ((3, 4), (5, 6))
_OBSERVED_COLOUR = (0, 230, 0)     # BGR green  -- span the clicks actually cover
_ESTIMATED_COLOUR = (0, 200, 255)  # amber      -- computed from WFDF geometry
_GOAL_COLOUR = (255, 0, 255)       # magenta    -- goal lines, either way


def draw_dashed_line(image, p1, p2, colour, thickness: int = 3, dash: int = 22):
    """Dashes in IMAGE space (uniform on screen), so a dashed line reads the
    same whether it is near or far. Shared with field_model.draw_field_model
    so the two drawings mean the same thing by the same marks."""
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    length = float(np.hypot(*(p2 - p1)))
    if length < 1e-6 or not np.isfinite(length):
        return
    steps = max(2, int(length / dash))
    for k in range(0, steps, 2):
        a = p1 + (p2 - p1) * (k / steps)
        b = p1 + (p2 - p1) * min(1.0, (k + 1) / steps)
        cv2.line(image, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), colour, thickness)


def _evidence_points_by_edge(calib: Calibration, cfg: UltimatePitchConfiguration,
                              tol_cm: float = 200.0) -> dict:
    """The calibration's OWN clicks, in field coordinates, grouped by the edge
    they support. Keys are oriented as in cfg.edges."""
    verts = cfg.vertices
    known = [tuple(e) for e in cfg.edges]
    by_edge = {}
    if calib.line_details:
        # a line fit already says which edge each click belongs to
        for entry in calib.line_details:
            edge = tuple(entry.get("edge", ()))
            if len(edge) != 2:
                continue
            key = edge if edge in known else edge[::-1]
            if key not in known:
                continue
            for x, y in entry.get("points", []):
                by_edge.setdefault(key, []).append(pixel_to_field(x, y, calib))
    else:
        # a point fit does not, so ask geometry: a clicked corner is evidence
        # for every edge it lies on
        for fx, fy in (calib.field_points or []):
            for edge in cfg.edges:
                a = np.array(verts[edge[0] - 1], dtype=np.float64)
                b = np.array(verts[edge[1] - 1], dtype=np.float64)
                ab = b - a
                t = float(np.clip((np.array([fx, fy]) - a) @ ab / (ab @ ab), 0.0, 1.0))
                if np.hypot(*(a + ab * t - np.array([fx, fy]))) <= tol_cm:
                    by_edge.setdefault(tuple(edge), []).append((float(fx), float(fy)))
    return by_edge


def evidence_spans(calib: Calibration, cfg: UltimatePitchConfiguration = None) -> dict:
    """How much of each field edge did this calibration's clicks actually see?

    Returns {(vi,vj): (t_lo, t_hi)}, t being the fraction along the edge from
    vertex vi to vertex vj. An edge with fewer than two supporting clicks is
    absent from the dict -- nothing was observed there, so all of it is
    extrapolation.

    This exists because the question "is this line measured or computed?" has
    a per-SEGMENT answer, not a per-line one. On torovcanyon.mp4's canvas_002
    the six clicks along the blue near sideline cover x=35.4m to 47.8m of its
    100m and nothing else; drawing the whole sideline the same way claims
    evidence for the other 88% of it. See results/field_lines_finding.md
    Result 3.14."""
    cfg = cfg or UltimatePitchConfiguration()
    verts = cfg.vertices
    spans = {}
    for edge, pts in _evidence_points_by_edge(calib, cfg).items():
        a = np.array(verts[edge[0] - 1], dtype=np.float64)
        b = np.array(verts[edge[1] - 1], dtype=np.float64)
        ab = b - a
        ts = [float(np.clip((np.array(p) - a) @ ab / (ab @ ab), 0.0, 1.0))
              for p in pts if np.all(np.isfinite(p))]
        if len(ts) >= 2 and max(ts) - min(ts) > 1e-6:
            spans[edge] = (min(ts), max(ts))
    return spans


def is_measurable(x, y, calib: Calibration, cfg: UltimatePitchConfiguration = None,
                   margin_cm: float = 0.0) -> bool:
    """Is pixel (x,y) inside the field AND inside the region this calibration
    can actually represent?

    The check `is_in_field` cannot make, and the one anything numeric needs.
    A calibration fitted from evidence at one end of a field is LOCAL: past
    its horizon the projection is meaningless, but the coordinates it returns
    are ordinary finite numbers that frequently fall inside 0..100m x 0..37m.
    Filtering with `is_in_field` alone therefore admits positions that are
    pure noise, and a speed computed between two such positions is noise
    times a frame rate.

    MEASURED (results/field_lines_finding.md Result 3.14): canvas_002, the
    calibration behind the numbers in results/metrics_finding.md, is usable
    to x=54.7m of a 100m field. Roughly the far half of every position it
    reports was never measurable."""
    cfg = cfg or UltimatePitchConfiguration()
    fx, fy = pixel_to_field(x, y, calib)
    if not (np.isfinite(fx) and np.isfinite(fy)):
        return False
    if not (-margin_cm <= fx <= cfg.length + margin_cm and
            -margin_cm <= fy <= cfg.width + margin_cm):
        return False
    return fx <= field_validity_region(calib, cfg)["usable_x_max"]


def draw_field_overlay(image, calib: Calibration, cfg: UltimatePitchConfiguration = None,
                        clip_to_validity: bool = True, legend: bool = True,
                        samples: int = 96, spans: dict = None):
    """Project the field outline onto the image, drawn so it never claims more
    than it knows:

      SOLID   -- the part of an edge the clicks actually cover (measured)
      DASHED  -- the rest, computed from the homography + WFDF dimensions
      not drawn at all -- anything past `field_validity_region`'s usable_x_max,
                         where the fit is extrapolating past its own horizon

    The old version drew all six edges as one flat yellow outline. That is what
    made canvas_002's bowtie look like a broken calibration rather than what it
    is (a LOCAL one, good to x=54.7m -- results/field_lines_finding.md Result
    3.14): the fold lives entirely in the extrapolated far end, and a drawing
    that gives guesses and measurements the same ink cannot show that.
    `field_model.draw_field_model` already used solid/dashed this way; this is
    the same convention on the calibration's own outline.

    Pass `clip_to_validity=False` to see the raw projection including any fold
    -- that is a diagnostic view, not the one to judge a calibration by.

    `spans` overrides the evidence lookup, for callers drawing a calibration
    whose clicks live in a DIFFERENT image than the one being drawn on: a
    per-frame stand-in calibration (run_calibrate_preview.py) or a bank
    composition (calibration_bank.py) carries no clicks of its own, but the
    spans are field-space quantities, so the source calibration's are still
    exactly right. Compute them once with `evidence_spans(source_calib)`."""
    cfg = cfg or UltimatePitchConfiguration()
    vis = image.copy()
    verts = cfg.vertices
    spans = evidence_spans(calib, cfg) if spans is None else spans
    region = field_validity_region(calib, cfg)
    x_max = region["usable_x_max"] if clip_to_validity else float("inf")
    H_inv = np.linalg.inv(calib.H())
    goal_edges = [tuple(e) for e in GOAL_EDGES]
    h_img, w_img = vis.shape[:2]
    n_visible = 0  # samples that land inside the frame at all

    def project(field_pts):
        px = cv2.perspectiveTransform(
            np.asarray(field_pts, dtype=np.float64).reshape(-1, 1, 2), H_inv).reshape(-1, 2)
        return px

    for edge in cfg.edges:
        edge = tuple(edge)
        a = np.array(verts[edge[0] - 1], dtype=np.float64)
        b = np.array(verts[edge[1] - 1], dtype=np.float64)
        span = spans.get(edge)
        colour_est = _GOAL_COLOUR if edge in goal_edges else _ESTIMATED_COLOUR
        colour_obs = _GOAL_COLOUR if edge in goal_edges else _OBSERVED_COLOUR
        ts = np.linspace(0.0, 1.0, samples + 1)
        field_pts = [a + (b - a) * t for t in ts]
        px = project(field_pts)
        drawable = [bool(p[0] <= x_max) and bool(np.all(np.isfinite(q))) and
                    bool(np.max(np.abs(q)) < 1e5) for p, q in zip(field_pts, px)]
        observed = [span is not None and span[0] - 1e-9 <= t <= span[1] + 1e-9 for t in ts]
        n_visible += sum(1 for d, q in zip(drawable, px)
                         if d and 0 <= q[0] < w_img and 0 <= q[1] < h_img)
        i = 0
        while i < len(ts):
            if not drawable[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(ts) and drawable[j + 1] and observed[j + 1] == observed[i]:
                j += 1
            # carry one sample into the next run so observed/estimated meet
            # without a visible gap at the handover
            end = j + 1 if (j + 1 < len(ts) and drawable[j + 1]) else j
            p1 = tuple(int(round(v)) for v in px[i])
            p2 = tuple(int(round(v)) for v in px[end])
            if p1 != p2:
                if observed[i]:
                    cv2.line(vis, p1, p2, colour_obs, 2)
                else:
                    draw_dashed_line(vis, p1, p2, colour_est, 2)
            i = j + 1

    # corners: filled where a click landed on one, hollow where its position is
    # purely computed, absent where it is past the validity horizon
    n_hidden = 0
    for vi in range(1, 9):
        fx, fy = verts[vi - 1]
        if fx > x_max:
            n_hidden += 1
            continue
        p = project([(fx, fy)])[0]
        if not np.all(np.isfinite(p)) or np.max(np.abs(p)) >= 1e5:
            n_hidden += 1
            continue
        supported = False
        for edge, (t_lo, t_hi) in spans.items():
            if edge[0] == vi and t_lo <= 0.02:
                supported = True
            if edge[1] == vi and t_hi >= 0.98:
                supported = True
        centre = (int(round(p[0])), int(round(p[1])))
        cv2.circle(vis, centre, 6, (0, 0, 255), -1 if supported else 2)

    if legend:
        lines = ["solid = fitted to clicked paint    dashed = computed from WFDF geometry"]
        if not spans:
            # a bank-composed calibration keeps only its homography and its
            # source error; its clicks are not carried through composition
            # (calibration_bank.py), so NOTHING here can be shown as measured
            lines.append("no per-line evidence recorded for this calibration -- "
                         "all of it drawn as computed")
        if clip_to_validity:
            note = (f"valid to x={region['usable_x_max'] / 100:.1f}m "
                    f"({region['usable_fraction'] * 100:.0f}% of field length)")
            if n_hidden:
                note += f"; {n_hidden} corner(s) beyond it, not drawn"
            lines.append(note)
        else:
            lines.append("UNCLIPPED diagnostic view -- includes any fold past the horizon")
        if not n_visible:
            # canvas_004 does this: every part of the field it can still be
            # trusted about projects outside the frame entirely. An empty
            # overlay is the honest drawing, but say so rather than look broken
            lines.append("nothing drawn: no part of the valid region "
                         "projects into this frame")
        if calib.source_max_reprojection_px is not None:
            lines.append(f"worst clicked-line reprojection: "
                         f"{calib.source_max_reprojection_px:.1f}px")
        # shrink to fit rather than run off the edge -- bank canvases are not
        # all full-resolution frames (canvas_000 is 641px wide)
        font, base = cv2.FONT_HERSHEY_SIMPLEX, 0.6
        widest = max(cv2.getTextSize(t, font, base, 1)[0][0] for t in lines)
        scale = base * min(1.0, (vis.shape[1] - 24) / max(1.0, widest))
        step = max(11, int(round(26 * scale / base)))
        y = vis.shape[0] - step // 2 - step * (len(lines) - 1)
        for text in lines:
            for c, th in (((0, 0, 0), max(2, int(round(4 * scale / base)))),
                          ((255, 255, 255), 1)):
                cv2.putText(vis, text, (12, y), font, scale, c, th, cv2.LINE_AA)
            y += step
    return vis


def field_in_front_of_camera(calib: Calibration, cfg: UltimatePitchConfiguration = None) -> bool:
    """Does the whole field lie on ONE side of the horizon?

    THE check that the other two structurally cannot make, and the actual
    defect behind bank canvas_002 (see results/field_lines_finding.md).
    Under field->image, each vertex has a homogeneous w = H_inv[2] . (fx,fy,1).
    A sign change between vertices means the fitted camera puts part of the
    field BEHIND itself -- the image of the line at infinity cuts through the
    playing surface. No real camera filming a field can do that.

    Why the existing checks miss it:
      - `vertices_within_bounds` only catches a field that EXPLODES (a vertex
        projecting hundreds of thousands of px away). A field that collapses
        past the horizon has every vertex comfortably inside a generous box,
        just folded.
      - `field_lines.projected_field_is_sane` catches the resulting bowtie
        shape, which is a SYMPTOM. This catches the cause, and does so with a
        physical statement rather than a geometric one.

    MEASURED, and the separation is total:
      canvas_000 (verified good)      w signs + + + + + + + +   -> True
      canvas_002 / canvas_003         w signs - - - - + - + -   -> False
      canvas_004 (known bad, 55px)    w signs + - + - - - - -   -> False
      five synthetic sideline cameras                           -> True
    canvas_002 puts vertices 5 and 7 -- the far end of the field at y=0 --
    on the far side of the horizon from everything else, which is exactly why
    its far half projects nonsensically while its clicked stretches fit to
    2.7px."""
    cfg = cfg or UltimatePitchConfiguration()
    H_inv = np.linalg.inv(calib.H())
    ws = [float(H_inv[2, 0] * fx + H_inv[2, 1] * fy + H_inv[2, 2])
          for fx, fy in cfg.vertices[:8]]
    if any(abs(w) < 1e-12 for w in ws):
        return False   # a vertex sits exactly ON the horizon
    return all(w > 0 for w in ws) or all(w < 0 for w in ws)


def field_covers_points(calib: Calibration, image_points,
                         cfg: UltimatePitchConfiguration = None,
                         margin_cm: float = 800.0) -> bool:
    """Do the pixels this calibration was FITTED FROM land inside the field
    it produces?

    A fit is supposed to explain its own evidence. A homography that maps
    every clicked point outside the 100m x 37m boundary has explained nothing
    -- it has found a projective map that satisfies the line equations while
    placing the actual field somewhere the clicks never were. That is the
    collapse mode `vertices_within_bounds` cannot see, since it only tests
    for explosion.

    `margin_cm` is generous (8m) on purpose: clicks legitimately land just
    outside a boundary line, and a sideline clicked from the stands can sit a
    metre or two wide. This is meant to catch a fit that is off by a field,
    not one that is off by a metre."""
    cfg = cfg or UltimatePitchConfiguration()
    if not len(image_points):
        return True
    inside = 0
    for x, y in image_points:
        fx, fy = pixel_to_field(x, y, calib)
        if not (np.isfinite(fx) and np.isfinite(fy)):
            continue
        if (-margin_cm <= fx <= cfg.length + margin_cm and
                -margin_cm <= fy <= cfg.width + margin_cm):
            inside += 1
    return inside >= 0.75 * len(image_points)


def field_validity_region(calib: Calibration, cfg: UltimatePitchConfiguration = None,
                           safety: float = 0.85):
    """Where on the field is this calibration actually usable?

    MEASURED, and this function exists because a hard pass/fail was the wrong
    shape of answer (torovcanyon.mp4, canvas_002 -- see
    results/field_lines_finding.md). That calibration fails
    `field_in_front_of_camera`, and the person is right that it is still the
    most accurate one available: round-tripping a grid of field points through
    it gives 1 image pixel = 0.01-0.04 m on the ground everywhere from x=0 to
    x=60m. It is not degenerate. Its horizon simply falls ACROSS the field --
    the field vanishes at x=64.4m along one sideline and x=117.5m along the
    other -- so the far corners land on opposite sides of it, and that sign
    split is the bowtie. Everything the camera can actually see is fine.

    A calibration fitted from evidence at one end of a field is therefore not
    "wrong"; it is LOCAL. Rejecting it throws away a good measurement of the
    near half. Reporting its validity region keeps it, and keeps the far half
    honestly unavailable.

    Returns a dict:
      horizon_x_at(y)     -- callable, field x where the plane vanishes at that y
      usable_x_max        -- smallest horizon x across the field's width, scaled
                             by `safety` (perspective compresses violently as the
                             horizon is approached, so stop short of it)
      whole_field_usable  -- True when the entire field is comfortably in front
      usable_fraction     -- fraction of the field's length that is usable
    """
    cfg = cfg or UltimatePitchConfiguration()
    H_inv = np.linalg.inv(calib.H())
    a, b, c = H_inv[2, 0], H_inv[2, 1], H_inv[2, 2]

    def horizon_x_at(y):
        if abs(a) < 1e-15:
            return float("inf")
        return -(b * y + c) / a

    xs = [horizon_x_at(y) for y in (0.0, cfg.width / 2.0, float(cfg.width))]
    positive = [x for x in xs if x > 0]
    if not positive:
        # No horizon crossing at any positive x. That is NOT "unusable" -- it
        # means the vanishing line sits behind x=0, so w keeps one sign across
        # the whole 0..length extent and every part of the field is equally
        # projectable. (Measured on a synthetic camera placed 15m behind the
        # near back line looking down the field: every horizon_x_at(y) is
        # negative, and the field projects perfectly. The earlier version
        # returned usable_x_max=0 for exactly this camera, which only stayed
        # invisible while the number was reported rather than used -- it now
        # decides what draw_field_overlay is allowed to draw.)
        if any(np.isnan(x) for x in xs) or any(x == 0 for x in xs):
            # degenerate: the vanishing line passes through the field's own
            # x=0 edge (or the fit is not a usable map at all)
            return {"horizon_x_at": horizon_x_at, "usable_x_max": 0.0,
                    "whole_field_usable": False, "usable_fraction": 0.0}
        return {"horizon_x_at": horizon_x_at, "usable_x_max": float(cfg.length),
                "whole_field_usable": True, "usable_fraction": 1.0}
    usable = min(positive) * safety
    return {"horizon_x_at": horizon_x_at,
            "usable_x_max": float(usable),
            "whole_field_usable": bool(usable >= cfg.length),
            "usable_fraction": float(min(1.0, usable / cfg.length))}


def calibration_covers_evidence(calib: Calibration, image_points,
                                 cfg: UltimatePitchConfiguration = None) -> bool:
    """Is the calibration valid where its own evidence lives?

    The replacement for using `field_in_front_of_camera` as a veto. A local
    calibration is fine; a calibration whose valid region does not even contain
    the points it was fitted from is not, because then nothing it says is
    supported by anything."""
    cfg = cfg or UltimatePitchConfiguration()
    if not len(image_points):
        return True
    region = field_validity_region(calib, cfg)
    ok = 0
    for x, y in image_points:
        fx, fy = pixel_to_field(x, y, calib)
        if np.isfinite(fx) and np.isfinite(fy) and 0.0 <= fx <= region["usable_x_max"]:
            ok += 1
    return ok >= 0.75 * len(image_points)


def _evidence_field_points(calib: Calibration, cfg: UltimatePitchConfiguration = None):
    """Every field-space point this calibration was actually fitted from:
    the explicit field_points, plus both endpoint vertices of each labelled
    line edge. Used to decide which SIDE of the horizon the camera is on."""
    cfg = cfg or UltimatePitchConfiguration()
    pts = [tuple(p) for p in (calib.field_points or [])]
    for det in (calib.line_details or []):
        vi, vj = det["edge"]
        pts.append(tuple(cfg.vertices[vi - 1]))
        pts.append(tuple(cfg.vertices[vj - 1]))
    return np.array(pts, dtype=float) if pts else np.zeros((0, 2))


def field_visibility(calib: Calibration, fx_cm, fy_cm,
                     cfg: UltimatePitchConfiguration = None,
                     max_m_per_px: float = 0.5):
    """Boolean mask over arbitrary field points: can this calibration
    represent the ground there at all?

    WHY THIS EXISTS ALONGSIDE field_validity_region. That function answers
    with a single scalar, `usable_x_max`, and every caller reads it as
    "usable where x <= this". That silently assumes the camera stands at the
    x=0 end looking up-field. MEASURED (calibration_corners.json, the
    t=1700s corner-clicked fit): that calibration is anchored on vertices
    5-8, i.e. the FAR endzone at x=82..100m, and field_validity_region
    reports usable_x_max=11.7m -- the opposite end from everything it can
    see. Masking with `x <= 11.7m` paints the one strip of field that
    calibration knows nothing about, and drops the endzone it was fitted on.
    That is the whole reason two endzone rectifications could not be
    combined: one of them was being masked backwards.

    Two side-agnostic tests replace the scalar:

      IN FRONT OF THE CAMERA -- w, the projective denominator of
      field->pixel, is zero exactly on the horizon and flips sign across it.
      Points whose w-sign matches the sign at the fitted evidence are on the
      camera's side of it; the rest are the bowtie, projecting to arithmetic
      rather than geometry. This needs no assumption about which end the
      camera is at, because the evidence itself says which side is real.

      RESOLVED FINELY ENOUGH -- near the horizon perspective compresses
      violently, so being merely in front is not enough. Ground resolution
      (metres of field per image pixel, from the singular values of the
      field->pixel Jacobian) is the honest measure of that, and it is the
      one this codebase already reasons in: field_validity_region's own
      docstring cites "1 image pixel = 0.01-0.04 m on the ground from x=0 to
      x=60m" as the evidence a local calibration is good. Default cutoff
      0.5 m/px is ~15x coarser than that measured good range, so it excludes
      only genuinely collapsed geometry, and replaces the old `safety=0.85`
      fudge with a number that means something.
    """
    cfg = cfg or UltimatePitchConfiguration()
    fx_cm = np.asarray(fx_cm, dtype=float)
    fy_cm = np.asarray(fy_cm, dtype=float)
    H_f2p = np.linalg.inv(np.asarray(calib.H()))

    def w_of(x, y):
        return H_f2p[2, 0] * x + H_f2p[2, 1] * y + H_f2p[2, 2]

    ev = _evidence_field_points(calib, cfg)
    if len(ev) == 0:
        raise ValueError("calibration records no field-space evidence, so which side of "
                         "the horizon the camera is on cannot be determined")
    want = np.sign(np.median(w_of(ev[:, 0], ev[:, 1])))
    w = w_of(fx_cm, fy_cm)
    in_front = np.sign(w) == want

    # ground resolution from the Jacobian, by central differences in cm
    d = 1.0
    def proj(x, y):
        p = np.stack([x, y, np.ones_like(x)], axis=-1) @ H_f2p.T
        return p[..., :2] / p[..., 2:3]
    base = proj(fx_cm, fy_cm)
    jx = (proj(fx_cm + d, fy_cm) - base) / d          # px per cm, along field x
    jy = (proj(fx_cm, fy_cm + d) - base) / d          # px per cm, along field y
    J = np.stack([jx, jy], axis=-1)                   # (..., 2 px, 2 field)
    with np.errstate(invalid="ignore", divide="ignore"):
        smin = np.linalg.svd(J, compute_uv=False)[..., -1]   # px per cm, worst direction
        m_per_px = 1.0 / (smin * 100.0)                      # cm->m
    fine_enough = np.isfinite(m_per_px) & (m_per_px <= max_m_per_px)
    return in_front & fine_enough
