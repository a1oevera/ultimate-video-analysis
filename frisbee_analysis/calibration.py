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


def is_in_field(x, y, calib: Calibration, cfg: UltimatePitchConfiguration = None,
                 margin_cm: float = 0.0) -> bool:
    """Is pixel (x,y) inside the field boundary (including endzones), per the
    fitted calibration? Works even for field regions (e.g. a far endzone)
    that were never directly visible/clicked -- see module docstring."""
    cfg = cfg or UltimatePitchConfiguration()
    fx, fy = pixel_to_field(x, y, calib)
    return (-margin_cm <= fx <= cfg.length + margin_cm) and (-margin_cm <= fy <= cfg.width + margin_cm)


def draw_field_overlay(image, calib: Calibration, cfg: UltimatePitchConfiguration = None):
    """Project the FULL field outline (including any off-camera portions,
    e.g. an endzone never directly clicked) onto the image via the fitted
    homography, for visual sanity-checking."""
    cfg = cfg or UltimatePitchConfiguration()
    vis = image.copy()
    verts_px = [field_to_pixel(fx, fy, calib) for fx, fy in cfg.vertices]
    for i, j in cfg.edges:
        p1 = tuple(int(v) for v in verts_px[i - 1])
        p2 = tuple(int(v) for v in verts_px[j - 1])
        cv2.line(vis, p1, p2, (0, 255, 255), 2)
    # explicitly draw both goal lines (endzone boundaries), whether or not
    # their own corner cones were ever visible/clicked -- this is the "can we
    # estimate an off-camera endzone" answer made visible
    for gx in (cfg.goal_line_near, cfg.goal_line_far):
        p1 = tuple(int(v) for v in field_to_pixel(gx, 0, calib))
        p2 = tuple(int(v) for v in field_to_pixel(gx, cfg.width, calib))
        cv2.line(vis, p1, p2, (255, 0, 255), 2)
    for x, y in verts_px:
        cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 255), -1)
    return vis
