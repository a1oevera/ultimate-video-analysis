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
                     cfg: UltimatePitchConfiguration = None) -> Calibration:
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
                        image_path=str(image_path))


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
