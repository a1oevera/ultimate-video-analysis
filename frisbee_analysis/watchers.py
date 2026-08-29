"""
People as field evidence: a row of watchers marks a sideline that has no paint.

CONTRIBUTED AND MEASURED BY THE PERSON, reproduced here as a module. On
`torovcanyon.mp4` t=3945-4005s (canvas_002's own window), YOLO11n over 60
frames gives 2,710 person detections in ~45s of CPU, foot points mapped into
the segment's reference coordinates through the existing registration. It
finds exactly the line that has no paint.

THE DISCRIMINATOR IS FREE. A watcher's foot point sits in nearly the same
registered spot all window; a running player's does not. Counting distinct
frames with a foot within 12px:

    >= 5 frames   1,929 / 2,710
    >= 20 frames  1,228
    >= 30 frames    898

RANSAC over the >=20 set yields one dominant row: **1,060 points, straight to
4.5px RMS over a 1,330px extent, present in all 60 frames**. Fed to
`field_lines.group_by_vanishing_point` alongside the 12 tracked paint lines it
lands in family B with the blue sideline (support 410k) with no special
handling -- the existing pipeline already accepts it as a sideline-direction
line.

IT IS A DIRECTION CUE, NOT A POSITION ONE. Through canvas_002 those feet sit
at field y ~ 41.1m (IQR 40.3-41.9) on a 37m-wide field: the row stands ~4m
OUTSIDE. The trend is y = 37.33 + 0.107x with 0.78m residual RMS over
x = 17.3-54.4m. Used directly as the sideline it is off by 4-6m. Its direction
disagrees with canvas_002's far sideline by 2.28 degrees (16.8px mean offset)
-- unresolved, and either people not standing parallel or canvas_002 being
slightly rotated. It could not be settled against paint, because no tracked
paint line lands anywhere near y=37m in the valid x range, which independently
confirms the far sideline is unpainted at midfield.

THE FAILURE MODE IS REAL AND IMMEDIATE. The second static cluster -- 167
points, straight to 1.0px, *straighter than the players* -- is the fence/path
beyond the field, 10.2 degrees off the sideline. A straight row of standing
people is not by itself a sideline. Two separators worked: those people are
smaller (median box 36px vs 47px), and they sit outside
`field_lines.playing_surface_mask`, which this project already computes.

SO THE INTENDED USES, in order of confidence:
  1. A LABELLING GATE. Watchers stand outside the field, so any labelling
     that puts a dense static collinear row INSIDE the boundary is wrong.
     This is evidence the fit never consumed, so it can rank labellings the
     way reprojection error structurally cannot -- the same role as cone
     hints, with 1,060 points behind it instead of 1 of 8.
  2. A DIRECTION CONSTRAINT for the unpainted far sideline, as a
     vanishing-point family member.
  3. NOT a position, unless the standoff is calibrated -- which is possible,
     because the far sideline IS painted inside the endzones: measure the
     offset where paint and people coexist and carry it across midfield.
"""
from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass, field


@dataclass
class StaticCluster:
    """One registered position a person occupied across many frames."""
    x: float
    y: float
    n_frames: int
    n_detections: int
    median_box_h: float


@dataclass
class WatcherRow:
    """A dense, static, collinear row of people -- candidate sideline marker."""
    line: np.ndarray              # homogeneous [a,b,c], normalized
    points: list                  # cluster centres on the row
    rms_px: float
    n_frames: int
    extent_px: float
    median_box_h: float
    fraction_on_surface: float    # how much of the row sits on the playing surface

    @property
    def angle_deg(self) -> float:
        return float(np.degrees(np.arctan2(-self.line[0], self.line[1])) % 180.0)


def foot_points(boxes) -> list:
    """Bottom-centre of each person box -- where they touch the ground."""
    return [((x0 + x1) / 2.0, y1) for x0, y0, x1, y1 in boxes]


def collect_static_clusters(frames, registrations, segment_id=None, yolo_model=None,
                             cluster_px: float = 12.0, min_frames: int = 20,
                             conf: float = 0.20, progress: bool = False) -> list:
    """Foot points across a segment, fused in reference coordinates, keeping
    only positions a person actually stood in.

    `min_frames=20` and `cluster_px=12` are the person's measured settings:
    at >=20 distinct frames within 12px, 1,228 of 2,710 detections survive,
    which is where the standing row separates cleanly from running players."""
    from .field_lines import person_boxes_from_yolo, _transform_points
    by_segment = {}
    for r in registrations:
        if r.ok and r.homography is not None:
            by_segment.setdefault(r.segment_id, []).append(r)
    if not by_segment:
        return []
    if segment_id is None:
        segment_id = max(by_segment, key=lambda s: len(by_segment[s]))
    members = by_segment.get(segment_id, [])

    clusters = []
    for n, reg in enumerate(members):
        frame = frames[reg.frame_idx]
        boxes = person_boxes_from_yolo(yolo_model, frame, conf) if yolo_model else []
        if progress:
            print(f"  frame {reg.frame_idx} ({n + 1}/{len(members)}): {len(boxes)} people")
        for box, (fx, fy) in zip(boxes, foot_points(boxes)):
            (rx, ry), = _transform_points([(fx, fy)], reg.homography)
            box_h = float(box[3] - box[1])
            for c in clusters:
                if np.hypot(rx - c["x"], ry - c["y"]) <= cluster_px:
                    c["n"] += 1
                    c["frames"].add(reg.frame_idx)
                    c["x"] += (rx - c["x"]) / c["n"]
                    c["y"] += (ry - c["y"]) / c["n"]
                    c["box_h"].append(box_h)
                    break
            else:
                clusters.append({"x": rx, "y": ry, "n": 1, "frames": {reg.frame_idx},
                                  "box_h": [box_h]})
    return [StaticCluster(c["x"], c["y"], len(c["frames"]), c["n"],
                           float(np.median(c["box_h"])))
            for c in clusters if len(c["frames"]) >= min_frames]


def find_rows(clusters, surface_mask=None, min_points: int = 30,
               inlier_px: float = 12.0, max_rows: int = 3,
               iterations: int = 800, seed: int = 0) -> list:
    """RANSAC the static clusters into dominant collinear rows.

    Returns rows sorted by point count. Each carries the measurements that
    separate a watcher row from the OTHER thing this finds -- the fence/path
    beyond the field, which on the test window was 167 points straight to
    1.0px, i.e. *straighter* than the people. Straightness is therefore not a
    usable criterion; `median_box_h` (36px vs 47px there) and
    `fraction_on_surface` are."""
    if len(clusters) < min_points:
        return []
    pts = np.array([[c.x, c.y] for c in clusters], dtype=np.float64)
    rng = np.random.default_rng(seed)
    remaining = list(range(len(clusters)))
    rows = []
    for _ in range(max_rows):
        if len(remaining) < min_points:
            break
        best = (0, None)
        for _ in range(iterations):
            i, j = rng.choice(len(remaining), size=2, replace=False)
            a, b = pts[remaining[i]], pts[remaining[j]]
            line = np.cross([a[0], a[1], 1.0], [b[0], b[1], 1.0])
            norm = np.hypot(line[0], line[1])
            if norm < 1e-9:
                continue
            line = line / norm
            d = np.abs(pts[remaining] @ line[:2] + line[2])
            inliers = [remaining[k] for k in np.nonzero(d <= inlier_px)[0]]
            if len(inliers) > best[0]:
                best = (len(inliers), inliers)
        if best[1] is None or best[0] < min_points:
            break
        idx = best[1]
        sub = pts[idx]
        # refit by total least squares over all inliers
        A = np.column_stack([sub[:, 0], sub[:, 1], np.ones(len(sub))])
        _, _, Vt = np.linalg.svd(A)
        line = Vt[-1]
        line = line / np.hypot(line[0], line[1])
        residuals = np.abs(sub @ line[:2] + line[2])
        direction = np.array([-line[1], line[0]])
        t = sub @ direction
        frames = max(clusters[k].n_frames for k in idx)
        on_surface = 1.0
        if surface_mask is not None:
            h, w = surface_mask.shape[:2]
            hits = 0
            for k in idx:
                xi, yi = int(round(clusters[k].x)), int(round(clusters[k].y))
                if 0 <= xi < w and 0 <= yi < h and surface_mask[yi, xi] > 0:
                    hits += 1
            on_surface = hits / len(idx)
        rows.append(WatcherRow(line=line, points=[(float(p[0]), float(p[1])) for p in sub],
                                rms_px=float(np.sqrt((residuals ** 2).mean())),
                                n_frames=int(frames),
                                extent_px=float(t.max() - t.min()),
                                median_box_h=float(np.median([clusters[k].median_box_h for k in idx])),
                                fraction_on_surface=float(on_surface)))
        remaining = [k for k in remaining if k not in set(idx)]
    rows.sort(key=lambda r: -len(r.points))
    return rows[:max_rows]


def watchers_outside_field(calib, rows, cfg=None, margin_cm: float = 200.0,
                            min_points: int = 30) -> bool:
    """THE GATE, and the person's highest-confidence use of this evidence.

    Watchers stand OUTSIDE the boundary. So a labelling that places a dense,
    static, collinear row of people inside the field is wrong. Crucially this
    is evidence the homography fit never consumed, so unlike reprojection
    error it can actually discriminate between labellings -- the same role as
    cone corroboration, but backed by 1,060 points rather than 1 cone in 8.

    Returns False when any qualifying row sits mostly inside the field."""
    from .calibration import pixel_to_field
    from .ultimate_config import UltimatePitchConfiguration
    cfg = cfg or UltimatePitchConfiguration()
    for row in rows:
        if len(row.points) < min_points:
            continue
        inside = 0
        counted = 0
        for x, y in row.points:
            fx, fy = pixel_to_field(x, y, calib)
            if not (np.isfinite(fx) and np.isfinite(fy)):
                continue
            counted += 1
            if (margin_cm <= fx <= cfg.length - margin_cm and
                    margin_cm <= fy <= cfg.width - margin_cm):
                inside += 1
        if counted and inside / counted > 0.5:
            return False
    return True


def standoff_from_paint(row: WatcherRow, calib, cfg=None, sideline_y: float = None):
    """How far outside the sideline this row of people is standing, in cm.

    The person's route to using watchers as a POSITION cue rather than only a
    direction: the far sideline IS painted inside the endzones, so measure the
    offset where paint and people coexist and carry it across midfield, where
    no paint exists. Measured on the test window through canvas_002: feet at
    field y ~ 41.1m (IQR 40.3-41.9) against a 37m-wide field, i.e. ~4m out,
    with the trend y = 37.33 + 0.107x and 0.78m residual RMS.

    Returns (median_offset_cm, iqr_cm, n) -- report all three, because a
    single median hides that the standoff DRIFTS along the field."""
    from .calibration import pixel_to_field
    from .ultimate_config import UltimatePitchConfiguration
    cfg = cfg or UltimatePitchConfiguration()
    if sideline_y is None:
        sideline_y = float(cfg.width)
    offsets = []
    for x, y in row.points:
        fx, fy = pixel_to_field(x, y, calib)
        if np.isfinite(fx) and np.isfinite(fy):
            offsets.append(fy - sideline_y)
    if not offsets:
        return (float("nan"), float("nan"), 0)
    arr = np.asarray(offsets)
    return (float(np.median(arr)),
            float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            len(arr))
