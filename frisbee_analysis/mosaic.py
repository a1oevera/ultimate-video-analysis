"""
Frame-to-mosaic registration: recovers each frame's homography against a
running reference so downstream code (player positions, speed/distance) can
work in one consistent frame even though the camera pans/tilts/zooms and
often shows only part of the field.

MEASURED (results/mosaic_finding.md, real footage): wide-framed background
(treeline, tents) registers reliably via ORB+RANSAC -- 74-84 inliers at
realistic 2-5s frame gaps, still 16+ at 63min apart. Tight close-ups get ~0
inliers -- no shared background to match. So registration is NOT forced
unconditionally: frames below `min_inliers` are flagged `ok=False` and left
for `interpolate_missing` to bridge (the fixed mount means true camera pose
changes smoothly, so interpolating across a short tight-zoom gap is a
reasonable bridge, not a hack -- see the finding doc for why).

Static broadcast overlay graphics (scoreboard, watermark, station bug) MUST
be masked out before feature detection -- unmasked, they're free "always
aligned" keypoints that say nothing about real camera motion. There's no
universal default box; pass your footage's actual overlay regions in
`MosaicConfig.overlay_mask_boxes`.
"""
from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass, field


@dataclass
class MosaicConfig:
    n_features: int = 3000
    ratio_test: float = 0.75
    ransac_thresh_px: float = 5.0
    min_inliers: int = 15  # measured cutoff: wide-vs-wide clears this easily; wide-vs-tight (0 inliers) never does
    overlay_mask_boxes: tuple = field(default_factory=tuple)  # (x0,y0,x1,y1) static-overlay regions to exclude


@dataclass
class FrameRegistration:
    frame_idx: int
    homography: np.ndarray | None  # maps this frame's pixel coords -> segment_id's own reference frame's coords
    n_inliers: int
    ok: bool  # True if n_inliers >= cfg.min_inliers (or this IS a segment's reference frame)
    segment_id: int = 0  # frames only share a coordinate system within the same segment_id -- see register_sequence


def _build_mask(shape, boxes):
    mask = np.full(shape, 255, dtype=np.uint8)
    for (x0, y0, x1, y1) in boxes:
        mask[y0:y1, x0:x1] = 0
    return mask


def _to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def register_pair(frame_a: np.ndarray, frame_b: np.ndarray, cfg: MosaicConfig = MosaicConfig()):
    """Homography mapping frame_b's pixel coords -> frame_a's pixel coords,
    and the RANSAC inlier count. Returns (None, 0) if there isn't enough
    shared background to trust (caller decides the threshold via n_inliers,
    or just check `>= cfg.min_inliers`)."""
    gray_a, gray_b = _to_gray(frame_a), _to_gray(frame_b)
    mask = _build_mask(gray_a.shape, cfg.overlay_mask_boxes)

    orb = cv2.ORB_create(nfeatures=cfg.n_features)
    kp_a, des_a = orb.detectAndCompute(gray_a, mask)
    kp_b, des_b = orb.detectAndCompute(gray_b, mask)
    if des_a is None or des_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        return None, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des_b, des_a, k=2)
    good = [m for m, n in matches if n and m.distance < cfg.ratio_test * n.distance]
    if len(good) < 4:
        return None, 0

    src = np.float32([kp_b[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_a[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, cfg.ransac_thresh_px)
    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    return H, n_inliers


def register_pair_with_coverage(frame_a: np.ndarray, frame_b: np.ndarray, cfg: MosaicConfig = MosaicConfig()):
    """Like register_pair, but also returns what fraction of frame_b's area
    is spanned by the convex hull of its INLIER keypoints.

    MEASURED (person caught it by eye, see results/calibration_finding.md):
    a calibration bank auto-match had visible drift on part of the frame far
    from where the matched keypoints actually were, despite a healthy inlier
    COUNT (1900+). Root cause: RANSAC's threshold only bounds error AT the
    matched points themselves -- it says nothing about how well-spread those
    points are. A homography fit from points clustered in one corner (e.g.
    background scenery) can be locally excellent there and still drift
    significantly when extrapolated to project a field line on the other
    side of the frame, where no matched point was anywhere near. Coverage
    (how much of the frame area the inlier points actually span) is a cheap
    proxy for "how far is this extrapolating" that a bare inlier count can't
    tell you."""
    gray_a, gray_b = _to_gray(frame_a), _to_gray(frame_b)
    mask = _build_mask(gray_a.shape, cfg.overlay_mask_boxes)

    orb = cv2.ORB_create(nfeatures=cfg.n_features)
    kp_a, des_a = orb.detectAndCompute(gray_a, mask)
    kp_b, des_b = orb.detectAndCompute(gray_b, mask)
    if des_a is None or des_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        return None, 0, 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des_b, des_a, k=2)
    good = [m for m, n in matches if n and m.distance < cfg.ratio_test * n.distance]
    if len(good) < 4:
        return None, 0, 0.0

    src = np.float32([kp_b[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_a[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, cfg.ransac_thresh_px)
    if H is None or inlier_mask is None:
        return None, 0, 0.0
    n_inliers = int(inlier_mask.sum())

    inlier_pts = src[inlier_mask.ravel().astype(bool)].reshape(-1, 2)
    frame_area = frame_b.shape[0] * frame_b.shape[1]
    if len(inlier_pts) >= 3 and frame_area > 0:
        hull_area = cv2.contourArea(cv2.convexHull(inlier_pts.astype(np.float32)))
        coverage = float(hull_area / frame_area)
    else:
        coverage = 0.0
    return H, n_inliers, coverage


def register_sequence(frames: list[np.ndarray], cfg: MosaicConfig = MosaicConfig()) -> list[FrameRegistration]:
    """Register each frame against the most recently successfully-registered
    frame (NOT simply the previous raw frame -- if that one failed, it isn't
    anchored in the mosaic either). Consecutive frames share far more
    background than a frame vs. a fixed reference tens of minutes away (see
    results/mosaic_finding.md), so registration is deliberately incremental.

    MEASURED (run_mosaic_test.py on real footage, including a broadcast intro
    card before gameplay starts): if the anchor itself is unrepresentative
    (e.g. frame 0 is a title card, not camera footage), every later frame
    fails against it even though later frames would match EACH OTHER fine --
    the anchor never advances and the whole rest of the sequence is wrongly
    marked unregisterable. Fix: when a frame fails against the stale anchor,
    also try it against the immediately preceding raw frame. If THAT
    succeeds, it's a coherent new shot (e.g. a broadcast cut from the intro
    into continuous gameplay) -- start a new segment there rather than
    stalling. Frames only share a coordinate system within the same
    `segment_id`; don't compare homographies across a segment boundary."""
    results = [FrameRegistration(0, np.eye(3), n_inliers=10**9, ok=True, segment_id=0)]
    last_good_idx = 0
    segment_id = 0
    for i in range(1, len(frames)):
        H_step, n_inliers = register_pair(frames[last_good_idx], frames[i], cfg)
        if H_step is not None and n_inliers >= cfg.min_inliers:
            H_full = results[last_good_idx].homography @ H_step
            results.append(FrameRegistration(i, H_full, n_inliers, True, segment_id))
            last_good_idx = i
            continue

        H_local, n_local = register_pair(frames[i - 1], frames[i], cfg)
        if H_local is not None and n_local >= cfg.min_inliers:
            segment_id += 1
            results.append(FrameRegistration(i, np.eye(3), n_local, True, segment_id))
            last_good_idx = i
        else:
            results.append(FrameRegistration(i, None, n_inliers, False, segment_id))
    return _reclaim_failed_frames(frames, results, cfg)


def _reclaim_failed_frames(frames, results: list[FrameRegistration], cfg: MosaicConfig) -> list[FrameRegistration]:
    """Second-chance pass for ok=False frames. MEASURED on real footage
    (results/calibration_finding.md): a frame that fails against the segment
    BEFORE it can still register excellently against the segment AFTER it --
    e.g. one frame scored only 5 inliers against the prior segment's end but
    1017 inliers against the next segment's start. The main forward pass
    never checks this direction (it only ever looks backward: against the
    stale anchor, then against the immediately preceding raw frame), so a
    frame like this gets permanently stranded even though it's really just
    the start of the next segment misclassified as a gap. Try each failed
    frame against the next available segment's reference frame; fold it in
    if it matches instead of leaving a browsable gap in the middle of good
    footage."""
    out = list(results)
    for i, r in enumerate(out):
        if r.ok:
            continue
        next_ok = next((j for j in range(r.frame_idx + 1, len(frames)) if out[j].ok), None)
        if next_ok is None:
            continue
        target = out[next_ok]
        H_step, n_step = register_pair(frames[target.frame_idx], frames[r.frame_idx], cfg)
        if H_step is not None and n_step >= cfg.min_inliers:
            H_full = target.homography @ H_step
            out[i] = FrameRegistration(r.frame_idx, H_full, n_step, True, target.segment_id)
    return out


def interpolate_missing(results: list[FrameRegistration]) -> list[FrameRegistration]:
    """Fill ok=False gaps by linearly interpolating the homography matrix
    entries between the nearest ok=True neighbors -- but ONLY within the same
    segment_id. Frames in different segments (e.g. either side of a broadcast
    cut) don't share a coordinate system, so interpolating across that
    boundary would silently produce a meaningless homography. An approximation
    (a homography isn't a linear space) but reasonable for the short gaps this
    is meant to bridge, given the fixed mount's pose changes smoothly within a
    segment -- do not use this to bridge a long run of failures with no nearby
    same-segment anchor."""
    good_idx = [r.frame_idx for r in results if r.ok]
    if not good_idx:
        return results
    out = list(results)
    for i, r in enumerate(out):
        if r.ok:
            continue
        same_seg = [g for g in good_idx if out[g].segment_id == r.segment_id]
        earlier = [g for g in same_seg if g < r.frame_idx]
        later = [g for g in same_seg if g > r.frame_idx]
        if earlier and later:
            a, b = max(earlier), min(later)
            t = (r.frame_idx - a) / (b - a)
            Ha, Hb = out[a].homography, out[b].homography
            H = (1 - t) * Ha + t * Hb
        elif earlier:
            H = out[max(earlier)].homography
        elif later:
            H = out[min(later)].homography
        else:
            H = None  # no same-segment anchor on either side -- nothing to interpolate from
        out[i] = FrameRegistration(r.frame_idx, H, r.n_inliers, ok=False, segment_id=r.segment_id)
    return out
