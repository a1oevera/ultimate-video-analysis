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
    # MEASURED (results/calibration_finding.md, torovcanyon.mp4 near-goal/near-back
    # line calibration): inlier COUNT alone let a step-to-step registration through
    # whose matched features were all clustered in one region (e.g. background
    # scenery), giving a homography that was locally fine there and wildly wrong
    # (thousands of px off) when a click far from that cluster got projected
    # through it. Coverage (inlier convex-hull area / frame area) is a cheap proxy
    # for "how far is this extrapolating" that count can't tell you.
    #
    # First set to 0.25 (calibration_bank.py's BANK_MIN_COVERAGE), carried over
    # unvalidated for THIS regime -- MEASURED WRONG: real consecutive-frame
    # footage during faster camera motion (torovcanyon.mp4 t=3400s) legitimately
    # dips to ~0.14-0.24 coverage between adjacent 1fps samples, and 0.25 was
    # fragmenting a genuinely continuous 60-frame window into 6 tiny segments
    # (worst: 1 frame). Re-measured directly: walked the real chain on both the
    # window that had the original bug (t=300s) and the one that was
    # over-fragmenting (t=3400s) -- at 0.15, both hold together as one segment,
    # and the worst individual step accepted is 0.144-0.186, nowhere near the
    # ~0.02-0.09 seen in genuinely bad/unrelated-content matches (see
    # calibration_bank cross-session tests). 0.15 is the current best estimate,
    # not a final answer -- re-measure again if either failure mode recurs.
    min_coverage: float = 0.15
    overlay_mask_boxes: tuple = field(default_factory=tuple)  # (x0,y0,x1,y1) static-overlay regions to exclude


@dataclass
class FrameRegistration:
    frame_idx: int
    homography: np.ndarray | None  # maps this frame's pixel coords -> segment_id's own reference frame's coords
    n_inliers: int
    ok: bool  # True if n_inliers >= cfg.min_inliers AND coverage >= cfg.min_coverage (or this IS a segment's reference frame)
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
    `segment_id`; don't compare homographies across a segment boundary.

    MEASURED (results/calibration_finding.md, torovcanyon.mp4): count-only
    gating let a step through whose inliers all clustered in one part of the
    frame -- fine near that cluster, wildly wrong (thousands of px) once a
    click far from it got projected through the chained homography. Now
    gated on coverage too (see MosaicConfig.min_coverage) via
    register_pair_with_coverage, not just register_pair -- a step with
    plenty of inliers but too little spread starts a new segment instead of
    silently poisoning the chain."""
    results = [FrameRegistration(0, np.eye(3), n_inliers=10**9, ok=True, segment_id=0)]
    last_good_idx = 0
    segment_id = 0
    for i in range(1, len(frames)):
        H_step, n_inliers, coverage = register_pair_with_coverage(frames[last_good_idx], frames[i], cfg)
        if H_step is not None and n_inliers >= cfg.min_inliers and coverage >= cfg.min_coverage:
            H_full = results[last_good_idx].homography @ H_step
            results.append(FrameRegistration(i, H_full, n_inliers, True, segment_id))
            last_good_idx = i
            continue

        H_local, n_local, coverage_local = register_pair_with_coverage(frames[i - 1], frames[i], cfg)
        if H_local is not None and n_local >= cfg.min_inliers and coverage_local >= cfg.min_coverage:
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
        H_step, n_step, coverage = register_pair_with_coverage(frames[target.frame_idx], frames[r.frame_idx], cfg)
        if H_step is not None and n_step >= cfg.min_inliers and coverage >= cfg.min_coverage:
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


def refine_to_anchor(frames: list, results: list[FrameRegistration], cfg: MosaicConfig = MosaicConfig(),
                      max_hops: int = 2, max_candidates: int = 12, progress: bool = False):
    """Re-register each frame to its segment anchor by the SHORTEST reliable
    path, instead of the long incremental chain register_sequence produces.

    MEASURED (torovcanyon.mp4 t=3945-4005s, 60 frames): chained homographies
    diverge from a direct anchor registration by a median 2.2px at chain depth
    <=20 and **7.7px past depth 40**, max 8.9px. That is exactly the gap
    between the human-clicked lines (0.5-2.7px) and the tracked lines built on
    the chain (rms 4.0-8.4px) -- drift, not detection, is what degrades them.
    It is also why a 159-frame segment produced a pan extent 2,354 frame-widths
    wide.

    Direct-to-anchor is NOT a drop-in replacement: it succeeded on only 28 of
    59 frames here, the failures being coverage-limited (7-14% against the 15%
    floor) because a wide pan leaves distant frames sharing too little with the
    anchor. So this keeps chaining, but makes the chain short: every frame that
    registers directly becomes a hub, and the rest reach the anchor through one
    of those in a single extra hop. Chain depth goes from "up to 59" to "1 or
    2" for most frames.

    Non-destructive: returns a new list, and any frame that cannot be improved
    keeps its original chained homography."""
    by_segment = {}
    for r in results:
        if r.ok and r.homography is not None:
            by_segment.setdefault(r.segment_id, []).append(r.frame_idx)
    out = list(results)
    for segment_id, members in by_segment.items():
        anchor = next((i for i in members if np.allclose(results[i].homography, np.eye(3))),
                      members[0])
        # pass 1: who can reach the anchor directly?
        hubs = {}
        for fi in members:
            if fi == anchor:
                continue
            H_direct, n_inliers, coverage = register_pair_with_coverage(
                frames[anchor], frames[fi], cfg)
            if H_direct is not None and n_inliers >= cfg.min_inliers and coverage >= cfg.min_coverage:
                hubs[fi] = H_direct
                out[fi] = FrameRegistration(fi, H_direct, n_inliers, True, segment_id)
        if progress:
            print(f"  segment {segment_id}: {len(hubs)}/{len(members) - 1} frames reach the "
                  f"anchor directly")
        if max_hops < 2 or not hubs:
            continue
        # pass 2: everyone else goes through the best hub available
        hub_order = sorted(hubs, key=lambda h: abs(h - anchor))
        for fi in members:
            if fi == anchor or fi in hubs:
                continue
            best = None
            for hub in sorted(hub_order, key=lambda h: abs(h - fi))[:max_candidates]:
                H_step, n_inliers, coverage = register_pair_with_coverage(
                    frames[hub], frames[fi], cfg)
                if H_step is None or n_inliers < cfg.min_inliers or coverage < cfg.min_coverage:
                    continue
                if best is None or n_inliers > best[1]:
                    best = (hubs[hub] @ H_step, n_inliers)
            if best is not None:
                out[fi] = FrameRegistration(fi, best[0], best[1], True, segment_id)
    return out


def segment_canvas(frames: list, registrations: list[FrameRegistration],
                    segment_id: int = None, max_pixels: int = 12_000_000,
                    progress: bool = False):
    """Composite every registered frame of one segment into a single wide
    image covering the WHOLE pan, in that segment's reference coordinates.

    Returns `(canvas_bgr, T)` where `T` is the 3x3 that maps segment
    reference coordinates -> canvas pixels, or `(None, None)` if the segment
    has no registered frames. Everything `track_lines` and
    `segment_pan_polygon` return lives in reference coordinates, so pushing
    it through `T` puts it on this image.

    Why this exists: a broadcast camera pans, and the person labelling lines
    told us plainly that their footage will NEVER show all four needed lines
    in one frame. It doesn't have to -- `track_lines` already fuses
    detections from every frame of the pan into one shared coordinate system,
    so a sideline seen only at the start and an end line seen only at the end
    are both in the candidate list. But a human can only label what they can
    SEE, and drawing those candidates on a single frame hides every line that
    fell outside it. This renders the union of what the camera swept, so the
    labelling image shows the same thing the tracker knows about.

    Composited as a COLLAGE (each output pixel painted once, by the first
    frame to cover it) rather than a blend, for the reason
    calibration_bank._grow_and_composite gives: averaging overlapping frames
    smears exactly the thin, low-contrast markings this tool is about. The
    anchor frame is painted first so the familiar reference view stays crisp
    and un-seamed in the middle."""
    by_segment = {}
    for r in registrations:
        if r.ok and r.homography is not None:
            by_segment.setdefault(r.segment_id, []).append(r)
    if not by_segment:
        return None, None
    if segment_id is None:
        segment_id = max(by_segment, key=lambda s: len(by_segment[s]))
    members = by_segment.get(segment_id, [])
    if not members:
        return None, None

    # anchor (identity homography) first -- collage gives priority to whoever
    # paints a pixel first, and the anchor is the frame with zero warp error
    members = sorted(members, key=lambda r: not np.allclose(r.homography, np.eye(3)))

    corners = []
    for reg in members:
        h, w = frames[reg.frame_idx].shape[:2]
        box = np.array([[[0.0, 0.0]], [[w, 0.0]], [[w, h]], [[0.0, h]]], dtype=np.float64)
        corners.append(cv2.perspectiveTransform(box, reg.homography).reshape(-1, 2))
    corners = np.concatenate(corners, axis=0)
    min_xy = corners.min(axis=0)
    max_xy = corners.max(axis=0)

    width = float(max_xy[0] - min_xy[0])
    height = float(max_xy[1] - min_xy[1])
    if width < 1 or height < 1:
        return None, None
    # a long pan can imply an enormous canvas; scale down rather than refuse,
    # since this image exists to be looked at, not measured on
    scale = min(1.0, (max_pixels / (width * height)) ** 0.5)
    T = np.array([[scale, 0.0, -min_xy[0] * scale],
                  [0.0, scale, -min_xy[1] * scale],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    out_w = max(1, int(np.ceil(width * scale)))
    out_h = max(1, int(np.ceil(height * scale)))

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    filled = np.zeros((out_h, out_w), dtype=np.uint8)
    for n, reg in enumerate(members):
        frame = frames[reg.frame_idx]
        H = T @ reg.homography
        warped = cv2.warpPerspective(frame, H, (out_w, out_h))
        ones = np.full(frame.shape[:2], 255, dtype=np.uint8)
        cover = cv2.warpPerspective(ones, H, (out_w, out_h))
        paint = (cover > 0) & (filled == 0)
        canvas[paint] = warped[paint]
        filled[paint] = 255
        if progress:
            print(f"  canvas {n + 1}/{len(members)}: {100.0 * filled.mean() / 255:.1f}% filled")
    return canvas, T
