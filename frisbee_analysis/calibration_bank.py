"""
Calibration bank: reuse an already-calibrated camera framing instead of
re-clicking lines every time the broadcast cuts to a "new" shot.

MEASURED, corrected from the original single-frame version (see
results/shot_scan_finding.md): matching a new shot against ONE saved frame
at a time under-counts real overlap -- ~30 "distinct" framings turned up in
just 15 real minutes of one broadcast camera's zoom/pan range, because two
different zoom states of the SAME camera often don't share enough features
to register against each other directly, even though a human recognizes
them as the same broadcast angle. Upgraded to matching against a GROWING
COMPOSITE CANVAS per distinct framing family instead: a new shot only needs
to overlap ANY part of what's already been seen in that family, not one
specific previous shot.

Mechanism: reuses the SAME ORB+RANSAC registration already validated for
within-segment frame chaining (mosaic.py's register_pair) -- applied
against each canvas's current accumulated image. A match warps the new
frame into that canvas's coordinate system (the growing-canvas math is the
same bounding-box/translation approach run_mosaic_visualize.py already
uses) and composites it in as a COLLAGE, not a blend -- each canvas pixel is
painted from exactly ONE source frame, never averaged, to avoid the
washout problem measured in results/calibration_finding.md (blending
smears/hides small objects). A canvas's calibration (canvas pixel -> field
cm) is established once by whichever frame first seeded it; every frame
merged in afterward inherits that SAME coordinate system through the
registration homography -- the same way frames within one continuous
segment already share coordinates via mosaic.py's register_sequence, just
extended ACROSS separately-calibrated sessions too.

Bank layout: one subdirectory per CANVAS under `bank_dir`, holding:
  canvas.jpg        -- the accumulated collage image (grows over time)
  mask.png          -- which canvas pixels are filled (0=empty, 255=filled)
  calibration.json  -- ONE calibration for the whole canvas
Backward compatible with the older single-frame bank format (a bare
`reference_frame.jpg` with no mask -- treated as a canvas that just hasn't
grown yet, mask = fully filled).
"""
from __future__ import annotations
import os
import cv2
import numpy as np
from .mosaic import MosaicConfig, register_pair_with_coverage
from .calibration import Calibration

# MEASURED FALSE POSITIVE (person caught it: an auto-match landed on a frame
# 26 minutes away from the canvas it "matched," visibly wrong, and the match
# wobbled between reruns of the identical command -- a sign of a coincidental
# match, not a real one). It cleared cfg.min_inliers (15) with just 17
# inliers and 15% coverage -- that threshold was validated for a DIFFERENT
# job (mosaic.py's within-segment frame-to-frame chaining, where consecutive
# frames share most of their background) and is far too permissive for
# matching across arbitrary, possibly-unrelated moments in the game. Known
# GOOD bank matches in this project measured 1,800-2,500+ inliers and 55-58%
# coverage -- a huge gap above these floors. Grounded in that one real data
# point, not exhaustively tuned; revisit if a genuine match ever gets
# rejected by this gate.
BANK_MIN_INLIERS = 50
BANK_MIN_COVERAGE = 0.25


def _canvas_dirs(bank_dir):
    if not os.path.isdir(bank_dir):
        return []
    return sorted(
        os.path.join(bank_dir, d) for d in os.listdir(bank_dir)
        if os.path.isdir(os.path.join(bank_dir, d))
    )


def _load_canvas(canvas_dir):
    """Returns (image, mask, calib) or None if this isn't a valid canvas
    dir. Backward-compatible with the older bare-frame bank format (no
    mask.png / canvas.jpg -- just reference_frame.jpg)."""
    calib_path = os.path.join(canvas_dir, "calibration.json")
    if not os.path.exists(calib_path):
        return None
    canvas_path = os.path.join(canvas_dir, "canvas.jpg")
    legacy_path = os.path.join(canvas_dir, "reference_frame.jpg")
    image_path = canvas_path if os.path.exists(canvas_path) else legacy_path
    if not os.path.exists(image_path):
        return None
    image = cv2.imread(image_path)
    if image is None:
        return None
    mask_path = os.path.join(canvas_dir, "mask.png")
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    else:
        mask = np.full(image.shape[:2], 255, dtype=np.uint8)
    calib = Calibration.load(calib_path)
    return image, mask, calib


def _save_canvas(canvas_dir, image, mask, calib):
    os.makedirs(canvas_dir, exist_ok=True)
    cv2.imwrite(os.path.join(canvas_dir, "canvas.jpg"), image)
    cv2.imwrite(os.path.join(canvas_dir, "mask.png"), mask)
    calib.save(os.path.join(canvas_dir, "calibration.json"))
    # this canvas may have started life in the old bare-frame format --
    # remove the stale legacy file so _load_canvas doesn't prefer it later
    legacy_path = os.path.join(canvas_dir, "reference_frame.jpg")
    if os.path.exists(legacy_path):
        os.remove(legacy_path)


def try_auto_calibrate(new_ref_frame: np.ndarray, bank_dir: str, mosaic_cfg: MosaicConfig = None):
    """Try to match new_ref_frame against every canvas's CURRENT accumulated
    image in the bank (not a single fixed frame -- the composite grows every
    time add_entry merges a new shot into it). Returns (calibration,
    matched_canvas_name, n_inliers, coverage) for the best match clearing
    min_inliers, or (None, None, 0, 0.0) if nothing matched -- caller should
    fall back to manual calibration in that case.

    `coverage` (0-1): fraction of new_ref_frame's area spanned by this
    match's inlier keypoints -- see mosaic.py's register_pair_with_coverage
    docstring for why this matters (a healthy inlier COUNT doesn't rule out
    the points being clustered in one small region, which risks drift when
    the fitted homography is extrapolated to project something far from
    that cluster). MEASURED CAVEAT: tested directly against a real case
    where a person spotted visible overlay drift -- that specific case had
    GOOD coverage (55.6%) and was NOT explained by this check (the drift
    traced back to the underlying calibration's own reprojection error
    instead, see Calibration.source_max_reprojection_px). Coverage is still
    a real, legitimate signal for a genuinely different failure mode (a
    poorly-distributed bank match) -- just don't expect it alone to catch
    every source of drift."""
    mosaic_cfg = mosaic_cfg or MosaicConfig()
    best = (None, None, 0, 0.0)
    for canvas_dir in _canvas_dirs(bank_dir):
        loaded = _load_canvas(canvas_dir)
        if loaded is None:
            continue
        canvas_image, _, canvas_calib = loaded
        # H_step maps new_ref_frame's pixels -> canvas's coords (register_pair's
        # convention: homography maps the SECOND arg's coords -> the FIRST arg's)
        H_step, n_inliers, coverage = register_pair_with_coverage(canvas_image, new_ref_frame, mosaic_cfg)
        if H_step is None or n_inliers < BANK_MIN_INLIERS or coverage < BANK_MIN_COVERAGE or n_inliers <= best[2]:
            continue
        new_H = canvas_calib.H() @ H_step
        new_calib = Calibration(homography=new_H.tolist(), keypoint_labels=canvas_calib.keypoint_labels,
                                 image_points=[], field_points=[],
                                 image_path=f"auto-matched via bank canvas {os.path.basename(canvas_dir)}",
                                 source_frame_indices=None, line_details=None,
                                 source_max_reprojection_px=canvas_calib.source_max_reprojection_px)
        best = (new_calib, os.path.basename(canvas_dir), n_inliers, coverage)
    return best


def _grow_and_composite(canvas_image, canvas_mask, canvas_calib: Calibration,
                         new_frame: np.ndarray, H_step: np.ndarray):
    """H_step maps new_frame's pixels -> the CURRENT canvas's coordinate
    system (register_pair's convention). Grows the canvas if new_frame's
    warped extent falls outside current bounds, then composites new_frame in
    as a COLLAGE -- paints only currently-EMPTY canvas pixels, never
    overwrites/blends existing ones, so merging overlapping content can't
    wash out or smear anything already there. Returns (image, mask, calib)
    for the grown, composited canvas."""
    ch, cw = canvas_image.shape[:2]
    nh, nw = new_frame.shape[:2]
    corners = np.float32([[0, 0], [nw, 0], [nw, nh], [0, nh]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H_step).reshape(-1, 2)
    all_x = np.concatenate([warped_corners[:, 0], [0, cw]])
    all_y = np.concatenate([warped_corners[:, 1], [0, ch]])
    min_x, min_y = float(np.floor(all_x.min())), float(np.floor(all_y.min()))
    max_x, max_y = float(np.ceil(all_x.max())), float(np.ceil(all_y.max()))
    new_cw, new_ch = int(max_x - min_x), int(max_y - min_y)

    # T maps OLD canvas coords -> GROWN canvas coords (pure translation, so
    # it's cheaply invertible -- needed below to re-express the calibration,
    # which was defined in OLD canvas coords, in the new GROWN coords)
    T = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64)
    ox, oy = int(-min_x), int(-min_y)

    grown_image = np.zeros((new_ch, new_cw, 3), dtype=np.uint8)
    grown_mask = np.zeros((new_ch, new_cw), dtype=np.uint8)
    grown_image[oy:oy + ch, ox:ox + cw] = canvas_image
    grown_mask[oy:oy + ch, ox:ox + cw] = canvas_mask

    H_new_to_grown = T @ H_step
    warped_new = cv2.warpPerspective(new_frame, H_new_to_grown, (new_cw, new_ch))
    warped_new_mask = cv2.warpPerspective(
        np.full((nh, nw), 255, dtype=np.uint8), H_new_to_grown, (new_cw, new_ch))

    paint = (grown_mask == 0) & (warped_new_mask > 0)  # collage: only fill previously-empty pixels
    grown_image[paint] = warped_new[paint]
    grown_mask[paint] = 255

    # new_calib(grown_pt) must still equal old_calib(old_pt) for the same
    # real-world point; grown_pt = T @ old_pt, so old_pt = inv(T) @ grown_pt
    new_calib_H = canvas_calib.H() @ np.linalg.inv(T)
    new_calib = Calibration(homography=new_calib_H.tolist(), keypoint_labels=canvas_calib.keypoint_labels,
                             image_points=[], field_points=[], image_path=canvas_calib.image_path,
                             source_frame_indices=None, line_details=None,
                             source_max_reprojection_px=canvas_calib.source_max_reprojection_px)
    return grown_image, grown_mask, new_calib


def add_entry(bank_dir: str, ref_frame: np.ndarray, calib: Calibration,
              name: str = None, mosaic_cfg: MosaicConfig = None) -> str:
    """Merge ref_frame into the bank: if it matches an existing canvas,
    grow+composite it into that canvas (so future matching against that
    canvas benefits from the wider accumulated coverage); otherwise seed a
    brand-new canvas from it. `name` only controls the folder name when a
    NEW canvas is created -- it has no effect when merging into an existing
    one (that canvas keeps its own existing name)."""
    mosaic_cfg = mosaic_cfg or MosaicConfig()
    os.makedirs(bank_dir, exist_ok=True)

    best_dir, best_H, best_n = None, None, 0
    for canvas_dir in _canvas_dirs(bank_dir):
        loaded = _load_canvas(canvas_dir)
        if loaded is None:
            continue
        canvas_image, canvas_mask, canvas_calib = loaded
        # same stricter bank-specific gate as try_auto_calibrate (see
        # BANK_MIN_INLIERS/BANK_MIN_COVERAGE above) -- a weak/coincidental
        # match merged into a canvas would corrupt it for every FUTURE match
        # too, not just this one, so this loop needs to be at least as
        # strict as the read-only matching path, not looser.
        H_step, n_inliers, coverage = register_pair_with_coverage(canvas_image, ref_frame, mosaic_cfg)
        if (H_step is not None and n_inliers >= BANK_MIN_INLIERS
                and coverage >= BANK_MIN_COVERAGE and n_inliers > best_n):
            best_dir, best_H, best_n = canvas_dir, H_step, n_inliers

    if best_dir is not None:
        canvas_image, canvas_mask, canvas_calib = _load_canvas(best_dir)

        # MEASURED BUG (person redid a manual calibration specifically to
        # fix a bad line, but the bank kept serving the OLD, worse one
        # afterward): merging into an existing canvas always kept THAT
        # canvas's calibration as authoritative, regardless of which one --
        # old canvas or new incoming frame -- was actually more accurate.
        # Prefer whichever has the LOWER known source_max_reprojection_px;
        # only keep the existing canvas's when the incoming one has no known
        # error to compare against (don't downgrade a verified canvas for
        # an unverified one).
        incoming_better = (
            calib.source_max_reprojection_px is not None and
            (canvas_calib.source_max_reprojection_px is None or
             calib.source_max_reprojection_px < canvas_calib.source_max_reprojection_px)
        )
        if incoming_better:
            # re-base the canvas on the NEW (better) calibration: warp the
            # OLD canvas's accumulated content INTO ref_frame's coordinate
            # system instead -- symmetric to the normal path below, just
            # with the roles of "base" and "merged in" swapped.
            H_reverse, n_inliers_reverse, coverage_reverse = register_pair_with_coverage(
                ref_frame, canvas_image, mosaic_cfg)
            if (H_reverse is not None and n_inliers_reverse >= BANK_MIN_INLIERS
                    and coverage_reverse >= BANK_MIN_COVERAGE):
                base_mask = np.full(ref_frame.shape[:2], 255, dtype=np.uint8)
                grown_image, grown_mask, grown_calib = _grow_and_composite(
                    ref_frame, base_mask, calib, canvas_image, H_reverse)
                _save_canvas(best_dir, grown_image, grown_mask, grown_calib)
                return best_dir
            # reverse registration failed for some reason -- fall through to
            # the normal path rather than losing this frame's contribution

        grown_image, grown_mask, grown_calib = _grow_and_composite(
            canvas_image, canvas_mask, canvas_calib, ref_frame, best_H)
        _save_canvas(best_dir, grown_image, grown_mask, grown_calib)
        return best_dir

    if name is None:
        existing = [d for d in os.listdir(bank_dir) if d.startswith("canvas_")]
        n = len(existing)
        name = f"canvas_{n:03d}"
        while os.path.isdir(os.path.join(bank_dir, name)):
            n += 1
            name = f"canvas_{n:03d}"
    canvas_dir = os.path.join(bank_dir, name)
    mask = np.full(ref_frame.shape[:2], 255, dtype=np.uint8)
    _save_canvas(canvas_dir, ref_frame, mask, calib)
    return canvas_dir
