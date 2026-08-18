"""
Calibration bank: reuse an already-calibrated camera framing instead of
re-clicking lines every time the broadcast cuts to a "new" shot.

MOTIVATING MEASUREMENT (person's own observation): this footage cuts/pans
often enough that a full game could need dozens of separate manual
calibration sessions if every camera segment needs its own. But the camera
is a FIXED mount (pure rotation/zoom, no translation -- see CONTEXT.md's
footage constraints), so the broadcast almost certainly reuses a small set
of recurring framings (the standard wide shot, a close-up near each
endzone) far more than it invents brand new ones. Worth checking whether a
"new" segment's reference frame is actually one you've already calibrated
before asking for more manual clicks.

Mechanism: reuses the SAME ORB+RANSAC registration already validated for
within-segment frame chaining (mosaic.py's register_pair) -- just applied
BETWEEN segments' reference frames instead of within one segment. If a new
segment's reference frame registers against a bank entry's reference frame
with enough inliers, the calibration composes for free:
  new_segment_pixels -> (bank entry's own homography) -> field cm
via new_calib.H = bank_entry_calib.H() @ H_step, where H_step is
register_pair's new-ref-frame -> bank-entry-ref-frame homography. No new
clicking needed. Falls back to None (caller does manual calibration) if no
bank entry matches -- and per calibration_finding.md's wrong-field warning,
a spurious match is still possible in principle (two coincidentally similar
framings); the caller should sanity-check the resulting overlay same as any
other calibration, not skip that step just because it was automatic.

Bank layout: one subdirectory per entry under `bank_dir`, each holding
`reference_frame.jpg` (the raw frame image, needed for future matching) and
`calibration.json` (a Calibration, same format run_calibrate.py saves).
"""
from __future__ import annotations
import os
import json
import cv2
import numpy as np
from .mosaic import MosaicConfig, register_pair
from .calibration import Calibration


def _entry_dirs(bank_dir):
    if not os.path.isdir(bank_dir):
        return []
    return sorted(
        os.path.join(bank_dir, d) for d in os.listdir(bank_dir)
        if os.path.isdir(os.path.join(bank_dir, d))
    )


def try_auto_calibrate(new_ref_frame: np.ndarray, bank_dir: str,
                        mosaic_cfg: MosaicConfig = None) -> tuple[Calibration | None, str | None, int]:
    """Try to match new_ref_frame against every entry in the bank. Returns
    (calibration, matched_entry_name, n_inliers) for the BEST match clearing
    min_inliers, or (None, None, 0) if nothing matched -- caller should fall
    back to manual calibration in that case."""
    mosaic_cfg = mosaic_cfg or MosaicConfig()
    best = (None, None, 0)
    for entry_dir in _entry_dirs(bank_dir):
        frame_path = os.path.join(entry_dir, "reference_frame.jpg")
        calib_path = os.path.join(entry_dir, "calibration.json")
        if not (os.path.exists(frame_path) and os.path.exists(calib_path)):
            continue
        bank_frame = cv2.imread(frame_path)
        if bank_frame is None:
            continue
        # H_step maps new_ref_frame's pixels -> bank_frame's pixels (register_pair's
        # convention: homography maps the SECOND arg's coords -> the FIRST arg's)
        H_step, n_inliers = register_pair(bank_frame, new_ref_frame, mosaic_cfg)
        if H_step is None or n_inliers < mosaic_cfg.min_inliers or n_inliers <= best[2]:
            continue
        bank_calib = Calibration.load(calib_path)
        new_H = bank_calib.H() @ H_step
        new_calib = Calibration(homography=new_H.tolist(), keypoint_labels=bank_calib.keypoint_labels,
                                 image_points=[], field_points=[],
                                 image_path=f"auto-matched via bank entry {os.path.basename(entry_dir)}",
                                 source_frame_indices=None, line_details=None)
        best = (new_calib, os.path.basename(entry_dir), n_inliers)
    return best


def add_entry(bank_dir: str, ref_frame: np.ndarray, calib: Calibration, name: str = None) -> str:
    """Save a reference frame + its calibration into the bank so future
    segments can auto-match against it. `name` defaults to a timestamp-free
    incrementing id."""
    os.makedirs(bank_dir, exist_ok=True)
    if name is None:
        existing = [d for d in os.listdir(bank_dir) if d.startswith("entry_")]
        n = len(existing)
        name = f"entry_{n:03d}"
        while os.path.isdir(os.path.join(bank_dir, name)):
            n += 1
            name = f"entry_{n:03d}"
    entry_dir = os.path.join(bank_dir, name)
    os.makedirs(entry_dir, exist_ok=True)
    cv2.imwrite(os.path.join(entry_dir, "reference_frame.jpg"), ref_frame)
    calib.save(os.path.join(entry_dir, "calibration.json"))
    return entry_dir
