"""
Cheap heuristic: does this frame plausibly show the field from the elevated
broadcast camera, or is it a ground-level/close-up/crowd/interview cutaway
that can never be usefully calibrated anyway (too little field visible)?

MEASURED (see results/shot_scan_finding.md): plain grass-green HSV fraction
correctly separated confirmed ground-level shots (green_frac 0.03-0.15) from
the confirmed wide elevated shot (0.72-0.76) -- but had at least one false
positive (a near-ground sideline shot with a lot of background tree foliage,
also green, scored 0.50). A texture-based refinement (edge density within
the green mask, on the hypothesis that mowed turf is smoother than foliage)
did NOT cleanly fix that case either -- tree foliage in that specific frame
scored a LOWER edge density than the real turf shots, the opposite of what
was hypothesized.

NOT a precise classifier -- used as a CANDIDATE filter, not a hard gate:
anything it lets through still has to pass actual calibration (enough field
lines found, low reprojection error) to be useful, so a false positive here
just costs one wasted calibration attempt downstream, not a wrong final
result. A false negative (wrongly skipping a real field view) is the more
costly failure mode, so the default threshold is deliberately lenient.
"""
import cv2
import numpy as np


def field_view_score(frame: np.ndarray) -> float:
    """Fraction of the frame that's grass-green (HSV hue 35-85, decent
    saturation). Not field-specific -- also fires on trees/other foliage,
    see module docstring."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85) & (hsv[:, :, 1] > 40)
    return float(mask.mean())


def looks_like_field_view(frame: np.ndarray, threshold: float = 0.35) -> bool:
    return field_view_score(frame) >= threshold
