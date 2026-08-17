"""Frisbee video analysis -- Track A (possession, coordinate-space, numpy/scipy
only) + Track B (CV pipeline, needs opencv/torch -- see requirements.txt)."""
from .schema import Track, FieldConfig, UFA_FIELD, WFDF_FIELD
from .ufatrack_loader import load_ufatrack, load_raw_rows
from .features import compute_features, FeatureConfig
from .possession import viterbi_decode, naive_argmax_baseline, HMMConfig
from .evaluate import (evaluate_sequence, derive_events, frame_accuracy,
                       transition_metrics, possession_segments,
                       transition_frames)
from .ultimate_config import UltimatePitchConfiguration

try:
    # Track B: needs opencv-python (see requirements.txt). Kept optional so
    # Track A (numpy/scipy only) still imports cleanly without it.
    from .mosaic import (MosaicConfig, FrameRegistration, register_pair,
                         register_sequence, interpolate_missing)
    from .calibration import (Calibration, fit_calibration, pixel_to_field,
                              field_to_pixel, is_in_field, draw_field_overlay)
except ImportError:
    pass
