"""Frisbee possession analysis -- coordinate-space core (Track A)."""
from .schema import Track, FieldConfig, UFA_FIELD, WFDF_FIELD
from .ufatrack_loader import load_ufatrack, load_raw_rows
from .features import compute_features, FeatureConfig
from .possession import viterbi_decode, naive_argmax_baseline, HMMConfig
from .evaluate import (evaluate_sequence, derive_events, frame_accuracy,
                       transition_metrics, possession_segments,
                       transition_frames)
from .ultimate_config import UltimatePitchConfiguration
