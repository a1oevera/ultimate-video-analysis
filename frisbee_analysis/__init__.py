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
                         register_sequence, interpolate_missing,
                         refine_to_anchor, segment_canvas)
    from .calibration import (Calibration, fit_calibration, fit_calibration_with_lines,
                              pixel_to_field, field_to_pixel, is_in_field, draw_field_overlay,
                              check_line_reprojection_error, vertices_within_bounds,
                              field_in_front_of_camera, field_covers_points,
                              field_validity_region, calibration_covers_evidence,
                              is_measurable)
    from .field_model import (FieldModel, ModelLine, FIELD_EDGES, EDGE_NAMES,
                              fit_field_model, build_field_model, draw_field_model,
                              VisibleExtent, visible_extent, extent_supports_observation,
                              parallel_edges_meet_outside_field,
                              extent_is_plausible)
    from .watchers import (StaticCluster, WatcherRow, foot_points,
                           collect_static_clusters, find_rows,
                           watchers_outside_field, standoff_from_paint)
    from .calibration_bank import try_auto_calibrate, add_entry as add_bank_entry
    from .shot_filter import looks_like_field_view, field_view_score
    from .field_lines import (SurfaceConfig, LineConfig, ConeConfig,
                              LineSegment, ConeHint, TrackedLine,
                              playing_surface_mask, line_pixel_mask,
                              detect_line_segments, merge_collinear,
                              detect_cone_hints, person_boxes_from_yolo,
                              classify_paint_colour,
                              track_lines, track_cone_hints, segment_pan_polygon,
                              group_by_vanishing_point, direction_hypotheses,
                              confusable_groups, draw_extended, transform_lines,
                              detect_conics, arc_fractions, conic_distance,
                              split_merged_lines, cross_field_candidates,
                              LEGAL_CROSS_POSITIONS_M,
                              sideline_family_candidates,
                              projected_field_is_sane,
                              no_parallel_crossings, PARALLEL_GROUPS,
                              rank_line_labelings,
                              draw_detections)
    from .video_overlays import overlay_boxes_for
except ImportError:
    pass
