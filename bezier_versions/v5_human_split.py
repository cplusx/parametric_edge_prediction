from bezier_refiner_core import run_bezier_refinement

VERSION_NAME = 'v5_anchor_consistent'
DESCRIPTION = 'Natural-anchor-aware global chunk segmentation with smoother, more consistent split placement.'
DEFAULT_CONFIG = {
    'max_degree': 5,
    'mean_error_threshold': 0.6,
    'max_error_threshold': 2.0,
    'max_segment_length': 88.0,
    'angle_threshold_deg': 50.0,
    'min_points': 6,
    'tiny_segment_length': 3.0,
    'min_path_length_for_bezier': 6.0,
    'cleanup_long_path_threshold': 20.0,
    'enable_tiny_cleanup': True,
    'enable_easy_merge': True,
    'use_global_chunk_dp': True,
    'split_anchor_weight': 0.8,
    'split_extrema_window': 5,
    'prefer_anchor_for_length_split': True,
    'length_split_lookahead': 48.0,
    'length_split_min_strength': 0.62,
    'enable_smooth_consistency': True,
    'smooth_consistency_min_path_length': 120.0,
    'smooth_consistency_q90_turn_threshold': 42.0,
    'smooth_consistency_max_strong_anchors': 4,
    'smooth_consistency_strong_anchor_threshold': 0.82,
    'smooth_consistency_target_length_factor': 1.0,
    'smooth_consistency_snap_window': 40.0,
    'smooth_consistency_snap_anchor_strength': 0.6,
    'enable_bundle_consistency': True,
    'bundle_consistency_min_path_length': 100.0,
    'bundle_consistency_max_length_ratio': 1.45,
    'bundle_consistency_descriptor_threshold': 0.09,
    'bundle_consistency_snap_window': 28.0,
    'bundle_consistency_snap_anchor_strength': 0.58,
}


def run_refiner(image_path, output_dir=None, **overrides):
    config = DEFAULT_CONFIG.copy()
    config.update(overrides)
    return run_bezier_refinement(image_path=image_path, output_dir=output_dir, **config)
