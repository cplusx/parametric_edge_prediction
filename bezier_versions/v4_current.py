from bezier_refiner_core import run_bezier_refinement

VERSION_NAME = 'v4_adjacent_merge'
DESCRIPTION = 'Trivial-path filtering, tiny-segment cleanup, and easy adjacent-curve merging.'
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
}


def run_refiner(image_path, output_dir=None, **overrides):
    config = DEFAULT_CONFIG.copy()
    config.update(overrides)
    return run_bezier_refinement(image_path=image_path, output_dir=output_dir, **config)
