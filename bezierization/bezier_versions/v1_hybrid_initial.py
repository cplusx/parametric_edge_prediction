from bezierization.bezier_refiner_core import run_bezier_refinement

VERSION_NAME = 'v1_greedy_hybrid'
DESCRIPTION = 'Hybrid pre-split plus greedy Bezier fitting without trivial-path filtering or post-merge cleanup.'
DEFAULT_CONFIG = {
    'max_degree': 5,
    'mean_error_threshold': 0.75,
    'max_error_threshold': 2.5,
    'max_segment_length': 96.0,
    'angle_threshold_deg': 55.0,
    'min_points': 6,
    'tiny_segment_length': 0.0,
    'min_path_length_for_bezier': 0.0,
    'cleanup_long_path_threshold': 1e9,
    'enable_tiny_cleanup': False,
    'enable_easy_merge': False,
}


def run_refiner(image_path, output_dir=None, **overrides):
    config = DEFAULT_CONFIG.copy()
    config.update(overrides)
    return run_bezier_refinement(image_path=image_path, output_dir=output_dir, **config)
