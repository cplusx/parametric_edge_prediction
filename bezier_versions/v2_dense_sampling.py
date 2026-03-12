from bezier_refiner_core import run_bezier_refinement

VERSION_NAME = 'v2_tiny_cleanup'
DESCRIPTION = 'Adds dense rasterization and tiny-segment cleanup, but still keeps trivial original paths.'
DEFAULT_CONFIG = {
    'max_degree': 5,
    'mean_error_threshold': 0.6,
    'max_error_threshold': 2.0,
    'max_segment_length': 88.0,
    'angle_threshold_deg': 50.0,
    'min_points': 6,
    'tiny_segment_length': 3.0,
    'min_path_length_for_bezier': 0.0,
    'cleanup_long_path_threshold': 40.0,
    'enable_tiny_cleanup': True,
    'enable_easy_merge': False,
}


def run_refiner(image_path, output_dir=None, **overrides):
    config = DEFAULT_CONFIG.copy()
    config.update(overrides)
    return run_bezier_refinement(image_path=image_path, output_dir=output_dir, **config)
