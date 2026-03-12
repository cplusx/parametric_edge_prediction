import argparse

from bezierization.bezier_refiner_core import path_length
from bezierization.bezier_versions.v5_human_split import DEFAULT_CONFIG, run_refiner


run_bezier_refinement = run_refiner


def main():
    parser = argparse.ArgumentParser(description='Current default Bezier refiner entrypoint.')
    parser.add_argument('--input', type=str, help='Path to a binary edge map.')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory for debug visualizations.')
    parser.add_argument('--max-degree', type=int, default=DEFAULT_CONFIG['max_degree'])
    parser.add_argument('--mean-error-threshold', type=float, default=DEFAULT_CONFIG['mean_error_threshold'])
    parser.add_argument('--max-error-threshold', type=float, default=DEFAULT_CONFIG['max_error_threshold'])
    parser.add_argument('--max-segment-length', type=float, default=DEFAULT_CONFIG['max_segment_length'])
    parser.add_argument('--angle-threshold', type=float, default=DEFAULT_CONFIG['angle_threshold_deg'])
    parser.add_argument('--min-points', type=int, default=DEFAULT_CONFIG['min_points'])
    parser.add_argument('--tiny-segment-length', type=float, default=DEFAULT_CONFIG['tiny_segment_length'])
    parser.add_argument('--min-path-length', type=float, default=DEFAULT_CONFIG['min_path_length_for_bezier'])
    args = parser.parse_args()

    if not args.input:
        raise SystemExit('Please provide --input path/to/binary_edge_map.png')

    result = run_refiner(
        image_path=args.input,
        output_dir=args.output_dir,
        max_degree=args.max_degree,
        mean_error_threshold=args.mean_error_threshold,
        max_error_threshold=args.max_error_threshold,
        max_segment_length=args.max_segment_length,
        angle_threshold_deg=args.angle_threshold,
        min_points=args.min_points,
        tiny_segment_length=args.tiny_segment_length,
        min_path_length_for_bezier=args.min_path_length,
    )

    print('paths', result['summary']['path_count'])
    print('segments', result['summary']['segment_count'])
    print('dropped_paths', len(result['dropped_paths']))
    print('degree_histogram', result['summary']['degree_histogram'])
    print('mean_segment_error', f"{result['summary']['mean_segment_error']:.4f}")
    print('max_segment_error', f"{result['summary']['max_segment_error']:.4f}")
    print('precision', f"{result['metrics']['precision']:.4f}")
    print('recall', f"{result['metrics']['recall']:.4f}")
    print('f1', f"{result['metrics']['f1']:.4f}")
    print('chamfer', f"{result['metrics']['chamfer']:.4f}")
    if 'overlay_path' in result:
        print('overlay_path', result['overlay_path'])
    if 'colored_overlay_path' in result:
        print('colored_overlay_path', result['colored_overlay_path'])
    if 'fitted_path' in result:
        print('fitted_path', result['fitted_path'])


if __name__ == '__main__':
    main()
