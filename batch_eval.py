import json
import os
import sys
from edge_refiner import run_bezier_refinement, path_length


def main(path: str) -> None:
    res = run_bezier_refinement(path, output_dir=None)
    short20 = 0
    short40 = 0
    for path_fit in res['fitted_paths']:
        parent_len = path_length(path_fit['original_points'])
        for seg in path_fit['segments']:
            seg_len = path_length(seg['points'])
            if parent_len >= 20 and seg_len <= 3.0:
                short20 += 1
            if parent_len >= 40 and seg_len <= 3.0:
                short40 += 1
    print(json.dumps({
        'file': path,
        'f1': res['metrics']['f1'],
        'precision': res['metrics']['precision'],
        'recall': res['metrics']['recall'],
        'chamfer': res['metrics']['chamfer'],
        'path_count': res['summary']['path_count'],
        'segment_count': res['summary']['segment_count'],
        'mean_segment_error': res['summary']['mean_segment_error'],
        'max_segment_error': res['summary']['max_segment_error'],
        'short20': short20,
        'short40': short40,
    }))


if __name__ == '__main__':
    main(sys.argv[1])
