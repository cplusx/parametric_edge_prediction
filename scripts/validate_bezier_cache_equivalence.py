import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_extract_graph_segments():
    module = importlib.import_module('misc_utils.bezier_target_utils')
    return module.extract_graph_segments, module.unpack_polylines


def _summary(result, unpack_polylines):
    polylines = unpack_polylines(result['graph_points'], result['graph_offsets'])
    return {
        'num_polylines': len(polylines),
        'offsets': result['graph_offsets'].tolist(),
        'points_sha': str(np.asarray(result['graph_points'], dtype=np.float32).round(6).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate fast/reference bezier cache generation equivalence.')
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--version', default='v5_anchor_consistent')
    args = parser.parse_args()

    report = {'version': args.version, 'images': []}
    for image_path in args.images:
        edge_path = Path(image_path)

        os.environ['BEZIER_USE_FAST_NUMBA'] = '0'
        importlib.invalidate_caches()
        ref_mod = importlib.import_module('misc_utils.bezier_target_utils')
        ref_mod = importlib.reload(ref_mod)
        ref_result = ref_mod.extract_graph_segments(edge_path, version_name=args.version)
        ref_summary = _summary(ref_result, ref_mod.unpack_polylines)

        os.environ['BEZIER_USE_FAST_NUMBA'] = '1'
        importlib.invalidate_caches()
        fast_backend = importlib.import_module('bezierization.fast_backend')
        importlib.reload(fast_backend)
        core_mod = importlib.import_module('bezierization.bezier_refiner_core')
        importlib.reload(core_mod)
        fast_mod = importlib.import_module('misc_utils.bezier_target_utils')
        fast_mod = importlib.reload(fast_mod)
        fast_result = fast_mod.extract_graph_segments(edge_path, version_name=args.version)
        fast_summary = _summary(fast_result, fast_mod.unpack_polylines)

        same_points = np.array_equal(ref_result['graph_points'], fast_result['graph_points'])
        same_offsets = np.array_equal(ref_result['graph_offsets'], fast_result['graph_offsets'])
        report['images'].append({
            'image': edge_path.name,
            'same_points': bool(same_points),
            'same_offsets': bool(same_offsets),
            'reference': ref_summary,
            'fast': fast_summary,
        })

    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
