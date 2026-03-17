import argparse
import cProfile
import importlib.util
import io
import pstats
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def profile_extract_graph_segments(edge_path: Path, version_name: str, top_k: int) -> str:
    bezier_target_utils = _load_module('bezier_target_utils', 'misc_utils/bezier_target_utils.py')

    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    result = bezier_target_utils.extract_graph_segments(edge_path, version_name=version_name)
    profiler.disable()
    elapsed = time.perf_counter() - start

    polylines = bezier_target_utils.unpack_polylines(result['graph_points'], result['graph_offsets'])

    buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer).sort_stats('cumulative')
    stats.print_stats(top_k)

    summary = [
        f'=== {edge_path.name} ===',
        f'elapsed_sec: {elapsed:.3f}',
        f'num_polylines: {len(polylines)}',
        '',
        buffer.getvalue(),
    ]
    return '\n'.join(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description='Profile bezier graph-cache generation on one or more edge maps.')
    parser.add_argument(
        '--images',
        nargs='+',
        required=True,
        help='One or more edge-map paths to profile.',
    )
    parser.add_argument(
        '--version',
        default='v5_anchor_consistent',
        help='Bezierization version name to profile.',
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='How many cProfile rows to print.',
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Optional path to save the full profiling report.',
    )
    args = parser.parse_args()

    reports = []
    for image_path in args.images:
        reports.append(
            profile_extract_graph_segments(
                edge_path=Path(image_path),
                version_name=str(args.version),
                top_k=int(args.top_k),
            )
        )

    report_text = '\n\n'.join(reports)
    print(report_text)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text, encoding='utf-8')


if __name__ == '__main__':
    main()
