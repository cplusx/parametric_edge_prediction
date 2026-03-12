import argparse
import glob
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ablation_api import run_version
from bezier_refiner_core import path_length


def process_one(image_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = run_version('v5_anchor_consistent', image_path=image_path, output_dir=None)

    short20 = 0
    short40 = 0
    for path_fit in result['fitted_paths']:
        parent_len = path_length(path_fit['original_points'])
        for segment in path_fit['segments']:
            seg_len = path_length(segment['points'])
            if parent_len >= 20 and seg_len <= 3.0:
                short20 += 1
            if parent_len >= 40 and seg_len <= 3.0:
                short40 += 1

    return {
        'file': image_path,
        'f1': result['metrics']['f1'],
        'precision': result['metrics']['precision'],
        'recall': result['metrics']['recall'],
        'chamfer': result['metrics']['chamfer'],
        'path_count': result['summary']['path_count'],
        'segment_count': result['summary']['segment_count'],
        'dropped_paths': len(result['dropped_paths']),
        'mean_segment_error': result['summary']['mean_segment_error'],
        'max_segment_error': result['summary']['max_segment_error'],
        'short20': short20,
        'short40': short40,
        'warnings': [str(item.message) for item in caught],
    }


def build_summary(rows):
    worst_f1 = sorted(rows, key=lambda row: (row['f1'], -row['chamfer']))[:10]
    worst_chamfer = sorted(rows, key=lambda row: (-row['chamfer'], row['f1']))[:10]
    warning_rows = [row for row in rows if row['warnings']][:20]
    return {
        'images': len(rows),
        'mean_f1': sum(row['f1'] for row in rows) / len(rows),
        'mean_chamfer': sum(row['chamfer'] for row in rows) / len(rows),
        'mean_segment_count': sum(row['segment_count'] for row in rows) / len(rows),
        'mean_dropped_paths': sum(row['dropped_paths'] for row in rows) / len(rows),
        'short20_cases': sum(1 for row in rows if row['short20'] > 0),
        'short40_cases': sum(1 for row in rows if row['short40'] > 0),
        'total_short20': sum(row['short20'] for row in rows),
        'total_short40': sum(row['short40'] for row in rows),
        'warning_cases': sum(1 for row in rows if row['warnings']),
        'total_warnings': sum(len(row['warnings']) for row in rows),
        'worst_f1': worst_f1,
        'worst_chamfer': worst_chamfer,
        'warning_rows': warning_rows,
    }


def main():
    parser = argparse.ArgumentParser(description='Run full-dataset v5 evaluation.')
    parser.add_argument('--data-glob', default='gt_rgb/test/*.png')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output-root', default='bezierize_ablation')
    args = parser.parse_args()

    files = sorted(glob.glob(args.data_glob))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_root = os.path.join(args.output_root, f'run_full_v5_{timestamp}')
    os.makedirs(run_root, exist_ok=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_one, image_path) for image_path in files]
        total = len(futures)
        for idx, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if idx % 100 == 0 or idx == total:
                print(f'completed {idx}/{total}', flush=True)

    rows.sort(key=lambda row: row['file'])
    summary = build_summary(rows)

    with open(os.path.join(run_root, 'per_image_results.jsonl'), 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')

    with open(os.path.join(run_root, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(run_root, 'report.md'), 'w') as f:
        f.write('# Full V5 Evaluation\n\n')
        f.write(json.dumps(summary, indent=2))
        f.write('\n')

    print(run_root)


if __name__ == '__main__':
    main()
