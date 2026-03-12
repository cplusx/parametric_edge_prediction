import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from glob import glob

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from bezierization.ablation_api import run_version
from bezierization.bezierize_ablation.experiments import EXPERIMENTS
from bezierization.bezierize_ablation.metrics_utils import aggregate_experiment_rows, summarize_path_records
from bezierization.bezier_refiner_core import path_length


def choose_files(data_glob, sample_size=None, seed=0):
    files = sorted(glob(data_glob))
    if sample_size is None or sample_size >= len(files):
        return files
    rng = random.Random(seed)
    files = files[:]
    rng.shuffle(files)
    return sorted(files[:sample_size])


def process_one(experiment, image_path, output_root=None):
    exp_name = experiment['name']
    output_dir = None
    if output_root is not None:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join(output_root, exp_name, stem)
    result = run_version(experiment['version'], image_path=image_path, output_dir=output_dir, **experiment['overrides'])

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
        'experiment': exp_name,
        'version': experiment['version'],
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
        'bucket_summary': summarize_path_records(result['fitted_paths'], result['dropped_paths']),
    }


def build_report(manifest, summary):
    experiment_meta = {exp['name']: exp for exp in manifest['experiments']}
    lines = []
    lines.append('# Bezierize Ablation Report')
    lines.append('')
    lines.append(f"- Timestamp: {manifest['timestamp']}")
    lines.append(f"- Dataset glob: `{manifest['data_glob']}`")
    lines.append(f"- Images evaluated: {manifest['num_images']}")
    lines.append(f"- Sample mode: `{manifest['sample_mode']}`")
    lines.append('')
    lines.append('## Experiments')
    lines.append('')
    for experiment in manifest['experiments']:
        lines.append(f"### {experiment['name']}")
        lines.append(f"- version: `{experiment['version']}`")
        lines.append(f"- kind: `{experiment['kind']}`")
        lines.append(f"- notes: {experiment['notes']}")
        if experiment['overrides']:
            lines.append(f"- overrides: `{json.dumps(experiment['overrides'], sort_keys=True)}`")
        lines.append('')
    lines.append('## Experiment Ranking')
    lines.append('')
    ranked = sorted(summary.items(), key=lambda kv: (-kv[1]['mean_f1'], kv[1]['total_short20'], kv[1]['mean_chamfer']))
    for name, stats in ranked:
        meta = experiment_meta.get(name, {})
        lines.append(f"### {name}")
        if meta:
            lines.append(f"- version: `{meta['version']}`")
            lines.append(f"- notes: {meta['notes']}")
        lines.append(f"- mean_f1: {stats['mean_f1']:.4f}")
        lines.append(f"- mean_chamfer: {stats['mean_chamfer']:.4f}")
        lines.append(f"- short20_cases: {stats['short20_cases']}")
        lines.append(f"- short40_cases: {stats['short40_cases']}")
        lines.append(f"- total_short20: {stats['total_short20']}")
        lines.append(f"- total_short40: {stats['total_short40']}")
        lines.append(f"- mean_segment_count: {stats['mean_segment_count']:.2f}")
        lines.append(f"- mean_dropped_paths: {stats['mean_dropped_paths']:.2f}")
        lines.append('')

    lines.append('## Bucketed Summary')
    lines.append('')
    for name, stats in ranked:
        meta = experiment_meta.get(name, {})
        lines.append(f"### {name}")
        if meta:
            lines.append(f"- version: `{meta['version']}`")
        for bucket_name in sorted(stats['bucket_summary'].keys()):
            bucket = stats['bucket_summary'][bucket_name]
            lines.append(
                f"- {bucket_name}: total_paths={bucket['total_paths']}, dropped={bucket['dropped_paths']}, "
                f"drop_rate={bucket['drop_rate']:.3f}, mean_segments_per_fitted_path={bucket['mean_segments_per_fitted_path']:.3f}, "
                f"short_output_segments={bucket['short_output_segments']}"
            )
        lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Run reusable Bezier ablations.')
    parser.add_argument('--data-glob', default='gt_rgb/test/*.png')
    parser.add_argument('--sample-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--with-visuals', action='store_true')
    parser.add_argument('--output-root', default='bezierize_ablation')
    args = parser.parse_args()

    files = choose_files(args.data_glob, sample_size=args.sample_size, seed=args.seed)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_root = os.path.join(args.output_root, f'run_{timestamp}')
    os.makedirs(run_root, exist_ok=True)
    visuals_root = os.path.join(run_root, 'visuals') if args.with_visuals else None
    if visuals_root is not None:
        os.makedirs(visuals_root, exist_ok=True)

    manifest = {
        'timestamp': timestamp,
        'data_glob': args.data_glob,
        'num_images': len(files),
        'sample_mode': 'full' if args.sample_size is None else f'sample_{len(files)}_seed_{args.seed}',
        'experiments': EXPERIMENTS,
    }
    with open(os.path.join(run_root, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    rows = []
    jobs = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for experiment in EXPERIMENTS:
            for image_path in files:
                jobs.append(pool.submit(process_one, experiment, image_path, visuals_root))
        total = len(jobs)
        for idx, future in enumerate(as_completed(jobs), 1):
            rows.append(future.result())
            if idx % 100 == 0 or idx == total:
                print(f'completed {idx}/{total}', flush=True)

    per_experiment = {}
    for experiment in EXPERIMENTS:
        exp_rows = [row for row in rows if row['experiment'] == experiment['name']]
        per_experiment[experiment['name']] = aggregate_experiment_rows(exp_rows)

    with open(os.path.join(run_root, 'per_image_results.jsonl'), 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')

    with open(os.path.join(run_root, 'summary.json'), 'w') as f:
        json.dump(per_experiment, f, indent=2)

    report = build_report(manifest, per_experiment)
    with open(os.path.join(run_root, 'report.md'), 'w') as f:
        f.write(report)

    print(run_root)


if __name__ == '__main__':
    main()
