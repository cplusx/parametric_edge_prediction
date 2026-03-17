import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from misc_utils.config_utils import load_config


TERM_TO_METRIC = {
    'ce': 'val/loss_ce',
    'ctrl': 'val/loss_ctrl',
    'sample': 'val/loss_sample',
    'endpoint': 'val/loss_endpoint',
    'bbox': 'val/loss_bbox',
    'giou': 'val/loss_giou',
    'curve_dist': 'val/loss_curve_dist',
    'extent': 'val/loss_extent',
}

TERM_TO_WEIGHT = {
    'ce': 'ce_weight',
    'ctrl': 'ctrl_weight',
    'sample': 'sample_weight',
    'endpoint': 'endpoint_weight',
    'bbox': 'bbox_weight',
    'giou': 'giou_weight',
    'curve_dist': 'curve_distance_weight',
    'extent': 'extent_weight',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare run metrics with a fair shared main-loss score that excludes extra loss terms.'
    )
    parser.add_argument(
        '--run',
        action='append',
        nargs=3,
        metavar=('LABEL', 'RUN_DIR', 'OVERRIDE_CONFIG'),
        required=True,
        help='Run label, output directory, and override config path.',
    )
    parser.add_argument(
        '--base-config',
        default='configs/parametric_edge/formal_bsds_train_val_test.yaml',
        help='Base config used together with each override config.',
    )
    parser.add_argument(
        '--exclude-term',
        action='append',
        default=[],
        choices=sorted(TERM_TO_METRIC.keys()),
        help='Loss term to exclude from the fair shared score.',
    )
    parser.add_argument(
        '--tail',
        type=int,
        default=5,
        help='How many trailing validation points to average for the final report.',
    )
    return parser.parse_args()


def latest_metrics_csv(run_dir: Path) -> Path:
    csv_root = run_dir / 'csv_logs'
    versions = sorted(csv_root.glob('version_*'))
    if not versions:
        raise FileNotFoundError(f'No csv_logs/version_* found under {run_dir}')
    metrics_path = versions[-1] / 'metrics.csv'
    if not metrics_path.exists():
        raise FileNotFoundError(f'No metrics.csv found at {metrics_path}')
    return metrics_path


def load_val_rows(metrics_path: Path):
    rows = []
    with metrics_path.open('r', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get('val/loss_main'):
                rows.append(row)
    if not rows:
        raise ValueError(f'No validation rows found in {metrics_path}')
    return rows


def shared_score(row, weight_cfg, excluded_terms):
    total = 0.0
    per_term = {}
    for term, metric_key in TERM_TO_METRIC.items():
        if term in excluded_terms:
            continue
        raw = float(row[metric_key])
        weight = float(weight_cfg.get(TERM_TO_WEIGHT[term], 0.0))
        value = raw * weight
        per_term[term] = value
        total += value
    return total, per_term


def summarize_run(label, run_dir, config, tail, excluded_terms):
    metrics_path = latest_metrics_csv(run_dir)
    rows = load_val_rows(metrics_path)
    weight_cfg = config['loss']

    fair_scores = []
    for row in rows:
        score, per_term = shared_score(row, weight_cfg, excluded_terms)
        fair_scores.append((score, row, per_term))

    best_fair_score, best_fair_row, best_fair_terms = min(fair_scores, key=lambda item: item[0])
    best_raw_row = min(rows, key=lambda row: float(row['val/loss_main']))
    tail_rows = rows[-tail:]

    tail_fair_scores = [shared_score(row, weight_cfg, excluded_terms)[0] for row in tail_rows]
    tail_term_means = {}
    for term in TERM_TO_METRIC:
        if term in excluded_terms:
            continue
        values = [shared_score(row, weight_cfg, excluded_terms)[1][term] for row in tail_rows]
        tail_term_means[term] = sum(values) / len(values)

    return {
        'label': label,
        'run_dir': run_dir,
        'metrics_path': metrics_path,
        'excluded_terms': sorted(excluded_terms),
        'best_raw_loss_main': float(best_raw_row['val/loss_main']),
        'best_raw_epoch': int(float(best_raw_row['epoch'])),
        'best_fair_shared_main': best_fair_score,
        'best_fair_epoch': int(float(best_fair_row['epoch'])),
        'best_fair_terms': best_fair_terms,
        'final_raw_loss_main_mean': sum(float(row['val/loss_main']) for row in tail_rows) / len(tail_rows),
        'final_fair_shared_main_mean': sum(tail_fair_scores) / len(tail_fair_scores),
        'final_fair_terms_mean': tail_term_means,
    }


def format_terms(term_values):
    return ', '.join(f'{term}={value:.4f}' for term, value in term_values.items())


def main():
    args = parse_args()
    excluded_terms = set(args.exclude_term)
    summaries = []
    for label, run_dir_str, override_config in args.run:
        run_dir = Path(run_dir_str)
        config = load_config(args.base_config, override_config)
        summaries.append(summarize_run(label, run_dir, config, args.tail, excluded_terms))

    print(f'Excluded fair-score terms: {sorted(excluded_terms)}')
    for summary in summaries:
        print(f'[{summary["label"]}] run_dir={summary["run_dir"]}')
        print(
            f'  best_raw_loss_main={summary["best_raw_loss_main"]:.4f} '
            f'(epoch {summary["best_raw_epoch"]})'
        )
        print(
            f'  best_fair_shared_main={summary["best_fair_shared_main"]:.4f} '
            f'(epoch {summary["best_fair_epoch"]})'
        )
        print(f'  best_fair_terms: {format_terms(summary["best_fair_terms"])}')
        print(f'  final_raw_loss_main_mean(last {args.tail})={summary["final_raw_loss_main_mean"]:.4f}')
        print(f'  final_fair_shared_main_mean(last {args.tail})={summary["final_fair_shared_main_mean"]:.4f}')
        print(f'  final_fair_terms_mean: {format_terms(summary["final_fair_terms_mean"])}')


if __name__ == '__main__':
    main()