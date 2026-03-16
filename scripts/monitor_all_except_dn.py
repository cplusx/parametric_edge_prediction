import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


EXPERIMENT_ROOT = Path('outputs/parametric_edge_training/overfit_diverse16_2000_all_except_dn')
BASELINE_REPORT = Path('outputs/parametric_edge_training/current_sweep_analysis/addback_ablation_report.md')
STATUS_MD = Path('outputs/parametric_edge_training/current_sweep_analysis/all_except_dn_status.md')
STATUS_JSON = Path('outputs/parametric_edge_training/current_sweep_analysis/all_except_dn_status.json')
RESULT_MARKER = '## All Except DN Result'
BASELINE_MEMORIZATION = 0.0142


def _version_index(path: Path) -> int:
    try:
        return int(path.parent.name.split('_')[-1])
    except (ValueError, IndexError):
        return -1


def _latest_metrics_file(root_dir: Path) -> Optional[Path]:
    candidates = sorted((root_dir / 'csv_logs').glob('version_*/metrics.csv'), key=_version_index)
    return candidates[-1] if candidates else None


def _read_max_epochs(metrics_path: Optional[Path]) -> Optional[int]:
    if metrics_path is None:
        return None
    hparams_path = metrics_path.with_name('hparams.yaml')
    if not hparams_path.exists():
        return None
    for line in hparams_path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if stripped.startswith('max_epochs:'):
            return int(stripped.split(':', 1)[1].strip())
    return None


def _read_metrics(metrics_path: Optional[Path]) -> Dict:
    summary = {
        'metrics_path': None if metrics_path is None else str(metrics_path),
        'best_val_loss': None,
        'best_epoch': None,
        'final_val_loss': None,
        'final_epoch': None,
        'val_rows': 0,
        'train_rows': 0,
    }
    if metrics_path is None or not metrics_path.exists():
        return summary

    with metrics_path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get('train/loss') not in {'', None}:
                summary['train_rows'] += 1
            val_loss_raw = row.get('val_loss') or row.get('val/loss')
            if val_loss_raw in {'', None}:
                continue
            summary['val_rows'] += 1
            val_loss = float(val_loss_raw)
            epoch = int(float(row['epoch']))
            summary['final_val_loss'] = val_loss
            summary['final_epoch'] = epoch
            if summary['best_val_loss'] is None or val_loss < summary['best_val_loss']:
                summary['best_val_loss'] = val_loss
                summary['best_epoch'] = epoch
    return summary


def _best_checkpoint(root_dir: Path) -> Optional[str]:
    candidates = sorted((root_dir / 'checkpoints').glob('best-*.ckpt'))
    return str(candidates[-1]) if candidates else None


def _parse_visualization_epoch(name: str) -> int:
    match = re.search(r'epoch_(\d+)_', name)
    return int(match.group(1)) if match else -1


def _visualization_snapshot(root_dir: Path) -> Dict:
    vis_dir = root_dir / 'visualizations'
    if not vis_dir.exists():
        return {
            'count': 0,
            'latest_epoch': None,
            'latest_files': [],
        }
    files = sorted(
        [
            path
            for pattern in ('epoch_*_*.jpg', 'epoch_*_*.png')
            for path in vis_dir.glob(pattern)
        ],
        key=lambda path: (_parse_visualization_epoch(path.name), path.name),
    )
    latest_epoch = _parse_visualization_epoch(files[-1].name) if files else None
    latest_files = [str(path) for path in files if _parse_visualization_epoch(path.name) == latest_epoch] if latest_epoch is not None else []
    return {
        'count': len(files),
        'latest_epoch': latest_epoch,
        'latest_files': latest_files,
    }


def _build_status(root_dir: Path) -> Dict:
    metrics_path = _latest_metrics_file(root_dir)
    metrics = _read_metrics(metrics_path)
    max_epochs = _read_max_epochs(metrics_path)
    visualizations = _visualization_snapshot(root_dir)
    checkpoint = _best_checkpoint(root_dir)
    completed = (
        max_epochs is not None
        and metrics['final_epoch'] is not None
        and metrics['final_epoch'] >= max_epochs - 1
    )
    return {
        'root_dir': str(root_dir),
        'status': 'completed' if completed else 'running',
        'max_epochs': max_epochs,
        'best_checkpoint': checkpoint,
        'delta_vs_memorization': None if metrics['best_val_loss'] is None else metrics['best_val_loss'] - BASELINE_MEMORIZATION,
        **metrics,
        'visualizations': visualizations,
        'updated_at_unix': int(time.time()),
    }


def _write_status_files(status: Dict) -> None:
    STATUS_JSON.write_text(json.dumps(status, indent=2), encoding='utf-8')
    best_val = '-' if status['best_val_loss'] is None else f"{status['best_val_loss']:.4f}"
    best_epoch = '-' if status['best_epoch'] is None else str(status['best_epoch'])
    final_val = '-' if status['final_val_loss'] is None else f"{status['final_val_loss']:.4f}"
    final_epoch = '-' if status['final_epoch'] is None else str(status['final_epoch'])
    delta = '-' if status['delta_vs_memorization'] is None else f"{status['delta_vs_memorization']:+.4f}"
    checkpoint = '-' if status['best_checkpoint'] is None else status['best_checkpoint']
    latest_vis_epoch = '-' if status['visualizations']['latest_epoch'] is None else str(status['visualizations']['latest_epoch'])
    lines = [
        '# All Except DN Status',
        '',
        f"- Status: {status['status']}",
        f"- Best val_loss: {best_val}",
        f"- Best epoch: {best_epoch}",
        f"- Final val_loss: {final_val}",
        f"- Final epoch: {final_epoch}",
        f"- Delta vs memorization: {delta}",
        f"- Best checkpoint: {checkpoint}",
        f"- Visualization file count: {status['visualizations']['count']}",
        f"- Latest visualization epoch: {latest_vis_epoch}",
        '',
        '## Latest visualization files',
        '',
    ]
    if status['visualizations']['latest_files']:
        for file_path in status['visualizations']['latest_files']:
            lines.append(f'- {file_path}')
    else:
        lines.append('- No visualization files emitted yet.')
    STATUS_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _append_final_result_to_report(status: Dict) -> None:
    if not BASELINE_REPORT.exists():
        return
    report = BASELINE_REPORT.read_text(encoding='utf-8')
    section = '\n'.join([
        RESULT_MARKER,
        '',
        '| Experiment | Best val_loss | Best epoch | Final val_loss | Delta vs memorization |',
        '| --- | ---: | ---: | ---: | ---: |',
        f"| all_except_dn | {status['best_val_loss']:.4f} | {status['best_epoch']} | {status['final_val_loss']:.4f} | {status['delta_vs_memorization']:+.4f} |",
        '',
        '### Interpretation',
        '',
        '- This run restores auxiliary loss, one-to-many supervision, top-k positive supervision, and distinct-query regularization while keeping denoising disabled.',
        f"- Final performance should be compared directly against `aux`, `one_to_many`, `topk`, and `distinct` to measure the combined non-DN regularization cost. Best checkpoint: `{status['best_checkpoint']}`.",
        '',
    ])
    if RESULT_MARKER in report:
        report = report.split(RESULT_MARKER, 1)[0].rstrip() + '\n\n' + section
    else:
        report = report.rstrip() + '\n\n' + section
    BASELINE_REPORT.write_text(report + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Monitor all-except-DN overfit experiment and update status/report files.')
    parser.add_argument('--interval-seconds', type=int, default=300)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()

    while True:
        status = _build_status(EXPERIMENT_ROOT)
        _write_status_files(status)
        if status['status'] == 'completed':
            _append_final_result_to_report(status)
            print(json.dumps(status, indent=2))
            return
        print(json.dumps(status, indent=2))
        if args.once:
            return
        time.sleep(max(30, args.interval_seconds))


if __name__ == '__main__':
    main()