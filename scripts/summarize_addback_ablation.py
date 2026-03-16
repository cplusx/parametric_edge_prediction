import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


EXPERIMENTS = [
    ('memorization', 'outputs/parametric_edge_training/overfit_diverse16_2000_memorization_base'),
    ('aux', 'outputs/parametric_edge_training/overfit_diverse16_2000_addback_aux'),
    ('dn', 'outputs/parametric_edge_training/overfit_diverse16_2000_addback_dn'),
    ('one_to_many', 'outputs/parametric_edge_training/overfit_diverse16_2000_addback_onetomany'),
    ('topk', 'outputs/parametric_edge_training/overfit_diverse16_2000_addback_topk'),
    ('distinct', 'outputs/parametric_edge_training/overfit_diverse16_2000_addback_distinct'),
]

MAIN_LOSS_COMPONENTS = [
    ('val/loss_ce', 'ce_weight', 1.0),
    ('val/loss_ctrl', 'ctrl_weight', 5.0),
    ('val/loss_endpoint', 'endpoint_weight', 2.0),
    ('val/loss_bbox', 'bbox_weight', 2.0),
    ('val/loss_giou', 'giou_weight', 1.0),
    ('val/loss_curve_dist', 'curve_distance_weight', 2.0),
    ('val/loss_extent', 'extent_weight', 0.0),
]


def _version_index(path: Path) -> int:
    try:
        return int(path.parent.name.split('_')[-1])
    except (ValueError, IndexError):
        return -1


def _metrics_files(root_dir: Path) -> List[Path]:
    csv_root = root_dir / 'csv_logs'
    if not csv_root.exists():
        return []
    return sorted(csv_root.glob('version_*/metrics.csv'), key=_version_index)


def _hparams_for_metrics(metrics_path: Path) -> Optional[Path]:
    hparams_path = metrics_path.with_name('hparams.yaml')
    return hparams_path if hparams_path.exists() else None


def _read_max_epochs(hparams_path: Optional[Path]) -> Optional[int]:
    if hparams_path is None or not hparams_path.exists():
        return None
    for line in hparams_path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if stripped.startswith('max_epochs:'):
            return int(stripped.split(':', 1)[1].strip())
    return None


def _read_loss_weights(hparams_path: Optional[Path]) -> Dict[str, float]:
    if hparams_path is None or not hparams_path.exists():
        return {}

    weights: Dict[str, float] = {}
    in_loss = False
    for line in hparams_path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        if not line.startswith(' '):
            in_loss = line.startswith('loss:')
            continue
        if not in_loss:
            continue
        if not line.startswith('  '):
            break
        key, _, raw_value = line.strip().partition(':')
        value = raw_value.strip()
        try:
            weights[key] = float(value)
        except ValueError:
            continue
    return weights


def _to_float(value: Optional[str]) -> Optional[float]:
    if value in {'', None}:
        return None
    return float(value)


def _compute_main_loss(row: Dict[str, str], loss_weights: Dict[str, float]) -> Optional[float]:
    total = 0.0
    found_component = False
    for metric_key, weight_key, default_weight in MAIN_LOSS_COMPONENTS:
        metric_value = _to_float(row.get(metric_key))
        if metric_value is None:
            continue
        total += loss_weights.get(weight_key, default_weight) * metric_value
        found_component = True
    return total if found_component else None


def _read_metrics(metrics_path: Path, loss_weights: Dict[str, float]) -> Dict:
    best_val = None
    best_epoch = None
    best_step = None
    best_val_main = None
    best_main_epoch = None
    best_main_step = None
    final_val = None
    final_epoch = None
    final_step = None
    final_val_main = None
    train_rows = 0
    val_rows = 0

    with metrics_path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get('train/loss') not in {'', None}:
                train_rows += 1
            val_loss_raw = row.get('val_loss') or row.get('val/loss')
            if val_loss_raw in {'', None}:
                continue
            val_rows += 1
            val_loss = float(val_loss_raw)
            epoch = int(float(row['epoch'])) if row.get('epoch') not in {'', None} else None
            step = int(float(row['step'])) if row.get('step') not in {'', None} else None
            val_loss_main = _compute_main_loss(row, loss_weights)
            final_val = val_loss
            final_epoch = epoch
            final_step = step
            final_val_main = val_loss_main
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_step = step
            if val_loss_main is not None and (best_val_main is None or val_loss_main < best_val_main):
                best_val_main = val_loss_main
                best_main_epoch = epoch
                best_main_step = step

    return {
        'metrics_path': str(metrics_path),
        'train_rows': train_rows,
        'val_rows': val_rows,
        'best_val_loss': best_val,
        'best_epoch': best_epoch,
        'best_step': best_step,
        'best_val_loss_main': best_val_main,
        'best_main_epoch': best_main_epoch,
        'best_main_step': best_main_step,
        'final_val_loss': final_val,
        'final_epoch': final_epoch,
        'final_step': final_step,
        'final_val_loss_main': final_val_main,
    }


def _best_checkpoint(root_dir: Path) -> Optional[str]:
    checkpoint_dir = root_dir / 'checkpoints'
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob('best-*.ckpt'))
    return str(candidates[-1]) if candidates else None


def _status_from_summary(summary: Dict) -> str:
    max_epochs = summary.get('max_epochs')
    final_epoch = summary.get('final_epoch')
    val_rows = summary.get('val_rows', 0)
    if val_rows == 0:
        return 'pending'
    if max_epochs is not None and final_epoch is not None and final_epoch >= max_epochs - 1:
        return 'completed'
    return 'running'


def _collect_versions(root_dir: Path) -> List[Dict]:
    versions: List[Dict] = []
    for metrics_path in _metrics_files(root_dir):
        hparams_path = _hparams_for_metrics(metrics_path)
        max_epochs = _read_max_epochs(hparams_path)
        loss_weights = _read_loss_weights(hparams_path)
        version_summary = {
            'version': metrics_path.parent.name,
            'metrics_path': str(metrics_path),
            'hparams_path': None if hparams_path is None else str(hparams_path),
            'max_epochs': max_epochs,
            'loss_weights': loss_weights,
        }
        version_summary.update(_read_metrics(metrics_path, loss_weights))
        version_summary['status'] = _status_from_summary(version_summary)
        versions.append(version_summary)
    return versions


def _select_primary_version(versions: List[Dict]) -> Optional[Dict]:
    if not versions:
        return None
    completed = [version for version in versions if version['status'] == 'completed']
    if completed:
        return min(
            completed,
            key=lambda version: (
                version['best_val_loss'] is None,
                version['best_val_loss'] if version['best_val_loss'] is not None else float('inf'),
                version['version'],
            ),
        )
    return versions[-1]


def summarize_experiment(name: str, root: str) -> Dict:
    root_dir = Path(root)
    versions = _collect_versions(root_dir)
    primary = _select_primary_version(versions)
    summary = {
        'name': name,
        'root_dir': str(root_dir),
        'status': 'pending',
        'metrics_path': None,
        'hparams_path': None,
        'max_epochs': None,
        'best_checkpoint': _best_checkpoint(root_dir),
        'train_rows': 0,
        'val_rows': 0,
        'best_val_loss': None,
        'best_epoch': None,
        'best_step': None,
        'best_val_loss_main': None,
        'best_main_epoch': None,
        'best_main_step': None,
        'final_val_loss': None,
        'final_epoch': None,
        'final_step': None,
        'final_val_loss_main': None,
        'delta_vs_memorization_best': None,
        'delta_vs_memorization_main': None,
        'selected_version': None,
        'versions': versions,
    }
    if primary is None:
        return summary

    summary.update(primary)
    summary['selected_version'] = primary['version']
    summary['status'] = primary['status']
    return summary


def build_markdown(rows: List[Dict]) -> str:
    memorization_best = next((row['best_val_loss'] for row in rows if row['name'] == 'memorization'), None)
    memorization_main = next((row['best_val_loss_main'] for row in rows if row['name'] == 'memorization'), None)
    for row in rows:
        if memorization_best is not None and row['best_val_loss'] is not None:
            row['delta_vs_memorization_best'] = row['best_val_loss'] - memorization_best
        if memorization_main is not None and row['best_val_loss_main'] is not None:
            row['delta_vs_memorization_main'] = row['best_val_loss_main'] - memorization_main

    lines = [
        '# Add-Back Ablation Summary',
        '',
        '| Experiment | Version | Status | Best val_loss_main | Main epoch | Delta main vs memorization | Best val_loss | Best epoch | Final val_loss | Val rows | Checkpoint |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |',
    ]

    for row in rows:
        lines.append(
            '| {name} | {version} | {status} | {best_val_loss_main} | {best_main_epoch} | {delta_main} | {best_val_loss} | {best_epoch} | {final_val_loss} | {val_rows} | {checkpoint} |'.format(
                name=row['name'],
                version='-' if row['selected_version'] is None else row['selected_version'],
                status=row['status'],
                best_val_loss_main='-' if row['best_val_loss_main'] is None else f"{row['best_val_loss_main']:.4f}",
                best_main_epoch='-' if row['best_main_epoch'] is None else row['best_main_epoch'],
                delta_main='-' if row['delta_vs_memorization_main'] is None else f"{row['delta_vs_memorization_main']:+.4f}",
                best_val_loss='-' if row['best_val_loss'] is None else f"{row['best_val_loss']:.4f}",
                best_epoch='-' if row['best_epoch'] is None else row['best_epoch'],
                final_val_loss='-' if row['final_val_loss'] is None else f"{row['final_val_loss']:.4f}",
                val_rows=row['val_rows'],
                checkpoint='-' if row['best_checkpoint'] is None else Path(row['best_checkpoint']).name,
            )
        )

    main_degradations = [
        row for row in rows
        if row['name'] != 'memorization' and row['delta_vs_memorization_main'] is not None
    ]
    main_degradations.sort(key=lambda row: row['delta_vs_memorization_main'], reverse=True)

    lines.extend(['', '## Current ranking by best val/loss_main degradation', ''])
    if not main_degradations:
        lines.append('No add-back run has produced a validation point yet.')
    else:
        for index, row in enumerate(main_degradations, start=1):
            lines.append(
                f"{index}. {row['name']}: best val/loss_main {row['best_val_loss_main']:.4f} ({row['delta_vs_memorization_main']:+.4f} vs memorization, total best val/loss {row['best_val_loss']:.4f}, {row['selected_version']})"
            )

    total_degradations = [
        row for row in rows
        if row['name'] != 'memorization' and row['delta_vs_memorization_best'] is not None
    ]
    total_degradations.sort(key=lambda row: row['delta_vs_memorization_best'], reverse=True)

    lines.extend(['', '## Reference ranking by best val/loss degradation', ''])
    if not total_degradations:
        lines.append('No add-back run has produced a validation point yet.')
    else:
        for index, row in enumerate(total_degradations, start=1):
            lines.append(
                f"{index}. {row['name']}: best val/loss {row['best_val_loss']:.4f} ({row['delta_vs_memorization_best']:+.4f} vs memorization, best val/loss_main {row['best_val_loss_main']:.4f}, {row['selected_version']})"
            )

    extra_incomplete = []
    for row in rows:
        for version in row.get('versions', []):
            if version['version'] != row.get('selected_version') and version['status'] == 'running':
                extra_incomplete.append((row['name'], version['version'], version['final_epoch']))

    lines.extend(['', '## Extra incomplete duplicate versions', ''])
    if not extra_incomplete:
        lines.append('No extra incomplete duplicate versions were found.')
    else:
        for name, version, final_epoch in extra_incomplete:
            lines.append(f'- {name}: {version} stopped early at epoch {final_epoch}')

    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize memorization add-back ablation runs.')
    parser.add_argument(
        '--output-dir',
        default='outputs/parametric_edge_training/current_sweep_analysis',
        help='Directory where markdown and json summaries will be written.',
    )
    args = parser.parse_args()

    rows = [summarize_experiment(name, root) for name, root in EXPERIMENTS]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown = build_markdown(rows)
    (output_dir / 'addback_ablation_summary.md').write_text(markdown, encoding='utf-8')
    (output_dir / 'addback_ablation_summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')

    print(markdown)


if __name__ == '__main__':
    main()