import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge_datasets.parametric_edge_dataset import ParametricEdgeDataset, parametric_edge_collate
from misc_utils.bezier_target_utils import ensure_graph_cache
from misc_utils.config_utils import load_config
from misc_utils.visualization_utils import render_curve_grid


def _resolve_split_edge_paths(config: Dict, full_dataset: bool) -> Dict[str, List[Path]]:
    data_cfg = config['data']
    explicit_split_globs = any(
        data_cfg.get(key) is not None
        for key in ('train_edge_glob', 'train_edge_globs', 'val_edge_glob', 'val_edge_globs', 'test_edge_glob', 'test_edge_globs')
    )
    if explicit_split_globs:
        split_to_paths: Dict[str, List[Path]] = {'train': [], 'val': [], 'test': []}
        for split in ('train', 'val', 'test'):
            patterns = data_cfg.get(f'{split}_edge_globs', data_cfg.get(f'{split}_edge_glob'))
            if patterns is None:
                continue
            if isinstance(patterns, str):
                patterns = [patterns]
            for pattern in patterns:
                split_to_paths[split].extend(sorted(Path().glob(str(pattern))))
        for split, paths in split_to_paths.items():
            deduped = []
            seen = set()
            for path in paths:
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                deduped.append(path)
            split_to_paths[split] = deduped
        return split_to_paths
    edge_paths = sorted(Path().glob(data_cfg['edge_glob']))
    if not edge_paths:
        raise FileNotFoundError(f"No edge maps matched {data_cfg['edge_glob']}")
    if not full_dataset and data_cfg.get('sample_names'):
        wanted = set(data_cfg['sample_names'])
        edge_paths = [path for path in edge_paths if path.name in wanted or path.stem in wanted]
    if not full_dataset and data_cfg.get('limit_samples'):
        edge_paths = edge_paths[: int(data_cfg['limit_samples'])]
    return {'test': edge_paths}


def _build_dataset(config: Dict, edge_paths: Sequence[Path], split: str, train_augment: bool) -> ParametricEdgeDataset:
    data_cfg = config['data']
    input_root = data_cfg.get(f'{split}_input_root', data_cfg.get('input_root'))
    return ParametricEdgeDataset(
        edge_paths=edge_paths,
        cache_root=Path(data_cfg['cache_dir']),
        image_size=int(data_cfg['image_size']),
        version_name=data_cfg.get('target_version', 'v5_anchor_consistent'),
        input_root=Path(input_root) if input_root else None,
        rgb_input=bool(data_cfg.get('rgb_input', False)),
        target_degree=int(data_cfg.get('target_degree', 3)),
        min_curve_length=float(data_cfg.get('min_curve_length', 3.0)),
        max_targets=int(data_cfg.get('max_targets', 128)),
        split=split,
        train_augment=train_augment,
        augment_cfg=dict(data_cfg.get('augment', {})),
    )


def _flatten_split_paths(split_to_paths: Dict[str, List[Path]]) -> List[Path]:
    edge_paths: List[Path] = []
    seen = set()
    for split in ('train', 'val', 'test'):
        for path in split_to_paths.get(split, []):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            edge_paths.append(path)
    return edge_paths


def _build_loader(dataset: ParametricEdgeDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=parametric_edge_collate,
        pin_memory=False,
        persistent_workers=bool(num_workers > 0),
    )


def _prepare_one_graph_cache(task: Dict) -> str:
    cache_path = ensure_graph_cache(
        edge_path=Path(task['edge_path']),
        cache_root=Path(task['cache_root']),
        version_name=str(task['version_name']),
    )
    return str(cache_path)


def _extract_sample_ids(batch: Dict) -> List[str]:
    return [str(target['sample_id']) for target in batch['targets']]


def _extract_num_targets(batch: Dict) -> List[int]:
    return [int(target['num_targets']) for target in batch['targets']]


def precompute_graph_caches(config: Dict, edge_paths: Sequence[Path], cache_workers: int) -> Dict[str, float]:
    data_cfg = config['data']
    tasks = [
        {
            'edge_path': str(edge_path),
            'cache_root': str(Path(data_cfg['cache_dir'])),
            'version_name': data_cfg.get('target_version', 'v5_anchor_consistent'),
        }
        for edge_path in edge_paths
    ]
    prepared = 0
    start_time = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, cache_workers)) as executor:
        for prepared, _ in enumerate(executor.map(_prepare_one_graph_cache, tasks), start=1):
            if prepared % 50 == 0 or prepared == len(tasks):
                elapsed = time.perf_counter() - start_time
                print(
                    json.dumps({
                        'stage': 'cache_precompute',
                        'prepared': prepared,
                        'total': len(tasks),
                        'elapsed_sec': elapsed,
                        'samples_per_sec': prepared / max(elapsed, 1e-6),
                    }),
                    flush=True,
                )
    elapsed = time.perf_counter() - start_time
    return {
        'prepared_samples': prepared,
        'elapsed_sec': elapsed,
        'samples_per_sec': prepared / max(elapsed, 1e-6),
    }


def benchmark_loader(dataset: ParametricEdgeDataset, batch_size: int, num_workers: int, split_name: str) -> Dict:
    loader = _build_loader(dataset, batch_size=batch_size, num_workers=num_workers)
    rows: List[Dict] = []
    start_time = time.perf_counter()
    last_time = start_time
    for batch in loader:
        now = time.perf_counter()
        delta = now - last_time
        sample_ids = _extract_sample_ids(batch)
        num_targets = _extract_num_targets(batch)
        per_sample_delta = delta / max(len(sample_ids), 1)
        for sample_id, target_count in zip(sample_ids, num_targets):
            rows.append({
                'split': split_name,
                'sample_id': sample_id,
                'num_targets': target_count,
                'arrival_sec': per_sample_delta,
            })
        last_time = now
    total_elapsed = last_time - start_time
    arrival_values = np.asarray([row['arrival_sec'] for row in rows], dtype=np.float64) if rows else np.zeros((0,), dtype=np.float64)
    return {
        'num_samples': len(rows),
        'total_elapsed_sec': float(total_elapsed),
        'samples_per_sec': float(len(rows) / max(total_elapsed, 1e-6)),
        'mean_arrival_sec': float(arrival_values.mean()) if arrival_values.size else 0.0,
        'median_arrival_sec': float(np.median(arrival_values)) if arrival_values.size else 0.0,
        'p95_arrival_sec': float(np.quantile(arrival_values, 0.95)) if arrival_values.size else 0.0,
        'max_arrival_sec': float(arrival_values.max()) if arrival_values.size else 0.0,
        'min_arrival_sec': float(arrival_values.min()) if arrival_values.size else 0.0,
        'rows': rows,
    }


def _select_visual_indices(rows: Sequence[Dict], max_count: int) -> List[int]:
    if not rows:
        return []
    sorted_pairs = sorted(enumerate(rows), key=lambda item: (item[1]['num_targets'], item[0]))
    if len(sorted_pairs) <= max_count:
        return sorted(index for index, _ in sorted_pairs)
    selected = set()
    for step in np.linspace(0, len(sorted_pairs) - 1, num=max_count):
        selected.add(sorted_pairs[int(round(step))][0])
    return sorted(selected)


def benchmark_split_datasets(datasets: List[Tuple[str, ParametricEdgeDataset]], batch_size: int, num_workers: int) -> Dict:
    split_summaries = {}
    all_rows: List[Dict] = []
    total_elapsed_sec = 0.0
    for split_name, dataset in datasets:
        summary = benchmark_loader(dataset, batch_size=batch_size, num_workers=num_workers, split_name=split_name)
        split_summaries[split_name] = {key: value for key, value in summary.items() if key != 'rows'}
        total_elapsed_sec += float(summary['total_elapsed_sec'])
        all_rows.extend(summary['rows'])
    arrival_values = np.asarray([row['arrival_sec'] for row in all_rows], dtype=np.float64) if all_rows else np.zeros((0,), dtype=np.float64)
    return {
        'num_samples': len(all_rows),
        'total_elapsed_sec': float(total_elapsed_sec),
        'samples_per_sec': float(len(all_rows) / max(total_elapsed_sec, 1e-6)),
        'mean_arrival_sec': float(arrival_values.mean()) if arrival_values.size else 0.0,
        'median_arrival_sec': float(np.median(arrival_values)) if arrival_values.size else 0.0,
        'p95_arrival_sec': float(np.quantile(arrival_values, 0.95)) if arrival_values.size else 0.0,
        'max_arrival_sec': float(arrival_values.max()) if arrival_values.size else 0.0,
        'min_arrival_sec': float(arrival_values.min()) if arrival_values.size else 0.0,
        'split_summaries': split_summaries,
        'rows': all_rows,
    }


def _visualize_dataset(dataset: ParametricEdgeDataset, indices: Sequence[int], output_dir: Path, prefix: str) -> List[str]:
    saved_files: List[str] = []
    for chunk_id in range(int(math.ceil(len(indices) / 4.0))):
        chunk = list(indices[chunk_id * 4: (chunk_id + 1) * 4])
        if not chunk:
            continue
        samples = [dataset[index] for index in chunk]
        images = torch.stack([sample['image'] for sample in samples], dim=0)
        targets = [sample['target'] for sample in samples]
        predictions = [target['curves'] for target in targets]
        output_path = output_dir / f'{prefix}_{chunk_id + 1:02d}.jpg'
        render_curve_grid(
            images,
            targets,
            predictions,
            output_path,
            titles=('Input', 'GT Overlay', 'GT Overlay Copy', 'GT Curves Only'),
        )
        saved_files.append(str(output_path))
    return saved_files


def main() -> None:
    parser = argparse.ArgumentParser(description='Warm graph caches, benchmark dataloader read time, and export verification visualizations.')
    parser.add_argument('--config', default='configs/parametric_edge/default.yaml')
    parser.add_argument('--output-dir', default='outputs/parametric_edge_training/data_debug/cache_benchmark')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--cache-workers', type=int, default=12)
    parser.add_argument('--visualize-count', type=int, default=12)
    parser.add_argument('--full-dataset', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    split_to_paths = _resolve_split_edge_paths(config, full_dataset=bool(args.full_dataset))
    edge_paths = _flatten_split_paths(split_to_paths)
    eval_datasets = []
    if split_to_paths.get('val'):
        eval_datasets.append(('val', _build_dataset(config, edge_paths=split_to_paths['val'], split='val', train_augment=False)))
    if split_to_paths.get('test'):
        eval_datasets.append(('test', _build_dataset(config, edge_paths=split_to_paths['test'], split='test', train_augment=False)))
    if not eval_datasets:
        eval_datasets.append(('test', _build_dataset(config, edge_paths=edge_paths, split='test', train_augment=False)))
    train_dataset = _build_dataset(
        config,
        edge_paths=split_to_paths.get('train', edge_paths),
        split='train',
        train_augment=bool(config['data'].get('train_augment', True)),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warmup = precompute_graph_caches(config, edge_paths=edge_paths, cache_workers=args.cache_workers)
    benchmark = benchmark_split_datasets(eval_datasets, batch_size=args.batch_size, num_workers=args.num_workers)
    primary_eval_dataset = eval_datasets[-1][1]
    eval_rows_for_selection = [row for row in benchmark['rows'] if row['split'] == eval_datasets[-1][0]]
    visual_indices = _select_visual_indices(eval_rows_for_selection, max_count=max(1, int(args.visualize_count)))
    eval_visuals = _visualize_dataset(primary_eval_dataset, visual_indices, output_dir=output_dir, prefix='eval_samples')
    train_visuals = _visualize_dataset(train_dataset, visual_indices, output_dir=output_dir, prefix='train_samples')

    benchmark_rows_path = output_dir / 'per_image_read_times.csv'
    with benchmark_rows_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['split', 'sample_id', 'num_targets', 'arrival_sec'])
        writer.writeheader()
        writer.writerows(benchmark['rows'])

    summary = {
        'config': args.config,
        'num_workers': int(args.num_workers),
        'cache_workers': int(args.cache_workers),
        'batch_size': int(args.batch_size),
        'num_edge_paths': len(edge_paths),
        'cache_warmup': warmup,
        'benchmark': {key: value for key, value in benchmark.items() if key != 'rows'},
        'visual_sample_indices': visual_indices,
        'eval_visuals': eval_visuals,
        'train_visuals': train_visuals,
        'per_image_csv': str(benchmark_rows_path),
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()