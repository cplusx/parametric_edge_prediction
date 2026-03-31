from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from edge_datasets.laion_synthetic_dataset import _read_laion_entry_cache
from misc_utils.bezier_target_utils import build_cache_key, ensure_target_cache


def _select_records(
    records: Sequence[Dict[str, Path]],
    shard_index: int,
    num_shards: int,
) -> List[Dict[str, Path]]:
    selected: List[Dict[str, Path]] = []
    for record_index, record in enumerate(records):
        if int(record_index) % int(num_shards) == int(shard_index):
            selected.append(record)
    return selected


def _cache_path_for_record(
    record: Dict[str, Path],
    cache_root: Path,
    version_name: str,
    target_degree: int,
    min_curve_length: float,
) -> Path:
    edge_path = Path(record['edge_path'])
    cache_key = build_cache_key(
        edge_path=edge_path,
        version_name=version_name,
        target_degree=target_degree,
        min_curve_length=min_curve_length,
    )
    return cache_root / str(record['batch_name']) / f'{edge_path.stem}_{cache_key}.npz'


def _build_one(task: Tuple[str, str, str, str, int, float]) -> Dict[str, object]:
    edge_path_str, batch_name, cache_root_str, version_name, target_degree, min_curve_length = task
    edge_path = Path(edge_path_str)
    cache_root = Path(cache_root_str) / str(batch_name)
    cache_path = _cache_path_for_record(
        {'edge_path': edge_path, 'batch_name': batch_name},
        cache_root=Path(cache_root_str),
        version_name=version_name,
        target_degree=target_degree,
        min_curve_length=min_curve_length,
    )
    existed_before = cache_path.exists()
    start_time = time.perf_counter()
    ensure_target_cache(
        edge_path=edge_path,
        cache_root=cache_root,
        version_name=version_name,
        target_degree=int(target_degree),
        min_curve_length=float(min_curve_length),
    )
    elapsed = time.perf_counter() - start_time
    return {
        'edge_path': str(edge_path),
        'cache_path': str(cache_path),
        'existed_before': bool(existed_before),
        'built_now': not bool(existed_before),
        'elapsed_s': float(elapsed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--version-name', type=str, default='v5_anchor_consistent')
    parser.add_argument('--target-degree', type=int, default=5)
    parser.add_argument('--min-curve-length', type=float, default=3.0)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--log-every', type=int, default=500)
    args = parser.parse_args()

    entry_cache_path = Path(args.data_root) / 'laion_entry_cache.txt'
    records = _read_laion_entry_cache(entry_cache_path)
    selected = _select_records(records, shard_index=args.shard_index, num_shards=args.num_shards)
    if int(args.limit) > 0:
        selected = selected[: int(args.limit)]

    tasks = [
        (
            str(record['edge_path']),
            str(record['batch_name']),
            str(args.cache_root),
            str(args.version_name),
            int(args.target_degree),
            float(args.min_curve_length),
        )
        for record in selected
    ]

    summary = {
        'data_root': str(args.data_root),
        'cache_root': str(args.cache_root),
        'entry_cache_path': str(entry_cache_path),
        'version_name': str(args.version_name),
        'target_degree': int(args.target_degree),
        'min_curve_length': float(args.min_curve_length),
        'workers': int(args.workers),
        'shard_index': int(args.shard_index),
        'num_shards': int(args.num_shards),
        'limit': int(args.limit),
        'total_records': len(records),
        'selected_records': len(selected),
        'completed': 0,
        'existing': 0,
        'built': 0,
        'failed': 0,
        'wall_time_s': 0.0,
        'failures': [],
    }

    start_time = time.perf_counter()
    if tasks:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            future_to_task = {executor.submit(_build_one, task): task for task in tasks}
            for future_index, future in enumerate(as_completed(future_to_task), start=1):
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - best effort logging
                    task = future_to_task[future]
                    summary['failed'] += 1
                    summary['failures'].append({
                        'edge_path': task[0],
                        'error': repr(exc),
                    })
                else:
                    summary['completed'] += 1
                    summary['existing'] += int(bool(result['existed_before']))
                    summary['built'] += int(bool(result['built_now']))
                if future_index % int(args.log_every) == 0 or future_index == len(tasks):
                    elapsed = time.perf_counter() - start_time
                    rate = float(future_index) / max(elapsed, 1e-6)
                    print(
                        json.dumps(
                            {
                                'progress': future_index,
                                'total': len(tasks),
                                'built': summary['built'],
                                'existing': summary['existing'],
                                'failed': summary['failed'],
                                'elapsed_s': round(elapsed, 2),
                                'items_per_s': round(rate, 3),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

    summary['wall_time_s'] = float(time.perf_counter() - start_time)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
