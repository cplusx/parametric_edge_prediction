import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from misc_utils.bezier_target_utils import ensure_target_cache, image_id_from_stem, load_binary_edge_annotation, load_cached_targets
from misc_utils.config_utils import load_config


def _proxy_row(edge_path_str: str) -> Dict:
    edge_path = Path(edge_path_str)
    edge_binary = load_binary_edge_annotation(edge_path)
    edge_mask = edge_binary > 0
    component_count = int(ndimage.label(edge_mask)[1])
    edge_pixels = int(edge_mask.sum())
    return {
        'sample_name': edge_path.name,
        'image_id': image_id_from_stem(edge_path.stem),
        'num_curves': component_count,
        'total_curve_length': float(edge_pixels),
    }


def _collect_row(task: Tuple[str, str, str, int, float]) -> Dict:
    edge_path_str, cache_root_str, version_name, target_degree, min_curve_length = task
    edge_path = Path(edge_path_str)
    cache_path = ensure_target_cache(
        edge_path=edge_path,
        cache_root=Path(cache_root_str),
        version_name=version_name,
        target_degree=target_degree,
        min_curve_length=min_curve_length,
    )
    payload = load_cached_targets(cache_path)
    curve_lengths = payload.get('curve_lengths')
    return {
        'sample_name': edge_path.name,
        'image_id': image_id_from_stem(edge_path.stem),
        'num_curves': int(payload['curves'].shape[0]),
        'total_curve_length': float(curve_lengths.sum()) if curve_lengths is not None and curve_lengths.size else 0.0,
    }


def _run_parallel(items: List, worker_fn, workers: int, progress_label: str) -> List[Dict]:
    if workers <= 1:
        rows: List[Dict] = []
        for idx, item in enumerate(items, start=1):
            rows.append(worker_fn(item))
            if idx % 50 == 0 or idx == len(items):
                print(f'{progress_label}: processed {idx}/{len(items)}')
        return rows

    rows_by_name: Dict[str, Dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker_fn, item) for item in items]
        for idx, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows_by_name[row['sample_name']] = row
            if idx % 50 == 0 or idx == len(futures):
                print(f'{progress_label}: processed {idx}/{len(futures)}')
    ordered_names = [Path(item[0]).name if isinstance(item, tuple) else Path(item).name for item in items]
    return [rows_by_name[name] for name in ordered_names if name in rows_by_name]


def collect_rows(config: Dict, workers: int, prescreen_multiplier: int) -> List[Dict]:
    data_cfg = config['data']
    edge_paths = sorted(Path().glob(data_cfg['edge_glob']))
    candidate_paths = edge_paths
    if prescreen_multiplier > 1 and len(edge_paths) > prescreen_multiplier:
        proxy_rows = _run_parallel([str(path) for path in edge_paths], _proxy_row, workers, progress_label='proxy prescreen')
        candidate_count = min(len(proxy_rows), max(1, prescreen_multiplier))
        candidate_rows = choose_diverse_samples(proxy_rows, count=candidate_count)
        candidate_names = {row['sample_name'] for row in candidate_rows}
        candidate_paths = [path for path in edge_paths if path.name in candidate_names]
        print(f'proxy prescreen: reduced expensive Bezier evaluation from {len(edge_paths)} to {len(candidate_paths)} candidates')

    tasks = [
        (
            str(edge_path),
            str(Path(data_cfg['cache_dir'])),
            data_cfg.get('target_version', 'v5_anchor_consistent'),
            int(data_cfg.get('target_degree', 5)),
            float(data_cfg.get('min_curve_length', 3.0)),
        )
        for edge_path in candidate_paths
    ]
    return _run_parallel(tasks, _collect_row, workers, progress_label='bezier stats')


def choose_diverse_samples(rows: List[Dict], count: int) -> List[Dict]:
    if count <= 0 or not rows:
        return []
    ordered = sorted(rows, key=lambda row: (row['num_curves'], row['total_curve_length'], row['sample_name']))
    positions = np.linspace(0, len(ordered) - 1, num=min(count, len(ordered)))
    selected: List[Dict] = []
    used_names = set()
    used_image_ids = set()
    for position in positions:
        center = int(round(float(position)))
        search_order = [center]
        for radius in range(1, len(ordered)):
            left = center - radius
            right = center + radius
            if left >= 0:
                search_order.append(left)
            if right < len(ordered):
                search_order.append(right)
            if left < 0 and right >= len(ordered):
                break
        chosen = None
        for idx in search_order:
            row = ordered[idx]
            if row['sample_name'] in used_names:
                continue
            if row['image_id'] in used_image_ids:
                continue
            chosen = row
            break
        if chosen is None:
            for idx in search_order:
                row = ordered[idx]
                if row['sample_name'] not in used_names:
                    chosen = row
                    break
        if chosen is None:
            continue
        selected.append(chosen)
        used_names.add(chosen['sample_name'])
        used_image_ids.add(chosen['image_id'])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description='Select an overfit subset with diverse Bezier curve counts.')
    parser.add_argument('--config', default='configs/parametric_edge/default.yaml')
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument('--prescreen-multiplier', type=int, default=64)
    parser.add_argument('--output', type=str, default='outputs/parametric_edge_training/overfit_sample_selection.json')
    args = parser.parse_args()

    config = load_config(args.config)
    rows = collect_rows(
        config,
        workers=max(1, int(args.workers)),
        prescreen_multiplier=max(int(args.count), int(args.prescreen_multiplier)),
    )
    selected = choose_diverse_samples(rows, count=args.count)

    counts = np.asarray([row['num_curves'] for row in rows], dtype=np.float32)
    result = {
        'count': len(selected),
        'summary': {
            'num_candidates': len(rows),
            'min_curves': float(counts.min()) if counts.size else 0.0,
            'p25_curves': float(np.percentile(counts, 25)) if counts.size else 0.0,
            'p50_curves': float(np.percentile(counts, 50)) if counts.size else 0.0,
            'p75_curves': float(np.percentile(counts, 75)) if counts.size else 0.0,
            'max_curves': float(counts.max()) if counts.size else 0.0,
        },
        'selected': selected,
        'sample_names': [row['sample_name'] for row in selected],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()