#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, median

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge_datasets.endpoint_datamodule import EndpointDetectionDataModule


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Profile v3 endpoint dataloader runtime.')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--bezier-root', type=Path, required=True)
    parser.add_argument('--entry-cache-path', type=Path, required=True)
    parser.add_argument('--batch-glob', type=str, default='batch*')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--num-batches', type=int, default=100)
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--max-targets', type=int, default=512)
    parser.add_argument('--target-degree', type=int, default=5)
    parser.add_argument('--endpoint-dedupe-distance-px', type=float, default=2.0)
    parser.add_argument('--rgb-input', action='store_true', default=True)
    parser.add_argument('--train-augment', action='store_true', default=True)
    parser.add_argument('--selection-seed', type=int, default=None)
    parser.add_argument('--selection-offset', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--json-out', type=Path, required=True)
    return parser


def build_config(args: argparse.Namespace) -> dict:
    dataset_cfg = {
        'dataset_type': 'laion_synthetic',
        'data_root': str(args.data_root),
        'image_root': str(args.image_root),
        'bezier_root': str(args.bezier_root),
        'entry_cache_path': str(args.entry_cache_path),
        'batch_glob': str(args.batch_glob),
        'selection_seed': args.selection_seed,
        'selection_offset': int(args.selection_offset),
        'max_samples': args.max_samples,
    }
    return {
        'data': {
            'include_primary_train_dataset': False,
            'rgb_input': bool(args.rgb_input),
            'image_size': int(args.image_size),
            'batch_size': int(args.batch_size),
            'val_batch_size': int(args.batch_size),
            'num_workers': int(args.num_workers),
            'pin_memory': False,
            'persistent_workers': bool(args.num_workers > 0),
            'target_degree': int(args.target_degree),
            'max_targets': int(args.max_targets),
            'endpoint_dedupe_distance_px': float(args.endpoint_dedupe_distance_px),
            'train_augment': bool(args.train_augment),
            'augment': {
                'resize_scale_range': [1.0, 1.5],
                'affine_scale_range': [0.9, 1.1],
                'affine_max_rotate_deg': 10.0,
                'affine_max_translate_ratio': 0.06,
                'hflip_prob': 0.5,
                'vflip_prob': 0.0,
            },
            'extra_train_datasets': [dataset_cfg],
            'val_dataset': dataset_cfg,
            'test_dataset': dataset_cfg,
        },
        'model': {
            'arch': 'dab_endpoint_detr',
        },
        'trainer': {},
    }


def main() -> None:
    args = build_argparser().parse_args()
    config = build_config(args)

    setup_start = time.perf_counter()
    dm = EndpointDetectionDataModule(config)
    dm.setup('fit')
    setup_elapsed = time.perf_counter() - setup_start

    loader_build_start = time.perf_counter()
    loader = dm.train_dataloader()
    loader_build_elapsed = time.perf_counter() - loader_build_start

    batch_times = []
    batch_sizes = []
    point_counts = []
    curve_counts = []

    total_images = 0
    total_points = 0
    total_curves = 0
    first_batch_summary = None

    iter_start = time.perf_counter()
    iterator = iter(loader)
    iterator_build_elapsed = time.perf_counter() - iter_start

    for batch_index in range(int(args.num_batches)):
        start = time.perf_counter()
        batch = next(iterator)
        elapsed = time.perf_counter() - start
        batch_times.append(elapsed)

        images = batch['images']
        targets = batch['targets']
        batch_size = int(images.shape[0])
        batch_point_counts = [int(target['points'].shape[0]) for target in targets]
        batch_curve_counts = [int(target['curves'].shape[0]) for target in targets]

        batch_sizes.append(batch_size)
        point_counts.extend(batch_point_counts)
        curve_counts.extend(batch_curve_counts)
        total_images += batch_size
        total_points += sum(batch_point_counts)
        total_curves += sum(batch_curve_counts)

        if batch_index == 0:
            first_batch_summary = {
                'images_shape': list(images.shape),
                'batch_size': batch_size,
                'target_keys': sorted(targets[0].keys()) if targets else [],
                'batch_point_counts': batch_point_counts,
                'batch_curve_counts': batch_curve_counts,
            }

    total_batch_time = float(sum(batch_times))
    later_batch_times = batch_times[1:] if len(batch_times) > 1 else batch_times

    result = {
        'data_root': str(args.data_root),
        'image_root': str(args.image_root),
        'bezier_root': str(args.bezier_root),
        'entry_cache_path': str(args.entry_cache_path),
        'num_batches': int(args.num_batches),
        'batch_size': int(args.batch_size),
        'num_workers': int(args.num_workers),
        'train_len': len(dm.train_dataset) if dm.train_dataset is not None else 0,
        'setup_elapsed_sec': setup_elapsed,
        'loader_build_elapsed_sec': loader_build_elapsed,
        'iterator_build_elapsed_sec': iterator_build_elapsed,
        'first_batch_elapsed_sec': batch_times[0] if batch_times else None,
        'mean_batch_elapsed_sec': mean(batch_times) if batch_times else None,
        'median_batch_elapsed_sec': median(batch_times) if batch_times else None,
        'mean_later_batch_elapsed_sec': mean(later_batch_times) if later_batch_times else None,
        'median_later_batch_elapsed_sec': median(later_batch_times) if later_batch_times else None,
        'total_batch_elapsed_sec': total_batch_time,
        'images_per_sec': (total_images / total_batch_time) if total_batch_time > 0 else None,
        'batches_per_sec': (len(batch_times) / total_batch_time) if total_batch_time > 0 else None,
        'total_images': total_images,
        'total_points': total_points,
        'total_curves': total_curves,
        'mean_points_per_image': (total_points / total_images) if total_images > 0 else None,
        'mean_curves_per_image': (total_curves / total_images) if total_images > 0 else None,
        'first_batch_summary': first_batch_summary,
        'batch_times_sec': batch_times,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2), encoding='utf-8')

    print(json.dumps({
        'num_batches': result['num_batches'],
        'batch_size': result['batch_size'],
        'num_workers': result['num_workers'],
        'setup_elapsed_sec': round(result['setup_elapsed_sec'], 4),
        'first_batch_elapsed_sec': round(result['first_batch_elapsed_sec'], 4) if result['first_batch_elapsed_sec'] is not None else None,
        'mean_batch_elapsed_sec': round(result['mean_batch_elapsed_sec'], 4) if result['mean_batch_elapsed_sec'] is not None else None,
        'mean_later_batch_elapsed_sec': round(result['mean_later_batch_elapsed_sec'], 4) if result['mean_later_batch_elapsed_sec'] is not None else None,
        'images_per_sec': round(result['images_per_sec'], 4) if result['images_per_sec'] is not None else None,
        'mean_points_per_image': round(result['mean_points_per_image'], 2) if result['mean_points_per_image'] is not None else None,
        'mean_curves_per_image': round(result['mean_curves_per_image'], 2) if result['mean_curves_per_image'] is not None else None,
        'json_out': str(args.json_out),
    }, indent=2))


if __name__ == '__main__':
    main()
