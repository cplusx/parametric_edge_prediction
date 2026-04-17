#!/usr/bin/env python
from __future__ import annotations

import argparse
import cProfile
import importlib.util
import io
import json
import pstats
import sys
import time
from pathlib import Path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Profile v3 dataloader setup speed.')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--bezier-root', type=Path, required=True)
    parser.add_argument('--entry-cache-path', type=Path, required=True)
    parser.add_argument('--batch-glob', type=str, default='batch*')
    parser.add_argument('--val-max-samples', type=int, default=1024)
    parser.add_argument('--test-max-samples', type=int, default=1024)
    parser.add_argument('--train-max-samples', type=int, default=None)
    parser.add_argument('--profile-first-batch', action='store_true')
    parser.add_argument('--first-batch-train-max-samples', type=int, default=8)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--json-out', type=Path, default=None)
    return parser


def make_dataset_cfg(args: argparse.Namespace, *, train_max_samples=None) -> dict:
    return {
        'dataset_type': 'laion_synthetic',
        'data_root': str(args.data_root),
        'image_root': str(args.image_root),
        'bezier_root': str(args.bezier_root),
        'entry_cache_path': str(args.entry_cache_path),
        'batch_glob': args.batch_glob,
        'selection_seed': None,
        'max_samples': train_max_samples,
    }


def make_dm_config(args: argparse.Namespace, *, train_max_samples=None) -> dict:
    train_cfg = make_dataset_cfg(args, train_max_samples=train_max_samples)
    val_cfg = make_dataset_cfg(args)
    test_cfg = make_dataset_cfg(args)
    val_cfg['max_samples'] = int(args.val_max_samples)
    test_cfg['max_samples'] = int(args.test_max_samples)
    test_cfg['selection_offset'] = 0
    return {
        'data': {
            'include_primary_train_dataset': False,
            'image_size': 512,
            'version_name': 'v5_anchor_consistent',
            'target_degree': 5,
            'min_curve_length': 8.0,
            'max_targets': 256,
            'rgb_input': True,
            'batch_size': 8,
            'val_batch_size': 8,
            'num_workers': 16,
            'pin_memory': False,
            'persistent_workers': False,
            'bezier_root': None,
            'input_root': None,
            'extra_train_datasets': [train_cfg],
            'val_dataset': val_cfg,
            'test_dataset': test_cfg,
        },
        'model': {
            'arch': 'dab_curve_detr',
        },
    }


def load_discover_fn(repo_root: Path):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    module_path = repo_root / 'edge_datasets' / 'laion_synthetic_dataset.py'
    spec = importlib.util.spec_from_file_location('laion_synthetic_dataset_standalone', module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Unable to load {module_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.discover_laion_bezier_samples


def profile_call(fn, top_k: int):
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    result = fn()
    profiler.disable()
    elapsed = time.perf_counter() - start
    stats_stream = io.StringIO()
    pstats.Stats(profiler, stream=stats_stream).sort_stats('cumulative').print_stats(top_k)
    return result, elapsed, stats_stream.getvalue()


def main() -> None:
    args = build_argparser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    discover_laion_bezier_samples = load_discover_fn(repo_root)

    sample_records, discover_elapsed, discover_stats = profile_call(
        lambda: discover_laion_bezier_samples(
            data_root=args.data_root,
            image_root=args.image_root,
            bezier_root=args.bezier_root,
            entry_cache_path=args.entry_cache_path,
            batch_glob=args.batch_glob,
        ),
        top_k=args.top_k,
    )

    setup_elapsed = None
    setup_stats = 'skipped: local environment is missing torch/pytorch_lightning\n'
    train_len = 0
    val_len = 0
    test_len = 0
    first_batch_elapsed = None
    first_batch_stats = 'skipped\n'
    first_batch_summary = None
    try:
        from edge_datasets.parametric_edge_datamodule import ParametricEdgeDataModule

        dm = ParametricEdgeDataModule(make_dm_config(args, train_max_samples=args.train_max_samples))
        _, setup_elapsed, setup_stats = profile_call(lambda: dm.setup('fit'), top_k=args.top_k)
        train_len = len(dm.train_dataset) if dm.train_dataset is not None else 0
        val_len = len(dm.val_dataset) if dm.val_dataset is not None else 0
        test_len = len(dm.test_dataset) if dm.test_dataset is not None else 0

        if args.profile_first_batch:
            dm_batch = ParametricEdgeDataModule(
                make_dm_config(args, train_max_samples=int(args.first_batch_train_max_samples))
            )
            dm_batch.setup('fit')

            def load_first_batch():
                loader = dm_batch.train_dataloader()
                batch = next(iter(loader))
                return {
                    'images_shape': tuple(batch['images'].shape),
                    'targets_in_batch': len(batch['targets']),
                    'num_curves': [int(target['curves'].shape[0]) for target in batch['targets']],
                }

            first_batch_summary, first_batch_elapsed, first_batch_stats = profile_call(
                load_first_batch,
                top_k=args.top_k,
            )
    except ModuleNotFoundError:
        pass

    result = {
        'entry_cache_path': str(args.entry_cache_path),
        'discover_records': len(sample_records),
        'discover_elapsed_sec': discover_elapsed,
        'datamodule_setup_elapsed_sec': setup_elapsed,
        'train_len': train_len,
        'val_len': val_len,
        'test_len': test_len,
        'first_batch_elapsed_sec': first_batch_elapsed,
        'first_batch_summary': first_batch_summary,
        'discover_profile_top': discover_stats,
        'setup_profile_top': setup_stats,
        'first_batch_profile_top': first_batch_stats,
    }

    print(json.dumps({
        'entry_cache_path': result['entry_cache_path'],
        'discover_records': result['discover_records'],
        'discover_elapsed_sec': round(result['discover_elapsed_sec'], 4),
        'datamodule_setup_elapsed_sec': (
            round(result['datamodule_setup_elapsed_sec'], 4)
            if result['datamodule_setup_elapsed_sec'] is not None else None
        ),
        'train_len': result['train_len'],
        'val_len': result['val_len'],
        'test_len': result['test_len'],
        'first_batch_elapsed_sec': (
            round(result['first_batch_elapsed_sec'], 4)
            if result['first_batch_elapsed_sec'] is not None else None
        ),
        'first_batch_summary': result['first_batch_summary'],
    }, indent=2))

    print('\n=== discover profile ===')
    print(result['discover_profile_top'])
    print('\n=== datamodule setup profile ===')
    print(result['setup_profile_top'])
    print('\n=== first batch profile ===')
    print(result['first_batch_profile_top'])

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
