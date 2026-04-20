import argparse
import json
import time
from pathlib import Path

import torch

from edge_datasets import build_datamodule
from misc_utils.config_utils import load_config
from models import build_model
from models.losses import compute_losses


def main() -> None:
    parser = argparse.ArgumentParser(description='Smoke test curve DAB on a local v3 bezier subset.')
    parser.add_argument('--config', default='configs/parametric_edge/laion_pretrain_cluster_dn_aux_ce.yaml')
    parser.add_argument(
        '--fixture-root',
        default='outputs/curve_dab_smoke20_fixture',
    )
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-batches', type=int, default=2)
    parser.add_argument('--output', default='outputs/curve_dab_smoke20_fixture/smoke_test_summary.json')
    args = parser.parse_args()

    fixture_root = Path(args.fixture_root).resolve()
    entry_cache_path = fixture_root / 'laion_entry_cache_v3_bezier_20_local.txt'
    if not entry_cache_path.exists():
        raise FileNotFoundError(entry_cache_path)

    config = load_config(args.config)
    config['data']['include_primary_train_dataset'] = False
    config['data']['batch_size'] = int(args.batch_size)
    config['data']['val_batch_size'] = int(args.batch_size)
    config['data']['num_workers'] = 0
    config['data']['persistent_workers'] = False
    config['trainer']['devices'] = 1
    config['trainer']['accumulate_grad_batches'] = 1
    config['trainer']['effective_batch_size'] = int(args.batch_size)
    config['trainer']['effective_train_batches_per_epoch'] = None
    config['data']['extra_train_datasets'] = [{
        'dataset_type': 'laion_synthetic',
        'data_root': str(fixture_root),
        'image_root': str(fixture_root / 'edge_detection'),
        'bezier_root': str(fixture_root / 'laion_edge_v3_bezier'),
        'batch_glob': 'batch*',
        'entry_cache_path': str(entry_cache_path),
        'selection_seed': 20260318,
        'selection_offset': 0,
        'max_samples': 20,
    }]
    config['data']['val_dataset'] = {
        'dataset_type': 'laion_synthetic',
        'data_root': str(fixture_root),
        'image_root': str(fixture_root / 'edge_detection'),
        'bezier_root': str(fixture_root / 'laion_edge_v3_bezier'),
        'batch_glob': 'batch*',
        'entry_cache_path': str(entry_cache_path),
        'selection_seed': 20260318,
        'selection_offset': 0,
        'max_samples': 20,
    }
    config['data']['test_dataset'] = config['data']['val_dataset']

    datamodule = build_datamodule(config)
    setup_t0 = time.perf_counter()
    datamodule.setup(stage='fit')
    setup_elapsed = time.perf_counter() - setup_t0

    model = build_model(config)
    model.train()

    loader = datamodule.train_dataloader()
    batch_summaries = []
    batch_times = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= int(args.num_batches):
            break
        t0 = time.perf_counter()
        outputs = model(batch['images'], targets=batch['targets'])
        losses = compute_losses(outputs, batch['targets'], config)
        elapsed = time.perf_counter() - t0
        batch_times.append(elapsed)
        cls_key = 'loss_focal' if 'loss_focal' in losses else 'loss_ce'
        batch_summaries.append({
            'batch_idx': batch_idx,
            'image_shape': list(batch['images'].shape),
            'num_targets': [int(target['curves'].shape[0]) for target in batch['targets']],
            'loss': float(losses['loss'].detach().cpu().item()),
            cls_key: float(losses[cls_key].detach().cpu().item()),
            'loss_chamfer': float(losses['loss_chamfer'].detach().cpu().item()),
            'elapsed_sec': elapsed,
        })

    summary = {
        'fixture_root': str(fixture_root),
        'entry_cache_path': str(entry_cache_path),
        'train_len': len(datamodule.train_dataset),
        'setup_elapsed_sec': setup_elapsed,
        'batch_size': int(args.batch_size),
        'num_batches': len(batch_summaries),
        'mean_batch_elapsed_sec': (sum(batch_times) / len(batch_times)) if batch_times else None,
        'batches': batch_summaries,
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
