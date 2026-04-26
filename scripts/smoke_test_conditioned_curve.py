from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge_datasets import build_datamodule
from misc_utils.config_utils import load_config
from models import build_model
from models.losses import compute_losses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Smoke test conditioned-curve DAB.')
    parser.add_argument('--config', default='configs/parametric_edge/laion_curve_pretrain_lab34_v3_edgeprob05.yaml')
    parser.add_argument('--mode', choices=['eval', 'train_aug'], default='train_aug')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-batches', type=int, default=1)
    parser.add_argument('--output', default='outputs/conditioned_curve_debug/smoke_test_summary.json')
    return parser.parse_args()


def _prepare_config(config_path: str, mode: str, batch_size: int) -> dict:
    config = load_config(config_path)
    config['model']['arch'] = 'dab_cond_curve_detr'
    config['data']['batch_size'] = int(batch_size)
    config['data']['val_batch_size'] = int(batch_size)
    config['data']['num_workers'] = 0
    config['data']['persistent_workers'] = False
    config['data']['pin_memory'] = False
    config['data']['train_augment'] = mode == 'train_aug'
    config['trainer']['devices'] = 1
    config['trainer']['precision'] = '32-true'
    config['trainer']['accumulate_grad_batches'] = 1
    config['trainer']['effective_batch_size'] = int(batch_size)
    return config


def main() -> None:
    args = parse_args()
    config = _prepare_config(args.config, args.mode, args.batch_size)
    datamodule = build_datamodule(config)
    datamodule.setup('fit')
    loader = datamodule.train_dataloader() if args.mode == 'train_aug' else datamodule.val_dataloader()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config).to(device)
    model.train()

    batch_summaries = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= int(args.num_batches):
            break
        images = batch['images'].to(device)
        targets = []
        for target in batch['targets']:
            targets.append({
                key: (value.to(device) if torch.is_tensor(value) else value)
                for key, value in target.items()
            })
        model_inputs = {}
        for key, value in (batch.get('model_inputs') or {}).items():
            model_inputs[key] = value.to(device) if torch.is_tensor(value) else value

        t0 = time.perf_counter()
        outputs = model(images, targets=targets, **model_inputs)
        losses = compute_losses(outputs, targets, config)
        elapsed = time.perf_counter() - t0
        batch_summaries.append({
            'batch_idx': batch_idx,
            'image_shape': list(images.shape),
            'num_targets': [int(target['curves'].shape[0]) for target in targets],
            'condition_counts': [
                int((~model_inputs['condition_padding_mask'][row]).sum().item())
                for row in range(model_inputs['condition_padding_mask'].shape[0])
            ] if 'condition_padding_mask' in model_inputs else [],
            'pred_logits_shape': list(outputs['pred_logits'].shape),
            'pred_curves_shape': list(outputs['pred_curves'].shape),
            'loss': float(losses['loss'].detach().cpu().item()),
            'loss_focal': float(losses['loss_focal'].detach().cpu().item()) if 'loss_focal' in losses else None,
            'loss_chamfer': float(losses['loss_chamfer'].detach().cpu().item()) if 'loss_chamfer' in losses else None,
            'elapsed_sec': elapsed,
        })

    summary = {
        'config': args.config,
        'mode': args.mode,
        'device': str(device),
        'num_batches': len(batch_summaries),
        'batches': batch_summaries,
    }
    output_path = (REPO_ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
