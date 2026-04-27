from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge_datasets import build_datamodule
from misc_utils import load_config, maybe_load_conditioned_curve_init
from misc_utils.visualization_utils import render_curve_grid
from models.curve_coordinates import curve_internal_to_external
from pl_trainer.parametric_edge_trainer import ParametricEdgeLightningModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run one conditioned-curve forward pass and render predictions.')
    parser.add_argument('--config', default='configs/parametric_edge/archive/conditioned_curve_overfit1_lab34.yaml')
    parser.add_argument('--mode', choices=['train', 'val'], default='val')
    parser.add_argument('--output-dir', default='outputs/conditioned_curve_forward_once')
    parser.add_argument('--score-threshold', type=float, default=0.3)
    parser.add_argument('--max-curves', type=int, default=300)
    return parser.parse_args()


def _prepare_config(config_path: str, mode: str) -> dict:
    config = load_config(config_path)
    config['data']['batch_size'] = 1
    config['data']['val_batch_size'] = 1
    config['data']['num_workers'] = 0
    config['data']['persistent_workers'] = False
    config['data']['pin_memory'] = False
    config['trainer']['devices'] = 1
    config['trainer']['accumulate_grad_batches'] = 1
    config['trainer']['effective_batch_size'] = 1
    config['data']['train_augment'] = mode == 'train'
    return config


def main() -> None:
    args = parse_args()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _prepare_config(args.config, args.mode)
    datamodule = build_datamodule(config)
    datamodule.setup('fit')
    loader = datamodule.train_dataloader() if args.mode == 'train' else datamodule.val_dataloader()
    batch = next(iter(loader))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    module = ParametricEdgeLightningModule(config).to(device)
    init_stats = maybe_load_conditioned_curve_init(module, config)
    module.eval()

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

    with torch.no_grad():
        outputs = module(images, targets=targets, **model_inputs)

    probs = outputs['pred_logits'].softmax(-1)[..., 0]
    scored_curves = []
    pred_count = 0
    for pred_curves, pred_keep in zip(outputs['pred_curves'], probs > args.score_threshold):
        selected = pred_curves[pred_keep][: args.max_curves]
        pred_count = int(selected.shape[0])
        scored_curves.append(curve_internal_to_external(selected, config))

    image_path = output_dir / 'forward_scores.jpg'
    render_curve_grid(batch['images'], batch['targets'], scored_curves, image_path)

    summary = {
        'config': args.config,
        'mode': args.mode,
        'device': str(device),
        'init_stats': init_stats,
        'sample_id': str(batch['targets'][0]['sample_id']),
        'input_path': str(batch['targets'][0]['input_path']),
        'bezier_path': str(batch['targets'][0]['bezier_path']),
        'pred_curve_count': pred_count,
        'score_threshold': float(args.score_threshold),
        'image_path': str(image_path),
    }
    summary_path = output_dir / 'forward_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
