import argparse
import csv
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge_datasets.parametric_edge_datamodule import ParametricEdgeDataModule
from misc_utils.config_utils import load_config
from misc_utils.visualization_utils import render_curve_grid
from models.losses import compute_losses
from models.matcher import hungarian_curve_matching
from pl_trainer.parametric_edge_trainer import ParametricEdgeLightningModule


def load_model(checkpoint_path: Path, config):
    model = ParametricEdgeLightningModule(config)
    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=True)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / 'per_sample_vis'
    vis_dir.mkdir(parents=True, exist_ok=True)

    dm = ParametricEdgeDataModule(cfg)
    dm.setup('fit')
    dataset = dm.train_dataset
    model = load_model(Path(args.checkpoint), cfg)

    rows = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            batch = dataset[idx]
            images = batch['image'].unsqueeze(0)
            targets = [batch['target']]
            outputs = model(images, targets=targets)
            losses = compute_losses(outputs, targets, cfg)
            probs = outputs['pred_logits'].softmax(-1)[..., 0]
            keep = probs[0] > float(cfg['callbacks'].get('visualization_score_threshold', 0.15))
            scored_curves = [outputs['pred_curves'][0, keep]]
            matched_indices = hungarian_curve_matching(
                outputs['pred_logits'],
                outputs['pred_curves'],
                targets,
                class_cost=float(cfg['loss'].get('class_cost', 1.0)),
                control_cost=float(cfg['loss'].get('control_cost', 5.0)),
                sample_cost=float(cfg['loss'].get('sample_cost', 2.0)),
                box_cost=float(cfg['loss'].get('box_cost', 1.0)),
                num_curve_samples=int(cfg['loss'].get('num_curve_samples', 16)),
            )
            src_idx, _ = matched_indices[0]
            matched_curves = [outputs['pred_curves'][0, src_idx]]
            sample_id = targets[0]['sample_id']
            render_curve_grid(images, targets, matched_curves, vis_dir / f'{sample_id}_matched.png')
            rows.append({
                'sample_id': sample_id,
                'num_targets': int(targets[0]['num_targets']),
                'num_scored_curves': int(scored_curves[0].shape[0]),
                'num_matched_curves': int(matched_curves[0].shape[0]),
                'loss': float(losses['loss'].detach()),
                'loss_ctrl': float(losses['loss_ctrl'].detach()),
                'loss_sample': float(losses['loss_sample'].detach()),
                'loss_endpoint': float(losses['loss_endpoint'].detach()),
                'loss_bbox': float(losses['loss_bbox'].detach()),
                'loss_topk_pos': float(losses.get('loss_topk_pos', torch.tensor(0.0)).detach()),
            })

    rows = sorted(rows, key=lambda x: x['num_targets'])
    small = [row for row in rows if row['num_targets'] <= 40]
    large = [row for row in rows if row['num_targets'] >= 150]

    def avg(rows, key):
        return sum(r[key] for r in rows) / max(len(rows), 1)

    summary = {
        'num_samples': len(rows),
        'small_count': len(small),
        'large_count': len(large),
        'avg_loss_small': avg(small, 'loss'),
        'avg_loss_large': avg(large, 'loss'),
        'avg_sample_loss_small': avg(small, 'loss_sample'),
        'avg_sample_loss_large': avg(large, 'loss_sample'),
        'avg_endpoint_loss_small': avg(small, 'loss_endpoint'),
        'avg_endpoint_loss_large': avg(large, 'loss_endpoint'),
        'rows': rows,
    }

    with (output_dir / 'per_sample_metrics.json').open('w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)
    with (output_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    with (output_dir / 'per_sample_metrics.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
