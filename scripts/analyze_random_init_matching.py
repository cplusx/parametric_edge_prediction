import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch

from edge_datasets.parametric_edge_datamodule import ParametricEdgeDataModule
from misc_utils.config_utils import load_config
from misc_utils.train_utils import sample_bezier_curves_torch
from models import build_model
from models.curve_coordinates import curve_internal_to_external
from models.geometry import reverse_curve_points
from models.matcher import HungarianCurveMatcher


def _align_target_to_prediction(
    pred_curves_ext: torch.Tensor,
    tgt_curves_ext: torch.Tensor,
    image_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tgt_rev_ext = reverse_curve_points(tgt_curves_ext)

    pred_px = pred_curves_ext * image_size
    tgt_px = tgt_curves_ext * image_size
    tgt_rev_px = tgt_rev_ext * image_size

    ctrl_forward = torch.linalg.norm(pred_px - tgt_px, dim=-1).mean(dim=1)
    ctrl_reverse = torch.linalg.norm(pred_px - tgt_rev_px, dim=-1).mean(dim=1)

    pred_endpoints_px = pred_px[:, [0, -1]]
    endpoint_forward = torch.linalg.norm(pred_endpoints_px - tgt_px[:, [0, -1]], dim=-1).mean(dim=1)
    endpoint_reverse = torch.linalg.norm(pred_endpoints_px - tgt_rev_px[:, [0, -1]], dim=-1).mean(dim=1)

    use_reverse = (ctrl_reverse + endpoint_reverse) < (ctrl_forward + endpoint_forward)
    oriented_tgt_ext = torch.where(use_reverse[:, None, None], tgt_rev_ext, tgt_curves_ext)
    oriented_tgt_px = torch.where(use_reverse[:, None, None], tgt_rev_px, tgt_px)
    return oriented_tgt_ext, oriented_tgt_px, pred_px


def _sample_curve_mean_px_error(
    pred_curves_ext: torch.Tensor,
    tgt_curves_ext: torch.Tensor,
    image_size: int,
    num_samples: int,
) -> torch.Tensor:
    pred_samples = sample_bezier_curves_torch(pred_curves_ext, num_samples=num_samples) * image_size
    tgt_samples = sample_bezier_curves_torch(tgt_curves_ext, num_samples=num_samples) * image_size
    tgt_rev_samples = torch.flip(tgt_samples, dims=(-2,))
    forward = torch.linalg.norm(pred_samples - tgt_samples, dim=-1).mean(dim=1)
    reverse = torch.linalg.norm(pred_samples - tgt_rev_samples, dim=-1).mean(dim=1)
    return torch.minimum(forward, reverse)


def _tensor_stats(values: List[torch.Tensor]) -> Dict[str, float]:
    merged = torch.cat(values, dim=0) if values else torch.empty(0)
    if merged.numel() == 0:
        return {'count': 0}
    return {
        'count': int(merged.numel()),
        'mean_px': float(merged.mean().item()),
        'median_px': float(merged.median().item()),
        'p90_px': float(torch.quantile(merged, 0.9).item()),
        'max_px': float(merged.max().item()),
        'min_px': float(merged.min().item()),
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--override-config', default=None)
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    config = load_config(args.config, args.override_config)
    seed = int(config.get('trainer', {}).get('seed', 20260325))
    pl.seed_everything(seed, workers=True)

    datamodule = ParametricEdgeDataModule(config)
    datamodule.setup('fit')
    if args.split == 'train':
        loader = datamodule.train_dataloader()
    elif args.split == 'test':
        loader = datamodule.test_dataloader()
    else:
        loader = datamodule.val_dataloader()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = build_model(config).to(device)
    model.eval()
    matcher = HungarianCurveMatcher.from_config(config)

    image_size = int(config['data']['image_size'])
    loss_cfg = config['loss']
    endpoint_errors = []
    ctrl_errors = []
    sample_errors = []
    per_sample = []

    for batch in loader:
        images = batch['images'].to(device)
        targets = batch['targets']
        outputs = model(images, targets=targets)
        indices = matcher(
            logits=outputs['pred_logits'],
            curves=outputs['pred_curves'],
            targets=targets,
        )

        pred_curves_ext = curve_internal_to_external(outputs['pred_curves'], config)

        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            pred = pred_curves_ext[batch_idx, src_idx].detach().cpu()
            tgt = targets[batch_idx]['curves'][tgt_idx.cpu()].detach().cpu()
            oriented_tgt_ext, oriented_tgt_px, pred_px = _align_target_to_prediction(pred, tgt, image_size=image_size)

            endpoint_err = torch.linalg.norm(pred_px[:, [0, -1]] - oriented_tgt_px[:, [0, -1]], dim=-1).mean(dim=1)
            if pred_px.shape[1] > 2:
                ctrl_pred_px = pred_px[:, 1:-1]
                ctrl_tgt_px = oriented_tgt_px[:, 1:-1]
                ctrl_err = torch.linalg.norm(ctrl_pred_px - ctrl_tgt_px, dim=-1).mean(dim=1)
            else:
                ctrl_err = torch.empty((pred_px.shape[0],), dtype=pred_px.dtype)
            sample_err = _sample_curve_mean_px_error(
                pred_curves_ext=pred,
                tgt_curves_ext=oriented_tgt_ext,
                image_size=image_size,
                num_samples=int(loss_cfg.get('num_curve_samples', 16)),
            )

            endpoint_errors.append(endpoint_err)
            if ctrl_err.numel() > 0:
                ctrl_errors.append(ctrl_err)
            sample_errors.append(sample_err)

            sample_id = targets[batch_idx].get('sample_id', f'batch_{batch_idx}')
            per_sample.append({
                'sample_id': sample_id,
                'matched_count': int(src_idx.numel()),
                'endpoint_mean_px': float(endpoint_err.mean().item()),
                'ctrl_mean_px': float(ctrl_err.mean().item()) if ctrl_err.numel() > 0 else None,
                'sample_mean_px': float(sample_err.mean().item()),
            })

    summary = {
        'config': args.config,
        'override_config': args.override_config,
        'split': args.split,
        'seed': seed,
        'image_size': image_size,
        'matching': {
            'control_cost': float(loss_cfg.get('control_cost', 5.0)),
            'endpoint_cost': float(loss_cfg.get('endpoint_cost', 5.0)),
            'sample_cost': float(loss_cfg.get('sample_cost', 0.0)),
            'curve_distance_cost': float(loss_cfg.get('curve_distance_cost', 0.0)),
            'num_curve_samples': int(loss_cfg.get('num_curve_samples', 16)),
        },
        'endpoint_error_px': _tensor_stats(endpoint_errors),
        'control_error_px': _tensor_stats(ctrl_errors),
        'sampled_curve_error_px': _tensor_stats(sample_errors),
        'per_sample': per_sample,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
