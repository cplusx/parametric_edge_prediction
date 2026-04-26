from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge_datasets import build_datamodule
from misc_utils.config_utils import load_config
from misc_utils.visualization_utils import PALETTE, draw_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Render conditioned-curve dataset samples.')
    parser.add_argument('--config', default='configs/parametric_edge/laion_curve_pretrain_lab34_v3_edgeprob05.yaml')
    parser.add_argument('--mode', choices=['eval', 'train_aug'], default='train_aug')
    parser.add_argument('--num-samples', type=int, default=8)
    parser.add_argument('--output-dir', default='outputs/conditioned_curve_debug')
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
    config.setdefault('trainer', {})
    config['trainer']['devices'] = 1
    config['trainer']['accumulate_grad_batches'] = 1
    config['trainer']['effective_batch_size'] = int(batch_size)
    return config


def _to_image_hwc(image_chw: np.ndarray) -> np.ndarray:
    if image_chw.shape[0] == 1:
        return np.repeat(image_chw[0][..., None], 3, axis=2)
    return np.transpose(image_chw, (1, 2, 0))


def _setup_axis(ax, image_np: np.ndarray) -> tuple[int, int]:
    ax.imshow(image_np, vmin=0.0, vmax=1.0)
    h, w = image_np.shape[:2]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    return h, w


def _draw_points(ax, points: np.ndarray, image_w: int, image_h: int) -> None:
    if points.size == 0:
        return
    pts = points.astype(np.float32).copy()
    pts[:, 0] *= float(image_w)
    pts[:, 1] *= float(image_h)
    for idx, point in enumerate(pts):
        ax.scatter(
            point[0],
            point[1],
            s=28.0,
            c=PALETTE[idx % len(PALETTE)],
            edgecolors='black',
            linewidths=0.5,
            zorder=3,
        )


def _draw_curves(ax, curves: np.ndarray, image_w: int, image_h: int) -> None:
    for idx, curve in enumerate(curves):
        draw_curve(ax, curve, image_w, image_h, PALETTE[idx % len(PALETTE)], linewidth=2.0, show_control_points=True)


def main() -> None:
    args = parse_args()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _prepare_config(args.config, args.mode, args.num_samples)
    datamodule = build_datamodule(config)
    datamodule.setup('fit')
    loader = datamodule.train_dataloader() if args.mode == 'train_aug' else datamodule.val_dataloader()
    batch = next(iter(loader))

    images = batch['images'].detach().cpu().numpy()
    targets = batch['targets']
    model_inputs = batch.get('model_inputs') or {}
    condition_points = model_inputs.get('condition_points')
    if condition_points is None:
        raise KeyError('missing model_inputs.condition_points in conditioned curve batch')
    condition_points = condition_points.detach().cpu().numpy()
    padding_mask = model_inputs.get('condition_padding_mask')
    padding_mask_np = padding_mask.detach().cpu().numpy() if padding_mask is not None else None

    sample_count = min(int(args.num_samples), len(targets))
    fig, axes = plt.subplots(sample_count, 4, figsize=(16, 4 * sample_count))
    if sample_count == 1:
        axes = np.asarray([axes])

    summaries = []
    for idx in range(sample_count):
        image_np = _to_image_hwc(images[idx])
        target = targets[idx]
        curves = target['curves'].detach().cpu().numpy()
        points = condition_points[idx]
        if padding_mask_np is not None and padding_mask_np.shape[1] > 0:
            points = points[~padding_mask_np[idx]]

        h, w = _setup_axis(axes[idx, 0], image_np)
        axes[idx, 0].set_title('Input')

        _setup_axis(axes[idx, 1], image_np)
        _draw_points(axes[idx, 1], points, w, h)
        axes[idx, 1].set_title(f'Condition points ({points.shape[0]})')

        _setup_axis(axes[idx, 2], image_np)
        _draw_curves(axes[idx, 2], curves, w, h)
        axes[idx, 2].set_title(f'Target curves ({curves.shape[0]})')

        _setup_axis(axes[idx, 3], image_np)
        _draw_curves(axes[idx, 3], curves, w, h)
        _draw_points(axes[idx, 3], points, w, h)
        sample_id = str(target['sample_id'])
        axes[idx, 3].set_title(sample_id)

        summaries.append({
            'sample_id': sample_id,
            'input_path': str(target['input_path']),
            'bezier_path': str(target['bezier_path']),
            'num_curves': int(curves.shape[0]),
            'num_condition_points': int(points.shape[0]),
            'image_size': target['image_size'].detach().cpu().tolist(),
        })

    fig.tight_layout()
    image_path = output_dir / f'conditioned_curve_{args.mode}.jpg'
    fig.savefig(image_path, dpi=180, format='jpg')
    plt.close(fig)

    summary = {
        'config': args.config,
        'mode': args.mode,
        'sample_count': sample_count,
        'image_path': str(image_path),
        'samples': summaries,
    }
    summary_path = output_dir / f'conditioned_curve_{args.mode}_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
