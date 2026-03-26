import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

from edge_datasets.parametric_edge_datamodule import ParametricEdgeDataModule
from misc_utils.config_utils import load_config
from misc_utils.visualization_utils import PALETTE, draw_curve
from models.curve_coordinates import curve_internal_to_external
from pl_trainer.parametric_edge_trainer import ParametricEdgeLightningModule


def _prepare_display_image(image: torch.Tensor) -> tuple[np.ndarray, str | None]:
    image_np = image.detach().cpu().numpy()
    if image_np.shape[0] == 1:
        return image_np[0], 'gray'
    return np.transpose(image_np, (1, 2, 0)), None


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--max-samples', type=int, default=4)
    parser.add_argument('--score-threshold', type=float, default=0.5)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    config = load_config(args.config)
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
    batch = next(iter(loader))

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    module = ParametricEdgeLightningModule.load_from_checkpoint(
        args.checkpoint,
        config=config,
        map_location=device,
    )
    module.to(device)
    module.eval()

    images = batch['images'][: args.max_samples].to(device)
    targets = batch['targets'][: args.max_samples]
    outputs = module(images, targets=targets)

    layers = []
    aux_outputs = outputs.get('aux_outputs', [])
    for idx, aux in enumerate(aux_outputs):
        layers.append((f'aux_{idx}', aux))
    layers.append(('main', {'pred_logits': outputs['pred_logits'], 'pred_curves': outputs['pred_curves']}))

    num_rows = len(layers)
    num_cols = len(targets)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4.2 * num_cols, 4.2 * num_rows))
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_idx, (layer_name, layer_outputs) in enumerate(layers):
        probs = layer_outputs['pred_logits'].softmax(-1)[..., 0]
        pred_curves_ext = curve_internal_to_external(layer_outputs['pred_curves'], config)
        for col_idx in range(num_cols):
            ax = axes[row_idx, col_idx]
            image_np, cmap = _prepare_display_image(images[col_idx].cpu())
            ax.imshow(image_np, cmap=cmap, vmin=0.0, vmax=1.0)
            ax.axis('off')
            height, width = image_np.shape[:2]
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
            ax.set_aspect('equal', adjustable='box')

            target_curves = targets[col_idx]['curves'].detach().cpu().numpy()
            for idx, curve in enumerate(target_curves):
                draw_curve(ax, curve, width, height, PALETTE[idx % len(PALETTE)], linewidth=2.2)

            keep = probs[col_idx] > args.score_threshold
            kept_curves = pred_curves_ext[col_idx, keep].detach().cpu().numpy()
            for idx, curve in enumerate(kept_curves):
                draw_curve(ax, curve, width, height, '#ffffff', linewidth=1.8, show_control_points=False)

            sample_id = targets[col_idx].get('sample_id', f'sample_{col_idx}')
            if row_idx == 0:
                ax.set_title(sample_id, fontsize=10)
            if col_idx == 0:
                ax.text(
                    -0.02,
                    0.5,
                    layer_name,
                    transform=ax.transAxes,
                    rotation=90,
                    va='center',
                    ha='right',
                    fontsize=12,
                    fontweight='bold',
                    color='black',
                )

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(output_path)


if __name__ == '__main__':
    main()
