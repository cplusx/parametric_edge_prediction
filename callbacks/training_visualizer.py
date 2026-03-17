from pathlib import Path
from typing import List

import pytorch_lightning as pl
import torch

from misc_utils.visualization_utils import render_curve_grid
from models.matcher import hungarian_curve_matching


class ParametricEdgeVisualizer(pl.Callback):
    def __init__(self, every_n_epochs: int = 1, max_score_curves: int = 24, score_threshold: float = 0.3) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.max_score_curves = max_score_curves
        self.score_threshold = score_threshold

    @staticmethod
    def _wandb_log_image(trainer, key: str, image_path: Path, caption: str) -> None:
        image_path = Path(image_path)
        if not image_path.exists():
            return
        loggers = getattr(trainer, 'loggers', None)
        if loggers is None:
            single_logger = getattr(trainer, 'logger', None)
            loggers = [] if single_logger is None else [single_logger]
        for logger in loggers:
            if logger is None or logger.__class__.__name__ != 'WandbLogger':
                continue
            import wandb

            logger.experiment.log({
                key: wandb.Image(str(image_path), caption=caption),
                'trainer/current_epoch': trainer.current_epoch,
            })

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.sanity_checking:
            return
        if batch_idx != 0:
            return
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        with torch.no_grad():
            predictions = pl_module(batch['images'], targets=batch['targets'])
        probs = predictions['pred_logits'].softmax(-1)[..., 0]
        scored_curves: List[torch.Tensor] = []
        for pred_curves, pred_keep in zip(predictions['pred_curves'], probs > self.score_threshold):
            scored_curves.append(pred_curves[pred_keep][: self.max_score_curves])
        matched_indices = hungarian_curve_matching(
            predictions['pred_logits'],
            predictions['pred_curves'],
            batch['targets'],
            control_cost=float(pl_module.config['loss'].get('control_cost', 5.0)),
            sample_cost=float(pl_module.config['loss'].get('sample_cost', 2.0)),
            box_cost=float(pl_module.config['loss'].get('box_cost', 1.0)),
            giou_cost=float(pl_module.config['loss'].get('giou_cost', 1.0)),
            curve_distance_cost=float(pl_module.config['loss'].get('curve_distance_cost', 1.0)),
            curve_match_point_count=int(pl_module.config['loss'].get('curve_match_point_count', 4)),
            num_curve_samples=int(pl_module.config['loss'].get('num_curve_samples', 16)),
        )
        matched_curves: List[torch.Tensor] = []
        for batch_id, (src_idx, _) in enumerate(matched_indices):
            matched_curves.append(predictions['pred_curves'][batch_id, src_idx])
        vis_dir = Path(trainer.default_root_dir) / 'visualizations'
        score_path = vis_dir / f'epoch_{trainer.current_epoch:03d}_scores.jpg'
        matched_path = vis_dir / f'epoch_{trainer.current_epoch:03d}_matched.jpg'
        render_curve_grid(batch['images'], batch['targets'], scored_curves, score_path)
        render_curve_grid(batch['images'], batch['targets'], matched_curves, matched_path)
        self._wandb_log_image(trainer, 'visualizations/scored_curves', score_path, f'epoch={trainer.current_epoch}')
        self._wandb_log_image(trainer, 'visualizations/matched_curves', matched_path, f'epoch={trainer.current_epoch}')
