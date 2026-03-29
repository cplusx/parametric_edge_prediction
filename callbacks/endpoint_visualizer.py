from pathlib import Path
from typing import List

import pytorch_lightning as pl
import torch

from misc_utils.visualization_utils import render_point_grid
from models.curve_coordinates import curve_internal_to_external
from models.endpoint_matcher import HungarianPointMatcher


class ParametricEndpointVisualizer(pl.Callback):
    def __init__(
        self,
        val_every_n_epochs: int = 1,
        train_every_n_steps: int = 0,
        max_score_points: int = 64,
        score_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self.val_every_n_epochs = val_every_n_epochs
        self.train_every_n_steps = train_every_n_steps
        self.max_score_points = max_score_points
        self.score_threshold = score_threshold
        self.train_batches_seen = 0

    def _predict_points(self, pl_module, batch):
        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            predictions = pl_module(batch['images'], targets=batch['targets'])
        if was_training:
            pl_module.train()

        probs = predictions['pred_logits'].softmax(-1)[..., 0]
        scored_points: List[torch.Tensor] = []
        for pred_points, pred_keep in zip(predictions['pred_points'], probs > self.score_threshold):
            scored_points.append(curve_internal_to_external(pred_points[pred_keep][: self.max_score_points], pl_module.config))

        matcher = HungarianPointMatcher.from_config(pl_module.config)
        matched_indices = matcher(
            logits=predictions['pred_logits'],
            points=predictions['pred_points'],
            targets=batch['targets'],
        )
        matched_points: List[torch.Tensor] = []
        for batch_id, (src_idx, _) in enumerate(matched_indices):
            matched_points.append(curve_internal_to_external(predictions['pred_points'][batch_id, src_idx], pl_module.config))
        return scored_points, matched_points

    def _render_batch(self, trainer, batch, scored_points, matched_points, split: str, token: str) -> None:
        vis_dir = Path(trainer.default_root_dir) / 'visualizations'
        score_path = vis_dir / f'{split}_{token}_scores.jpg'
        matched_path = vis_dir / f'{split}_{token}_matched.jpg'
        render_point_grid(batch['images'], batch['targets'], scored_points, score_path)
        render_point_grid(batch['images'], batch['targets'], matched_points, matched_path)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        if self.train_every_n_steps <= 0:
            return
        self.train_batches_seen += 1
        if self.train_batches_seen % self.train_every_n_steps != 0:
            return
        scored_points, matched_points = self._predict_points(pl_module, batch)
        self._render_batch(trainer, batch, scored_points, matched_points, 'train', f'step_{self.train_batches_seen:07d}')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        if batch_idx != 0:
            return
        if self.val_every_n_epochs <= 0:
            return
        completed_epoch = trainer.current_epoch + 1
        if completed_epoch % self.val_every_n_epochs != 0:
            return
        scored_points, matched_points = self._predict_points(pl_module, batch)
        self._render_batch(trainer, batch, scored_points, matched_points, 'val', f'epoch_{completed_epoch:03d}')
