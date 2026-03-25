from pathlib import Path
from typing import List

import pytorch_lightning as pl
import torch

from misc_utils.visualization_utils import render_curve_grid
from models.curve_coordinates import curve_internal_to_external
from models.matcher import hungarian_curve_matching


class ParametricEdgeVisualizer(pl.Callback):
    def __init__(
        self,
        val_every_n_epochs: int = 1,
        train_every_n_steps: int = 0,
        max_score_curves: int = 24,
        score_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        self.val_every_n_epochs = val_every_n_epochs
        self.train_every_n_steps = train_every_n_steps
        self.max_score_curves = max_score_curves
        self.score_threshold = score_threshold
        self.train_batches_seen = 0

    @staticmethod
    def _wandb_log_image(trainer, key: str, image_path: Path, caption: str, global_step: int, epoch: int) -> None:
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
                'trainer/current_epoch': epoch,
                'trainer/global_step': global_step,
            })

    def _predict_curves(self, pl_module, batch):
        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            predictions = pl_module(batch['images'], targets=batch['targets'])
        if was_training:
            pl_module.train()

        probs = predictions['pred_logits'].softmax(-1)[..., 0]
        scored_curves: List[torch.Tensor] = []
        for pred_curves, pred_keep in zip(predictions['pred_curves'], probs > self.score_threshold):
            scored_curves.append(curve_internal_to_external(pred_curves[pred_keep][: self.max_score_curves], pl_module.config))

        matched_indices = hungarian_curve_matching(
            predictions['pred_logits'],
            predictions['pred_curves'],
            batch['targets'],
            control_cost=float(pl_module.config['loss'].get('control_cost', 5.0)),
            sample_cost=float(pl_module.config['loss'].get('sample_cost', 2.0)),
            curve_distance_cost=float(pl_module.config['loss'].get('curve_distance_cost', 0.0)),
            curve_match_point_count=int(pl_module.config['loss'].get('curve_match_point_count', 4)),
            num_curve_samples=int(pl_module.config['loss'].get('num_curve_samples', 16)),
            direction_invariant=bool(pl_module.config['loss'].get('direction_invariant', True)),
            config=pl_module.config,
        )
        matched_curves: List[torch.Tensor] = []
        for batch_id, (src_idx, _) in enumerate(matched_indices):
            matched_curves.append(curve_internal_to_external(predictions['pred_curves'][batch_id, src_idx], pl_module.config))
        return scored_curves, matched_curves

    def _render_batch(self, trainer, batch, scored_curves, matched_curves, split: str, token: str) -> None:
        vis_dir = Path(trainer.default_root_dir) / 'visualizations'
        score_path = vis_dir / f'{split}_{token}_scores.jpg'
        matched_path = vis_dir / f'{split}_{token}_matched.jpg'
        render_curve_grid(batch['images'], batch['targets'], scored_curves, score_path)
        render_curve_grid(batch['images'], batch['targets'], matched_curves, matched_path)
        caption = f'{split}:{token}'
        self._wandb_log_image(
            trainer,
            f'visualizations/{split}_scored_curves',
            score_path,
            caption,
            global_step=trainer.global_step,
            epoch=trainer.current_epoch,
        )
        self._wandb_log_image(
            trainer,
            f'visualizations/{split}_matched_curves',
            matched_path,
            caption,
            global_step=trainer.global_step,
            epoch=trainer.current_epoch,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        if self.train_every_n_steps <= 0:
            return
        self.train_batches_seen += 1
        if self.train_batches_seen % self.train_every_n_steps != 0:
            return
        scored_curves, matched_curves = self._predict_curves(pl_module, batch)
        self._render_batch(trainer, batch, scored_curves, matched_curves, 'train', f'step_{self.train_batches_seen:07d}')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        if batch_idx != 0:
            return
        if self.val_every_n_epochs <= 0:
            return
        if trainer.current_epoch % self.val_every_n_epochs != 0:
            return
        scored_curves, matched_curves = self._predict_curves(pl_module, batch)
        self._render_batch(trainer, batch, scored_curves, matched_curves, 'val', f'epoch_{trainer.current_epoch:03d}')
