from pathlib import Path
from typing import Iterable, Sequence

import pytorch_lightning as pl
import torch

from misc_utils.visualization_utils import render_flow_training_grid, render_point_grid, render_point_trajectory_animation
from models.pipelines.endpoint_flow_pipeline import EndpointFlowPipeline


class EndpointFlowVisualizer(pl.Callback):
    def __init__(
        self,
        val_every_n_epochs: int = 1,
        train_every_n_steps: int = 1000,
        val_every_n_steps: int | None = None,
        inference_steps: int = 20,
        guidance_scales: Sequence[float] = (1.0, 3.0, 5.0, 7.0),
    ) -> None:
        super().__init__()
        self.val_every_n_epochs = int(val_every_n_epochs)
        self.train_every_n_steps = int(train_every_n_steps)
        self.val_every_n_steps = int(val_every_n_steps) if val_every_n_steps is not None else int(train_every_n_steps)
        self.inference_steps = int(inference_steps)
        self.guidance_scales = tuple(float(scale) for scale in guidance_scales)
        self._last_val_render_step = -1
        self._last_train_render_step = -1

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

    def _build_pipeline(self, pl_module) -> EndpointFlowPipeline:
        pipeline = EndpointFlowPipeline(model=pl_module.model)
        pipeline = pipeline.to(device=pl_module.device)
        return pipeline

    def _render_one(self, trainer, batch, predicted_points, split: str, token: str, guidance_scale: float) -> None:
        vis_dir = Path(trainer.default_root_dir) / 'visualizations'
        output_path = vis_dir / f'{split}_{token}_cfg{int(guidance_scale):02d}.jpg'
        render_point_grid(
            batch['images'],
            batch['targets'],
            predicted_points,
            output_path,
            titles=('Input', 'GT Edge', 'Target Endpoints', f'Generated Endpoints (cfg={guidance_scale:g})', 'Generated Endpoints Only'),
        )
        caption = f'{split}:{token}:cfg{guidance_scale:g}'
        self._wandb_log_image(
            trainer,
            f'visualizations/{split}_flow_cfg_{guidance_scale:g}',
            output_path,
            caption,
            global_step=trainer.global_step,
            epoch=trainer.current_epoch,
        )

    @staticmethod
    def _wandb_log_video(trainer, key: str, media_path: Path, caption: str, global_step: int, epoch: int) -> None:
        media_path = Path(media_path)
        if not media_path.exists():
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
                key: wandb.Video(str(media_path), fps=5, format=media_path.suffix.lstrip('.')),
                'trainer/current_epoch': epoch,
                'trainer/global_step': global_step,
            })

    def _predict_train_flow(self, pl_module, batch):
        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            outputs = pl_module.model(batch['images'].to(pl_module.device), targets=batch['targets'])
        if was_training:
            pl_module.train()
        return outputs

    def _render_train_flow(self, trainer, batch, outputs, token: str) -> None:
        vis_dir = Path(trainer.default_root_dir) / 'visualizations'
        output_path = vis_dir / f'train_{token}_flow_vectors.jpg'
        lightning_module = getattr(trainer, 'lightning_module', None)
        num_train_timesteps = int(getattr(getattr(lightning_module, 'model', None), 'num_train_timesteps', 1000))
        render_flow_training_grid(
            batch['images'],
            batch['targets'],
            outputs['noisy_points'].detach().cpu(),
            outputs['target_velocity'].detach().cpu(),
            outputs['pred_velocity'].detach().cpu(),
            outputs['timestep_indices'].detach().cpu(),
            outputs['valid_mask'].detach().cpu().bool(),
            output_path,
            num_train_timesteps=num_train_timesteps,
            titles=('Input', 'GT Edge', 'Noise + GT Velocity', 'Noise + Pred Velocity'),
        )
        self._wandb_log_image(
            trainer,
            'visualizations/train_flow_vectors',
            output_path,
            f'train:{token}',
            global_step=trainer.global_step,
            epoch=trainer.current_epoch,
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        if batch_idx != 0:
            return
        if self.val_every_n_steps <= 0:
            return
        step = int(trainer.global_step)
        if step <= 0 or step % self.val_every_n_steps != 0:
            return
        if step == self._last_val_render_step:
            return
        self._last_val_render_step = step
        was_training = pl_module.training
        pl_module.eval()
        pipeline = self._build_pipeline(pl_module)
        with torch.no_grad():
            batch_max_points = max(max(int(target['points'].shape[0]), 1) for target in batch['targets'])
            for guidance_scale in self.guidance_scales:
                result = pipeline(
                    batch['images'].to(pl_module.device),
                    num_inference_steps=self.inference_steps,
                    num_points=batch_max_points,
                    guidance_scale=guidance_scale,
                )
                token = f'step_{step:07d}'
                self._render_one(
                    trainer,
                    batch,
                    [points.detach().cpu() for points in result.selected_points],
                    'val',
                    token,
                    guidance_scale,
                )
                vis_dir = Path(trainer.default_root_dir) / 'visualizations'
                gif_path = vis_dir / f'val_{token}_cfg{int(guidance_scale):02d}_trajectory.gif'
                mp4_path = vis_dir / f'val_{token}_cfg{int(guidance_scale):02d}_trajectory.mp4'
                render_point_trajectory_animation(
                    batch['images'],
                    batch['targets'],
                    result.trajectory.detach().cpu(),
                    gif_path,
                    output_mp4_path=mp4_path,
                    title=f'Validation Trajectory (cfg={guidance_scale:g})',
                    fps=5,
                )
                self._wandb_log_video(
                    trainer,
                    f'visualizations/val_flow_cfg_{guidance_scale:g}_trajectory',
                    gif_path,
                    f'val:{token}:cfg{guidance_scale:g}:trajectory',
                    global_step=trainer.global_step,
                    epoch=trainer.current_epoch,
                )
        if was_training:
            pl_module.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        if self.train_every_n_steps <= 0:
            return
        step = int(trainer.global_step)
        if step <= 0 or step % self.train_every_n_steps != 0:
            return
        if step == self._last_train_render_step:
            return
        self._last_train_render_step = step
        train_outputs = self._predict_train_flow(pl_module, batch)
        self._render_train_flow(trainer, batch, train_outputs, f'step_{step:07d}')
