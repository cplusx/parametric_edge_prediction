from pathlib import Path
from typing import Iterable, Sequence

import pytorch_lightning as pl
import torch

from misc_utils.visualization_utils import render_point_grid
from models.pipelines.endpoint_flow_pipeline import EndpointFlowPipeline


class EndpointFlowVisualizer(pl.Callback):
    def __init__(
        self,
        val_every_n_epochs: int = 1,
        inference_steps: int = 20,
        guidance_scales: Sequence[float] = (1.0, 3.0, 5.0, 7.0),
    ) -> None:
        super().__init__()
        self.val_every_n_epochs = int(val_every_n_epochs)
        self.inference_steps = int(inference_steps)
        self.guidance_scales = tuple(float(scale) for scale in guidance_scales)

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

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        if batch_idx != 0 or self.val_every_n_epochs <= 0:
            return
        completed_epoch = trainer.current_epoch + 1
        if completed_epoch % self.val_every_n_epochs != 0:
            return
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
                self._render_one(
                    trainer,
                    batch,
                    [points.detach().cpu() for points in result.selected_points],
                    'val',
                    f'epoch_{completed_epoch:03d}',
                    guidance_scale,
                )
        if was_training:
            pl_module.train()
