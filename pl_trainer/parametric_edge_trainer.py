from typing import Dict

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, MultiStepLR, SequentialLR

from misc_utils.endpoint_flow_utils import scale_points_from_flow
from models import build_model
from models.losses import compute_losses


class ParametricEdgeLightningModule(pl.LightningModule):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        self.model = build_model(config)
        self.use_channels_last = bool(config.get('trainer', {}).get('channels_last', False))
        self._last_flow_debug_step = -1
        if self.use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.save_hyperparameters(config)

    def forward(self, images: torch.Tensor, targets=None):
        if self.use_channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        return self.model(images, targets=targets)

    def on_train_epoch_start(self) -> None:
        self.model.set_epoch(self.current_epoch)
        datamodule = getattr(self.trainer, 'datamodule', None)
        if datamodule is not None and hasattr(datamodule, 'set_epoch'):
            datamodule.set_epoch(self.current_epoch)

    def _maybe_print_endpoint_flow_debug(self, outputs: Dict, stage: str) -> None:
        if stage != 'train' or not self.trainer.is_global_zero:
            return
        debug_cfg = self.config.get('debug', {})
        every_n = int(debug_cfg.get('flow_target_print_every_n_steps', 0))
        if every_n <= 0:
            return
        step = int(self.global_step)
        if step % every_n != 0:
            return
        if step == self._last_flow_debug_step:
            return
        self._last_flow_debug_step = step
        max_points = int(debug_cfg.get('flow_target_print_max_points', 8))
        valid_mask = outputs['valid_mask'][0].detach().cpu().bool()
        if valid_mask.numel() == 0 or not bool(valid_mask.any()):
            print(f'[flow-debug] step={step} no valid matched points')
            return
        image_size = outputs['image_sizes'][0].detach().cpu()
        width = float(image_size[1].item())
        height = float(image_size[0].item())
        point_scale = torch.tensor([width, height], dtype=torch.float32)
        vel_scale = point_scale * 0.5

        source_px = scale_points_from_flow(outputs['source_points'][0].detach().cpu()[valid_mask]) * point_scale
        noisy_px = scale_points_from_flow(outputs['noisy_points'][0].detach().cpu()[valid_mask]) * point_scale
        target_px = scale_points_from_flow(outputs['target_points'][0].detach().cpu()[valid_mask]) * point_scale
        target_vel_px = outputs['target_velocity'][0].detach().cpu()[valid_mask] * vel_scale
        pred_vel_px = outputs['pred_velocity'][0].detach().cpu()[valid_mask] * vel_scale
        timestep = int(outputs['timestep_indices'][0].detach().cpu().item())

        print(f'[flow-debug] step={step} t={timestep} valid={int(valid_mask.sum().item())} size=({int(height)},{int(width)})')
        limit = min(max_points, int(source_px.shape[0]))
        for idx in range(limit):
            src = source_px[idx].tolist()
            noisy = noisy_px[idx].tolist()
            tgt = target_px[idx].tolist()
            tgt_vel = target_vel_px[idx].tolist()
            pred_vel = pred_vel_px[idx].tolist()
            print(
                f'  idx={idx:02d} src=({src[0]:.1f},{src[1]:.1f}) '
                f'noisy=({noisy[0]:.1f},{noisy[1]:.1f}) '
                f'tgt=({tgt[0]:.1f},{tgt[1]:.1f}) '
                f'gt_vel=({tgt_vel[0]:.1f},{tgt_vel[1]:.1f}) '
                f'pred_vel=({pred_vel[0]:.1f},{pred_vel[1]:.1f})'
            )

    def _shared_step(self, batch: Dict, stage: str) -> torch.Tensor:
        self.model.set_epoch(self.current_epoch)
        outputs = self(batch['images'], targets=batch['targets'])
        self._maybe_print_endpoint_flow_debug(outputs, stage)
        sync_dist = stage != 'train'
        if 'pred_group_count' in outputs:
            self.log(f'{stage}/group_count', float(outputs['pred_group_count']), batch_size=batch['images'].shape[0], sync_dist=sync_dist)
        losses = compute_losses(outputs, batch['targets'], self.config)
        self.log(f'{stage}/loss', losses['loss'], prog_bar=True, batch_size=batch['images'].shape[0], sync_dist=sync_dist)
        if stage == 'val':
            self.log('val_loss', losses['loss'], prog_bar=False, batch_size=batch['images'].shape[0], sync_dist=True)
            if 'loss_main' in losses:
                self.log('val_loss_main', losses['loss_main'], prog_bar=False, batch_size=batch['images'].shape[0], sync_dist=True)
        for key, value in losses.items():
            if key in {'loss', 'matching'}:
                continue
            self.log(f'{stage}/{key}', value, batch_size=batch['images'].shape[0], sync_dist=sync_dist)
        return losses['loss']

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, 'train')

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, 'val')

    def test_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, 'test')

    def configure_optimizers(self):
        opt_cfg = self.config['optimizer']
        base_lr = float(opt_cfg['lr'])
        backbone_lr = float(opt_cfg.get('backbone_lr', base_lr))
        weight_decay = float(opt_cfg.get('weight_decay', 1e-4))
        beta1 = float(opt_cfg.get('beta1', 0.9))
        beta2 = float(opt_cfg.get('beta2', 0.999))

        backbone_params = []
        other_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('model.backbone.'):
                backbone_params.append(param)
            else:
                other_params.append(param)

        param_groups = []
        if other_params:
            param_groups.append({'params': other_params, 'lr': base_lr, 'weight_decay': weight_decay})
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay})

        optimizer = torch.optim.AdamW(param_groups, betas=(beta1, beta2))

        max_epochs = int(self.config['trainer'].get('max_epochs', 1))
        warmup_epochs = int(opt_cfg.get('warmup_epochs', 0))
        scheduler_type = str(opt_cfg.get('scheduler', 'multistep')).lower()
        min_lr_ratio = float(opt_cfg.get('min_lr_ratio', 0.01))

        if scheduler_type == 'cosine':
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(1, max_epochs - warmup_epochs),
                eta_min=base_lr * min_lr_ratio,
            )
        else:
            milestones = opt_cfg.get('lr_milestones')
            if milestones is None:
                milestones = [max(1, int(max_epochs * 0.7)), max(1, int(max_epochs * 0.9))]
            main_scheduler = MultiStepLR(
                optimizer,
                milestones=[int(milestone) for milestone in milestones],
                gamma=float(opt_cfg.get('lr_gamma', 0.1)),
            )

        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=float(opt_cfg.get('warmup_start_factor', 0.1)),
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = main_scheduler

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            },
        }
