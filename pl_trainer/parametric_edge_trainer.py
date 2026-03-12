from typing import Dict

import pytorch_lightning as pl
import torch

from models.losses import compute_losses
from models.parametric_detr import ParametricDETR


class ParametricEdgeLightningModule(pl.LightningModule):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        self.model = ParametricDETR(config)
        self.save_hyperparameters(config)

    def forward(self, images: torch.Tensor, targets=None):
        return self.model(images, targets=targets)

    def _shared_step(self, batch: Dict, stage: str) -> torch.Tensor:
        self.model.set_epoch(self.current_epoch)
        outputs = self(batch['images'], targets=batch['targets'])
        losses = compute_losses(outputs, batch['targets'], self.config)
        self.log(f'{stage}/loss', losses['loss'], prog_bar=True, batch_size=batch['images'].shape[0])
        for key, value in losses.items():
            if key in {'loss', 'matching'}:
                continue
            self.log(f'{stage}/{key}', value, batch_size=batch['images'].shape[0])
        return losses['loss']

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, 'train')

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, 'val')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.config['optimizer']['lr']),
            weight_decay=float(self.config['optimizer'].get('weight_decay', 1e-4)),
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(self.config['optimizer'].get('lr_step_size', 50)),
            gamma=float(self.config['optimizer'].get('lr_gamma', 0.5)),
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
