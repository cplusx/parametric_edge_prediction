from typing import Dict

import torch
import torch.nn.functional as F

from models.curve_coordinates import curve_external_to_internal
from models.losses.base import BaseLossComponent
from models.losses.matched import ClassificationLoss


class DenoisingLoss(BaseLossComponent):
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.classification = ClassificationLoss(config)

    def __call__(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if 'dn_pred_logits' not in outputs or 'dn_meta' not in outputs:
            zero = outputs['pred_curves'].sum() * 0.0
            return {'loss_dn': zero, 'loss_dn_ce': zero, 'loss_dn_curve': zero}
        device = outputs['dn_pred_logits'].device
        dn_meta = outputs['dn_meta']
        logits = outputs['dn_pred_logits']
        curves = outputs['dn_pred_curves']
        labels = dn_meta['labels'].to(device)
        mask = dn_meta['mask'].to(device)
        curve_mask = dn_meta.get('curve_mask', mask).to(device)
        gt_curves = curve_external_to_internal(dn_meta['curves'].to(device), self.config)
        query_weights = mask.to(dtype=logits.dtype)
        loss_ce = self.classification(logits, labels, query_weights=query_weights)
        if curve_mask.any():
            pred_valid = curves[curve_mask]
            target_valid = gt_curves[curve_mask]
            loss_curve = F.l1_loss(pred_valid, target_valid)
        else:
            loss_curve = curves.sum() * 0.0
        total = float(self.config['loss'].get('dn_weight', 1.0)) * (loss_ce + float(self.config['loss'].get('dn_curve_weight', 5.0)) * loss_curve)
        return {'loss_dn': total, 'loss_dn_ce': loss_ce, 'loss_dn_curve': loss_curve}
