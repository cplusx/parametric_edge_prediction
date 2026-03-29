from typing import Dict

import torch
import torch.nn.functional as F

from models.curve_coordinates import curve_external_to_internal
from models.losses.base import BaseLossComponent


class EndpointDenoisingLoss(BaseLossComponent):
    def __call__(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pred_points = outputs.get('pred_points')
        if 'dn_pred_points' not in outputs or 'dn_meta' not in outputs:
            zero = pred_points.sum() * 0.0 if pred_points is not None else torch.tensor(0.0)
            return {'loss_dn': zero, 'loss_dn_ce': zero, 'loss_dn_point': zero}

        dn_meta = outputs['dn_meta']
        device = outputs['dn_pred_points'].device
        point_mask = dn_meta['mask'].to(device)
        gt_points = curve_external_to_internal(dn_meta['points'].to(device), self.config)
        if point_mask.any():
            loss_point = F.l1_loss(outputs['dn_pred_points'][point_mask], gt_points[point_mask])
        else:
            loss_point = outputs['dn_pred_points'].sum() * 0.0

        zero = outputs['dn_pred_points'].sum() * 0.0
        total = float(self.config['loss'].get('dn_weight', 1.0)) * (
            zero + float(self.config['loss'].get('dn_point_weight', 5.0)) * loss_point
        )
        return {
            'loss_dn': total,
            'loss_dn_ce': zero,
            'loss_dn_point': loss_point,
        }
