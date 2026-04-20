from typing import Dict, List, Optional, Tuple

import torch

from models.curve_coordinates import curve_external_to_internal
from models.losses.base import BaseLossComponent
from models.losses.matched import ClassificationLoss


class MatchedPointLoss(BaseLossComponent):
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.classification = ClassificationLoss(config)

    def __call__(
        self,
        pred_points: torch.Tensor,
        pred_logits: torch.Tensor,
        targets: List[dict],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        runtime_outputs: Dict[str, torch.Tensor],
        query_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del runtime_outputs
        loss_cfg = self.config['loss']
        device = pred_logits.device
        target_classes = torch.full(pred_logits.shape[:2], 1, dtype=torch.long, device=device)

        matched_pred_points = []
        matched_tgt_points = []
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            target_classes[batch_idx, src_idx] = 0
            matched_pred_points.append(pred_points[batch_idx, src_idx])
            matched_tgt_points.append(curve_external_to_internal(targets[batch_idx]['points'][tgt_idx].to(device), self.config))

        classification_terms = self.classification.compute(pred_logits, target_classes, query_weights=query_weights)
        class_loss_name = self.classification.loss_name()
        classification_loss = classification_terms[class_loss_name]
        if matched_pred_points:
            matched_pred_points = torch.cat(matched_pred_points, dim=0)
            matched_tgt_points = torch.cat(matched_tgt_points, dim=0)
            point_abs = torch.abs(matched_pred_points - matched_tgt_points).mean(dim=1)
            loss_point = point_abs.mean()
        else:
            loss_point = pred_points.sum() * 0.0

        total = (
            float(loss_cfg.get('ce_weight', 1.0)) * classification_loss
            + float(loss_cfg.get('point_weight', 5.0)) * loss_point
        )
        return {
            'loss_total': total,
            **classification_terms,
            'loss_point': loss_point,
        }
