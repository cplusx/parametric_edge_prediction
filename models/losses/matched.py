from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from models.curve_coordinates import curve_external_to_internal
from models.curve_distances import (
    build_curve_distance_from_config,
    closed_curve_endpoint_dist_threshold_from_config,
    curve_loss_name_from_config,
    curve_loss_weight_from_config,
    infer_closed_curves,
)
from models.losses.base import BaseLossComponent, balanced_class_weights


def classification_loss_name_from_config(config: Dict) -> str:
    loss_type = str(config.get('loss', {}).get('class_loss_type', 'ce')).lower()
    return 'loss_focal' if loss_type == 'focal' else 'loss_ce'


class ClassificationLoss(BaseLossComponent):
    def loss_name(self) -> str:
        return classification_loss_name_from_config(self.config)

    @staticmethod
    def _masked_weighted_mean(
        values: torch.Tensor,
        mask: torch.Tensor,
        query_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if values.numel() == 0:
            return values.sum() * 0.0
        weights = mask.to(device=values.device, dtype=values.dtype)
        if query_weights is not None:
            weights = weights * query_weights.to(device=values.device, dtype=values.dtype)
        denom = weights.sum().clamp_min(1e-6)
        return (values * weights).sum() / denom

    @staticmethod
    def _weighted_mean(values: torch.Tensor, query_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        if values.numel() == 0:
            return values.sum() * 0.0
        if query_weights is None:
            return values.mean()
        weights = query_weights.to(device=values.device, dtype=values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1e-6)

    def compute(self, pred_logits: torch.Tensor, target_classes: torch.Tensor, query_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        loss_cfg = self.config['loss']
        dynamic_class_balance = bool(loss_cfg.get('dynamic_class_balance', True))
        loss_type = str(loss_cfg.get('class_loss_type', 'ce'))
        focal_alpha = float(loss_cfg.get('focal_alpha', 0.5))
        focal_gamma = float(loss_cfg.get('focal_gamma', 1.0))

        balance_weights = None
        balance_normalizer = None
        balance_active = None
        if dynamic_class_balance:
            balance_weights, balance_normalizer, balance_active = balanced_class_weights(
                target_classes,
                positive_class=0,
                query_weights=query_weights,
            )

        def reduce_per_sample(loss_values: torch.Tensor) -> torch.Tensor:
            if balance_weights is not None:
                per_sample = (loss_values * balance_weights).sum(dim=1) / balance_normalizer
                return per_sample[balance_active].mean() if bool(balance_active.any()) else loss_values.sum() * 0.0
            if query_weights is None:
                return loss_values.mean()
            query_weights_local = query_weights.to(loss_values.device, loss_values.dtype)
            return (loss_values * query_weights_local).sum() / query_weights_local.sum().clamp_min(1e-6)

        if loss_type == 'focal':
            object_logits = pred_logits[..., 0] - pred_logits[..., 1]
            target_object = (target_classes == 0).float()
            bce = F.binary_cross_entropy_with_logits(object_logits, target_object, reduction='none')
            prob = torch.sigmoid(object_logits)
            pt = torch.where(target_object > 0.5, prob, 1.0 - prob)
            alpha_t = torch.where(
                target_object > 0.5,
                torch.full_like(target_object, focal_alpha),
                torch.full_like(target_object, 1.0 - focal_alpha),
            )
            focal_loss = bce * alpha_t * (1.0 - pt).pow(focal_gamma)
            plain_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, reduction='none')
            return {
                self.loss_name(): reduce_per_sample(focal_loss),
                'loss_focal_posi': self._masked_weighted_mean(focal_loss, target_classes == 0, query_weights=query_weights),
                'loss_focal_nega': self._masked_weighted_mean(focal_loss, target_classes != 0, query_weights=query_weights),
                'loss_ce': self._weighted_mean(plain_ce, query_weights=query_weights),
            }

        no_object_weight = float(loss_cfg.get('no_object_weight', 0.2))
        class_weight = torch.tensor([1.0, no_object_weight], device=pred_logits.device)
        ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, weight=class_weight, reduction='none')
        return {self.loss_name(): reduce_per_sample(ce)}

    def __call__(
        self,
        pred_logits: torch.Tensor,
        target_classes: torch.Tensor,
        query_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.compute(pred_logits, target_classes, query_weights=query_weights)[self.loss_name()]


class MatchedCurveLoss(BaseLossComponent):
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.classification = ClassificationLoss(config)
        self.curve_loss_name = curve_loss_name_from_config(config)
        self.curve_distance = build_curve_distance_from_config(config, for_matching=False)
        self.closed_curve_endpoint_dist_threshold = closed_curve_endpoint_dist_threshold_from_config(config)

    def __call__(
        self,
        pred_curves: torch.Tensor,
        pred_logits: torch.Tensor,
        targets: List[dict],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        runtime_outputs: Dict[str, torch.Tensor],
        match_weights: Optional[List[torch.Tensor]] = None,
        query_weights: Optional[torch.Tensor] = None,
        loss_weight_overrides: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        loss_cfg = self.config['loss']
        weight_cfg = loss_cfg if loss_weight_overrides is None else {**loss_cfg, **loss_weight_overrides}
        device = pred_logits.device
        target_classes = torch.full(pred_logits.shape[:2], 1, dtype=torch.long, device=device)
        matched_pred_curves = []
        matched_tgt_curves = []
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            target_classes[batch_idx, src_idx] = 0
            matched_pred_curves.append(pred_curves[batch_idx, src_idx])
            matched_tgt_curves.append(curve_external_to_internal(targets[batch_idx]['curves'][tgt_idx].to(device), self.config))
        classification_terms = self.classification.compute(pred_logits, target_classes, query_weights=query_weights)
        class_loss_name = self.classification.loss_name()
        classification_loss = classification_terms[class_loss_name]
        if matched_pred_curves:
            matched_pred_curves = torch.cat(matched_pred_curves, dim=0)
            matched_tgt_curves = torch.cat(matched_tgt_curves, dim=0)
            target_is_closed = infer_closed_curves(
                matched_tgt_curves,
                threshold=self.closed_curve_endpoint_dist_threshold,
            )
            loss_curve = self.curve_distance.matched_cost_from_curves(
                matched_pred_curves,
                matched_tgt_curves,
                target_is_closed=target_is_closed,
            ).total.mean()
        else:
            zero = pred_curves.sum() * 0.0
            loss_curve = zero
        total = (
            float(weight_cfg.get('ce_weight', 1.0)) * classification_loss
            + curve_loss_weight_from_config(self.config, overrides=loss_weight_overrides) * loss_curve
        )
        out = {
            'loss_total': total,
            **classification_terms,
            'loss_curve': loss_curve,
        }
        out[self.curve_loss_name] = loss_curve
        return out
