from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from models.geometry import curve_boxes_xyxy, matched_generalized_box_iou, symmetric_curve_distance
from models.losses.base import BaseLossComponent, balanced_class_weights, matched_curve_weights, weighted_mean


class ClassificationLoss(BaseLossComponent):
    def __call__(
        self,
        pred_logits: torch.Tensor,
        target_classes: torch.Tensor,
        query_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss_cfg = self.config['loss']
        no_object_weight = float(loss_cfg.get('no_object_weight', 0.2))
        dynamic_class_balance = bool(loss_cfg.get('dynamic_class_balance', True))
        loss_type = str(loss_cfg.get('class_loss_type', 'ce'))
        focal_alpha = float(loss_cfg.get('focal_alpha', 0.25))
        focal_gamma = float(loss_cfg.get('focal_gamma', 2.0))

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
            alpha_t = torch.where(target_object > 0.5, torch.full_like(target_object, focal_alpha), torch.full_like(target_object, 1.0 - focal_alpha))
            weight = alpha_t * (1.0 - pt).pow(focal_gamma)
            neg_scale = torch.where(target_object > 0.5, torch.ones_like(target_object), torch.full_like(target_object, no_object_weight))
            loss = bce * weight * neg_scale
            return reduce_per_sample(loss)
        class_weight = torch.tensor([1.0, no_object_weight], device=pred_logits.device)
        ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, weight=class_weight, reduction='none')
        return reduce_per_sample(ce)


class MatchedCurveLoss(BaseLossComponent):
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.classification = ClassificationLoss(config)

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
        matched_pred_boxes = []
        matched_tgt_boxes = []
        matched_pair_weights = []
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            target_classes[batch_idx, src_idx] = 0
            matched_pred_curves.append(pred_curves[batch_idx, src_idx])
            matched_tgt_curves.append(targets[batch_idx]['curves'][tgt_idx].to(device))
            matched_pred_boxes.append(curve_boxes_xyxy(pred_curves[batch_idx, src_idx]))
            matched_tgt_boxes.append(targets[batch_idx]['boxes'][tgt_idx].to(device))
            if match_weights is not None:
                matched_pair_weights.append(match_weights[batch_idx].to(device=device, dtype=pred_logits.dtype))
        loss_ce = self.classification(pred_logits, target_classes, query_weights=query_weights)
        if matched_pred_curves:
            matched_pred_curves = torch.cat(matched_pred_curves, dim=0)
            matched_tgt_curves = torch.cat(matched_tgt_curves, dim=0)
            matched_pred_boxes = torch.cat(matched_pred_boxes, dim=0)
            matched_tgt_boxes = torch.cat(matched_tgt_boxes, dim=0)
            ctrl_weights = matched_curve_weights(
                targets,
                indices,
                device,
                alpha=float(loss_cfg.get('curvature_weight_alpha', 0.0)),
                power=float(loss_cfg.get('curvature_weight_power', 1.0)),
            )
            endpoint_weights = matched_curve_weights(
                targets,
                indices,
                device,
                alpha=float(loss_cfg.get('curvature_endpoint_alpha', loss_cfg.get('curvature_weight_alpha', 0.0))),
                power=float(loss_cfg.get('curvature_endpoint_power', loss_cfg.get('curvature_weight_power', 1.0))),
            )
            curve_weights = matched_curve_weights(
                targets,
                indices,
                device,
                alpha=float(loss_cfg.get('curvature_curve_distance_alpha', loss_cfg.get('curvature_weight_alpha', 0.0))),
                power=float(loss_cfg.get('curvature_curve_distance_power', loss_cfg.get('curvature_weight_power', 1.0))),
            )
            if matched_pair_weights:
                pair_weights = torch.cat(matched_pair_weights, dim=0)
                ctrl_weights = ctrl_weights * pair_weights
                endpoint_weights = endpoint_weights * pair_weights
                curve_weights = curve_weights * pair_weights
            ctrl_abs = torch.abs(matched_pred_curves - matched_tgt_curves).mean(dim=(1, 2))
            loss_ctrl = weighted_mean(ctrl_abs, ctrl_weights)
            endpoint_abs = torch.abs(matched_pred_curves[:, [0, -1]] - matched_tgt_curves[:, [0, -1]]).mean(dim=(1, 2))
            loss_endpoint = weighted_mean(endpoint_abs, endpoint_weights)
            bbox_abs = torch.abs(matched_pred_boxes - matched_tgt_boxes).mean(dim=1)
            loss_bbox = weighted_mean(bbox_abs, ctrl_weights)
            loss_giou = weighted_mean(1.0 - matched_generalized_box_iou(matched_pred_boxes, matched_tgt_boxes), ctrl_weights)
            curve_dist, curve_chamfer, curve_length = symmetric_curve_distance(
                matched_pred_curves,
                matched_tgt_curves,
                num_samples=int(loss_cfg.get('num_curve_samples', 16)),
                length_weight=float(loss_cfg.get('curve_distance_length_weight', 0.25)),
            )
            loss_curve_dist = weighted_mean(curve_dist, curve_weights)
            loss_curve_chamfer = weighted_mean(curve_chamfer, curve_weights)
            loss_curve_length = weighted_mean(curve_length, curve_weights)
        else:
            zero = pred_curves.sum() * 0.0
            loss_ctrl = zero
            loss_endpoint = zero
            loss_bbox = zero
            loss_giou = zero
            loss_curve_dist = zero
            loss_curve_chamfer = zero
            loss_curve_length = zero
        total = (
            float(weight_cfg.get('ce_weight', 1.0)) * loss_ce
            + float(weight_cfg.get('ctrl_weight', 5.0)) * loss_ctrl
            + float(weight_cfg.get('sample_weight', 0.0)) * loss_curve_chamfer
            + float(weight_cfg.get('endpoint_weight', 2.0)) * loss_endpoint
            + float(weight_cfg.get('bbox_weight', 2.0)) * loss_bbox
            + float(weight_cfg.get('giou_weight', 1.0)) * loss_giou
            + float(weight_cfg.get('curve_distance_weight', 2.0)) * loss_curve_dist
        )
        return {
            'loss_total': total,
            'loss_ce': loss_ce,
            'loss_ctrl': loss_ctrl,
            'loss_sample': loss_curve_chamfer,
            'loss_endpoint': loss_endpoint,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou,
            'loss_curve_dist': loss_curve_dist,
            'loss_curve_chamfer': loss_curve_chamfer,
            'loss_curve_length': loss_curve_length,
        }