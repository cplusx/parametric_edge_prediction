from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from models.losses.base import BaseLossComponent
from models.matcher import repeated_hungarian_curve_matching, topk_curve_positive_indices


class PositiveObjectLoss(BaseLossComponent):
    def __call__(self, pred_logits: torch.Tensor, indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        positives = []
        for batch_idx, (src_idx, _) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            positives.append(pred_logits[batch_idx, src_idx])
        if not positives:
            return pred_logits.sum() * 0.0
        logits = torch.cat(positives, dim=0)
        labels = torch.zeros((logits.shape[0],), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)


class TopKPositiveLoss(BaseLossComponent):
    def __call__(self, outputs: Dict[str, torch.Tensor], targets: List[dict]) -> torch.Tensor:
        loss_cfg = self.config['loss']
        topk_per_target = int(loss_cfg.get('topk_positive_per_gt', 0))
        if topk_per_target <= 0:
            return outputs['pred_logits'].sum() * 0.0
        positive_indices = topk_curve_positive_indices(
            logits=outputs['pred_logits'],
            curves=outputs['pred_curves'],
            targets=targets,
            topk_per_target=topk_per_target,
            control_cost=float(loss_cfg.get('control_cost', 5.0)),
            sample_cost=0.0,
            box_cost=float(loss_cfg.get('box_cost', 1.0)),
            giou_cost=float(loss_cfg.get('giou_cost', 1.0)),
            curve_distance_cost=float(loss_cfg.get('curve_distance_cost', 1.0)),
            curve_match_point_count=int(loss_cfg.get('curve_match_point_count', 4)),
            num_curve_samples=int(loss_cfg.get('num_curve_samples', 16)),
            active_counts=outputs.get('pred_active_counts'),
        )
        positives = []
        for batch_idx, src_idx in enumerate(positive_indices):
            if src_idx.numel() == 0:
                continue
            positives.append(outputs['pred_logits'][batch_idx, src_idx])
        if not positives:
            return outputs['pred_logits'].sum() * 0.0
        logits = torch.cat(positives, dim=0)
        labels = torch.zeros((logits.shape[0],), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)


class DenoisingLoss(BaseLossComponent):
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
        gt_curves = dn_meta['curves'].to(device)
        class_weight = torch.tensor([1.0, float(self.config['loss'].get('no_object_weight', 0.2))], device=device)
        loss_ce = F.cross_entropy(logits.transpose(1, 2), labels, weight=class_weight)
        if mask.any():
            pred_valid = curves[mask]
            target_valid = gt_curves[mask]
            loss_curve = F.l1_loss(pred_valid, target_valid)
        else:
            loss_curve = curves.sum() * 0.0
        total = float(self.config['loss'].get('dn_weight', 1.0)) * (loss_ce + float(self.config['loss'].get('dn_curve_weight', 5.0)) * loss_curve)
        return {'loss_dn': total, 'loss_dn_ce': loss_ce, 'loss_dn_curve': loss_curve}


class CountLoss(BaseLossComponent):
    def __call__(self, outputs: Dict[str, torch.Tensor], targets: List[dict]) -> torch.Tensor:
        pred_ratio = outputs.get('pred_count_ratio')
        if pred_ratio is None:
            return outputs['pred_logits'].sum() * 0.0
        max_q = max(1, int(self.config['model'].get('count_max_queries', self.config['model'].get('active_query_topk', outputs['pred_logits'].shape[1]))))
        target_counts = torch.tensor(
            [min(max_q, int(target['num_targets'])) / float(max_q) for target in targets],
            device=pred_ratio.device,
            dtype=pred_ratio.dtype,
        )
        return F.l1_loss(pred_ratio, target_counts)


class DistinctQueryLoss(BaseLossComponent):
    def __call__(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = outputs.get('pred_query_hidden')
        ref_points = outputs.get('pred_ref_points')
        logits = outputs.get('pred_logits')
        active_counts = outputs.get('pred_active_counts')
        if hidden is None or ref_points is None or logits is None:
            return outputs['pred_logits'].sum() * 0.0
        topk = int(self.config['loss'].get('distinct_topk', 24))
        spatial_sigma = float(self.config['loss'].get('distinct_spatial_sigma', 0.12))
        if hidden.shape[1] <= 1:
            return hidden.sum() * 0.0
        obj_prob = logits.softmax(-1)[..., 0]
        batch_losses = []
        for batch_idx in range(hidden.shape[0]):
            limit = hidden.shape[1] if active_counts is None else int(active_counts[batch_idx].item())
            if limit <= 1:
                continue
            limit = min(limit, hidden.shape[1], topk)
            prob = obj_prob[batch_idx]
            idx = torch.topk(prob, k=limit, dim=0).indices
            h = F.normalize(hidden[batch_idx, idx], dim=-1)
            r = ref_points[batch_idx, idx]
            p = prob[idx]
            sim = torch.matmul(h, h.transpose(0, 1)).pow(2)
            dist = torch.cdist(r, r)
            spatial_weight = torch.exp(-(dist.pow(2)) / max(spatial_sigma * spatial_sigma, 1e-6))
            weight = (p.unsqueeze(0) * p.unsqueeze(1)) * spatial_weight
            eye = torch.eye(limit, device=hidden.device, dtype=torch.bool)
            penalty = sim[~eye] * weight[~eye]
            if penalty.numel() > 0:
                batch_losses.append(penalty.mean())
        if not batch_losses:
            return hidden.sum() * 0.0
        return torch.stack(batch_losses).mean()


class OneToManyLoss(BaseLossComponent):
    def __init__(self, config: Dict, matched_curve_loss, positive_object_loss) -> None:
        super().__init__(config)
        self.matched_curve_loss = matched_curve_loss
        self.positive_object_loss = positive_object_loss

    def __call__(self, outputs: Dict[str, torch.Tensor], targets: List[dict]) -> Dict[str, torch.Tensor]:
        loss_cfg = self.config['loss']
        repeat_factor = int(loss_cfg.get('one_to_many_repeat', 1))
        if repeat_factor <= 1:
            zero = outputs['pred_curves'].sum() * 0.0
            return {
                'loss_om_total': zero,
                'loss_om_ce': zero,
                'loss_om_ctrl': zero,
                'loss_om_endpoint': zero,
                'loss_om_bbox': zero,
                'loss_om_giou': zero,
                'loss_om_curve_dist': zero,
                'loss_om_extent': zero,
            }
        om_indices = repeated_hungarian_curve_matching(
            logits=outputs['pred_logits'],
            curves=outputs['pred_curves'],
            targets=targets,
            repeat_factor=repeat_factor,
            control_cost=float(loss_cfg.get('control_cost', 5.0)),
            sample_cost=0.0,
            box_cost=float(loss_cfg.get('box_cost', 1.0)),
            giou_cost=float(loss_cfg.get('giou_cost', 1.0)),
            curve_distance_cost=float(loss_cfg.get('curve_distance_cost', 1.0)),
            curve_match_point_count=int(loss_cfg.get('curve_match_point_count', 4)),
            num_curve_samples=int(loss_cfg.get('num_curve_samples', 16)),
            active_counts=outputs.get('pred_active_counts'),
        )
        matched = self.matched_curve_loss(outputs['pred_curves'], outputs['pred_logits'], targets, om_indices, outputs)
        loss_pos_ce = self.positive_object_loss(outputs['pred_logits'], om_indices)
        total = (
            float(loss_cfg.get('one_to_many_ce_weight', loss_cfg.get('ce_weight', 1.0))) * loss_pos_ce
            + float(loss_cfg.get('one_to_many_ctrl_weight', loss_cfg.get('ctrl_weight', 5.0))) * matched['loss_ctrl']
            + float(loss_cfg.get('one_to_many_endpoint_weight', loss_cfg.get('endpoint_weight', 2.0))) * matched['loss_endpoint']
            + float(loss_cfg.get('one_to_many_bbox_weight', loss_cfg.get('bbox_weight', 2.0))) * matched['loss_bbox']
            + float(loss_cfg.get('one_to_many_giou_weight', loss_cfg.get('giou_weight', 1.0))) * matched['loss_giou']
            + float(loss_cfg.get('one_to_many_curve_distance_weight', loss_cfg.get('curve_distance_weight', 2.0))) * matched['loss_curve_dist']
            + float(loss_cfg.get('one_to_many_extent_weight', loss_cfg.get('extent_weight', 0.0))) * matched['loss_extent']
        )
        return {
            'loss_om_total': total,
            'loss_om_ce': loss_pos_ce,
            'loss_om_ctrl': matched['loss_ctrl'],
            'loss_om_endpoint': matched['loss_endpoint'],
            'loss_om_bbox': matched['loss_bbox'],
            'loss_om_giou': matched['loss_giou'],
            'loss_om_curve_dist': matched['loss_curve_dist'],
            'loss_om_extent': matched['loss_extent'],
        }