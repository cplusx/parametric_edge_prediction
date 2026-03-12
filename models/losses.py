from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from misc_utils.train_utils import sample_bezier_curves_torch
from models.matcher import hungarian_curve_matching, repeated_hungarian_curve_matching, topk_curve_positive_indices


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.sum() * 0.0
    weights = weights.to(values.device, values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    denom = weights.sum().clamp_min(1e-6)
    return (values * weights).sum() / denom


def _classification_loss(pred_logits: torch.Tensor, target_classes: torch.Tensor, no_object_weight: float, loss_type: str, focal_alpha: float, focal_gamma: float) -> torch.Tensor:
    if loss_type == 'focal':
        object_logits = pred_logits[..., 0] - pred_logits[..., 1]
        target_object = (target_classes == 0).float()
        bce = F.binary_cross_entropy_with_logits(object_logits, target_object, reduction='none')
        prob = torch.sigmoid(object_logits)
        pt = torch.where(target_object > 0.5, prob, 1.0 - prob)
        alpha_t = torch.where(target_object > 0.5, torch.full_like(target_object, focal_alpha), torch.full_like(target_object, 1.0 - focal_alpha))
        weight = alpha_t * (1.0 - pt).pow(focal_gamma)
        neg_scale = torch.where(target_object > 0.5, torch.ones_like(target_object), torch.full_like(target_object, no_object_weight))
        return (bce * weight * neg_scale).mean()
    class_weight = torch.tensor([1.0, no_object_weight], device=pred_logits.device)
    return F.cross_entropy(pred_logits.transpose(1, 2), target_classes, weight=class_weight)


def _matched_curve_weights(targets: List[dict], indices: List[Tuple[torch.Tensor, torch.Tensor]], device: torch.device, alpha: float, power: float) -> torch.Tensor:
    if alpha <= 0.0:
        total = sum(int(src_idx.numel()) for src_idx, _ in indices)
        return torch.ones((total,), device=device)
    weights = []
    for batch_idx, (_, tgt_idx) in enumerate(indices):
        if tgt_idx.numel() == 0:
            continue
        curvatures = targets[batch_idx].get('curve_curvatures')
        if curvatures is None or curvatures.numel() == 0:
            weights.append(torch.ones((tgt_idx.numel(),), device=device))
            continue
        curvatures = curvatures[tgt_idx].to(device=device, dtype=torch.float32)
        max_curv = curvatures.max().clamp_min(1e-6)
        normalized = (curvatures / max_curv).clamp(0.0, 1.0)
        weights.append(1.0 + alpha * normalized.pow(power))
    if not weights:
        return torch.ones((0,), device=device)
    return torch.cat(weights, dim=0)


def _matched_curve_losses(pred_curves: torch.Tensor, pred_logits: torch.Tensor, targets: List[dict], indices: List[Tuple[torch.Tensor, torch.Tensor]], config: Dict) -> Dict[str, torch.Tensor]:
    loss_cfg = config['loss']
    device = pred_logits.device
    target_classes = torch.full(pred_logits.shape[:2], 1, dtype=torch.long, device=device)
    matched_pred_curves = []
    matched_tgt_curves = []
    matched_pred_boxes = []
    matched_tgt_boxes = []
    matched_pred_extents = []
    matched_tgt_norm_lengths = []
    for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
        if src_idx.numel() == 0:
            continue
        target_classes[batch_idx, src_idx] = 0
        matched_pred_curves.append(pred_curves[batch_idx, src_idx])
        matched_tgt_curves.append(targets[batch_idx]['curves'][tgt_idx].to(device))
        matched_pred_boxes.append(torch.stack([
            pred_curves[batch_idx, src_idx, :, 0].min(dim=1).values,
            pred_curves[batch_idx, src_idx, :, 1].min(dim=1).values,
            pred_curves[batch_idx, src_idx, :, 0].max(dim=1).values,
            pred_curves[batch_idx, src_idx, :, 1].max(dim=1).values,
        ], dim=1))
        matched_tgt_boxes.append(targets[batch_idx]['boxes'][tgt_idx].to(device))
        if config['model'].get('curve_anchor_mode', 'sigmoid') == 'ref_point_delta' and 'pred_curve_extent' in config.get('_runtime_outputs', {}):
            matched_pred_extents.append(config['_runtime_outputs']['pred_curve_extent'][batch_idx, src_idx])
            matched_tgt_norm_lengths.append(targets[batch_idx]['curve_norm_lengths'][tgt_idx].to(device))
    loss_ce = _classification_loss(
        pred_logits,
        target_classes,
        no_object_weight=float(loss_cfg.get('no_object_weight', 0.2)),
        loss_type=str(loss_cfg.get('class_loss_type', 'ce')),
        focal_alpha=float(loss_cfg.get('focal_alpha', 0.25)),
        focal_gamma=float(loss_cfg.get('focal_gamma', 2.0)),
    )
    if matched_pred_curves:
        matched_pred_curves = torch.cat(matched_pred_curves, dim=0)
        matched_tgt_curves = torch.cat(matched_tgt_curves, dim=0)
        matched_pred_boxes = torch.cat(matched_pred_boxes, dim=0)
        matched_tgt_boxes = torch.cat(matched_tgt_boxes, dim=0)
        ctrl_weights = _matched_curve_weights(
            targets, indices, device,
            alpha=float(loss_cfg.get('curvature_weight_alpha', 0.0)),
            power=float(loss_cfg.get('curvature_weight_power', 1.0)),
        )
        sample_weights = _matched_curve_weights(
            targets, indices, device,
            alpha=float(loss_cfg.get('curvature_sample_alpha', loss_cfg.get('curvature_weight_alpha', 0.0))),
            power=float(loss_cfg.get('curvature_sample_power', loss_cfg.get('curvature_weight_power', 1.0))),
        )
        endpoint_weights = _matched_curve_weights(
            targets, indices, device,
            alpha=float(loss_cfg.get('curvature_endpoint_alpha', loss_cfg.get('curvature_weight_alpha', 0.0))),
            power=float(loss_cfg.get('curvature_endpoint_power', loss_cfg.get('curvature_weight_power', 1.0))),
        )
        ctrl_abs = torch.abs(matched_pred_curves - matched_tgt_curves).mean(dim=(1, 2))
        loss_ctrl = _weighted_mean(ctrl_abs, ctrl_weights)
        sampled_pred = sample_bezier_curves_torch(matched_pred_curves, num_samples=int(loss_cfg.get('num_curve_samples', 16)))
        sampled_tgt = sample_bezier_curves_torch(matched_tgt_curves, num_samples=int(loss_cfg.get('num_curve_samples', 16)))
        sample_abs = torch.abs(sampled_pred - sampled_tgt).mean(dim=(1, 2))
        loss_sample = _weighted_mean(sample_abs, sample_weights)
        endpoint_abs = torch.abs(matched_pred_curves[:, [0, -1]] - matched_tgt_curves[:, [0, -1]]).mean(dim=(1, 2))
        loss_endpoint = _weighted_mean(endpoint_abs, endpoint_weights)
        bbox_abs = torch.abs(matched_pred_boxes - matched_tgt_boxes).mean(dim=1)
        loss_bbox = _weighted_mean(bbox_abs, ctrl_weights)
        if matched_pred_extents:
            pred_extent = torch.cat(matched_pred_extents, dim=0)
            tgt_norm_length = torch.cat(matched_tgt_norm_lengths, dim=0)
            loss_extent = _weighted_mean(torch.abs(pred_extent - tgt_norm_length), ctrl_weights)
        else:
            loss_extent = pred_curves.sum() * 0.0
    else:
        zero = pred_curves.sum() * 0.0
        loss_ctrl = zero
        loss_sample = zero
        loss_endpoint = zero
        loss_bbox = zero
        loss_extent = zero
    total = (
        float(loss_cfg.get('ce_weight', 1.0)) * loss_ce
        + float(loss_cfg.get('ctrl_weight', 5.0)) * loss_ctrl
        + float(loss_cfg.get('sample_weight', 3.0)) * loss_sample
        + float(loss_cfg.get('endpoint_weight', 2.0)) * loss_endpoint
        + float(loss_cfg.get('bbox_weight', 2.0)) * loss_bbox
        + float(loss_cfg.get('extent_weight', 0.0)) * loss_extent
    )
    return {
        'loss_total': total,
        'loss_ce': loss_ce,
        'loss_ctrl': loss_ctrl,
        'loss_sample': loss_sample,
        'loss_endpoint': loss_endpoint,
        'loss_bbox': loss_bbox,
        'loss_extent': loss_extent,
    }


def _positive_object_loss(pred_logits: torch.Tensor, indices: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
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


def _topk_positive_object_loss(outputs: Dict[str, torch.Tensor], targets: List[dict], config: Dict) -> torch.Tensor:
    loss_cfg = config['loss']
    topk_per_target = int(loss_cfg.get('topk_positive_per_gt', 0))
    if topk_per_target <= 0:
        return outputs['pred_logits'].sum() * 0.0
    positive_indices = topk_curve_positive_indices(
        logits=outputs['pred_logits'],
        curves=outputs['pred_curves'],
        targets=targets,
        topk_per_target=topk_per_target,
        control_cost=float(loss_cfg.get('control_cost', 5.0)),
        sample_cost=float(loss_cfg.get('sample_cost', 2.0)),
        box_cost=float(loss_cfg.get('box_cost', 1.0)),
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


def _one_to_many_losses(outputs: Dict[str, torch.Tensor], targets: List[dict], config: Dict) -> Dict[str, torch.Tensor]:
    loss_cfg = config['loss']
    repeat_factor = int(loss_cfg.get('one_to_many_repeat', 1))
    if repeat_factor <= 1:
        zero = outputs['pred_curves'].sum() * 0.0
        return {
            'loss_om_total': zero,
            'loss_om_ce': zero,
            'loss_om_ctrl': zero,
            'loss_om_sample': zero,
            'loss_om_endpoint': zero,
            'loss_om_bbox': zero,
            'loss_om_extent': zero,
        }
    om_indices = repeated_hungarian_curve_matching(
        logits=outputs['pred_logits'],
        curves=outputs['pred_curves'],
        targets=targets,
        repeat_factor=repeat_factor,
        class_cost=float(loss_cfg.get('class_cost', 1.0)),
        control_cost=float(loss_cfg.get('control_cost', 5.0)),
        sample_cost=float(loss_cfg.get('sample_cost', 2.0)),
        box_cost=float(loss_cfg.get('box_cost', 1.0)),
        num_curve_samples=int(loss_cfg.get('num_curve_samples', 16)),
        active_counts=outputs.get('pred_active_counts'),
    )
    matched = _matched_curve_losses(outputs['pred_curves'], outputs['pred_logits'], targets, om_indices, config)
    loss_pos_ce = _positive_object_loss(outputs['pred_logits'], om_indices)
    total = (
        float(loss_cfg.get('one_to_many_ce_weight', loss_cfg.get('ce_weight', 1.0))) * loss_pos_ce
        + float(loss_cfg.get('one_to_many_ctrl_weight', loss_cfg.get('ctrl_weight', 5.0))) * matched['loss_ctrl']
        + float(loss_cfg.get('one_to_many_sample_weight', loss_cfg.get('sample_weight', 3.0))) * matched['loss_sample']
        + float(loss_cfg.get('one_to_many_endpoint_weight', loss_cfg.get('endpoint_weight', 2.0))) * matched['loss_endpoint']
        + float(loss_cfg.get('one_to_many_bbox_weight', loss_cfg.get('bbox_weight', 2.0))) * matched['loss_bbox']
        + float(loss_cfg.get('one_to_many_extent_weight', loss_cfg.get('extent_weight', 0.0))) * matched['loss_extent']
    )
    return {
        'loss_om_total': total,
        'loss_om_ce': loss_pos_ce,
        'loss_om_ctrl': matched['loss_ctrl'],
        'loss_om_sample': matched['loss_sample'],
        'loss_om_endpoint': matched['loss_endpoint'],
        'loss_om_bbox': matched['loss_bbox'],
        'loss_om_extent': matched['loss_extent'],
    }


def _dn_losses(outputs: Dict[str, torch.Tensor], config: Dict) -> Dict[str, torch.Tensor]:
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
    class_weight = torch.tensor([1.0, float(config['loss'].get('no_object_weight', 0.2))], device=device)
    loss_ce = F.cross_entropy(logits.transpose(1, 2), labels, weight=class_weight)
    if mask.any():
        pred_valid = curves[mask]
        target_valid = gt_curves[mask]
        loss_curve = F.l1_loss(pred_valid, target_valid)
    else:
        loss_curve = curves.sum() * 0.0
    total = float(config['loss'].get('dn_weight', 1.0)) * (loss_ce + float(config['loss'].get('dn_curve_weight', 5.0)) * loss_curve)
    return {'loss_dn': total, 'loss_dn_ce': loss_ce, 'loss_dn_curve': loss_curve}


def _count_loss(outputs: Dict[str, torch.Tensor], targets: List[dict], config: Dict) -> torch.Tensor:
    pred_ratio = outputs.get('pred_count_ratio')
    if pred_ratio is None:
        return outputs['pred_logits'].sum() * 0.0
    max_q = max(1, int(config['model'].get('count_max_queries', config['model'].get('active_query_topk', outputs['pred_logits'].shape[1]))))
    target_counts = torch.tensor(
        [min(max_q, int(target['num_targets'])) / float(max_q) for target in targets],
        device=pred_ratio.device,
        dtype=pred_ratio.dtype,
    )
    return F.l1_loss(pred_ratio, target_counts)


def _distinct_query_loss(outputs: Dict[str, torch.Tensor], config: Dict) -> torch.Tensor:
    hidden = outputs.get('pred_query_hidden')
    ref_points = outputs.get('pred_ref_points')
    logits = outputs.get('pred_logits')
    active_counts = outputs.get('pred_active_counts')
    if hidden is None or ref_points is None or logits is None:
        return outputs['pred_logits'].sum() * 0.0
    topk = int(config['loss'].get('distinct_topk', 24))
    spatial_sigma = float(config['loss'].get('distinct_spatial_sigma', 0.12))
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


def compute_losses(outputs: Dict[str, torch.Tensor], targets: List[dict], config: Dict) -> Dict[str, torch.Tensor]:
    loss_cfg = config['loss']
    config = dict(config)
    config['_runtime_outputs'] = outputs
    indices = hungarian_curve_matching(
        logits=outputs['pred_logits'],
        curves=outputs['pred_curves'],
        targets=targets,
        class_cost=float(loss_cfg.get('class_cost', 1.0)),
        control_cost=float(loss_cfg.get('control_cost', 5.0)),
        sample_cost=float(loss_cfg.get('sample_cost', 2.0)),
        box_cost=float(loss_cfg.get('box_cost', 1.0)),
        num_curve_samples=int(loss_cfg.get('num_curve_samples', 16)),
        active_counts=outputs.get('pred_active_counts'),
    )
    base = _matched_curve_losses(outputs['pred_curves'], outputs['pred_logits'], targets, indices, config)
    total = base['loss_total']
    log_values = {k: v.detach() for k, v in base.items() if k != 'loss_total'}

    aux_weight = float(loss_cfg.get('aux_weight', 0.5))
    for level_idx, aux in enumerate(outputs.get('aux_outputs', [])):
        aux_indices = hungarian_curve_matching(
            logits=aux['pred_logits'],
            curves=aux['pred_curves'],
            targets=targets,
            class_cost=float(loss_cfg.get('class_cost', 1.0)),
            control_cost=float(loss_cfg.get('control_cost', 5.0)),
            sample_cost=float(loss_cfg.get('sample_cost', 2.0)),
            box_cost=float(loss_cfg.get('box_cost', 1.0)),
            num_curve_samples=int(loss_cfg.get('num_curve_samples', 16)),
            active_counts=aux.get('pred_active_counts'),
        )
        aux_losses = _matched_curve_losses(aux['pred_curves'], aux['pred_logits'], targets, aux_indices, config)
        total = total + aux_weight * aux_losses['loss_total']
        log_values[f'loss_aux_{level_idx}'] = aux_losses['loss_total'].detach()

    one_to_many = _one_to_many_losses(outputs, targets, config)
    total = total + float(loss_cfg.get('one_to_many_weight', 0.0)) * one_to_many['loss_om_total']
    log_values['loss_om_total'] = one_to_many['loss_om_total'].detach()
    log_values['loss_om_ce'] = one_to_many['loss_om_ce'].detach()
    log_values['loss_om_ctrl'] = one_to_many['loss_om_ctrl'].detach()
    log_values['loss_om_sample'] = one_to_many['loss_om_sample'].detach()
    log_values['loss_om_endpoint'] = one_to_many['loss_om_endpoint'].detach()
    log_values['loss_om_bbox'] = one_to_many['loss_om_bbox'].detach()
    log_values['loss_om_extent'] = one_to_many['loss_om_extent'].detach()

    loss_topk_pos = _topk_positive_object_loss(outputs, targets, config)
    total = total + float(loss_cfg.get('topk_positive_weight', 0.0)) * loss_topk_pos
    log_values['loss_topk_pos'] = loss_topk_pos.detach()

    loss_count = _count_loss(outputs, targets, config)
    total = total + float(loss_cfg.get('count_weight', 0.0)) * loss_count
    log_values['loss_count'] = loss_count.detach()

    loss_distinct = _distinct_query_loss(outputs, config)
    total = total + float(loss_cfg.get('distinct_weight', 0.0)) * loss_distinct
    log_values['loss_distinct'] = loss_distinct.detach()

    dn = _dn_losses(outputs, config)
    total = total + dn['loss_dn']
    log_values['loss_dn_ce'] = dn['loss_dn_ce'].detach()
    log_values['loss_dn_curve'] = dn['loss_dn_curve'].detach()
    return {'loss': total, 'matching': indices, **log_values}
