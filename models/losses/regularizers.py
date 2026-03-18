from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from models.losses.base import BaseLossComponent, balanced_class_weights
from models.matcher import build_curve_cost_matrix, hungarian_curve_matching


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
    def __init__(self, config: Dict, matched_curve_loss) -> None:
        super().__init__(config)
        self.matched_curve_loss = matched_curve_loss

    def _topk_truncation_enabled(self) -> bool:
        return bool(self.config['loss'].get('topk_positive_enabled', False))

    def _layer_topk_per_target(self, layer_idx: int) -> int:
        layer_values = self.config['loss'].get('topk_positive_layer_k')
        if isinstance(layer_values, list) and layer_values:
            index = min(layer_idx, len(layer_values) - 1)
            return int(layer_values[index])
        return int(self.config['loss'].get('topk_positive_per_gt', 0))

    def _layer_rank_weights(self, layer_idx: int, max_rank: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        del layer_idx
        tau = float(self.config['loss'].get('topk_positive_tau', 1.5))
        tau = max(tau, 1e-6)
        return torch.exp(-torch.arange(max_rank, device=device, dtype=dtype) / tau)

    def _flatten_group_outputs(self, aux_outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        grouped_logits = aux_outputs['group_pred_logits']
        grouped_curves = aux_outputs['group_pred_curves']
        batch_size, num_groups, num_queries = grouped_logits.shape[:3]
        flat_logits = grouped_logits.reshape(batch_size, num_groups * num_queries, -1)
        flat_curves = grouped_curves.reshape(batch_size, num_groups * num_queries, grouped_curves.shape[-2], grouped_curves.shape[-1])
        runtime_outputs = {
            'pred_curve_extent': None,
        }
        grouped_extent = aux_outputs.get('group_pred_curve_extent')
        if grouped_extent is not None:
            runtime_outputs['pred_curve_extent'] = grouped_extent.reshape(batch_size, num_groups * num_queries)
        return flat_logits, flat_curves, runtime_outputs

    def _select_group_topk_matches(
        self,
        aux_outputs: Dict[str, torch.Tensor],
        targets: List[dict],
        topk_per_target: int,
        layer_idx: int,
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor], torch.Tensor]:
        loss_cfg = self.config['loss']
        grouped_logits = aux_outputs['group_pred_logits']
        grouped_curves = aux_outputs['group_pred_curves']
        batch_size, num_groups, num_queries = grouped_logits.shape[:3]
        device = grouped_logits.device
        dtype = grouped_logits.dtype
        topk_enabled = self._topk_truncation_enabled()
        selected_indices: List[Tuple[torch.Tensor, torch.Tensor]] = []
        selected_weights: List[torch.Tensor] = []
        query_weights = torch.ones((batch_size, num_groups * num_queries), device=device, dtype=dtype)

        for batch_idx, target in enumerate(targets):
            tgt_curves = target['curves'].to(device)
            tgt_boxes = target['boxes'].to(device)
            if tgt_curves.numel() == 0:
                selected_indices.append((torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)))
                selected_weights.append(torch.empty(0, dtype=dtype, device=device))
                continue

            candidates: List[List[Tuple[float, int]]] = [[] for _ in range(tgt_curves.shape[0])]
            for group_idx in range(num_groups):
                total_cost = build_curve_cost_matrix(
                    logits=grouped_logits[batch_idx, group_idx],
                    curves=grouped_curves[batch_idx, group_idx],
                    tgt_curves=tgt_curves,
                    tgt_boxes=tgt_boxes,
                    control_cost=float(loss_cfg.get('control_cost', 5.0)),
                    sample_cost=float(loss_cfg.get('sample_cost', 2.0)),
                    box_cost=float(loss_cfg.get('box_cost', 1.0)),
                    giou_cost=float(loss_cfg.get('giou_cost', 1.0)),
                    curve_distance_cost=float(loss_cfg.get('curve_distance_cost', 1.0)),
                    curve_match_point_count=int(loss_cfg.get('curve_match_point_count', 4)),
                    num_curve_samples=int(loss_cfg.get('num_curve_samples', 16)),
                )
                row_ind, col_ind = linear_sum_assignment(total_cost.detach().cpu().numpy())
                for src_idx, tgt_idx in zip(row_ind.tolist(), col_ind.tolist()):
                    candidates[tgt_idx].append((float(total_cost[src_idx, tgt_idx].item()), group_idx * num_queries + src_idx))

            batch_src = []
            batch_tgt = []
            batch_pair_weights = []
            for tgt_idx, tgt_candidates in enumerate(candidates):
                if not tgt_candidates:
                    continue
                tgt_candidates.sort(key=lambda item: item[0])
                keep_count = len(tgt_candidates)
                if topk_enabled:
                    keep_count = min(keep_count, max(0, topk_per_target))
                if keep_count <= 0:
                    continue
                rank_weights = self._layer_rank_weights(layer_idx, keep_count, device=device, dtype=dtype)
                for rank_idx, (_, flat_src_idx) in enumerate(tgt_candidates[:keep_count]):
                    batch_src.append(flat_src_idx)
                    batch_tgt.append(tgt_idx)
                    batch_pair_weights.append(rank_weights[rank_idx])
                    query_weights[batch_idx, flat_src_idx] = rank_weights[rank_idx]

            if batch_src:
                selected_indices.append((
                    torch.tensor(batch_src, dtype=torch.long, device=device),
                    torch.tensor(batch_tgt, dtype=torch.long, device=device),
                ))
                selected_weights.append(torch.stack(batch_pair_weights).to(device=device, dtype=dtype))
            else:
                selected_indices.append((torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)))
                selected_weights.append(torch.empty(0, dtype=dtype, device=device))
        return selected_indices, selected_weights, query_weights

    def __call__(self, outputs: Dict[str, torch.Tensor], targets: List[dict]) -> torch.Tensor:
        loss_cfg = self.config['loss']
        aux_outputs = outputs.get('aux_outputs', [])
        if not aux_outputs or outputs.get('pred_group_count', 1) <= 1:
            zero = outputs['pred_logits'].sum() * 0.0
            return {
                'loss_topk_pos': zero,
                'loss_topk_pos_ce': zero,
                'loss_topk_pos_ctrl': zero,
                'loss_topk_pos_sample': zero,
                'loss_topk_pos_endpoint': zero,
                'loss_topk_pos_bbox': zero,
                'loss_topk_pos_giou': zero,
                'loss_topk_pos_curve_dist': zero,
                'loss_topk_pos_extent': zero,
            }

        layer_summaries = []
        for layer_idx, aux in enumerate(aux_outputs):
            grouped_logits = aux.get('group_pred_logits')
            grouped_curves = aux.get('group_pred_curves')
            if grouped_logits is None or grouped_curves is None or grouped_logits.shape[1] <= 1:
                continue
            topk_per_target = self._layer_topk_per_target(layer_idx)
            if self._topk_truncation_enabled() and topk_per_target <= 0:
                continue
            indices, match_weights, query_weights = self._select_group_topk_matches(aux, targets, topk_per_target, layer_idx)
            flat_logits, flat_curves, runtime_outputs = self._flatten_group_outputs(aux)
            layer_summaries.append(
                self.matched_curve_loss(
                    flat_curves,
                    flat_logits,
                    targets,
                    indices,
                    runtime_outputs,
                    match_weights=match_weights,
                    query_weights=query_weights,
                )
            )

        if not layer_summaries:
            zero = outputs['pred_logits'].sum() * 0.0
            return {
                'loss_topk_pos': zero,
                'loss_topk_pos_ce': zero,
                'loss_topk_pos_ctrl': zero,
                'loss_topk_pos_sample': zero,
                'loss_topk_pos_endpoint': zero,
                'loss_topk_pos_bbox': zero,
                'loss_topk_pos_giou': zero,
                'loss_topk_pos_curve_dist': zero,
                'loss_topk_pos_extent': zero,
            }

        return {
            'loss_topk_pos': torch.stack([summary['loss_total'] for summary in layer_summaries]).mean(),
            'loss_topk_pos_ce': torch.stack([summary['loss_ce'] for summary in layer_summaries]).mean(),
            'loss_topk_pos_ctrl': torch.stack([summary['loss_ctrl'] for summary in layer_summaries]).mean(),
            'loss_topk_pos_sample': torch.stack([summary['loss_sample'] for summary in layer_summaries]).mean(),
            'loss_topk_pos_endpoint': torch.stack([summary['loss_endpoint'] for summary in layer_summaries]).mean(),
            'loss_topk_pos_bbox': torch.stack([summary['loss_bbox'] for summary in layer_summaries]).mean(),
            'loss_topk_pos_giou': torch.stack([summary['loss_giou'] for summary in layer_summaries]).mean(),
            'loss_topk_pos_curve_dist': torch.stack([summary['loss_curve_dist'] for summary in layer_summaries]).mean(),
            'loss_topk_pos_extent': torch.stack([summary['loss_extent'] for summary in layer_summaries]).mean(),
        }


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
        if bool(self.config['loss'].get('dynamic_class_balance', True)):
            ce_per_query = F.cross_entropy(logits.transpose(1, 2), labels, reduction='none')
            class_weights, normalizer, active_mask = balanced_class_weights(labels, positive_class=0)
            per_sample = (ce_per_query * class_weights).sum(dim=1) / normalizer
            loss_ce = per_sample[active_mask].mean() if bool(active_mask.any()) else logits.sum() * 0.0
        else:
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


class DistinctQueryLoss(BaseLossComponent):
    def __call__(self, outputs: Dict[str, torch.Tensor], targets: List[dict]) -> torch.Tensor:
        grouped_hidden = outputs.get('group_pred_query_hidden')
        grouped_refs = outputs.get('group_pred_ref_points')
        grouped_logits = outputs.get('group_pred_logits')
        grouped_curves = outputs.get('group_pred_curves')
        if grouped_hidden is None or grouped_refs is None or grouped_logits is None or grouped_curves is None:
            hidden = outputs.get('pred_query_hidden')
            ref_points = outputs.get('pred_ref_points')
            logits = outputs.get('pred_logits')
            curves = outputs.get('pred_curves')
            if hidden is None or ref_points is None or logits is None or curves is None:
                return outputs['pred_logits'].sum() * 0.0
            grouped_hidden = hidden.unsqueeze(1)
            grouped_refs = ref_points.unsqueeze(1)
            grouped_logits = logits.unsqueeze(1)
            grouped_curves = curves.unsqueeze(1)
        topk = int(self.config['loss'].get('distinct_topk', 24))
        spatial_sigma = float(self.config['loss'].get('distinct_spatial_sigma', 0.12))
        if grouped_hidden.shape[2] <= 1:
            return grouped_hidden.sum() * 0.0
        batch_losses = []
        loss_cfg = self.config['loss']
        for batch_idx in range(grouped_hidden.shape[0]):
            tgt_curves = targets[batch_idx]['curves'].to(grouped_hidden.device)
            tgt_boxes = targets[batch_idx]['boxes'].to(grouped_hidden.device)
            if tgt_curves.numel() == 0:
                continue
            for group_idx in range(grouped_hidden.shape[1]):
                max_queries = grouped_hidden.shape[2]
                limit = max_queries
                if limit <= 1:
                    continue
                limit = min(limit, max_queries, topk)
                prob = grouped_logits[batch_idx, group_idx].softmax(-1)[:, 0]
                idx = torch.topk(prob, k=limit, dim=0).indices
                selected_hidden = grouped_hidden[batch_idx, group_idx, idx]
                selected_refs = grouped_refs[batch_idx, group_idx, idx]
                selected_prob = prob[idx]
                selected_curves = grouped_curves[batch_idx, group_idx, idx]
                total_cost = build_curve_cost_matrix(
                    logits=grouped_logits[batch_idx, group_idx, idx],
                    curves=selected_curves,
                    tgt_curves=tgt_curves,
                    tgt_boxes=tgt_boxes,
                    control_cost=float(loss_cfg.get('control_cost', 5.0)),
                    sample_cost=float(loss_cfg.get('sample_cost', 2.0)),
                    box_cost=float(loss_cfg.get('box_cost', 1.0)),
                    giou_cost=float(loss_cfg.get('giou_cost', 1.0)),
                    curve_distance_cost=float(loss_cfg.get('curve_distance_cost', 1.0)),
                    curve_match_point_count=int(loss_cfg.get('curve_match_point_count', 4)),
                    num_curve_samples=int(loss_cfg.get('num_curve_samples', 16)),
                )
                assigned_gt = total_cost.argmin(dim=1)
                h = F.normalize(selected_hidden, dim=-1)
                sim = torch.matmul(h, h.transpose(0, 1)).pow(2)
                dist = torch.cdist(selected_refs, selected_refs)
                spatial_weight = torch.exp(-(dist.pow(2)) / max(spatial_sigma * spatial_sigma, 1e-6))
                weight = (selected_prob.unsqueeze(0) * selected_prob.unsqueeze(1)) * spatial_weight
                same_gt = assigned_gt.unsqueeze(0) == assigned_gt.unsqueeze(1)
                eye = torch.eye(limit, device=grouped_hidden.device, dtype=torch.bool)
                pair_mask = same_gt & (~eye)
                penalty = sim[pair_mask] * weight[pair_mask]
                if penalty.numel() > 0:
                    batch_losses.append(penalty.mean())
        if not batch_losses:
            return grouped_hidden.sum() * 0.0
        return torch.stack(batch_losses).mean()


class OneToManyLoss(BaseLossComponent):
    def __init__(self, config: Dict, matched_curve_loss, positive_object_loss) -> None:
        super().__init__(config)
        self.matched_curve_loss = matched_curve_loss
        self.positive_object_loss = positive_object_loss

    def __call__(self, outputs: Dict[str, torch.Tensor], targets: List[dict]) -> Dict[str, torch.Tensor]:
        grouped_logits = outputs.get('group_pred_logits')
        grouped_curves = outputs.get('group_pred_curves')
        if grouped_logits is None or grouped_curves is None or grouped_logits.shape[1] <= 1:
            zero = outputs['pred_curves'].sum() * 0.0
            return {
                'loss_om_total': zero,
                'loss_om_ce': zero,
                'loss_om_ctrl': zero,
                'loss_om_sample': zero,
                'loss_om_endpoint': zero,
                'loss_om_bbox': zero,
                'loss_om_giou': zero,
                'loss_om_curve_dist': zero,
                'loss_om_extent': zero,
            }
        loss_cfg = self.config['loss']
        om_weight_overrides = {
            'ce_weight': float(loss_cfg.get('one_to_many_ce_weight', loss_cfg.get('ce_weight', 1.0))),
            'ctrl_weight': float(loss_cfg.get('one_to_many_ctrl_weight', loss_cfg.get('ctrl_weight', 5.0))),
            'sample_weight': float(loss_cfg.get('one_to_many_sample_weight', loss_cfg.get('sample_weight', 0.0))),
            'endpoint_weight': float(loss_cfg.get('one_to_many_endpoint_weight', loss_cfg.get('endpoint_weight', 2.0))),
            'bbox_weight': float(loss_cfg.get('one_to_many_bbox_weight', loss_cfg.get('bbox_weight', 2.0))),
            'giou_weight': float(loss_cfg.get('one_to_many_giou_weight', loss_cfg.get('giou_weight', 1.0))),
            'curve_distance_weight': float(loss_cfg.get('one_to_many_curve_distance_weight', loss_cfg.get('curve_distance_weight', 2.0))),
            'extent_weight': float(loss_cfg.get('one_to_many_extent_weight', loss_cfg.get('extent_weight', 0.0))),
        }
        group_summaries = []
        for group_idx in range(1, grouped_logits.shape[1]):
            group_indices = hungarian_curve_matching(
                logits=grouped_logits[:, group_idx],
                curves=grouped_curves[:, group_idx],
                targets=targets,
                control_cost=float(loss_cfg.get('control_cost', 5.0)),
                sample_cost=float(loss_cfg.get('sample_cost', 2.0)),
                box_cost=float(loss_cfg.get('box_cost', 1.0)),
                giou_cost=float(loss_cfg.get('giou_cost', 1.0)),
                curve_distance_cost=float(loss_cfg.get('curve_distance_cost', 1.0)),
                curve_match_point_count=int(loss_cfg.get('curve_match_point_count', 4)),
                num_curve_samples=int(loss_cfg.get('num_curve_samples', 16)),
            )
            group_outputs = {
                'pred_curve_extent': None if outputs.get('group_pred_curve_extent') is None else outputs['group_pred_curve_extent'][:, group_idx],
            }
            group_summaries.append(
                self.matched_curve_loss(
                    grouped_curves[:, group_idx],
                    grouped_logits[:, group_idx],
                    targets,
                    group_indices,
                    group_outputs,
                    loss_weight_overrides=om_weight_overrides,
                )
            )

        total = torch.stack([summary['loss_total'] for summary in group_summaries]).mean()
        return {
            'loss_om_total': total,
            'loss_om_ce': torch.stack([summary['loss_ce'] for summary in group_summaries]).mean(),
            'loss_om_ctrl': torch.stack([summary['loss_ctrl'] for summary in group_summaries]).mean(),
            'loss_om_sample': torch.stack([summary['loss_sample'] for summary in group_summaries]).mean(),
            'loss_om_endpoint': torch.stack([summary['loss_endpoint'] for summary in group_summaries]).mean(),
            'loss_om_bbox': torch.stack([summary['loss_bbox'] for summary in group_summaries]).mean(),
            'loss_om_giou': torch.stack([summary['loss_giou'] for summary in group_summaries]).mean(),
            'loss_om_curve_dist': torch.stack([summary['loss_curve_dist'] for summary in group_summaries]).mean(),
            'loss_om_extent': torch.stack([summary['loss_extent'] for summary in group_summaries]).mean(),
        }