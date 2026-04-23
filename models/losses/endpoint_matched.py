from typing import Dict, List, Optional, Tuple

import torch

from models.curve_coordinates import curve_external_to_internal
from models.geometry import point_to_incident_curves_attach_distance
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
        matched_l1_terms = []
        matched_attach_terms = []
        matched_degrees = []
        matched_loop_only = []
        use_attach_loss = str(loss_cfg.get('endpoint_loss_type', 'l1')).lower() == 'attach'
        for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            target_classes[batch_idx, src_idx] = 0
            batch_pred_points = pred_points[batch_idx, src_idx]
            batch_tgt_points = curve_external_to_internal(targets[batch_idx]['points'][tgt_idx].to(device), self.config)
            matched_pred_points.append(batch_pred_points)
            matched_tgt_points.append(batch_tgt_points)
            matched_l1_terms.append(torch.abs(batch_pred_points - batch_tgt_points).mean(dim=1))
            if use_attach_loss:
                attach_distance = self._attach_distance_for_batch(
                    batch_pred_points,
                    targets[batch_idx],
                    tgt_idx,
                    device=device,
                    num_curve_samples=int(loss_cfg.get('point_attach_num_curve_samples', 8)),
                )
                matched_attach_terms.append(attach_distance)
                point_count = int(targets[batch_idx]['points'].shape[0])
                default_degrees = torch.ones((point_count,), dtype=torch.long)
                default_loop_only = torch.zeros((point_count,), dtype=torch.bool)
                matched_degrees.append(targets[batch_idx].get('point_degree', default_degrees).to(device)[tgt_idx])
                matched_loop_only.append(targets[batch_idx].get('point_is_loop_only', default_loop_only).to(device)[tgt_idx])

        classification_terms = self.classification.compute(pred_logits, target_classes, query_weights=query_weights)
        class_loss_name = self.classification.loss_name()
        classification_loss = classification_terms[class_loss_name]
        if matched_pred_points:
            matched_pred_points = torch.cat(matched_pred_points, dim=0)
            matched_tgt_points = torch.cat(matched_tgt_points, dim=0)
            del matched_pred_points, matched_tgt_points
            l1_terms = torch.cat(matched_l1_terms, dim=0)
            if use_attach_loss:
                attach_terms = torch.cat(matched_attach_terms, dim=0) if matched_attach_terms else torch.zeros_like(l1_terms)
                degrees = torch.cat(matched_degrees, dim=0) if matched_degrees else torch.ones_like(l1_terms, dtype=torch.long)
                loop_only = torch.cat(matched_loop_only, dim=0) if matched_loop_only else torch.zeros_like(degrees, dtype=torch.bool)
                degree_threshold = int(loss_cfg.get('point_attach_degree_threshold', 3))
                low_degree_multiplier = float(loss_cfg.get('point_attach_low_degree_multiplier', 5.0))
                attach_weight = float(loss_cfg.get('point_attach_weight', 1.0))
                high_degree = (degrees >= degree_threshold) & (~loop_only)
                low_degree = (degrees < degree_threshold) & (~loop_only)
                l1_weights = torch.ones_like(l1_terms)
                l1_weights = torch.where(loop_only, torch.zeros_like(l1_weights), l1_weights)
                attach_weights = torch.ones_like(attach_terms)
                attach_weights = torch.where(low_degree | loop_only, attach_weights.new_full(attach_weights.shape, low_degree_multiplier), attach_weights)
                point_terms = l1_weights * l1_terms + attach_weight * attach_weights * attach_terms
                loss_point = point_terms.mean()
                loss_point_l1 = l1_terms.mean()
                loss_point_attach = attach_terms.mean()
                loss_point_loop_only_attach = attach_terms[loop_only].mean() if bool(loop_only.any()) else attach_terms.sum() * 0.0
                loss_point_low_degree_attach = attach_terms[low_degree].mean() if bool(low_degree.any()) else attach_terms.sum() * 0.0
                loss_point_high_degree_attach = attach_terms[high_degree].mean() if bool(high_degree.any()) else attach_terms.sum() * 0.0
                matched_loop_only_count = loop_only.to(dtype=l1_terms.dtype).sum()
                matched_low_degree_count = low_degree.to(dtype=l1_terms.dtype).sum()
                matched_high_degree_count = high_degree.to(dtype=l1_terms.dtype).sum()
            else:
                loss_point = l1_terms.mean()
                zero = l1_terms.sum() * 0.0
                loss_point_l1 = loss_point
                loss_point_attach = zero
                loss_point_loop_only_attach = zero
                loss_point_low_degree_attach = zero
                loss_point_high_degree_attach = zero
                matched_loop_only_count = zero
                matched_low_degree_count = zero
                matched_high_degree_count = zero
        else:
            loss_point = pred_points.sum() * 0.0
            loss_point_l1 = loss_point
            loss_point_attach = loss_point
            loss_point_loop_only_attach = loss_point
            loss_point_low_degree_attach = loss_point
            loss_point_high_degree_attach = loss_point
            matched_loop_only_count = loss_point
            matched_low_degree_count = loss_point
            matched_high_degree_count = loss_point

        total = (
            float(loss_cfg.get('ce_weight', 1.0)) * classification_loss
            + float(loss_cfg.get('point_weight', 5.0)) * loss_point
        )
        return {
            'loss_total': total,
            **classification_terms,
            'loss_point': loss_point,
            'loss_point_l1': loss_point_l1,
            'loss_point_attach': loss_point_attach,
            'loss_point_loop_only_attach': loss_point_loop_only_attach,
            'loss_point_low_degree_attach': loss_point_low_degree_attach,
            'loss_point_high_degree_attach': loss_point_high_degree_attach,
            'matched_loop_only_count': matched_loop_only_count,
            'matched_low_degree_count': matched_low_degree_count,
            'matched_high_degree_count': matched_high_degree_count,
        }

    def _attach_distance_for_batch(
        self,
        batch_pred_points: torch.Tensor,
        target: dict,
        tgt_idx: torch.Tensor,
        *,
        device: torch.device,
        num_curve_samples: int,
    ) -> torch.Tensor:
        if batch_pred_points.numel() == 0:
            return batch_pred_points.new_zeros((0,))
        target_curves = target.get('curves')
        if target_curves is None or target_curves.numel() == 0:
            return batch_pred_points.sum(dim=1) * 0.0
        full_offsets = target.get('point_curve_offsets')
        full_indices = target.get('point_curve_indices')
        if full_offsets is None or full_indices is None or full_offsets.numel() == 0:
            return batch_pred_points.sum(dim=1) * 0.0
        full_offsets = full_offsets.to(device)
        full_indices = full_indices.to(device)
        compact_offsets = [0]
        compact_indices = []
        for matched_target_idx in tgt_idx.to(device):
            start = int(full_offsets[matched_target_idx].item())
            end = int(full_offsets[matched_target_idx + 1].item())
            incident = full_indices[start:end]
            if incident.numel() > 0:
                compact_indices.append(incident)
            compact_offsets.append(compact_offsets[-1] + int(incident.numel()))
        point_curve_offsets = torch.as_tensor(compact_offsets, dtype=torch.long, device=device)
        if compact_indices:
            point_curve_indices = torch.cat(compact_indices, dim=0).to(dtype=torch.long, device=device)
        else:
            point_curve_indices = torch.zeros((0,), dtype=torch.long, device=device)
        target_curves_int = curve_external_to_internal(target_curves.to(device), self.config)
        return point_to_incident_curves_attach_distance(
            pred_points=batch_pred_points,
            target_curves=target_curves_int,
            point_curve_offsets=point_curve_offsets,
            point_curve_indices=point_curve_indices,
            num_curve_samples=num_curve_samples,
        )
