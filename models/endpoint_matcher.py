from typing import List, Optional, Tuple

import torch
from scipy.optimize import linear_sum_assignment

from models.curve_coordinates import curve_external_to_internal
from models.geometry import point_to_endpoint_incident_curve_distance_matrix


def _finite_debug_summary(name: str, tensor: torch.Tensor) -> str:
    flat = tensor.detach().reshape(-1)
    finite_mask = torch.isfinite(flat)
    finite_count = int(finite_mask.sum().item())
    total_count = int(flat.numel())
    if finite_count > 0:
        finite_vals = flat[finite_mask]
        min_val = float(finite_vals.min().item())
        max_val = float(finite_vals.max().item())
        mean_val = float(finite_vals.mean().item())
    else:
        min_val = float('nan')
        max_val = float('nan')
        mean_val = float('nan')
    return (
        f"{name}: shape={tuple(tensor.shape)} "
        f"finite={finite_count}/{total_count} "
        f"min={min_val:.6g} max={max_val:.6g} mean={mean_val:.6g}"
    )


class HungarianPointMatcher:
    def __init__(
        self,
        *,
        point_cost: float = 5.0,
        edge_prob_cost: float = 1.0,
        config: Optional[dict] = None,
    ) -> None:
        self.point_cost = float(point_cost)
        self.edge_prob_cost = float(edge_prob_cost)
        self.config = config
        loss_cfg = config.get('loss', {}) if isinstance(config, dict) else {}
        self.attach_matching_enabled = bool(loss_cfg.get('point_attach_matching_enabled', False))
        self.attach_matching_cost = float(loss_cfg.get('point_attach_matching_cost', 1.0))
        self.attach_low_degree_multiplier = float(loss_cfg.get('point_attach_low_degree_multiplier', 5.0))
        self.attach_degree_threshold = int(loss_cfg.get('point_attach_degree_threshold', 3))
        self.attach_num_curve_samples = int(loss_cfg.get('point_attach_num_curve_samples', 8))

    @classmethod
    def from_config(cls, config: dict) -> "HungarianPointMatcher":
        loss_cfg = config.get('loss', {})
        return cls(
            point_cost=float(loss_cfg.get('point_cost', 5.0)),
            edge_prob_cost=float(loss_cfg.get('edge_prob_cost', 1.0)),
            config=config,
        )

    @staticmethod
    def _edge_prob_cost_matrix(logits: torch.Tensor, target_count: int) -> torch.Tensor:
        if target_count <= 0:
            return logits.new_zeros((logits.shape[0], 0))
        edge_prob = logits.softmax(-1)[:, 0]
        return (-edge_prob)[:, None].expand(-1, target_count)

    def build_cost_matrix(
        self,
        logits: torch.Tensor,
        points: torch.Tensor,
        tgt_points: torch.Tensor,
        target: Optional[dict] = None,
    ) -> torch.Tensor:
        point_distance = torch.cdist(points, tgt_points, p=1)
        total = self.point_cost * point_distance
        if self.attach_matching_enabled and target is not None:
            attach_distance = self._attach_matching_distance_matrix(points, target)
            if attach_distance.shape == point_distance.shape:
                point_degree = target.get('point_degree')
                point_is_loop_only = target.get('point_is_loop_only')
                if point_degree is None:
                    point_degree = torch.ones((tgt_points.shape[0],), dtype=torch.long, device=points.device)
                else:
                    point_degree = point_degree.to(points.device)
                if point_is_loop_only is None:
                    point_is_loop_only = torch.zeros((tgt_points.shape[0],), dtype=torch.bool, device=points.device)
                else:
                    point_is_loop_only = point_is_loop_only.to(points.device)
                low_degree = (point_degree < self.attach_degree_threshold) & (~point_is_loop_only)
                attach_multiplier = torch.ones((tgt_points.shape[0],), dtype=points.dtype, device=points.device)
                attach_multiplier = torch.where(
                    low_degree | point_is_loop_only,
                    attach_multiplier.new_full(attach_multiplier.shape, self.attach_low_degree_multiplier),
                    attach_multiplier,
                )
                endpoint_total = point_distance + self.attach_matching_cost * attach_distance * attach_multiplier[None, :]
                endpoint_total = torch.where(point_is_loop_only[None, :], self.attach_matching_cost * attach_distance * attach_multiplier[None, :], endpoint_total)
                total = self.point_cost * endpoint_total
        return total + self.edge_prob_cost * self._edge_prob_cost_matrix(logits, tgt_points.shape[0])

    def _attach_matching_distance_matrix(self, points: torch.Tensor, target: dict) -> torch.Tensor:
        target_curves = target.get('curves')
        offsets = target.get('point_curve_offsets')
        indices = target.get('point_curve_indices')
        target_count = int(target['points'].shape[0])
        if target_curves is None or offsets is None or indices is None:
            return points.new_zeros((points.shape[0], target_count))
        if target_curves.numel() == 0 or offsets.numel() == 0:
            return points.new_zeros((points.shape[0], target_count))
        target_curves_int = curve_external_to_internal(target_curves.to(points.device), self.config) if self.config is not None else target_curves.to(points.device)
        return point_to_endpoint_incident_curve_distance_matrix(
            pred_points=points,
            target_curves=target_curves_int,
            point_curve_offsets=offsets.to(points.device),
            point_curve_indices=indices.to(points.device),
            num_curve_samples=self.attach_num_curve_samples,
        )

    @torch.no_grad()
    def __call__(
        self,
        logits: torch.Tensor,
        points: torch.Tensor,
        targets: List[dict],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        indices = []
        for batch_idx, target in enumerate(targets):
            tgt_points = target['points'].to(points.device)
            if self.config is not None:
                tgt_points = curve_external_to_internal(tgt_points, self.config)
            if tgt_points.numel() == 0:
                indices.append((
                    torch.empty(0, dtype=torch.long, device=points.device),
                    torch.empty(0, dtype=torch.long, device=points.device),
                ))
                continue
            total_cost = self.build_cost_matrix(
                logits=logits[batch_idx],
                points=points[batch_idx],
                tgt_points=tgt_points,
                target=target,
            )
            if not torch.isfinite(total_cost).all():
                sample_id = target.get('sample_id', f'batch_{batch_idx}')
                raise ValueError(
                    'Hungarian point cost matrix contains invalid numeric entries.\n'
                    f'sample_id={sample_id}\n'
                    + _finite_debug_summary('pred_logits', logits[batch_idx]) + '\n'
                    + _finite_debug_summary('pred_points', points[batch_idx]) + '\n'
                    + _finite_debug_summary('tgt_points', tgt_points) + '\n'
                    + _finite_debug_summary('total_cost', total_cost)
                )
            row_ind, col_ind = linear_sum_assignment(total_cost.detach().cpu().numpy())
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.long, device=points.device),
                torch.as_tensor(col_ind, dtype=torch.long, device=points.device),
            ))
        return indices
