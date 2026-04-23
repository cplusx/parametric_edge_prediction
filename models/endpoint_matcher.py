from typing import List, Optional, Tuple

import torch
from scipy.optimize import linear_sum_assignment

from models.curve_coordinates import curve_external_to_internal


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

    def build_cost_matrix(self, logits: torch.Tensor, points: torch.Tensor, tgt_points: torch.Tensor) -> torch.Tensor:
        total = self.point_cost * torch.cdist(points, tgt_points, p=1)
        return total + self.edge_prob_cost * self._edge_prob_cost_matrix(logits, tgt_points.shape[0])

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
