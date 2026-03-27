from typing import List, Optional, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from misc_utils.train_utils import sample_bezier_curves_torch
from models.curve_coordinates import curve_external_to_internal
from models.geometry import (
    pairwise_curve_l1_forward_reverse_cost,
    pairwise_endpoint_l1_forward_reverse_cost,
    pairwise_sample_l1_forward_reverse_cost,
    pairwise_curve_chamfer_cost,
)


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


class HungarianCurveMatcher:
    def __init__(
        self,
        *,
        control_cost: float = 5.0,
        endpoint_cost: float = 5.0,
        sample_cost: float = 0.0,
        curve_distance_cost: float = 0.0,
        curve_match_point_count: int = 4,
        num_curve_samples: int = 16,
        direction_invariant: bool = True,
        use_edge_prob_cost_in_matching: bool = False,
        edge_prob_cost: float = 1.0,
        config: dict = None,
    ) -> None:
        self.control_cost = float(control_cost)
        self.endpoint_cost = float(endpoint_cost)
        self.sample_cost = float(sample_cost)
        self.curve_distance_cost = float(curve_distance_cost)
        self.curve_match_point_count = int(curve_match_point_count)
        self.num_curve_samples = int(num_curve_samples)
        self.direction_invariant = bool(direction_invariant)
        self.use_edge_prob_cost_in_matching = bool(use_edge_prob_cost_in_matching)
        self.edge_prob_cost = float(edge_prob_cost)
        self.config = config

    @classmethod
    def from_config(
        cls,
        config: dict,
        *,
        use_edge_prob_cost_in_matching: Optional[bool] = None,
        edge_prob_cost: Optional[float] = None,
    ) -> "HungarianCurveMatcher":
        loss_cfg = config.get('loss', {})
        return cls(
            control_cost=float(loss_cfg.get('control_cost', 5.0)),
            endpoint_cost=float(loss_cfg.get('endpoint_cost', 5.0)),
            sample_cost=float(loss_cfg.get('sample_cost', 0.0)),
            curve_distance_cost=float(loss_cfg.get('curve_distance_cost', 0.0)),
            curve_match_point_count=int(loss_cfg.get('curve_match_point_count', 4)),
            num_curve_samples=int(loss_cfg.get('num_curve_samples', 16)),
            direction_invariant=bool(loss_cfg.get('direction_invariant', True)),
            use_edge_prob_cost_in_matching=(
                bool(loss_cfg.get('use_edge_prob_cost_in_matching', False))
                if use_edge_prob_cost_in_matching is None
                else bool(use_edge_prob_cost_in_matching)
            ),
            edge_prob_cost=(
                float(loss_cfg.get('edge_prob_cost', 1.0))
                if edge_prob_cost is None
                else float(edge_prob_cost)
            ),
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
        curves: torch.Tensor,
        tgt_curves: torch.Tensor,
    ) -> torch.Tensor:
        if self.direction_invariant:
            ctrl_forward, ctrl_reverse = pairwise_curve_l1_forward_reverse_cost(curves, tgt_curves)
            endpoint_forward, endpoint_reverse = pairwise_endpoint_l1_forward_reverse_cost(curves, tgt_curves)
            sample_forward, sample_reverse = pairwise_sample_l1_forward_reverse_cost(
                curves,
                tgt_curves,
                num_samples=self.num_curve_samples,
            )
        else:
            ctrl_cost = torch.cdist(curves.reshape(curves.shape[0], -1), tgt_curves.reshape(tgt_curves.shape[0], -1), p=1)
            endpoint_cost_matrix = torch.cdist(
                curves[:, [0, -1]].reshape(curves.shape[0], -1),
                tgt_curves[:, [0, -1]].reshape(tgt_curves.shape[0], -1),
                p=1,
            )
            sampled_pred = sample_bezier_curves_torch(curves, num_samples=self.num_curve_samples).reshape(curves.shape[0], -1)
            sampled_tgt = sample_bezier_curves_torch(tgt_curves, num_samples=self.num_curve_samples).reshape(tgt_curves.shape[0], -1)
            sample_cost_matrix = torch.cdist(sampled_pred, sampled_tgt, p=1)
        curve_distance_cost_matrix = pairwise_curve_chamfer_cost(
            curves,
            tgt_curves,
            point_count=self.curve_match_point_count,
        )
        if self.direction_invariant:
            oriented_cost = torch.minimum(
                self.control_cost * ctrl_forward + self.endpoint_cost * endpoint_forward + self.sample_cost * sample_forward,
                self.control_cost * ctrl_reverse + self.endpoint_cost * endpoint_reverse + self.sample_cost * sample_reverse,
            )
            total = oriented_cost + self.curve_distance_cost * curve_distance_cost_matrix
        else:
            total = (
                self.control_cost * ctrl_cost
                + self.endpoint_cost * endpoint_cost_matrix
                + self.sample_cost * sample_cost_matrix
                + self.curve_distance_cost * curve_distance_cost_matrix
            )
        if self.use_edge_prob_cost_in_matching:
            total = total + self.edge_prob_cost * self._edge_prob_cost_matrix(logits, tgt_curves.shape[0])
        return total

    @torch.no_grad()
    def __call__(
        self,
        logits: torch.Tensor,
        curves: torch.Tensor,
        targets: List[dict],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        indices = []
        for batch_idx, target in enumerate(targets):
            tgt_curves = target['curves'].to(curves.device)
            if self.config is not None:
                tgt_curves = curve_external_to_internal(tgt_curves, self.config)
            if tgt_curves.numel() == 0:
                indices.append((
                    torch.empty(0, dtype=torch.long, device=curves.device),
                    torch.empty(0, dtype=torch.long, device=curves.device),
                ))
                continue
            total_cost = self.build_cost_matrix(
                logits=logits[batch_idx],
                curves=curves[batch_idx],
                tgt_curves=tgt_curves,
            )
            if not torch.isfinite(total_cost).all():
                sample_id = target.get('sample_id', f'batch_{batch_idx}')
                raise ValueError(
                    'Hungarian cost matrix contains invalid numeric entries.\n'
                    f'sample_id={sample_id}\n'
                    + _finite_debug_summary('pred_logits', logits[batch_idx]) + '\n'
                    + _finite_debug_summary('pred_curves', curves[batch_idx]) + '\n'
                    + _finite_debug_summary('tgt_curves', tgt_curves) + '\n'
                    + _finite_debug_summary('total_cost', total_cost)
                )
            row_ind, col_ind = linear_sum_assignment(total_cost.detach().cpu().numpy())
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.long, device=curves.device),
                torch.as_tensor(col_ind, dtype=torch.long, device=curves.device),
            ))
        return indices


def build_curve_cost_matrix(
    logits: torch.Tensor,
    curves: torch.Tensor,
    tgt_curves: torch.Tensor,
    control_cost: float,
    endpoint_cost: float,
    sample_cost: float,
    curve_distance_cost: float,
    curve_match_point_count: int,
    num_curve_samples: int,
    direction_invariant: bool = True,
    use_edge_prob_cost_in_matching: Optional[bool] = None,
    edge_prob_cost: Optional[float] = None,
    config: dict = None,
) -> torch.Tensor:
    matcher = (
        HungarianCurveMatcher.from_config(
            config,
            use_edge_prob_cost_in_matching=use_edge_prob_cost_in_matching,
            edge_prob_cost=edge_prob_cost,
        )
        if config is not None and (use_edge_prob_cost_in_matching is None or edge_prob_cost is None)
        else HungarianCurveMatcher(
            control_cost=control_cost,
            endpoint_cost=endpoint_cost,
            sample_cost=sample_cost,
            curve_distance_cost=curve_distance_cost,
            curve_match_point_count=curve_match_point_count,
            num_curve_samples=num_curve_samples,
            direction_invariant=direction_invariant,
            use_edge_prob_cost_in_matching=bool(use_edge_prob_cost_in_matching) if use_edge_prob_cost_in_matching is not None else False,
            edge_prob_cost=float(edge_prob_cost) if edge_prob_cost is not None else 1.0,
            config=config,
        )
    )
    return matcher.build_cost_matrix(logits=logits, curves=curves, tgt_curves=tgt_curves)


@torch.no_grad()
def hungarian_curve_matching(
    logits: torch.Tensor,
    curves: torch.Tensor,
    targets: List[dict],
    control_cost: float = 5.0,
    endpoint_cost: float = 5.0,
    sample_cost: float = 0.0,
    curve_distance_cost: float = 0.0,
    curve_match_point_count: int = 4,
    num_curve_samples: int = 16,
    direction_invariant: bool = True,
    use_edge_prob_cost_in_matching: Optional[bool] = None,
    edge_prob_cost: Optional[float] = None,
    config: dict = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    matcher = (
        HungarianCurveMatcher.from_config(
            config,
            use_edge_prob_cost_in_matching=use_edge_prob_cost_in_matching,
            edge_prob_cost=edge_prob_cost,
        )
        if config is not None and (use_edge_prob_cost_in_matching is None or edge_prob_cost is None)
        else HungarianCurveMatcher(
            control_cost=control_cost,
            endpoint_cost=endpoint_cost,
            sample_cost=sample_cost,
            curve_distance_cost=curve_distance_cost,
            curve_match_point_count=curve_match_point_count,
            num_curve_samples=num_curve_samples,
            direction_invariant=direction_invariant,
            use_edge_prob_cost_in_matching=bool(use_edge_prob_cost_in_matching) if use_edge_prob_cost_in_matching is not None else False,
            edge_prob_cost=float(edge_prob_cost) if edge_prob_cost is not None else 1.0,
            config=config,
        )
    )
    return matcher(logits=logits, curves=curves, targets=targets)
