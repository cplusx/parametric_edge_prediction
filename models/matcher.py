from typing import List, Tuple

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


def _build_cost_matrix(
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
) -> torch.Tensor:
    if direction_invariant:
        ctrl_forward, ctrl_reverse = pairwise_curve_l1_forward_reverse_cost(curves, tgt_curves)
        endpoint_forward, endpoint_reverse = pairwise_endpoint_l1_forward_reverse_cost(curves, tgt_curves)
        sample_forward, sample_reverse = pairwise_sample_l1_forward_reverse_cost(
            curves,
            tgt_curves,
            num_samples=num_curve_samples,
        )
    else:
        ctrl_cost = torch.cdist(curves.reshape(curves.shape[0], -1), tgt_curves.reshape(tgt_curves.shape[0], -1), p=1)
        endpoint_cost_matrix = torch.cdist(
            curves[:, [0, -1]].reshape(curves.shape[0], -1),
            tgt_curves[:, [0, -1]].reshape(tgt_curves.shape[0], -1),
            p=1,
        )
        sampled_pred = sample_bezier_curves_torch(curves, num_samples=num_curve_samples).reshape(curves.shape[0], -1)
        sampled_tgt = sample_bezier_curves_torch(tgt_curves, num_samples=num_curve_samples).reshape(tgt_curves.shape[0], -1)
        sample_cost_matrix = torch.cdist(sampled_pred, sampled_tgt, p=1)
    curve_distance_cost_matrix = pairwise_curve_chamfer_cost(curves, tgt_curves, point_count=curve_match_point_count)
    if direction_invariant:
        oriented_cost = torch.minimum(
            control_cost * ctrl_forward + endpoint_cost * endpoint_forward + sample_cost * sample_forward,
            control_cost * ctrl_reverse + endpoint_cost * endpoint_reverse + sample_cost * sample_reverse,
        )
        total = oriented_cost + curve_distance_cost * curve_distance_cost_matrix
    else:
        total = (
            control_cost * ctrl_cost
            + endpoint_cost * endpoint_cost_matrix
            + sample_cost * sample_cost_matrix
            + curve_distance_cost * curve_distance_cost_matrix
        )
    return total


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
) -> torch.Tensor:
    return _build_cost_matrix(
        logits=logits,
        curves=curves,
        tgt_curves=tgt_curves,
        control_cost=control_cost,
        endpoint_cost=endpoint_cost,
        sample_cost=sample_cost,
        curve_distance_cost=curve_distance_cost,
        curve_match_point_count=curve_match_point_count,
        num_curve_samples=num_curve_samples,
        direction_invariant=direction_invariant,
    )


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
    config: dict = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    indices = []
    for batch_idx, target in enumerate(targets):
        tgt_curves = target['curves'].to(curves.device)
        if config is not None:
            tgt_curves = curve_external_to_internal(tgt_curves, config)
        if tgt_curves.numel() == 0:
            indices.append((torch.empty(0, dtype=torch.long, device=curves.device), torch.empty(0, dtype=torch.long, device=curves.device)))
            continue
        total_cost = _build_cost_matrix(
            logits=logits[batch_idx],
            curves=curves[batch_idx],
            tgt_curves=tgt_curves,
            control_cost=control_cost,
            endpoint_cost=endpoint_cost,
            sample_cost=sample_cost,
            curve_distance_cost=curve_distance_cost,
            curve_match_point_count=curve_match_point_count,
            num_curve_samples=num_curve_samples,
            direction_invariant=direction_invariant,
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
        indices.append((torch.as_tensor(row_ind, dtype=torch.long, device=curves.device), torch.as_tensor(col_ind, dtype=torch.long, device=curves.device)))
    return indices
