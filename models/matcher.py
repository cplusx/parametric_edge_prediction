from typing import Dict, List, Optional, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from models.curve_coordinates import curve_external_to_internal
from models.curve_distances import (
    build_curve_distance_from_config,
    closed_curve_endpoint_dist_threshold_from_config,
    curve_distance_type_from_config,
    curve_matching_cost_weight_from_config,
    infer_closed_curves,
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
        curve_cost: float = 5.0,
        edge_prob_cost: float = 1.0,
        config: dict = None,
    ) -> None:
        self.curve_cost = float(curve_cost)
        self.edge_prob_cost = float(edge_prob_cost)
        self.config = config
        self.curve_distance_type = curve_distance_type_from_config(config or {})
        self.closed_curve_endpoint_dist_threshold = closed_curve_endpoint_dist_threshold_from_config(config or {})
        self.curve_distance = build_curve_distance_from_config(config or {}, for_matching=True)

    @classmethod
    def from_config(
        cls,
        config: dict,
        *,
        edge_prob_cost: Optional[float] = None,
    ) -> "HungarianCurveMatcher":
        loss_cfg = config.get('loss', {})
        return cls(
            curve_cost=curve_matching_cost_weight_from_config(config),
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
        return self.build_cost_components(logits=logits, curves=curves, tgt_curves=tgt_curves)["total"]

    def build_cost_components(
        self,
        logits: torch.Tensor,
        curves: torch.Tensor,
        tgt_curves: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        target_is_closed = infer_closed_curves(
            tgt_curves,
            threshold=self.closed_curve_endpoint_dist_threshold,
        )
        curve_components = self.curve_distance.pairwise_cost_from_curves(
            curves,
            tgt_curves,
            target_is_closed=target_is_closed,
        )
        curve_cost_matrix = curve_components["total"]
        curve_weighted = self.curve_cost * curve_cost_matrix
        edge_prob_raw = self._edge_prob_cost_matrix(logits, tgt_curves.shape[0])
        edge_prob_weighted = self.edge_prob_cost * edge_prob_raw
        total = curve_weighted + edge_prob_weighted
        out = {
            "curve_raw": curve_cost_matrix,
            "curve": curve_weighted,
            "edge_prob_raw": edge_prob_raw,
            "edge_prob": edge_prob_weighted,
            "total": total,
        }
        if self.curve_distance_type == "emd":
            out["emd_raw"] = curve_cost_matrix
            out["emd"] = curve_weighted
        else:
            out["chamfer_raw"] = curve_cost_matrix
            out["chamfer"] = curve_weighted
        return out

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
    chamfer_cost: float,
    chamfer_match_point_count: int,
    edge_prob_cost: Optional[float] = None,
    config: dict = None,
) -> torch.Tensor:
    matcher = (
        HungarianCurveMatcher.from_config(
            config,
            edge_prob_cost=edge_prob_cost,
        )
        if config is not None and edge_prob_cost is None
        else HungarianCurveMatcher(
            curve_cost=chamfer_cost,
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
    chamfer_cost: float = 5.0,
    chamfer_match_point_count: int = 20,
    edge_prob_cost: Optional[float] = None,
    config: dict = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    matcher = (
        HungarianCurveMatcher.from_config(
            config,
            edge_prob_cost=edge_prob_cost,
        )
        if config is not None and edge_prob_cost is None
        else HungarianCurveMatcher(
            curve_cost=chamfer_cost,
            edge_prob_cost=float(edge_prob_cost) if edge_prob_cost is not None else 1.0,
            config=config,
        )
    )
    return matcher(logits=logits, curves=curves, targets=targets)
