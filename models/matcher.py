from typing import List, Optional, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from models.curve_coordinates import curve_external_to_internal
from models.geometry import pairwise_curve_chamfer_cost


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
        chamfer_cost: float = 5.0,
        chamfer_match_point_count: int = 20,
        use_edge_prob_cost_in_matching: bool = False,
        edge_prob_cost: float = 1.0,
        config: dict = None,
    ) -> None:
        self.chamfer_cost = float(chamfer_cost)
        self.chamfer_match_point_count = int(chamfer_match_point_count)
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
            chamfer_cost=float(loss_cfg.get('chamfer_cost', 5.0)),
            chamfer_match_point_count=int(loss_cfg.get('chamfer_match_point_count', 20)),
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
        chamfer_cost_matrix = pairwise_curve_chamfer_cost(
            curves,
            tgt_curves,
            point_count=self.chamfer_match_point_count,
        )
        total = self.chamfer_cost * chamfer_cost_matrix
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
    chamfer_cost: float,
    chamfer_match_point_count: int,
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
            chamfer_cost=chamfer_cost,
            chamfer_match_point_count=chamfer_match_point_count,
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
    chamfer_cost: float = 5.0,
    chamfer_match_point_count: int = 20,
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
            chamfer_cost=chamfer_cost,
            chamfer_match_point_count=chamfer_match_point_count,
            use_edge_prob_cost_in_matching=bool(use_edge_prob_cost_in_matching) if use_edge_prob_cost_in_matching is not None else False,
            edge_prob_cost=float(edge_prob_cost) if edge_prob_cost is not None else 1.0,
            config=config,
        )
    )
    return matcher(logits=logits, curves=curves, targets=targets)
