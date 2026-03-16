from typing import List, Tuple

import torch
from scipy.optimize import linear_sum_assignment

from misc_utils.train_utils import sample_bezier_curves_torch
from models.geometry import curve_boxes_xyxy, pairwise_curve_chamfer_cost, pairwise_generalized_box_iou


def _build_cost_matrix(
    logits: torch.Tensor,
    curves: torch.Tensor,
    tgt_curves: torch.Tensor,
    tgt_boxes: torch.Tensor,
    control_cost: float,
    sample_cost: float,
    box_cost: float,
    giou_cost: float,
    curve_distance_cost: float,
    curve_match_point_count: int,
    num_curve_samples: int,
    valid_query_count: int = -1,
) -> torch.Tensor:
    num_queries = logits.shape[0]
    sampled_pred = sample_bezier_curves_torch(curves, num_samples=num_curve_samples)
    ctrl_cost = torch.cdist(curves.reshape(num_queries, -1), tgt_curves.reshape(tgt_curves.shape[0], -1), p=1)
    sampled_tgt = sample_bezier_curves_torch(tgt_curves, num_samples=num_curve_samples)
    sample_cost_matrix = torch.cdist(sampled_pred.reshape(num_queries, -1), sampled_tgt.reshape(tgt_curves.shape[0], -1), p=1)
    pred_boxes = curve_boxes_xyxy(curves)
    bbox_cost = torch.cdist(pred_boxes, tgt_boxes, p=1)
    giou_cost_matrix = 1.0 - pairwise_generalized_box_iou(pred_boxes, tgt_boxes)
    curve_distance_cost_matrix = pairwise_curve_chamfer_cost(curves, tgt_curves, point_count=curve_match_point_count)
    total = (
        control_cost * ctrl_cost
        + sample_cost * sample_cost_matrix
        + box_cost * bbox_cost
        + giou_cost * giou_cost_matrix
        + curve_distance_cost * curve_distance_cost_matrix
    )
    if valid_query_count > 0 and valid_query_count < num_queries:
        total[valid_query_count:] = total[valid_query_count:] + 1e6
    return total


def build_curve_cost_matrix(
    logits: torch.Tensor,
    curves: torch.Tensor,
    tgt_curves: torch.Tensor,
    tgt_boxes: torch.Tensor,
    control_cost: float,
    sample_cost: float,
    box_cost: float,
    giou_cost: float,
    curve_distance_cost: float,
    curve_match_point_count: int,
    num_curve_samples: int,
    valid_query_count: int = -1,
) -> torch.Tensor:
    return _build_cost_matrix(
        logits=logits,
        curves=curves,
        tgt_curves=tgt_curves,
        tgt_boxes=tgt_boxes,
        control_cost=control_cost,
        sample_cost=sample_cost,
        box_cost=box_cost,
        giou_cost=giou_cost,
        curve_distance_cost=curve_distance_cost,
        curve_match_point_count=curve_match_point_count,
        num_curve_samples=num_curve_samples,
        valid_query_count=valid_query_count,
    )


@torch.no_grad()
def hungarian_curve_matching(
    logits: torch.Tensor,
    curves: torch.Tensor,
    targets: List[dict],
    control_cost: float = 5.0,
    sample_cost: float = 2.0,
    box_cost: float = 1.0,
    giou_cost: float = 1.0,
    curve_distance_cost: float = 0.0,
    curve_match_point_count: int = 4,
    num_curve_samples: int = 16,
    active_counts: torch.Tensor = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    batch_size, num_queries = logits.shape[:2]
    indices = []
    for batch_idx, target in enumerate(targets):
        tgt_curves = target['curves'].to(curves.device)
        tgt_boxes = target['boxes'].to(curves.device)
        if tgt_curves.numel() == 0:
            indices.append((torch.empty(0, dtype=torch.long, device=curves.device), torch.empty(0, dtype=torch.long, device=curves.device)))
            continue
        total_cost = _build_cost_matrix(
            logits=logits[batch_idx],
            curves=curves[batch_idx],
            tgt_curves=tgt_curves,
            tgt_boxes=tgt_boxes,
            control_cost=control_cost,
            sample_cost=sample_cost,
            box_cost=box_cost,
            giou_cost=giou_cost,
            curve_distance_cost=curve_distance_cost,
            curve_match_point_count=curve_match_point_count,
            num_curve_samples=num_curve_samples,
            valid_query_count=int(active_counts[batch_idx].item()) if active_counts is not None else -1,
        )
        row_ind, col_ind = linear_sum_assignment(total_cost.detach().cpu().numpy())
        indices.append((torch.as_tensor(row_ind, dtype=torch.long, device=curves.device), torch.as_tensor(col_ind, dtype=torch.long, device=curves.device)))
    return indices


@torch.no_grad()
def topk_curve_positive_indices(
    logits: torch.Tensor,
    curves: torch.Tensor,
    targets: List[dict],
    topk_per_target: int,
    control_cost: float = 5.0,
    sample_cost: float = 2.0,
    box_cost: float = 1.0,
    giou_cost: float = 1.0,
    curve_distance_cost: float = 0.0,
    curve_match_point_count: int = 4,
    num_curve_samples: int = 16,
    active_counts: torch.Tensor = None,
) -> List[torch.Tensor]:
    positives = []
    for batch_idx, target in enumerate(targets):
        tgt_curves = target['curves'].to(curves.device)
        tgt_boxes = target['boxes'].to(curves.device)
        if tgt_curves.numel() == 0 or topk_per_target <= 0:
            positives.append(torch.empty(0, dtype=torch.long, device=curves.device))
            continue
        total_cost = _build_cost_matrix(
            logits=logits[batch_idx],
            curves=curves[batch_idx],
            tgt_curves=tgt_curves,
            tgt_boxes=tgt_boxes,
            control_cost=control_cost,
            sample_cost=sample_cost,
            box_cost=box_cost,
            giou_cost=giou_cost,
            curve_distance_cost=curve_distance_cost,
            curve_match_point_count=curve_match_point_count,
            num_curve_samples=num_curve_samples,
            valid_query_count=int(active_counts[batch_idx].item()) if active_counts is not None else -1,
        )
        chosen = []
        used = set()
        for tgt_idx in range(total_cost.shape[1]):
            ordering = torch.argsort(total_cost[:, tgt_idx], dim=0)
            picked_for_target = 0
            for src_idx in ordering.tolist():
                if src_idx in used:
                    continue
                chosen.append(src_idx)
                used.add(src_idx)
                picked_for_target += 1
                if picked_for_target >= topk_per_target:
                    break
        positives.append(torch.tensor(chosen, dtype=torch.long, device=curves.device))
    return positives
