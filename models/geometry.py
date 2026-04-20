from typing import Tuple

import torch

from misc_utils.train_utils import sample_bezier_curves_torch

try:
    from models import native_cost_cpp
except Exception:
    native_cost_cpp = None


def curve_boxes_xyxy(curves: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        curves[..., 0].min(dim=-1).values,
        curves[..., 1].min(dim=-1).values,
        curves[..., 0].max(dim=-1).values,
        curves[..., 1].max(dim=-1).values,
    ], dim=-1)


def box_area_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    wh = (boxes[..., 2:] - boxes[..., :2]).clamp_min(0.0)
    return wh[..., 0] * wh[..., 1]


def pairwise_generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter_wh = (bottom_right - top_left).clamp_min(0.0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]

    area1 = box_area_xyxy(boxes1)
    area2 = box_area_xyxy(boxes2)
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp_min(1e-6)

    outer_top_left = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    outer_bottom_right = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    outer_wh = (outer_bottom_right - outer_top_left).clamp_min(0.0)
    outer_area = outer_wh[..., 0] * outer_wh[..., 1]
    return iou - (outer_area - union) / outer_area.clamp_min(1e-6)


def matched_generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0:
        return boxes1.new_zeros((0,))
    giou = pairwise_generalized_box_iou(boxes1, boxes2)
    diag = torch.arange(boxes1.shape[0], device=boxes1.device)
    return giou[diag, diag]


def _curve_arc_length(samples: torch.Tensor) -> torch.Tensor:
    if samples.shape[1] <= 1:
        return samples.new_zeros((samples.shape[0],))
    deltas = samples[:, 1:, :] - samples[:, :-1, :]
    return deltas.norm(dim=-1).sum(dim=-1)


def reverse_curve_points(curves: torch.Tensor) -> torch.Tensor:
    return torch.flip(curves, dims=(-2,))


def symmetric_curve_chamfer_distance(
    pred_curves: torch.Tensor,
    tgt_curves: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    if pred_curves.numel() == 0:
        return pred_curves.new_zeros((0,))
    pred_samples = sample_bezier_curves_torch(pred_curves, num_samples=num_samples)
    tgt_samples = sample_bezier_curves_torch(tgt_curves, num_samples=num_samples)
    pairwise = torch.cdist(pred_samples, tgt_samples, p=2.0)
    pred_to_tgt = pairwise.min(dim=2).values.mean(dim=1)
    tgt_to_pred = pairwise.min(dim=1).values.mean(dim=1)
    return pred_to_tgt + tgt_to_pred


def pairwise_curve_chamfer_cost(
    pred_curves: torch.Tensor,
    tgt_curves: torch.Tensor,
    point_count: int,
) -> torch.Tensor:
    if pred_curves.numel() == 0 or tgt_curves.numel() == 0:
        return pred_curves.new_zeros((pred_curves.shape[0], tgt_curves.shape[0]))
    point_count = max(2, int(point_count))
    pred_samples = sample_bezier_curves_torch(pred_curves, num_samples=point_count)
    tgt_samples = sample_bezier_curves_torch(tgt_curves, num_samples=point_count)
    if native_cost_cpp is not None and (not torch.is_grad_enabled()) and (not pred_curves.requires_grad) and (not tgt_curves.requires_grad):
        try:
            return native_cost_cpp.pairwise_curve_chamfer_from_samples(pred_samples, tgt_samples).to(device=pred_curves.device, dtype=pred_curves.dtype)
        except Exception:
            pass
    pairwise = torch.linalg.norm(
        pred_samples[:, None, :, None, :] - tgt_samples[None, :, None, :, :],
        dim=-1,
    )
    pred_to_tgt = pairwise.min(dim=3).values.mean(dim=2)
    tgt_to_pred = pairwise.min(dim=2).values.mean(dim=2)
    return 0.5 * (pred_to_tgt + tgt_to_pred)
