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


def pairwise_curve_l1_forward_reverse_cost(
    pred_curves: torch.Tensor,
    tgt_curves: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if pred_curves.numel() == 0 or tgt_curves.numel() == 0:
        zero = pred_curves.new_zeros((pred_curves.shape[0], tgt_curves.shape[0]))
        return zero, zero
    pred_flat = pred_curves.reshape(pred_curves.shape[0], -1)
    tgt_flat = tgt_curves.reshape(tgt_curves.shape[0], -1)
    tgt_rev_flat = reverse_curve_points(tgt_curves).reshape(tgt_curves.shape[0], -1)
    forward = torch.cdist(pred_flat, tgt_flat, p=1)
    reverse = torch.cdist(pred_flat, tgt_rev_flat, p=1)
    return forward, reverse


def pairwise_sample_l1_forward_reverse_cost(
    pred_curves: torch.Tensor,
    tgt_curves: torch.Tensor,
    num_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if pred_curves.numel() == 0 or tgt_curves.numel() == 0:
        zero = pred_curves.new_zeros((pred_curves.shape[0], tgt_curves.shape[0]))
        return zero, zero
    pred_samples = sample_bezier_curves_torch(pred_curves, num_samples=num_samples).reshape(pred_curves.shape[0], -1)
    tgt_samples = sample_bezier_curves_torch(tgt_curves, num_samples=num_samples)
    tgt_forward = tgt_samples.reshape(tgt_curves.shape[0], -1)
    tgt_reverse = torch.flip(tgt_samples, dims=(-2,)).reshape(tgt_curves.shape[0], -1)
    forward = torch.cdist(pred_samples, tgt_forward, p=1)
    reverse = torch.cdist(pred_samples, tgt_reverse, p=1)
    return forward, reverse


def aligned_curve_l1(
    pred_curves: torch.Tensor,
    tgt_curves: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if pred_curves.numel() == 0:
        zero = pred_curves.new_zeros((0,))
        return zero, pred_curves
    tgt_reversed = reverse_curve_points(tgt_curves)
    forward = torch.abs(pred_curves - tgt_curves).mean(dim=(1, 2))
    reverse = torch.abs(pred_curves - tgt_reversed).mean(dim=(1, 2))
    use_reverse = reverse < forward
    best = torch.where(use_reverse, reverse, forward)
    oriented_target = torch.where(
        use_reverse[:, None, None],
        tgt_reversed,
        tgt_curves,
    )
    return best, oriented_target


def aligned_curve_endpoint_l1(
    pred_curves: torch.Tensor,
    tgt_curves: torch.Tensor,
    ctrl_weight: float = 1.0,
    endpoint_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pred_curves.numel() == 0:
        zero = pred_curves.new_zeros((0,))
        return zero, zero, pred_curves
    tgt_reversed = reverse_curve_points(tgt_curves)
    ctrl_forward = torch.abs(pred_curves - tgt_curves).mean(dim=(1, 2))
    ctrl_reverse = torch.abs(pred_curves - tgt_reversed).mean(dim=(1, 2))
    pred_endpoints = pred_curves[:, [0, -1]]
    endpoint_forward = torch.abs(pred_endpoints - tgt_curves[:, [0, -1]]).mean(dim=(1, 2))
    endpoint_reverse = torch.abs(pred_endpoints - tgt_reversed[:, [0, -1]]).mean(dim=(1, 2))
    forward_total = float(ctrl_weight) * ctrl_forward + float(endpoint_weight) * endpoint_forward
    reverse_total = float(ctrl_weight) * ctrl_reverse + float(endpoint_weight) * endpoint_reverse
    use_reverse = reverse_total < forward_total
    oriented_target = torch.where(use_reverse[:, None, None], tgt_reversed, tgt_curves)
    ctrl_best = torch.where(use_reverse, ctrl_reverse, ctrl_forward)
    endpoint_best = torch.where(use_reverse, endpoint_reverse, endpoint_forward)
    return ctrl_best, endpoint_best, oriented_target


def aligned_endpoint_l1(
    pred_curves: torch.Tensor,
    tgt_curves: torch.Tensor,
) -> torch.Tensor:
    if pred_curves.numel() == 0:
        return pred_curves.new_zeros((0,))
    tgt_reversed = reverse_curve_points(tgt_curves)
    pred_endpoints = pred_curves[:, [0, -1]]
    forward = torch.abs(pred_endpoints - tgt_curves[:, [0, -1]]).mean(dim=(1, 2))
    reverse = torch.abs(pred_endpoints - tgt_reversed[:, [0, -1]]).mean(dim=(1, 2))
    return torch.minimum(forward, reverse)


def pairwise_aligned_curve_l1_cost(
    pred_curves: torch.Tensor,
    tgt_curves: torch.Tensor,
) -> torch.Tensor:
    if pred_curves.numel() == 0 or tgt_curves.numel() == 0:
        return pred_curves.new_zeros((pred_curves.shape[0], tgt_curves.shape[0]))
    if native_cost_cpp is not None and (not torch.is_grad_enabled()) and (not pred_curves.requires_grad) and (not tgt_curves.requires_grad):
        try:
            return native_cost_cpp.aligned_curve_l1_cost(pred_curves, tgt_curves).to(device=pred_curves.device, dtype=pred_curves.dtype)
        except Exception:
            pass
    forward, reverse = pairwise_curve_l1_forward_reverse_cost(pred_curves, tgt_curves)
    return torch.minimum(forward, reverse)


def pairwise_aligned_sample_l1_cost(
    pred_curves: torch.Tensor,
    tgt_curves: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    if pred_curves.numel() == 0 or tgt_curves.numel() == 0:
        return pred_curves.new_zeros((pred_curves.shape[0], tgt_curves.shape[0]))
    if native_cost_cpp is not None and (not torch.is_grad_enabled()) and (not pred_curves.requires_grad) and (not tgt_curves.requires_grad):
        try:
            pred_samples = sample_bezier_curves_torch(pred_curves, num_samples=num_samples)
            tgt_samples = sample_bezier_curves_torch(tgt_curves, num_samples=num_samples)
            return native_cost_cpp.aligned_sample_l1_cost(pred_samples, tgt_samples).to(device=pred_curves.device, dtype=pred_curves.dtype)
        except Exception:
            pass
    forward, reverse = pairwise_sample_l1_forward_reverse_cost(pred_curves, tgt_curves, num_samples=num_samples)
    return torch.minimum(forward, reverse)


def symmetric_curve_distance(
    pred_curves: torch.Tensor,
    tgt_curves: torch.Tensor,
    num_samples: int,
    length_weight: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pred_curves.numel() == 0:
        zero = pred_curves.new_zeros((0,))
        return zero, zero, zero
    pred_samples = sample_bezier_curves_torch(pred_curves, num_samples=num_samples)
    tgt_samples = sample_bezier_curves_torch(tgt_curves, num_samples=num_samples)
    pairwise = torch.cdist(pred_samples, tgt_samples)
    chamfer = 0.5 * (
        pairwise.min(dim=2).values.mean(dim=1) +
        pairwise.min(dim=1).values.mean(dim=1)
    )
    pred_length = _curve_arc_length(pred_samples)
    tgt_length = _curve_arc_length(tgt_samples)
    length_delta = torch.abs(pred_length - tgt_length)
    total = chamfer + float(length_weight) * length_delta
    return total, chamfer, length_delta


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
