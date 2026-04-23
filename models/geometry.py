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


def sample_curve_segments(curves: torch.Tensor, num_samples: int) -> torch.Tensor:
    if curves.numel() == 0:
        return curves.new_zeros((curves.shape[0] if curves.ndim > 0 else 0, 0, 2, 2))
    num_samples = max(2, int(num_samples))
    samples = sample_bezier_curves_torch(curves, num_samples=num_samples)
    return torch.stack([samples[:, :-1, :], samples[:, 1:, :]], dim=2)


def point_to_segments_distance(points: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    if points.numel() == 0:
        return points.new_zeros((points.shape[0],) + tuple(segments.shape[:-2]))
    if segments.numel() == 0:
        return points.new_zeros((points.shape[0],) + tuple(segments.shape[:-2]))
    starts = segments[..., 0, :]
    ends = segments[..., 1, :]
    segment_vec = ends - starts
    point_vec = points.view(points.shape[0], *([1] * (starts.ndim - 1)), 2) - starts.unsqueeze(0)
    denom = (segment_vec * segment_vec).sum(dim=-1).clamp_min(1e-12)
    t = (point_vec * segment_vec.unsqueeze(0)).sum(dim=-1) / denom.unsqueeze(0)
    projection = starts.unsqueeze(0) + t.clamp(0.0, 1.0).unsqueeze(-1) * segment_vec.unsqueeze(0)
    return (points.view(points.shape[0], *([1] * (starts.ndim - 1)), 2) - projection).norm(dim=-1)


def point_to_curve_distance_matrix(
    points: torch.Tensor,
    target_curves: torch.Tensor,
    num_curve_samples: int,
) -> torch.Tensor:
    if points.numel() == 0:
        curve_count = int(target_curves.shape[0]) if target_curves.ndim >= 1 else 0
        return points.new_zeros((points.shape[0], curve_count))
    if target_curves.numel() == 0:
        return points.new_zeros((points.shape[0], 0))
    segments = sample_curve_segments(target_curves, num_samples=num_curve_samples)
    segment_distances = point_to_segments_distance(points, segments)
    return segment_distances.min(dim=2).values


def point_to_endpoint_incident_curve_distance_matrix(
    pred_points: torch.Tensor,
    target_curves: torch.Tensor,
    point_curve_offsets: torch.Tensor,
    point_curve_indices: torch.Tensor,
    num_curve_samples: int,
) -> torch.Tensor:
    if pred_points.numel() == 0:
        target_count = max(0, int(point_curve_offsets.numel()) - 1)
        return pred_points.new_zeros((pred_points.shape[0], target_count))
    target_count = max(0, int(point_curve_offsets.numel()) - 1)
    if target_count == 0:
        return pred_points.new_zeros((pred_points.shape[0], 0))
    if target_curves.numel() == 0:
        return pred_points.new_zeros((pred_points.shape[0], target_count))
    curve_distances = point_to_curve_distance_matrix(
        pred_points,
        target_curves,
        num_curve_samples=num_curve_samples,
    )
    out = pred_points.new_zeros((pred_points.shape[0], target_count))
    curve_count = int(target_curves.shape[0])
    for point_idx in range(target_count):
        start = int(point_curve_offsets[point_idx].item())
        end = int(point_curve_offsets[point_idx + 1].item())
        incident = point_curve_indices[start:end]
        if incident.numel() == 0:
            continue
        valid = incident[(incident >= 0) & (incident < curve_count)].to(device=pred_points.device, dtype=torch.long)
        if valid.numel() == 0:
            continue
        out[:, point_idx] = curve_distances.index_select(1, valid).min(dim=1).values
    return out


def point_to_incident_curves_attach_distance(
    pred_points: torch.Tensor,
    target_curves: torch.Tensor,
    point_curve_offsets: torch.Tensor,
    point_curve_indices: torch.Tensor,
    num_curve_samples: int,
) -> torch.Tensor:
    if pred_points.numel() == 0:
        return pred_points.new_zeros((0,))
    if target_curves.numel() == 0 or point_curve_offsets.numel() == 0:
        return pred_points.new_zeros((pred_points.shape[0],))

    segments = sample_curve_segments(target_curves, num_samples=num_curve_samples)
    distances = []
    curve_count = int(target_curves.shape[0])
    for point_idx in range(pred_points.shape[0]):
        start = int(point_curve_offsets[point_idx].item())
        end = int(point_curve_offsets[point_idx + 1].item())
        incident = point_curve_indices[start:end]
        if incident.numel() == 0:
            distances.append(pred_points[point_idx].sum() * 0.0)
            continue
        valid = incident[(incident >= 0) & (incident < curve_count)]
        if valid.numel() == 0:
            distances.append(pred_points[point_idx].sum() * 0.0)
            continue
        incident_segments = segments.index_select(0, valid.to(device=segments.device, dtype=torch.long))
        point_dist = point_to_segments_distance(pred_points[point_idx: point_idx + 1], incident_segments)
        distances.append(point_dist.reshape(-1).min())
    return torch.stack(distances, dim=0)
