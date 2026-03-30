from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


def scale_points_to_flow(points: torch.Tensor) -> torch.Tensor:
    return points * 2.0 - 1.0


def scale_points_from_flow(points: torch.Tensor) -> torch.Tensor:
    return (points + 1.0) * 0.5


def sample_uniform_points(
    batch_size: int,
    num_points: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    return torch.rand((batch_size, num_points, 2), device=device, dtype=dtype, generator=generator)


def build_uniform_flow_batch(
    source_points: torch.Tensor,
    targets: Sequence[dict],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not targets:
        raise ValueError('targets must be non-empty')
    batch_size, num_points, _ = source_points.shape
    aligned_targets = source_points.clone()
    valid_mask = torch.zeros((batch_size, num_points), dtype=torch.bool, device=source_points.device)
    for batch_idx, target in enumerate(targets):
        points = target['points'].to(device=source_points.device, dtype=source_points.dtype)
        if points.numel() == 0:
            continue
        count = min(int(points.shape[0]), num_points)
        points = points[:count]
        cost = torch.cdist(source_points[batch_idx, :count], points, p=1)
        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
        if len(row_ind) == 0:
            continue
        row_idx = torch.as_tensor(row_ind, dtype=torch.long, device=source_points.device)
        col_idx = torch.as_tensor(col_ind, dtype=torch.long, device=source_points.device)
        aligned_targets[batch_idx, row_idx] = points[col_idx]
        valid_mask[batch_idx, :count] = True
    return source_points, aligned_targets, valid_mask


def select_predicted_points(points: torch.Tensor) -> List[torch.Tensor]:
    selected: List[torch.Tensor] = []
    for batch_idx in range(points.shape[0]):
        selected.append(points[batch_idx])
    return selected
