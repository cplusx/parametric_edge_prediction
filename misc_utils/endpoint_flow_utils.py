from typing import List, Sequence, Tuple

import numpy as np
import torch


def scale_points_to_flow(points: torch.Tensor) -> torch.Tensor:
    return points * 2.0 - 1.0


def scale_points_from_flow(points: torch.Tensor) -> torch.Tensor:
    return (points + 1.0) * 0.5


def _sorted_indices(points: torch.Tensor) -> torch.Tensor:
    if points.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=points.device)
    points_np = points.detach().cpu().numpy()
    order = np.lexsort((points_np[:, 1], points_np[:, 0]))
    return torch.from_numpy(order.astype(np.int64)).to(device=points.device)


def pad_endpoint_targets(targets: Sequence[dict], max_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if not targets:
        raise ValueError('targets must be non-empty')
    device = targets[0]['points'].device
    batch_size = len(targets)
    padded = torch.zeros((batch_size, max_points, 2), dtype=torch.float32, device=device)
    valid_mask = torch.zeros((batch_size, max_points), dtype=torch.bool, device=device)
    for batch_idx, target in enumerate(targets):
        points = target['points'].to(device=device, dtype=torch.float32)
        if points.numel() == 0:
            continue
        order = _sorted_indices(points)
        points = points[order]
        count = min(max_points, points.shape[0])
        padded[batch_idx, :count] = points[:count]
        valid_mask[batch_idx, :count] = True
    return padded, valid_mask


def select_predicted_points(points: torch.Tensor, presence_probs: torch.Tensor, threshold: float) -> List[torch.Tensor]:
    selected: List[torch.Tensor] = []
    keep_mask = presence_probs > float(threshold)
    for batch_idx in range(points.shape[0]):
        selected.append(points[batch_idx, keep_mask[batch_idx]])
    return selected
