from typing import Dict, List, Tuple

import torch


class BaseLossComponent:
    def __init__(self, config: Dict) -> None:
        self.config = config


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.sum() * 0.0
    weights = weights.to(values.device, values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    denom = weights.sum().clamp_min(1e-6)
    return (values * weights).sum() / denom


def matched_curve_weights(
    targets: List[dict],
    indices: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    alpha: float,
    power: float,
) -> torch.Tensor:
    if alpha <= 0.0:
        total = sum(int(src_idx.numel()) for src_idx, _ in indices)
        return torch.ones((total,), device=device)
    weights = []
    for batch_idx, (_, tgt_idx) in enumerate(indices):
        if tgt_idx.numel() == 0:
            continue
        curvatures = targets[batch_idx].get('curve_curvatures')
        if curvatures is None or curvatures.numel() == 0:
            weights.append(torch.ones((tgt_idx.numel(),), device=device))
            continue
        curvatures = curvatures[tgt_idx].to(device=device, dtype=torch.float32)
        max_curv = curvatures.max().clamp_min(1e-6)
        normalized = (curvatures / max_curv).clamp(0.0, 1.0)
        weights.append(1.0 + alpha * normalized.pow(power))
    if not weights:
        return torch.ones((0,), device=device)
    return torch.cat(weights, dim=0)