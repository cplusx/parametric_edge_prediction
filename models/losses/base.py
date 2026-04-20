from typing import Dict, Optional, Tuple

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


def balanced_class_weights(
    target_classes: torch.Tensor,
    positive_class: int = 0,
    query_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_classes = target_classes.long()
    positive_mask = target_classes == int(positive_class)
    negative_mask = ~positive_mask
    if query_weights is not None:
        query_weights = query_weights.to(device=target_classes.device, dtype=torch.float32)
    else:
        query_weights = torch.ones_like(target_classes, dtype=torch.float32)

    pos_weight_sum = (query_weights * positive_mask.to(dtype=torch.float32)).sum(dim=1, keepdim=True)
    neg_weight_sum = (query_weights * negative_mask.to(dtype=torch.float32)).sum(dim=1, keepdim=True)
    total_weight = query_weights.sum(dim=1, keepdim=True)
    has_pos = pos_weight_sum > 0
    has_neg = neg_weight_sum > 0
    has_both = has_pos & has_neg

    target_pos_total = torch.where(has_both, 0.5 * total_weight, torch.where(has_pos, total_weight, torch.zeros_like(total_weight)))
    target_neg_total = torch.where(has_both, 0.5 * total_weight, torch.where(has_neg, total_weight, torch.zeros_like(total_weight)))

    pos_multiplier = torch.where(has_pos, target_pos_total / pos_weight_sum.clamp_min(1e-6), torch.zeros_like(total_weight))
    neg_multiplier = torch.where(has_neg, target_neg_total / neg_weight_sum.clamp_min(1e-6), torch.zeros_like(total_weight))
    weights = torch.where(positive_mask, query_weights * pos_multiplier, query_weights * neg_multiplier).to(dtype=torch.float32)
    normalizer = weights.sum(dim=1).clamp_min(1e-6)
    active_mask = normalizer > 0
    return weights, normalizer, active_mask
