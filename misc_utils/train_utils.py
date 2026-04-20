import math
from typing import Dict, List, Tuple

import torch


_BASIS_CACHE: Dict[Tuple[int, int, str, str], torch.Tensor] = {}


def bernstein_basis_torch(degree: int, t_values: torch.Tensor) -> torch.Tensor:
    basis = []
    for i in range(degree + 1):
        coeff = torch.tensor(float(math.comb(degree, i)), dtype=t_values.dtype, device=t_values.device)
        basis.append(coeff * (1.0 - t_values) ** (degree - i) * t_values ** i)
    return torch.stack(basis, dim=-1)


def get_bezier_basis_torch(
    degree: int,
    num_samples: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (int(degree), int(num_samples), str(device), str(dtype))
    cached = _BASIS_CACHE.get(key)
    if cached is not None:
        return cached
    t_values = torch.linspace(0.0, 1.0, int(num_samples), device=device, dtype=dtype)
    basis = bernstein_basis_torch(int(degree), t_values)
    _BASIS_CACHE[key] = basis
    return basis


def sample_bezier_curves_torch(control_points: torch.Tensor, num_samples: int = 16) -> torch.Tensor:
    if control_points.ndim != 3 or control_points.shape[-1] != 2:
        raise ValueError(f'Expected [N, K, 2], got {tuple(control_points.shape)}')
    degree = control_points.shape[1] - 1
    basis = get_bezier_basis_torch(
        degree,
        num_samples,
        device=control_points.device,
        dtype=control_points.dtype,
    )
    return torch.einsum('sk,nkd->nsd', basis, control_points)


def build_dn_queries(
    targets: List[dict],
    num_dn_groups: int,
    noise_scale: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_targets = max((target['curves'].shape[0] for target in targets), default=0)
    control_dim = targets[0]['curves'].shape[1] * 2 if targets and targets[0]['curves'].ndim == 3 else 12
    if max_targets == 0 or num_dn_groups <= 0:
        return (
            torch.zeros((len(targets), 0, control_dim), device=device),
            torch.zeros((len(targets), 0, control_dim), device=device),
            torch.zeros((len(targets), 0), dtype=torch.bool, device=device),
            torch.zeros((len(targets), 0), dtype=torch.long, device=device),
        )
    dn_count = max_targets * num_dn_groups
    dn_curves = torch.zeros((len(targets), dn_count, control_dim), dtype=torch.float32, device=device)
    dn_targets = torch.zeros((len(targets), dn_count, control_dim), dtype=torch.float32, device=device)
    dn_mask = torch.zeros((len(targets), dn_count), dtype=torch.bool, device=device)
    dn_labels = torch.ones((len(targets), dn_count), dtype=torch.long, device=device)
    for batch_idx, target in enumerate(targets):
        curves = target['curves'].to(device)
        count = curves.shape[0]
        if count == 0:
            continue
        flat = curves.reshape(count, control_dim)
        for group_idx in range(num_dn_groups):
            start = group_idx * max_targets
            end = start + count
            noise = torch.randn_like(flat) * noise_scale
            dn_curves[batch_idx, start:end] = (flat + noise).clamp(0.0, 1.0)
            dn_targets[batch_idx, start:end] = flat
            dn_mask[batch_idx, start:end] = True
            dn_labels[batch_idx, start:end] = 0
    return dn_curves, dn_targets, dn_mask, dn_labels
