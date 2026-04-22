from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict

import torch


class CurveQueryInitializer(ABC):
    @abstractmethod
    def initialize(
        self,
        *,
        num_queries: int,
        num_control_points: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return initial curves with shape [num_queries, num_control_points, 2]."""


class RandomCurveQueryInitializer(CurveQueryInitializer):
    """Initialize each control point independently at random."""

    def initialize(
        self,
        *,
        num_queries: int,
        num_control_points: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.rand((num_queries, num_control_points, 2), device=device, dtype=dtype).clamp(1e-4, 1.0 - 1e-4)


class PointCurveQueryInitializer(CurveQueryInitializer):
    """Initialize each query as a repeated single point on a regular grid."""

    def initialize(
        self,
        *,
        num_queries: int,
        num_control_points: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        side = int(math.ceil(math.sqrt(float(num_queries))))
        ys = torch.linspace(0.05, 0.95, side, device=device, dtype=dtype)
        xs = torch.linspace(0.05, 0.95, side, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        refs = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)[:num_queries].clamp(1e-4, 1.0 - 1e-4)
        return refs.unsqueeze(1).expand(-1, num_control_points, -1).clone()


class LineCurveQueryInitializer(CurveQueryInitializer):
    """Randomize only the endpoints, then linearly interpolate the middle control points."""

    def initialize(
        self,
        *,
        num_queries: int,
        num_control_points: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        start = torch.rand((num_queries, 2), device=device, dtype=dtype).clamp(1e-4, 1.0 - 1e-4)
        end = torch.rand((num_queries, 2), device=device, dtype=dtype).clamp(1e-4, 1.0 - 1e-4)
        alphas = torch.linspace(0.0, 1.0, num_control_points, device=device, dtype=dtype)
        return start[:, None, :] * (1.0 - alphas)[None, :, None] + end[:, None, :] * alphas[None, :, None]


def build_curve_query_initializer(config: Dict) -> CurveQueryInitializer:
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    if "curve_query_init_type" not in model_cfg or model_cfg.get("curve_query_init_type") is None:
        raise KeyError("model.curve_query_init_type must be set explicitly")
    init_type = str(model_cfg["curve_query_init_type"]).strip().lower()
    if not init_type:
        raise ValueError("model.curve_query_init_type must be a non-empty string")
    if init_type in {"random"}:
        return RandomCurveQueryInitializer()
    if init_type in {"current", "point", "grid_point", "grid"}:
        return PointCurveQueryInitializer()
    if init_type in {"line", "line_segment", "straight_line"}:
        return LineCurveQueryInitializer()
    raise ValueError(f"Unsupported curve_query_init_type: {init_type}")
