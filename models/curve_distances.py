from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from misc_utils.train_utils import sample_bezier_curves_torch


def curve_arc_length_from_samples(samples: torch.Tensor) -> torch.Tensor:
    if samples.ndim != 3 or samples.shape[-1] != 2:
        raise ValueError(f"Expected [B, N, 2], got {tuple(samples.shape)}")
    if samples.shape[1] <= 1:
        return samples.new_zeros((samples.shape[0],))
    deltas = samples[:, 1:, :] - samples[:, :-1, :]
    return deltas.norm(dim=-1).sum(dim=-1)


def _normalize_closed_flags(is_closed: Optional[torch.Tensor], batch_size: int, *, device: torch.device) -> torch.Tensor:
    if is_closed is None:
        return torch.zeros((batch_size,), dtype=torch.bool, device=device)
    is_closed = torch.as_tensor(is_closed, dtype=torch.bool, device=device)
    if is_closed.ndim == 0:
        is_closed = is_closed.expand(batch_size)
    if is_closed.shape != (batch_size,):
        raise ValueError(f"Expected closed flags shape {(batch_size,)}, got {tuple(is_closed.shape)}")
    return is_closed


def build_curve_sample_weights(
    batch_size: int,
    num_samples: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    is_closed: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if num_samples < 2:
        raise ValueError(f"num_samples must be >= 2, got {num_samples}")
    weights = torch.ones((batch_size, num_samples), device=device, dtype=dtype)
    closed = _normalize_closed_flags(is_closed, batch_size, device=device)
    if bool(closed.any()):
        weights[closed, 0] = 0.5
        weights[closed, -1] = 0.5
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)


@dataclass
class CurveDistanceOutput:
    total: torch.Tensor
    primary: torch.Tensor
    length_gap: Optional[torch.Tensor] = None
    max_transport: Optional[torch.Tensor] = None
    pred_length: Optional[torch.Tensor] = None
    tgt_length: Optional[torch.Tensor] = None

    def as_dict(self) -> Dict[str, torch.Tensor]:
        out = {
            "total": self.total,
            "primary": self.primary,
        }
        if self.length_gap is not None:
            out["length_gap"] = self.length_gap
        if self.max_transport is not None:
            out["max_transport"] = self.max_transport
        if self.pred_length is not None:
            out["pred_length"] = self.pred_length
        if self.tgt_length is not None:
            out["tgt_length"] = self.tgt_length
        return out


def curve_distance_type_from_config(config: Dict) -> str:
    loss_cfg = config.get("loss", {}) if isinstance(config, dict) else {}
    return str(loss_cfg.get("curve_distance_type", "chamfer")).lower()


def curve_loss_name_from_config(config: Dict) -> str:
    curve_type = curve_distance_type_from_config(config)
    if curve_type == "emd":
        return "loss_emd"
    return "loss_chamfer"


def curve_matching_cost_weight_from_config(config: Dict) -> float:
    loss_cfg = config.get("loss", {}) if isinstance(config, dict) else {}
    curve_type = curve_distance_type_from_config(config)
    curve_cost = loss_cfg.get("curve_cost")
    if curve_cost is not None:
        return float(curve_cost)
    if curve_type == "emd":
        return float(loss_cfg.get("emd_cost", loss_cfg.get("chamfer_cost", 5.0)))
    return float(loss_cfg.get("chamfer_cost", 5.0))


def curve_loss_weight_from_config(config: Dict, *, overrides: Optional[Dict[str, float]] = None) -> float:
    weight_cfg = config.get("loss", {}) if overrides is None else {**config.get("loss", {}), **overrides}
    curve_type = curve_distance_type_from_config(config)
    curve_weight = weight_cfg.get("curve_weight")
    if curve_weight is not None:
        return float(curve_weight)
    if curve_type == "emd":
        return float(weight_cfg.get("emd_weight", weight_cfg.get("chamfer_weight", 5.0)))
    return float(weight_cfg.get("chamfer_weight", 5.0))


def closed_curve_endpoint_dist_threshold_from_config(config: Dict) -> float:
    loss_cfg = config.get("loss", {}) if isinstance(config, dict) else {}
    return float(loss_cfg.get("closed_curve_endpoint_dist_threshold", 0.03))


def infer_closed_curves(curves: torch.Tensor, *, threshold: float) -> torch.Tensor:
    if curves.ndim != 3 or curves.shape[-1] != 2:
        raise ValueError(f"Expected [N, K, 2], got {tuple(curves.shape)}")
    if curves.shape[0] == 0:
        return torch.zeros((0,), dtype=torch.bool, device=curves.device)
    endpoint_gap = (curves[:, 0, :] - curves[:, -1, :]).norm(dim=-1)
    return endpoint_gap <= float(threshold)


def _sample_count_from_config(config: Dict, *, for_matching: bool) -> int:
    loss_cfg = config.get("loss", {}) if isinstance(config, dict) else {}
    curve_type = curve_distance_type_from_config(config)
    generic_key = "curve_match_point_count" if for_matching else "curve_num_samples"
    if generic_key in loss_cfg:
        return int(loss_cfg[generic_key])
    if curve_type == "emd":
        specific_key = "emd_match_point_count" if for_matching else "emd_num_samples"
        fallback_key = "chamfer_match_point_count" if for_matching else "chamfer_num_samples"
        return int(loss_cfg.get(specific_key, loss_cfg.get(fallback_key, 20)))
    fallback_key = "chamfer_match_point_count" if for_matching else "chamfer_num_samples"
    return int(loss_cfg.get(fallback_key, 20))


def build_curve_distance_from_config(config: Dict, *, for_matching: bool):
    loss_cfg = config.get("loss", {}) if isinstance(config, dict) else {}
    curve_type = curve_distance_type_from_config(config)
    sample_count = _sample_count_from_config(config, for_matching=for_matching)
    if curve_type == "emd":
        return SinkhornEMDCurveDistance(
            num_samples=sample_count,
            sinkhorn_epsilon=float(loss_cfg.get("emd_sinkhorn_epsilon", 0.02)),
            sinkhorn_iters=int(loss_cfg.get("emd_sinkhorn_iters", 10)),
            length_weight=float(loss_cfg.get("emd_length_weight", 1.0)),
            transport_presence_power=float(loss_cfg.get("emd_transport_presence_power", 0.5)),
            chunk_size=loss_cfg.get("emd_chunk_size"),
        )
    return ChamferCurveDistance(num_samples=sample_count)


class ChamferCurveDistance:
    def __init__(self, *, num_samples: int = 20) -> None:
        self.num_samples = int(num_samples)

    def sample_curves(self, curves: torch.Tensor) -> torch.Tensor:
        return sample_bezier_curves_torch(curves, num_samples=self.num_samples)

    def matched_cost_from_samples(self, pred_samples: torch.Tensor, tgt_samples: torch.Tensor) -> CurveDistanceOutput:
        if pred_samples.shape != tgt_samples.shape:
            raise ValueError(
                f"pred_samples and tgt_samples must share shape, got {tuple(pred_samples.shape)} vs {tuple(tgt_samples.shape)}"
            )
        pairwise = torch.cdist(pred_samples, tgt_samples, p=2.0)
        pred_to_tgt = pairwise.min(dim=2).values.mean(dim=1)
        tgt_to_pred = pairwise.min(dim=1).values.mean(dim=1)
        total = 0.5 * (pred_to_tgt + tgt_to_pred)
        return CurveDistanceOutput(total=total, primary=total)

    def matched_cost_from_curves(
        self,
        pred_curves: torch.Tensor,
        tgt_curves: torch.Tensor,
        *,
        target_is_closed: Optional[torch.Tensor] = None,
    ) -> CurveDistanceOutput:
        del target_is_closed
        return self.matched_cost_from_samples(self.sample_curves(pred_curves), self.sample_curves(tgt_curves))

    def pairwise_cost_from_curves(
        self,
        pred_curves: torch.Tensor,
        tgt_curves: torch.Tensor,
        *,
        target_is_closed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del target_is_closed
        pred_samples = self.sample_curves(pred_curves)
        tgt_samples = self.sample_curves(tgt_curves)
        return self.pairwise_cost_from_samples(pred_samples, tgt_samples)

    def pairwise_cost_from_samples(self, pred_samples: torch.Tensor, tgt_samples: torch.Tensor) -> Dict[str, torch.Tensor]:
        if pred_samples.ndim != 3 or tgt_samples.ndim != 3:
            raise ValueError(
                f"Expected pred/tgt samples with shape [B, N, 2], got {tuple(pred_samples.shape)} vs {tuple(tgt_samples.shape)}"
            )
        pred_count, _, _ = pred_samples.shape
        tgt_count, _, _ = tgt_samples.shape
        if pred_count == 0 or tgt_count == 0:
            zero = pred_samples.new_zeros((pred_count, tgt_count))
            return {
                "total": zero,
                "primary": zero,
            }
        pairwise = torch.linalg.norm(
            pred_samples[:, None, :, None, :] - tgt_samples[None, :, None, :, :],
            dim=-1,
        )
        pred_to_tgt = pairwise.min(dim=3).values.mean(dim=2)
        tgt_to_pred = pairwise.min(dim=2).values.mean(dim=2)
        total = 0.5 * (pred_to_tgt + tgt_to_pred)
        return {
            "total": total,
            "primary": total,
        }


class SinkhornEMDCurveDistance:
    def __init__(
        self,
        *,
        num_samples: int = 20,
        sinkhorn_epsilon: float = 0.02,
        sinkhorn_iters: int = 80,
        length_weight: float = 1.0,
        transport_presence_power: float = 0.5,
        chunk_size: Optional[int] = None,
    ) -> None:
        self.num_samples = int(num_samples)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.length_weight = float(length_weight)
        self.transport_presence_power = float(transport_presence_power)
        self.chunk_size = None if chunk_size is None else int(chunk_size)

    def sample_curves(self, curves: torch.Tensor) -> torch.Tensor:
        return sample_bezier_curves_torch(curves, num_samples=self.num_samples)

    def _transport_presence(self, plan: torch.Tensor) -> torch.Tensor:
        plan_max = plan.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        normalized = (plan / plan_max).clamp_min(0.0)
        if self.transport_presence_power != 1.0:
            normalized = normalized.pow(self.transport_presence_power)
        return normalized

    def _sinkhorn_ot_cost(
        self,
        pred_samples: torch.Tensor,
        tgt_samples: torch.Tensor,
        *,
        pred_weights: torch.Tensor,
        tgt_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cost_matrix = torch.cdist(pred_samples, tgt_samples, p=2.0)
        kernel = torch.exp(-cost_matrix / max(self.sinkhorn_epsilon, 1e-6)).clamp_min(1e-8)

        u = torch.ones_like(pred_weights)
        v = torch.ones_like(tgt_weights)
        for _ in range(self.sinkhorn_iters):
            kv = torch.bmm(kernel, v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
            u = pred_weights / kv
            ktu = torch.bmm(kernel.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
            v = tgt_weights / ktu

        plan = u.unsqueeze(-1) * kernel * v.unsqueeze(-2)
        ot_cost = (plan * cost_matrix).sum(dim=(1, 2))
        transport_presence = self._transport_presence(plan)
        max_transport = (cost_matrix * transport_presence).amax(dim=(1, 2))
        return ot_cost, max_transport

    def sinkhorn_distance_from_samples(
        self,
        pred_samples: torch.Tensor,
        tgt_samples: torch.Tensor,
        *,
        target_is_closed: Optional[torch.Tensor] = None,
    ) -> CurveDistanceOutput:
        if pred_samples.ndim != 3 or tgt_samples.ndim != 3:
            raise ValueError(
                f"Expected pred/tgt samples with shape [B, N, 2], got {tuple(pred_samples.shape)} vs {tuple(tgt_samples.shape)}"
            )
        if pred_samples.shape != tgt_samples.shape:
            raise ValueError(
                f"pred_samples and tgt_samples must share shape, got {tuple(pred_samples.shape)} vs {tuple(tgt_samples.shape)}"
            )
        batch_size, num_samples, _ = pred_samples.shape
        closed = _normalize_closed_flags(target_is_closed, batch_size, device=pred_samples.device)
        pred_weights = build_curve_sample_weights(
            batch_size,
            num_samples,
            device=pred_samples.device,
            dtype=pred_samples.dtype,
            is_closed=closed,
        )
        tgt_weights = build_curve_sample_weights(
            batch_size,
            num_samples,
            device=tgt_samples.device,
            dtype=tgt_samples.dtype,
            is_closed=closed,
        )
        cross_cost, max_transport = self._sinkhorn_ot_cost(
            pred_samples,
            tgt_samples,
            pred_weights=pred_weights,
            tgt_weights=tgt_weights,
        )
        pred_self_cost, _ = self._sinkhorn_ot_cost(
            pred_samples,
            pred_samples,
            pred_weights=pred_weights,
            tgt_weights=pred_weights,
        )
        tgt_self_cost, _ = self._sinkhorn_ot_cost(
            tgt_samples,
            tgt_samples,
            pred_weights=tgt_weights,
            tgt_weights=tgt_weights,
        )
        emd = (cross_cost - 0.5 * pred_self_cost - 0.5 * tgt_self_cost).clamp_min(0.0)
        pred_length = curve_arc_length_from_samples(pred_samples)
        tgt_length = curve_arc_length_from_samples(tgt_samples)
        length_gap = (pred_length - tgt_length).abs()
        total = emd + self.length_weight * max_transport * length_gap
        return CurveDistanceOutput(
            total=total,
            primary=emd,
            length_gap=length_gap,
            max_transport=max_transport,
            pred_length=pred_length,
            tgt_length=tgt_length,
        )

    def matched_cost_from_curves(
        self,
        pred_curves: torch.Tensor,
        tgt_curves: torch.Tensor,
        *,
        target_is_closed: Optional[torch.Tensor] = None,
    ) -> CurveDistanceOutput:
        pred_samples = self.sample_curves(pred_curves)
        tgt_samples = self.sample_curves(tgt_curves)
        return self.sinkhorn_distance_from_samples(
            pred_samples,
            tgt_samples,
            target_is_closed=target_is_closed,
        )

    def pairwise_cost_from_curves(
        self,
        pred_curves: torch.Tensor,
        tgt_curves: torch.Tensor,
        *,
        target_is_closed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pred_samples = self.sample_curves(pred_curves)
        tgt_samples = self.sample_curves(tgt_curves)
        return self.pairwise_cost_from_samples(
            pred_samples,
            tgt_samples,
            target_is_closed=target_is_closed,
        )

    def pairwise_cost_from_samples(
        self,
        pred_samples: torch.Tensor,
        tgt_samples: torch.Tensor,
        *,
        target_is_closed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if pred_samples.ndim != 3 or tgt_samples.ndim != 3:
            raise ValueError(
                f"Expected pred/tgt samples with shape [B, N, 2], got {tuple(pred_samples.shape)} vs {tuple(tgt_samples.shape)}"
            )
        pred_count, num_samples, _ = pred_samples.shape
        tgt_count, tgt_num_samples, _ = tgt_samples.shape
        if num_samples != tgt_num_samples:
            raise ValueError(f"Sample count mismatch: {num_samples} vs {tgt_num_samples}")
        if pred_count == 0 or tgt_count == 0:
            shape = (pred_count, tgt_count)
            zero = pred_samples.new_zeros(shape)
            return {
                "total": zero,
                "emd": zero,
                "length_gap": zero,
                "max_transport": zero,
                "pred_length": zero,
                "tgt_length": zero,
            }

        closed = _normalize_closed_flags(target_is_closed, tgt_count, device=tgt_samples.device)
        pair_total = pred_count * tgt_count
        chunk_size = pair_total if self.chunk_size is None else max(1, self.chunk_size)
        pair_outputs: list[CurveDistanceOutput] = []

        pred_indices = torch.arange(pred_count, device=pred_samples.device)[:, None].expand(pred_count, tgt_count).reshape(-1)
        tgt_indices = torch.arange(tgt_count, device=tgt_samples.device)[None, :].expand(pred_count, tgt_count).reshape(-1)
        pair_closed = closed[tgt_indices]

        for start in range(0, pair_total, chunk_size):
            end = min(pair_total, start + chunk_size)
            chunk_out = self.sinkhorn_distance_from_samples(
                pred_samples[pred_indices[start:end]],
                tgt_samples[tgt_indices[start:end]],
                target_is_closed=pair_closed[start:end],
            )
            pair_outputs.append(chunk_out)

        total = torch.cat([item.total for item in pair_outputs], dim=0).reshape(pred_count, tgt_count)
        emd = torch.cat([item.primary for item in pair_outputs], dim=0).reshape(pred_count, tgt_count)
        length_gap = torch.cat([item.length_gap for item in pair_outputs], dim=0).reshape(pred_count, tgt_count)
        max_transport = torch.cat([item.max_transport for item in pair_outputs], dim=0).reshape(pred_count, tgt_count)
        pred_length = torch.cat([item.pred_length for item in pair_outputs], dim=0).reshape(pred_count, tgt_count)
        tgt_length = torch.cat([item.tgt_length for item in pair_outputs], dim=0).reshape(pred_count, tgt_count)
        return {
            "total": total,
            "emd": emd,
            "length_gap": length_gap,
            "max_transport": max_transport,
            "pred_length": pred_length,
            "tgt_length": tgt_length,
        }
