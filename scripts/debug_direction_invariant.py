import json
from pathlib import Path
import sys

import torch
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from misc_utils.bezier_target_utils import extract_cubic_targets
from misc_utils.train_utils import sample_bezier_curves_torch
from models.geometry import (
    aligned_curve_endpoint_l1,
    curve_boxes_xyxy,
    pairwise_aligned_curve_l1_cost,
    pairwise_aligned_sample_l1_cost,
    pairwise_curve_chamfer_cost,
    pairwise_generalized_box_iou,
    reverse_curve_points,
)


def _raw_pairwise_ctrl_cost(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    return torch.cdist(pred.reshape(pred.shape[0], -1), tgt.reshape(tgt.shape[0], -1), p=1)


def _raw_pairwise_sample_cost(pred: torch.Tensor, tgt: torch.Tensor, num_samples: int) -> torch.Tensor:
    pred_samples = sample_bezier_curves_torch(pred, num_samples=num_samples).reshape(pred.shape[0], -1)
    tgt_samples = sample_bezier_curves_torch(tgt, num_samples=num_samples).reshape(tgt.shape[0], -1)
    return torch.cdist(pred_samples, tgt_samples, p=1)


def _analyze_pair(pred_curve: torch.Tensor, tgt_curve: torch.Tensor, num_samples: int) -> dict:
    pred = pred_curve.unsqueeze(0)
    tgt = tgt_curve.unsqueeze(0)
    tgt_rev = reverse_curve_points(tgt)

    ctrl_forward_raw = float(_raw_pairwise_ctrl_cost(pred, tgt)[0, 0].item())
    ctrl_reverse_raw = float(_raw_pairwise_ctrl_cost(pred, tgt_rev)[0, 0].item())
    ctrl_aligned_impl = float(pairwise_aligned_curve_l1_cost(pred, tgt)[0, 0].item())

    sample_forward_raw = float(_raw_pairwise_sample_cost(pred, tgt, num_samples=num_samples)[0, 0].item())
    sample_reverse_raw = float(_raw_pairwise_sample_cost(pred, tgt_rev, num_samples=num_samples)[0, 0].item())
    sample_aligned_impl = float(pairwise_aligned_sample_l1_cost(pred, tgt, num_samples=num_samples)[0, 0].item())

    ctrl_abs, endpoint_abs, oriented_tgt = aligned_curve_endpoint_l1(pred, tgt)

    ctrl_dim = pred.numel()
    sample_dim = num_samples * 2

    return {
        "ctrl_forward_raw": ctrl_forward_raw,
        "ctrl_reverse_raw": ctrl_reverse_raw,
        "ctrl_min_raw": min(ctrl_forward_raw, ctrl_reverse_raw),
        "ctrl_aligned_impl": ctrl_aligned_impl,
        "ctrl_expected_if_only_min": min(ctrl_forward_raw, ctrl_reverse_raw),
        "ctrl_impl_times_dim": ctrl_aligned_impl * ctrl_dim,
        "sample_forward_raw": sample_forward_raw,
        "sample_reverse_raw": sample_reverse_raw,
        "sample_min_raw": min(sample_forward_raw, sample_reverse_raw),
        "sample_aligned_impl": sample_aligned_impl,
        "sample_expected_if_only_min": min(sample_forward_raw, sample_reverse_raw),
        "sample_impl_times_dim": sample_aligned_impl * sample_dim,
        "loss_ctrl_aligned": float(ctrl_abs[0].item()),
        "loss_endpoint_aligned": float(endpoint_abs[0].item()),
        "oriented_target_is_reversed": bool(torch.allclose(oriented_tgt, tgt_rev)),
    }


def _expected_direction_invariant_cost(
    pred_curves: torch.Tensor,
    tgt_curves: torch.Tensor,
    control_cost: float,
    sample_cost: float,
    giou_cost: float,
    curve_distance_cost: float,
    num_curve_samples: int,
    curve_match_point_count: int,
) -> torch.Tensor:
    ctrl_forward = _raw_pairwise_ctrl_cost(pred_curves, tgt_curves)
    ctrl_reverse = _raw_pairwise_ctrl_cost(pred_curves, reverse_curve_points(tgt_curves))
    ctrl = torch.minimum(ctrl_forward, ctrl_reverse)

    sample_forward = _raw_pairwise_sample_cost(pred_curves, tgt_curves, num_curve_samples)
    sample_reverse = _raw_pairwise_sample_cost(pred_curves, reverse_curve_points(tgt_curves), num_curve_samples)
    sample = torch.minimum(sample_forward, sample_reverse)

    pred_boxes = curve_boxes_xyxy(pred_curves)
    tgt_boxes = curve_boxes_xyxy(tgt_curves)
    giou = 1.0 - pairwise_generalized_box_iou(pred_boxes, tgt_boxes)
    curve_dist = pairwise_curve_chamfer_cost(pred_curves, tgt_curves, point_count=curve_match_point_count)
    return control_cost * ctrl + sample_cost * sample + giou_cost * giou + curve_distance_cost * curve_dist


def _assignment_accuracy(cost_matrix: torch.Tensor, true_targets: torch.Tensor) -> dict:
    row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
    assigned = torch.full((cost_matrix.shape[0],), -1, dtype=torch.long)
    assigned[torch.as_tensor(row_ind, dtype=torch.long)] = torch.as_tensor(col_ind, dtype=torch.long)
    accuracy = float((assigned == true_targets.cpu()).float().mean().item())
    return {
        "assignment": assigned.tolist(),
        "true_targets": true_targets.tolist(),
        "accuracy": accuracy,
    }


def _analyze_real_matching_case() -> dict:
    sample_path = REPO_ROOT / "edge_data/HED-BSDS/gt_rgb/test/100007_ann5.png"
    targets = extract_cubic_targets(sample_path, version_name="v5_anchor_consistent", target_degree=5, min_curve_length=3.0)
    tgt_curves = torch.from_numpy(targets["curves"][:12]).float()
    generator = torch.Generator().manual_seed(7)
    perm = torch.randperm(tgt_curves.shape[0], generator=generator)
    pred_curves = tgt_curves[perm].clone()
    reverse_mask = torch.tensor([(i % 2) == 0 for i in range(pred_curves.shape[0])], dtype=torch.bool)
    pred_curves[reverse_mask] = reverse_curve_points(pred_curves[reverse_mask])
    pred_curves = (pred_curves + 0.003 * torch.randn_like(pred_curves, generator=generator)).clamp(0.0, 1.0)

    logits = torch.zeros((pred_curves.shape[0], 2), dtype=torch.float32)
    control_cost = 1.5
    sample_cost = 2.0
    giou_cost = 1.0
    curve_distance_cost = 6.0
    num_curve_samples = 24
    curve_match_point_count = 4

    current_aligned = (
        control_cost * pairwise_aligned_curve_l1_cost(pred_curves, tgt_curves)
        + sample_cost * pairwise_aligned_sample_l1_cost(pred_curves, tgt_curves, num_samples=num_curve_samples)
        + giou_cost * (1.0 - pairwise_generalized_box_iou(curve_boxes_xyxy(pred_curves), curve_boxes_xyxy(tgt_curves)))
        + curve_distance_cost * pairwise_curve_chamfer_cost(pred_curves, tgt_curves, point_count=curve_match_point_count)
    )
    direct_off = (
        control_cost * _raw_pairwise_ctrl_cost(pred_curves, tgt_curves)
        + sample_cost * _raw_pairwise_sample_cost(pred_curves, tgt_curves, num_curve_samples)
        + giou_cost * (1.0 - pairwise_generalized_box_iou(curve_boxes_xyxy(pred_curves), curve_boxes_xyxy(tgt_curves)))
        + curve_distance_cost * pairwise_curve_chamfer_cost(pred_curves, tgt_curves, point_count=curve_match_point_count)
    )
    expected_di = _expected_direction_invariant_cost(
        pred_curves,
        tgt_curves,
        control_cost=control_cost,
        sample_cost=sample_cost,
        giou_cost=giou_cost,
        curve_distance_cost=curve_distance_cost,
        num_curve_samples=num_curve_samples,
        curve_match_point_count=curve_match_point_count,
    )

    return {
        "sample": sample_path.name,
        "num_curves": int(tgt_curves.shape[0]),
        "reversed_predictions": int(reverse_mask.sum().item()),
        "direct_off": _assignment_accuracy(direct_off, perm),
        "current_direction_invariant": _assignment_accuracy(current_aligned, perm),
        "expected_direction_invariant_raw_scale": _assignment_accuracy(expected_di, perm),
        "cost_stats": {
            "direct_off_mean": float(direct_off.mean().item()),
            "current_direction_invariant_mean": float(current_aligned.mean().item()),
            "expected_direction_invariant_mean": float(expected_di.mean().item()),
        },
    }


def main() -> None:
    out_dir = Path("outputs/direction_invariant_debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_curve = torch.tensor(
        [
            [0.10, 0.20],
            [0.25, 0.30],
            [0.45, 0.35],
            [0.62, 0.42],
            [0.78, 0.48],
            [0.92, 0.60],
        ],
        dtype=torch.float32,
    )
    reversed_curve = torch.flip(base_curve, dims=(0,))
    perturbed_curve = base_curve.clone()
    perturbed_curve[1, 1] += 0.03
    perturbed_curve[4, 0] -= 0.04

    report = {
        "same_orientation": _analyze_pair(base_curve, base_curve, num_samples=24),
        "reversed_orientation": _analyze_pair(base_curve, reversed_curve, num_samples=24),
        "perturbed_orientation": _analyze_pair(base_curve, perturbed_curve, num_samples=24),
        "real_matching_case": _analyze_real_matching_case(),
    }

    text_lines = []
    for case_name, stats in report.items():
        text_lines.append(f"[{case_name}]")
        for key, value in stats.items():
            text_lines.append(f"{key}: {value}")
        text_lines.append("")

    (out_dir / "report.txt").write_text("\n".join(text_lines), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
