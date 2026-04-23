#!/usr/bin/env python3
# ruff: noqa: E402
import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge_datasets import build_datamodule
from misc_utils.config_utils import load_config
from misc_utils.debug_vis_endpoint_attach import render_endpoint_attach_frame, save_frames_as_gif
from models.curve_coordinates import curve_external_to_internal, curve_internal_to_external
from models.endpoint_matcher import HungarianPointMatcher
from models.losses.endpoint_matched import MatchedPointLoss


def _force_attach_config(config: Dict) -> Dict:
    config = copy.deepcopy(config)
    config.setdefault("data", {})["endpoint_target_mode"] = "attach"
    config["data"].setdefault("endpoint_closed_curve_threshold_px", 2.0)
    config.setdefault("loss", {})["endpoint_loss_type"] = "attach"
    config["loss"].setdefault("point_attach_weight", 1.0)
    config["loss"].setdefault("point_attach_low_degree_multiplier", 5.0)
    config["loss"].setdefault("point_attach_degree_threshold", 3)
    config["loss"].setdefault("point_attach_num_curve_samples", 8)
    config.setdefault("trainer", {})["devices"] = 1
    config["trainer"]["accelerator"] = "cpu"
    return config


def _target_has_required_types(target: dict) -> bool:
    points = target["points"]
    if points.shape[0] == 0:
        return False
    degree = target["point_degree"]
    loop_only = target["point_is_loop_only"]
    low = (degree < 3) & (~loop_only)
    high = (degree >= 3) & (~loop_only)
    return bool(loop_only.any()) and bool(low.any()) and bool(high.any())


def _pick_sample(dataset, max_scan: int) -> dict:
    fallback = None
    for idx in range(min(len(dataset), int(max_scan))):
        item = dataset[idx]
        target = item["target"]
        if fallback is None and target["points"].shape[0] > 0 and target["curves"].shape[0] > 0:
            fallback = item
        if _target_has_required_types(target):
            return item
    if fallback is None:
        raise RuntimeError(f"No usable attach sample found in first {max_scan} samples")
    return fallback


def _make_logits(query_count: int, device: torch.device) -> torch.Tensor:
    logits = torch.full((1, query_count, 2), -6.0, dtype=torch.float32, device=device)
    logits[..., 0] = 6.0
    logits[..., 1] = -6.0
    return logits


def _loop_focus_init(target: dict, pred_points_int: torch.Tensor, config: Dict) -> Optional[int]:
    loop_indices = torch.nonzero(target["point_is_loop_only"], as_tuple=False).flatten()
    if loop_indices.numel() == 0:
        return None
    point_idx = int(loop_indices[0].item())
    offsets = target["point_curve_offsets"]
    indices = target["point_curve_indices"]
    start = int(offsets[point_idx].item())
    end = int(offsets[point_idx + 1].item())
    if end <= start:
        return point_idx
    curve_idx = int(indices[start].item())
    curve = target["curves"][curve_idx]
    pred_points_int[point_idx] = curve_external_to_internal(curve[curve.shape[0] // 2], config)
    return point_idx


def _optimize(
    *,
    target: dict,
    config: Dict,
    output_path: Path,
    use_attach: bool,
    iterations: int,
    lr: float,
    frame_every: int,
    loop_focus: bool,
) -> None:
    run_config = copy.deepcopy(config)
    run_config.setdefault("loss", {})["endpoint_loss_type"] = "attach" if use_attach else "l1"
    device = torch.device("cpu")
    target = {key: value.to(device) if hasattr(value, "to") else value for key, value in target.items()}
    gt_points_int = curve_external_to_internal(target["points"], run_config)
    if gt_points_int.shape[0] == 0:
        raise RuntimeError("Selected sample has no GT endpoints")
    generator = torch.Generator(device=device)
    generator.manual_seed(13)
    initial = (gt_points_int + torch.randn(gt_points_int.shape, generator=generator, device=device) * 0.08).clamp(1e-4, 1.0 - 1e-4)
    pred_points = torch.nn.Parameter(initial)
    highlight_idx = _loop_focus_init(target, pred_points.data, run_config) if loop_focus else None
    logits = _make_logits(pred_points.shape[0], device=device)
    matcher = HungarianPointMatcher.from_config(run_config)
    loss_fn = MatchedPointLoss(run_config)
    optimizer = torch.optim.Adam([pred_points], lr=float(lr))
    frames = []
    for step in range(int(iterations) + 1):
        optimizer.zero_grad(set_to_none=True)
        outputs = {"pred_points": pred_points.unsqueeze(0), "pred_logits": logits}
        indices = matcher(logits=outputs["pred_logits"], points=outputs["pred_points"], targets=[target])
        losses = loss_fn(outputs["pred_points"], outputs["pred_logits"], [target], indices, outputs)
        if step % int(frame_every) == 0 or step == int(iterations):
            pred_external = curve_internal_to_external(pred_points.detach(), run_config).clamp(0.0, 1.0).cpu().numpy()
            matched = (indices[0][0].detach().cpu().numpy(), indices[0][1].detach().cpu().numpy())
            title = (
                f"{'attach' if use_attach else 'l1-only'} iter={step} "
                f"loss={float(losses['loss_total'].detach()):.4f} "
                f"l1={float(losses['loss_point_l1'].detach()):.4f} "
                f"attach={float(losses['loss_point_attach'].detach()):.4f}"
            )
            frames.append(
                render_endpoint_attach_frame(
                    target=target,
                    pred_points=pred_external,
                    matched_indices=matched,
                    title=title,
                    highlight_point_index=highlight_idx,
                )
            )
        if step == int(iterations):
            break
        losses["loss_total"].backward()
        optimizer.step()
        pred_points.data.clamp_(1e-4, 1.0 - 1e-4)
    save_frames_as_gif(frames, output_path, fps=3.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize standalone endpoint predictions with L1 vs attach loss.")
    parser.add_argument("--config", default="configs/parametric_edge/laion_endpoint_attach_lab34_v3_2gpu.yaml")
    parser.add_argument("--output-dir", default="outputs/debug_endpoint_attach_opt")
    parser.add_argument("--scan", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--frame-every", type=int, default=4)
    parser.add_argument("--loop-focus", action="store_true")
    args = parser.parse_args()

    config = _force_attach_config(load_config(args.config))
    datamodule = build_datamodule(config)
    datamodule.setup("fit")
    item = _pick_sample(datamodule.train_dataset, max_scan=args.scan)
    target = item["target"]
    output_dir = Path(args.output_dir)
    _optimize(
        target=target,
        config=config,
        output_path=output_dir / "baseline_l1_only.gif",
        use_attach=False,
        iterations=args.iterations,
        lr=args.lr,
        frame_every=args.frame_every,
        loop_focus=args.loop_focus,
    )
    _optimize(
        target=target,
        config=config,
        output_path=output_dir / "attach_loss.gif",
        use_attach=True,
        iterations=args.iterations,
        lr=args.lr,
        frame_every=args.frame_every,
        loop_focus=args.loop_focus,
    )
    print(f"sample_id={target.get('sample_id')}")
    print(f"wrote {output_dir / 'baseline_l1_only.gif'}")
    print(f"wrote {output_dir / 'attach_loss.gif'}")


if __name__ == "__main__":
    main()
