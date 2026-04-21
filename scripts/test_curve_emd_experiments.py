from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Dict, Sequence

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from misc_utils.train_utils import sample_bezier_curves_torch
from models.curve_distances import ChamferCurveDistance, SinkhornEMDCurveDistance


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def close_curve_control_points(curves: torch.Tensor) -> torch.Tensor:
    closed = curves.clone()
    closed[..., -1, :] = closed[..., 0, :]
    return closed


def make_open_curve_bank(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(
        [
            [[0.08, 0.20], [0.18, 0.86], [0.35, 0.10], [0.55, 0.90], [0.78, 0.24], [0.92, 0.78]],
            [[0.12, 0.72], [0.26, 0.16], [0.44, 0.83], [0.57, 0.20], [0.77, 0.70], [0.88, 0.28]],
            [[0.10, 0.52], [0.18, 0.22], [0.40, 0.88], [0.62, 0.18], [0.82, 0.72], [0.90, 0.44]],
        ],
        device=device,
        dtype=dtype,
    )


def make_circle_samples(num_samples: int, phase_steps: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    unique_count = max(2, num_samples - 1)
    phase = (2.0 * math.pi * phase_steps) / unique_count
    theta = torch.linspace(phase, phase + 2.0 * math.pi, num_samples, device=device, dtype=dtype)
    radius_x = 0.28
    radius_y = 0.22
    cx = 0.5
    cy = 0.5
    return torch.stack([cx + radius_x * torch.cos(theta), cy + radius_y * torch.sin(theta)], dim=-1)


def jitter_like(base: torch.Tensor, scale: float, *, generator: torch.Generator | None = None) -> torch.Tensor:
    noise = torch.randn(base.shape, generator=generator, device=base.device, dtype=base.dtype) * scale
    return (base + noise).clamp(0.02, 0.98)


def sample_curve_for_plot(curves: torch.Tensor, num_samples: int = 200) -> torch.Tensor:
    with torch.no_grad():
        return sample_bezier_curves_torch(curves.detach(), num_samples=num_samples).cpu()


def fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return np.ascontiguousarray(rgba[..., :3])


def save_gif(frames: Sequence[np.ndarray], path: Path, fps: int = 1) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, list(frames), duration=1.0 / fps)


def plot_curve_overlay(ax: plt.Axes, gt_curves: torch.Tensor, pred_curves: torch.Tensor, *, title: str) -> None:
    gt_samples = sample_curve_for_plot(gt_curves)
    pred_samples = sample_curve_for_plot(pred_curves)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(gt_samples), len(pred_samples), 1)))
    for idx, pts in enumerate(gt_samples):
        ax.plot(pts[:, 0], pts[:, 1], color=colors[idx], linewidth=3.0, alpha=0.9, label="GT" if idx == 0 else None)
    for idx, pts in enumerate(pred_samples):
        ax.plot(pts[:, 0], pts[:, 1], color=colors[idx], linewidth=2.0, linestyle="--", alpha=0.9, label="Pred" if idx == 0 else None)
        cps = pred_curves[idx].detach().cpu()
        ax.scatter(cps[:, 0], cps[:, 1], color=colors[idx], s=18)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_title(title)
    ax.legend(loc="lower right")


def plot_closed_bezier_overlay(
    ax: plt.Axes,
    gt_samples: torch.Tensor,
    pred_curve: torch.Tensor,
    *,
    title: str,
    render_samples: int = 200,
) -> None:
    gt_np = gt_samples.detach().cpu().numpy()
    pred_curve_cpu = pred_curve.detach().cpu()
    pred_render = sample_bezier_curves_torch(pred_curve_cpu, num_samples=render_samples)[0].numpy()
    pred_sparse = sample_bezier_curves_torch(pred_curve_cpu, num_samples=gt_samples.shape[1])[0].numpy()
    cps = pred_curve_cpu[0].numpy()

    ax.plot(gt_np[0, :, 0], gt_np[0, :, 1], color="#1f77b4", linewidth=3.0, alpha=0.85, label="GT loop")
    ax.scatter(gt_np[0, :, 0], gt_np[0, :, 1], color="#1f77b4", s=18)

    ax.plot(pred_render[:, 0], pred_render[:, 1], color="#d62728", linewidth=2.4, linestyle="--", alpha=0.95, label="Pred bezier")
    ax.scatter(pred_sparse[:, 0], pred_sparse[:, 1], color="#d62728", s=16)
    ax.plot(cps[:, 0], cps[:, 1], color="#ff9896", linewidth=1.5, alpha=0.9, linestyle=":", label="Control polygon")
    ax.scatter(cps[:, 0], cps[:, 1], color="#ff9896", s=28)

    gt_start = gt_np[0, 0]
    gt_end = gt_np[0, -1]
    pred_start = cps[0]
    pred_end = cps[-1]
    ax.scatter([gt_start[0]], [gt_start[1]], color="#1f77b4", s=170, marker="*", edgecolors="black", linewidths=0.8, label="GT p0")
    ax.scatter([gt_end[0]], [gt_end[1]], color="#17becf", s=140, marker="P", edgecolors="black", linewidths=0.8, label="GT p19")
    ax.scatter([pred_start[0]], [pred_start[1]], color="#d62728", s=130, marker="X", edgecolors="black", linewidths=0.8, label="Pred c0")
    ax.scatter([pred_end[0]], [pred_end[1]], color="#ff7f0e", s=120, marker="D", edgecolors="black", linewidths=0.8, label="Pred c5")
    ax.annotate("GT p0", xy=(gt_start[0], gt_start[1]), xytext=(8, -14), textcoords="offset points", color="#1f77b4", fontsize=9)
    ax.annotate("GT p19", xy=(gt_end[0], gt_end[1]), xytext=(8, 8), textcoords="offset points", color="#17becf", fontsize=9)
    ax.annotate("Pred c0", xy=(pred_start[0], pred_start[1]), xytext=(8, -18), textcoords="offset points", color="#d62728", fontsize=9)
    ax.annotate("Pred c5", xy=(pred_end[0], pred_end[1]), xytext=(8, 12), textcoords="offset points", color="#ff7f0e", fontsize=9)
    ax.plot([pred_start[0], pred_end[0]], [pred_start[1], pred_end[1]], color="#555555", linewidth=1.0, alpha=0.6, linestyle=":")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_title(title)
    ax.legend(loc="lower right")


def plot_losses(ax: plt.Axes, history: Sequence[Dict[str, float]], keys: Sequence[str], *, title: str) -> None:
    if not history:
        return
    steps = [item["step"] for item in history]
    for key in keys:
        if key not in history[0]:
            continue
        ax.plot(steps, [item[key] for item in history], label=key)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")


def closed_curve_bend_penalty(samples: torch.Tensor, *, eps: float = 1e-4) -> torch.Tensor:
    if samples.ndim != 3 or samples.shape[-1] != 2:
        raise ValueError(f"Expected [B, N, 2], got {tuple(samples.shape)}")
    if samples.shape[1] < 3:
        return samples.new_zeros((samples.shape[0],))

    edges = torch.roll(samples, shifts=-1, dims=1) - samples
    edge_norm = edges.norm(dim=-1, keepdim=True).clamp_min(eps)
    edge_dir = edges / edge_norm

    prev_dir = torch.roll(edge_dir, shifts=1, dims=1)
    cos_turn = (prev_dir * edge_dir).sum(dim=-1).clamp(-1.0, 1.0)

    # Penalize only true fold-backs: angles > 90 deg (cos < 0).
    # As cos -> -1 (U-turn), the denominator collapses and the penalty spikes.
    foldness = torch.relu(-cos_turn)
    turn_penalty = (foldness * foldness) / (1.0 + cos_turn + eps)
    return turn_penalty.mean(dim=1)


def render_experiment_frame(
    *,
    output_path: Path,
    overlay_fn: Callable[[plt.Axes], None],
    history: Sequence[Dict[str, float]],
    loss_keys: Sequence[str],
    title: str,
) -> np.ndarray:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    overlay_fn(axes[0])
    plot_losses(axes[1], history, loss_keys, title=title)
    frame = fig_to_rgb(fig)
    plt.close(fig)
    return frame


def optimize_single_open_curve(
    *,
    distance: SinkhornEMDCurveDistance,
    output_dir: Path,
    steps: int,
    lr: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, float]:
    gt = make_open_curve_bank(device, dtype)[:1]
    pred_raw = torch.nn.Parameter(jitter_like(gt, 0.18))
    optimizer = torch.optim.Adam([pred_raw], lr=lr)
    history: list[dict[str, float]] = []
    frames: list[np.ndarray] = []

    for step in range(steps + 1):
        pred = pred_raw.clamp(0.02, 0.98)
        out = distance.matched_cost_from_curves(pred, gt, target_is_closed=torch.zeros(1, dtype=torch.bool, device=device))
        loss = out.total.mean()
        if step < steps:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred_raw.clamp_(0.02, 0.98)
        history.append(
            {
                "step": step,
                "loss_total": float(loss.detach().cpu().item()),
                "emd": float(out.primary.mean().detach().cpu().item()),
                "length_gap": float(out.length_gap.mean().detach().cpu().item()),
                "max_transport": float(out.max_transport.mean().detach().cpu().item()),
            }
        )
        if step % 4 == 0 or step == steps:
            pred_vis = pred.detach()
            frames.append(
                render_experiment_frame(
                    output_path=output_dir / "exp1_open_curve.gif",
                    overlay_fn=lambda ax, gt_curves=gt.detach(), pred_curves=pred_vis: plot_curve_overlay(
                        ax, gt_curves, pred_curves, title=f"Exp1 Open Curve | step={step}"
                    ),
                    history=history,
                    loss_keys=("loss_total", "emd", "length_gap"),
                    title="Exp1 losses",
                )
            )

    save_gif(frames, output_dir / "exp1_open_curve.gif")
    imageio.imwrite(output_dir / "exp1_open_curve_final.jpg", frames[-1])
    return history[-1]


def optimize_single_closed_bezier(
    *,
    distance: SinkhornEMDCurveDistance,
    open_distance: SinkhornEMDCurveDistance,
    output_dir: Path,
    steps: int,
    lr: float,
    device: torch.device,
    dtype: torch.dtype,
    bend_weight: float,
) -> Dict[str, float]:
    gt_samples = make_circle_samples(distance.num_samples, phase_steps=0, device=device, dtype=dtype)[None, ...]
    shifted_samples = make_circle_samples(distance.num_samples, phase_steps=5, device=device, dtype=dtype)[None, ...]
    ctrl_idx = torch.linspace(0, distance.num_samples - 1, steps=6, device=device).round().long()
    init_curve = shifted_samples[:, ctrl_idx, :]
    pred_curve_raw = torch.nn.Parameter((init_curve + 0.03 * torch.randn_like(init_curve)).clamp(0.02, 0.98))
    optimizer = torch.optim.Adam([pred_curve_raw], lr=lr)
    history: list[dict[str, float]] = []
    frames: list[np.ndarray] = []

    closed_zero = distance.sinkhorn_distance_from_samples(
        shifted_samples,
        gt_samples,
        target_is_closed=torch.ones(1, dtype=torch.bool, device=device),
    )
    open_nonzero = open_distance.sinkhorn_distance_from_samples(
        shifted_samples,
        gt_samples,
        target_is_closed=torch.zeros(1, dtype=torch.bool, device=device),
    )
    rotation_endpoint_to_gt = (shifted_samples[:, 0, :] - gt_samples[:, 0, :]).norm(dim=-1)

    for step in range(steps + 1):
        pred_curve = pred_curve_raw.clamp(0.02, 0.98)
        pred_samples = sample_bezier_curves_torch(pred_curve, num_samples=distance.num_samples)
        out = distance.sinkhorn_distance_from_samples(
            pred_samples,
            gt_samples,
            target_is_closed=torch.ones(1, dtype=torch.bool, device=device),
        )
        bend_penalty = closed_curve_bend_penalty(pred_samples).mean()
        loss = out.total.mean() + bend_weight * bend_penalty
        if step < steps:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred_curve_raw.clamp_(0.02, 0.98)
        endpoint_gap = (pred_curve[:, 0, :] - pred_curve[:, -1, :]).norm(dim=-1)
        endpoint0_to_gt = (pred_curve[:, 0, :] - gt_samples[:, 0, :]).norm(dim=-1)
        endpoint5_to_gt = (pred_curve[:, -1, :] - gt_samples[:, 0, :]).norm(dim=-1)
        history.append(
            {
                "step": step,
                "loss_total": float(loss.detach().cpu().item()),
                "emd_total": float(out.total.mean().detach().cpu().item()),
                "emd": float(out.primary.mean().detach().cpu().item()),
                "length_gap": float(out.length_gap.mean().detach().cpu().item()),
                "bend_penalty": float(bend_penalty.detach().cpu().item()),
                "rotation_closed_loss": float(closed_zero.total.mean().detach().cpu().item()),
                "rotation_open_loss": float(open_nonzero.total.mean().detach().cpu().item()),
                "rotation_endpoint_to_gt": float(rotation_endpoint_to_gt.mean().detach().cpu().item()),
                "endpoint_gap": float(endpoint_gap.mean().detach().cpu().item()),
                "pred_c0_to_gt_endpoint": float(endpoint0_to_gt.mean().detach().cpu().item()),
                "pred_c5_to_gt_endpoint": float(endpoint5_to_gt.mean().detach().cpu().item()),
            }
        )
        if step % 4 == 0 or step == steps:
            pred_curve_vis = pred_curve.detach()
            frames.append(
                render_experiment_frame(
                    output_path=output_dir / "exp2_closed_curve.gif",
                    overlay_fn=lambda ax, gt_pts=gt_samples.detach(), pred_curve_local=pred_curve_vis: plot_closed_bezier_overlay(
                        ax,
                        gt_pts,
                        pred_curve_local,
                        title=(
                            f"Exp2 Closed Bezier | step={step}\n"
                            f"shift-only closed={closed_zero.total.item():.4e} open={open_nonzero.total.item():.4e}\n"
                            f"emd_total={out.total.mean().item():.4f} | bend={bend_penalty.item():.4f} | w={bend_weight:.2e}\n"
                            f"shift endpoint->gt={rotation_endpoint_to_gt.item():.4f} | "
                            f"pred c0/c5 -> gt={endpoint0_to_gt.item():.4f}/{endpoint5_to_gt.item():.4f}"
                        ),
                    ),
                    history=history,
                    loss_keys=(
                        "loss_total",
                        "emd_total",
                        "emd",
                        "bend_penalty",
                        "rotation_closed_loss",
                        "rotation_open_loss",
                        "endpoint_gap",
                        "pred_c0_to_gt_endpoint",
                        "pred_c5_to_gt_endpoint",
                    ),
                    title="Exp2 losses",
                )
            )

    save_gif(frames, output_dir / "exp2_closed_curve.gif")
    imageio.imwrite(output_dir / "exp2_closed_curve_final.jpg", frames[-1])
    return history[-1]


def optimize_multiple_curves_batch(
    *,
    distance: SinkhornEMDCurveDistance,
    output_dir: Path,
    steps: int,
    lr: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, float]:
    gt = make_open_curve_bank(device, dtype)
    pred_raw = torch.nn.Parameter(jitter_like(gt, 0.12))
    optimizer = torch.optim.Adam([pred_raw], lr=lr)
    history: list[dict[str, float]] = []
    frames: list[np.ndarray] = []

    for step in range(steps + 1):
        pred = pred_raw.clamp(0.02, 0.98)
        out = distance.matched_cost_from_curves(pred, gt, target_is_closed=torch.zeros(gt.shape[0], dtype=torch.bool, device=device))
        loss = out.total.mean()
        if step < steps:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred_raw.clamp_(0.02, 0.98)
        history.append(
            {
                "step": step,
                "loss_total": float(loss.detach().cpu().item()),
                "emd": float(out.primary.mean().detach().cpu().item()),
                "length_gap": float(out.length_gap.mean().detach().cpu().item()),
            }
        )
        if step % 4 == 0 or step == steps:
            pred_vis = pred.detach()
            frames.append(
                render_experiment_frame(
                    output_path=output_dir / "exp3_batch_aligned.gif",
                    overlay_fn=lambda ax, gt_curves=gt.detach(), pred_curves=pred_vis: plot_curve_overlay(
                        ax, gt_curves, pred_curves, title=f"Exp3 Batch aligned | step={step}"
                    ),
                    history=history,
                    loss_keys=("loss_total", "emd", "length_gap"),
                    title="Exp3 losses",
                )
            )

    save_gif(frames, output_dir / "exp3_batch_aligned.gif")
    imageio.imwrite(output_dir / "exp3_batch_aligned_final.jpg", frames[-1])
    return history[-1]


def render_hungarian_overlay(
    ax: plt.Axes,
    *,
    gt_curves: torch.Tensor,
    pred_curves: torch.Tensor,
    matched_pred: Sequence[int],
    matched_tgt: Sequence[int],
    title: str,
) -> None:
    gt_samples = sample_curve_for_plot(gt_curves)
    pred_samples = sample_curve_for_plot(pred_curves)
    gt_colors = plt.cm.Set2(np.linspace(0, 1, max(len(gt_samples), 1)))
    matched_pred_set = set(int(v) for v in matched_pred)
    matched_map = {int(p): int(t) for p, t in zip(matched_pred, matched_tgt)}

    for idx, pts in enumerate(gt_samples):
        ax.plot(pts[:, 0], pts[:, 1], color=gt_colors[idx], linewidth=3.5, alpha=0.95)
        ax.text(float(pts[0, 0]), float(pts[0, 1]), f"GT{idx}", color=gt_colors[idx], fontsize=10)

    for idx, pts in enumerate(pred_samples):
        if idx in matched_pred_set:
            color = gt_colors[matched_map[idx] % len(gt_colors)]
            label = f"P{idx}->GT{matched_map[idx]}"
            alpha = 0.95
        else:
            color = "#808080"
            label = f"P{idx} FP"
            alpha = 0.65
        ax.plot(pts[:, 0], pts[:, 1], color=color, linestyle="--", linewidth=2.2, alpha=alpha)
        ax.text(float(pts[-1, 0]), float(pts[-1, 1]), label, color=color, fontsize=9)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_title(title)


def optimize_with_hungarian_false_positives(
    *,
    distance: SinkhornEMDCurveDistance,
    output_dir: Path,
    steps: int,
    lr: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, float]:
    gt = make_open_curve_bank(device, dtype)
    base_pred = jitter_like(gt, 0.14)
    false_positives = torch.tensor(
        [
            [[0.74, 0.10], [0.82, 0.16], [0.90, 0.24], [0.94, 0.34], [0.90, 0.42], [0.82, 0.48]],
            [[0.08, 0.90], [0.14, 0.82], [0.22, 0.76], [0.28, 0.70], [0.34, 0.62], [0.38, 0.56]],
        ],
        device=device,
        dtype=dtype,
    )
    pred_raw = torch.nn.Parameter(torch.cat([base_pred, false_positives], dim=0))
    optimizer = torch.optim.Adam([pred_raw], lr=lr)
    history: list[dict[str, float]] = []
    frames: list[np.ndarray] = []

    last_match_pred = np.arange(gt.shape[0])
    last_match_tgt = np.arange(gt.shape[0])

    for step in range(steps + 1):
        pred = pred_raw.clamp(0.02, 0.98)
        pairwise = distance.pairwise_cost_from_curves(
            pred,
            gt,
            target_is_closed=torch.zeros(gt.shape[0], dtype=torch.bool, device=device),
        )
        row_ind, col_ind = linear_sum_assignment(pairwise["total"].detach().cpu().numpy())
        matched = distance.matched_cost_from_curves(
            pred[row_ind],
            gt[col_ind],
            target_is_closed=torch.zeros(len(row_ind), dtype=torch.bool, device=device),
        )
        loss = matched.total.mean()
        if step < steps:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred_raw.clamp_(0.02, 0.98)
        last_match_pred = row_ind
        last_match_tgt = col_ind
        history.append(
            {
                "step": step,
                "loss_total": float(loss.detach().cpu().item()),
                "emd": float(matched.primary.mean().detach().cpu().item()),
                "length_gap": float(matched.length_gap.mean().detach().cpu().item()),
                "pairwise_min": float(pairwise["total"].min().detach().cpu().item()),
            }
        )
        if step % 4 == 0 or step == steps:
            pred_vis = pred.detach()
            frames.append(
                render_experiment_frame(
                    output_path=output_dir / "exp4_hungarian_false_positive.gif",
                    overlay_fn=lambda ax, gt_curves=gt.detach(), pred_curves=pred_vis, pred_ids=row_ind.copy(), tgt_ids=col_ind.copy(): render_hungarian_overlay(
                        ax,
                        gt_curves=gt_curves,
                        pred_curves=pred_curves,
                        matched_pred=pred_ids,
                        matched_tgt=tgt_ids,
                        title=f"Exp4 Hungarian FP | step={step}",
                    ),
                    history=history,
                    loss_keys=("loss_total", "emd", "length_gap", "pairwise_min"),
                    title="Exp4 losses",
                )
            )

    save_gif(frames, output_dir / "exp4_hungarian_false_positive.gif")
    imageio.imwrite(output_dir / "exp4_hungarian_false_positive_final.jpg", frames[-1])
    return {
        **history[-1],
        "matched_pred_indices": [int(v) for v in last_match_pred.tolist()],
        "matched_tgt_indices": [int(v) for v in last_match_tgt.tolist()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Local EMD curve-distance experiments with optimization GIFs.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/curve_emd_experiments"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.02)
    parser.add_argument("--sinkhorn-iters", type=int, default=80)
    parser.add_argument("--length-weight", type=float, default=1.0)
    parser.add_argument("--closed-bend-weight", type=float, default=0.0)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = choose_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    args.output_dir.mkdir(parents=True, exist_ok=True)

    emd_distance = SinkhornEMDCurveDistance(
        num_samples=args.num_samples,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        sinkhorn_iters=args.sinkhorn_iters,
        length_weight=args.length_weight,
    )
    open_distance = SinkhornEMDCurveDistance(
        num_samples=args.num_samples,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        sinkhorn_iters=args.sinkhorn_iters,
        length_weight=args.length_weight,
    )
    chamfer_distance = ChamferCurveDistance(num_samples=args.num_samples)

    summary = {
        "config": {
            "device": str(device),
            "dtype": args.dtype,
            "steps": args.steps,
            "lr": args.lr,
            "num_samples": args.num_samples,
            "sinkhorn_epsilon": args.sinkhorn_epsilon,
            "sinkhorn_iters": args.sinkhorn_iters,
            "length_weight": args.length_weight,
        },
        "exp1_open_curve": optimize_single_open_curve(
            distance=emd_distance,
            output_dir=args.output_dir,
            steps=args.steps,
            lr=args.lr,
            device=device,
            dtype=dtype,
        ),
        "exp2_closed_curve": optimize_single_closed_bezier(
            distance=emd_distance,
            open_distance=open_distance,
            output_dir=args.output_dir,
            steps=args.steps,
            lr=args.lr,
            device=device,
            dtype=dtype,
            bend_weight=args.closed_bend_weight,
        ),
        "exp3_batch_aligned": optimize_multiple_curves_batch(
            distance=emd_distance,
            output_dir=args.output_dir,
            steps=args.steps,
            lr=args.lr,
            device=device,
            dtype=dtype,
        ),
        "exp4_hungarian_false_positive": optimize_with_hungarian_false_positives(
            distance=emd_distance,
            output_dir=args.output_dir,
            steps=args.steps,
            lr=args.lr,
            device=device,
            dtype=dtype,
        ),
    }

    gt_open = make_open_curve_bank(device, dtype)[:1]
    pred_open = jitter_like(gt_open, 0.18)
    summary["sanity"] = {
        "chamfer_open_initial": float(
            chamfer_distance.matched_cost_from_curves(pred_open, gt_open).total.mean().detach().cpu().item()
        ),
        "emd_open_initial": float(
            emd_distance.matched_cost_from_curves(
                pred_open,
                gt_open,
                target_is_closed=torch.zeros(1, dtype=torch.bool, device=device),
            ).total.mean().detach().cpu().item()
        ),
    }

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
