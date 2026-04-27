from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Dict, List, Sequence

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from misc_utils.train_utils import sample_bezier_curves_torch
from models.curve_distances import OrderedBidirectionalCurveDistance


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_open_curve_bank(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(
        [
            [[0.08, 0.20], [0.18, 0.86], [0.35, 0.10], [0.55, 0.90], [0.78, 0.24], [0.92, 0.78]],
            [[0.12, 0.72], [0.26, 0.16], [0.44, 0.83], [0.57, 0.20], [0.77, 0.70], [0.88, 0.28]],
            [[0.10, 0.52], [0.18, 0.22], [0.40, 0.88], [0.62, 0.18], [0.82, 0.72], [0.90, 0.44]],
            [[0.14, 0.28], [0.20, 0.78], [0.42, 0.16], [0.58, 0.86], [0.76, 0.30], [0.86, 0.82]],
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


def jitter_like(base: torch.Tensor, scale: float) -> torch.Tensor:
    noise = torch.randn_like(base) * scale
    return (base + noise).clamp(0.02, 0.98)


def sample_curve_for_plot(curves: torch.Tensor, num_samples: int = 160) -> torch.Tensor:
    with torch.no_grad():
        return sample_bezier_curves_torch(curves.detach(), num_samples=num_samples).cpu()


def fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return np.ascontiguousarray(rgba[..., :3])


def save_gif(frames: Sequence[np.ndarray], path: Path, fps: int) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, list(frames), duration=1.0 / fps)


def plot_curve_overlay(
    ax: plt.Axes,
    gt_curves: torch.Tensor,
    pred_curves: torch.Tensor,
    *,
    title: str,
    matched_pred_mask: torch.Tensor | None = None,
) -> None:
    gt_samples = sample_curve_for_plot(gt_curves)
    pred_samples = sample_curve_for_plot(pred_curves)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(gt_samples), len(pred_samples), 1)))
    for idx, pts in enumerate(gt_samples):
        ax.plot(pts[:, 0], pts[:, 1], color=colors[idx], linewidth=3.0, alpha=0.9, label="GT" if idx == 0 else None)
    for idx, pts in enumerate(pred_samples):
        if matched_pred_mask is not None and not bool(matched_pred_mask[idx]):
            color = "#999999"
            alpha = 0.55
        else:
            color = colors[idx]
            alpha = 0.9
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2.0, linestyle="--", alpha=alpha, label="Pred" if idx == 0 else None)
        cps = pred_curves[idx].detach().cpu()
        ax.scatter(cps[:, 0], cps[:, 1], color=color, s=14, alpha=alpha)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_title(title)
    ax.legend(loc="lower right")


def plot_losses(ax: plt.Axes, history: Sequence[Dict[str, float]], *, title: str) -> None:
    if not history:
        return
    steps = [item["step"] for item in history]
    for key in ("loss_total", "forward", "reverse", "ordered"):
        if key in history[0]:
            ax.plot(steps, [item[key] for item in history], label=key)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")


def render_frame(
    *,
    gt_curves: torch.Tensor,
    pred_curves: torch.Tensor,
    history: Sequence[Dict[str, float]],
    title: str,
    matched_pred_mask: torch.Tensor | None = None,
) -> np.ndarray:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    plot_curve_overlay(axes[0], gt_curves, pred_curves, title=title, matched_pred_mask=matched_pred_mask)
    plot_losses(axes[1], history, title="Loss")
    frame = fig_to_rgb(fig)
    plt.close(fig)
    return frame


def ordered_forward_reverse_terms(pred_curves: torch.Tensor, tgt_curves: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred_samples = sample_bezier_curves_torch(pred_curves, num_samples=20)
    tgt_samples = sample_bezier_curves_torch(tgt_curves, num_samples=20)
    forward = (pred_samples - tgt_samples).norm(dim=-1).mean(dim=1)
    reverse = (pred_samples - tgt_samples.flip(dims=(1,))).norm(dim=-1).mean(dim=1)
    return forward, reverse


def optimize_single_open_curve(
    *,
    distance: OrderedBidirectionalCurveDistance,
    output_dir: Path,
    steps: int,
    lr: float,
    device: torch.device,
    dtype: torch.dtype,
    fps: int,
) -> Dict[str, float]:
    gt = make_open_curve_bank(device, dtype)[:1]
    pred_raw = torch.nn.Parameter(jitter_like(gt, 0.18))
    optimizer = torch.optim.Adam([pred_raw], lr=lr)
    history: list[dict[str, float]] = []
    frames: list[np.ndarray] = []

    for step in range(steps + 1):
        pred = pred_raw.clamp(0.02, 0.98)
        out = distance.matched_cost_from_curves(pred, gt, target_is_closed=torch.zeros(1, dtype=torch.bool, device=device))
        forward, reverse = ordered_forward_reverse_terms(pred, gt)
        loss = out.total.mean()
        if step < steps:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred_raw.clamp_(0.02, 0.98)
        history.append({
            "step": step,
            "loss_total": float(loss.detach().cpu().item()),
            "ordered": float(out.total.mean().detach().cpu().item()),
            "forward": float(forward.mean().detach().cpu().item()),
            "reverse": float(reverse.mean().detach().cpu().item()),
        })
        frames.append(render_frame(gt_curves=gt.detach(), pred_curves=pred.detach(), history=history, title=f"Open curve | step={step}"))

    save_gif(frames, output_dir / "exp1_open_curve.gif", fps=fps)
    imageio.imwrite(output_dir / "exp1_open_curve_final.jpg", frames[-1])
    return history[-1]


def optimize_single_closed_curve(
    *,
    distance: OrderedBidirectionalCurveDistance,
    output_dir: Path,
    steps: int,
    lr: float,
    device: torch.device,
    dtype: torch.dtype,
    fps: int,
) -> Dict[str, float]:
    gt_samples = make_circle_samples(distance.num_samples, phase_steps=0, device=device, dtype=dtype)[None, ...]
    shifted_samples = make_circle_samples(distance.num_samples, phase_steps=5, device=device, dtype=dtype)[None, ...]
    ctrl_idx = torch.linspace(0, distance.num_samples - 1, steps=6, device=device).round().long()
    gt_curve = gt_samples[:, ctrl_idx, :].clone()
    gt_curve[:, -1, :] = gt_curve[:, 0, :]
    pred_init = shifted_samples[:, ctrl_idx, :].clone()
    pred_init[:, -1, :] = pred_init[:, 0, :]
    pred_raw = torch.nn.Parameter((pred_init + 0.03 * torch.randn_like(pred_init)).clamp(0.02, 0.98))
    optimizer = torch.optim.Adam([pred_raw], lr=lr)
    history: list[dict[str, float]] = []
    frames: list[np.ndarray] = []

    for step in range(steps + 1):
        pred = pred_raw.clamp(0.02, 0.98)
        out = distance.matched_cost_from_curves(pred, gt_curve, target_is_closed=torch.ones(1, dtype=torch.bool, device=device))
        forward, reverse = ordered_forward_reverse_terms(pred, gt_curve)
        loss = out.total.mean()
        if step < steps:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred_raw.clamp_(0.02, 0.98)
        history.append({
            "step": step,
            "loss_total": float(loss.detach().cpu().item()),
            "ordered": float(out.total.mean().detach().cpu().item()),
            "forward": float(forward.mean().detach().cpu().item()),
            "reverse": float(reverse.mean().detach().cpu().item()),
        })
        frames.append(render_frame(gt_curves=gt_curve.detach(), pred_curves=pred.detach(), history=history, title=f"Closed curve | step={step}"))

    save_gif(frames, output_dir / "exp2_closed_curve.gif", fps=fps)
    imageio.imwrite(output_dir / "exp2_closed_curve_final.jpg", frames[-1])
    return history[-1]


def optimize_multiple_curves_batch(
    *,
    distance: OrderedBidirectionalCurveDistance,
    output_dir: Path,
    steps: int,
    lr: float,
    device: torch.device,
    dtype: torch.dtype,
    fps: int,
) -> Dict[str, float]:
    gt = make_open_curve_bank(device, dtype)[:3]
    pred_raw = torch.nn.Parameter(jitter_like(gt, 0.20))
    optimizer = torch.optim.Adam([pred_raw], lr=lr)
    history: list[dict[str, float]] = []
    frames: list[np.ndarray] = []

    for step in range(steps + 1):
        pred = pred_raw.clamp(0.02, 0.98)
        out = distance.matched_cost_from_curves(pred, gt, target_is_closed=torch.zeros(gt.shape[0], dtype=torch.bool, device=device))
        forward, reverse = ordered_forward_reverse_terms(pred, gt)
        loss = out.total.mean()
        if step < steps:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred_raw.clamp_(0.02, 0.98)
        history.append({
            "step": step,
            "loss_total": float(loss.detach().cpu().item()),
            "ordered": float(out.total.mean().detach().cpu().item()),
            "forward": float(forward.mean().detach().cpu().item()),
            "reverse": float(reverse.mean().detach().cpu().item()),
        })
        frames.append(render_frame(gt_curves=gt.detach(), pred_curves=pred.detach(), history=history, title=f"Aligned batch | step={step}"))

    save_gif(frames, output_dir / "exp3_batch_aligned.gif", fps=fps)
    imageio.imwrite(output_dir / "exp3_batch_aligned_final.jpg", frames[-1])
    return history[-1]


def optimize_with_hungarian_false_positives(
    *,
    distance: OrderedBidirectionalCurveDistance,
    output_dir: Path,
    steps: int,
    lr: float,
    device: torch.device,
    dtype: torch.dtype,
    fps: int,
) -> Dict[str, float]:
    gt = make_open_curve_bank(device, dtype)[:3]
    pred_init = torch.cat(
        [
            jitter_like(gt, 0.22),
            torch.tensor(
                [
                    [[0.05, 0.10], [0.12, 0.16], [0.18, 0.22], [0.24, 0.30], [0.30, 0.36], [0.38, 0.44]],
                    [[0.62, 0.08], [0.68, 0.18], [0.74, 0.25], [0.82, 0.35], [0.88, 0.42], [0.94, 0.50]],
                ],
                device=device,
                dtype=dtype,
            ),
        ],
        dim=0,
    )
    pred_raw = torch.nn.Parameter(pred_init.clamp(0.02, 0.98))
    optimizer = torch.optim.Adam([pred_raw], lr=lr)
    history: list[dict[str, float]] = []
    frames: list[np.ndarray] = []
    last_match_pred = torch.empty(0, dtype=torch.long)
    last_match_tgt = torch.empty(0, dtype=torch.long)

    for step in range(steps + 1):
        pred = pred_raw.clamp(0.02, 0.98)
        pairwise = distance.pairwise_cost_from_curves(
            pred,
            gt,
            target_is_closed=torch.zeros(gt.shape[0], dtype=torch.bool, device=device),
        )["total"]
        row_ind, col_ind = linear_sum_assignment(pairwise.detach().cpu().numpy())
        src_idx = torch.as_tensor(row_ind, dtype=torch.long, device=device)
        tgt_idx = torch.as_tensor(col_ind, dtype=torch.long, device=device)
        matched_pred = pred[src_idx]
        matched_gt = gt[tgt_idx]
        out = distance.matched_cost_from_curves(
            matched_pred,
            matched_gt,
            target_is_closed=torch.zeros(src_idx.shape[0], dtype=torch.bool, device=device),
        )
        forward, reverse = ordered_forward_reverse_terms(matched_pred, matched_gt)
        loss = out.total.mean()
        if step < steps:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred_raw.clamp_(0.02, 0.98)
        history.append({
            "step": step,
            "loss_total": float(loss.detach().cpu().item()),
            "ordered": float(out.total.mean().detach().cpu().item()),
            "forward": float(forward.mean().detach().cpu().item()),
            "reverse": float(reverse.mean().detach().cpu().item()),
        })
        matched_mask = torch.zeros(pred.shape[0], dtype=torch.bool, device=device)
        matched_mask[src_idx] = True
        frames.append(
            render_frame(
                gt_curves=gt.detach(),
                pred_curves=pred.detach(),
                history=history,
                title=f"Hungarian + false positives | step={step}",
                matched_pred_mask=matched_mask.detach().cpu(),
            )
        )
        last_match_pred = src_idx.detach().cpu()
        last_match_tgt = tgt_idx.detach().cpu()

    save_gif(frames, output_dir / "exp4_hungarian_false_positive.gif", fps=fps)
    imageio.imwrite(output_dir / "exp4_hungarian_false_positive_final.jpg", frames[-1])
    return {
        **history[-1],
        "matched_pred_indices": [int(v) for v in last_match_pred.tolist()],
        "matched_tgt_indices": [int(v) for v in last_match_tgt.tolist()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Local ordered curve-distance experiments with optimization GIFs.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/curve_ordered_experiments"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--fps", type=int, default=3)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = choose_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    args.output_dir.mkdir(parents=True, exist_ok=True)

    distance = OrderedBidirectionalCurveDistance(num_samples=args.num_samples)
    summary = {
        "config": {
            "device": str(device),
            "dtype": args.dtype,
            "steps": args.steps,
            "lr": args.lr,
            "num_samples": args.num_samples,
            "fps": args.fps,
        },
        "exp1_open_curve": optimize_single_open_curve(
            distance=distance,
            output_dir=args.output_dir,
            steps=args.steps,
            lr=args.lr,
            device=device,
            dtype=dtype,
            fps=args.fps,
        ),
        "exp2_closed_curve": optimize_single_closed_curve(
            distance=distance,
            output_dir=args.output_dir,
            steps=args.steps,
            lr=args.lr,
            device=device,
            dtype=dtype,
            fps=args.fps,
        ),
        "exp3_batch_aligned": optimize_multiple_curves_batch(
            distance=distance,
            output_dir=args.output_dir,
            steps=args.steps,
            lr=args.lr,
            device=device,
            dtype=dtype,
            fps=args.fps,
        ),
        "exp4_hungarian_false_positive": optimize_with_hungarian_false_positives(
            distance=distance,
            output_dir=args.output_dir,
            steps=args.steps,
            lr=args.lr,
            device=device,
            dtype=dtype,
            fps=args.fps,
        ),
    }

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
