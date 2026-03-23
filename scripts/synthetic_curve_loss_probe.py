import argparse
import copy
import csv
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

from misc_utils.config_utils import load_config
from misc_utils.train_utils import sample_bezier_curves_torch
from models.geometry import (
    aligned_curve_endpoint_l1,
    curve_boxes_xyxy,
    matched_generalized_box_iou,
    reverse_curve_points,
)
from models.losses.composite import build_loss_computer


@dataclass
class PresetResult:
    name: str
    output_dir: Path
    history: List[Dict[str, float]]
    summary: Dict[str, float]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_spread_centers(count: int, min_dist: float, rng: np.random.Generator) -> np.ndarray:
    centers: List[np.ndarray] = []
    attempts = 0
    while len(centers) < count and attempts < 10000:
        candidate = rng.uniform(0.15, 0.85, size=(2,))
        if all(np.linalg.norm(candidate - center) >= min_dist for center in centers):
            centers.append(candidate)
        attempts += 1
    if len(centers) < count:
        raise RuntimeError(f'Failed to sample {count} separated curve centers.')
    return np.stack(centers, axis=0)


def generate_gt_curves(num_targets: int, degree: int, rng: np.random.Generator) -> torch.Tensor:
    centers = sample_spread_centers(num_targets, min_dist=0.18, rng=rng)
    controls: List[np.ndarray] = []
    for center in centers:
        theta = rng.uniform(0.0, 2.0 * math.pi)
        tangent = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        half_extent = rng.uniform(0.06, 0.14)
        arc_amplitude = rng.uniform(-0.08, 0.08)
        ts = np.linspace(-1.0, 1.0, degree + 1, dtype=np.float32)
        base = center[None, :] + ts[:, None] * half_extent * tangent[None, :]
        arc = ((1.0 - ts**2) * arc_amplitude)[:, None] * normal[None, :]
        jitter = rng.normal(scale=0.01, size=base.shape).astype(np.float32)
        jitter[0] = 0.0
        jitter[-1] = 0.0
        curve = np.clip(base + arc + jitter, 0.03, 0.97)
        controls.append(curve.astype(np.float32))
    return torch.from_numpy(np.stack(controls, axis=0))


def build_target(curves: torch.Tensor) -> List[Dict]:
    return [{
        'curves': curves,
        'boxes': curve_boxes_xyxy(curves),
    }]


def preset_loss_config(base_config: Dict, preset_name: str) -> Dict:
    config = copy.deepcopy(base_config)
    loss_cfg = config['loss']
    loss_cfg['group_detr_num_groups'] = 1
    loss_cfg['aux_weight'] = 0.0
    loss_cfg['one_to_many_weight'] = 0.0
    loss_cfg['topk_positive_enabled'] = False
    loss_cfg['topk_positive_weight'] = 0.0
    loss_cfg['distinct_weight'] = 0.0
    loss_cfg['dn_weight'] = 0.0
    loss_cfg['dn_curve_weight'] = 0.0
    if preset_name == 'current_main':
        return config
    if preset_name == 'ctrl_endpoint_giou_only':
        loss_cfg['sample_cost'] = 0.0
        loss_cfg['curve_distance_cost'] = 0.0
        loss_cfg['sample_weight'] = 0.0
        loss_cfg['curve_distance_weight'] = 0.0
        return config
    raise ValueError(f'Unknown preset: {preset_name}')


def orientation_aware_sample_l1(pred_curves: torch.Tensor, tgt_curves: torch.Tensor, num_samples: int) -> torch.Tensor:
    pred_samples = sample_bezier_curves_torch(pred_curves, num_samples=num_samples)
    tgt_samples = sample_bezier_curves_torch(tgt_curves, num_samples=num_samples)
    tgt_rev_samples = torch.flip(tgt_samples, dims=(-2,))
    forward = torch.abs(pred_samples - tgt_samples).mean(dim=(1, 2))
    reverse = torch.abs(pred_samples - tgt_rev_samples).mean(dim=(1, 2))
    return torch.minimum(forward, reverse)


def orientation_aware_chamfer(pred_curves: torch.Tensor, tgt_curves: torch.Tensor, num_samples: int) -> torch.Tensor:
    pred_samples = sample_bezier_curves_torch(pred_curves, num_samples=num_samples)
    tgt_samples = sample_bezier_curves_torch(tgt_curves, num_samples=num_samples)
    pairwise = torch.cdist(pred_samples, tgt_samples)
    return 0.5 * (
        pairwise.min(dim=2).values.mean(dim=1) +
        pairwise.min(dim=1).values.mean(dim=1)
    )


def evaluate_state(
    pred_curves: torch.Tensor,
    pred_logits: torch.Tensor,
    target_curves: torch.Tensor,
    matching: List[Tuple[torch.Tensor, torch.Tensor]],
    num_curve_samples: int,
) -> Dict[str, float]:
    src_idx, tgt_idx = matching[0]
    object_prob = pred_logits.softmax(dim=-1)[..., 0]
    matched_prob = object_prob[src_idx].mean().item() if src_idx.numel() > 0 else 0.0
    unmatched_mask = torch.ones((pred_curves.shape[0],), dtype=torch.bool, device=pred_curves.device)
    unmatched_mask[src_idx] = False
    unmatched_prob = object_prob[unmatched_mask].mean().item() if bool(unmatched_mask.any()) else 0.0
    if src_idx.numel() == 0:
        zero = 0.0
        return {
            'matched_ctrl_l1': zero,
            'matched_endpoint_l1': zero,
            'matched_sample_l1': zero,
            'matched_chamfer': zero,
            'matched_one_minus_giou': zero,
            'matched_prob_mean': matched_prob,
            'unmatched_prob_mean': unmatched_prob,
        }
    matched_pred = pred_curves[src_idx]
    matched_tgt = target_curves[tgt_idx]
    ctrl_l1, endpoint_l1, oriented_tgt = aligned_curve_endpoint_l1(
        matched_pred,
        matched_tgt,
        ctrl_weight=1.0,
        endpoint_weight=1.0,
    )
    sample_l1 = orientation_aware_sample_l1(matched_pred, oriented_tgt, num_samples=num_curve_samples)
    chamfer = orientation_aware_chamfer(matched_pred, oriented_tgt, num_samples=num_curve_samples)
    giou = matched_generalized_box_iou(curve_boxes_xyxy(matched_pred), curve_boxes_xyxy(oriented_tgt))
    return {
        'matched_ctrl_l1': float(ctrl_l1.mean().item()),
        'matched_endpoint_l1': float(endpoint_l1.mean().item()),
        'matched_sample_l1': float(sample_l1.mean().item()),
        'matched_chamfer': float(chamfer.mean().item()),
        'matched_one_minus_giou': float((1.0 - giou).mean().item()),
        'matched_prob_mean': matched_prob,
        'unmatched_prob_mean': unmatched_prob,
    }


def plot_curves(ax, curves: torch.Tensor, colors: Sequence, linewidth: float = 2.0, alpha: float = 1.0, background: str = 'black') -> None:
    ax.set_facecolor(background)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 0.0)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    samples = sample_bezier_curves_torch(curves.detach().cpu(), num_samples=64).numpy()
    for pts, color in zip(samples, colors):
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=linewidth, alpha=alpha)


def render_snapshot(
    step: int,
    max_steps: int,
    pred_curves: torch.Tensor,
    pred_logits: torch.Tensor,
    target_curves: torch.Tensor,
    matching: List[Tuple[torch.Tensor, torch.Tensor]],
    history: List[Dict[str, float]],
    preset_name: str,
) -> plt.Figure:
    src_idx, tgt_idx = matching[0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle(f'{preset_name} | step {step}/{max_steps}', fontsize=16)

    gt_colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(target_curves.shape[0], 1)))
    plot_curves(axes[0, 0], target_curves, gt_colors, linewidth=2.5)
    axes[0, 0].set_title('GT Curves')

    probs = pred_logits.softmax(dim=-1)[:, 0].detach().cpu().numpy()
    samples = sample_bezier_curves_torch(pred_curves.detach().cpu(), num_samples=64).numpy()
    axes[0, 1].set_facecolor('black')
    axes[0, 1].set_xlim(0.0, 1.0)
    axes[0, 1].set_ylim(1.0, 0.0)
    axes[0, 1].set_aspect('equal')
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    matched_set = set(src_idx.detach().cpu().tolist())
    for idx, pts in enumerate(samples):
        color = plt.cm.viridis(float(probs[idx]))
        linewidth = 2.5 if idx in matched_set else 1.2
        alpha = 0.95 if idx in matched_set else 0.4
        axes[0, 1].plot(pts[:, 0], pts[:, 1], color=color, linewidth=linewidth, alpha=alpha)
    axes[0, 1].set_title('All Predictions (color = p(object))')

    axes[1, 0].set_facecolor('black')
    axes[1, 0].set_xlim(0.0, 1.0)
    axes[1, 0].set_ylim(1.0, 0.0)
    axes[1, 0].set_aspect('equal')
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    gt_samples = sample_bezier_curves_torch(target_curves.detach().cpu(), num_samples=64).numpy()
    for pts in gt_samples:
        axes[1, 0].plot(pts[:, 0], pts[:, 1], color='white', linewidth=2.0, alpha=0.4)
    matched_src = src_idx.detach().cpu().tolist()
    matched_tgt = tgt_idx.detach().cpu().tolist()
    for pair_idx, (s_idx, t_idx) in enumerate(zip(matched_src, matched_tgt)):
        color = gt_colors[t_idx % len(gt_colors)]
        pts_pred = samples[s_idx]
        pts_tgt = gt_samples[t_idx]
        axes[1, 0].plot(pts_tgt[:, 0], pts_tgt[:, 1], color=color, linewidth=2.0, alpha=0.35)
        axes[1, 0].plot(pts_pred[:, 0], pts_pred[:, 1], color=color, linewidth=2.2, alpha=1.0)
    axes[1, 0].set_title('Matched Pairs Overlay')

    axes[1, 1].plot([item['step'] for item in history], [item['loss_total'] for item in history], label='loss_total', linewidth=2.0)
    axes[1, 1].plot([item['step'] for item in history], [item['matched_ctrl_l1'] for item in history], label='matched_ctrl_l1', linewidth=1.5)
    axes[1, 1].plot([item['step'] for item in history], [item['matched_endpoint_l1'] for item in history], label='matched_endpoint_l1', linewidth=1.5)
    axes[1, 1].plot([item['step'] for item in history], [item['matched_sample_l1'] for item in history], label='matched_sample_l1', linewidth=1.5)
    axes[1, 1].set_title('Optimization Curves')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].legend(loc='upper right', fontsize=9)
    axes[1, 1].grid(alpha=0.3)

    latest = history[-1]
    metrics_text = (
        f"matched ctrl L1: {latest['matched_ctrl_l1']:.4f}\n"
        f"matched endpoint L1: {latest['matched_endpoint_l1']:.4f}\n"
        f"matched sample L1: {latest['matched_sample_l1']:.4f}\n"
        f"matched 1-GIoU: {latest['matched_one_minus_giou']:.4f}\n"
        f"matched p(obj): {latest['matched_prob_mean']:.3f}\n"
        f"unmatched p(obj): {latest['unmatched_prob_mean']:.3f}"
    )
    axes[1, 1].text(
        0.02, 0.98, metrics_text,
        transform=axes[1, 1].transAxes,
        va='top',
        ha='left',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
    )
    return fig


def figure_to_array(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def save_mp4(frame_paths: List[Path], output_path: Path, fps: int) -> None:
    if not frame_paths:
        return
    with tempfile.TemporaryDirectory(prefix='synthetic_probe_frames_', dir=output_path.parent) as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        for idx, frame_path in enumerate(frame_paths):
            dst = tmp_dir_path / f'frame_{idx:05d}.png'
            try:
                os.symlink(frame_path.resolve(), dst)
            except OSError:
                shutil.copy2(frame_path, dst)
        input_pattern = str(tmp_dir_path / 'frame_%05d.png')
        cmd = [
            'ffmpeg',
            '-y',
            '-framerate',
            str(fps),
            '-i',
            input_pattern,
            '-pix_fmt',
            'yuv420p',
            str(output_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_single_preset(
    preset_name: str,
    base_config: Dict,
    target_curves: torch.Tensor,
    init_curve_params: torch.Tensor,
    init_logits: torch.Tensor,
    output_root: Path,
    steps: int,
    lr_curves: float,
    lr_logits: float,
    save_every: int,
    device: torch.device,
) -> PresetResult:
    config = preset_loss_config(base_config, preset_name)
    loss_computer = build_loss_computer(config)
    preset_dir = output_root / preset_name
    frames_dir = preset_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    curve_params = torch.nn.Parameter(init_curve_params.clone().to(device))
    logits_params = torch.nn.Parameter(init_logits.clone().to(device))
    optimizer = torch.optim.Adam([
        {'params': [curve_params], 'lr': lr_curves},
        {'params': [logits_params], 'lr': lr_logits},
    ])

    targets = build_target(target_curves.to(device))
    history: List[Dict[str, float]] = []
    snapshot_steps = sorted(set([0, 1, 2, 5, 10, 20, 50, 100, steps - 1]))
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pred_curves = curve_params.sigmoid()
        pred_logits = logits_params
        outputs = {
            'pred_curves': pred_curves.unsqueeze(0),
            'pred_logits': pred_logits.unsqueeze(0),
        }
        losses = loss_computer(outputs, targets)
        total_loss = losses['loss']
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            current_curves = curve_params.sigmoid().detach()
            current_logits = logits_params.detach()
            metrics = evaluate_state(
                current_curves,
                current_logits,
                target_curves.to(device),
                losses['matching'],
                num_curve_samples=int(config['loss'].get('num_curve_samples', 24)),
            )
            record = {
                'step': step,
                'loss_total': float(total_loss.item()),
                'loss_ce': float(losses.get('loss_ce', total_loss * 0.0).item()),
                'loss_ctrl': float(losses.get('loss_ctrl', total_loss * 0.0).item()),
                'loss_sample': float(losses.get('loss_sample', total_loss * 0.0).item()),
                'loss_endpoint': float(losses.get('loss_endpoint', total_loss * 0.0).item()),
                'loss_giou': float(losses.get('loss_giou', total_loss * 0.0).item()),
                'loss_curve_dist': float(losses.get('loss_curve_dist', total_loss * 0.0).item()),
                **metrics,
            }
            history.append(record)

            if step in snapshot_steps or step % save_every == 0 or step == steps - 1:
                fig = render_snapshot(
                    step=step,
                    max_steps=steps - 1,
                    pred_curves=current_curves.cpu(),
                    pred_logits=current_logits.cpu(),
                    target_curves=target_curves.cpu(),
                    matching=losses['matching'],
                    history=history,
                    preset_name=preset_name,
                )
                frame_path = frames_dir / f'frame_{step:05d}.png'
                fig.savefig(frame_path, dpi=150)
                plt.close(fig)

    history_path = preset_dir / 'history.json'
    history_path.write_text(json.dumps(history, indent=2))
    with open(preset_dir / 'history.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    frame_paths = sorted(frames_dir.glob('frame_*.png'))
    pdf_path = preset_dir / 'optimization_snapshots.pdf'
    with PdfPages(pdf_path) as pdf:
        for frame_path in frame_paths:
            image = plt.imread(frame_path)
            fig, ax = plt.subplots(figsize=(12, 9))
            ax.imshow(image)
            ax.axis('off')
            ax.set_title(frame_path.stem)
            pdf.savefig(fig)
            plt.close(fig)

    mp4_path = preset_dir / 'optimization.mp4'
    save_mp4(frame_paths, mp4_path, fps=8)

    summary = dict(history[-1])
    summary['best_loss_total'] = min(item['loss_total'] for item in history)
    summary['best_matched_sample_l1'] = min(item['matched_sample_l1'] for item in history)
    summary['pdf'] = str(pdf_path)
    summary['mp4'] = str(mp4_path)
    (preset_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    return PresetResult(name=preset_name, output_dir=preset_dir, history=history, summary=summary)


def build_compare_pdf(results: List[PresetResult], output_path: Path) -> None:
    with PdfPages(output_path) as pdf:
        fig, axes = plt.subplots(2, max(len(results), 1), figsize=(6 * max(len(results), 1), 10), constrained_layout=True)
        if len(results) == 1:
            axes = np.asarray(axes).reshape(2, 1)
        for col, result in enumerate(results):
            final_frame = sorted((result.output_dir / 'frames').glob('frame_*.png'))[-1]
            image = plt.imread(final_frame)
            axes[0, col].imshow(image)
            axes[0, col].axis('off')
            axes[0, col].set_title(f'{result.name} final snapshot')

            history = result.history
            axes[1, col].plot([item['step'] for item in history], [item['loss_total'] for item in history], label='loss_total', linewidth=2.0)
            axes[1, col].plot([item['step'] for item in history], [item['matched_sample_l1'] for item in history], label='matched_sample_l1', linewidth=1.5)
            axes[1, col].plot([item['step'] for item in history], [item['matched_one_minus_giou'] for item in history], label='matched_1-GIoU', linewidth=1.5)
            axes[1, col].grid(alpha=0.3)
            axes[1, col].legend(fontsize=9)
            axes[1, col].set_title(result.name)
            axes[1, col].set_xlabel('Step')
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Synthetic Bezier loss sanity probe')
    parser.add_argument('--presets', nargs='+', default=['current_main', 'ctrl_endpoint_giou_only'])
    parser.add_argument('--num-targets', type=int, default=8)
    parser.add_argument('--num-preds', type=int, default=16)
    parser.add_argument('--degree', type=int, default=5)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--lr-curves', type=float, default=0.05)
    parser.add_argument('--lr-logits', type=float, default=0.05)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20260323)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    if args.num_preds <= args.num_targets:
        raise ValueError('num-preds must be greater than num-targets so unmatched queries exist.')

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    base_config = load_config('configs/parametric_edge/default.yaml')

    target_curves = generate_gt_curves(args.num_targets, args.degree, rng)
    init_curve_params = torch.randn((args.num_preds, args.degree + 1, 2), dtype=torch.float32)
    init_logits = torch.randn((args.num_preds, 2), dtype=torch.float32) * 0.05

    if args.output_dir is None:
        output_root = Path('outputs/synthetic_loss_probe') / f'seed_{args.seed}_N{args.num_targets}_M{args.num_preds}_T{args.steps}'
    else:
        output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    gt_path = output_root / 'gt_curves.pt'
    torch.save(target_curves, gt_path)
    (output_root / 'run_config.json').write_text(json.dumps({
        'seed': args.seed,
        'num_targets': args.num_targets,
        'num_preds': args.num_preds,
        'degree': args.degree,
        'steps': args.steps,
        'lr_curves': args.lr_curves,
        'lr_logits': args.lr_logits,
        'presets': args.presets,
        'device': str(device),
    }, indent=2))

    results = []
    for preset_name in args.presets:
        results.append(run_single_preset(
            preset_name=preset_name,
            base_config=base_config,
            target_curves=target_curves,
            init_curve_params=init_curve_params,
            init_logits=init_logits,
            output_root=output_root,
            steps=args.steps,
            lr_curves=args.lr_curves,
            lr_logits=args.lr_logits,
            save_every=args.save_every,
            device=device,
        ))

    compare = {
        result.name: result.summary
        for result in results
    }
    compare_path = output_root / 'compare_summary.json'
    compare_path.write_text(json.dumps(compare, indent=2))
    build_compare_pdf(results, output_root / 'compare_report.pdf')
    print(json.dumps({
        'output_root': str(output_root),
        'compare_summary': str(compare_path),
        'compare_report_pdf': str(output_root / 'compare_report.pdf'),
        'presets': [result.name for result in results],
    }, indent=2))


if __name__ == '__main__':
    main()
