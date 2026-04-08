from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from misc_utils.bezier_target_utils import denormalize_control_points, sample_bezier_numpy
from misc_utils.endpoint_flow_utils import scale_points_from_flow

PALETTE = ['#ff595e', '#ffca3a', '#8ac926', '#1982c4', '#6a4c93', '#f72585', '#4cc9f0', '#fb5607']


def _edge_overlay(image_np: np.ndarray, edge_mask_np: np.ndarray) -> np.ndarray:
    edge_mask_np = np.asarray(edge_mask_np, dtype=np.float32) > 0.0
    if edge_mask_np.any():
        edge_mask_np = ndimage.binary_dilation(edge_mask_np, iterations=1)
    if image_np.ndim == 2:
        base = np.repeat(image_np[..., None], 3, axis=2)
    else:
        base = np.asarray(image_np, dtype=np.float32).copy()
    overlay = base.copy()
    overlay[edge_mask_np] = np.asarray([1.0, 0.2, 0.1], dtype=np.float32)
    return np.clip(overlay, 0.0, 1.0)


def draw_curve(
    ax,
    control_points: np.ndarray,
    width: int,
    height: int,
    color: str,
    linewidth: float = 2.0,
    show_control_points: bool = True,
) -> None:
    curve = denormalize_control_points(control_points, width, height)
    samples = sample_bezier_numpy(curve)
    ax.plot(samples[:, 0], samples[:, 1], color=color, linewidth=linewidth)
    if show_control_points:
        ax.scatter(curve[:, 0], curve[:, 1], color=color, s=10)


def render_curve_grid(
    images: torch.Tensor,
    targets: List[dict],
    predictions: List[torch.Tensor],
    output_path: Path,
    titles: Iterable[str] = ('Input', 'GT Edge', 'Target Curves', 'Predicted Curves', 'Predicted Curves Only'),
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = min(len(targets), 4)
    has_edge_mask = any('edge_mask' in target for target in targets[:batch_size])
    num_cols = 5 if has_edge_mask else 4
    fig, axes = plt.subplots(batch_size, num_cols, figsize=(4 * num_cols, 4 * batch_size))
    if batch_size == 1:
        axes = np.expand_dims(axes, axis=0)
    titles = list(titles)
    for row in range(batch_size):
        image = images[row].detach().cpu().numpy()
        target_curves = targets[row]['curves'].detach().cpu().numpy()
        pred_curves = predictions[row].detach().cpu().numpy() if isinstance(predictions[row], torch.Tensor) else np.asarray(predictions[row])
        edge_mask = targets[row].get('edge_mask')
        edge_mask_np = edge_mask.detach().cpu().numpy()[0] if edge_mask is not None else None
        if image.shape[0] == 1:
            image_np = image[0]
            cmap = 'gray'
        else:
            image_np = np.transpose(image, (1, 2, 0))
            cmap = None
        axes[row, 0].imshow(image_np, cmap=cmap, vmin=0.0, vmax=1.0)
        axes[row, 0].axis('off')
        axes[row, 0].set_title(titles[0])
        target_col = 1
        pred_col = 2
        pred_only_col = 3
        if has_edge_mask:
            edge_display = _edge_overlay(image_np, edge_mask_np) if edge_mask_np is not None else image_np
            axes[row, 1].imshow(edge_display, vmin=0.0, vmax=1.0)
            axes[row, 1].axis('off')
            axes[row, 1].set_title(titles[1])
            target_col = 2
            pred_col = 3
            pred_only_col = 4
        for col in (target_col, pred_col):
            axes[row, col].imshow(image_np, cmap=cmap, vmin=0.0, vmax=1.0)
            axes[row, col].axis('off')
            axes[row, col].set_title(titles[col])
        axes[row, pred_only_col].imshow(np.zeros((image_np.shape[0], image_np.shape[1])), cmap='gray', vmin=0.0, vmax=1.0)
        axes[row, pred_only_col].axis('off')
        axes[row, pred_only_col].set_title(titles[pred_only_col])
        height, width = image_np.shape[:2]
        for col in range(num_cols):
            axes[row, col].set_xlim(0, width)
            axes[row, col].set_ylim(height, 0)
            axes[row, col].set_aspect('equal', adjustable='box')
        for idx, curve in enumerate(target_curves):
            draw_curve(axes[row, target_col], curve, width, height, PALETTE[idx % len(PALETTE)])
        for idx, curve in enumerate(pred_curves):
            draw_curve(axes[row, pred_col], curve, width, height, PALETTE[idx % len(PALETTE)])
            draw_curve(axes[row, pred_only_col], curve, width, height, '#ffffff', linewidth=2.0, show_control_points=False)
        axes[row, pred_only_col].set_facecolor('black')
        axes[row, pred_only_col].set_xlim(0, width)
        axes[row, pred_only_col].set_ylim(height, 0)
        axes[row, pred_only_col].set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _draw_points(ax, points: np.ndarray, width: int, height: int, color: str, size: float = 18.0) -> None:
    if points.size == 0:
        return
    pts = np.asarray(points, dtype=np.float32).copy()
    pts[:, 0] *= float(width)
    pts[:, 1] *= float(height)
    ax.scatter(pts[:, 0], pts[:, 1], s=size, c=color, edgecolors='none')


def render_point_grid(
    images: torch.Tensor,
    targets: List[dict],
    predictions: List[torch.Tensor],
    output_path: Path,
    titles: Iterable[str] = ('Input', 'GT Edge', 'Target Endpoints', 'Predicted Endpoints', 'Predicted Endpoints Only'),
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = min(len(targets), 4)
    has_edge_mask = any('edge_mask' in target for target in targets[:batch_size])
    num_cols = 5 if has_edge_mask else 4
    fig, axes = plt.subplots(batch_size, num_cols, figsize=(4 * num_cols, 4 * batch_size))
    if batch_size == 1:
        axes = np.expand_dims(axes, axis=0)
    titles = list(titles)
    for row in range(batch_size):
        image = images[row].detach().cpu().numpy()
        target_points = targets[row]['points'].detach().cpu().numpy()
        pred_points = predictions[row].detach().cpu().numpy() if isinstance(predictions[row], torch.Tensor) else np.asarray(predictions[row])
        edge_mask = targets[row].get('edge_mask')
        edge_mask_np = edge_mask.detach().cpu().numpy()[0] if edge_mask is not None else None
        if image.shape[0] == 1:
            image_np = image[0]
            cmap = 'gray'
        else:
            image_np = np.transpose(image, (1, 2, 0))
            cmap = None
        axes[row, 0].imshow(image_np, cmap=cmap, vmin=0.0, vmax=1.0)
        axes[row, 0].axis('off')
        axes[row, 0].set_title(titles[0])
        target_col = 1
        pred_col = 2
        pred_only_col = 3
        if has_edge_mask:
            edge_display = _edge_overlay(image_np, edge_mask_np) if edge_mask_np is not None else image_np
            axes[row, 1].imshow(edge_display, vmin=0.0, vmax=1.0)
            axes[row, 1].axis('off')
            axes[row, 1].set_title(titles[1])
            target_col = 2
            pred_col = 3
            pred_only_col = 4
        for col in (target_col, pred_col):
            axes[row, col].imshow(image_np, cmap=cmap, vmin=0.0, vmax=1.0)
            axes[row, col].axis('off')
            axes[row, col].set_title(titles[col])
        axes[row, pred_only_col].imshow(np.zeros((image_np.shape[0], image_np.shape[1])), cmap='gray', vmin=0.0, vmax=1.0)
        axes[row, pred_only_col].axis('off')
        axes[row, pred_only_col].set_title(titles[pred_only_col])
        height, width = image_np.shape[:2]
        for col in range(num_cols):
            axes[row, col].set_xlim(0, width)
            axes[row, col].set_ylim(height, 0)
            axes[row, col].set_aspect('equal', adjustable='box')
        for idx, point in enumerate(target_points):
            _draw_points(axes[row, target_col], point[None, :], width, height, PALETTE[idx % len(PALETTE)], size=20.0)
        _draw_points(axes[row, pred_col], pred_points, width, height, '#ffffff', size=16.0)
        _draw_points(axes[row, pred_only_col], pred_points, width, height, '#ffffff', size=16.0)
        axes[row, pred_only_col].set_facecolor('black')
        axes[row, pred_only_col].set_xlim(0, width)
        axes[row, pred_only_col].set_ylim(height, 0)
        axes[row, pred_only_col].set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _flow_points_to_pixels(points_flow: np.ndarray, width: int, height: int) -> np.ndarray:
    points = scale_points_from_flow(torch.as_tensor(points_flow)).detach().cpu().numpy().astype(np.float32)
    points[:, 0] *= float(width)
    points[:, 1] *= float(height)
    return points


def _flow_velocity_to_pixels(velocity_flow: np.ndarray, width: int, height: int) -> np.ndarray:
    velocity = (np.asarray(velocity_flow, dtype=np.float32) * 0.5).copy()
    velocity[:, 0] *= float(width)
    velocity[:, 1] *= float(height)
    return velocity


def render_flow_training_grid(
    images: torch.Tensor,
    targets: List[dict],
    noisy_points: torch.Tensor,
    gt_velocity: torch.Tensor,
    pred_velocity: torch.Tensor,
    timestep_indices: torch.Tensor,
    valid_mask: torch.Tensor,
    output_path: Path,
    num_train_timesteps: int = 1000,
    titles: Iterable[str] = ('Input', 'GT Edge', 'Noise + GT Velocity', 'Noise + Pred Velocity'),
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid_per_sample = valid_mask.detach().cpu().sum(dim=1)
    sample_idx = int(valid_per_sample.argmax().item()) if valid_per_sample.numel() > 0 else 0

    image = images[sample_idx].detach().cpu().numpy()
    edge_mask = targets[sample_idx].get('edge_mask')
    edge_mask_np = edge_mask.detach().cpu().numpy()[0] if edge_mask is not None else None
    if image.shape[0] == 1:
        image_np = image[0]
        cmap = 'gray'
    else:
        image_np = np.transpose(image, (1, 2, 0))
        cmap = None
    height, width = image_np.shape[:2]

    mask_np = valid_mask[sample_idx].detach().cpu().numpy().astype(bool)
    noisy_np = noisy_points[sample_idx].detach().cpu().numpy()[mask_np]
    gt_vel_np = gt_velocity[sample_idx].detach().cpu().numpy()[mask_np]
    pred_vel_np = pred_velocity[sample_idx].detach().cpu().numpy()[mask_np]
    noisy_px = _flow_points_to_pixels(noisy_np, width, height) if noisy_np.size else np.zeros((0, 2), dtype=np.float32)
    timestep = int(timestep_indices[sample_idx].detach().cpu().item())
    sigma = (float(timestep) + 0.5) / float(max(int(num_train_timesteps), 1))
    gt_vel_px = _flow_velocity_to_pixels(gt_vel_np * sigma, width, height) if gt_vel_np.size else np.zeros((0, 2), dtype=np.float32)
    pred_vel_px = _flow_velocity_to_pixels(pred_vel_np * sigma, width, height) if pred_vel_np.size else np.zeros((0, 2), dtype=np.float32)

    has_edge_mask = edge_mask_np is not None
    num_cols = 4 if has_edge_mask else 3
    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))
    if num_cols == 1:
        axes = np.asarray([axes])
    titles = list(titles)

    axes[0].imshow(image_np, cmap=cmap, vmin=0.0, vmax=1.0)
    axes[0].axis('off')
    axes[0].set_title(titles[0])

    gt_col = 1
    pred_col = 2
    if has_edge_mask:
        edge_display = _edge_overlay(image_np, edge_mask_np)
        axes[1].imshow(edge_display, vmin=0.0, vmax=1.0)
        axes[1].axis('off')
        axes[1].set_title(titles[1])
        gt_col = 2
        pred_col = 3

    for col, title in ((gt_col, titles[gt_col]), (pred_col, titles[pred_col])):
        axes[col].imshow(image_np, cmap=cmap, vmin=0.0, vmax=1.0)
        axes[col].axis('off')
        axes[col].set_title(title)
        axes[col].text(
            0.01,
            0.99,
            f't={timestep}',
            transform=axes[col].transAxes,
            ha='left',
            va='top',
            color='white',
            fontsize=10,
            bbox={'facecolor': 'black', 'alpha': 0.65, 'pad': 2},
        )

    for ax in axes:
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_aspect('equal', adjustable='box')

    if noisy_px.size:
        for ax in (axes[gt_col], axes[pred_col]):
            ax.scatter(noisy_px[:, 0], noisy_px[:, 1], s=10, c='#4cc9f0', edgecolors='none', alpha=0.9)
        axes[gt_col].quiver(
            noisy_px[:, 0], noisy_px[:, 1], gt_vel_px[:, 0], gt_vel_px[:, 1],
            angles='xy', scale_units='xy', scale=1.0, color='white', width=0.003, alpha=0.9,
        )
        axes[pred_col].quiver(
            noisy_px[:, 0], noisy_px[:, 1], pred_vel_px[:, 0], pred_vel_px[:, 1],
            angles='xy', scale_units='xy', scale=1.0, color='#ff595e', width=0.003, alpha=0.9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_point_trajectory_animation(
    images: torch.Tensor,
    targets: List[dict],
    trajectory_points: torch.Tensor,
    output_gif_path: Path,
    output_mp4_path: Optional[Path] = None,
    title: str = 'Validation Flow Trajectory',
    fps: int = 5,
) -> None:
    output_gif_path = Path(output_gif_path)
    output_gif_path.parent.mkdir(parents=True, exist_ok=True)
    if output_mp4_path is not None:
        output_mp4_path = Path(output_mp4_path)
        output_mp4_path.parent.mkdir(parents=True, exist_ok=True)

    batch_idx = 0
    if targets:
        batch_idx = int(max(range(len(targets)), key=lambda i: int(targets[i]['points'].shape[0])))

    image = images[batch_idx].detach().cpu().numpy()
    target_points = targets[batch_idx]['points'].detach().cpu().numpy()
    edge_mask = targets[batch_idx].get('edge_mask')
    edge_mask_np = edge_mask.detach().cpu().numpy()[0] if edge_mask is not None else None
    if image.shape[0] == 1:
        image_np = np.repeat(image[0][..., None], 3, axis=2)
    else:
        image_np = np.transpose(image, (1, 2, 0))
    background = _edge_overlay(image_np, edge_mask_np) if edge_mask_np is not None else image_np
    height, width = image_np.shape[:2]

    trajectory_np = trajectory_points[batch_idx].detach().cpu().numpy()
    num_frames = int(trajectory_np.shape[0])
    frames: List[Image.Image] = []

    gt_px = target_points.astype(np.float32).copy()
    if gt_px.size:
        gt_px[:, 0] *= float(width)
        gt_px[:, 1] *= float(height)

    for frame_idx in range(num_frames):
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5))
        ax.imshow(background, vmin=0.0, vmax=1.0)
        ax.axis('off')
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title)
        ax.text(
            0.01,
            0.99,
            f'step={frame_idx}/{num_frames - 1}',
            transform=ax.transAxes,
            ha='left',
            va='top',
            color='white',
            fontsize=10,
            bbox={'facecolor': 'black', 'alpha': 0.65, 'pad': 2},
        )

        current = trajectory_np[frame_idx].astype(np.float32).copy()
        current[:, 0] *= float(width)
        current[:, 1] *= float(height)

        history = trajectory_np[: frame_idx + 1].astype(np.float32).copy()
        history[..., 0] *= float(width)
        history[..., 1] *= float(height)

        if gt_px.size:
            ax.scatter(gt_px[:, 0], gt_px[:, 1], s=22, c='#ffca3a', edgecolors='#ff595e', linewidths=0.6, alpha=0.95, label='GT')

        for point_idx in range(history.shape[1]):
            trail = history[:, point_idx, :]
            ax.plot(trail[:, 0], trail[:, 1], color='white', linewidth=0.8, alpha=0.28)

        ax.scatter(current[:, 0], current[:, 1], s=16, c='#4cc9f0', edgecolors='none', alpha=0.95, label='Current')
        fig.tight_layout()
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(Image.fromarray(rgb))
        plt.close(fig)

    if not frames:
        return
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(int(1000 / max(int(fps), 1)), 1),
        loop=0,
    )

    if output_mp4_path is not None:
        try:
            import imageio.v2 as imageio

            imageio.mimsave(
                output_mp4_path,
                [np.asarray(frame) for frame in frames],
                fps=max(int(fps), 1),
            )
        except Exception:
            pass
