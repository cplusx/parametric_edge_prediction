from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage

from misc_utils.bezier_target_utils import denormalize_control_points, sample_bezier_numpy

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
