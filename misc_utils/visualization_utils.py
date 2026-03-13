from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from misc_utils.bezier_target_utils import denormalize_control_points, sample_bezier_numpy

PALETTE = ['#ff595e', '#ffca3a', '#8ac926', '#1982c4', '#6a4c93', '#f72585', '#4cc9f0', '#fb5607']


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
    titles: Iterable[str] = ('Input', 'Target Curves', 'Predicted Curves', 'Predicted Curves Only'),
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = min(len(targets), 4)
    fig, axes = plt.subplots(batch_size, 4, figsize=(16, 4 * batch_size))
    if batch_size == 1:
        axes = np.expand_dims(axes, axis=0)
    titles = list(titles)
    for row in range(batch_size):
        image = images[row].detach().cpu().numpy()
        target_curves = targets[row]['curves'].detach().cpu().numpy()
        pred_curves = predictions[row].detach().cpu().numpy() if isinstance(predictions[row], torch.Tensor) else np.asarray(predictions[row])
        if image.shape[0] == 1:
            image_np = image[0]
            cmap = 'gray'
        else:
            image_np = np.transpose(image, (1, 2, 0))
            cmap = None
        for col in range(3):
            axes[row, col].imshow(image_np, cmap=cmap, vmin=0.0, vmax=1.0)
            axes[row, col].axis('off')
            axes[row, col].set_title(titles[col])
        axes[row, 3].imshow(np.zeros((image_np.shape[0], image_np.shape[1])), cmap='gray', vmin=0.0, vmax=1.0)
        axes[row, 3].axis('off')
        axes[row, 3].set_title(titles[3])
        height, width = image_np.shape[:2]
        for idx, curve in enumerate(target_curves):
            draw_curve(axes[row, 1], curve, width, height, PALETTE[idx % len(PALETTE)])
        for idx, curve in enumerate(pred_curves):
            draw_curve(axes[row, 2], curve, width, height, PALETTE[idx % len(PALETTE)])
            draw_curve(axes[row, 3], curve, width, height, '#ffffff', linewidth=2.0, show_control_points=False)
        axes[row, 3].set_facecolor('black')
        axes[row, 3].set_xlim(0, width)
        axes[row, 3].set_ylim(height, 0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
