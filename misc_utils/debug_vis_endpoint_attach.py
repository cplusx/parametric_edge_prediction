from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from misc_utils.bezier_target_utils import sample_bezier_numpy


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _draw_curves(ax, curves: np.ndarray, *, color: str = "#1f77b4", linewidth: float = 1.5, alpha: float = 0.85) -> None:
    for curve in np.asarray(curves, dtype=np.float32):
        if curve.ndim != 2 or curve.shape[0] < 2:
            continue
        samples = sample_bezier_numpy(curve, num_samples=48)
        ax.plot(samples[:, 0], samples[:, 1], color=color, linewidth=linewidth, alpha=alpha)


def render_endpoint_attach_frame(
    *,
    target: dict,
    pred_points: np.ndarray,
    title: str,
    matched_indices: Optional[Tuple[Iterable[int], Iterable[int]]] = None,
    highlight_point_index: Optional[int] = None,
) -> np.ndarray:
    curves = _as_numpy(target.get("curves", np.zeros((0, 2, 2), dtype=np.float32)))
    points = _as_numpy(target.get("points", np.zeros((0, 2), dtype=np.float32)))
    degrees = _as_numpy(target.get("point_degree", np.zeros((points.shape[0],), dtype=np.int64))).astype(np.int64)
    loop_only = _as_numpy(target.get("point_is_loop_only", np.zeros((points.shape[0],), dtype=bool))).astype(bool)
    pred_points = np.asarray(pred_points, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    _draw_curves(ax, curves, color="#2f6fdb", linewidth=1.4)

    if points.size:
        high = (degrees >= 3) & (~loop_only)
        low = (degrees < 3) & (~loop_only)
        if np.any(low):
            ax.scatter(points[low, 0], points[low, 1], s=38, c="#2ca02c", label="degree<=2", zorder=5)
        if np.any(high):
            ax.scatter(points[high, 0], points[high, 1], s=42, c="#d62728", label="degree>=3", zorder=6)
        if np.any(loop_only):
            ax.scatter(
                points[loop_only, 0],
                points[loop_only, 1],
                s=68,
                facecolors="none",
                edgecolors="#ff7f0e",
                linewidths=1.8,
                label="loop-only",
                zorder=7,
            )

    if pred_points.size:
        ax.scatter(pred_points[:, 0], pred_points[:, 1], s=32, c="#ffd92f", edgecolors="#6b5b00", linewidths=0.5, label="pred", zorder=8)

    if matched_indices is not None and points.size and pred_points.size:
        src_idx, tgt_idx = matched_indices
        for src, tgt in zip(list(src_idx), list(tgt_idx)):
            src = int(src)
            tgt = int(tgt)
            if 0 <= src < pred_points.shape[0] and 0 <= tgt < points.shape[0]:
                ax.plot(
                    [pred_points[src, 0], points[tgt, 0]],
                    [pred_points[src, 1], points[tgt, 1]],
                    linestyle="--",
                    color="#777777",
                    linewidth=0.7,
                    alpha=0.55,
                    zorder=3,
                )

    if highlight_point_index is not None and points.size:
        idx = int(highlight_point_index)
        if 0 <= idx < points.shape[0]:
            ax.scatter([points[idx, 0]], [points[idx, 1]], s=150, facecolors="none", edgecolors="#000000", linewidths=2.0, zorder=10)

    ax.set_title(title, fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(1.05, -0.05)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.legend(loc="lower right", fontsize=6)
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    plt.close(fig)
    return frame


def save_frames_as_gif(frames: List[np.ndarray], path: Path, *, fps: float = 3.0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = 1.0 / max(float(fps), 1e-6)
    imageio.mimsave(path, frames, duration=duration)
