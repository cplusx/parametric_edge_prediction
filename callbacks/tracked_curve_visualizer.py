from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

from misc_utils.train_utils import sample_bezier_curves_torch
from models.curve_coordinates import curve_internal_to_external
from models.matcher import HungarianCurveMatcher


def _plot_curve(ax, curves: torch.Tensor, color: str, linewidth: float = 2.0, show_points: bool = True) -> None:
    if curves.numel() == 0:
        return
    samples = sample_bezier_curves_torch(curves, num_samples=96)
    for curve, sample in zip(curves.detach().cpu(), samples.detach().cpu()):
        xy = sample.numpy()
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=linewidth)
        if show_points:
            ctrl = curve.numpy()
            ax.scatter(ctrl[:, 0], ctrl[:, 1], color=color, s=10)


class TrackedCurveVisualizer(pl.Callback):
    def __init__(
        self,
        sample_id: str,
        target_idx: int,
        every_n_epochs: int = 10,
        output_subdir: str = "tracked_curve",
    ) -> None:
        super().__init__()
        self.sample_id = sample_id
        self.target_idx = int(target_idx)
        self.every_n_epochs = int(every_n_epochs)
        self.output_subdir = output_subdir

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        if batch_idx != 0 or self.every_n_epochs <= 0:
            return
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        sample_ids = [target.get("sample_id") for target in batch["targets"]]
        if self.sample_id not in sample_ids:
            return
        sample_batch_idx = sample_ids.index(self.sample_id)
        if self.target_idx >= batch["targets"][sample_batch_idx]["curves"].shape[0]:
            return

        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            predictions = pl_module(batch["images"], targets=batch["targets"])
        if was_training:
            pl_module.train()

        matcher = HungarianCurveMatcher.from_config(pl_module.config)
        matched = matcher(
            logits=predictions["pred_logits"],
            curves=predictions["pred_curves"],
            targets=batch["targets"],
        )[sample_batch_idx]
        src_idx, tgt_idx = matched
        matched_pred = None
        mask = tgt_idx == self.target_idx
        if mask.any():
            matched_pred = int(src_idx[mask][0].item())
        if matched_pred is None:
            return

        probs = predictions["pred_logits"][sample_batch_idx].softmax(-1)[:, 0]
        pred_curves_external = curve_internal_to_external(predictions["pred_curves"][sample_batch_idx], pl_module.config)
        target_curve_external = batch["targets"][sample_batch_idx]["curves"][self.target_idx:self.target_idx + 1]
        pred_curve_external = pred_curves_external[matched_pred:matched_pred + 1]
        pred_end = pred_curve_external[:, [0, -1]]
        tgt_end = target_curve_external[:, [0, -1]]
        endpoint_l1_external = torch.abs(pred_end - tgt_end).mean()

        image = batch["images"][sample_batch_idx].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        h, w = image.shape[:2]
        out_dir = Path(trainer.default_root_dir) / self.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
        axes[0].imshow(image)
        _plot_curve(axes[0], target_curve_external, color="red", linewidth=3.0)
        _plot_curve(axes[0], pred_curve_external, color="lime", linewidth=2.2)
        axes[0].set_title(
            f"epoch {trainer.current_epoch}  pred {matched_pred}\n"
            f"score={float(probs[matched_pred].item()):.3f}  endpointL1={float(endpoint_l1_external.item()):.4f}"
        )
        axes[0].axis("off")

        axes[1].imshow(np.zeros((h, w)), cmap="gray", vmin=0.0, vmax=1.0)
        _plot_curve(axes[1], target_curve_external, color="red", linewidth=3.0, show_points=False)
        _plot_curve(axes[1], pred_curve_external, color="white", linewidth=2.2, show_points=False)
        axes[1].set_title("Target (red) vs matched pred (white)")
        axes[1].axis("off")
        axes[1].set_facecolor("black")

        fig.tight_layout()
        frame_path = out_dir / f"epoch_{trainer.current_epoch:03d}.jpg"
        fig.savefig(frame_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
