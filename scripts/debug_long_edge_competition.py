import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import yaml

from edge_datasets.parametric_edge_datamodule import ParametricEdgeDataModule
from misc_utils.train_utils import sample_bezier_curves_torch
from models.curve_coordinates import curve_external_to_internal, curve_internal_to_external
from models.geometry import (
    aligned_endpoint_l1,
    pairwise_curve_chamfer_cost,
    pairwise_curve_l1_forward_reverse_cost,
    pairwise_sample_l1_forward_reverse_cost,
)
from models.matcher import hungarian_curve_matching
from pl_trainer.parametric_edge_trainer import ParametricEdgeLightningModule


ROOT = Path("outputs/parametric_edge_training/overfit_diverse8_dab_curve_coordmargin")
CKPT = ROOT / "checkpoints" / "best-599-0.0612.ckpt"
HPARAMS = ROOT / "csv_logs" / "version_0" / "hparams.yaml"
OUT_DIR = Path("outputs/debug_long_edge_competition")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET_SAMPLE_ID = "183066_ann2"
TARGET_TGT_IDX = 0
TOPK = 8


def _load_config():
    with open(HPARAMS, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_model_and_batch():
    config = _load_config()
    seed = config.get("trainer", {}).get("seed")
    if seed is not None:
        pl.seed_everything(int(seed), workers=True)
    datamodule = ParametricEdgeDataModule(config)
    datamodule.setup("fit")
    val_loader = datamodule.val_dataloader()
    batch = next(iter(val_loader))
    model = ParametricEdgeLightningModule.load_from_checkpoint(str(CKPT), config=config, map_location="cpu")
    model.eval()
    with torch.no_grad():
        outputs = model(batch["images"], targets=batch["targets"])
    return config, batch, outputs


def _compute_cost_breakdown(config, batch, outputs, batch_idx, tgt_idx):
    loss_cfg = config["loss"]
    pred_logits = outputs["pred_logits"][batch_idx]
    pred_curves = outputs["pred_curves"][batch_idx]
    pred_probs = pred_logits.softmax(-1)[:, 0]
    tgt_curves_external = batch["targets"][batch_idx]["curves"]
    tgt_curves = curve_external_to_internal(tgt_curves_external, config)
    tgt_curve = tgt_curves[tgt_idx:tgt_idx + 1]

    ctrl_forward, ctrl_reverse = pairwise_curve_l1_forward_reverse_cost(pred_curves, tgt_curve)
    sample_forward, sample_reverse = pairwise_sample_l1_forward_reverse_cost(
        pred_curves,
        tgt_curve,
        num_samples=int(loss_cfg.get("num_curve_samples", 24)),
    )
    orientation_forward = (
        float(loss_cfg.get("control_cost", 1.5)) * ctrl_forward.squeeze(1)
        + float(loss_cfg.get("sample_cost", 2.0)) * sample_forward.squeeze(1)
    )
    orientation_reverse = (
        float(loss_cfg.get("control_cost", 1.5)) * ctrl_reverse.squeeze(1)
        + float(loss_cfg.get("sample_cost", 2.0)) * sample_reverse.squeeze(1)
    )
    use_reverse = orientation_reverse < orientation_forward
    orientation_cost = torch.minimum(orientation_forward, orientation_reverse)
    curve_dist = (
        float(loss_cfg.get("curve_distance_cost", 0.0))
        * pairwise_curve_chamfer_cost(
            pred_curves,
            tgt_curve,
            point_count=int(loss_cfg.get("curve_match_point_count", 4)),
        ).squeeze(1)
    )
    total_cost = orientation_cost + curve_dist

    endpoint_min = aligned_endpoint_l1(pred_curves, tgt_curve.expand_as(pred_curves))
    sorted_idx = torch.argsort(total_cost)

    matched = hungarian_curve_matching(
        outputs["pred_logits"][batch_idx:batch_idx + 1],
        outputs["pred_curves"][batch_idx:batch_idx + 1],
        [batch["targets"][batch_idx]],
        control_cost=float(loss_cfg.get("control_cost", 1.5)),
        sample_cost=float(loss_cfg.get("sample_cost", 2.0)),
        giou_cost=float(loss_cfg.get("giou_cost", 0.0)),
        curve_distance_cost=float(loss_cfg.get("curve_distance_cost", 0.0)),
        curve_match_point_count=int(loss_cfg.get("curve_match_point_count", 4)),
        num_curve_samples=int(loss_cfg.get("num_curve_samples", 24)),
        direction_invariant=bool(loss_cfg.get("direction_invariant", True)),
        config=config,
    )[0]
    src_idx, tgt_match_idx = matched
    assigned_target_for_pred = torch.full((pred_curves.shape[0],), -1, dtype=torch.long)
    assigned_target_for_pred[src_idx] = tgt_match_idx
    matched_pred_idx = int(src_idx[tgt_match_idx == tgt_idx][0].item())
    matched_rank = int((sorted_idx == matched_pred_idx).nonzero(as_tuple=False)[0].item())

    rows = []
    for rank, pred_idx in enumerate(sorted_idx.tolist()):
        rows.append({
            "rank": rank + 1,
            "pred_idx": pred_idx,
            "score": float(pred_probs[pred_idx].item()),
            "total_cost": float(total_cost[pred_idx].item()),
            "orientation_cost": float(orientation_cost[pred_idx].item()),
            "curve_distance_cost": float(curve_dist[pred_idx].item()),
            "ctrl_forward_cost": float((float(loss_cfg.get("control_cost", 1.5)) * ctrl_forward[pred_idx, 0]).item()),
            "ctrl_reverse_cost": float((float(loss_cfg.get("control_cost", 1.5)) * ctrl_reverse[pred_idx, 0]).item()),
            "sample_forward_cost": float((float(loss_cfg.get("sample_cost", 2.0)) * sample_forward[pred_idx, 0]).item()),
            "sample_reverse_cost": float((float(loss_cfg.get("sample_cost", 2.0)) * sample_reverse[pred_idx, 0]).item()),
            "use_reverse": bool(use_reverse[pred_idx].item()),
            "endpoint_l1_min": float(endpoint_min[pred_idx].item()),
            "assigned_target_idx": int(assigned_target_for_pred[pred_idx].item()),
            "is_matched_to_long_edge": bool(pred_idx == matched_pred_idx),
        })
    return rows, matched_pred_idx, matched_rank


def _plot_curve(ax, curves, color, linewidth=2.0, alpha=1.0):
    if curves.numel() == 0:
        return
    samples = sample_bezier_curves_torch(curves, num_samples=64)
    for sample in samples:
        xy = sample.detach().cpu()
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=linewidth, alpha=alpha)


def _visualize_topk(config, batch, outputs, batch_idx, tgt_idx, rows, matched_pred_idx, matched_rank):
    image = batch["images"][batch_idx].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    tgt_curve = batch["targets"][batch_idx]["curves"][tgt_idx:tgt_idx + 1]
    pred_curves_external = curve_internal_to_external(outputs["pred_curves"][batch_idx], config)

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    axes = axes.reshape(-1)
    top_rows = rows[:TOPK]

    overview = axes[0]
    overview.imshow(image)
    _plot_curve(overview, tgt_curve, color="red", linewidth=3.0)
    for row in top_rows:
        pred_idx = row["pred_idx"]
        color = "lime" if pred_idx == matched_pred_idx else "deepskyblue"
        _plot_curve(overview, pred_curves_external[pred_idx:pred_idx + 1], color=color, linewidth=1.8, alpha=0.9)
    overview.set_title(
        f"GT long edge vs top-{TOPK} competitors\nmatched pred={matched_pred_idx}, cost-rank={matched_rank + 1}",
        fontsize=11,
    )
    overview.axis("off")

    for ax, row in zip(axes[1:], top_rows):
        pred_idx = row["pred_idx"]
        ax.imshow(image)
        _plot_curve(ax, tgt_curve, color="red", linewidth=3.0)
        color = "lime" if pred_idx == matched_pred_idx else "deepskyblue"
        _plot_curve(ax, pred_curves_external[pred_idx:pred_idx + 1], color=color, linewidth=2.2)
        title = (
            f"rank {row['rank']} pred {pred_idx}\n"
            f"cost {row['total_cost']:.3f}, endpoint {row['endpoint_l1_min']:.3f}\n"
            f"score {row['score']:.3f}, assigned_tgt {row['assigned_target_idx']}"
        )
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    for ax in axes[1 + len(top_rows):]:
        ax.axis("off")

    fig.tight_layout()
    out_path = OUT_DIR / "183066_ann2_tgt0_topk_competitors.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _synthetic_endpoint_competition(config):
    device = torch.device("cpu")
    target = torch.tensor(
        [[[0.18, 0.28], [0.36, 0.28], [0.54, 0.28], [0.72, 0.28], [0.88, 0.28], [1.00, 0.28]]],
        dtype=torch.float32,
        device=device,
    )
    pred_full = torch.tensor(
        [[[0.18, 0.28], [0.36, 0.28], [0.54, 0.28], [0.72, 0.28], [0.88, 0.28], [0.995, 0.28]]],
        dtype=torch.float32,
        device=device,
    )
    pred_short = torch.tensor(
        [[[0.18, 0.28], [0.34, 0.28], [0.50, 0.28], [0.66, 0.28], [0.78, 0.28], [0.88, 0.28]]],
        dtype=torch.float32,
        device=device,
    )
    preds = torch.cat([pred_full, pred_short], dim=0)
    tgt_internal = curve_external_to_internal(target, config)
    pred_internal = curve_external_to_internal(preds, config)

    ctrl_forward, ctrl_reverse = pairwise_curve_l1_forward_reverse_cost(pred_internal, tgt_internal)
    sample_forward, sample_reverse = pairwise_sample_l1_forward_reverse_cost(
        pred_internal, tgt_internal, num_samples=int(config["loss"].get("num_curve_samples", 24))
    )
    orientation_cost = torch.minimum(
        float(config["loss"].get("control_cost", 1.5)) * ctrl_forward.squeeze(1)
        + float(config["loss"].get("sample_cost", 2.0)) * sample_forward.squeeze(1),
        float(config["loss"].get("control_cost", 1.5)) * ctrl_reverse.squeeze(1)
        + float(config["loss"].get("sample_cost", 2.0)) * sample_reverse.squeeze(1),
    )
    curve_dist = (
        float(config["loss"].get("curve_distance_cost", 0.0))
        * pairwise_curve_chamfer_cost(
            pred_internal, tgt_internal, point_count=int(config["loss"].get("curve_match_point_count", 4))
        ).squeeze(1)
    )
    endpoint = aligned_endpoint_l1(pred_internal, tgt_internal.expand_as(pred_internal))
    return {
        "full_reach": {
            "orientation_cost": float(orientation_cost[0].item()),
            "curve_distance_cost": float(curve_dist[0].item()),
            "endpoint_l1_internal": float(endpoint[0].item()),
            "total_cost": float((orientation_cost[0] + curve_dist[0]).item()),
        },
        "short_truncated": {
            "orientation_cost": float(orientation_cost[1].item()),
            "curve_distance_cost": float(curve_dist[1].item()),
            "endpoint_l1_internal": float(endpoint[1].item()),
            "total_cost": float((orientation_cost[1] + curve_dist[1]).item()),
        },
    }


def main():
    config, batch, outputs = _load_model_and_batch()
    sample_ids = [target["sample_id"] for target in batch["targets"]]
    batch_idx = sample_ids.index(TARGET_SAMPLE_ID)
    rows, matched_pred_idx, matched_rank = _compute_cost_breakdown(config, batch, outputs, batch_idx, TARGET_TGT_IDX)
    viz_path = _visualize_topk(config, batch, outputs, batch_idx, TARGET_TGT_IDX, rows, matched_pred_idx, matched_rank)
    synthetic = _synthetic_endpoint_competition(config)

    summary = {
        "sample_id": TARGET_SAMPLE_ID,
        "tgt_idx": TARGET_TGT_IDX,
        "matched_pred_idx": matched_pred_idx,
        "matched_cost_rank": matched_rank + 1,
        "topk": rows[:TOPK],
        "synthetic_endpoint_competition": synthetic,
        "viz_path": str(viz_path),
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(OUT_DIR / "topk_rows.json", "w", encoding="utf-8") as handle:
        json.dump(rows[:32], handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
