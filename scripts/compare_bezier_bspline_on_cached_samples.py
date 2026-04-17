from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.segmentation import find_boundaries

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam_mask_bezierization.bspline import (
    draw_bspline_control_points_on_image,
    draw_colored_bspline_curves_on_image,
    run_bspline_from_source_edge,
)
from sam_mask_bezierization.mask_to_edge_methods import masks_to_label_image
from sam_mask_bezierization.pipeline import (
    detect_small_bubbles,
    draw_colored_curves_on_image,
    draw_endpoints_control_points_on_image,
    raster_to_rgb,
    repair_global_band_thin,
    run_bezier_from_source_edge,
    prune_tiny_edge_cc,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("experiments/sam2_binarization_rebuild/cache/final_strategy_30_samples_images"),
    )
    parser.add_argument(
        "--mask-cache-dir",
        type=Path,
        default=Path("experiments/sam2_binarization_rebuild/cache/final_strategy_30_samples_masks"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bspline_vs_bezier_cached_20"),
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--samples-per-sheet", type=int, default=5)
    parser.add_argument("--bubble-area-ratio", type=float, default=0.0008)
    parser.add_argument("--tiny-cc-limit", type=int, default=15)
    return parser.parse_args()


def _load_kept_masks(mask_dir: Path) -> list[dict]:
    data = np.load(mask_dir / "kept_masks.npz")
    segs = np.asarray(data["segmentation"], dtype=bool)
    areas = np.asarray(data["area"], dtype=np.int32)
    masks: list[dict] = []
    for seg, area in zip(segs, areas):
        masks.append({"segmentation": seg.astype(bool), "area": int(area)})
    return masks


def _build_source_edge_from_kept_masks(
    kept_masks: list[dict],
    bubble_area_ratio: float,
    tiny_cc_limit: int,
) -> np.ndarray:
    label_image = masks_to_label_image(kept_masks)
    raw_edge = find_boundaries(label_image, mode="inner", background=0).astype(bool)
    bubble_limit = max(1, int(raw_edge.shape[0] * raw_edge.shape[1] * bubble_area_ratio))
    selected_bubbles = detect_small_bubbles(raw_edge, max_bubble_area=bubble_limit)
    source_edge = repair_global_band_thin(raw_edge, selected_bubbles, band_radius=1)
    source_edge, _, _ = prune_tiny_edge_cc(source_edge, pixel_limit=tiny_cc_limit)
    return source_edge


def _row_label(record: dict) -> str:
    return (
        f"{record['sample_id']}\n"
        f"Bz {record['bezier_paths']}/{record['bezier_segments']} | "
        f"BS2 {record['bspline2_paths']}/{record['bspline2_segments']} | "
        f"BS3 {record['bspline3_paths']}/{record['bspline3_segments']}"
    )


def _save_sheet(records: list[dict], path: Path, title: str, panel_key: str) -> None:
    rows = len(records)
    fig, axes = plt.subplots(rows, 5, figsize=(24, 4.2 * rows))
    if rows == 1:
        axes = np.asarray([axes])
    headers = ["Input", "Source edge", "Bezier", "B-spline mid=2", "B-spline mid=3"]
    for col, header in enumerate(headers):
        axes[0, col].set_title(header, fontsize=12)
    for row, record in enumerate(records):
        panels = [
            record["input"],
            record["source_edge_rgb"],
            record[panel_key]["bezier"],
            record[panel_key]["bspline2"],
            record[panel_key]["bspline3"],
        ]
        for col, panel in enumerate(panels):
            axes[row, col].imshow(panel)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(_row_label(record), rotation=0, ha="right", va="center", labelpad=66, fontsize=8.5)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(args.image_dir.glob("*.jpg"), key=lambda p: p.stem)[: args.limit]
    records: list[dict] = []

    for image_path in image_files:
        sample_id = image_path.stem
        mask_dir = args.mask_cache_dir / sample_id
        if not mask_dir.exists():
            continue

        image_np = np.asarray(Image.open(image_path).convert("RGB"))
        kept_masks = _load_kept_masks(mask_dir)
        source_edge = _build_source_edge_from_kept_masks(
            kept_masks,
            bubble_area_ratio=args.bubble_area_ratio,
            tiny_cc_limit=args.tiny_cc_limit,
        )

        bezier_result = run_bezier_from_source_edge(source_edge, compute_final_raster=True)
        bspline2_result = run_bspline_from_source_edge(source_edge, num_middle_ctrl_points=2, compute_final_raster=True)
        bspline3_result = run_bspline_from_source_edge(source_edge, num_middle_ctrl_points=3, compute_final_raster=True)

        records.append(
            {
                "sample_id": sample_id,
                "input": image_np,
                "source_edge_rgb": raster_to_rgb(source_edge),
                "overlay": {
                    "bezier": draw_colored_curves_on_image(image_np, bezier_result["final_paths"]),
                    "bspline2": draw_colored_bspline_curves_on_image(image_np, bspline2_result["final_paths"]),
                    "bspline3": draw_colored_bspline_curves_on_image(image_np, bspline3_result["final_paths"]),
                },
                "controls": {
                    "bezier": draw_endpoints_control_points_on_image(image_np, bezier_result["final_paths"]),
                    "bspline2": draw_bspline_control_points_on_image(image_np, bspline2_result["final_paths"]),
                    "bspline3": draw_bspline_control_points_on_image(image_np, bspline3_result["final_paths"]),
                },
                "bezier_paths": int(bezier_result["summary"]["final_path_count"]),
                "bezier_segments": int(bezier_result["summary"]["final_segment_count"]),
                "bspline2_paths": int(bspline2_result["summary"]["path_count"]),
                "bspline2_segments": int(bspline2_result["summary"]["segment_count"]),
                "bspline3_paths": int(bspline3_result["summary"]["path_count"]),
                "bspline3_segments": int(bspline3_result["summary"]["segment_count"]),
            }
        )

    for start in range(0, len(records), args.samples_per_sheet):
        chunk = records[start:start + args.samples_per_sheet]
        if not chunk:
            continue
        sidx = start + 1
        eidx = start + len(chunk)
        _save_sheet(
            chunk,
            args.output_dir / f"overlay_samples_{sidx:02d}_{eidx:02d}.png",
            title=f"Bezier vs B-spline overlays | samples {sidx}-{eidx}",
            panel_key="overlay",
        )
        _save_sheet(
            chunk,
            args.output_dir / f"controls_samples_{sidx:02d}_{eidx:02d}.png",
            title=f"Bezier vs B-spline control points | samples {sidx}-{eidx}",
            panel_key="controls",
        )

    summary = {
        "num_samples": len(records),
        "sample_ids": [record["sample_id"] for record in records],
        "samples": [
            {
                "sample_id": record["sample_id"],
                "bezier_paths": record["bezier_paths"],
                "bezier_segments": record["bezier_segments"],
                "bspline2_paths": record["bspline2_paths"],
                "bspline2_segments": record["bspline2_segments"],
                "bspline3_paths": record["bspline3_paths"],
                "bspline3_segments": record["bspline3_segments"],
            }
            for record in records
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
