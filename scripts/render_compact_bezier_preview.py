from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bezierization.bezier_refiner_core import render_piecewise_fits
from sam_mask_bezierization.pipeline import (
    draw_colored_curves_on_image,
    draw_endpoints_control_points_on_image,
    raster_to_rgb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bezier-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--source-edge-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--samples-per-sheet", type=int, default=5)
    return parser.parse_args()


def _load_compact_paths(npz_path: Path) -> list[dict]:
    data = np.load(npz_path)
    control_points = np.asarray(data["control_points"], dtype=np.float32)
    segment_degrees = np.asarray(data["segment_degrees"], dtype=np.uint8)
    segment_ctrl_offsets = np.asarray(data["segment_ctrl_offsets"], dtype=np.int32)
    path_segment_offsets = np.asarray(data["path_segment_offsets"], dtype=np.int32)

    paths: list[dict] = []
    for path_idx in range(len(path_segment_offsets) - 1):
        seg_start = int(path_segment_offsets[path_idx])
        seg_end = int(path_segment_offsets[path_idx + 1])
        segments: list[dict] = []
        for seg_idx in range(seg_start, seg_end):
            c0 = int(segment_ctrl_offsets[seg_idx])
            c1 = int(segment_ctrl_offsets[seg_idx + 1])
            ctrl = control_points[c0:c1]
            segments.append(
                {
                    "degree": int(segment_degrees[seg_idx]),
                    "control_points": np.asarray(ctrl, dtype=np.float64),
                    "points": np.asarray(ctrl, dtype=np.float64).copy(),
                    "fitted_points": np.asarray(ctrl, dtype=np.float64).copy(),
                }
            )
        paths.append({"segments": segments, "original_points": np.zeros((0, 2), dtype=np.float64)})
    return paths


def _save_sheet(records: list[dict], sheet_path: Path, title: str) -> None:
    rows = len(records)
    fig, axes = plt.subplots(rows, 5, figsize=(22, 4.1 * rows))
    if rows == 1:
        axes = np.asarray([axes])
    headers = ["Input", "Source edge", "Colored curves", "Endpoints + control points", "Binarized rasterized"]
    for col, header in enumerate(headers):
        axes[0, col].set_title(header, fontsize=12)

    for row, rec in enumerate(records):
        sid = rec["sample_id"]
        subtitle = f"{sid} | paths={rec['num_paths']} segs={rec['num_segments']}"
        panels = [
            rec["input"],
            rec["source_edge"],
            rec["colored_curves"],
            rec["endpoints_control_points"],
            rec["binarized_rasterized"],
        ]
        for col, panel in enumerate(panels):
            axes[row, col].imshow(panel, cmap="gray" if panel.ndim == 2 else None)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(subtitle, rotation=0, ha="right", va="center", labelpad=56, fontsize=9)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(sheet_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(args.bezier_dir.glob("*.npz"), key=lambda p: p.stem)[: args.limit]
    rendered: list[dict] = []

    for npz_path in npz_files:
        sample_id = npz_path.stem
        image_path = args.image_dir / f"{sample_id}.jpg"
        if not image_path.exists():
            image_path = args.image_dir / f"{sample_id}.png"
        source_edge_path = args.source_edge_dir / f"{sample_id}.png"
        if not image_path.exists() or not source_edge_path.exists():
            continue

        image_np = np.asarray(Image.open(image_path).convert("RGB"))
        source_edge = np.asarray(Image.open(source_edge_path).convert("L")) > 127
        paths = _load_compact_paths(npz_path)
        final_raster, _ = render_piecewise_fits(source_edge.shape, paths)

        rendered.append(
            {
                "sample_id": sample_id,
                "num_paths": len(paths),
                "num_segments": int(sum(len(p["segments"]) for p in paths)),
                "input": image_np,
                "source_edge": source_edge.astype(np.uint8) * 255,
                "colored_curves": draw_colored_curves_on_image(image_np, paths),
                "endpoints_control_points": draw_endpoints_control_points_on_image(image_np, paths),
                "binarized_rasterized": raster_to_rgb(final_raster),
            }
        )

    for start in range(0, len(rendered), args.samples_per_sheet):
        chunk = rendered[start:start + args.samples_per_sheet]
        if not chunk:
            continue
        sidx = start + 1
        eidx = start + len(chunk)
        _save_sheet(
            chunk,
            args.output_dir / f"samples_{sidx:02d}_{eidx:02d}.png",
            title=f"Compact bezier preview | samples {sidx}-{eidx}",
        )


if __name__ == "__main__":
    main()
