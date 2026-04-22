from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from misc_utils.train_utils import sample_bezier_curves_torch  # noqa: E402
from models.curve_query_initializers import build_curve_query_initializer  # noqa: E402


def build_config(init_type: str) -> dict:
    return {
        "model": {
            "curve_query_init_type": init_type,
        }
    }


def draw_initializer(ax, curves: torch.Tensor, *, title: str, sample_limit: int = 25) -> None:
    curves = curves[:sample_limit].detach().cpu()
    rendered = sample_bezier_curves_torch(curves, num_samples=60).numpy()
    cps = curves.numpy()
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(sample_limit, 1)))

    for idx in range(curves.shape[0]):
        color = colors[idx % len(colors)]
        ax.plot(rendered[idx, :, 0], rendered[idx, :, 1], color=color, linewidth=1.8, alpha=0.9)
        ax.plot(cps[idx, :, 0], cps[idx, :, 1], color=color, linewidth=0.8, linestyle=":", alpha=0.8)
        ax.scatter(cps[idx, :, 0], cps[idx, :, 1], color=color, s=12)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_title(title)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize curve query initializers.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "curve_query_initializers")
    parser.add_argument("--num-queries", type=int, default=64)
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--sample-limit", type=int, default=25)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    num_control_points = args.degree + 1
    device = torch.device("cpu")
    dtype = torch.float32

    init_specs = [
        ("random", "Random Curves"),
        ("line", "Line Interpolation"),
    ]

    records = []
    fig, axes = plt.subplots(1, len(init_specs), figsize=(7 * len(init_specs), 7), constrained_layout=True)
    if len(init_specs) == 1:
        axes = [axes]

    for ax, (init_type, title) in zip(axes, init_specs):
        initializer = build_curve_query_initializer(build_config(init_type))
        curves = initializer.initialize(
            num_queries=args.num_queries,
            num_control_points=num_control_points,
            device=device,
            dtype=dtype,
        )
        draw_initializer(ax, curves, title=title, sample_limit=args.sample_limit)
        np.save(args.output_dir / f"{init_type}_init_curves.npy", curves.numpy())
        records.append(
            {
                "init_type": init_type,
                "title": title,
                "num_queries": int(args.num_queries),
                "num_control_points": int(num_control_points),
                "example_curve_0": curves[0].tolist(),
            }
        )

    figure_path = args.output_dir / "curve_query_initializers.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "figure_path": str(figure_path),
                "records": records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(json.dumps({"figure_path": str(figure_path), "summary_path": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
