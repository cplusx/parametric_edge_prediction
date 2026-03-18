import argparse
import importlib.util
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from misc_utils.bezier_target_utils import sample_bezier_numpy


def _load_graph_pipeline():
    path = Path('edge_datasets/graph_pipeline.py').resolve()
    spec = importlib.util.spec_from_file_location('graph_pipeline_local', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_edge(edge_npz: Path) -> np.ndarray:
    obj = np.load(edge_npz, allow_pickle=False)
    mat = sparse.csr_matrix((obj['data'], obj['indices'], obj['indptr']), shape=tuple(obj['shape']))
    return (mat.toarray() > 0).astype(np.uint8)


def _load_polylines(graph_npz: Path) -> List[np.ndarray]:
    obj = np.load(graph_npz, allow_pickle=False)
    pts = np.asarray(obj['graph_points'], dtype=np.float32)
    off = np.asarray(obj['graph_offsets'], dtype=np.int64)
    return [pts[s:e] for s, e in zip(off[:-1], off[1:]) if e - s >= 2]


def _render_strategy_panel(ax, edge, curves, weights, title):
    h, w = edge.shape[:2]
    ax.imshow(edge, cmap='gray')
    ax.set_title(title)
    for idx, curve in enumerate(curves):
        ctrl = np.asarray(curve, dtype=np.float32) * np.array([max(w - 1, 1), max(h - 1, 1)], dtype=np.float32)
        samples_curve = sample_bezier_numpy(ctrl)
        weight = float(weights[idx])
        if weight < 0.95:
            color = plt.colormaps['autumn'](1.0 - weight)
            ax.plot(samples_curve[:, 0], samples_curve[:, 1], color=color, linewidth=2.2)
        else:
            ax.plot(samples_curve[:, 0], samples_curve[:, 1], color='#666666', linewidth=0.7, alpha=0.35)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare clutter-weight heuristics on cached remote samples.')
    parser.add_argument('--sample-root', default='outputs/remote_cache_debug_samples')
    parser.add_argument('--output-dir', default='outputs/remote_cache_debug_samples/clutter_strategy_compare')
    parser.add_argument('--samples', nargs='*', default=[
        'batch1_000000', 'batch1_018857', 'batch2_150890',
        'batch1_024442', 'batch2_125853', 'batch2_126615',
    ])
    parser.add_argument('--target-degree', type=int, default=5)
    parser.add_argument('--min-curve-length', type=float, default=3.0)
    parser.add_argument('--max-targets', type=int, default=2000)
    args = parser.parse_args()

    module = _load_graph_pipeline()
    sample_root = Path(args.sample_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    strategies = ['aggressive', 'overlap_only', 'balanced']
    summary_rows = []

    for sample in args.samples:
        edge_path = sample_root / f'{sample}_edge.npz'
        graph_path = sample_root / f'{sample}_graph.npz'
        if not edge_path.exists() or not graph_path.exists():
            continue

        edge = _load_edge(edge_path)
        polylines = _load_polylines(graph_path)
        targets = module.build_targets_from_polylines(
            polylines,
            image_shape=edge.shape[:2],
            target_degree=int(args.target_degree),
            min_curve_length=float(args.min_curve_length),
            max_targets=int(args.max_targets),
        )
        curves = targets['curves']
        lengths = targets['curve_lengths']
        boxes = targets['curve_boxes']
        curvatures = targets['curve_curvatures']

        fig, axes = plt.subplots(1, 1 + len(strategies), figsize=(5 * (1 + len(strategies)), 4.5))
        axes = np.asarray(axes).reshape(-1)
        axes[0].imshow(edge, cmap='gray')
        axes[0].set_title('Binary Edge')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_aspect('equal', adjustable='box')
        axes[0].set_xlim(0, edge.shape[1])
        axes[0].set_ylim(edge.shape[0], 0)

        for ax, strategy in zip(axes[1:], strategies):
            weights = module._compute_clutter_match_weights(curves, lengths, boxes, curvatures, strategy=strategy)
            low_count = int((weights < 0.95).sum())
            summary_rows.append((sample, strategy, len(weights), low_count, float(weights.min()) if len(weights) else 1.0, float(weights.mean()) if len(weights) else 1.0))
            _render_strategy_panel(ax, edge, curves, weights, f'{strategy} | low={low_count}')

        fig.suptitle(sample)
        fig.tight_layout()
        fig.savefig(output_dir / f'{sample}_compare.png', dpi=160)
        plt.close(fig)

    summary_path = output_dir / 'summary.txt'
    with summary_path.open('w', encoding='utf-8') as f:
        for row in summary_rows:
            f.write(f'{row[0]}\t{row[1]}\tcurves={row[2]}\tlow={row[3]}\tmin={row[4]:.3f}\tmean={row[5]:.3f}\n')


if __name__ == '__main__':
    main()
