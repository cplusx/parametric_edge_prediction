from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from edge_datasets.laion_synthetic_dataset import _read_laion_entry_cache
from misc_utils.bezier_target_utils import load_binary_edge_annotation, load_cached_targets, load_image_array_original, sample_bezier_numpy, target_cache_path
from misc_utils.visualization_utils import PALETTE


def _find_records(data_root: Path, sample_specs: Sequence[str]) -> List[Dict[str, Path]]:
    records = _read_laion_entry_cache(data_root / 'laion_entry_cache.txt')
    wanted = {}
    for spec in sample_specs:
        batch, image_id = spec.split(':', 1)
        wanted[(batch, image_id)] = spec
    found: Dict[str, Dict[str, Path]] = {}
    for record in records:
        key = (str(record['batch_name']), str(record['image_id']))
        if key in wanted:
            found[wanted[key]] = record
    missing = [spec for spec in sample_specs if spec not in found]
    if missing:
        raise FileNotFoundError(f'Missing records for specs: {missing}')
    return [found[spec] for spec in sample_specs]


def _cache_path(edge_path: Path, cache_root: Path, version_name: str, target_degree: int, min_curve_length: float) -> Path:
    return target_cache_path(edge_path=edge_path, cache_root=cache_root)


def _draw_curve(ax, control_points: np.ndarray, color: str) -> None:
    pts = np.asarray(control_points, dtype=np.float32)
    samples = sample_bezier_numpy(pts, num_samples=96)
    ax.plot(samples[:, 0], samples[:, 1], color=color, linewidth=2.0, alpha=0.95)
    ax.scatter(pts[:, 0], pts[:, 1], s=12, c='white', edgecolors=color, linewidths=0.9, zorder=4)
    ax.scatter([pts[0, 0], pts[-1, 0]], [pts[0, 1], pts[-1, 1]], s=30, c='#ff3b30', edgecolors='#ffd60a', linewidths=1.0, zorder=5)


def _render_record(record: Dict[str, Path], cache_path: Path, output_path: Path) -> Dict[str, object]:
    image = load_image_array_original(Path(record['image_path']), rgb=True)
    edge = load_binary_edge_annotation(Path(record['edge_path']))
    cache = load_cached_targets(cache_path)
    curves = np.asarray(cache['curves'], dtype=np.float32)
    image_h, image_w = image.shape[:2]
    curves_px = curves.copy()
    curves_px[..., 0] *= float(image_w)
    curves_px[..., 1] *= float(image_h)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    panels = [
        ('Input', image),
        ('Edge', edge),
        ('Curve Overlay', image),
        ('Curve Only', np.zeros((image_h, image_w, 3), dtype=np.float32)),
    ]
    for ax, (title, panel) in zip(axes, panels):
        if panel.ndim == 2:
            ax.imshow(panel, cmap='gray', vmin=0.0, vmax=255.0)
        else:
            panel_f = panel.astype(np.float32)
            if panel_f.max() > 1.0:
                panel_f = panel_f / 255.0
            ax.imshow(panel_f, vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.axis('off')
        ax.set_xlim(0, image_w)
        ax.set_ylim(image_h, 0)
        ax.set_aspect('equal', adjustable='box')

    for idx, curve in enumerate(curves_px):
        color = PALETTE[idx % len(PALETTE)]
        _draw_curve(axes[2], curve, color)
        _draw_curve(axes[3], curve, color)

    fig.suptitle(f"{record['batch_name']}_{record['image_id']}\ncache={cache_path.name}", fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return {
        'sample_id': f"{record['batch_name']}_{record['image_id']}",
        'image_path': str(record['image_path']),
        'edge_path': str(record['edge_path']),
        'curve_cache_path': str(cache_path),
        'cache_exists': bool(cache_path.exists()),
        'num_curves': int(curves.shape[0]),
        'image_size': [int(image_h), int(image_w)],
        'output_path': str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sample', action='append', required=True, help='batch:image_id')
    parser.add_argument('--version-name', type=str, default='v5_anchor_consistent')
    parser.add_argument('--target-degree', type=int, default=5)
    parser.add_argument('--min-curve-length', type=float, default=3.0)
    args = parser.parse_args()

    records = _find_records(args.data_root, args.sample)
    summary = []
    vis_dir = args.output_dir / 'visualizations'
    for record in records:
        cache_path = _cache_path(Path(record['edge_path']), args.cache_root, args.version_name, args.target_degree, args.min_curve_length)
        if not cache_path.exists():
            summary.append({
                'sample_id': f"{record['batch_name']}_{record['image_id']}",
                'image_path': str(record['image_path']),
                'edge_path': str(record['edge_path']),
                'curve_cache_path': str(cache_path),
                'cache_exists': False,
            })
            continue
        output_path = vis_dir / f"{record['batch_name']}_{record['image_id']}_curve_cache.png"
        summary.append(_render_record(record, cache_path, output_path))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'summary.json').write_text(json.dumps({'records': summary}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
