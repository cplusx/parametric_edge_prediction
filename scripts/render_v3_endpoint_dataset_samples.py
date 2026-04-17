from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge_datasets.endpoint_datamodule import EndpointDetectionDataModule
from misc_utils.visualization_utils import PALETTE, draw_curve

FIXTURE_ROOT = REPO_ROOT / 'outputs' / 'v3_dataloader_profile_fixture'
SAMPLE_LIMIT = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['eval', 'train_aug'], default='eval')
    return parser.parse_args()


def _build_local_config(mode: str) -> dict:
    dataset_cfg = {
        'dataset_type': 'laion_synthetic',
        'data_root': str(FIXTURE_ROOT),
        'image_root': str(FIXTURE_ROOT / 'edge_detection'),
        'bezier_root': str(FIXTURE_ROOT / 'laion_edge_v3_bezier'),
        'entry_cache_path': str(FIXTURE_ROOT / 'laion_entry_cache_v3_bezier.txt'),
        'batch_glob': 'batch*',
        'selection_seed': 20260318,
        'selection_offset': 0,
        'max_samples': SAMPLE_LIMIT,
    }
    return {
        'data': {
            'include_primary_train_dataset': False,
            'rgb_input': True,
            'image_size': 256,
            'batch_size': SAMPLE_LIMIT,
            'val_batch_size': SAMPLE_LIMIT,
            'num_workers': 0,
            'pin_memory': False,
            'persistent_workers': False,
            'target_degree': 5,
            'min_curve_length': 3.0,
            'max_targets': 512,
            'endpoint_dedupe_distance_px': 2.0,
            'train_augment': mode == 'train_aug',
            'augment': {
                'resize_scale_range': [1.0, 1.5],
                'affine_scale_range': [0.9, 1.1],
                'affine_max_rotate_deg': 10.0,
                'affine_max_translate_ratio': 0.06,
                'hflip_prob': 0.5,
                'vflip_prob': 0.0,
            },
            'extra_train_datasets': [dataset_cfg],
            'val_dataset': dataset_cfg,
            'test_dataset': dataset_cfg,
        },
        'model': {
            'arch': 'dab_endpoint_detr',
        },
        'trainer': {},
    }


def _to_image_hwc(image_chw: np.ndarray) -> np.ndarray:
    if image_chw.shape[0] == 1:
        return np.repeat(image_chw[0][..., None], 3, axis=2)
    return np.transpose(image_chw, (1, 2, 0))


def _draw_point_overlay(ax, image_np: np.ndarray, points: np.ndarray) -> None:
    ax.imshow(image_np, vmin=0.0, vmax=1.0)
    h, w = image_np.shape[:2]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    if points.size == 0:
        return
    pts = points.astype(np.float32).copy()
    pts[:, 0] *= float(w)
    pts[:, 1] *= float(h)
    for idx, point in enumerate(pts):
        ax.scatter(point[0], point[1], s=20.0, c=PALETTE[idx % len(PALETTE)], edgecolors='none')


def _draw_curve_overlay(ax, image_np: np.ndarray, curves: np.ndarray) -> None:
    ax.imshow(image_np, vmin=0.0, vmax=1.0)
    h, w = image_np.shape[:2]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    for idx, curve in enumerate(curves):
        draw_curve(ax, curve, w, h, PALETTE[idx % len(PALETTE)], linewidth=2.0, show_control_points=True)


def main() -> None:
    args = parse_args()
    output_dir = REPO_ROOT / 'outputs' / f'v3_endpoint_dataset_debug_{args.mode}'
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _build_local_config(args.mode)
    dm = EndpointDetectionDataModule(config)
    dm.setup('fit')
    loader = dm.train_dataloader() if args.mode == 'train_aug' else dm.val_dataloader()
    batch = next(iter(loader))
    images = batch['images'].detach().cpu().numpy()
    targets = batch['targets']

    sample_count = min(SAMPLE_LIMIT, len(targets))
    point_fig, point_axes = plt.subplots(sample_count, 1, figsize=(6, 4 * sample_count))
    curve_fig, curve_axes = plt.subplots(sample_count, 1, figsize=(6, 4 * sample_count))
    if sample_count == 1:
        point_axes = np.asarray([point_axes])
        curve_axes = np.asarray([curve_axes])

    summaries = []
    for idx in range(sample_count):
        image_np = _to_image_hwc(images[idx])
        target = targets[idx]
        points = target['points'].detach().cpu().numpy()
        curves = target['curves'].detach().cpu().numpy()
        bezier_path = Path(target['bezier_path'])
        input_path = Path(target['input_path'])

        _draw_point_overlay(point_axes[idx], image_np, points)
        _draw_curve_overlay(curve_axes[idx], image_np, curves)
        sample_id = str(target['sample_id'])
        point_axes[idx].set_title(f'{sample_id} | endpoints={points.shape[0]}')
        curve_axes[idx].set_title(f'{sample_id} | curves={curves.shape[0]}')
        summaries.append(
            {
                'sample_id': sample_id,
                'input_path': str(input_path),
                'bezier_path': str(bezier_path),
                'num_points': int(points.shape[0]),
                'num_curves': int(curves.shape[0]),
                'image_size': target['image_size'].detach().cpu().tolist(),
                'target_keys': sorted(target.keys()),
            }
        )

    point_fig.tight_layout()
    point_fig.savefig(output_dir / 'endpoint_points_overlay.png', dpi=180)
    plt.close(point_fig)

    curve_fig.tight_layout()
    curve_fig.savefig(output_dir / 'endpoint_curves_overlay.png', dpi=180)
    plt.close(curve_fig)

    with (output_dir / 'summary.json').open('w', encoding='utf-8') as handle:
        json.dump(
            {
                'mode': args.mode,
                'sample_count': sample_count,
                'samples': summaries,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(json.dumps({'output_dir': str(output_dir), 'sample_count': sample_count, 'mode': args.mode}, ensure_ascii=False))


if __name__ == '__main__':
    main()
