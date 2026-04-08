import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from edge_datasets.endpoint_flow_overfit_dataset import SingleLaionEndpointFlowOverfitDataset


def _match(uniform_points_norm: np.ndarray, gt_points_norm: np.ndarray, p: int):
    src = torch.from_numpy(uniform_points_norm).float()
    tgt = torch.from_numpy(gt_points_norm).float()
    cost = torch.cdist(src, tgt, p=p)
    row_ind, col_ind = linear_sum_assignment(cost.numpy())
    matched_cost = cost[row_ind, col_ind].numpy()
    return row_ind, col_ind, matched_cost, cost.numpy()


def _render(image_hwc: np.ndarray, gt_points_norm: np.ndarray, uniform_points_norm: np.ndarray, row_ind, col_ind, title: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image_hwc, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis('off')
    h, w = image_hwc.shape[:2]
    gt_points = gt_points_norm.copy()
    gt_points[:, 0] *= float(w - 1)
    gt_points[:, 1] *= float(h - 1)
    uniform_points = uniform_points_norm.copy()
    uniform_points[:, 0] *= float(w - 1)
    uniform_points[:, 1] *= float(h - 1)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect('equal', adjustable='box')

    ax.scatter(uniform_points[:, 0], uniform_points[:, 1], s=18, c='#4cc9f0', edgecolors='black', linewidths=0.25, label='uniform')
    ax.scatter(gt_points[:, 0], gt_points[:, 1], s=22, c='#ffd166', edgecolors='#d00000', linewidths=0.6, label='gt')

    for src_idx, tgt_idx in zip(row_ind.tolist(), col_ind.tolist()):
        src = uniform_points[src_idx]
        tgt = gt_points[tgt_idx]
        ax.plot([src[0], tgt[0]], [src[1], tgt[1]], color='white', alpha=0.7, linewidth=0.9)

    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Debug Hungarian matching between uniform points and GT endpoints.')
    parser.add_argument('--image-path', required=True)
    parser.add_argument('--edge-path', required=True)
    parser.add_argument('--bezier-cache-path', required=True)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    dataset = SingleLaionEndpointFlowOverfitDataset(
        image_path=Path(args.image_path),
        edge_path=Path(args.edge_path),
        bezier_cache_path=Path(args.bezier_cache_path),
        image_size=int(args.image_size),
        repeats=1,
    )
    sample = dataset[0]
    image = np.transpose(sample['image'].numpy(), (1, 2, 0)).astype(np.float32)
    gt_points_norm = sample['target']['points'].numpy().astype(np.float32)
    if gt_points_norm.shape[0] == 0:
        raise ValueError('No GT endpoints in sample.')

    rng = np.random.default_rng(int(args.seed))
    uniform_points_norm = rng.random((gt_points_norm.shape[0], 2), dtype=np.float32)

    l1_row, l1_col, l1_cost, _ = _match(uniform_points_norm, gt_points_norm, p=1)
    l2_row, l2_col, l2_cost, _ = _match(uniform_points_norm, gt_points_norm, p=2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _render(
        image,
        gt_points_norm,
        uniform_points_norm,
        l1_row,
        l1_col,
        f'L1 Hungarian | points={gt_points_norm.shape[0]} | mean={float(l1_cost.mean()):.4f}',
        output_dir / 'uniform_match_l1.png',
    )
    _render(
        image,
        gt_points_norm,
        uniform_points_norm,
        l2_row,
        l2_col,
        f'L2 Hungarian | points={gt_points_norm.shape[0]} | mean={float(l2_cost.mean()):.4f}',
        output_dir / 'uniform_match_l2.png',
    )

    summary = {
        'num_points': int(gt_points_norm.shape[0]),
        'seed': int(args.seed),
        'l1': {
            'mean_cost': float(l1_cost.mean()),
            'median_cost': float(np.median(l1_cost)),
            'max_cost': float(l1_cost.max()),
        },
        'l2': {
            'mean_cost': float(l2_cost.mean()),
            'median_cost': float(np.median(l2_cost)),
            'max_cost': float(l2_cost.max()),
        },
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
