import argparse
import importlib.util
import copy
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from misc_utils.train_utils import sample_bezier_curves_torch


def load_geometry_helpers():
    repo_root = Path(__file__).resolve().parents[1]
    geometry_path = repo_root / 'models' / 'geometry.py'
    spec = importlib.util.spec_from_file_location('synthetic_geometry', geometry_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load geometry helpers from {geometry_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEOMETRY = load_geometry_helpers()
curve_boxes_xyxy = GEOMETRY.curve_boxes_xyxy
pairwise_curve_chamfer_cost = GEOMETRY.pairwise_curve_chamfer_cost
pairwise_generalized_box_iou = GEOMETRY.pairwise_generalized_box_iou
symmetric_curve_distance = GEOMETRY.symmetric_curve_distance


def load_config(config_path: Path) -> Dict:
    with config_path.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def build_canonical_curve(num_control_points: int) -> np.ndarray:
    x_values = np.linspace(-1.0, 1.0, num_control_points)
    y_values = (
        0.38 * np.sin(1.35 * math.pi * (x_values + 1.0) * 0.5)
        - 0.16 * x_values
        + 0.06 * np.cos(2.4 * math.pi * x_values)
    )
    return np.stack([x_values, y_values], axis=1)


def transform_curve(
    canonical_curve: np.ndarray,
    center: np.ndarray,
    scale_xy: Tuple[float, float],
    rotation_rad: float,
    noise_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    scale = np.array(scale_xy, dtype=np.float32)
    local = canonical_curve * scale[None, :]
    noise = rng.normal(loc=0.0, scale=noise_scale, size=local.shape).astype(np.float32)
    local = local + noise
    cos_theta = math.cos(rotation_rad)
    sin_theta = math.sin(rotation_rad)
    rotation = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=np.float32)
    curve = local @ rotation.T + center[None, :]
    return np.clip(curve, 0.02, 0.98)


def build_target_curve(num_control_points: int) -> np.ndarray:
    canonical = build_canonical_curve(num_control_points)
    target = transform_curve(
        canonical_curve=canonical,
        center=np.array([0.5, 0.5], dtype=np.float32),
        scale_xy=(0.18, 0.13),
        rotation_rad=0.2,
        noise_scale=0.0,
        rng=np.random.default_rng(0),
    )
    return target.astype(np.float32)


def build_query_curves(num_queries: int, num_control_points: int, rng: np.random.Generator) -> np.ndarray:
    canonical = build_canonical_curve(num_control_points)
    grid_side = int(math.ceil(math.sqrt(num_queries + 1)))
    xs = np.linspace(0.14, 0.86, grid_side)
    ys = np.linspace(0.14, 0.86, grid_side)
    centers = []
    for y_value in ys:
        for x_value in xs:
            if abs(x_value - 0.5) < 0.12 and abs(y_value - 0.5) < 0.12:
                continue
            centers.append(np.array([x_value, y_value], dtype=np.float32))
    rng.shuffle(centers)
    curves = []
    for query_index in range(num_queries):
        center = centers[query_index % len(centers)].copy()
        center += rng.uniform(-0.035, 0.035, size=2).astype(np.float32)
        center = np.clip(center, 0.08, 0.92)
        scale_x = float(rng.uniform(0.08, 0.24))
        scale_y = float(rng.uniform(0.05, 0.18))
        rotation = float(rng.uniform(-math.pi, math.pi))
        noise_scale = float(rng.uniform(0.005, 0.03))
        if query_index < max(4, num_queries // 6):
            center = np.array([0.5, 0.5], dtype=np.float32) + rng.uniform(-0.08, 0.08, size=2).astype(np.float32)
            scale_x = float(rng.uniform(0.14, 0.22))
            scale_y = float(rng.uniform(0.10, 0.15))
            rotation = float(rng.uniform(-0.45, 0.45))
            noise_scale = float(rng.uniform(0.002, 0.018))
        curve = transform_curve(
            canonical_curve=canonical,
            center=center,
            scale_xy=(scale_x, scale_y),
            rotation_rad=rotation,
            noise_scale=noise_scale,
            rng=rng,
        )
        curves.append(curve)
    return np.stack(curves, axis=0).astype(np.float32)


def make_logits(probabilities: np.ndarray) -> torch.Tensor:
    probabilities = np.clip(probabilities, 1e-4, 1.0 - 1e-4)
    logits = np.log(probabilities / (1.0 - probabilities))
    return torch.from_numpy(np.stack([logits, np.zeros_like(logits)], axis=-1).astype(np.float32))


def build_probabilities(mode: str, num_queries: int, rng: np.random.Generator) -> np.ndarray:
    if mode == 'constant':
        return np.full((num_queries,), 0.75, dtype=np.float32)
    if mode == 'random':
        return rng.uniform(0.1, 0.95, size=(num_queries,)).astype(np.float32)
    raise ValueError(f'Unsupported probability mode: {mode}')


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind='mergesort')
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and np.isclose(sorted_values[end], sorted_values[start], rtol=1e-8, atol=1e-10):
            end += 1
        average_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def spearman_corr(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if len(values_a) < 2:
        return float('nan')
    rank_a = rankdata(values_a)
    rank_b = rankdata(values_b)
    std_a = rank_a.std()
    std_b = rank_b.std()
    if std_a < 1e-8 or std_b < 1e-8:
        return float('nan')
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def compute_cost_breakdown(config: Dict, target_curve: torch.Tensor, query_curves: torch.Tensor) -> Dict[str, torch.Tensor]:
    loss_cfg = config['loss']
    target_boxes = curve_boxes_xyxy(target_curve)
    query_boxes = curve_boxes_xyxy(query_curves)
    control_term = float(loss_cfg.get('control_cost', 5.0)) * torch.cdist(
        query_curves.reshape(query_curves.shape[0], -1),
        target_curve.reshape(target_curve.shape[0], -1),
        p=1,
    )[:, 0]
    giou_term = float(loss_cfg.get('giou_cost', 1.0)) * (1.0 - pairwise_generalized_box_iou(query_boxes, target_boxes)[:, 0])
    curve_term = float(loss_cfg.get('curve_distance_cost', 1.0)) * pairwise_curve_chamfer_cost(
        query_curves,
        target_curve,
        point_count=int(loss_cfg.get('curve_match_point_count', 4)),
    )[:, 0]
    total_term = control_term + giou_term + curve_term
    reference_total, reference_chamfer, reference_length = symmetric_curve_distance(
        query_curves,
        target_curve.repeat(query_curves.shape[0], 1, 1),
        num_samples=max(64, int(loss_cfg.get('num_curve_samples', 24))),
        length_weight=float(loss_cfg.get('curve_distance_length_weight', 0.25)),
    )
    return {
        'control_term': control_term,
        'giou_term': giou_term,
        'curve_term': curve_term,
        'total_term': total_term,
        'reference_total': reference_total,
        'reference_chamfer': reference_chamfer,
        'reference_length': reference_length,
    }


def curve_center(curve: np.ndarray) -> np.ndarray:
    samples = sample_bezier_curves_torch(torch.from_numpy(curve[None, ...]), num_samples=32)[0].numpy()
    return samples.mean(axis=0)


def format_annotation(record: Dict) -> str:
    return (
        f"q{record['query_index']:02d} r{record['rank_total']:02d}\n"
        f"tot={record['total_term']:.2f}\n"
        f"ctl={record['control_term']:.2f}\n"
        f"giou={record['giou_term']:.2f}\n"
        f"curve={record['curve_term']:.2f} ref={record['reference_total']:.2f}"
    )


def draw_curve(ax, curve: np.ndarray, color: str, linewidth: float, alpha: float = 1.0) -> None:
    samples = sample_bezier_curves_torch(torch.from_numpy(curve[None, ...]), num_samples=80)[0].numpy()
    ax.plot(samples[:, 0], samples[:, 1], color=color, linewidth=linewidth, alpha=alpha)
    ax.scatter(curve[:, 0], curve[:, 1], color=color, s=12, alpha=alpha)


def build_records(costs: Dict[str, torch.Tensor]) -> List[Dict]:
    total_order = torch.argsort(costs['total_term']).tolist()
    reference_order = torch.argsort(costs['reference_total']).tolist()
    total_rank = {query_index: rank + 1 for rank, query_index in enumerate(total_order)}
    reference_rank = {query_index: rank + 1 for rank, query_index in enumerate(reference_order)}
    records = []
    for query_index in range(costs['total_term'].shape[0]):
        record = {'query_index': int(query_index)}
        for key, value in costs.items():
            record[key] = float(value[query_index].item())
        record['rank_total'] = int(total_rank[query_index])
        record['rank_reference'] = int(reference_rank[query_index])
        record['rank_gap'] = int(record['rank_total'] - record['rank_reference'])
        records.append(record)
    records.sort(key=lambda item: item['total_term'])
    return records


def visualize_scenario(
    output_dir: Path,
    scenario_name: str,
    target_curve: np.ndarray,
    query_curves: np.ndarray,
    records: List[Dict],
    weights: Dict[str, float],
) -> None:
    totals = np.array([record['total_term'] for record in records], dtype=np.float32)
    color_norm = plt.Normalize(vmin=float(totals.min()), vmax=float(totals.max()) if totals.max() > totals.min() else float(totals.min()) + 1.0)
    cmap = plt.cm.viridis_r
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect('equal')
    ax.set_facecolor('#0f1115')
    ax.grid(color='#2a2f3a', linestyle='--', linewidth=0.5, alpha=0.5)
    draw_curve(ax, target_curve, color='#ff6b6b', linewidth=4.0)
    ax.text(
        0.5,
        0.57,
        'Target A',
        color='#ffb3b3',
        fontsize=14,
        ha='center',
        va='bottom',
        fontweight='bold',
    )
    query_lookup = {record['query_index']: record for record in records}
    for query_index, curve in enumerate(query_curves):
        record = query_lookup[query_index]
        color = cmap(color_norm(record['total_term']))
        draw_curve(ax, curve, color=color, linewidth=2.2, alpha=0.95)
        center = curve_center(curve)
        ax.text(
            float(center[0]),
            float(center[1]),
            format_annotation(record),
            fontsize=8.2,
            color='white',
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#111723', edgecolor=color, alpha=0.86),
        )
    weight_text = '  '.join([
        f"ctrl={weights['control_cost']:.1f}",
        f"giou={weights['giou_cost']:.1f}",
        f"curve={weights['curve_distance_cost']:.1f}",
    ])
    ax.set_title(f'Synthetic Matching Cost Breakdown ({scenario_name})\n{weight_text}', color='white', fontsize=16, pad=18)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#586174')
    fig.tight_layout()
    fig.savefig(output_dir / f'{scenario_name}_matching_visualization.png', dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_summary(output_dir: Path, scenario_name: str, records: List[Dict], correlations: Dict[str, float]) -> None:
    summary_lines = [
        f'# Synthetic Matching Summary: {scenario_name}',
        '',
        'Spearman correlation against dense curve reference distance:',
        f"- total_term: {correlations['total_term']:.4f}",
        f"- control_term: {correlations['control_term']:.4f}",
        f"- giou_term: {correlations['giou_term']:.4f}",
        f"- curve_term: {correlations['curve_term']:.4f}",
        '',
        'Largest ranking disagreements between total cost and dense curve reference:',
    ]
    disagreements = sorted(records, key=lambda item: abs(item['rank_gap']), reverse=True)[:8]
    for record in disagreements:
        summary_lines.append(
            f"- q{record['query_index']:02d}: total_rank={record['rank_total']}, ref_rank={record['rank_reference']}, gap={record['rank_gap']}, total={record['total_term']:.3f}, ref={record['reference_total']:.3f}"
        )
    summary_lines.append('')
    summary_lines.append('Lowest total cost queries:')
    for record in records[:8]:
        summary_lines.append(
            f"- q{record['query_index']:02d}: total={record['total_term']:.3f}, ctrl={record['control_term']:.3f}, giou={record['giou_term']:.3f}, curve={record['curve_term']:.3f}, ref={record['reference_total']:.3f}"
        )
    (output_dir / f'{scenario_name}_matching_summary.md').write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')


def save_records(output_dir: Path, scenario_name: str, records: List[Dict], correlations: Dict[str, float]) -> None:
    (output_dir / f'{scenario_name}_matching_records.json').write_text(
        json.dumps({'correlations': correlations, 'records': records}, indent=2),
        encoding='utf-8',
    )
    header = [
        'query_index', 'rank_total', 'rank_reference', 'rank_gap',
        'total_term', 'control_term', 'giou_term', 'curve_term',
        'reference_total', 'reference_chamfer', 'reference_length',
    ]
    csv_lines = [','.join(header)]
    for record in records:
        csv_lines.append(','.join(str(record[key]) for key in header))
    (output_dir / f'{scenario_name}_matching_records.csv').write_text('\n'.join(csv_lines) + '\n', encoding='utf-8')


def run_scenario(config: Dict, output_dir: Path, scenario_name: str, output_name: str, num_queries: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    num_control_points = int(config['model']['target_degree']) + 1
    target_curve = build_target_curve(num_control_points)
    query_curves = build_query_curves(num_queries=num_queries, num_control_points=num_control_points, rng=rng)
    probabilities = build_probabilities(mode=scenario_name, num_queries=num_queries, rng=rng)

    target_curve_t = torch.from_numpy(target_curve[None, ...])
    query_curves_t = torch.from_numpy(query_curves)
    costs = compute_cost_breakdown(config=config, target_curve=target_curve_t, query_curves=query_curves_t)
    records = build_records(costs)
    reference_values = np.array([record['reference_total'] for record in records], dtype=np.float64)
    correlations = {
        key: spearman_corr(np.array([record[key] for record in records], dtype=np.float64), reference_values)
        for key in ['total_term', 'control_term', 'giou_term', 'curve_term']
    }
    save_records(output_dir=output_dir, scenario_name=output_name, records=records, correlations=correlations)
    write_summary(output_dir=output_dir, scenario_name=output_name, records=records, correlations=correlations)
    weights = {
        'control_cost': float(config['loss'].get('control_cost', 5.0)),
        'giou_cost': float(config['loss'].get('giou_cost', 1.0)),
        'curve_distance_cost': float(config['loss'].get('curve_distance_cost', 1.0)),
    }
    visualize_scenario(output_dir=output_dir, scenario_name=output_name, target_curve=target_curve, query_curves=query_curves, records=records, weights=weights)
    return correlations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize synthetic matching cost breakdown for Bezier curves.')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/parametric_edge/default.yaml'),
        help='Path to config YAML.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('outputs/parametric_edge_training/current_sweep_analysis/matching_synthetic'),
        help='Directory where diagnostics will be written.',
    )
    parser.add_argument('--num-queries', type=int, default=24, help='Number of synthetic query curves.')
    parser.add_argument('--seed', type=int, default=13, help='Random seed for synthetic data generation.')
    parser.add_argument('--control-cost', type=float, default=None, help='Override control-point matching cost.')
    parser.add_argument('--giou-cost', type=float, default=None, help='Override gIoU matching cost.')
    parser.add_argument('--curve-distance-cost', type=float, default=None, help='Override lightweight curve-distance matching cost.')
    parser.add_argument('--name-suffix', type=str, default='', help='Optional suffix appended to scenario output names.')
    parser.add_argument(
        '--scenarios',
        nargs='+',
        default=['constant', 'random'],
        choices=['constant', 'random'],
        help='Probability modes to render.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = copy.deepcopy(config)
    if args.control_cost is not None:
        config['loss']['control_cost'] = float(args.control_cost)
    if args.giou_cost is not None:
        config['loss']['giou_cost'] = float(args.giou_cost)
    if args.curve_distance_cost is not None:
        config['loss']['curve_distance_cost'] = float(args.curve_distance_cost)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = {}
    for offset, scenario_name in enumerate(args.scenarios):
        effective_name = f'{scenario_name}{args.name_suffix}'
        correlations = run_scenario(
            config=config,
            output_dir=args.output_dir,
            scenario_name=scenario_name,
            output_name=effective_name,
            num_queries=args.num_queries,
            seed=args.seed + offset,
        )
        aggregate[effective_name] = correlations
    (args.output_dir / 'matching_correlation_overview.json').write_text(json.dumps(aggregate, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()