import argparse
import math
import os
import sys

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from bezier_refiner_core import (
    Image,
    cleanup_tiny_bezier_segments,
    evaluate_bezier,
    extract_ordered_edge_paths,
    fit_polyline_with_piecewise_bezier,
    harmonize_similar_smooth_paths,
    merge_easy_adjacent_segments,
    path_length,
)
from bezier_versions.v1_hybrid_initial import DEFAULT_CONFIG as V1_CONFIG, VERSION_NAME as V1_NAME
from bezier_versions.v2_dense_sampling import DEFAULT_CONFIG as V2_CONFIG, VERSION_NAME as V2_NAME
from bezier_versions.v3_trivial_filter import DEFAULT_CONFIG as V3_CONFIG, VERSION_NAME as V3_NAME
from bezier_versions.v4_current import DEFAULT_CONFIG as V4_CONFIG, VERSION_NAME as V4_NAME
from bezier_versions.v5_human_split import DEFAULT_CONFIG as V5_CONFIG, VERSION_NAME as V5_NAME

PALETTE = [
    '#ff595e', '#ffca3a', '#8ac926', '#1982c4', '#6a4c93', '#f72585',
    '#b5179e', '#7209b7', '#4361ee', '#4cc9f0', '#fb5607', '#8338ec',
]

VERSIONS = [
    (V1_NAME, 'V1 Greedy Hybrid', V1_CONFIG),
    (V2_NAME, 'V2 Tiny Cleanup', V2_CONFIG),
    (V3_NAME, 'V3 Trivial Pruned', V3_CONFIG),
    (V4_NAME, 'V4 Adjacent Merge', V4_CONFIG),
    (V5_NAME, 'V5 Anchor Consistent', V5_CONFIG),
]


def color_for(path_idx, seg_idx=0):
    return PALETTE[(path_idx * 3 + seg_idx) % len(PALETTE)]


def build_segment_records(fitted_paths):
    records = []
    for path_idx, path_fit in enumerate(fitted_paths):
        for seg_idx, segment in enumerate(path_fit['segments']):
            control_points = np.asarray(segment['control_points'], dtype=np.float64)
            degree = len(control_points) - 1
            start = control_points[0]
            end = control_points[-1]
            initial_control = np.linspace(start, end, degree + 1)
            records.append({
                'path_idx': path_idx,
                'seg_idx': seg_idx,
                'color': color_for(path_idx, seg_idx),
                'initial_control': initial_control,
                'final_control': control_points,
            })
    return records


def sample_segment_record(record, alpha, samples=72):
    control = (1.0 - alpha) * record['initial_control'] + alpha * record['final_control']
    t_values = np.linspace(0.0, 1.0, samples)
    return evaluate_bezier(control, t_values)


def stage_polyline_curves(paths):
    curves = []
    for path_idx, path in enumerate(paths):
        curves.append({
            'color': color_for(path_idx, 0),
            'points': np.asarray(path, dtype=np.float64),
        })
    return curves


def transition_frames(prev_stage, next_stage, fit_frames=36, hold_frames=18, fade_frames=12):
    frames = []
    if prev_stage is None:
        for _ in range(hold_frames):
            frames.append({'mode': 'hold', 'stage': next_stage})
        return frames

    if next_stage['mode'] == 'fit':
        for alpha in np.linspace(0.0, 1.0, fit_frames):
            frames.append({'mode': 'fit', 'stage': next_stage, 'alpha': float(alpha)})
        for _ in range(hold_frames):
            frames.append({'mode': 'hold', 'stage': {'mode': 'curves', 'label': next_stage['label'], 'curves': [
                {'color': record['color'], 'points': sample_segment_record(record, 1.0)}
                for record in next_stage['records']
            ]}})
        return frames

    for alpha in np.linspace(0.0, 1.0, fade_frames):
        frames.append({'mode': 'crossfade', 'prev': prev_stage, 'next': next_stage, 'alpha': float(alpha)})
    for _ in range(hold_frames):
        frames.append({'mode': 'hold', 'stage': next_stage})
    return frames


def drawable_curves(stage):
    if stage['mode'] == 'curves':
        return stage['curves']
    if stage['mode'] == 'fit':
        return [
            {'color': record['color'], 'points': sample_segment_record(record, 1.0)}
            for record in stage['records']
        ]
    return []


def render_stage(ax, edge_map, version_label, frame_payload):
    ax.clear()
    ax.imshow(edge_map, cmap='gray', alpha=0.28)
    ax.set_axis_off()

    if frame_payload['mode'] == 'hold':
        stage = frame_payload['stage']
        curves = drawable_curves(stage)
        for curve in curves:
            pts = np.asarray(curve['points'])
            ax.plot(pts[:, 1], pts[:, 0], color=curve['color'], linewidth=2.0)
        stage_label = stage['label']

    elif frame_payload['mode'] == 'crossfade':
        prev_stage = frame_payload['prev']
        next_stage = frame_payload['next']
        alpha = frame_payload['alpha']
        for curve in drawable_curves(prev_stage):
            pts = np.asarray(curve['points'])
            ax.plot(pts[:, 1], pts[:, 0], color=curve['color'], linewidth=1.6, alpha=max(0.0, 1.0 - alpha))
        for curve in drawable_curves(next_stage):
            pts = np.asarray(curve['points'])
            ax.plot(pts[:, 1], pts[:, 0], color=curve['color'], linewidth=2.0, alpha=alpha)
        stage_label = next_stage['label']

    elif frame_payload['mode'] == 'fit':
        stage = frame_payload['stage']
        alpha = frame_payload['alpha']
        for record in stage['records']:
            pts = sample_segment_record(record, alpha)
            ax.plot(pts[:, 1], pts[:, 0], color=record['color'], linewidth=2.0)
        stage_label = stage['label']

    ax.set_title(stage_label, fontsize=11)
    ax.text(0.5, -0.045, version_label, transform=ax.transAxes, ha='center', va='top', fontsize=11)
    ax.set_xlim(0, edge_map.shape[1])
    ax.set_ylim(edge_map.shape[0], 0)


def build_version_stages(image_path, config):
    image = np.array(Image.open(image_path).convert('L'))
    edge_map = (image > 127).astype(np.uint8)
    paths, _, _, _, _ = extract_ordered_edge_paths(edge_map, connectivity=2)

    stages = []
    original_curves = stage_polyline_curves(paths)
    stages.append({'mode': 'curves', 'label': 'Stage: Extract Paths', 'curves': original_curves})

    kept_paths = [np.asarray(path, dtype=np.float64) for path in paths]
    dropped_paths = []
    min_path_length = config.get('min_path_length_for_bezier', 0.0)
    if min_path_length > 0:
        next_kept = []
        for path in kept_paths:
            if path_length(path) < min_path_length:
                dropped_paths.append(path)
            else:
                next_kept.append(path)
        kept_paths = next_kept
        stages.append({'mode': 'curves', 'label': 'Stage: Trivial Path Filter', 'curves': stage_polyline_curves(kept_paths)})

    fitted_paths = []
    fit_kwargs = {
        'max_degree': config['max_degree'],
        'mean_error_threshold': config['mean_error_threshold'],
        'max_error_threshold': config['max_error_threshold'],
        'max_segment_length': config['max_segment_length'],
        'angle_threshold_deg': config['angle_threshold_deg'],
        'min_points': config['min_points'],
    }
    optional_fit_keys = [
        'use_global_chunk_dp',
        'split_anchor_weight',
        'split_extrema_window',
        'prefer_anchor_for_length_split',
        'length_split_lookahead',
        'length_split_min_strength',
        'short_segment_penalty_weight',
        'preferred_min_segment_ratio',
        'enable_smooth_consistency',
        'smooth_consistency_min_path_length',
        'smooth_consistency_q90_turn_threshold',
        'smooth_consistency_max_strong_anchors',
        'smooth_consistency_strong_anchor_threshold',
        'smooth_consistency_target_length_factor',
        'smooth_consistency_snap_window',
        'smooth_consistency_snap_anchor_strength',
    ]
    for key in optional_fit_keys:
        if key in config:
            fit_kwargs[key] = config[key]
    for path in kept_paths:
        segments = fit_polyline_with_piecewise_bezier(path, **fit_kwargs)
        fitted_paths.append({'original_points': path, 'segments': segments})
    if config.get('enable_bundle_consistency', False) and len(fitted_paths) > 1:
        fitted_paths = harmonize_similar_smooth_paths(
            fitted_paths,
            max_degree=config['max_degree'],
            mean_error_threshold=config['mean_error_threshold'],
            max_error_threshold=config['max_error_threshold'],
            min_points=config['min_points'],
            extrema_window=config.get('split_extrema_window', 5),
            min_path_length=config.get('bundle_consistency_min_path_length', 100.0),
            max_length_ratio=config.get('bundle_consistency_max_length_ratio', 1.45),
            descriptor_distance_threshold=config.get('bundle_consistency_descriptor_threshold', 0.09),
            snap_window=config.get('bundle_consistency_snap_window', 28.0),
            snap_anchor_strength=config.get('bundle_consistency_snap_anchor_strength', 0.58),
        )
    fit_records = build_segment_records(fitted_paths)
    stages.append({'mode': 'fit', 'label': 'Stage: Fit Bezier', 'records': fit_records})

    if config.get('enable_tiny_cleanup', False):
        cleaned_paths = []
        for path_fit in fitted_paths:
            cleaned_segments = cleanup_tiny_bezier_segments(
                path_fit['segments'],
                parent_length=path_length(path_fit['original_points']),
                max_degree=config['max_degree'],
                mean_error_threshold=config['mean_error_threshold'],
                max_error_threshold=config['max_error_threshold'],
                tiny_length_threshold=config['tiny_segment_length'],
                long_path_threshold=config.get('cleanup_long_path_threshold', 20.0),
                min_points=config['min_points'],
            )
            cleaned_paths.append({'original_points': path_fit['original_points'], 'segments': cleaned_segments})
        fitted_paths = cleaned_paths
        stages.append({'mode': 'curves', 'label': 'Stage: Tiny Cleanup', 'curves': [
            {'color': color_for(path_idx, seg_idx), 'points': sample_segment_record({
                'path_idx': path_idx,
                'seg_idx': seg_idx,
                'color': color_for(path_idx, seg_idx),
                'initial_control': np.asarray(segment['control_points'], dtype=np.float64),
                'final_control': np.asarray(segment['control_points'], dtype=np.float64),
            }, 1.0)}
            for path_idx, path_fit in enumerate(fitted_paths)
            for seg_idx, segment in enumerate(path_fit['segments'])
        ]})

    if config.get('enable_easy_merge', False):
        from bezier_refiner_core import merge_easy_adjacent_segments
        merged_paths = []
        for path_fit in fitted_paths:
            merged_segments = merge_easy_adjacent_segments(path_fit['segments'], max_degree=config['max_degree'])
            merged_paths.append({'original_points': path_fit['original_points'], 'segments': merged_segments})
        fitted_paths = merged_paths
        stages.append({'mode': 'curves', 'label': 'Stage: Adjacent Merge', 'curves': [
            {'color': color_for(path_idx, seg_idx), 'points': sample_segment_record({
                'path_idx': path_idx,
                'seg_idx': seg_idx,
                'color': color_for(path_idx, seg_idx),
                'initial_control': np.asarray(segment['control_points'], dtype=np.float64),
                'final_control': np.asarray(segment['control_points'], dtype=np.float64),
            }, 1.0)}
            for path_idx, path_fit in enumerate(fitted_paths)
            for seg_idx, segment in enumerate(path_fit['segments'])
        ]})

    return edge_map, stages


def build_timeline(version_stages, fit_frames=36, hold_frames=18, fade_frames=12):
    timelines = []
    max_frames = 0
    for _, _, _, stages in version_stages:
        frames = []
        prev = None
        for stage in stages:
            frames.extend(
                transition_frames(
                    prev,
                    stage,
                    fit_frames=fit_frames,
                    hold_frames=hold_frames,
                    fade_frames=fade_frames,
                )
            )
            prev = stage
        timelines.append(frames)
        max_frames = max(max_frames, len(frames))

    for frames in timelines:
        if len(frames) < max_frames:
            frames.extend([frames[-1]] * (max_frames - len(frames)))
    return timelines, max_frames


def render_animation(
    version_stages,
    output_path_mp4,
    output_path_gif,
    fps=10,
    fit_frames=36,
    hold_frames=18,
    fade_frames=12,
):
    timelines, max_frames = build_timeline(
        version_stages,
        fit_frames=fit_frames,
        hold_frames=hold_frames,
        fade_frames=fade_frames,
    )
    n_versions = len(version_stages)
    ncols = 2
    nrows = int(math.ceil(n_versions / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4.8 * nrows))
    axes = np.atleast_1d(axes).flatten()
    canvas = FigureCanvasAgg(fig)
    frames = []

    for frame_idx in range(max_frames):
        for ax, (version_name, version_label, edge_map, _), timeline in zip(axes, version_stages, timelines):
            render_stage(ax, edge_map, version_label, timeline[frame_idx])
        for ax in axes[len(version_stages):]:
            ax.clear()
            ax.set_axis_off()
        fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.04, wspace=0.04, hspace=0.12)
        canvas.draw()
        width, height = fig.canvas.get_width_height()
        frame = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3]
        frames.append(frame.copy())

    iio.imwrite(output_path_gif, frames, duration=1000 / fps, loop=0)
    iio.imwrite(output_path_mp4, frames, fps=fps)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Create side-by-side Bezier version comparison animation.')
    parser.add_argument('--input', default='gt_rgb/test/51084_ann1.png')
    parser.add_argument('--output-dir', default='bezierize_animation/outputs')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--fit-frames', type=int, default=36)
    parser.add_argument('--hold-frames', type=int, default=18)
    parser.add_argument('--fade-frames', type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    version_stages = []
    for version_name, version_label, config in VERSIONS:
        edge_map, stages = build_version_stages(args.input, config)
        version_stages.append((version_name, version_label, edge_map, stages))

    stem = os.path.splitext(os.path.basename(args.input))[0]
    mp4_path = os.path.join(args.output_dir, f'{stem}_version_comparison.mp4')
    gif_path = os.path.join(args.output_dir, f'{stem}_version_comparison.gif')
    render_animation(
        version_stages,
        mp4_path,
        gif_path,
        fps=args.fps,
        fit_frames=args.fit_frames,
        hold_frames=args.hold_frames,
        fade_frames=args.fade_frames,
    )
    print(mp4_path)
    print(gif_path)


if __name__ == '__main__':
    main()
