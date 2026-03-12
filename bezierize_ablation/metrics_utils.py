from collections import defaultdict
import math
import numpy as np

from bezier_refiner_core import compute_turning_angles, path_length


def classify_length(length):
    if length < 20:
        return 'short'
    if length < 60:
        return 'medium'
    return 'long'


def classify_topology(points):
    if len(points) < 3:
        return 'open'
    end_dist = np.linalg.norm(np.asarray(points[0], dtype=np.float64) - np.asarray(points[-1], dtype=np.float64))
    if end_dist <= 2.5 and path_length(points) >= 10:
        return 'closed'
    return 'open'


def classify_curvature(points):
    length = path_length(points)
    if length < 2:
        return 'flat'
    end_dist = np.linalg.norm(np.asarray(points[0], dtype=np.float64) - np.asarray(points[-1], dtype=np.float64))
    tortuosity = length / max(end_dist, 1.0)
    angles = compute_turning_angles(np.asarray(points, dtype=np.float64))
    mean_turn = float(np.mean(angles[1:-1])) if len(angles) > 2 else 0.0
    max_turn = float(np.max(angles)) if len(angles) > 0 else 0.0
    if tortuosity < 1.08 and max_turn < 25 and mean_turn < 10:
        return 'flat'
    if tortuosity < 1.25 and max_turn < 65 and mean_turn < 22:
        return 'medium_curvature'
    return 'very_curvy'


def path_bucket_record(points):
    length = path_length(points)
    return {
        'length': length,
        'length_bucket': classify_length(length),
        'curvature_bucket': classify_curvature(points),
        'topology_bucket': classify_topology(points),
    }


def summarize_path_records(fitted_paths, dropped_paths):
    bucket_stats = defaultdict(lambda: {
        'total_paths': 0,
        'fitted_paths': 0,
        'dropped_paths': 0,
        'total_length': 0.0,
        'total_segments': 0,
        'short_output_segments': 0,
        'sum_segment_mean_error': 0.0,
        'sum_segment_max_error': 0.0,
        'segment_counter': 0,
    })

    def update_bucket(bucket_name, bucket_value, record, fitted_segments=None):
        key = f'{bucket_name}:{bucket_value}'
        stats = bucket_stats[key]
        stats['total_paths'] += 1
        stats['total_length'] += record['length']
        if fitted_segments is None:
            stats['dropped_paths'] += 1
            return
        stats['fitted_paths'] += 1
        stats['total_segments'] += len(fitted_segments)
        for segment in fitted_segments:
            stats['segment_counter'] += 1
            stats['sum_segment_mean_error'] += float(segment['mean_error'])
            stats['sum_segment_max_error'] += float(segment['max_error'])
            if path_length(segment['points']) <= 3.0:
                stats['short_output_segments'] += 1

    for path_fit in fitted_paths:
        record = path_bucket_record(path_fit['original_points'])
        for bucket_name in ['length_bucket', 'curvature_bucket', 'topology_bucket']:
            update_bucket(bucket_name, record[bucket_name], record, fitted_segments=path_fit['segments'])

    for path in dropped_paths:
        record = path_bucket_record(path)
        for bucket_name in ['length_bucket', 'curvature_bucket', 'topology_bucket']:
            update_bucket(bucket_name, record[bucket_name], record, fitted_segments=None)

    final = {}
    for key, stats in bucket_stats.items():
        total_paths = stats['total_paths']
        fitted_paths_count = stats['fitted_paths']
        seg_count = stats['segment_counter']
        final[key] = {
            'total_paths': total_paths,
            'fitted_paths': fitted_paths_count,
            'dropped_paths': stats['dropped_paths'],
            'drop_rate': stats['dropped_paths'] / total_paths if total_paths else 0.0,
            'mean_input_length': stats['total_length'] / total_paths if total_paths else 0.0,
            'mean_segments_per_fitted_path': stats['total_segments'] / fitted_paths_count if fitted_paths_count else 0.0,
            'short_output_segments': stats['short_output_segments'],
            'mean_segment_mean_error': stats['sum_segment_mean_error'] / seg_count if seg_count else 0.0,
            'mean_segment_max_error': stats['sum_segment_max_error'] / seg_count if seg_count else 0.0,
        }
    return final


def aggregate_experiment_rows(rows):
    def avg(key):
        return sum(row[key] for row in rows) / len(rows) if rows else 0.0

    bucket_agg = defaultdict(list)
    for row in rows:
        for key, value in row['bucket_summary'].items():
            bucket_agg[key].append(value)

    bucket_summary = {}
    for key, values in bucket_agg.items():
        bucket_summary[key] = {
            'total_paths': sum(v['total_paths'] for v in values),
            'fitted_paths': sum(v['fitted_paths'] for v in values),
            'dropped_paths': sum(v['dropped_paths'] for v in values),
            'drop_rate': sum(v['dropped_paths'] for v in values) / max(sum(v['total_paths'] for v in values), 1),
            'mean_input_length': sum(v['mean_input_length'] * v['total_paths'] for v in values) / max(sum(v['total_paths'] for v in values), 1),
            'mean_segments_per_fitted_path': sum(v['mean_segments_per_fitted_path'] * v['fitted_paths'] for v in values) / max(sum(v['fitted_paths'] for v in values), 1),
            'short_output_segments': sum(v['short_output_segments'] for v in values),
            'mean_segment_mean_error': sum(v['mean_segment_mean_error'] * max(v['fitted_paths'], 1) for v in values) / max(sum(max(v['fitted_paths'], 1) for v in values), 1),
            'mean_segment_max_error': sum(v['mean_segment_max_error'] * max(v['fitted_paths'], 1) for v in values) / max(sum(max(v['fitted_paths'], 1) for v in values), 1),
        }

    return {
        'images': len(rows),
        'mean_f1': avg('f1'),
        'mean_precision': avg('precision'),
        'mean_recall': avg('recall'),
        'mean_chamfer': avg('chamfer'),
        'mean_path_count': avg('path_count'),
        'mean_segment_count': avg('segment_count'),
        'mean_dropped_paths': avg('dropped_paths'),
        'mean_mean_segment_error': avg('mean_segment_error'),
        'mean_max_segment_error': avg('max_segment_error'),
        'short20_cases': sum(1 for row in rows if row['short20'] > 0),
        'short40_cases': sum(1 for row in rows if row['short40'] > 0),
        'total_short20': sum(row['short20'] for row in rows),
        'total_short40': sum(row['short40'] for row in rows),
        'bucket_summary': bucket_summary,
    }
