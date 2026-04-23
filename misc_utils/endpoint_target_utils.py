from typing import Dict, List, Sequence, Tuple

import numpy as np


def _as_hw(image_size: Sequence[int]) -> Tuple[int, int]:
    if len(image_size) < 2:
        raise ValueError(f'image_size must have at least two entries, got {image_size}')
    height = int(image_size[0])
    width = int(image_size[1])
    return height, width


def curves_to_unique_endpoints(
    curves: np.ndarray,
    image_size: Sequence[int],
    dedupe_distance_px: float = 2.0,
) -> np.ndarray:
    curves = np.asarray(curves, dtype=np.float32)
    if curves.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if curves.ndim != 3 or curves.shape[-1] != 2:
        raise ValueError(f'curves must have shape [N, P, 2], got {curves.shape}')

    height, width = _as_hw(image_size)
    endpoints = np.concatenate([curves[:, 0, :], curves[:, -1, :]], axis=0)
    scale = np.asarray([float(width), float(height)], dtype=np.float32)
    endpoints_px = endpoints * scale[None, :]

    centers_px = []
    counts = []
    threshold = float(dedupe_distance_px)
    for point_px in endpoints_px:
        if not centers_px:
            centers_px.append(point_px.copy())
            counts.append(1)
            continue
        distances = np.linalg.norm(np.stack(centers_px, axis=0) - point_px[None, :], axis=1)
        best_idx = int(np.argmin(distances))
        if float(distances[best_idx]) <= threshold:
            count = counts[best_idx]
            centers_px[best_idx] = (centers_px[best_idx] * count + point_px) / float(count + 1)
            counts[best_idx] = count + 1
        else:
            centers_px.append(point_px.copy())
            counts.append(1)

    unique_points = np.stack(centers_px, axis=0) / scale[None, :]
    return np.clip(unique_points.astype(np.float32), 0.0, 1.0)


def curves_to_endpoint_clusters_with_incidence(
    curves: np.ndarray,
    image_size: Sequence[int],
    dedupe_distance_px: float = 2.0,
    closed_curve_threshold_px: float = 2.0,
) -> Dict[str, np.ndarray]:
    curves = np.asarray(curves, dtype=np.float32)
    height, width = _as_hw(image_size)
    if curves.size == 0:
        return {
            'points': np.zeros((0, 2), dtype=np.float32),
            'point_degree': np.zeros((0,), dtype=np.int64),
            'point_is_loop_only': np.zeros((0,), dtype=bool),
            'point_curve_offsets': np.zeros((1,), dtype=np.int64),
            'point_curve_indices': np.zeros((0,), dtype=np.int64),
        }
    if curves.ndim != 3 or curves.shape[-1] != 2:
        raise ValueError(f'curves must have shape [N, P, 2], got {curves.shape}')

    scale = np.asarray([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)
    curves_px = curves * scale[None, None, :]
    endpoint_gap = np.linalg.norm(curves_px[:, 0, :] - curves_px[:, -1, :], axis=1)
    curve_is_closed = endpoint_gap <= float(closed_curve_threshold_px)

    contributions = []
    for curve_idx, is_closed in enumerate(curve_is_closed.tolist()):
        contributions.append((curves_px[curve_idx, 0].copy(), int(curve_idx), bool(is_closed)))
        contributions.append((curves_px[curve_idx, -1].copy(), int(curve_idx), bool(is_closed)))

    centers: List[np.ndarray] = []
    counts: List[int] = []
    cluster_curve_indices: List[List[int]] = []
    cluster_has_closed: List[bool] = []
    cluster_has_open: List[bool] = []
    threshold = float(dedupe_distance_px)
    for point_px, curve_idx, is_closed in contributions:
        if not centers:
            centers.append(point_px.copy())
            counts.append(1)
            cluster_curve_indices.append([curve_idx])
            cluster_has_closed.append(is_closed)
            cluster_has_open.append(not is_closed)
            continue
        distances = np.linalg.norm(np.stack(centers, axis=0) - point_px[None, :], axis=1)
        best_idx = int(np.argmin(distances))
        if float(distances[best_idx]) <= threshold:
            count = counts[best_idx]
            centers[best_idx] = (centers[best_idx] * count + point_px) / float(count + 1)
            counts[best_idx] = count + 1
            if curve_idx not in cluster_curve_indices[best_idx]:
                cluster_curve_indices[best_idx].append(curve_idx)
            cluster_has_closed[best_idx] = cluster_has_closed[best_idx] or is_closed
            cluster_has_open[best_idx] = cluster_has_open[best_idx] or (not is_closed)
        else:
            centers.append(point_px.copy())
            counts.append(1)
            cluster_curve_indices.append([curve_idx])
            cluster_has_closed.append(is_closed)
            cluster_has_open.append(not is_closed)

    if not centers:
        points = np.zeros((0, 2), dtype=np.float32)
    else:
        points = np.clip(np.stack(centers, axis=0).astype(np.float32) / scale[None, :], 0.0, 1.0)
    offsets = [0]
    flat_curve_indices: List[int] = []
    for incident in cluster_curve_indices:
        unique_incident = sorted(set(int(idx) for idx in incident))
        flat_curve_indices.extend(unique_incident)
        offsets.append(len(flat_curve_indices))

    return {
        'points': points.astype(np.float32),
        'point_degree': np.asarray(counts, dtype=np.int64),
        'point_is_loop_only': np.asarray(
            [has_closed and (not has_open) for has_closed, has_open in zip(cluster_has_closed, cluster_has_open)],
            dtype=bool,
        ),
        'point_curve_offsets': np.asarray(offsets, dtype=np.int64),
        'point_curve_indices': np.asarray(flat_curve_indices, dtype=np.int64),
    }


def polylines_to_unique_endpoint_count(
    polylines: Sequence[np.ndarray],
    dedupe_distance_px: float = 2.0,
) -> int:
    if not polylines:
        return 0

    endpoints = []
    for polyline in polylines:
        polyline = np.asarray(polyline, dtype=np.float32)
        if polyline.ndim != 2 or polyline.shape[0] < 2 or polyline.shape[1] != 2:
            continue
        endpoints.append(polyline[0])
        endpoints.append(polyline[-1])
    if not endpoints:
        return 0

    centers = []
    counts = []
    threshold = float(dedupe_distance_px)
    for point in endpoints:
        point = np.asarray(point, dtype=np.float32)
        if not centers:
            centers.append(point.copy())
            counts.append(1)
            continue
        distances = np.linalg.norm(np.stack(centers, axis=0) - point[None, :], axis=1)
        best_idx = int(np.argmin(distances))
        if float(distances[best_idx]) <= threshold:
            count = counts[best_idx]
            centers[best_idx] = (centers[best_idx] * count + point) / float(count + 1)
            counts[best_idx] = count + 1
        else:
            centers.append(point.copy())
            counts.append(1)
    return len(centers)
