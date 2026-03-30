from typing import Sequence, Tuple

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
