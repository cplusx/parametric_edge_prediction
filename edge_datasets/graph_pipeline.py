from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

from misc_utils.bezier_target_utils import control_point_bbox, curve_length, denormalize_control_points, fit_polyline_to_bezier, polyline_curvature_score, sample_bezier_numpy


def _size_tuple(image_size) -> Tuple[int, int]:
    if isinstance(image_size, int):
        return int(image_size), int(image_size)
    if isinstance(image_size, Sequence) and len(image_size) == 2:
        return int(image_size[0]), int(image_size[1])
    raise ValueError(f'Unsupported image_size: {image_size}')


def _resize_image(image: np.ndarray, height: int, width: int) -> np.ndarray:
    image_uint8 = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    if image_uint8.shape[2] == 1:
        pil = Image.fromarray(image_uint8[..., 0], mode='L')
        resized = np.asarray(pil.resize((width, height), resample=Image.BILINEAR), dtype=np.float32) / 255.0
        return resized[..., None]
    pil = Image.fromarray(image_uint8, mode='RGB')
    resized = np.asarray(pil.resize((width, height), resample=Image.BILINEAR), dtype=np.float32) / 255.0
    return resized


def _resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 2:
        mask = mask[..., None]
    channels = []
    src_h, src_w = mask.shape[:2]
    downscaling = height < src_h or width < src_w
    for channel_idx in range(mask.shape[2]):
        channel_mask = np.asarray(mask[..., channel_idx] > 0.0, dtype=np.uint8)
        if downscaling and channel_mask.any():
            channel_mask = ndimage.binary_dilation(channel_mask, iterations=1).astype(np.uint8)
            pil = Image.fromarray(channel_mask * 255, mode='L')
            resized = np.asarray(pil.resize((width, height), resample=Image.BOX), dtype=np.uint8)
            channel = (resized > 0).astype(np.float32)
        else:
            pil = Image.fromarray(channel_mask * 255, mode='L')
            resized = np.asarray(pil.resize((width, height), resample=Image.NEAREST), dtype=np.uint8)
            channel = (resized > 0).astype(np.float32)
        channels.append(channel)
    return np.stack(channels, axis=-1)


def _scale_polylines(polylines: List[np.ndarray], scale_x: float, scale_y: float) -> List[np.ndarray]:
    scale = np.asarray([scale_x, scale_y], dtype=np.float32)
    return [np.asarray(polyline, dtype=np.float32) * scale for polyline in polylines]




def _scale_curves(curves: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    curves = np.asarray(curves, dtype=np.float32)
    if curves.size == 0:
        return np.zeros_like(curves, dtype=np.float32)
    scale = np.asarray([scale_x, scale_y], dtype=np.float32)
    return (curves * scale[None, None, :]).astype(np.float32)


def apply_horizontal_flip_to_curves(curves: np.ndarray, width: int) -> np.ndarray:
    curves = np.asarray(curves, dtype=np.float32).copy()
    if curves.size == 0:
        return curves
    curves[..., 0] = (width - 1) - curves[..., 0]
    return curves


def apply_vertical_flip_to_curves(curves: np.ndarray, height: int) -> np.ndarray:
    curves = np.asarray(curves, dtype=np.float32).copy()
    if curves.size == 0:
        return curves
    curves[..., 1] = (height - 1) - curves[..., 1]
    return curves


def apply_affine_to_curves(curves: np.ndarray, matrix_xy: np.ndarray) -> np.ndarray:
    curves = np.asarray(curves, dtype=np.float32)
    if curves.size == 0:
        return np.zeros_like(curves, dtype=np.float32)
    linear = matrix_xy[:2, :2]
    offset = matrix_xy[:2, 2]
    return ((curves @ linear.T) + offset).astype(np.float32)

def _normalize_xy_control_points(control_points: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = np.asarray([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)
    return np.asarray(control_points, dtype=np.float32) / scale


def _constant_pad_image(
    image: np.ndarray,
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
) -> np.ndarray:
    if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
        return image
    fill_value = np.asarray(image, dtype=np.float32).mean(axis=(0, 1), keepdims=True)
    output = np.broadcast_to(fill_value, (image.shape[0] + pad_top + pad_bottom, image.shape[1] + pad_left + pad_right, image.shape[2])).copy()
    output[pad_top: pad_top + image.shape[0], pad_left: pad_left + image.shape[1]] = image
    return output.astype(np.float32)


def resize_with_aspect_ratio(
    image: np.ndarray,
    polylines: List[np.ndarray],
    output_size: Tuple[int, int],
    scale_multiplier: float = 1.0,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    target_h, target_w = output_size
    src_h, src_w = image.shape[:2]
    min_scale = max(target_h / max(src_h, 1), target_w / max(src_w, 1))
    scale = max(min_scale, 1e-6) * float(scale_multiplier)
    new_h = max(1, int(round(src_h * scale)))
    new_w = max(1, int(round(src_w * scale)))
    resized = _resize_image(image, new_h, new_w)
    return resized, _scale_polylines(polylines, new_w / max(src_w, 1), new_h / max(src_h, 1))


def apply_horizontal_flip(image: np.ndarray, polylines: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
    width = image.shape[1]
    flipped = image[:, ::-1].copy()
    flipped_polylines = []
    for polyline in polylines:
        points = np.asarray(polyline, dtype=np.float32).copy()
        points[:, 0] = (width - 1) - points[:, 0]
        flipped_polylines.append(points)
    return flipped, flipped_polylines


def apply_vertical_flip(image: np.ndarray, polylines: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
    height = image.shape[0]
    flipped = image[::-1].copy()
    flipped_polylines = []
    for polyline in polylines:
        points = np.asarray(polyline, dtype=np.float32).copy()
        points[:, 1] = (height - 1) - points[:, 1]
        flipped_polylines.append(points)
    return flipped, flipped_polylines


def _translation_matrix(tx: float, ty: float) -> np.ndarray:
    return np.asarray([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=np.float32)


def _rotation_scale_matrix(angle_deg: float, scale: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    cos_theta = np.cos(theta) * scale
    sin_theta = np.sin(theta) * scale
    return np.asarray(
        [[cos_theta, -sin_theta, 0.0], [sin_theta, cos_theta, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def build_centered_affine_matrix(width: int, height: int, angle_deg: float, scale: float, translate_xy: Tuple[float, float]) -> np.ndarray:
    center = np.asarray([(width - 1) * 0.5, (height - 1) * 0.5], dtype=np.float32)
    return (
        _translation_matrix(float(translate_xy[0]), float(translate_xy[1]))
        @ _translation_matrix(center[0], center[1])
        @ _rotation_scale_matrix(angle_deg, scale)
        @ _translation_matrix(-center[0], -center[1])
    )


def _xy_to_rc_homography(matrix_xy: np.ndarray) -> np.ndarray:
    swap = np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return swap @ matrix_xy @ swap


def apply_affine_to_image(image: np.ndarray, matrix_xy: np.ndarray) -> np.ndarray:
    matrix_rc = _xy_to_rc_homography(matrix_xy)
    inverse_rc = np.linalg.inv(matrix_rc)
    output = np.empty_like(image)
    for channel_idx in range(image.shape[2]):
        output[..., channel_idx] = ndimage.affine_transform(
            image[..., channel_idx],
            matrix=inverse_rc[:2, :2],
            offset=inverse_rc[:2, 2],
            output_shape=image.shape[:2],
            order=1,
            mode='reflect',
            prefilter=False,
        )
    return np.clip(output, 0.0, 1.0)


def apply_affine_to_mask(mask: np.ndarray, matrix_xy: np.ndarray) -> np.ndarray:
    matrix_rc = _xy_to_rc_homography(matrix_xy)
    inverse_rc = np.linalg.inv(matrix_rc)
    if mask.ndim == 2:
        mask = mask[..., None]
    output = np.empty_like(mask)
    for channel_idx in range(mask.shape[2]):
        output[..., channel_idx] = ndimage.affine_transform(
            mask[..., channel_idx],
            matrix=inverse_rc[:2, :2],
            offset=inverse_rc[:2, 2],
            output_shape=mask.shape[:2],
            order=0,
            mode='constant',
            cval=0.0,
            prefilter=False,
        )
    return (output > 0.0).astype(np.float32)


def apply_affine_to_polylines(polylines: List[np.ndarray], matrix_xy: np.ndarray) -> List[np.ndarray]:
    transformed = []
    linear = matrix_xy[:2, :2]
    offset = matrix_xy[:2, 2]
    for polyline in polylines:
        points = np.asarray(polyline, dtype=np.float32)
        transformed.append((points @ linear.T) + offset)
    return transformed


def _clip_segment_to_rect(p0: np.ndarray, p1: np.ndarray, width: int, height: int):
    xmin, ymin = 0.0, 0.0
    xmax, ymax = float(max(width - 1, 0)), float(max(height - 1, 0))
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    p = [-dx, dx, -dy, dy]
    q = [p0[0] - xmin, xmax - p0[0], p0[1] - ymin, ymax - p0[1]]
    u0, u1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-8:
            if qi < 0.0:
                return None
            continue
        t = qi / pi
        if pi < 0.0:
            if t > u1:
                return None
            u0 = max(u0, t)
        else:
            if t < u0:
                return None
            u1 = min(u1, t)
    clipped_start = p0 + u0 * np.asarray([dx, dy], dtype=np.float32)
    clipped_end = p0 + u1 * np.asarray([dx, dy], dtype=np.float32)
    return clipped_start.astype(np.float32), clipped_end.astype(np.float32)


def _deduplicate_points(points) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if points.shape[0] == 1:
        return points.astype(np.float32, copy=True)
    deltas = np.linalg.norm(points[1:] - points[:-1], axis=1)
    keep = np.concatenate(
        [np.ones(1, dtype=bool), deltas > 1e-4],
        axis=0,
    )
    return points[keep].astype(np.float32, copy=False)


def _clip_segments_to_bounds_vectorized(
    starts: np.ndarray,
    ends: np.ndarray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
):
    if starts.size == 0:
        empty = np.zeros((0, 2), dtype=np.float32)
        return np.zeros((0,), dtype=bool), empty, empty
    starts = np.asarray(starts, dtype=np.float32)
    ends = np.asarray(ends, dtype=np.float32)
    delta = ends - starts
    dx = delta[:, 0]
    dy = delta[:, 1]
    p = np.stack([-dx, dx, -dy, dy], axis=1)
    q = np.stack(
        [
            starts[:, 0] - xmin,
            xmax - starts[:, 0],
            starts[:, 1] - ymin,
            ymax - starts[:, 1],
        ],
        axis=1,
    )
    u0 = np.zeros((starts.shape[0],), dtype=np.float32)
    u1 = np.ones((starts.shape[0],), dtype=np.float32)
    valid = np.ones((starts.shape[0],), dtype=bool)
    eps = 1e-8
    for boundary_idx in range(4):
        pi = p[:, boundary_idx]
        qi = q[:, boundary_idx]
        parallel = np.abs(pi) < eps
        valid &= ~(parallel & (qi < 0.0))
        non_parallel = ~parallel
        if not np.any(non_parallel):
            continue
        t = np.zeros_like(pi, dtype=np.float32)
        t[non_parallel] = qi[non_parallel] / pi[non_parallel]
        negative = non_parallel & (pi < 0.0)
        positive = non_parallel & (pi > 0.0)
        if np.any(negative):
            u0[negative] = np.maximum(u0[negative], t[negative])
        if np.any(positive):
            u1[positive] = np.minimum(u1[positive], t[positive])
    valid &= u0 <= u1
    clipped_starts = starts + u0[:, None] * delta
    clipped_ends = starts + u1[:, None] * delta
    return valid, clipped_starts.astype(np.float32), clipped_ends.astype(np.float32)


def _clip_segments_to_rect_vectorized(starts: np.ndarray, ends: np.ndarray, width: int, height: int):
    return _clip_segments_to_bounds_vectorized(
        starts,
        ends,
        xmin=0.0,
        ymin=0.0,
        xmax=float(max(width - 1, 0)),
        ymax=float(max(height - 1, 0)),
    )


def _pad_image_mask_and_polylines_for_crop(
    image: np.ndarray,
    mask: np.ndarray,
    polylines: List[np.ndarray],
    crop_h: int,
    crop_w: int,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    height, width = image.shape[:2]
    if height >= crop_h and width >= crop_w:
        return image, mask, polylines
    pad_h = max(crop_h - height, 0)
    pad_w = max(crop_w - width, 0)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    image = np.pad(image, ((top, bottom), (left, right), (0, 0)), mode='reflect')
    mask = np.pad(mask, ((top, bottom), (left, right), (0, 0)), mode='constant', constant_values=0.0)
    polylines = [np.asarray(polyline, dtype=np.float32) + np.asarray([left, top], dtype=np.float32) for polyline in polylines]
    return image, mask, polylines


def random_crop_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    crop_size: Tuple[int, int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    if mask.ndim == 2:
        mask = mask[..., None]
    crop_h, crop_w = crop_size
    height, width = image.shape[:2]
    max_top = height - crop_h
    max_left = width - crop_w
    top = int(rng.integers(0, max_top + 1)) if max_top > 0 else 0
    left = int(rng.integers(0, max_left + 1)) if max_left > 0 else 0
    cropped_image = image[top:top + crop_h, left:left + crop_w].copy()
    cropped_mask = mask[top:top + crop_h, left:left + crop_w].copy()
    return cropped_image, cropped_mask, (left, top)


def _dedupe_point_cloud(points: List[np.ndarray], dedupe_distance_px: float) -> np.ndarray:
    points_px = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if points_px.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if points_px.shape[0] == 1:
        return points_px.astype(np.float32, copy=True)

    # Keep ordering deterministic before greedy suppression.
    order = np.lexsort((points_px[:, 1], points_px[:, 0]))
    sorted_points = points_px[order]
    threshold_sq = float(dedupe_distance_px) ** 2
    deltas = sorted_points[:, None, :] - sorted_points[None, :, :]
    distance_sq = np.sum(deltas * deltas, axis=-1)

    suppressed = np.zeros((sorted_points.shape[0],), dtype=bool)
    kept_points: List[np.ndarray] = []
    for idx in range(sorted_points.shape[0]):
        if suppressed[idx]:
            continue
        kept_points.append(sorted_points[idx])
        suppressed |= distance_sq[idx] <= threshold_sq
        suppressed[idx] = True

    return np.stack(kept_points, axis=0).astype(np.float32, copy=False)


def _curve_endpoint_points_px(curves: np.ndarray, width: int, height: int) -> np.ndarray:
    curves = np.asarray(curves, dtype=np.float32)
    if curves.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if curves.ndim != 3 or curves.shape[-1] != 2:
        raise ValueError(f'curves must have shape [N, P, 2], got {curves.shape}')
    scale = np.asarray([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)
    endpoints = np.concatenate([curves[:, 0, :], curves[:, -1, :]], axis=0)
    return (endpoints * scale[None, :]).astype(np.float32, copy=False)


def _scale_points(points: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    scale = np.asarray([scale_x, scale_y], dtype=np.float32)
    return (points * scale[None, :]).astype(np.float32)


def _apply_horizontal_flip_to_points(points: np.ndarray, width: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).copy()
    if points.size == 0:
        return points.reshape(0, 2)
    points[:, 0] = (width - 1) - points[:, 0]
    return points


def _apply_vertical_flip_to_points(points: np.ndarray, height: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).copy()
    if points.size == 0:
        return points.reshape(0, 2)
    points[:, 1] = (height - 1) - points[:, 1]
    return points


def _apply_affine_to_points(points: np.ndarray, matrix_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    linear = matrix_xy[:2, :2]
    offset = matrix_xy[:2, 2]
    return ((points @ linear.T) + offset).astype(np.float32)


def _build_endpoint_targets_from_points(
    points_px: np.ndarray,
    width: int,
    height: int,
    dedupe_distance_px: float,
) -> Dict[str, np.ndarray]:
    points_px = np.asarray(points_px, dtype=np.float32).reshape(-1, 2)
    if points_px.size:
        inside = (
            (points_px[:, 0] >= 0.0)
            & (points_px[:, 0] <= float(max(width - 1, 0)))
            & (points_px[:, 1] >= 0.0)
            & (points_px[:, 1] <= float(max(height - 1, 0)))
        )
        points_px = points_px[inside]
    if points_px.size == 0:
        point_array = np.zeros((0, 2), dtype=np.float32)
    else:
        deduped = _dedupe_point_cloud([point for point in points_px], dedupe_distance_px=dedupe_distance_px)
        scale = np.asarray([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)
        point_array = np.clip(deduped / scale[None, :], 0.0, 1.0).astype(np.float32)
    return {
        'points': point_array,
        'image_size': np.asarray([height, width], dtype=np.int64),
    }


def build_endpoint_targets_from_crop_union(
    curves: np.ndarray,
    crop_left: int,
    crop_top: int,
    crop_width: int,
    crop_height: int,
    dedupe_distance_px: float,
    num_curve_samples: int = 48,
) -> Dict[str, np.ndarray]:
    xmin = float(crop_left)
    ymin = float(crop_top)
    xmax = float(crop_left + max(crop_width - 1, 0))
    ymax = float(crop_top + max(crop_height - 1, 0))
    offset = np.asarray([crop_left, crop_top], dtype=np.float32)
    curves = np.asarray(curves, dtype=np.float32)
    candidate_points: List[np.ndarray] = []

    if curves.size == 0:
        point_array = np.zeros((0, 2), dtype=np.float32)
        return {
            'points': point_array,
            'image_size': np.asarray([crop_height, crop_width], dtype=np.int64),
        }

    endpoints = np.concatenate([curves[:, 0, :], curves[:, -1, :]], axis=0)
    inside = (
        (endpoints[:, 0] >= xmin)
        & (endpoints[:, 0] <= xmax)
        & (endpoints[:, 1] >= ymin)
        & (endpoints[:, 1] <= ymax)
    )
    if np.any(inside):
        for point in endpoints[inside]:
            candidate_points.append(point - offset)

    sampled_starts: List[np.ndarray] = []
    sampled_ends: List[np.ndarray] = []
    for control_points in curves:
        sampled = sample_bezier_numpy(control_points, num_samples=num_curve_samples)
        if sampled.shape[0] < 2:
            continue
        sampled_starts.append(sampled[:-1])
        sampled_ends.append(sampled[1:])

    if sampled_starts:
        starts = np.concatenate(sampled_starts, axis=0)
        ends = np.concatenate(sampled_ends, axis=0)
        valid, clipped_starts, clipped_ends = _clip_segments_to_bounds_vectorized(
            starts,
            ends,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )
        if np.any(valid):
            eps = 1e-4
            start_changed = valid & (np.linalg.norm(clipped_starts - starts, axis=1) > eps)
            end_changed = valid & (np.linalg.norm(clipped_ends - ends, axis=1) > eps)
            if np.any(start_changed):
                for point in clipped_starts[start_changed]:
                    candidate_points.append(point - offset)
            if np.any(end_changed):
                for point in clipped_ends[end_changed]:
                    candidate_points.append(point - offset)

    points_px = _dedupe_point_cloud(candidate_points, dedupe_distance_px=dedupe_distance_px)
    if points_px.size == 0:
        point_array = np.zeros((0, 2), dtype=np.float32)
    else:
        scale = np.asarray([max(crop_width - 1, 1), max(crop_height - 1, 1)], dtype=np.float32)
        point_array = np.clip(points_px / scale[None, :], 0.0, 1.0).astype(np.float32)
    return {
        'points': point_array,
        'image_size': np.asarray([crop_height, crop_width], dtype=np.int64),
    }


def clip_polylines_to_rect(polylines: List[np.ndarray], width: int, height: int) -> List[np.ndarray]:
    clipped_polylines: List[np.ndarray] = []
    for polyline in polylines:
        points = np.asarray(polyline, dtype=np.float32)
        if points.shape[0] < 2:
            continue
        starts = points[:-1]
        ends = points[1:]
        valid, clipped_starts, clipped_ends = _clip_segments_to_rect_vectorized(starts, ends, width, height)
        if not np.any(valid):
            continue
        valid_indices = np.flatnonzero(valid)
        run_start = 0
        for local_idx in range(1, valid_indices.shape[0] + 1):
            reached_end = local_idx == valid_indices.shape[0]
            if not reached_end:
                prev_idx = valid_indices[local_idx - 1]
                curr_idx = valid_indices[local_idx]
                contiguous_segments = curr_idx == prev_idx + 1
                connected_segments = (
                    contiguous_segments
                    and np.linalg.norm(clipped_ends[prev_idx] - clipped_starts[curr_idx]) <= 1e-3
                )
                if connected_segments:
                    continue
            run_indices = valid_indices[run_start:local_idx]
            run_points = np.concatenate(
                [clipped_starts[run_indices[:1]], clipped_ends[run_indices]],
                axis=0,
            )
            deduped = _deduplicate_points(run_points)
            if deduped.shape[0] >= 2:
                clipped_polylines.append(deduped)
            run_start = local_idx
    return [polyline for polyline in clipped_polylines if polyline.shape[0] >= 2]


def random_crop_image_and_polylines(
    image: np.ndarray,
    polylines: List[np.ndarray],
    crop_size: Tuple[int, int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    crop_h, crop_w = crop_size
    height, width = image.shape[:2]
    if height < crop_h or width < crop_w:
        pad_h = max(crop_h - height, 0)
        pad_w = max(crop_w - width, 0)
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        image = np.pad(image, ((top, bottom), (left, right), (0, 0)), mode='reflect')
        polylines = [np.asarray(polyline, dtype=np.float32) + np.asarray([left, top], dtype=np.float32) for polyline in polylines]
        height, width = image.shape[:2]
    max_top = height - crop_h
    max_left = width - crop_w
    top = int(rng.integers(0, max_top + 1)) if max_top > 0 else 0
    left = int(rng.integers(0, max_left + 1)) if max_left > 0 else 0
    cropped = image[top:top + crop_h, left:left + crop_w].copy()
    shifted = [np.asarray(polyline, dtype=np.float32) - np.asarray([left, top], dtype=np.float32) for polyline in polylines]
    return cropped, clip_polylines_to_rect(shifted, crop_w, crop_h)


def random_crop_image_mask_and_polylines(
    image: np.ndarray,
    mask: np.ndarray,
    polylines: List[np.ndarray],
    crop_size: Tuple[int, int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    if mask.ndim == 2:
        mask = mask[..., None]
    crop_h, crop_w = crop_size
    height, width = image.shape[:2]
    if height < crop_h or width < crop_w:
        pad_h = max(crop_h - height, 0)
        pad_w = max(crop_w - width, 0)
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        image = np.pad(image, ((top, bottom), (left, right), (0, 0)), mode='reflect')
        mask = np.pad(mask, ((top, bottom), (left, right), (0, 0)), mode='constant', constant_values=0.0)
        polylines = [np.asarray(polyline, dtype=np.float32) + np.asarray([left, top], dtype=np.float32) for polyline in polylines]
        height, width = image.shape[:2]
    max_top = height - crop_h
    max_left = width - crop_w
    top = int(rng.integers(0, max_top + 1)) if max_top > 0 else 0
    left = int(rng.integers(0, max_left + 1)) if max_left > 0 else 0
    cropped_image = image[top:top + crop_h, left:left + crop_w].copy()
    cropped_mask = mask[top:top + crop_h, left:left + crop_w].copy()
    shifted = [np.asarray(polyline, dtype=np.float32) - np.asarray([left, top], dtype=np.float32) for polyline in polylines]
    return cropped_image, cropped_mask, clip_polylines_to_rect(shifted, crop_w, crop_h)


def letterbox_image_and_polylines(image: np.ndarray, polylines: List[np.ndarray], output_size: Tuple[int, int]) -> Tuple[np.ndarray, List[np.ndarray]]:
    target_h, target_w = output_size
    src_h, src_w = image.shape[:2]
    scale = min(target_h / max(src_h, 1), target_w / max(src_w, 1))
    new_h = max(1, int(round(src_h * scale)))
    new_w = max(1, int(round(src_w * scale)))
    resized = _resize_image(image, new_h, new_w)
    scaled_polylines = _scale_polylines(polylines, new_w / max(src_w, 1), new_h / max(src_h, 1))
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    padded = _constant_pad_image(resized, pad_top=pad_top, pad_bottom=pad_bottom, pad_left=pad_left, pad_right=pad_right)
    shifted = [polyline + np.asarray([pad_left, pad_top], dtype=np.float32) for polyline in scaled_polylines]
    return padded, shifted


def letterbox_image_mask_and_polylines(
    image: np.ndarray,
    mask: np.ndarray,
    polylines: List[np.ndarray],
    output_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    if mask.ndim == 2:
        mask = mask[..., None]
    target_h, target_w = output_size
    src_h, src_w = image.shape[:2]
    scale = min(target_h / max(src_h, 1), target_w / max(src_w, 1))
    new_h = max(1, int(round(src_h * scale)))
    new_w = max(1, int(round(src_w * scale)))
    resized = _resize_image(image, new_h, new_w)
    resized_mask = _resize_mask(mask, new_h, new_w)
    scaled_polylines = _scale_polylines(polylines, new_w / max(src_w, 1), new_h / max(src_h, 1))
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    padded = _constant_pad_image(resized, pad_top=pad_top, pad_bottom=pad_bottom, pad_left=pad_left, pad_right=pad_right)
    padded_mask = np.pad(resized_mask, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0.0)
    shifted = [polyline + np.asarray([pad_left, pad_top], dtype=np.float32) for polyline in scaled_polylines]
    return padded, (padded_mask > 0.5).astype(np.float32), shifted


def build_targets_from_polylines(
    polylines: List[np.ndarray],
    image_shape: Tuple[int, int],
    target_degree: int,
    min_curve_length: float,
    max_targets: int,
) -> Dict[str, np.ndarray]:
    height, width = image_shape
    curves: List[np.ndarray] = []
    lengths: List[float] = []
    boxes: List[List[float]] = []
    curvatures: List[float] = []
    norm_lengths: List[float] = []
    image_diag = float(max(np.hypot(max(width - 1, 1), max(height - 1, 1)), 1.0))

    for polyline in polylines:
        points = np.asarray(polyline, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 2:
            continue
        control_points = fit_polyline_to_bezier(points, degree=target_degree)
        seg_length = curve_length(control_points)
        if seg_length < min_curve_length:
            continue
        normalized = np.clip(_normalize_xy_control_points(control_points, width, height), 0.0, 1.0)
        curves.append(normalized)
        lengths.append(seg_length)
        norm_lengths.append(seg_length / image_diag)
        boxes.append(list(control_point_bbox(normalized)))
        curvatures.append(polyline_curvature_score(points))

    if curves:
        order = np.argsort(np.asarray(lengths))[::-1][:max_targets]
        curve_array = np.stack([curves[idx] for idx in order]).astype(np.float32)
        length_array = np.asarray([lengths[idx] for idx in order], dtype=np.float32)
        box_array = np.asarray([boxes[idx] for idx in order], dtype=np.float32)
        curvature_array = np.asarray([curvatures[idx] for idx in order], dtype=np.float32)
        norm_length_array = np.asarray([norm_lengths[idx] for idx in order], dtype=np.float32)
    else:
        curve_array = np.zeros((0, target_degree + 1, 2), dtype=np.float32)
        length_array = np.zeros((0,), dtype=np.float32)
        box_array = np.zeros((0, 4), dtype=np.float32)
        curvature_array = np.zeros((0,), dtype=np.float32)
        norm_length_array = np.zeros((0,), dtype=np.float32)

    return {
        'curves': curve_array,
        'curve_lengths': length_array,
        'curve_boxes': box_array,
        'curve_curvatures': curvature_array,
        'curve_norm_lengths': norm_length_array,
        'image_size': np.asarray([height, width], dtype=np.int64),
    }


def build_targets_from_curves(
    curves_px: np.ndarray,
    image_shape: Tuple[int, int],
    max_targets: int,
) -> Dict[str, np.ndarray]:
    height, width = image_shape
    control_count = int(curves_px.shape[1]) if np.asarray(curves_px).ndim == 3 and curves_px.shape[1] > 0 else 6
    curves: List[np.ndarray] = []
    lengths: List[float] = []
    boxes: List[List[float]] = []
    curvatures: List[float] = []
    norm_lengths: List[float] = []
    image_diag = float(max(np.hypot(max(width - 1, 1), max(height - 1, 1)), 1.0))
    xmin = 0.0
    ymin = 0.0
    xmax = float(max(width - 1, 0))
    ymax = float(max(height - 1, 0))

    for control_points in np.asarray(curves_px, dtype=np.float32):
        if control_points.ndim != 2 or control_points.shape[0] < 2:
            continue
        min_x, min_y = np.min(control_points, axis=0)
        max_x, max_y = np.max(control_points, axis=0)
        if max_x < xmin or min_x > xmax or max_y < ymin or min_y > ymax:
            continue
        seg_length = curve_length(control_points)
        normalized = _normalize_xy_control_points(control_points, width, height)
        curves.append(normalized)
        lengths.append(seg_length)
        norm_lengths.append(seg_length / image_diag)
        boxes.append(list(control_point_bbox(normalized)))
        sampled = sample_bezier_numpy(control_points, num_samples=48)
        curvatures.append(polyline_curvature_score(sampled))

    if curves:
        order = np.argsort(np.asarray(lengths))[::-1][:max_targets]
        curve_array = np.stack([curves[idx] for idx in order]).astype(np.float32)
        length_array = np.asarray([lengths[idx] for idx in order], dtype=np.float32)
        box_array = np.asarray([boxes[idx] for idx in order], dtype=np.float32)
        curvature_array = np.asarray([curvatures[idx] for idx in order], dtype=np.float32)
        norm_length_array = np.asarray([norm_lengths[idx] for idx in order], dtype=np.float32)
    else:
        curve_array = np.zeros((0, control_count, 2), dtype=np.float32)
        length_array = np.zeros((0,), dtype=np.float32)
        box_array = np.zeros((0, 4), dtype=np.float32)
        curvature_array = np.zeros((0,), dtype=np.float32)
        norm_length_array = np.zeros((0,), dtype=np.float32)

    return {
        'curves': curve_array,
        'curve_lengths': length_array,
        'curve_boxes': box_array,
        'curve_curvatures': curvature_array,
        'curve_norm_lengths': norm_length_array,
        'image_size': np.asarray([height, width], dtype=np.int64),
    }


def clip_and_refit_curves_to_rect(
    curves_px: np.ndarray,
    width: int,
    height: int,
    num_samples: int = 64,
) -> np.ndarray:
    curves_px = np.asarray(curves_px, dtype=np.float32)
    if curves_px.size == 0:
        control_count = int(curves_px.shape[1]) if curves_px.ndim == 3 and curves_px.shape[1] > 0 else 0
        return np.zeros((0, control_count, 2), dtype=np.float32)

    clipped_curves: List[np.ndarray] = []
    degree = int(curves_px.shape[1] - 1)
    control_count = int(curves_px.shape[1])
    for control_points in curves_px:
        if control_points.ndim != 2 or control_points.shape[0] < 2:
            continue
        sampled = sample_bezier_numpy(control_points, num_samples=max(num_samples, control_points.shape[0]))
        clipped_polylines = clip_polylines_to_rect([sampled], width=width, height=height)
        for polyline in clipped_polylines:
            if polyline.shape[0] < 2:
                continue
            refit = fit_polyline_to_bezier(polyline, degree=degree).astype(np.float32)
            clipped_curves.append(refit)

    if clipped_curves:
        return np.stack(clipped_curves).astype(np.float32)
    return np.zeros((0, control_count, 2), dtype=np.float32)


def build_endpoint_targets_from_polylines(
    polylines: List[np.ndarray],
    image_shape: Tuple[int, int],
    dedupe_distance_px: float,
) -> Dict[str, np.ndarray]:
    height, width = image_shape
    endpoints: List[np.ndarray] = []
    for polyline in polylines:
        points = np.asarray(polyline, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 2:
            continue
        endpoints.append(points[0])
        endpoints.append(points[-1])

    if endpoints:
        scale = np.asarray([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)
        point_array = _dedupe_point_cloud(endpoints, dedupe_distance_px=dedupe_distance_px) / scale[None, :]
        point_array = np.clip(point_array, 0.0, 1.0)
    else:
        point_array = np.zeros((0, 2), dtype=np.float32)

    return {
        'points': point_array.astype(np.float32),
        'image_size': np.asarray([height, width], dtype=np.int64),
    }


def prepare_training_sample(
    image: np.ndarray,
    polylines: List[np.ndarray],
    image_size,
    target_degree: int,
    min_curve_length: float,
    max_targets: int,
    augment_cfg: Dict,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    output_size = _size_tuple(image_size)
    resize_min, resize_max = augment_cfg.get('resize_scale_range', [1.0, 1.25])
    resize_scale = float(rng.uniform(resize_min, resize_max))
    scale_min, scale_max = augment_cfg.get('affine_scale_range', [0.95, 1.05])
    affine_scale = float(rng.uniform(scale_min, scale_max))
    safe_resize_multiplier = resize_scale / max(affine_scale, 1e-6)
    image, polylines = resize_with_aspect_ratio(image, polylines, output_size=output_size, scale_multiplier=safe_resize_multiplier)

    if float(augment_cfg.get('hflip_prob', 0.0)) > 0.0 and rng.random() < float(augment_cfg.get('hflip_prob', 0.0)):
        image, polylines = apply_horizontal_flip(image, polylines)
    if float(augment_cfg.get('vflip_prob', 0.0)) > 0.0 and rng.random() < float(augment_cfg.get('vflip_prob', 0.0)):
        image, polylines = apply_vertical_flip(image, polylines)

    max_angle = float(augment_cfg.get('affine_max_rotate_deg', 0.0))
    angle = float(rng.uniform(-max_angle, max_angle)) if max_angle > 0.0 else 0.0
    translate_ratio = float(augment_cfg.get('affine_max_translate_ratio', 0.0))
    translate_x = float(rng.uniform(-translate_ratio, translate_ratio) * image.shape[1])
    translate_y = float(rng.uniform(-translate_ratio, translate_ratio) * image.shape[0])
    matrix_xy = build_centered_affine_matrix(image.shape[1], image.shape[0], angle_deg=angle, scale=affine_scale, translate_xy=(translate_x, translate_y))
    image = apply_affine_to_image(image, matrix_xy)
    polylines = apply_affine_to_polylines(polylines, matrix_xy)

    image, polylines = random_crop_image_and_polylines(image, polylines, crop_size=output_size, rng=rng)
    targets = build_targets_from_polylines(polylines, image_shape=image.shape[:2], target_degree=target_degree, min_curve_length=min_curve_length, max_targets=max_targets)
    return image, targets


def prepare_eval_sample(
    image: np.ndarray,
    polylines: List[np.ndarray],
    image_size,
    target_degree: int,
    min_curve_length: float,
    max_targets: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    output_size = _size_tuple(image_size)
    image, polylines = letterbox_image_and_polylines(image, polylines, output_size=output_size)
    targets = build_targets_from_polylines(polylines, image_shape=image.shape[:2], target_degree=target_degree, min_curve_length=min_curve_length, max_targets=max_targets)
    return image, targets


def prepare_training_sample_with_mask(
    image: np.ndarray,
    mask: np.ndarray,
    polylines: List[np.ndarray],
    image_size,
    target_degree: int,
    min_curve_length: float,
    max_targets: int,
    augment_cfg: Dict,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    output_size = _size_tuple(image_size)
    resize_min, resize_max = augment_cfg.get('resize_scale_range', [1.0, 1.25])
    resize_scale = float(rng.uniform(resize_min, resize_max))
    scale_min, scale_max = augment_cfg.get('affine_scale_range', [0.95, 1.05])
    affine_scale = float(rng.uniform(scale_min, scale_max))
    safe_resize_multiplier = resize_scale / max(affine_scale, 1e-6)
    image, polylines = resize_with_aspect_ratio(image, polylines, output_size=output_size, scale_multiplier=safe_resize_multiplier)
    resized_h, resized_w = image.shape[:2]
    mask = _resize_mask(mask if mask.ndim == 3 else mask[..., None], resized_h, resized_w)

    if float(augment_cfg.get('hflip_prob', 0.0)) > 0.0 and rng.random() < float(augment_cfg.get('hflip_prob', 0.0)):
        image, polylines = apply_horizontal_flip(image, polylines)
        mask = mask[:, ::-1].copy()
    if float(augment_cfg.get('vflip_prob', 0.0)) > 0.0 and rng.random() < float(augment_cfg.get('vflip_prob', 0.0)):
        image, polylines = apply_vertical_flip(image, polylines)
        mask = mask[::-1].copy()

    max_angle = float(augment_cfg.get('affine_max_rotate_deg', 0.0))
    angle = float(rng.uniform(-max_angle, max_angle)) if max_angle > 0.0 else 0.0
    translate_ratio = float(augment_cfg.get('affine_max_translate_ratio', 0.0))
    translate_x = float(rng.uniform(-translate_ratio, translate_ratio) * image.shape[1])
    translate_y = float(rng.uniform(-translate_ratio, translate_ratio) * image.shape[0])
    matrix_xy = build_centered_affine_matrix(image.shape[1], image.shape[0], angle_deg=angle, scale=affine_scale, translate_xy=(translate_x, translate_y))
    image = apply_affine_to_image(image, matrix_xy)
    mask = apply_affine_to_mask(mask, matrix_xy)
    polylines = apply_affine_to_polylines(polylines, matrix_xy)

    image, mask, polylines = random_crop_image_mask_and_polylines(image, mask, polylines, crop_size=output_size, rng=rng)
    targets = build_targets_from_polylines(polylines, image_shape=image.shape[:2], target_degree=target_degree, min_curve_length=min_curve_length, max_targets=max_targets)
    return image, (mask > 0.5).astype(np.float32), targets


def prepare_training_endpoint_sample_with_mask(
    image: np.ndarray,
    mask: np.ndarray,
    curves: np.ndarray,
    image_size,
    augment_cfg: Dict,
    rng: np.random.Generator,
    dedupe_distance_px: float,
    max_targets: int = 512,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    output_size = _size_tuple(image_size)
    src_h, src_w = image.shape[:2]
    points_px = _curve_endpoint_points_px(curves, src_w, src_h)

    resize_min, resize_max = augment_cfg.get('resize_scale_range', [1.0, 1.25])
    resize_scale = float(rng.uniform(resize_min, resize_max))
    scale_min, scale_max = augment_cfg.get('affine_scale_range', [0.95, 1.05])
    affine_scale = float(rng.uniform(scale_min, scale_max))
    safe_resize_multiplier = resize_scale / max(affine_scale, 1e-6)
    image, _ = resize_with_aspect_ratio(image, [], output_size=output_size, scale_multiplier=safe_resize_multiplier)
    resized_h, resized_w = image.shape[:2]
    mask = _resize_mask(mask if mask.ndim == 3 else mask[..., None], resized_h, resized_w)
    points_px = _scale_points(points_px, resized_w / max(src_w, 1), resized_h / max(src_h, 1))

    if float(augment_cfg.get('hflip_prob', 0.0)) > 0.0 and rng.random() < float(augment_cfg.get('hflip_prob', 0.0)):
        image = image[:, ::-1].copy()
        mask = mask[:, ::-1].copy()
        points_px = _apply_horizontal_flip_to_points(points_px, image.shape[1])
    if float(augment_cfg.get('vflip_prob', 0.0)) > 0.0 and rng.random() < float(augment_cfg.get('vflip_prob', 0.0)):
        image = image[::-1].copy()
        mask = mask[::-1].copy()
        points_px = _apply_vertical_flip_to_points(points_px, image.shape[0])

    max_angle = float(augment_cfg.get('affine_max_rotate_deg', 0.0))
    angle = float(rng.uniform(-max_angle, max_angle)) if max_angle > 0.0 else 0.0
    translate_ratio = float(augment_cfg.get('affine_max_translate_ratio', 0.0))
    translate_x = float(rng.uniform(-translate_ratio, translate_ratio) * image.shape[1])
    translate_y = float(rng.uniform(-translate_ratio, translate_ratio) * image.shape[0])
    matrix_xy = build_centered_affine_matrix(image.shape[1], image.shape[0], angle_deg=angle, scale=affine_scale, translate_xy=(translate_x, translate_y))
    image = apply_affine_to_image(image, matrix_xy)
    mask = apply_affine_to_mask(mask, matrix_xy)
    points_px = _apply_affine_to_points(points_px, matrix_xy)

    image, mask = _pad_image_mask_and_polylines_for_crop(image, mask if mask.ndim == 3 else mask[..., None], [], crop_h=output_size[0], crop_w=output_size[1])[:2]
    pad_h = image.shape[0] - resized_h
    pad_w = image.shape[1] - resized_w
    if points_px.size:
        points_px = points_px + np.asarray([pad_w // 2, pad_h // 2], dtype=np.float32)[None, :]
    image, mask, (crop_left, crop_top) = random_crop_image_and_mask(image, mask, crop_size=output_size, rng=rng)
    if points_px.size:
        points_px = points_px - np.asarray([crop_left, crop_top], dtype=np.float32)[None, :]
    targets = _build_endpoint_targets_from_points(
        points_px,
        width=output_size[1],
        height=output_size[0],
        dedupe_distance_px=dedupe_distance_px,
    )
    return image, (mask > 0.5).astype(np.float32), targets


def prepare_training_endpoint_sample(
    image: np.ndarray,
    curves: np.ndarray,
    image_size,
    augment_cfg: Dict,
    rng: np.random.Generator,
    dedupe_distance_px: float,
    max_targets: int = 512,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    output_size = _size_tuple(image_size)
    src_h, src_w = image.shape[:2]
    points_px = _curve_endpoint_points_px(curves, src_w, src_h)

    resize_min, resize_max = augment_cfg.get('resize_scale_range', [1.0, 1.25])
    resize_scale = float(rng.uniform(resize_min, resize_max))
    scale_min, scale_max = augment_cfg.get('affine_scale_range', [0.95, 1.05])
    affine_scale = float(rng.uniform(scale_min, scale_max))
    safe_resize_multiplier = resize_scale / max(affine_scale, 1e-6)
    image, _ = resize_with_aspect_ratio(image, [], output_size=output_size, scale_multiplier=safe_resize_multiplier)
    resized_h, resized_w = image.shape[:2]
    points_px = _scale_points(points_px, resized_w / max(src_w, 1), resized_h / max(src_h, 1))

    if float(augment_cfg.get('hflip_prob', 0.0)) > 0.0 and rng.random() < float(augment_cfg.get('hflip_prob', 0.0)):
        image = image[:, ::-1].copy()
        points_px = _apply_horizontal_flip_to_points(points_px, image.shape[1])
    if float(augment_cfg.get('vflip_prob', 0.0)) > 0.0 and rng.random() < float(augment_cfg.get('vflip_prob', 0.0)):
        image = image[::-1].copy()
        points_px = _apply_vertical_flip_to_points(points_px, image.shape[0])

    max_angle = float(augment_cfg.get('affine_max_rotate_deg', 0.0))
    angle = float(rng.uniform(-max_angle, max_angle)) if max_angle > 0.0 else 0.0
    translate_ratio = float(augment_cfg.get('affine_max_translate_ratio', 0.0))
    translate_x = float(rng.uniform(-translate_ratio, translate_ratio) * image.shape[1])
    translate_y = float(rng.uniform(-translate_ratio, translate_ratio) * image.shape[0])
    matrix_xy = build_centered_affine_matrix(image.shape[1], image.shape[0], angle_deg=angle, scale=affine_scale, translate_xy=(translate_x, translate_y))
    image = apply_affine_to_image(image, matrix_xy)
    points_px = _apply_affine_to_points(points_px, matrix_xy)

    crop_h, crop_w = output_size
    height, width = image.shape[:2]
    if height < crop_h or width < crop_w:
        pad_h = max(crop_h - height, 0)
        pad_w = max(crop_w - width, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='reflect')
        if points_px.size:
            points_px = points_px + np.asarray([pad_left, pad_top], dtype=np.float32)[None, :]
        height, width = image.shape[:2]
    max_top = height - crop_h
    max_left = width - crop_w
    crop_top = int(rng.integers(0, max_top + 1)) if max_top > 0 else 0
    crop_left = int(rng.integers(0, max_left + 1)) if max_left > 0 else 0
    image = image[crop_top:crop_top + crop_h, crop_left:crop_left + crop_w].copy()
    if points_px.size:
        points_px = points_px - np.asarray([crop_left, crop_top], dtype=np.float32)[None, :]
    targets = _build_endpoint_targets_from_points(
        points_px,
        width=output_size[1],
        height=output_size[0],
        dedupe_distance_px=dedupe_distance_px,
    )
    return image, targets


def prepare_training_curve_sample_with_mask(
    image: np.ndarray,
    mask: np.ndarray,
    curves: np.ndarray,
    image_size,
    max_targets: int,
    augment_cfg: Dict,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    output_size = _size_tuple(image_size)
    src_h, src_w = image.shape[:2]
    curves_px = np.asarray(
        [denormalize_control_points(control_points, src_w, src_h) for control_points in np.asarray(curves, dtype=np.float32)],
        dtype=np.float32,
    ) if np.asarray(curves).size else np.zeros((0, 0, 2), dtype=np.float32)

    resize_min, resize_max = augment_cfg.get('resize_scale_range', [1.0, 1.25])
    resize_scale = float(rng.uniform(resize_min, resize_max))
    scale_min, scale_max = augment_cfg.get('affine_scale_range', [0.95, 1.05])
    affine_scale = float(rng.uniform(scale_min, scale_max))
    safe_resize_multiplier = resize_scale / max(affine_scale, 1e-6)
    image, _ = resize_with_aspect_ratio(image, [], output_size=output_size, scale_multiplier=safe_resize_multiplier)
    resized_h, resized_w = image.shape[:2]
    mask = _resize_mask(mask if mask.ndim == 3 else mask[..., None], resized_h, resized_w)
    curves_px = _scale_curves(curves_px, resized_w / max(src_w, 1), resized_h / max(src_h, 1))

    if float(augment_cfg.get('hflip_prob', 0.0)) > 0.0 and rng.random() < float(augment_cfg.get('hflip_prob', 0.0)):
        image = image[:, ::-1].copy()
        mask = mask[:, ::-1].copy()
        curves_px = apply_horizontal_flip_to_curves(curves_px, image.shape[1])
    if float(augment_cfg.get('vflip_prob', 0.0)) > 0.0 and rng.random() < float(augment_cfg.get('vflip_prob', 0.0)):
        image = image[::-1].copy()
        mask = mask[::-1].copy()
        curves_px = apply_vertical_flip_to_curves(curves_px, image.shape[0])

    max_angle = float(augment_cfg.get('affine_max_rotate_deg', 0.0))
    angle = float(rng.uniform(-max_angle, max_angle)) if max_angle > 0.0 else 0.0
    translate_ratio = float(augment_cfg.get('affine_max_translate_ratio', 0.0))
    translate_x = float(rng.uniform(-translate_ratio, translate_ratio) * image.shape[1])
    translate_y = float(rng.uniform(-translate_ratio, translate_ratio) * image.shape[0])
    matrix_xy = build_centered_affine_matrix(image.shape[1], image.shape[0], angle_deg=angle, scale=affine_scale, translate_xy=(translate_x, translate_y))
    image = apply_affine_to_image(image, matrix_xy)
    mask = apply_affine_to_mask(mask, matrix_xy)
    curves_px = apply_affine_to_curves(curves_px, matrix_xy)

    image, mask = _pad_image_mask_and_polylines_for_crop(image, mask if mask.ndim == 3 else mask[..., None], [], crop_h=output_size[0], crop_w=output_size[1])[:2]
    pad_h = image.shape[0] - resized_h
    pad_w = image.shape[1] - resized_w
    if curves_px.size:
        curves_px = curves_px + np.asarray([pad_w // 2, pad_h // 2], dtype=np.float32)[None, None, :]
    image, mask, (crop_left, crop_top) = random_crop_image_and_mask(image, mask, crop_size=output_size, rng=rng)
    if curves_px.size:
        curves_px = curves_px - np.asarray([crop_left, crop_top], dtype=np.float32)[None, None, :]
    targets = build_targets_from_curves(curves_px, image_shape=image.shape[:2], max_targets=max_targets)
    return image, (mask > 0.5).astype(np.float32), targets


def prepare_training_curve_sample(
    image: np.ndarray,
    curves: np.ndarray,
    image_size,
    max_targets: int,
    augment_cfg: Dict,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    output_size = _size_tuple(image_size)
    src_h, src_w = image.shape[:2]
    curves_px = np.asarray(
        [denormalize_control_points(control_points, src_w, src_h) for control_points in np.asarray(curves, dtype=np.float32)],
        dtype=np.float32,
    ) if np.asarray(curves).size else np.zeros((0, 0, 2), dtype=np.float32)

    resize_min, resize_max = augment_cfg.get('resize_scale_range', [1.0, 1.25])
    resize_scale = float(rng.uniform(resize_min, resize_max))
    scale_min, scale_max = augment_cfg.get('affine_scale_range', [0.95, 1.05])
    affine_scale = float(rng.uniform(scale_min, scale_max))
    safe_resize_multiplier = resize_scale / max(affine_scale, 1e-6)
    image, _ = resize_with_aspect_ratio(image, [], output_size=output_size, scale_multiplier=safe_resize_multiplier)
    resized_h, resized_w = image.shape[:2]
    curves_px = _scale_curves(curves_px, resized_w / max(src_w, 1), resized_h / max(src_h, 1))

    if float(augment_cfg.get('hflip_prob', 0.0)) > 0.0 and rng.random() < float(augment_cfg.get('hflip_prob', 0.0)):
        image = image[:, ::-1].copy()
        curves_px = apply_horizontal_flip_to_curves(curves_px, image.shape[1])
    if float(augment_cfg.get('vflip_prob', 0.0)) > 0.0 and rng.random() < float(augment_cfg.get('vflip_prob', 0.0)):
        image = image[::-1].copy()
        curves_px = apply_vertical_flip_to_curves(curves_px, image.shape[0])

    max_angle = float(augment_cfg.get('affine_max_rotate_deg', 0.0))
    angle = float(rng.uniform(-max_angle, max_angle)) if max_angle > 0.0 else 0.0
    translate_ratio = float(augment_cfg.get('affine_max_translate_ratio', 0.0))
    translate_x = float(rng.uniform(-translate_ratio, translate_ratio) * image.shape[1])
    translate_y = float(rng.uniform(-translate_ratio, translate_ratio) * image.shape[0])
    matrix_xy = build_centered_affine_matrix(image.shape[1], image.shape[0], angle_deg=angle, scale=affine_scale, translate_xy=(translate_x, translate_y))
    image = apply_affine_to_image(image, matrix_xy)
    curves_px = apply_affine_to_curves(curves_px, matrix_xy)

    crop_h, crop_w = output_size
    height, width = image.shape[:2]
    if height < crop_h or width < crop_w:
        pad_h = max(crop_h - height, 0)
        pad_w = max(crop_w - width, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='reflect')
        if curves_px.size:
            curves_px = curves_px + np.asarray([pad_left, pad_top], dtype=np.float32)[None, None, :]
        height, width = image.shape[:2]
    max_top = height - crop_h
    max_left = width - crop_w
    crop_top = int(rng.integers(0, max_top + 1)) if max_top > 0 else 0
    crop_left = int(rng.integers(0, max_left + 1)) if max_left > 0 else 0
    image = image[crop_top:crop_top + crop_h, crop_left:crop_left + crop_w].copy()
    if curves_px.size:
        curves_px = curves_px - np.asarray([crop_left, crop_top], dtype=np.float32)[None, None, :]
    targets = build_targets_from_curves(curves_px, image_shape=image.shape[:2], max_targets=max_targets)
    return image, targets


def prepare_eval_endpoint_sample_with_mask(
    image: np.ndarray,
    mask: np.ndarray,
    curves: np.ndarray,
    image_size,
    dedupe_distance_px: float,
    max_targets: int = 512,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    output_size = _size_tuple(image_size)
    src_h, src_w = image.shape[:2]
    points_px = _curve_endpoint_points_px(curves, src_w, src_h)

    image, mask, _ = letterbox_image_mask_and_polylines(image, mask, [], output_size=output_size)
    scale = min(output_size[0] / max(src_h, 1), output_size[1] / max(src_w, 1))
    new_h = max(1, int(round(src_h * scale)))
    new_w = max(1, int(round(src_w * scale)))
    pad_top = (output_size[0] - new_h) // 2
    pad_left = (output_size[1] - new_w) // 2
    points_px = _scale_points(points_px, new_w / max(src_w, 1), new_h / max(src_h, 1))
    if points_px.size:
        points_px = points_px + np.asarray([pad_left, pad_top], dtype=np.float32)[None, :]
    targets = _build_endpoint_targets_from_points(
        points_px,
        width=output_size[1],
        height=output_size[0],
        dedupe_distance_px=dedupe_distance_px,
    )
    return image, (mask > 0.5).astype(np.float32), targets


def prepare_eval_endpoint_sample(
    image: np.ndarray,
    curves: np.ndarray,
    image_size,
    dedupe_distance_px: float,
    max_targets: int = 512,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    output_size = _size_tuple(image_size)
    src_h, src_w = image.shape[:2]
    points_px = _curve_endpoint_points_px(curves, src_w, src_h)

    image, _ = letterbox_image_and_polylines(image, [], output_size=output_size)
    scale = min(output_size[0] / max(src_h, 1), output_size[1] / max(src_w, 1))
    new_h = max(1, int(round(src_h * scale)))
    new_w = max(1, int(round(src_w * scale)))
    pad_top = (output_size[0] - new_h) // 2
    pad_left = (output_size[1] - new_w) // 2
    points_px = _scale_points(points_px, new_w / max(src_w, 1), new_h / max(src_h, 1))
    if points_px.size:
        points_px = points_px + np.asarray([pad_left, pad_top], dtype=np.float32)[None, :]
    targets = _build_endpoint_targets_from_points(
        points_px,
        width=output_size[1],
        height=output_size[0],
        dedupe_distance_px=dedupe_distance_px,
    )
    return image, targets


def prepare_eval_curve_sample_with_mask(
    image: np.ndarray,
    mask: np.ndarray,
    curves: np.ndarray,
    image_size,
    max_targets: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    output_size = _size_tuple(image_size)
    src_h, src_w = image.shape[:2]
    curves_px = np.asarray(
        [denormalize_control_points(control_points, src_w, src_h) for control_points in np.asarray(curves, dtype=np.float32)],
        dtype=np.float32,
    ) if np.asarray(curves).size else np.zeros((0, 0, 2), dtype=np.float32)

    image, mask, _ = letterbox_image_mask_and_polylines(image, mask, [], output_size=output_size)
    scale = min(output_size[0] / max(src_h, 1), output_size[1] / max(src_w, 1))
    new_h = max(1, int(round(src_h * scale)))
    new_w = max(1, int(round(src_w * scale)))
    pad_top = (output_size[0] - new_h) // 2
    pad_left = (output_size[1] - new_w) // 2
    curves_px = _scale_curves(curves_px, new_w / max(src_w, 1), new_h / max(src_h, 1))
    if curves_px.size:
        curves_px = curves_px + np.asarray([pad_left, pad_top], dtype=np.float32)[None, None, :]
    targets = build_targets_from_curves(curves_px, image_shape=image.shape[:2], max_targets=max_targets)
    return image, (mask > 0.5).astype(np.float32), targets


def prepare_eval_curve_sample(
    image: np.ndarray,
    curves: np.ndarray,
    image_size,
    max_targets: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    output_size = _size_tuple(image_size)
    src_h, src_w = image.shape[:2]
    curves_px = np.asarray(
        [denormalize_control_points(control_points, src_w, src_h) for control_points in np.asarray(curves, dtype=np.float32)],
        dtype=np.float32,
    ) if np.asarray(curves).size else np.zeros((0, 0, 2), dtype=np.float32)

    image, _ = letterbox_image_and_polylines(image, [], output_size=output_size)
    scale = min(output_size[0] / max(src_h, 1), output_size[1] / max(src_w, 1))
    new_h = max(1, int(round(src_h * scale)))
    new_w = max(1, int(round(src_w * scale)))
    pad_top = (output_size[0] - new_h) // 2
    pad_left = (output_size[1] - new_w) // 2
    curves_px = _scale_curves(curves_px, new_w / max(src_w, 1), new_h / max(src_h, 1))
    if curves_px.size:
        curves_px = curves_px + np.asarray([pad_left, pad_top], dtype=np.float32)[None, None, :]
    targets = build_targets_from_curves(curves_px, image_shape=image.shape[:2], max_targets=max_targets)
    return image, targets


def prepare_eval_sample_with_mask(
    image: np.ndarray,
    mask: np.ndarray,
    polylines: List[np.ndarray],
    image_size,
    target_degree: int,
    min_curve_length: float,
    max_targets: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    output_size = _size_tuple(image_size)
    image, mask, polylines = letterbox_image_mask_and_polylines(image, mask, polylines, output_size=output_size)
    targets = build_targets_from_polylines(polylines, image_shape=image.shape[:2], target_degree=target_degree, min_curve_length=min_curve_length, max_targets=max_targets)
    return image, (mask > 0.5).astype(np.float32), targets
