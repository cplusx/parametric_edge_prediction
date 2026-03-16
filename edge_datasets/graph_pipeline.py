from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

from misc_utils.bezier_target_utils import control_point_bbox, curve_length, fit_polyline_to_bezier, polyline_curvature_score


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


def _scale_polylines(polylines: List[np.ndarray], scale_x: float, scale_y: float) -> List[np.ndarray]:
    scale = np.asarray([scale_x, scale_y], dtype=np.float32)
    return [np.asarray(polyline, dtype=np.float32) * scale for polyline in polylines]


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


def _deduplicate_points(points: List[np.ndarray]) -> np.ndarray:
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    deduped = [np.asarray(points[0], dtype=np.float32)]
    for point in points[1:]:
        point = np.asarray(point, dtype=np.float32)
        if np.linalg.norm(point - deduped[-1]) > 1e-4:
            deduped.append(point)
    return np.stack(deduped, axis=0).astype(np.float32)


def clip_polylines_to_rect(polylines: List[np.ndarray], width: int, height: int) -> List[np.ndarray]:
    clipped_polylines: List[np.ndarray] = []
    for polyline in polylines:
        points = np.asarray(polyline, dtype=np.float32)
        if points.shape[0] < 2:
            continue
        current: List[np.ndarray] = []
        for start, end in zip(points[:-1], points[1:]):
            clipped = _clip_segment_to_rect(start, end, width, height)
            if clipped is None:
                if len(current) >= 2:
                    clipped_polylines.append(_deduplicate_points(current))
                current = []
                continue
            c0, c1 = clipped
            if not current:
                current = [c0, c1]
                continue
            if np.linalg.norm(current[-1] - c0) > 1e-3:
                if len(current) >= 2:
                    clipped_polylines.append(_deduplicate_points(current))
                current = [c0, c1]
            else:
                current.append(c1)
        if len(current) >= 2:
            clipped_polylines.append(_deduplicate_points(current))
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