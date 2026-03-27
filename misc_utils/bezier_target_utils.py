import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from scipy import sparse

from bezierization.ablation_api import run_version

SUPPORTED_INPUT_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
TARGET_CACHE_FORMAT_VERSION = 'bezier_refit_v7_xy_deg5_curvature_normlen_polarityfix'
GRAPH_CACHE_FORMAT_VERSION = 'graph_polyline_v1_segments_xy'


def rc_to_xy(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return points[..., [1, 0]]


def xy_to_rc(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return points[..., [1, 0]]


def image_id_from_stem(stem: str) -> str:
    return stem.split('_ann')[0]


def resolve_input_path(edge_path: Path, input_root: Optional[Union[Path, Sequence[Path]]] = None) -> Path:
    if input_root is None:
        return edge_path
    if isinstance(input_root, (str, Path)):
        input_roots = [Path(input_root)]
    else:
        input_roots = [Path(root) for root in input_root]
    stem = edge_path.stem
    image_id = image_id_from_stem(stem)
    candidates = []
    for root in input_roots:
        for suffix in SUPPORTED_INPUT_SUFFIXES:
            candidates.extend([
                root / f'{stem}{suffix}',
                root / f'{image_id}{suffix}',
                root / edge_path.name.replace(edge_path.suffix, suffix),
            ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'Could not resolve input image for {edge_path} under {input_roots}')


def bernstein_basis_numpy(degree: int, t_values: np.ndarray) -> np.ndarray:
    basis = []
    for i in range(degree + 1):
        coeff = float(__import__('math').comb(degree, i))
        basis.append(coeff * (1.0 - t_values) ** (degree - i) * t_values ** i)
    return np.stack(basis, axis=1).astype(np.float32)


def resample_polyline_by_arclength(points: np.ndarray, num_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f'Expected [N, 2] polyline, got {points.shape}')
    if points.shape[0] == 0:
        raise ValueError('Cannot resample empty polyline')
    if points.shape[0] == 1:
        return np.repeat(points, num_points, axis=0)
    if points.shape[0] == num_points:
        return points.astype(np.float32)

    deltas = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cumulative[-1]
    if total <= 1e-6:
        return np.repeat(points[:1], num_points, axis=0)

    samples = np.linspace(0.0, total, num_points, dtype=np.float32)
    resampled: List[np.ndarray] = []
    seg_idx = 0
    for sample_pos in samples:
        while seg_idx < len(seg_lengths) - 1 and cumulative[seg_idx + 1] < sample_pos:
            seg_idx += 1
        seg_start = cumulative[seg_idx]
        seg_length = max(seg_lengths[seg_idx], 1e-6)
        alpha = (sample_pos - seg_start) / seg_length
        alpha = float(np.clip(alpha, 0.0, 1.0))
        point = points[seg_idx] * (1.0 - alpha) + points[seg_idx + 1] * alpha
        resampled.append(point.astype(np.float32))
    return np.stack(resampled, axis=0).astype(np.float32)


def fit_polyline_to_bezier(points: np.ndarray, degree: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f'Expected [N, 2] polyline, got {points.shape}')
    if points.shape[0] == 0:
        raise ValueError('Cannot fit empty polyline')
    if points.shape[0] == 1:
        return np.repeat(points, degree + 1, axis=0)
    p0 = points[0]
    pN = points[-1]
    if points.shape[0] == 2:
        return np.linspace(p0, pN, degree + 1, dtype=np.float32)
    if points.shape[0] < degree + 1:
        # Cropped polylines can collapse to very few points. Resampling them to the
        # target count before fitting avoids underconstrained high-degree fits that
        # produce spurious sharp curves or control points far outside the image.
        points = resample_polyline_by_arclength(points, degree + 1)
        p0 = points[0]
        pN = points[-1]

    deltas = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cumulative[-1]
    if total <= 1e-6:
        return np.repeat(points[:1], degree + 1, axis=0)
    t_values = (cumulative / total).astype(np.float32)
    basis = bernstein_basis_numpy(degree, t_values)
    rhs = points - basis[:, [0]] * p0 - basis[:, [-1]] * pN
    if degree > 1:
        design = basis[:, 1:-1]
        inner, _, _, _ = np.linalg.lstsq(design, rhs, rcond=None)
        control_points = np.concatenate([p0[None], inner, pN[None]], axis=0)
    else:
        control_points = np.stack([p0, pN], axis=0)
    return control_points.astype(np.float32)


def normalize_control_points(control_points: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = np.array([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)
    return rc_to_xy(control_points) / scale


def denormalize_control_points(control_points: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = np.array([max(width - 1, 1), max(height - 1, 1)], dtype=np.float32)
    return np.asarray(control_points, dtype=np.float32) * scale


def sample_bezier_numpy(control_points: np.ndarray, num_samples: int = 48) -> np.ndarray:
    degree = control_points.shape[0] - 1
    t_values = np.linspace(0.0, 1.0, num_samples, dtype=np.float32)
    basis = bernstein_basis_numpy(degree, t_values)
    return basis @ np.asarray(control_points, dtype=np.float32)


def curve_length(control_points: np.ndarray, num_samples: int = 48) -> float:
    points = sample_bezier_numpy(np.asarray(control_points, dtype=np.float32), num_samples=num_samples)
    diffs = np.diff(points, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def polyline_curvature_score(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 3:
        return 0.0
    prev_vec = points[1:-1] - points[:-2]
    next_vec = points[2:] - points[1:-1]
    prev_norm = np.linalg.norm(prev_vec, axis=1)
    next_norm = np.linalg.norm(next_vec, axis=1)
    valid = (prev_norm > 1e-6) & (next_norm > 1e-6)
    if not np.any(valid):
        return 0.0
    cos_angles = np.sum(prev_vec[valid] * next_vec[valid], axis=1) / (prev_norm[valid] * next_norm[valid])
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    turn_angles = np.abs(np.arccos(cos_angles))
    path_len = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    if path_len <= 1e-6:
        return 0.0
    return float(turn_angles.sum() / path_len)


def control_point_bbox(control_points: np.ndarray) -> Tuple[float, float, float, float]:
    mins = np.min(control_points, axis=0)
    maxs = np.max(control_points, axis=0)
    return float(mins[0]), float(mins[1]), float(maxs[0]), float(maxs[1])


def build_cache_key(edge_path: Path, version_name: str, target_degree: int, min_curve_length: float) -> str:
    payload = {
        'edge_path': str(edge_path.resolve()),
        'version_name': version_name,
        'target_degree': target_degree,
        'min_curve_length': min_curve_length,
        'format_version': TARGET_CACHE_FORMAT_VERSION,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()[:16]


def build_graph_cache_key(edge_path: Path, version_name: str) -> str:
    payload = {
        'edge_path': str(edge_path.resolve()),
        'version_name': version_name,
        'format_version': GRAPH_CACHE_FORMAT_VERSION,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()[:16]


def load_binary_edge_annotation(edge_path: Path) -> np.ndarray:
    edge_path = Path(edge_path)
    if edge_path.suffix == '.npz':
        data = np.load(edge_path, allow_pickle=False)
        if {'indices', 'indptr', 'data', 'shape'}.issubset(data.files):
            matrix = sparse.csr_matrix((data['data'], data['indices'], data['indptr']), shape=tuple(int(value) for value in data['shape']))
            edge_mask = (matrix.toarray() > 0).astype(np.uint8)
            return (edge_mask * 255).astype(np.uint8)
        raise ValueError(f'Unsupported npz edge annotation format at {edge_path}')
    image = Image.open(edge_path).convert('L')
    array = np.asarray(image, dtype=np.uint8)
    high_mask = (array > 127).astype(np.uint8)
    high_ratio = float(high_mask.mean())
    # Treat the minority class as the edge foreground so both
    # black-on-white and white-on-black annotations are handled.
    edge_mask = high_mask if high_ratio <= 0.5 else (1 - high_mask)
    return (edge_mask * 255).astype(np.uint8)


def extract_cubic_targets(edge_path: Path, version_name: str, target_degree: int, min_curve_length: float) -> Dict:
    edge_binary = load_binary_edge_annotation(edge_path)
    result = run_version(
        version_name,
        image_array=edge_binary,
        output_dir=None,
        compute_raster=False,
        compute_summary=False,
        compute_metrics=False,
        include_debug_artifacts=False,
    )
    height, width = edge_binary.shape[:2]
    curves: List[np.ndarray] = []
    lengths: List[float] = []
    boxes: List[List[float]] = []
    source_degrees: List[int] = []
    curvature_scores: List[float] = []
    norm_lengths: List[float] = []
    image_diag = float(max(np.hypot(max(width - 1, 1), max(height - 1, 1)), 1.0))

    for path_fit in result['fitted_paths']:
        for segment in path_fit['segments']:
            polyline = np.asarray(segment.get('points', segment.get('fitted_points')), dtype=np.float32)
            if polyline.ndim != 2 or polyline.shape[0] < 2:
                continue
            control_points = fit_polyline_to_bezier(polyline, degree=target_degree)
            seg_length = curve_length(control_points)
            if seg_length < min_curve_length:
                continue
            curves.append(normalize_control_points(control_points, width, height))
            lengths.append(seg_length)
            norm_lengths.append(seg_length / image_diag)
            boxes.append(list(control_point_bbox(curves[-1])))
            source_degrees.append(int(segment['degree']))
            curvature_scores.append(polyline_curvature_score(polyline))

    if curves:
        order = np.argsort(np.asarray(lengths))[::-1]
        curves = [curves[idx] for idx in order]
        lengths = [lengths[idx] for idx in order]
        boxes = [boxes[idx] for idx in order]
        source_degrees = [source_degrees[idx] for idx in order]
        curvature_scores = [curvature_scores[idx] for idx in order]
        curve_array = np.stack(curves).astype(np.float32)
        length_array = np.asarray(lengths, dtype=np.float32)
        box_array = np.asarray(boxes, dtype=np.float32)
        degree_array = np.asarray(source_degrees, dtype=np.int64)
        curvature_array = np.asarray(curvature_scores, dtype=np.float32)
        norm_length_array = np.asarray(norm_lengths, dtype=np.float32)
    else:
        curve_array = np.zeros((0, target_degree + 1, 2), dtype=np.float32)
        length_array = np.zeros((0,), dtype=np.float32)
        box_array = np.zeros((0, 4), dtype=np.float32)
        degree_array = np.zeros((0,), dtype=np.int64)
        curvature_array = np.zeros((0,), dtype=np.float32)
        norm_length_array = np.zeros((0,), dtype=np.float32)

    return {
        'curves': curve_array,
        'curve_lengths': length_array,
        'curve_boxes': box_array,
        'source_degrees': degree_array,
        'curve_curvatures': curvature_array,
        'curve_norm_lengths': norm_length_array,
        'image_size': np.asarray([height, width], dtype=np.int64),
        'path': str(edge_path),
        'image_id': image_id_from_stem(edge_path.stem),
        'metrics_f1': 0.0,
        'metrics_chamfer': 0.0,
    }


def _collect_graph_polylines(result: Dict) -> List[np.ndarray]:
    polylines: List[np.ndarray] = []
    for path_fit in result['fitted_paths']:
        for segment in path_fit['segments']:
            polyline = np.asarray(segment.get('points', segment.get('fitted_points')), dtype=np.float32)
            if polyline.ndim != 2 or polyline.shape[0] < 2:
                continue
            polylines.append(rc_to_xy(polyline))
    return polylines


def _pack_polylines(polylines: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if not polylines:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((1,), dtype=np.int64)
    offsets = [0]
    packed = []
    for polyline in polylines:
        polyline = np.asarray(polyline, dtype=np.float32)
        packed.append(polyline)
        offsets.append(offsets[-1] + polyline.shape[0])
    return np.concatenate(packed, axis=0).astype(np.float32), np.asarray(offsets, dtype=np.int64)


def unpack_polylines(points: np.ndarray, offsets: np.ndarray) -> List[np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.int64)
    polylines: List[np.ndarray] = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        if end - start < 2:
            continue
        polylines.append(points[start:end].copy())
    return polylines


def extract_graph_segments(edge_path: Path, version_name: str) -> Dict:
    edge_binary = load_binary_edge_annotation(edge_path)
    result = run_version(
        version_name,
        image_array=edge_binary,
        output_dir=None,
        compute_raster=False,
        compute_summary=False,
        compute_metrics=False,
        include_debug_artifacts=False,
    )
    height, width = edge_binary.shape[:2]
    polylines = _collect_graph_polylines(result)
    packed_points, packed_offsets = _pack_polylines(polylines)
    return {
        'graph_points': packed_points,
        'graph_offsets': packed_offsets,
        'image_size': np.asarray([height, width], dtype=np.int64),
        'path': str(edge_path),
        'image_id': image_id_from_stem(edge_path.stem),
    }


def ensure_graph_cache(edge_path: Path, cache_root: Path, version_name: str) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = build_graph_cache_key(edge_path, version_name)
    cache_path = cache_root / f'{edge_path.stem}_{cache_key}_graph.npz'
    if cache_path.exists():
        return cache_path
    payload = extract_graph_segments(edge_path, version_name=version_name)
    np.savez_compressed(cache_path, **payload)
    return cache_path


def load_cached_graph(cache_path: Path) -> Dict:
    data = np.load(cache_path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def ensure_target_cache(edge_path: Path, cache_root: Path, version_name: str, target_degree: int = 5, min_curve_length: float = 3.0) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = build_cache_key(edge_path, version_name, target_degree, min_curve_length)
    cache_path = cache_root / f'{edge_path.stem}_{cache_key}.npz'
    if cache_path.exists():
        return cache_path
    payload = extract_cubic_targets(edge_path, version_name, target_degree=target_degree, min_curve_length=min_curve_length)
    np.savez_compressed(cache_path, **payload)
    return cache_path


def load_cached_targets(cache_path: Path) -> Dict:
    data = np.load(cache_path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def load_image_array(image_path: Path, image_size: int, rgb: bool) -> np.ndarray:
    image = Image.open(image_path)
    image = image.convert('RGB' if rgb else 'L')
    image = image.resize((image_size, image_size), resample=Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[None, ...]
    else:
        array = np.transpose(array, (2, 0, 1))
    return array


def load_image_array_original(image_path: Path, rgb: bool) -> np.ndarray:
    image = Image.open(image_path)
    image = image.convert('RGB' if rgb else 'L')
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[..., None]
    return array
