from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline

from bezierization.bezier_refiner_core import (
    extract_ordered_edge_paths,
    fit_error_metrics,
    path_length,
    split_polyline_by_corners,
)


def _chord_length_parameterize(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= 1:
        return np.zeros(len(points), dtype=np.float64)
    seg_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = float(cumulative[-1])
    if total <= 1e-8:
        return np.linspace(0.0, 1.0, len(points), dtype=np.float64)
    return cumulative / total


def _open_uniform_knots(n_ctrl: int, degree: int) -> np.ndarray:
    interior_count = n_ctrl - degree - 1
    if interior_count > 0:
        interior = np.linspace(0.0, 1.0, interior_count + 2, dtype=np.float64)[1:-1]
    else:
        interior = np.zeros((0,), dtype=np.float64)
    return np.concatenate(
        [
            np.zeros(degree + 1, dtype=np.float64),
            interior,
            np.ones(degree + 1, dtype=np.float64),
        ]
    )


def _bspline_basis_matrix(t_values: np.ndarray, knots: np.ndarray, degree: int, n_ctrl: int) -> np.ndarray:
    basis = np.zeros((len(t_values), n_ctrl), dtype=np.float64)
    for idx in range(n_ctrl):
        coeffs = np.zeros(n_ctrl, dtype=np.float64)
        coeffs[idx] = 1.0
        basis[:, idx] = BSpline(knots, coeffs, degree, extrapolate=False)(t_values)
    basis[~np.isfinite(basis)] = 0.0
    return basis


def fit_bspline_curve(
    points: np.ndarray,
    num_middle_ctrl_points: int,
) -> dict[str, Any] | None:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return None

    target_ctrl = int(num_middle_ctrl_points) + 2
    n_ctrl = int(min(max(2, target_ctrl), len(points)))
    degree = int(min(max(1, num_middle_ctrl_points), n_ctrl - 1))
    if n_ctrl < 2 or degree < 1:
        return None

    t_values = _chord_length_parameterize(points)
    knots = _open_uniform_knots(n_ctrl=n_ctrl, degree=degree)
    basis = _bspline_basis_matrix(t_values, knots, degree=degree, n_ctrl=n_ctrl)

    start_pt = np.asarray(points[0], dtype=np.float64)
    end_pt = np.asarray(points[-1], dtype=np.float64)

    if n_ctrl == 2:
        control_points = np.vstack([start_pt, end_pt])
    else:
        rhs = points - basis[:, [0]] * start_pt - basis[:, [-1]] * end_pt
        inner_basis = basis[:, 1:-1]
        if inner_basis.size == 0:
            return None
        inner_ctrl, _, _, _ = np.linalg.lstsq(inner_basis, rhs, rcond=None)
        control_points = np.vstack([start_pt, inner_ctrl, end_pt])

    if not np.isfinite(control_points).all():
        return None

    spline_r = BSpline(knots, control_points[:, 0], degree, extrapolate=False)
    spline_c = BSpline(knots, control_points[:, 1], degree, extrapolate=False)
    fitted = np.column_stack([spline_r(t_values), spline_c(t_values)])
    if not np.isfinite(fitted).all():
        return None

    metrics = fit_error_metrics(points, fitted)
    return {
        "curve_type": "bspline",
        "degree": degree,
        "num_middle_ctrl_points": int(num_middle_ctrl_points),
        "control_points": control_points,
        "knots": knots,
        "t_values": t_values,
        "fitted_points": fitted,
        "points": points,
        "mean_error": float(metrics["mean_error"]),
        "max_error": float(metrics["max_error"]),
    }


def fit_polyline_with_piecewise_bspline(
    points: np.ndarray,
    num_middle_ctrl_points: int,
    mean_error_threshold: float = 0.75,
    max_error_threshold: float = 2.5,
    max_segment_length: float = 120.0,
    angle_threshold_deg: float = 50.0,
    min_points: int = 6,
) -> list[dict[str, Any]]:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return []

    split_points = split_polyline_by_corners(
        points,
        max_segment_length=max_segment_length,
        angle_threshold_deg=angle_threshold_deg,
        min_index_gap=max(4, min_points - 1),
        min_split_arc_length=max(8.0, float(min_points)),
        prefer_anchor_for_length_split=False,
        split_extrema_window=5,
    )

    piecewise_segments: list[dict[str, Any]] = []
    for chunk_start, chunk_end in zip(split_points[:-1], split_points[1:]):
        if chunk_end <= chunk_start:
            continue
        chunk = points[chunk_start:chunk_end + 1]
        start_idx = 0
        while start_idx < len(chunk) - 1:
            best_candidate = None
            fallback_candidate = None
            candidate_end = start_idx + max(2, min_points)

            while candidate_end <= len(chunk):
                sub_points = chunk[start_idx:candidate_end]
                fit = fit_bspline_curve(sub_points, num_middle_ctrl_points=num_middle_ctrl_points)
                if fit is None:
                    break
                fallback_candidate = fit
                if fit["mean_error"] <= mean_error_threshold and fit["max_error"] <= max_error_threshold:
                    best_candidate = fit
                    candidate_end += 1
                    continue
                break

            selected = best_candidate
            if selected is None:
                selected = fallback_candidate
                if selected is None:
                    forced_end = min(len(chunk), start_idx + max(2, min_points))
                    selected = fit_bspline_curve(
                        chunk[start_idx:forced_end],
                        num_middle_ctrl_points=num_middle_ctrl_points,
                    )
                    if selected is None:
                        break

            piecewise_segments.append(selected)
            consumed = len(selected["points"]) - 1
            if consumed <= 0:
                break
            start_idx += consumed

    return piecewise_segments


def fit_paths_with_piecewise_bspline(
    paths: list[np.ndarray],
    num_middle_ctrl_points: int,
    mean_error_threshold: float = 0.75,
    max_error_threshold: float = 2.5,
    max_segment_length: float = 120.0,
    angle_threshold_deg: float = 50.0,
    min_points: int = 6,
    min_path_length_for_spline: float = 6.0,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    fitted_paths: list[dict[str, Any]] = []
    dropped_paths: list[np.ndarray] = []
    for path in paths:
        path = np.asarray(path, dtype=np.float64)
        if path_length(path) < float(min_path_length_for_spline):
            fit = fit_bspline_curve(path, num_middle_ctrl_points=num_middle_ctrl_points)
            if fit is None:
                dropped_paths.append(path)
                continue
            fitted_paths.append({"original_points": path, "segments": [fit]})
            continue
        segments = fit_polyline_with_piecewise_bspline(
            path,
            num_middle_ctrl_points=num_middle_ctrl_points,
            mean_error_threshold=mean_error_threshold,
            max_error_threshold=max_error_threshold,
            max_segment_length=max_segment_length,
            angle_threshold_deg=angle_threshold_deg,
            min_points=min_points,
        )
        if not segments:
            dropped_paths.append(path)
            continue
        fitted_paths.append({"original_points": path, "segments": segments})
    return fitted_paths, dropped_paths


def sample_piecewise_bspline(segments: list[dict[str, Any]], samples_per_segment: int = 64) -> np.ndarray:
    samples: list[np.ndarray] = []
    for segment in segments:
        ctrl = np.asarray(segment["control_points"], dtype=np.float64)
        knots = np.asarray(segment["knots"], dtype=np.float64)
        degree = int(segment["degree"])
        ctrl_len = path_length(ctrl)
        adaptive_samples = max(samples_per_segment, int(np.ceil(max(ctrl_len, 1.0) * 2.0)))
        adaptive_samples = min(adaptive_samples, 4096)
        t_values = np.linspace(0.0, 1.0, adaptive_samples, dtype=np.float64)
        spline_r = BSpline(knots, ctrl[:, 0], degree, extrapolate=False)
        spline_c = BSpline(knots, ctrl[:, 1], degree, extrapolate=False)
        curve = np.column_stack([spline_r(t_values), spline_c(t_values)])
        curve = curve[np.isfinite(curve).all(axis=1)]
        if len(curve) == 0:
            continue
        if samples:
            curve = curve[1:]
        samples.append(curve)
    if not samples:
        return np.zeros((0, 2), dtype=np.float64)
    return np.vstack(samples)


def rasterize_points(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    from bezierization.bezier_refiner_core import rasterize_points as _rasterize_points

    return _rasterize_points(shape, points)


def render_piecewise_bspline_fits(
    shape: tuple[int, int],
    fitted_paths: list[dict[str, Any]],
    samples_per_segment: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    raster = np.zeros(shape, dtype=np.uint8)
    sampled_points: list[np.ndarray] = []
    for path_fit in fitted_paths:
        curve_points = sample_piecewise_bspline(path_fit["segments"], samples_per_segment=samples_per_segment)
        if len(curve_points) == 0:
            continue
        sampled_points.append(curve_points)
        raster = np.maximum(raster, rasterize_points(shape, curve_points))
    if sampled_points:
        sampled = np.vstack(sampled_points)
    else:
        sampled = np.zeros((0, 2), dtype=np.float64)
    return raster, sampled


def draw_colored_bspline_curves_on_image(image_np: np.ndarray, paths: list[dict[str, Any]]) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(image_np)
    cmap = plt.colormaps.get_cmap("gist_rainbow")
    for idx, path in enumerate(paths):
        pts = sample_piecewise_bspline(path["segments"], samples_per_segment=64)
        if len(pts) == 0:
            continue
        color = cmap(idx / max(len(paths), 1))
        ax.plot(pts[:, 1], pts[:, 0], color=color, linewidth=2)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return arr


def draw_bspline_control_points_on_image(image_np: np.ndarray, paths: list[dict[str, Any]]) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(image_np)
    cmap = plt.colormaps.get_cmap("gist_rainbow")
    for idx, path in enumerate(paths):
        color = cmap(idx / max(len(paths), 1))
        for seg in path["segments"]:
            ctrl = np.asarray(seg["control_points"], dtype=np.float64)
            curve = sample_piecewise_bspline([seg], samples_per_segment=64)
            if len(curve) > 0:
                ax.plot(curve[:, 1], curve[:, 0], color=color, linewidth=1.8, alpha=0.95)
            ax.plot(ctrl[:, 1], ctrl[:, 0], color=color, linestyle="--", linewidth=1.0, alpha=0.65)
            if len(ctrl) > 2:
                ax.scatter(ctrl[1:-1, 1], ctrl[1:-1, 0], s=18, c=[color], edgecolors="white", linewidths=0.6)
            ax.scatter([ctrl[0, 1], ctrl[-1, 1]], [ctrl[0, 0], ctrl[-1, 0]], s=24, c="red", edgecolors="yellow", linewidths=0.8)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return arr


def run_bspline_from_source_edge(
    source_edge: np.ndarray,
    num_middle_ctrl_points: int,
    compute_final_raster: bool = False,
) -> dict[str, Any]:
    source_edge = np.asarray(source_edge, dtype=bool)
    paths, _, _, _, _ = extract_ordered_edge_paths(source_edge, connectivity=2)
    fitted_paths, dropped_paths = fit_paths_with_piecewise_bspline(
        paths,
        num_middle_ctrl_points=num_middle_ctrl_points,
        mean_error_threshold=0.75,
        max_error_threshold=2.5,
        max_segment_length=120.0,
        angle_threshold_deg=50.0,
        min_points=6,
        min_path_length_for_spline=6.0,
    )
    final_raster = None
    if compute_final_raster:
        final_raster, _ = render_piecewise_bspline_fits(source_edge.shape, fitted_paths)

    summary = {
        "path_count": int(len(fitted_paths)),
        "segment_count": int(sum(len(p["segments"]) for p in fitted_paths)),
        "dropped_path_count": int(len(dropped_paths)),
        "num_middle_ctrl_points": int(num_middle_ctrl_points),
    }
    result: dict[str, Any] = {
        "final_paths": fitted_paths,
        "dropped_paths": dropped_paths,
        "summary": summary,
    }
    if final_raster is not None:
        result["final_raster"] = final_raster
    return result
