import os
from functools import lru_cache

import numpy as np

try:
    from bezierization import native_backend_cffi
    NATIVE_CFFI_AVAILABLE = native_backend_cffi.backend_available()
except Exception:
    native_backend_cffi = None
    NATIVE_CFFI_AVAILABLE = False

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


USE_FAST_NUMBA_BACKEND = os.environ.get('BEZIER_USE_FAST_NUMBA', '1') != '0' and NUMBA_AVAILABLE
USE_NATIVE_CFFI_BACKEND = os.environ.get('BEZIER_USE_NATIVE_CFFI', '1') != '0' and NATIVE_CFFI_AVAILABLE


@njit(cache=True)
def _path_length_numba(points):
    n_points = points.shape[0]
    if n_points < 2:
        return 0.0
    total = 0.0
    for idx in range(n_points - 1):
        dy = points[idx + 1, 0] - points[idx, 0]
        dx = points[idx + 1, 1] - points[idx, 1]
        total += (dy * dy + dx * dx) ** 0.5
    return total


@njit(cache=True)
def _chord_length_parameterize_numba(points):
    n_points = points.shape[0]
    if n_points == 1:
        out = np.zeros((1,), dtype=np.float64)
        return out
    cumulative = np.zeros((n_points,), dtype=np.float64)
    total = 0.0
    for idx in range(n_points - 1):
        dy = points[idx + 1, 0] - points[idx, 0]
        dx = points[idx + 1, 1] - points[idx, 1]
        total += (dy * dy + dx * dx) ** 0.5
        cumulative[idx + 1] = total
    if total < 1e-8:
        out = np.empty((n_points,), dtype=np.float64)
        if n_points == 1:
            out[0] = 0.0
            return out
        for idx in range(n_points):
            out[idx] = idx / max(n_points - 1, 1)
        return out
    return cumulative / total


@njit(cache=True)
def _bernstein_basis_numba(coeffs, degree, t_values):
    n_t = t_values.shape[0]
    basis = np.empty((n_t, degree + 1), dtype=np.float64)
    for row in range(n_t):
        t = t_values[row]
        one_minus_t = 1.0 - t
        for col in range(degree + 1):
            basis[row, col] = coeffs[col] * (t ** col) * (one_minus_t ** (degree - col))
    return basis


@njit(cache=True)
def _evaluate_bezier_numba(control_points, basis):
    n_t = basis.shape[0]
    n_ctrl = control_points.shape[0]
    dims = control_points.shape[1]
    out = np.zeros((n_t, dims), dtype=np.float64)
    for row in range(n_t):
        for ctrl_idx in range(n_ctrl):
            weight = basis[row, ctrl_idx]
            for dim in range(dims):
                out[row, dim] += weight * control_points[ctrl_idx, dim]
    return out


@njit(cache=True)
def _fit_error_metrics_numba(points, fitted_points):
    n_points = points.shape[0]
    if n_points == 0:
        return 0.0, 0.0
    total = 0.0
    max_err = 0.0
    for idx in range(n_points):
        dy = points[idx, 0] - fitted_points[idx, 0]
        dx = points[idx, 1] - fitted_points[idx, 1]
        dist = (dy * dy + dx * dx) ** 0.5
        total += dist
        if dist > max_err:
            max_err = dist
    return total / n_points, max_err


@njit(cache=True)
def _points_center_numba(points):
    n_points = points.shape[0]
    dims = points.shape[1]
    center = np.zeros((dims,), dtype=np.float64)
    if n_points == 0:
        return center
    for i in range(n_points):
        for d in range(dims):
            center[d] += points[i, d]
    for d in range(dims):
        center[d] /= n_points
    return center


@njit(cache=True)
def _bbox_extent_numba(points):
    dims = points.shape[1]
    mins = points[0].copy()
    maxs = points[0].copy()
    for i in range(1, points.shape[0]):
        for d in range(dims):
            if points[i, d] < mins[d]:
                mins[d] = points[i, d]
            if points[i, d] > maxs[d]:
                maxs[d] = points[i, d]
    total = 0.0
    for d in range(dims):
        diff = maxs[d] - mins[d]
        total += diff * diff
    return total ** 0.5


@njit(cache=True)
def _max_center_offset_numba(control_points, center):
    max_offset = 0.0
    for i in range(control_points.shape[0]):
        total = 0.0
        for d in range(control_points.shape[1]):
            diff = control_points[i, d] - center[d]
            total += diff * diff
        dist = total ** 0.5
        if dist > max_offset:
            max_offset = dist
    return max_offset


@njit(cache=True)
def _control_polygon_is_stable_numba(control_points, points, data_scale, points_center, max_multiplier, max_offset_multiplier):
    for i in range(control_points.shape[0]):
        for d in range(control_points.shape[1]):
            value = control_points[i, d]
            if not np.isfinite(value):
                return False
    control_polygon_length = _path_length_numba(control_points)
    if (not np.isfinite(control_polygon_length)) or control_polygon_length > max_multiplier * data_scale:
        return False
    max_center_offset = _max_center_offset_numba(control_points, points_center)
    if (not np.isfinite(max_center_offset)) or max_center_offset > max_offset_multiplier * data_scale:
        return False
    return True


@lru_cache(maxsize=16)
def bernstein_coefficients(degree):
    from math import comb as math_comb
    return np.asarray([math_comb(degree, i) for i in range(degree + 1)], dtype=np.float64)


def path_length(points):
    pts = np.asarray(points, dtype=np.float64)
    if USE_NATIVE_CFFI_BACKEND:
        return native_backend_cffi.path_length(pts)
    if USE_FAST_NUMBA_BACKEND:
        return float(_path_length_numba(pts))
    if len(pts) < 2:
        return 0.0
    deltas = np.diff(pts, axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def chord_length_parameterize(points):
    pts = np.asarray(points, dtype=np.float64)
    if USE_FAST_NUMBA_BACKEND:
        return _chord_length_parameterize_numba(pts)
    if len(pts) == 1:
        return np.array([0.0], dtype=np.float64)
    deltas = np.diff(pts, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    total = cumulative[-1]
    if total < 1e-8:
        return np.linspace(0.0, 1.0, len(pts))
    return cumulative / total


def bernstein_basis(degree, t_values):
    t_values = np.asarray(t_values, dtype=np.float64)
    coeffs = bernstein_coefficients(int(degree))
    if USE_FAST_NUMBA_BACKEND:
        return _bernstein_basis_numba(coeffs, int(degree), t_values)
    basis = np.empty((len(t_values), degree + 1), dtype=np.float64)
    for i in range(degree + 1):
        basis[:, i] = coeffs[i] * (t_values ** i) * ((1.0 - t_values) ** (degree - i))
    return basis


def evaluate_bezier(control_points, basis):
    ctrl = np.asarray(control_points, dtype=np.float64)
    base = np.asarray(basis, dtype=np.float64)
    if USE_NATIVE_CFFI_BACKEND:
        return native_backend_cffi.evaluate_bezier(ctrl, base)
    if USE_FAST_NUMBA_BACKEND:
        return _evaluate_bezier_numba(ctrl, base)
    return base @ ctrl


def fit_error_metrics(points, fitted_points):
    pts = np.asarray(points, dtype=np.float64)
    fit = np.asarray(fitted_points, dtype=np.float64)
    if USE_NATIVE_CFFI_BACKEND:
        return native_backend_cffi.fit_error_metrics(pts, fit)
    if USE_FAST_NUMBA_BACKEND:
        mean_error, max_error = _fit_error_metrics_numba(pts, fit)
        return {
            'mean_error': float(mean_error),
            'max_error': float(max_error),
        }
    distances = np.linalg.norm(pts - fit, axis=1)
    return {
        'mean_error': float(distances.mean()) if len(distances) else 0.0,
        'max_error': float(distances.max()) if len(distances) else 0.0,
    }


def prepare_fit_context(points):
    pts = np.asarray(points, dtype=np.float64)
    t_values = chord_length_parameterize(pts)
    if USE_FAST_NUMBA_BACKEND:
        data_scale = max(
            float(_path_length_numba(pts)),
            float(np.linalg.norm(pts[-1] - pts[0])) if len(pts) >= 2 else 0.0,
            float(_bbox_extent_numba(pts)) if len(pts) else 0.0,
            1.0,
        )
        points_center = _points_center_numba(pts)
    else:
        data_scale = max(
            path_length(pts),
            np.linalg.norm(pts[-1] - pts[0]) if len(pts) >= 2 else 0.0,
            np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)) if len(pts) else 0.0,
            1.0,
        )
        points_center = pts.mean(axis=0)
    return pts, t_values, data_scale, points_center


def control_polygon_is_stable(control_points, points, data_scale, points_center, max_multiplier=100.0, max_offset_multiplier=25.0):
    ctrl = np.asarray(control_points, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    center = np.asarray(points_center, dtype=np.float64)
    if USE_NATIVE_CFFI_BACKEND:
        return native_backend_cffi.control_polygon_is_stable(
            ctrl,
            center,
            data_scale=float(data_scale),
            max_multiplier=float(max_multiplier),
            max_offset_multiplier=float(max_offset_multiplier),
        )
    if USE_FAST_NUMBA_BACKEND:
        return bool(_control_polygon_is_stable_numba(ctrl, pts, float(data_scale), center, float(max_multiplier), float(max_offset_multiplier)))
    if not np.isfinite(ctrl).all():
        return False
    control_polygon_length = path_length(ctrl)
    if not np.isfinite(control_polygon_length) or control_polygon_length > max_multiplier * data_scale:
        return False
    max_center_offset = float(np.linalg.norm(ctrl - center, axis=1).max())
    if not np.isfinite(max_center_offset) or max_center_offset > max_offset_multiplier * data_scale:
        return False
    return True
