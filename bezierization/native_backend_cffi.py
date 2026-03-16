import os
from pathlib import Path

import numpy as np

try:
    from cffi import FFI
    CFFI_AVAILABLE = True
except Exception:
    FFI = None
    CFFI_AVAILABLE = False


_LIB = None
_FFI = None
_LOAD_ATTEMPTED = False


def _build_source():
    return r"""
        #include <math.h>
        #include <stdbool.h>
        #include <stddef.h>

        double bez_path_length(const double* points, size_t n_points, size_t dims) {
            if (n_points < 2) {
                return 0.0;
            }
            double total = 0.0;
            for (size_t i = 0; i + 1 < n_points; ++i) {
                double sum = 0.0;
                for (size_t d = 0; d < dims; ++d) {
                    double diff = points[(i + 1) * dims + d] - points[i * dims + d];
                    sum += diff * diff;
                }
                total += sqrt(sum);
            }
            return total;
        }

        void bez_fit_error_metrics(
            const double* points,
            const double* fitted_points,
            size_t n_points,
            size_t dims,
            double* out_mean,
            double* out_max
        ) {
            if (n_points == 0) {
                *out_mean = 0.0;
                *out_max = 0.0;
                return;
            }
            double total = 0.0;
            double max_err = 0.0;
            for (size_t i = 0; i < n_points; ++i) {
                double sum = 0.0;
                for (size_t d = 0; d < dims; ++d) {
                    double diff = points[i * dims + d] - fitted_points[i * dims + d];
                    sum += diff * diff;
                }
                double dist = sqrt(sum);
                total += dist;
                if (dist > max_err) {
                    max_err = dist;
                }
            }
            *out_mean = total / (double)n_points;
            *out_max = max_err;
        }

        void bez_evaluate_bezier(
            const double* control_points,
            size_t n_ctrl,
            size_t dims,
            const double* basis,
            size_t n_t,
            double* out
        ) {
            for (size_t i = 0; i < n_t; ++i) {
                for (size_t d = 0; d < dims; ++d) {
                    double value = 0.0;
                    for (size_t j = 0; j < n_ctrl; ++j) {
                        value += basis[i * n_ctrl + j] * control_points[j * dims + d];
                    }
                    out[i * dims + d] = value;
                }
            }
        }

        bool bez_control_polygon_is_stable(
            const double* control_points,
            size_t n_ctrl,
            const double* center,
            size_t dims,
            double data_scale,
            double max_multiplier,
            double max_offset_multiplier
        ) {
            for (size_t i = 0; i < n_ctrl * dims; ++i) {
                if (!isfinite(control_points[i])) {
                    return false;
                }
            }
            double total_length = 0.0;
            if (n_ctrl >= 2) {
                for (size_t i = 0; i + 1 < n_ctrl; ++i) {
                    double sum = 0.0;
                    for (size_t d = 0; d < dims; ++d) {
                        double diff = control_points[(i + 1) * dims + d] - control_points[i * dims + d];
                        sum += diff * diff;
                    }
                    total_length += sqrt(sum);
                }
            }
            if (!isfinite(total_length) || total_length > max_multiplier * data_scale) {
                return false;
            }
            double max_center_offset = 0.0;
            for (size_t i = 0; i < n_ctrl; ++i) {
                double sum = 0.0;
                for (size_t d = 0; d < dims; ++d) {
                    double diff = control_points[i * dims + d] - center[d];
                    sum += diff * diff;
                }
                double dist = sqrt(sum);
                if (dist > max_center_offset) {
                    max_center_offset = dist;
                }
            }
            if (!isfinite(max_center_offset) || max_center_offset > max_offset_multiplier * data_scale) {
                return false;
            }
            return true;
        }
    """


def _build_cdef():
    return """
        double bez_path_length(const double* points, size_t n_points, size_t dims);
        void bez_fit_error_metrics(
            const double* points,
            const double* fitted_points,
            size_t n_points,
            size_t dims,
            double* out_mean,
            double* out_max
        );
        void bez_evaluate_bezier(
            const double* control_points,
            size_t n_ctrl,
            size_t dims,
            const double* basis,
            size_t n_t,
            double* out
        );
        bool bez_control_polygon_is_stable(
            const double* control_points,
            size_t n_ctrl,
            const double* center,
            size_t dims,
            double data_scale,
            double max_multiplier,
            double max_offset_multiplier
        );
    """


def _load_native_backend():
    global _LIB, _FFI, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _LIB, _FFI
    _LOAD_ATTEMPTED = True
    if not CFFI_AVAILABLE:
        return None, None
    try:
        ffibuilder = FFI()
        ffibuilder.cdef(_build_cdef())
        tmpdir = Path(__file__).resolve().parent / "_native_build"
        tmpdir.mkdir(parents=True, exist_ok=True)
        lib = ffibuilder.verify(
            _build_source(),
            extra_compile_args=['-O3'],
            tmpdir=str(tmpdir),
        )
        _LIB = lib
        _FFI = ffibuilder
    except Exception:
        _LIB = None
        _FFI = None
    return _LIB, _FFI


USE_NATIVE_CFFI_BACKEND = os.environ.get('BEZIER_USE_NATIVE_CFFI', '1') != '0'


def backend_available():
    lib, ffi = _load_native_backend()
    return USE_NATIVE_CFFI_BACKEND and lib is not None and ffi is not None


def path_length(points):
    lib, ffi = _load_native_backend()
    if not USE_NATIVE_CFFI_BACKEND or lib is None or ffi is None:
        raise RuntimeError('native CFFI backend unavailable')
    pts = np.ascontiguousarray(points, dtype=np.float64)
    ptr = ffi.from_buffer('double[]', pts.reshape(-1))
    return float(lib.bez_path_length(ptr, pts.shape[0], pts.shape[1]))


def fit_error_metrics(points, fitted_points):
    lib, ffi = _load_native_backend()
    if not USE_NATIVE_CFFI_BACKEND or lib is None or ffi is None:
        raise RuntimeError('native CFFI backend unavailable')
    pts = np.ascontiguousarray(points, dtype=np.float64)
    fit = np.ascontiguousarray(fitted_points, dtype=np.float64)
    out_mean = ffi.new('double *')
    out_max = ffi.new('double *')
    lib.bez_fit_error_metrics(
        ffi.from_buffer('double[]', pts.reshape(-1)),
        ffi.from_buffer('double[]', fit.reshape(-1)),
        pts.shape[0],
        pts.shape[1],
        out_mean,
        out_max,
    )
    return {
        'mean_error': float(out_mean[0]),
        'max_error': float(out_max[0]),
    }


def evaluate_bezier(control_points, basis):
    lib, ffi = _load_native_backend()
    if not USE_NATIVE_CFFI_BACKEND or lib is None or ffi is None:
        raise RuntimeError('native CFFI backend unavailable')
    ctrl = np.ascontiguousarray(control_points, dtype=np.float64)
    base = np.ascontiguousarray(basis, dtype=np.float64)
    out = np.empty((base.shape[0], ctrl.shape[1]), dtype=np.float64)
    lib.bez_evaluate_bezier(
        ffi.from_buffer('double[]', ctrl.reshape(-1)),
        ctrl.shape[0],
        ctrl.shape[1],
        ffi.from_buffer('double[]', base.reshape(-1)),
        base.shape[0],
        ffi.from_buffer('double[]', out.reshape(-1)),
    )
    return out


def control_polygon_is_stable(control_points, center, data_scale, max_multiplier, max_offset_multiplier):
    lib, ffi = _load_native_backend()
    if not USE_NATIVE_CFFI_BACKEND or lib is None or ffi is None:
        raise RuntimeError('native CFFI backend unavailable')
    ctrl = np.ascontiguousarray(control_points, dtype=np.float64)
    ctr = np.ascontiguousarray(center, dtype=np.float64)
    return bool(lib.bez_control_polygon_is_stable(
        ffi.from_buffer('double[]', ctrl.reshape(-1)),
        ctrl.shape[0],
        ffi.from_buffer('double[]', ctr.reshape(-1)),
        ctrl.shape[1],
        float(data_scale),
        float(max_multiplier),
        float(max_offset_multiplier),
    ))
