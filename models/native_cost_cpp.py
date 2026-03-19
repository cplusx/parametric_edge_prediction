from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import torch

_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <limits>

template <typename scalar_t>
torch::Tensor aligned_curve_l1_cost_impl(torch::Tensor pred, torch::Tensor tgt) {
    const auto num_pred = pred.size(0);
    const auto num_tgt = tgt.size(0);
    const auto num_ctrl = pred.size(1);
    auto out = torch::zeros({num_pred, num_tgt}, pred.options());

    auto pred_a = pred.accessor<scalar_t, 3>();
    auto tgt_a = tgt.accessor<scalar_t, 3>();
    auto out_a = out.accessor<scalar_t, 2>();

    const scalar_t denom = static_cast<scalar_t>(num_ctrl * 2);
    for (int64_t i = 0; i < num_pred; ++i) {
        for (int64_t j = 0; j < num_tgt; ++j) {
            scalar_t forward = 0;
            scalar_t reverse = 0;
            for (int64_t k = 0; k < num_ctrl; ++k) {
                const int64_t rk = num_ctrl - 1 - k;
                forward += std::abs(pred_a[i][k][0] - tgt_a[j][k][0]);
                forward += std::abs(pred_a[i][k][1] - tgt_a[j][k][1]);
                reverse += std::abs(pred_a[i][k][0] - tgt_a[j][rk][0]);
                reverse += std::abs(pred_a[i][k][1] - tgt_a[j][rk][1]);
            }
            out_a[i][j] = std::min(forward, reverse) / denom;
        }
    }
    return out;
}

template <typename scalar_t>
torch::Tensor aligned_sample_l1_cost_impl(torch::Tensor pred, torch::Tensor tgt) {
    const auto num_pred = pred.size(0);
    const auto num_tgt = tgt.size(0);
    const auto num_pts = pred.size(1);
    auto out = torch::zeros({num_pred, num_tgt}, pred.options());

    auto pred_a = pred.accessor<scalar_t, 3>();
    auto tgt_a = tgt.accessor<scalar_t, 3>();
    auto out_a = out.accessor<scalar_t, 2>();

    const scalar_t denom = static_cast<scalar_t>(num_pts * 2);
    for (int64_t i = 0; i < num_pred; ++i) {
        for (int64_t j = 0; j < num_tgt; ++j) {
            scalar_t forward = 0;
            scalar_t reverse = 0;
            for (int64_t k = 0; k < num_pts; ++k) {
                const int64_t rk = num_pts - 1 - k;
                forward += std::abs(pred_a[i][k][0] - tgt_a[j][k][0]);
                forward += std::abs(pred_a[i][k][1] - tgt_a[j][k][1]);
                reverse += std::abs(pred_a[i][k][0] - tgt_a[j][rk][0]);
                reverse += std::abs(pred_a[i][k][1] - tgt_a[j][rk][1]);
            }
            out_a[i][j] = std::min(forward, reverse) / denom;
        }
    }
    return out;
}

template <typename scalar_t>
torch::Tensor pairwise_curve_chamfer_from_samples_impl(torch::Tensor pred, torch::Tensor tgt) {
    const auto num_pred = pred.size(0);
    const auto num_tgt = tgt.size(0);
    const auto num_pts = pred.size(1);
    auto out = torch::zeros({num_pred, num_tgt}, pred.options());

    auto pred_a = pred.accessor<scalar_t, 3>();
    auto tgt_a = tgt.accessor<scalar_t, 3>();
    auto out_a = out.accessor<scalar_t, 2>();

    for (int64_t i = 0; i < num_pred; ++i) {
        for (int64_t j = 0; j < num_tgt; ++j) {
            scalar_t pred_to_tgt = 0;
            scalar_t pred_to_tgt_rev = 0;
            for (int64_t p = 0; p < num_pts; ++p) {
                scalar_t best = std::numeric_limits<scalar_t>::max();
                scalar_t best_rev = std::numeric_limits<scalar_t>::max();
                for (int64_t q = 0; q < num_pts; ++q) {
                    const scalar_t dx = pred_a[i][p][0] - tgt_a[j][q][0];
                    const scalar_t dy = pred_a[i][p][1] - tgt_a[j][q][1];
                    const scalar_t dist = std::sqrt(dx * dx + dy * dy);
                    if (dist < best) best = dist;

                    const int64_t rq = num_pts - 1 - q;
                    const scalar_t dxr = pred_a[i][p][0] - tgt_a[j][rq][0];
                    const scalar_t dyr = pred_a[i][p][1] - tgt_a[j][rq][1];
                    const scalar_t dist_rev = std::sqrt(dxr * dxr + dyr * dyr);
                    if (dist_rev < best_rev) best_rev = dist_rev;
                }
                pred_to_tgt += best;
                pred_to_tgt_rev += best_rev;
            }

            scalar_t tgt_to_pred = 0;
            scalar_t tgt_rev_to_pred = 0;
            for (int64_t q = 0; q < num_pts; ++q) {
                scalar_t best = std::numeric_limits<scalar_t>::max();
                scalar_t best_rev = std::numeric_limits<scalar_t>::max();
                const int64_t rq = num_pts - 1 - q;
                for (int64_t p = 0; p < num_pts; ++p) {
                    const scalar_t dx = tgt_a[j][q][0] - pred_a[i][p][0];
                    const scalar_t dy = tgt_a[j][q][1] - pred_a[i][p][1];
                    const scalar_t dist = std::sqrt(dx * dx + dy * dy);
                    if (dist < best) best = dist;

                    const scalar_t dxr = tgt_a[j][rq][0] - pred_a[i][p][0];
                    const scalar_t dyr = tgt_a[j][rq][1] - pred_a[i][p][1];
                    const scalar_t dist_rev = std::sqrt(dxr * dxr + dyr * dyr);
                    if (dist_rev < best_rev) best_rev = dist_rev;
                }
                tgt_to_pred += best;
                tgt_rev_to_pred += best_rev;
            }

            const scalar_t forward = static_cast<scalar_t>(0.5) * (
                pred_to_tgt / static_cast<scalar_t>(num_pts) +
                tgt_to_pred / static_cast<scalar_t>(num_pts)
            );
            const scalar_t reverse = static_cast<scalar_t>(0.5) * (
                pred_to_tgt_rev / static_cast<scalar_t>(num_pts) +
                tgt_rev_to_pred / static_cast<scalar_t>(num_pts)
            );
            out_a[i][j] = std::min(forward, reverse);
        }
    }
    return out;
}

torch::Tensor aligned_curve_l1_cost(torch::Tensor pred, torch::Tensor tgt) {
    TORCH_CHECK(pred.device().is_cpu(), "pred must be CPU");
    TORCH_CHECK(tgt.device().is_cpu(), "tgt must be CPU");
    TORCH_CHECK(pred.dim() == 3 && tgt.dim() == 3, "expected [N,K,2] and [M,K,2]");
    TORCH_CHECK(pred.size(1) == tgt.size(1) && pred.size(2) == 2 && tgt.size(2) == 2, "shape mismatch");
    auto pred_c = pred.contiguous();
    auto tgt_c = tgt.contiguous();
    AT_DISPATCH_FLOATING_TYPES(pred_c.scalar_type(), "aligned_curve_l1_cost", [&] {
        pred_c = pred_c.contiguous();
        tgt_c = tgt_c.contiguous();
    });
    return AT_DISPATCH_FLOATING_TYPES(pred_c.scalar_type(), "aligned_curve_l1_cost", [&] {
        return aligned_curve_l1_cost_impl<scalar_t>(pred_c, tgt_c);
    });
}

torch::Tensor aligned_sample_l1_cost(torch::Tensor pred, torch::Tensor tgt) {
    TORCH_CHECK(pred.device().is_cpu(), "pred must be CPU");
    TORCH_CHECK(tgt.device().is_cpu(), "tgt must be CPU");
    TORCH_CHECK(pred.dim() == 3 && tgt.dim() == 3, "expected [N,S,2] and [M,S,2]");
    TORCH_CHECK(pred.size(1) == tgt.size(1) && pred.size(2) == 2 && tgt.size(2) == 2, "shape mismatch");
    auto pred_c = pred.contiguous();
    auto tgt_c = tgt.contiguous();
    return AT_DISPATCH_FLOATING_TYPES(pred_c.scalar_type(), "aligned_sample_l1_cost", [&] {
        return aligned_sample_l1_cost_impl<scalar_t>(pred_c, tgt_c);
    });
}

torch::Tensor pairwise_curve_chamfer_from_samples(torch::Tensor pred, torch::Tensor tgt) {
    TORCH_CHECK(pred.device().is_cpu(), "pred must be CPU");
    TORCH_CHECK(tgt.device().is_cpu(), "tgt must be CPU");
    TORCH_CHECK(pred.dim() == 3 && tgt.dim() == 3, "expected [N,S,2] and [M,S,2]");
    TORCH_CHECK(pred.size(1) == tgt.size(1) && pred.size(2) == 2 && tgt.size(2) == 2, "shape mismatch");
    auto pred_c = pred.contiguous();
    auto tgt_c = tgt.contiguous();
    return AT_DISPATCH_FLOATING_TYPES(pred_c.scalar_type(), "pairwise_curve_chamfer_from_samples", [&] {
        return pairwise_curve_chamfer_from_samples_impl<scalar_t>(pred_c, tgt_c);
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("aligned_curve_l1_cost", &aligned_curve_l1_cost, "Aligned control-point pairwise cost (CPU)");
    m.def("aligned_sample_l1_cost", &aligned_sample_l1_cost, "Aligned sampled-point pairwise cost (CPU)");
    m.def("pairwise_curve_chamfer_from_samples", &pairwise_curve_chamfer_from_samples, "Pairwise curve chamfer from sampled points (CPU)");
}
"""


USE_NATIVE_MATCHER_CPP = os.environ.get("PARAM_EDGE_USE_NATIVE_MATCHER_CPP", "1") != "0"


@lru_cache(maxsize=1)
def _load_extension() -> Optional[object]:
    if not USE_NATIVE_MATCHER_CPP:
        return None
    try:
        from torch.utils.cpp_extension import load_inline
    except Exception:
        return None
    try:
        return load_inline(
            name="param_edge_native_cost_cpp",
            cpp_sources=_CPP_SOURCE,
            functions=None,
            extra_cflags=["-O3", "-std=c++17"],
            with_cuda=False,
            verbose=False,
        )
    except Exception:
        return None


def backend_available() -> bool:
    return _load_extension() is not None


def _ensure_cpu_contiguous(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(device="cpu").contiguous()


def aligned_curve_l1_cost(pred_curves: torch.Tensor, tgt_curves: torch.Tensor) -> torch.Tensor:
    module = _load_extension()
    if module is None:
        raise RuntimeError("native matcher cpp backend unavailable")
    return module.aligned_curve_l1_cost(_ensure_cpu_contiguous(pred_curves), _ensure_cpu_contiguous(tgt_curves))


def aligned_sample_l1_cost(pred_samples: torch.Tensor, tgt_samples: torch.Tensor) -> torch.Tensor:
    module = _load_extension()
    if module is None:
        raise RuntimeError("native matcher cpp backend unavailable")
    return module.aligned_sample_l1_cost(_ensure_cpu_contiguous(pred_samples), _ensure_cpu_contiguous(tgt_samples))


def pairwise_curve_chamfer_from_samples(pred_samples: torch.Tensor, tgt_samples: torch.Tensor) -> torch.Tensor:
    module = _load_extension()
    if module is None:
        raise RuntimeError("native matcher cpp backend unavailable")
    return module.pairwise_curve_chamfer_from_samples(_ensure_cpu_contiguous(pred_samples), _ensure_cpu_contiguous(tgt_samples))
