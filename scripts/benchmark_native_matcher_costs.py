from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import native_cost_cpp
from models.matcher import build_curve_cost_matrix
import models.geometry as geometry


def run_once(pred: torch.Tensor, tgt: torch.Tensor, use_native: bool) -> torch.Tensor:
    boxes = geometry.curve_boxes_xyxy(tgt)
    if use_native:
        with torch.no_grad():
            return build_curve_cost_matrix(
                torch.zeros(pred.shape[0], 2),
                pred,
                tgt,
                boxes,
                5.0,
                2.0,
                1.0,
                1.0,
                4,
                16,
            )
    with torch.enable_grad():
        pred_g = pred.clone().requires_grad_(True)
        tgt_g = tgt.clone().requires_grad_(True)
        boxes_g = geometry.curve_boxes_xyxy(tgt_g)
        return build_curve_cost_matrix(
            torch.zeros(pred_g.shape[0], 2),
            pred_g,
            tgt_g,
            boxes_g,
            5.0,
            2.0,
            1.0,
            1.0,
            4,
            16,
        ).detach()


def bench(pred: torch.Tensor, tgt: torch.Tensor, use_native: bool, iters: int) -> tuple[torch.Tensor, float]:
    run_once(pred, tgt, use_native)
    times = []
    last = None
    for _ in range(iters):
        t0 = time.perf_counter()
        last = run_once(pred, tgt, use_native)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return last, sum(times) / len(times)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark native matcher-side curve cost kernels.")
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    _ = native_cost_cpp._load_extension()

    for num_queries, num_targets in [(256, 180), (512, 220), (512, 512)]:
        torch.manual_seed(0)
        pred = torch.rand(num_queries, 6, 2)
        tgt = torch.rand(num_targets, 6, 2)
        ref, py_time = bench(pred, tgt, use_native=False, iters=args.iters)
        nat, native_time = bench(pred, tgt, use_native=True, iters=args.iters)
        print(f"shape {num_queries}x{num_targets}")
        print(f"  max_abs_diff: {float((ref - nat).abs().max()):.8f}")
        print(f"  py_time:      {py_time:.6f}s")
        print(f"  native_time:  {native_time:.6f}s")
        print(f"  speedup:      {py_time / max(native_time, 1e-9):.3f}x")


if __name__ == "__main__":
    main()
