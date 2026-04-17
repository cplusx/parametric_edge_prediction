from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam_mask_bezierization.pipeline import run_bezier_from_source_edge, save_compact_bezier_paths


VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-edge-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-name", type=str, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def iter_source_edges(source_edge_dir: Path, limit: int) -> list[Path]:
    edges = sorted(
        [p for p in source_edge_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS],
        key=lambda p: p.stem,
    )
    if limit > 0:
        edges = edges[:limit]
    return edges


def _process_one(source_edge_path: str, output_path: str) -> dict[str, object]:
    source_path = Path(source_edge_path)
    out_path = Path(output_path)
    if out_path.exists():
        return {"status": "skipped", "sample_id": source_path.stem}

    edge = np.asarray(Image.open(source_path).convert("L")) > 127
    result = run_bezier_from_source_edge(edge, compute_final_raster=False)
    tmp_path = out_path.with_name(f"{out_path.stem}.{os.getpid()}.tmp{out_path.suffix}")
    save_compact_bezier_paths(tmp_path, result["final_paths"], image_shape=edge.shape, summary=result["summary"])
    tmp_path.replace(out_path)
    return {
        "status": "processed",
        "sample_id": source_path.stem,
        "summary": result["summary"],
    }


def main() -> int:
    args = parse_args()

    output_batch_dir = args.output_root / args.batch_name
    output_batch_dir.mkdir(parents=True, exist_ok=True)

    source_edges = iter_source_edges(args.source_edge_dir, args.limit)
    if not source_edges:
        print("No source edge files found.", flush=True)
        return 1

    print(
        f"batch={args.batch_name} source_edges={len(source_edges)} "
        f"workers={args.workers} output={output_batch_dir}",
        flush=True,
    )

    pending_items: list[tuple[Path, Path]] = []
    skipped_existing = 0
    for source_edge_path in source_edges:
        output_path = output_batch_dir / f"{source_edge_path.stem}.npz"
        if output_path.exists():
            skipped_existing += 1
            continue
        pending_items.append((source_edge_path, output_path))

    print(
        f"batch={args.batch_name} pending={len(pending_items)} "
        f"skipped_existing={skipped_existing}",
        flush=True,
    )

    if not pending_items:
        elapsed = 0.0
        summary = {
            "batch_name": args.batch_name,
            "source_edge_dir": str(args.source_edge_dir),
            "output_dir": str(output_batch_dir),
            "num_source_edges": len(source_edges),
            "processed": 0,
            "skipped_existing": skipped_existing,
            "failed": 0,
            "elapsed_sec": elapsed,
            "mean_final_paths": 0.0,
            "mean_final_segments": 0.0,
            "samples": [],
        }
        (output_batch_dir / "_batch_summary.json").write_text(json.dumps(summary, indent=2))
        print(
            f"DONE processed=0 skipped_existing={skipped_existing} failed=0 elapsed_sec={elapsed:.1f}",
            flush=True,
        )
        return 0

    processed = 0
    failed = 0
    summaries: list[dict[str, object]] = []
    start_time = time.time()

    pending_queue = deque(pending_items)
    inflight_limit = max(args.workers * 4, 1)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures: dict[Any, Path] = {}

        def _submit_one() -> bool:
            if not pending_queue:
                return False
            source_edge_path, output_path = pending_queue.popleft()
            future = executor.submit(_process_one, str(source_edge_path), str(output_path))
            futures[future] = source_edge_path
            return True

        while len(futures) < inflight_limit and _submit_one():
            pass

        completed = 0
        while futures:
            future = next(as_completed(list(futures.keys())))
            source_edge_path = futures.pop(future)
            sample_id = source_edge_path.stem
            try:
                row = future.result()
                status = row["status"]
                if status == "processed":
                    processed += 1
                    summaries.append({"sample_id": sample_id, **row["summary"]})
                else:
                    skipped_existing += 1
            except Exception as exc:  # pragma: no cover - operational path
                failed += 1
                print(f"[ERROR] {sample_id}: {exc}", file=sys.stderr, flush=True)

            completed += 1
            while len(futures) < inflight_limit and _submit_one():
                pass

            if completed % max(args.log_every, 1) == 0:
                elapsed = time.time() - start_time
                print(
                    f"[{completed}/{len(pending_items)}] processed={processed} "
                    f"skipped_existing={skipped_existing} failed={failed} "
                    f"elapsed_sec={elapsed:.1f}",
                    flush=True,
                )

    elapsed = time.time() - start_time
    summary = {
        "batch_name": args.batch_name,
        "source_edge_dir": str(args.source_edge_dir),
        "output_dir": str(output_batch_dir),
        "num_source_edges": len(source_edges),
        "processed": processed,
        "skipped_existing": skipped_existing,
        "failed": failed,
        "elapsed_sec": elapsed,
        "mean_final_paths": float(np.mean([row["final_path_count"] for row in summaries])) if summaries else 0.0,
        "mean_final_segments": float(np.mean([row["final_segment_count"] for row in summaries])) if summaries else 0.0,
        "samples": summaries,
    }
    (output_batch_dir / "_batch_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"DONE processed={processed} skipped_existing={skipped_existing} "
        f"failed={failed} elapsed_sec={elapsed:.1f}",
        flush=True,
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
