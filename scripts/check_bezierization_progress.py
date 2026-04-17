from __future__ import annotations

import argparse
import json
from typing import Any

from check_source_edge_progress import (
    RemoteRunner,
    batch_sort_key,
    discover_jobs,
    enrich_metrics,
    format_duration,
    query_lab30_counts,
    query_cluster_status,
    query_lab_status,
    run_sync_to_lab30,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check bezierization batch progress and ETA.")
    parser.add_argument(
        "--jobs",
        default="all",
        help="Comma-separated bezier job names. Default: all discovered bezier jobs",
    )
    parser.add_argument("--skip-sync", action="store_true", help="Skip syncing results back to lab30 before checking.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    return parser.parse_args()


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = ["job", "status", "done/total", "progress", "speed(img/h)", "eta", "failed", "extra"]
    table_rows: list[list[str]] = []
    for row in rows:
        total = row.get("total")
        done = row.get("done")
        done_total = "-" if total is None or done is None else f"{done}/{total}"
        progress = "-" if row.get("progress") is None else f"{row['progress'] * 100:.1f}%"
        speed = "-" if row.get("rate_img_per_hour") is None else f"{row['rate_img_per_hour']:.1f}"
        eta = format_duration(row.get("eta_seconds"))
        failed = "-" if row.get("failed") is None else str(row["failed"])
        extra_parts = []
        if row.get("job_id"):
            extra_parts.append(f"id={row['job_id']}")
        if row.get("reason") and row["status"] == "PENDING":
            extra_parts.append(row["reason"])
        if row.get("error"):
            extra_parts.append(row["error"])
        table_rows.append([row["name"], row["status"], done_total, progress, speed, eta, failed, " | ".join(extra_parts)])

    widths = [len(h) for h in headers]
    for row in table_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in table_rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))

    totals = [r for r in rows if r.get("total") is not None and r.get("done") is not None]
    if totals:
        total_done = sum(int(r["done"]) for r in totals)
        total_total = sum(int(r["total"]) for r in totals)
        print()
        print(f"overall bezier: {total_done}/{total_total} ({(total_done / total_total) * 100:.1f}%)")


def main() -> int:
    args = parse_args()
    sync_errors: list[str] = []
    if not args.skip_sync:
        sync_error = run_sync_to_lab30()
        if sync_error:
            sync_errors.append(f"sync: {sync_error}")
    runner = RemoteRunner()
    discovered_jobs, cluster_jobs, discovery_errors = discover_jobs(runner)

    bezier_jobs = {
        name: spec for name, spec in discovered_jobs.items() if spec.stage == "bezier"
    }
    if args.jobs == "all":
        selected = list(bezier_jobs.values())
    else:
        names = [x.strip() for x in args.jobs.split(",") if x.strip()]
        missing = [n for n in names if n not in bezier_jobs]
        if missing:
            print(f"Unknown bezier jobs: {', '.join(missing)}")
            return 1
        selected = [bezier_jobs[name] for name in names]

    selected.sort(key=lambda spec: (spec.host, batch_sort_key(spec.batch_name)))
    lab30_counts, lab30_count_error = query_lab30_counts(runner, selected)
    if lab30_count_error:
        sync_errors.append(f"lab30-counts: {lab30_count_error}")

    rows: list[dict[str, Any]] = []
    for spec in selected:
        count_override = lab30_counts.get((spec.stage, spec.batch_name))
        if spec.host == "cluster":
            row = query_cluster_status(runner, spec, cluster_jobs, count_override=count_override)
        else:
            row = query_lab_status(runner, spec, count_override=count_override)
        row["name"] = spec.name
        rows.append(enrich_metrics(row))

    for error in discovery_errors + sync_errors:
        rows.append(
            {
                "name": error.split(":", 1)[0],
                "status": "ERROR",
                "total": None,
                "done": None,
                "failed": None,
                "elapsed": None,
                "progress": None,
                "remaining": None,
                "rate_img_per_hour": None,
                "eta_seconds": None,
                "error": error,
            }
        )

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
