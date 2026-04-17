from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_EXT_PATTERN = r"\.(jpg|jpeg|png|webp)$"


@dataclass(frozen=True)
class JobSpec:
    name: str
    host: str
    stage: str
    batch_name: str
    expected_dir: str
    expected_kind: str
    output_dir: str
    output_kind: str
    pid_file: str | None = None
    log_files: tuple[str, ...] | None = None
    cluster_job_name: str | None = None
    cluster_log_glob: str | None = None


LAB_HOST_CONFIGS = {
    "lab30": {
        "repo_root": "/home/viplab/jiaxin/parametric_edge_prediction",
        "data_root": "/home/devdata/laion/edge_detection",
    },
    "lab34": {
        "repo_root": "/data/jiaxin/parametric_edge_prediction",
        "data_root": "/data/jiaxin/laion/edge_detection",
    },
    "lab21": {
        "repo_root": "/media/jiaxin/parametric_edge_prediction",
        "data_root": "/media/jiaxin/laion/edge_detection",
    },
}

CLUSTER_CONFIG = {
    "repo_root": "/home/user/yc47434/parametric_edge_prediction",
    "data_root": "/home/user/yc47434/laion/edge_detection",
}


class RemoteRunner:
    def __init__(self) -> None:
        self.config = self._load_zshrc_config()

    def _load_zshrc_config(self) -> dict[str, str]:
        zshrc = Path.home() / ".zshrc"
        text = zshrc.read_text(encoding="utf-8")
        cfg: dict[str, str] = {}
        for match in re.finditer(r'^export\s+([A-Z0-9_]+)="([^"]*)"', text, flags=re.MULTILINE):
            cfg[match.group(1)] = match.group(2)
        log34_match = re.search(r'^alias\s+log34="sshpass -p ([^ ]+) ssh viplab@\$\{LAB34_IP\}"', text, flags=re.MULTILINE)
        if log34_match:
            cfg["LAB34_PSWD"] = log34_match.group(1)
        log21_match = re.search(r'^alias\s+log21="sshpass -p ([^ ]+) ssh viplab@\$\{LAB21_IP\}"', text, flags=re.MULTILINE)
        if log21_match:
            cfg["LAB21_PSWD"] = log21_match.group(1)
        return cfg

    def run(self, host: str, remote_cmd: str) -> subprocess.CompletedProcess[str]:
        if host == "lab30":
            args = [
                "sshpass",
                "-p",
                self.config["LAB30_PSWD"],
                "ssh",
                "-o",
                "ConnectTimeout=5",
                "-o",
                "PreferredAuthentications=password",
                "-o",
                "PubkeyAuthentication=no",
                "-o",
                "StrictHostKeyChecking=no",
                f"viplab@{self.config['LAB30_IP']}",
                remote_cmd,
            ]
        elif host == "lab34":
            args = [
                "sshpass",
                "-p",
                self.config["LAB34_PSWD"],
                "ssh",
                "-o",
                "ConnectTimeout=5",
                "-o",
                "PreferredAuthentications=password",
                "-o",
                "PubkeyAuthentication=no",
                "-o",
                "StrictHostKeyChecking=no",
                f"viplab@{self.config['LAB34_IP']}",
                remote_cmd,
            ]
        elif host == "lab21":
            args = [
                "sshpass",
                "-p",
                self.config["LAB21_PSWD"],
                "ssh",
                "-o",
                "ConnectTimeout=5",
                "-o",
                "PreferredAuthentications=password",
                "-o",
                "PubkeyAuthentication=no",
                "-o",
                "StrictHostKeyChecking=no",
                f"viplab@{self.config['LAB21_IP']}",
                remote_cmd,
            ]
        elif host == "cluster":
            inner = (
                f"sshpass -p {shlex.quote(self.config['CLUSTER_PSWD'])} "
                f"ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "
                f"yc47434@{self.config['CLUSTER_IP']} {shlex.quote(remote_cmd)}"
            )
            args = [
                "sshpass",
                "-p",
                self.config["LAB30_PSWD"],
                "ssh",
                "-o",
                "ConnectTimeout=5",
                "-o",
                "PreferredAuthentications=password",
                "-o",
                "PubkeyAuthentication=no",
                "-o",
                "StrictHostKeyChecking=no",
                f"viplab@{self.config['LAB30_IP']}",
                inner,
            ]
        else:
            raise ValueError(f"Unsupported host: {host}")
        return subprocess.run(args, capture_output=True, text=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check source-edge batch progress and ETA.")
    parser.add_argument(
        "--jobs",
        default="all",
        help="Comma-separated job names. Default: all",
    )
    parser.add_argument("--skip-sync", action="store_true", help="Skip syncing results back to lab30 before checking.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    return parser.parse_args()


def batch_sort_key(batch_name: str) -> tuple[int, str]:
    match = re.search(r"batch(\d+)", batch_name)
    if match:
        return int(match.group(1)), batch_name
    return 10**9, batch_name


def make_job_name(host: str, batch_name: str, stage: str) -> str:
    if stage == "source_edge":
        return f"{host}-{batch_name}"
    return f"{host}-{batch_name}-bezier"


def query_remote_lines(runner: RemoteRunner, host: str, cmd: str) -> tuple[list[str], str | None]:
    proc = runner.run(host, cmd)
    if proc.returncode != 0:
        return [], proc.stderr.strip() or proc.stdout.strip() or f"{host} command failed"
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()], None


def discover_lab_specs(runner: RemoteRunner, host: str) -> tuple[list[JobSpec], str | None]:
    cfg = LAB_HOST_CONFIGS[host]
    outputs_root = f"{cfg['repo_root']}/outputs"
    cmd = (
        f"find {shlex.quote(outputs_root)} -maxdepth 1 -mindepth 1 -type d "
        f"\\( -name 'sam2_source_edge_batch*_full_logs' -o "
        f"-name 'sam2_source_edge_batch*_delayed_logs' -o "
        f"-name 'bezier_from_source_edge_batch*_logs' \\) -printf '%f\\n' | sort -u"
    )
    lines, error = query_remote_lines(runner, host, cmd)
    if error:
        return [], error

    source_logs: dict[str, str] = {}
    bezier_logs: dict[str, str] = {}
    for line in lines:
        source_match = re.fullmatch(r"sam2_source_edge_(batch\d+)_(full|delayed)_logs", line)
        if source_match:
            batch_name = source_match.group(1)
            current = source_logs.get(batch_name)
            if current is None or line.endswith("_full_logs"):
                source_logs[batch_name] = line
            continue
        bezier_match = re.fullmatch(r"bezier_from_source_edge_(batch\d+)_logs", line)
        if bezier_match:
            batch_name = bezier_match.group(1)
            bezier_logs[batch_name] = line

    specs: list[JobSpec] = []
    for batch_name, log_dir_name in sorted(source_logs.items(), key=lambda item: batch_sort_key(item[0])):
        specs.append(
            JobSpec(
                name=make_job_name(host, batch_name, "source_edge"),
                host=host,
                stage="source_edge",
                batch_name=batch_name,
                expected_dir=f"{cfg['data_root']}/{batch_name}/images",
                expected_kind="image",
                output_dir=f"{cfg['data_root']}/laion_edge_v3_source_edge/{batch_name}",
                output_kind="png",
                pid_file=f"{outputs_root}/{log_dir_name}/launcher.pid",
                log_files=(
                    f"{outputs_root}/{log_dir_name}/gpu0.log",
                    f"{outputs_root}/{log_dir_name}/gpu1.log",
                ),
            )
        )
    for batch_name, log_dir_name in sorted(bezier_logs.items(), key=lambda item: batch_sort_key(item[0])):
        specs.append(
            JobSpec(
                name=make_job_name(host, batch_name, "bezier"),
                host=host,
                stage="bezier",
                batch_name=batch_name,
                expected_dir=f"{cfg['data_root']}/laion_edge_v3_source_edge/{batch_name}",
                expected_kind="png",
                output_dir=f"{cfg['data_root']}/laion_edge_v3_bezier/{batch_name}",
                output_kind="npz",
                pid_file=f"{outputs_root}/{log_dir_name}/launcher.pid",
                log_files=(f"{outputs_root}/{log_dir_name}/main.log",),
            )
        )
    return specs, None


def discover_cluster_specs(runner: RemoteRunner, cluster_jobs: dict[str, dict[str, str]]) -> tuple[list[JobSpec], str | None]:
    cfg = CLUSTER_CONFIG
    source_root = f"{cfg['data_root']}/laion_edge_v3_source_edge"
    bezier_root = f"{cfg['data_root']}/laion_edge_v3_bezier"
    cmd = (
        f"if [ -d {shlex.quote(source_root)} ]; then "
        f"  find {shlex.quote(source_root)} -maxdepth 1 -mindepth 1 -type d -name 'batch*' -printf 'source\\t%f\\n'; "
        f"fi; "
        f"if [ -d {shlex.quote(bezier_root)} ]; then "
        f"  find {shlex.quote(bezier_root)} -maxdepth 1 -mindepth 1 -type d -name 'batch*' -printf 'bezier\\t%f\\n'; "
        f"fi | sort -u"
    )
    lines, error = query_remote_lines(runner, "cluster", cmd)
    if error:
        return [], error

    source_batches: set[str] = set()
    bezier_batches: set[str] = set()
    for line in lines:
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        stage, batch_name = parts
        if stage == "source":
            source_batches.add(batch_name)
        elif stage == "bezier":
            bezier_batches.add(batch_name)
    for job_name in cluster_jobs:
        source_match = re.fullmatch(r"sam2-src-b(\d+)", job_name)
        if source_match:
            source_batches.add(f"batch{source_match.group(1)}")
        bezier_match = re.fullmatch(r"bezier-b(\d+)", job_name)
        if bezier_match:
            bezier_batches.add(f"batch{bezier_match.group(1)}")

    specs: list[JobSpec] = []
    for batch_name in sorted(source_batches, key=batch_sort_key):
        batch_num = batch_sort_key(batch_name)[0]
        specs.append(
            JobSpec(
                name=make_job_name("cluster", batch_name, "source_edge"),
                host="cluster",
                stage="source_edge",
                batch_name=batch_name,
                expected_dir=f"{cfg['data_root']}/edge_detection/{batch_name}/images",
                expected_kind="image",
                output_dir=f"{cfg['data_root']}/laion_edge_v3_source_edge/{batch_name}",
                output_kind="png",
                cluster_job_name=f"sam2-src-b{batch_num}",
                cluster_log_glob=f"{cfg['repo_root']}/cluster_runs/parametric_edge_prediction/sam2-src-b{batch_num}-*",
            )
        )
    for batch_name in sorted(bezier_batches, key=batch_sort_key):
        batch_num = batch_sort_key(batch_name)[0]
        specs.append(
            JobSpec(
                name=make_job_name("cluster", batch_name, "bezier"),
                host="cluster",
                stage="bezier",
                batch_name=batch_name,
                expected_dir=f"{cfg['data_root']}/laion_edge_v3_source_edge/{batch_name}",
                expected_kind="png",
                output_dir=f"{cfg['data_root']}/laion_edge_v3_bezier/{batch_name}",
                output_kind="npz",
                cluster_job_name=f"bezier-b{batch_num}",
            )
        )
    return specs, None


def discover_jobs(runner: RemoteRunner) -> tuple[dict[str, JobSpec], dict[str, dict[str, str]], list[str]]:
    cluster_jobs = query_cluster_jobs(runner)
    discovered: dict[str, JobSpec] = {}
    errors: list[str] = []
    for host in ("lab30", "lab34", "lab21"):
        specs, error = discover_lab_specs(runner, host)
        if error:
            if host != "lab21":
                errors.append(f"{host}: {error}")
            continue
        for spec in specs:
            discovered[spec.name] = spec
    specs, error = discover_cluster_specs(runner, cluster_jobs)
    if error:
        errors.append(f"cluster: {error}")
    else:
        for spec in specs:
            discovered[spec.name] = spec
    return discovered, cluster_jobs, errors


def run_sync_to_lab30() -> str | None:
    script = Path(__file__).resolve().parent / "sync_completed_results_to_lab30.py"
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    if proc.returncode == 0:
        return None
    return proc.stderr.strip() or proc.stdout.strip() or "sync to lab30 failed"


def lab30_mirror_paths(stage: str, batch_name: str) -> tuple[str, str, str, str]:
    data_root = "/home/devdata/laion/edge_detection"
    if stage == "source_edge":
        return (
            f"{data_root}/{batch_name}/images",
            "image",
            f"{data_root}/laion_edge_v3_source_edge/{batch_name}",
            "png",
        )
    return (
        f"{data_root}/laion_edge_v3_source_edge/{batch_name}",
        "png",
        f"{data_root}/laion_edge_v3_bezier/{batch_name}",
        "npz",
    )


def query_lab30_counts(runner: RemoteRunner, specs: list[JobSpec]) -> tuple[dict[tuple[str, str], tuple[int | None, int | None]], str | None]:
    jobs = []
    for spec in specs:
        expected_dir, expected_kind, output_dir, output_kind = lab30_mirror_paths(spec.stage, spec.batch_name)
        jobs.append(
            {
                "stage": spec.stage,
                "batch": spec.batch_name,
                "expected_dir": expected_dir,
                "expected_kind": expected_kind,
                "output_dir": output_dir,
                "output_kind": output_kind,
            }
        )
    payload = shlex.quote(json.dumps(jobs))
    cmd = (
        "python3 - <<'PY'\n"
        "import json\n"
        "from pathlib import Path\n"
        f"jobs = json.loads({payload})\n"
        "for job in jobs:\n"
        "    expected = Path(job['expected_dir'])\n"
        "    output = Path(job['output_dir'])\n"
        "    if job['expected_kind'] == 'image':\n"
        "        total = len([p for p in expected.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}]) if expected.exists() else -1\n"
        "    else:\n"
        "        total = len(list(expected.glob(f\"*.{job['expected_kind']}\"))) if expected.exists() else -1\n"
        "    done = len(list(output.glob(f\"*.{job['output_kind']}\"))) if output.exists() else 0\n"
        "    print(f\"{job['stage']}\\t{job['batch']}\\t{total}\\t{done}\")\n"
        "PY"
    )
    proc = runner.run("lab30", cmd)
    if proc.returncode != 0:
        return {}, proc.stderr.strip() or proc.stdout.strip() or "lab30 count query failed"
    result: dict[tuple[str, str], tuple[int | None, int | None]] = {}
    for line in proc.stdout.splitlines():
        parts = line.strip().split("\t")
        if len(parts) != 4:
            continue
        stage, batch_name, total_str, done_str = parts
        try:
            total = int(total_str)
            done = int(done_str)
        except ValueError:
            continue
        result[(stage, batch_name)] = (total if total >= 0 else None, done)
    return result, None


def parse_elapsed(text: str | None) -> int | None:
    if not text:
        return None
    text = text.strip()
    if not text:
        return None
    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        days = int(day_text)
    parts = text.split(":")
    if len(parts) != 3:
        return None
    hours, minutes, seconds = (int(x) for x in parts)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = max(int(round(seconds)), 0)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d{hours:02d}h"
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def parse_failed_counts(*texts: str) -> int | None:
    values: list[int] = []
    pattern = re.compile(r"failed=(\d+)")
    for text in texts:
        matches = pattern.findall(text)
        if matches:
            values.append(int(matches[-1]))
    return max(values) if values else None


def query_counts(runner: RemoteRunner, spec: JobSpec) -> tuple[int | None, int | None, str | None]:
    cmd = (
        (
            f"total=$(find {shlex.quote(spec.expected_dir)} -maxdepth 1 -type f "
            f"| grep -Ei {shlex.quote(VALID_EXT_PATTERN)} | wc -l); "
            if spec.expected_kind == "image"
            else f"total=$(find {shlex.quote(spec.expected_dir)} -maxdepth 1 -type f -name '*.{spec.expected_kind}' | wc -l); "
        )
        + f"done=$(find {shlex.quote(spec.output_dir)} -maxdepth 1 -type f -name '*.{spec.output_kind}' | wc -l); "
        f"printf 'TOTAL=%s\\nDONE=%s\\n' \"$total\" \"$done\""
    )
    proc = runner.run(spec.host, cmd)
    if proc.returncode != 0:
        return None, None, proc.stderr.strip() or proc.stdout.strip() or "remote command failed"
    total_match = re.search(r"TOTAL=(\d+)", proc.stdout)
    done_match = re.search(r"DONE=(\d+)", proc.stdout)
    if not total_match or not done_match:
        return None, None, "failed to parse counts"
    return int(total_match.group(1)), int(done_match.group(1)), None


def query_lab_status(
    runner: RemoteRunner,
    spec: JobSpec,
    count_override: tuple[int | None, int | None] | None = None,
) -> dict[str, Any]:
    if count_override is None:
        total, done, error = query_counts(runner, spec)
        if error:
            return {"error": error, "status": "ERROR", "total": total, "done": done}
    else:
        total, done = count_override
    pid_file = spec.pid_file
    cmd_parts = [
        f"status=stopped; elapsed=''; "
        f"if [ -f {shlex.quote(pid_file)} ]; then "
        f"  pid=$(cat {shlex.quote(pid_file)} 2>/dev/null || true); "
        f"  if [ -n \"$pid\" ] && ps -p \"$pid\" >/dev/null 2>&1; then "
        f"    status=RUNNING; "
        f"    elapsed=$(ps -o etimes= -p \"$pid\" | tr -d ' '); "
        f"  fi; "
        f"fi; "
        f"printf 'STATUS=%s\\nELAPSED=%s\\n' \"$status\" \"$elapsed\"; "
    ]
    for idx, log_file in enumerate(spec.log_files or ()):
        cmd_parts.append(
            f"if [ -f {shlex.quote(log_file)} ]; then "
            f"printf 'LOG{idx}_BEGIN\\n'; tail -n 40 {shlex.quote(log_file)}; printf '\\nLOG{idx}_END\\n'; fi; "
        )
    cmd = "".join(cmd_parts)
    proc = runner.run(spec.host, cmd)
    if proc.returncode != 0:
        return {"error": proc.stderr.strip() or proc.stdout.strip() or "remote command failed", "status": "ERROR", "total": total, "done": done}
    status_match = re.search(r"STATUS=(\S+)", proc.stdout)
    elapsed_match = re.search(r"ELAPSED=(\d*)", proc.stdout)
    log_texts = []
    for idx, _ in enumerate(spec.log_files or ()):
        match = re.search(rf"LOG{idx}_BEGIN\n(.*?)\nLOG{idx}_END", proc.stdout, flags=re.DOTALL)
        if match:
            log_texts.append(match.group(1))
    status = status_match.group(1) if status_match else "UNKNOWN"
    elapsed = int(elapsed_match.group(1)) if elapsed_match and elapsed_match.group(1) else None
    failed = parse_failed_counts(*log_texts)
    if status != "RUNNING":
        if done is not None and total is not None and done >= total:
            status = "COMPLETED"
        elif done:
            status = "PARTIAL"
        else:
            status = "IDLE"
    return {"status": status, "elapsed": elapsed, "total": total, "done": done, "failed": failed}


def query_cluster_jobs(runner: RemoteRunner) -> dict[str, dict[str, str]]:
    cmd = "squeue -h -u yc47434 -o '%j|%T|%M|%i|%R'"
    proc = runner.run("cluster", cmd)
    jobs: dict[str, dict[str, str]] = {}
    if proc.returncode != 0:
        return jobs
    for line in proc.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 5:
            continue
        job_name, status, elapsed, job_id, reason = parts
        jobs[job_name] = {
            "status": status,
            "elapsed": elapsed,
            "job_id": job_id,
            "reason": reason,
        }
    return jobs


def query_cluster_status(
    runner: RemoteRunner,
    spec: JobSpec,
    cluster_jobs: dict[str, dict[str, str]],
    count_override: tuple[int | None, int | None] | None = None,
) -> dict[str, Any]:
    if count_override is None:
        total, done, error = query_counts(runner, spec)
        if error:
            return {"error": error, "status": "ERROR", "total": total, "done": done}
    else:
        total, done = count_override
    job_info = cluster_jobs.get(spec.cluster_job_name or "")
    failed = None
    if spec.cluster_log_glob:
        log_cmd = (
            f"latest=$(ls -dt {spec.cluster_log_glob} 2>/dev/null | head -n 1); "
            f"if [ -n \"$latest\" ]; then "
            f"  if [ -f \"$latest/front.log\" ]; then printf 'LOG0_BEGIN\\n'; tail -n 40 \"$latest/front.log\"; printf '\\nLOG0_END\\n'; fi; "
            f"  if [ -f \"$latest/back.log\" ]; then printf 'LOG1_BEGIN\\n'; tail -n 40 \"$latest/back.log\"; printf '\\nLOG1_END\\n'; fi; "
            f"fi"
        )
        proc = runner.run("cluster", log_cmd)
        if proc.returncode == 0:
            log0_match = re.search(r"LOG0_BEGIN\n(.*?)\nLOG0_END", proc.stdout, flags=re.DOTALL)
            log1_match = re.search(r"LOG1_BEGIN\n(.*?)\nLOG1_END", proc.stdout, flags=re.DOTALL)
            failed = parse_failed_counts(
                log0_match.group(1) if log0_match else "",
                log1_match.group(1) if log1_match else "",
            )
    if job_info:
        elapsed = parse_elapsed(job_info["elapsed"])
        return {
            "status": job_info["status"],
            "elapsed": elapsed,
            "total": total,
            "done": done,
            "failed": failed,
            "job_id": job_info["job_id"],
            "reason": job_info["reason"],
        }
    if done is not None and total is not None and done >= total:
        status = "COMPLETED"
    elif done:
        status = "PARTIAL"
    else:
        status = "IDLE"
    return {"status": status, "elapsed": None, "total": total, "done": done, "failed": failed}


def enrich_metrics(row: dict[str, Any]) -> dict[str, Any]:
    total = row.get("total")
    done = row.get("done")
    elapsed = row.get("elapsed")
    rate = None
    eta = None
    progress = None
    remaining = None
    if total is not None and done is not None and total > 0:
        progress = done / total
        remaining = max(total - done, 0)
    if elapsed and done and elapsed > 0:
        rate = done / elapsed
    if rate and remaining is not None and rate > 0 and row.get("status") == "RUNNING":
        eta = remaining / rate
    row["progress"] = progress
    row["remaining"] = remaining
    row["rate_img_per_hour"] = rate * 3600 if rate is not None else None
    row["eta_seconds"] = eta
    return row


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = ["job", "stage", "status", "done/total", "progress", "speed(img/h)", "eta", "failed", "extra"]
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
        table_rows.append([row["name"], row["stage"], row["status"], done_total, progress, speed, eta, failed, " | ".join(extra_parts)])
    widths = [len(h) for h in headers]
    for row in table_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "  ".join("-" * widths[i] for i in range(len(headers)))
    print(header_line)
    print(sep_line)
    for row in table_rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    totals = [r for r in rows if r.get("total") is not None and r.get("done") is not None]
    if totals:
        total_done = sum(int(r["done"]) for r in totals)
        total_total = sum(int(r["total"]) for r in totals)
        print()
        print(f"overall: {total_done}/{total_total} ({(total_done / total_total) * 100:.1f}%)")


def main() -> int:
    args = parse_args()
    sync_errors: list[str] = []
    if not args.skip_sync:
        sync_error = run_sync_to_lab30()
        if sync_error:
            sync_errors.append(f"sync: {sync_error}")
    runner = RemoteRunner()
    discovered_jobs, cluster_jobs, discovery_errors = discover_jobs(runner)
    if args.jobs == "all":
        selected = list(discovered_jobs.values())
    else:
        names = [x.strip() for x in args.jobs.split(",") if x.strip()]
        missing = [n for n in names if n not in discovered_jobs]
        if missing:
            print(f"Unknown jobs: {', '.join(missing)}", file=sys.stderr)
            return 1
        selected = [discovered_jobs[name] for name in names]
    selected.sort(key=lambda spec: (spec.host, batch_sort_key(spec.batch_name), 0 if spec.stage == "source_edge" else 1))
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
        row["stage"] = spec.stage
        rows.append(enrich_metrics(row))
    for error in discovery_errors + sync_errors:
        rows.append(
            {
                "name": error.split(":", 1)[0],
                "stage": "discovery",
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
