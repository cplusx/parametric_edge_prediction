from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SyncSpec:
    name: str
    source_host: str
    stage: str
    source_dir: str
    expected_dir: str
    expected_ext: str
    dest_dir: str
    sync_ext: str


def make_spec(
    *,
    host: str,
    stage: str,
    batch: int,
    source_dir: str,
    expected_dir: str,
    expected_ext: str,
    dest_dir: str,
    sync_ext: str,
) -> SyncSpec:
    batch_name = f"batch{batch}"
    name = f"{host}-{batch_name}-{stage}"
    return SyncSpec(
        name=name,
        source_host=host,
        stage=stage,
        source_dir=source_dir,
        expected_dir=expected_dir,
        expected_ext=expected_ext,
        dest_dir=dest_dir,
        sync_ext=sync_ext,
    )


SYNC_SPECS: dict[str, SyncSpec] = {}

for batch in (1, 2, 3, 4, 5, 6):
    spec = make_spec(
        host="cluster",
        stage="source_edge",
        batch=batch,
        source_dir=f"/home/user/yc47434/laion/edge_detection/laion_edge_v3_source_edge/batch{batch}",
        expected_dir=f"/home/user/yc47434/laion/edge_detection/edge_detection/batch{batch}/images",
        expected_ext="image",
        dest_dir=f"/home/devdata/laion/edge_detection/laion_edge_v3_source_edge/batch{batch}",
        sync_ext="png",
    )
    SYNC_SPECS[spec.name] = spec

for batch in (2, 3, 5, 6, 9, 10, 11, 12):
    spec = make_spec(
        host="cluster",
        stage="bezier",
        batch=batch,
        source_dir=f"/home/user/yc47434/laion/edge_detection/laion_edge_v3_bezier/batch{batch}",
        expected_dir=f"/home/user/yc47434/laion/edge_detection/laion_edge_v3_source_edge/batch{batch}",
        expected_ext="png",
        dest_dir=f"/home/devdata/laion/edge_detection/laion_edge_v3_bezier/batch{batch}",
        sync_ext="npz",
    )
    SYNC_SPECS[spec.name] = spec

for batch in (11, 12, 13):
    spec = make_spec(
        host="lab34",
        stage="source_edge",
        batch=batch,
        source_dir=f"/data/jiaxin/laion/edge_detection/laion_edge_v3_source_edge/batch{batch}",
        expected_dir=f"/data/jiaxin/laion/edge_detection/batch{batch}/images",
        expected_ext="image",
        dest_dir=f"/home/devdata/laion/edge_detection/laion_edge_v3_source_edge/batch{batch}",
        sync_ext="png",
    )
    SYNC_SPECS[spec.name] = spec

for batch in (11, 12, 13):
    spec = make_spec(
        host="lab34",
        stage="bezier",
        batch=batch,
        source_dir=f"/data/jiaxin/laion/edge_detection/laion_edge_v3_bezier/batch{batch}",
        expected_dir=f"/data/jiaxin/laion/edge_detection/laion_edge_v3_source_edge/batch{batch}",
        expected_ext="png",
        dest_dir=f"/home/devdata/laion/edge_detection/laion_edge_v3_bezier/batch{batch}",
        sync_ext="npz",
    )
    SYNC_SPECS[spec.name] = spec

spec = make_spec(
    host="lab21",
    stage="bezier",
    batch=4,
    source_dir="/media/jiaxin/laion/edge_detection/laion_edge_v3_bezier/batch4",
    expected_dir="/media/jiaxin/laion/edge_detection/laion_edge_v3_source_edge/batch4",
    expected_ext="png",
    dest_dir="/home/devdata/laion/edge_detection/laion_edge_v3_bezier/batch4",
    sync_ext="npz",
)
SYNC_SPECS[spec.name] = spec
DEFAULT_SYNC_SPEC_NAMES = [name for name in SYNC_SPECS if not name.startswith("lab21-")]


class RemoteRunner:
    def __init__(self) -> None:
        self.config = self._load_zshrc_config()

    def _load_zshrc_config(self) -> dict[str, str]:
        zshrc = Path.home() / ".zshrc"
        text = zshrc.read_text(encoding="utf-8")
        cfg: dict[str, str] = {}
        for match in re.finditer(r'^export\s+([A-Z0-9_]+)="([^"]*)"', text, flags=re.MULTILINE):
            cfg[match.group(1)] = match.group(2)
        for alias_name, key in (
            ("log30", "LAB30_PSWD"),
            ("log34", "LAB34_PSWD"),
            ("log21", "LAB21_PSWD"),
        ):
            match = re.search(
                rf'^alias\s+{alias_name}="sshpass -p ([^ ]+) ssh viplab@\$\{{[A-Z0-9_]+_IP\}}"',
                text,
                flags=re.MULTILINE,
            )
            if match and key not in cfg:
                cfg[key] = match.group(1)
        return cfg

    def run_lab30(self, remote_cmd: str) -> subprocess.CompletedProcess[str]:
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
        last_proc: subprocess.CompletedProcess[str] | None = None
        for attempt in range(3):
            proc = subprocess.run(args, capture_output=True, text=True)
            last_proc = proc
            if proc.returncode == 0:
                return proc
            err = (proc.stderr or proc.stdout).strip()
            if "Permission denied" not in err or attempt == 2:
                return proc
            time.sleep(0.5)
        assert last_proc is not None
        return last_proc

    def _lab30_wrap_host_cmd(self, host: str, remote_cmd: str) -> str:
        if host == "cluster":
            return (
                f"sshpass -p {shlex.quote(self.config['CLUSTER_PSWD'])} "
                f"ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "
                f"yc47434@{self.config['CLUSTER_IP']} {shlex.quote(remote_cmd)}"
            )
        if host == "lab34":
            return (
                f"sshpass -p {shlex.quote(self.config['LAB34_PSWD'])} "
                f"ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "
                f"-o UserKnownHostsFile=/dev/null "
                f"viplab@{self.config['LAB34_IP']} {shlex.quote(remote_cmd)}"
            )
        if host == "lab21":
            return (
                f"sshpass -p {shlex.quote(self.config['LAB21_PSWD'])} "
                f"ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "
                f"-o UserKnownHostsFile=/dev/null "
                f"viplab@{self.config['LAB21_IP']} {shlex.quote(remote_cmd)}"
            )
        raise ValueError(f"Unsupported source host: {host}")

    def count_remote(self, host: str, directory: str, kind: str) -> tuple[int | None, str | None]:
        if kind == "image":
            cmd = (
                f"find {shlex.quote(directory)} -maxdepth 1 -type f "
                f"\\( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.webp' \\) | wc -l"
            )
        else:
            cmd = f"find {shlex.quote(directory)} -maxdepth 1 -type f -name '*.{kind}' | wc -l"
        proc = self.run_lab30(self._lab30_wrap_host_cmd(host, cmd))
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout).strip()
            return None, err or f"count failed on {host}"
        try:
            return int(proc.stdout.strip().splitlines()[-1]), None
        except Exception:
            return None, f"failed to parse count on {host}: {proc.stdout.strip()}"

    def count_lab30(self, directory: str, kind: str) -> tuple[int | None, str | None]:
        if kind == "image":
            cmd = (
                f"find {shlex.quote(directory)} -maxdepth 1 -type f "
                f"\\( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.webp' \\) | wc -l"
            )
        else:
            cmd = f"find {shlex.quote(directory)} -maxdepth 1 -type f -name '*.{kind}' | wc -l"
        proc = self.run_lab30(cmd)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout).strip()
            return None, err or "count failed on lab30"
        try:
            return int(proc.stdout.strip().splitlines()[-1]), None
        except Exception:
            return None, f"failed to parse count on lab30: {proc.stdout.strip()}"

    def sync_to_lab30(self, spec: SyncSpec) -> subprocess.CompletedProcess[str]:
        mkdir_cmd = f"mkdir -p {shlex.quote(spec.dest_dir)}"
        if spec.source_host == "cluster":
            rsync_cmd = (
                f"sshpass -p {shlex.quote(self.config['CLUSTER_PSWD'])} "
                f"rsync -a -e \"ssh -o StrictHostKeyChecking=no\" "
                f"yc47434@{self.config['CLUSTER_IP']}:{shlex.quote(spec.source_dir)}/ "
                f"{shlex.quote(spec.dest_dir)}/"
            )
        elif spec.source_host == "lab34":
            rsync_cmd = (
                f"sshpass -p {shlex.quote(self.config['LAB34_PSWD'])} "
                f"rsync -a -e \"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null\" "
                f"viplab@{self.config['LAB34_IP']}:{shlex.quote(spec.source_dir)}/ "
                f"{shlex.quote(spec.dest_dir)}/"
            )
        elif spec.source_host == "lab21":
            rsync_cmd = (
                f"sshpass -p {shlex.quote(self.config['LAB21_PSWD'])} "
                f"rsync -a -e \"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null\" "
                f"viplab@{self.config['LAB21_IP']}:{shlex.quote(spec.source_dir)}/ "
                f"{shlex.quote(spec.dest_dir)}/"
            )
        else:
            raise ValueError(f"Unsupported source host: {spec.source_host}")
        return self.run_lab30(f"{mkdir_cmd} && {rsync_cmd}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incrementally sync source_edge / bezier results from remote machines back to lab30."
    )
    parser.add_argument(
        "--specs",
        default="all",
        help="Comma-separated spec names. Default: all",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report what would be synced.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a text summary.")
    return parser.parse_args()


def resolve_specs(specs_arg: str) -> list[SyncSpec]:
    if specs_arg == "all":
        return [SYNC_SPECS[name] for name in DEFAULT_SYNC_SPEC_NAMES]
    names = [name.strip() for name in specs_arg.split(",") if name.strip()]
    missing = [name for name in names if name not in SYNC_SPECS]
    if missing:
        raise SystemExit(f"Unknown spec(s): {', '.join(missing)}")
    return [SYNC_SPECS[name] for name in names]


def evaluate_spec(runner: RemoteRunner, spec: SyncSpec, dry_run: bool) -> dict[str, Any]:
    src_count, src_err = runner.count_remote(spec.source_host, spec.source_dir, spec.sync_ext)
    expected_count, expected_err = runner.count_remote(spec.source_host, spec.expected_dir, spec.expected_ext)
    dest_count, dest_err = runner.count_lab30(spec.dest_dir, spec.sync_ext)

    result: dict[str, Any] = {
        "name": spec.name,
        "source_host": spec.source_host,
        "stage": spec.stage,
        "source_dir": spec.source_dir,
        "dest_dir": spec.dest_dir,
        "source_count": src_count,
        "expected_count": expected_count,
        "dest_count": dest_count,
        "complete": False,
        "action": "skip",
        "status": "OK",
    }

    errors = [err for err in (src_err, expected_err, dest_err) if err]
    if errors:
        result["status"] = "ERROR"
        result["errors"] = errors
        return result

    complete = expected_count is not None and src_count is not None and src_count >= expected_count
    result["complete"] = complete
    if src_count == 0:
        result["action"] = "nothing_to_sync"
        return result

    if dest_count is not None and src_count is not None and dest_count >= src_count:
        result["action"] = "up_to_date"
        return result

    result["action"] = "sync"
    if dry_run:
        return result

    proc = runner.sync_to_lab30(spec)
    result["sync_returncode"] = proc.returncode
    result["sync_stdout"] = proc.stdout.strip()[-1000:]
    result["sync_stderr"] = proc.stderr.strip()[-1000:]
    if proc.returncode != 0:
        result["status"] = "ERROR"
        result["action"] = "sync_failed"
        return result

    new_dest_count, new_dest_err = runner.count_lab30(spec.dest_dir, spec.sync_ext)
    result["dest_count_after"] = new_dest_count
    if new_dest_err:
        result["status"] = "ERROR"
        result["errors"] = [new_dest_err]
    return result


def main() -> int:
    args = parse_args()
    specs = resolve_specs(args.specs)
    runner = RemoteRunner()
    results = [evaluate_spec(runner, spec, dry_run=args.dry_run) for spec in specs]

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for row in results:
            msg = (
                f"{row['name']}: status={row['status']} action={row['action']} "
                f"source={row.get('source_count')} expected={row.get('expected_count')} "
                f"dest={row.get('dest_count')}"
            )
            if "dest_count_after" in row:
                msg += f" dest_after={row['dest_count_after']}"
            if row.get("errors"):
                msg += f" errors={row['errors']}"
            print(msg)

    return 0 if all(row["status"] == "OK" for row in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
