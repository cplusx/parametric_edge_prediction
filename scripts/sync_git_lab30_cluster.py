from __future__ import annotations

import argparse
import subprocess
import sys

from remote_hosts import run_lab30, wrap_cluster_cmd


LOCAL_REPO = "/Users/jiaxincheng/Desktop/expts/parametric_edge_prediction"
LAB30_REPO = "/home/viplab/jiaxin/parametric_edge_prediction"
CLUSTER_REPO = "/home/user/yc47434/parametric_edge_prediction"


def run_local(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=LOCAL_REPO)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or "local command failed")
    return proc.stdout.strip()


def ensure_local_git_ready() -> tuple[str, str]:
    dirty = run_local(["git", "status", "--porcelain"])
    if dirty:
        raise RuntimeError("local git tree is dirty; commit or stash first")
    branch = run_local(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    head = run_local(["git", "rev-parse", "HEAD"])
    upstream = run_local(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    upstream_head = run_local(["git", "rev-parse", "@{u}"])
    if head != upstream_head:
        raise RuntimeError(
            f"local HEAD {head[:12]} is not pushed to upstream {upstream} ({upstream_head[:12]}); push first"
        )
    return branch, head


def sync_remote_repo(label: str, repo: str, branch: str, *, via_cluster: bool = False) -> None:
    inner = (
        f"set -euo pipefail; "
        f"cd {repo}; "
        f"test -z \"$(git status --porcelain)\"; "
        f"git fetch origin; "
        f"git checkout {branch}; "
        f"git pull --ff-only origin {branch}; "
        f"printf '%s\\n' $(git rev-parse HEAD)"
    )
    remote_cmd = wrap_cluster_cmd(inner) if via_cluster else inner
    proc = run_lab30(remote_cmd, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} sync failed: {(proc.stderr or proc.stdout).strip()}")
    print(f"[{label}] {proc.stdout.strip().splitlines()[-1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync current committed branch to lab30 and cluster via git.")
    args = parser.parse_args()
    del args

    branch, head = ensure_local_git_ready()
    print(f"[local] branch={branch} head={head}")
    sync_remote_repo("lab30", LAB30_REPO, branch, via_cluster=False)
    sync_remote_repo("cluster", CLUSTER_REPO, branch, via_cluster=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
