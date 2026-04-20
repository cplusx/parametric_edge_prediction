from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Callable

try:
    from remote_hosts import run_lab30, wrap_cluster_cmd
except ModuleNotFoundError:  # pragma: no cover
    from scripts.remote_hosts import run_lab30, wrap_cluster_cmd


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


def _run_remote_sync_script(
    *,
    run_remote: Callable[[str], subprocess.CompletedProcess[str]],
    label: str,
    repo: str,
    branch: str,
) -> tuple[str, str]:
    inner = f"""set -euo pipefail
cd {repo}
git fetch origin
if git show-ref --verify --quiet refs/heads/{branch}; then
  git checkout {branch}
else
  git checkout -b {branch} origin/{branch}
fi
dirty="$(git status --porcelain)"
if [ -n "$dirty" ]; then
  echo "__ACTION__ reset_dirty"
  printf '%s\n' "$dirty"
  git reset --hard origin/{branch}
  git clean -fd
else
  current_head="$(git rev-parse HEAD)"
  upstream_head="$(git rev-parse origin/{branch})"
  if [ "$current_head" != "$upstream_head" ]; then
    echo "__ACTION__ merge_committed"
    git merge --no-edit -X theirs origin/{branch}
  else
    echo "__ACTION__ fast_forward_or_already_synced"
  fi
fi
echo "__HEAD__ $(git rev-parse HEAD)"
"""
    proc = run_remote(inner)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} sync failed: {(proc.stderr or proc.stdout).strip()}")
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    action = next((line.split(maxsplit=1)[1] for line in lines if line.startswith("__ACTION__ ")), "unknown")
    head = next((line.split(maxsplit=1)[1] for line in lines if line.startswith("__HEAD__ ")), "")
    if not head:
        raise RuntimeError(f"{label} sync failed: missing resulting HEAD in output")
    return action, head


def sync_remote_repo(label: str, repo: str, branch: str, *, via_cluster: bool = False) -> None:
    def _runner(cmd: str) -> subprocess.CompletedProcess[str]:
        remote_cmd = wrap_cluster_cmd(cmd) if via_cluster else cmd
        return run_lab30(remote_cmd, timeout=120)

    action, head = _run_remote_sync_script(
        run_remote=_runner,
        label=label,
        repo=repo,
        branch=branch,
    )
    print(f"[{label}] action={action} head={head}")


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
