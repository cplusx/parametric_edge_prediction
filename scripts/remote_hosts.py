from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HostConfig:
    lab30_ip: str
    lab30_pswd: str
    lab34_ip: str
    lab34_pswd: str
    cluster_ip: str
    cluster_pswd: str


def _extract_alias_password(text: str, alias_name: str) -> str:
    match = re.search(
        rf'^alias\s+{re.escape(alias_name)}="sshpass -p ([^ ]+) ssh [^"]+"',
        text,
        flags=re.MULTILINE,
    )
    if not match:
        raise KeyError(f"missing password for alias {alias_name}")
    return match.group(1)


def load_host_config() -> HostConfig:
    zshrc = Path.home() / ".zshrc"
    text = zshrc.read_text(encoding="utf-8")
    cfg: dict[str, str] = {}
    for match in re.finditer(r'^export\s+([A-Z0-9_]+)="([^"]*)"', text, flags=re.MULTILINE):
        cfg[match.group(1)] = match.group(2)
    if "LAB34_PSWD" not in cfg:
        cfg["LAB34_PSWD"] = _extract_alias_password(text, "log34")
    return HostConfig(
        lab30_ip=cfg["LAB30_IP"],
        lab30_pswd=cfg["LAB30_PSWD"],
        lab34_ip=cfg["LAB34_IP"],
        lab34_pswd=cfg["LAB34_PSWD"],
        cluster_ip=cfg["CLUSTER_IP"],
        cluster_pswd=cfg["CLUSTER_PSWD"],
    )


def run_lab30(remote_cmd: str, *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    cfg = load_host_config()
    args = [
        "sshpass",
        "-p",
        cfg.lab30_pswd,
        "ssh",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "PreferredAuthentications=password",
        "-o",
        "PubkeyAuthentication=no",
        "-o",
        "StrictHostKeyChecking=no",
        f"viplab@{cfg.lab30_ip}",
        remote_cmd,
    ]
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def run_lab34(remote_cmd: str, *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    cfg = load_host_config()
    args = [
        "sshpass",
        "-p",
        cfg.lab34_pswd,
        "ssh",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "PreferredAuthentications=password",
        "-o",
        "PubkeyAuthentication=no",
        "-o",
        "StrictHostKeyChecking=no",
        f"viplab@{cfg.lab34_ip}",
        remote_cmd,
    ]
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def wrap_cluster_cmd(remote_cmd: str) -> str:
    cfg = load_host_config()
    return (
        f"sshpass -p {shlex.quote(cfg.cluster_pswd)} "
        f"ssh -o ConnectTimeout=5 -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no "
        f"yc47434@{cfg.cluster_ip} {shlex.quote(remote_cmd)}"
    )
