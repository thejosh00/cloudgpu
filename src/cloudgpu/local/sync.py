"""Rsync remote scripts to instance."""

from __future__ import annotations

import subprocess
from pathlib import Path


def get_remote_dir() -> Path:
    """Get the path to the local remote/ scripts directory."""
    return Path(__file__).resolve().parent.parent / "remote"


def sync_remote(host: str, persistent_dir: str) -> None:
    """Rsync the remote scripts to the instance's persistent directory.

    Copies src/cloudgpu/remote/ -> <persistent>//cloudgpu/remote/
    """
    local_dir = str(get_remote_dir()) + "/"
    remote_path = f"{persistent_dir}/cloudgpu/remote/"

    # Ensure target directory exists
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, f"mkdir -p {remote_path}"],
        check=True,
        timeout=15,
    )

    subprocess.run(
        [
            "rsync",
            "-az",
            "--delete",
            local_dir,
            f"{host}:{remote_path}",
        ],
        check=True,
        timeout=60,
    )


def copy_file(local_path: str, host: str, remote_path: str) -> None:
    """Rsync a single local file to ``remote_path`` (home-relative if it has no dir part).

    The file content travels over the rsync/ssh stream, never as a command-line argument,
    so it's safe for secrets.
    """
    if "/" in remote_path:
        remote_dir = remote_path.rsplit("/", 1)[0]
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", host, f"mkdir -p {remote_dir}"],
            check=True,
            timeout=15,
        )
    subprocess.run(
        ["rsync", "-az", local_path, f"{host}:{remote_path}"],
        check=True,
        timeout=60,
    )


def copy_dir(local_dir: str, host: str, remote_dir: str, exclude: list[str] | None = None) -> None:
    """Mirror a local directory's contents to ``remote_dir`` on the instance (rsync).

    ``exclude`` is a list of rsync patterns to omit (e.g. tool state, secrets, VCS).
    """
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, f"mkdir -p {remote_dir}"],
        check=True,
        timeout=15,
    )
    cmd = ["rsync", "-az", "--delete"]
    for pat in exclude or []:
        cmd += ["--exclude", pat]
    cmd += [local_dir.rstrip("/") + "/", f"{host}:{remote_dir}/"]
    subprocess.run(cmd, check=True, timeout=300)
