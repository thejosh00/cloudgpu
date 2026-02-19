"""Rsync remote scripts to instance."""

from __future__ import annotations

import subprocess
from pathlib import Path


def get_remote_dir() -> Path:
    """Get the path to the local remote/ scripts directory."""
    return Path(__file__).resolve().parent.parent / "remote"


def sync_remote(host: str, persistent_dir: str) -> None:
    """Rsync the remote scripts to the instance's persistent directory.

    Copies src/cloudgpu/remote/ -> <persistent>/cloudgpu/remote/
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
