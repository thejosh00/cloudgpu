"""Is this host a cloudgpu instance? Stdlib-only.

The real installers change the host they run on (``useradd``, systemd units, venvs, app
directories), so they refuse to run anywhere that isn't a cloudgpu instance. An instance is
recognised by REAL_MARKER, an empty root-owned file the local tool writes over SSH (with
``sudo -n``) every time it syncs the remote tool (``cloudgpu.local.sync.sync_remote``, which
``setup``, ``up``, ``install``, ``recover`` and ``provision`` all run first).

Why /var/lib/cloudgpu/instance:
- It is on the instance's own root disk, not on the persistent filesystem, so it belongs to
  this instance and dies with it; a filesystem that moves to another instance doesn't carry it.
- It needs root to create, and macOS has no /var/lib at all, so a developer's Mac never has it
  unless someone deliberately runs ``sudo mkdir -p /var/lib/cloudgpu && sudo touch ...``.

The guard reads MARKER, which is REAL_MARKER; only tests reassign it (to a temp path). It
reads no environment variable, so nothing left set in a shell can switch it off.
"""

from __future__ import annotations

import os

REAL_MARKER = "/var/lib/cloudgpu/instance"
MARKER = REAL_MARKER


class NotOnInstance(RuntimeError):
    """Raised before an installer changes a host that isn't a cloudgpu instance."""


def write_marker_command() -> str:
    """Shell command (run as the SSH user) that leaves the marker; plain coreutils + sudo."""
    directory = os.path.dirname(REAL_MARKER)
    return f"sudo -n mkdir -p {directory} && sudo -n touch {REAL_MARKER}"


def require_instance() -> None:
    """Raise NotOnInstance unless the instance marker exists."""
    if not os.path.isfile(MARKER):
        raise NotOnInstance(
            f"Refusing to install or recover apps: this host is not a cloudgpu instance (missing marker {MARKER}). "
            "The installers change the host (create the cloudgpu-app user, systemd units, venvs), "
            "so they only run on an instance prepared from your machine: run "
            "`cloudgpu setup <host>` (or `cloudgpu up`, `install`, `recover`), which writes the "
            "marker over SSH."
        )
