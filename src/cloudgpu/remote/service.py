"""systemd service management for app servers. Stdlib-only; runs on the instance.

A *service spec* is a dict: {"name", "exec_start", "workdir", "port"}. Apps that are
long-running servers (e.g. ComfyUI) return one from ``AppInstaller.service_spec()``; we
install it as a system unit so it auto-starts on boot and restarts on crash.

The instance's filesystem (where exec_start/workdir live) is the persistent identity, but
``/etc/systemd/system`` is instance-local, so the unit is re-created on every fresh
instance through the normal install/recover path. Needs passwordless sudo (Lambda Ubuntu).
"""

from __future__ import annotations

import os
from typing import Any

from .utils import log, run


def unit_text(spec: dict[str, Any]) -> str:
    """Render the systemd unit for a service spec (pure; testable without sudo)."""
    return f"""[Unit]
Description=cloudgpu {spec['name']}
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
Environment=HOME=/home/ubuntu
WorkingDirectory={spec['workdir']}
ExecStart={spec['exec_start']}
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
"""


def install_service(spec: dict[str, Any]) -> None:
    """Write + enable + start the systemd unit for ``spec`` (idempotent)."""
    name = spec["name"]
    tmp = f"/tmp/cloudgpu-{name}.service"
    with open(tmp, "w") as f:
        f.write(unit_text(spec))
    run(["sudo", "install", "-m", "644", tmp, f"/etc/systemd/system/{name}.service"])
    os.remove(tmp)
    run(["sudo", "systemctl", "daemon-reload"])
    run(["sudo", "systemctl", "enable", "--now", name])
    log(f"Service '{name}' enabled and started (systemd, restarts on boot/crash).")


def restart_service(name: str) -> None:
    """Restart a service (e.g. after provisioning adds custom nodes)."""
    run(["sudo", "systemctl", "restart", name])
    log(f"Service '{name}' restarted.")


def service_active(name: str) -> str:
    """Return the service's systemd state ('active', 'inactive', ...); best-effort."""
    try:
        result = run(["systemctl", "is-active", name], capture=True, check=False)
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"
