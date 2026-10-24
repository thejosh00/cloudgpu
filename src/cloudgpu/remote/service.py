"""systemd service management for app servers. Stdlib-only; runs on the instance.

A *service spec* is a dict: {"name", "exec_start", "workdir", "port"} plus an optional
"rw_paths" list (the only directories the app may write; defaults to [workdir]). Apps that are
long-running servers (e.g. ComfyUI) return one from ``AppInstaller.service_spec()``; we
install it as a system unit so it auto-starts on boot and restarts on crash.

The instance's filesystem (where exec_start/workdir live) is the persistent identity, but
``/etc/systemd/system`` is instance-local, so the unit is re-created on every fresh
instance through the normal install/recover path. Needs passwordless sudo (Lambda Ubuntu).

App services never run as ``ubuntu`` (which has passwordless sudo, and so could read
the root-only Lambda key in /etc/cloudgpu). They run as the unprivileged system user
``cloudgpu-app`` (no login shell, no sudo) inside a systemd sandbox: the whole OS is
read-only, /home is hidden, and only the app's own dirs on the persistent filesystem plus
its private HOME/tmp are writable. Provisioning, setup and auto-terminate stay ubuntu/root.

Persistent filesystems written by earlier (ubuntu-run) versions are shared through the
``cloudgpu-app`` group: ``migrate_ownership`` (idempotent, run on every service install)
gives the group read/write on the app's dirs, with setgid dirs so new files inherit it;
``ubuntu`` joins the group so provisioning can keep writing there too.
"""

from __future__ import annotations

import os
import pwd
from typing import Any

from .utils import log, run


APP_USER = "cloudgpu-app"
APP_HOME = f"/var/lib/{APP_USER}"  # instance-local; owned by APP_USER (caches live here)


def rw_paths(spec: dict[str, Any]) -> list[str]:
    """The app's writable dirs on the persistent filesystem (excluding its HOME)."""
    return list(spec.get("rw_paths") or [spec["workdir"]])


def unit_text(spec: dict[str, Any]) -> str:
    """Render the systemd unit for a service spec (pure; testable without sudo)."""
    writable = " ".join([*rw_paths(spec), APP_HOME])
    return f"""[Unit]
Description=cloudgpu {spec['name']}
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User={APP_USER}
Group={APP_USER}
UMask=0002
Environment=HOME={APP_HOME}
WorkingDirectory={spec['workdir']}
ExecStart={spec['exec_start']}
Restart=always
RestartSec=3
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
ReadWritePaths={writable}

[Install]
WantedBy=multi-user.target
"""


def ensure_app_user() -> None:
    """Create the unprivileged ``cloudgpu-app`` user + its HOME (idempotent).

    System user, own group, no login shell, not in sudo. The provisioning user
    (``ubuntu``) joins its group so both can write the shared app dirs.
    """
    try:
        pwd.getpwnam(APP_USER)
    except KeyError:
        run(["sudo", "useradd", "--system", "--user-group", "--home-dir", APP_HOME,
             "--no-create-home", "--shell", "/usr/sbin/nologin", APP_USER])
        log(f"Created unprivileged service user '{APP_USER}'.")
    run(["sudo", "install", "-d", "-o", APP_USER, "-g", APP_USER, "-m", "750", APP_HOME])
    run(["sudo", "usermod", "-aG", APP_USER, pwd.getpwuid(os.getuid()).pw_name])


def migrate_ownership(paths: list[str]) -> None:
    """Share ``paths`` with the ``cloudgpu-app`` group (idempotent; safe to re-run).

    Group gets read/write (models readable; outputs, custom_nodes, venvs writable);
    dirs get setgid so files created later by either user keep the group. ``-h`` so a
    symlink planted by app code can't redirect the chgrp onto a system file (chmod -R
    already skips symlinks).
    """
    for path in paths:
        if not os.path.isdir(path):
            continue
        run(["sudo", "chgrp", "-hR", APP_USER, path])
        run(["sudo", "chmod", "-R", "g+rwX", path])
        run(["sudo", "find", path, "-type", "d", "-exec", "chmod", "g+s", "{}", "+"])


def install_service(spec: dict[str, Any]) -> None:
    """Write + enable + start the systemd unit for ``spec`` (idempotent)."""
    name = spec["name"]
    ensure_app_user()
    migrate_ownership(rw_paths(spec))
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
