"""Local app registry — per-app metadata the core needs, app-agnostic glue.

Installing/recovering an app and running it as a service live on the remote side
(``cloudgpu/remote/apps/*``). This module is the *local* source of truth for the small
amount the CLI needs to stay generic: which ports an app exposes (for ``cloudgpu forward``),
whether it's a managed service (restart after provisioning), and which files ``cloudgpu init``
should vendor into a profile folder.

Core code (cli.py, profiles.py) iterates a profile's ``apps`` through here and never
hardcodes a specific app. Keep ``AVAILABLE_APPS`` aligned with the remote ``APP_INSTALLERS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from ..remote.apps.comfyui import COMFYUI_PORT


@dataclass(frozen=True)
class AppSpec:
    """What the local CLI needs to know about an installable app."""

    name: str
    ports: tuple[int, ...] = ()          # ports `cloudgpu forward` opens for this app
    is_service: bool = False             # runs as a systemd service (restart after provision)
    template_pkg: str | None = None      # importlib package holding files to vendor
    templates: tuple[str, ...] = ()      # files `cloudgpu init` drops into the profile folder

    def scaffold(self, profile_dir: Path) -> list[str]:
        """Vendor this app's template files into ``profile_dir``. Returns names written."""
        if not self.template_pkg:
            return []
        written = []
        for fname in self.templates:
            dest = profile_dir / fname
            # Never clobber a user's edited provision.py.
            if fname == "provision.py" and dest.exists():
                continue
            dest.write_text(resources.files(self.template_pkg).joinpath(fname).read_text())
            written.append(fname)
        return written


APPS: dict[str, AppSpec] = {
    "comfyui": AppSpec(
        name="comfyui",
        ports=(COMFYUI_PORT,),
        is_service=True,
        template_pkg="cloudgpu.templates.comfyui",
        templates=("comfylib.py", "provision.py"),
    ),
}

# Installable apps the CLI offers (aligned with remote APP_INSTALLERS).
AVAILABLE_APPS: list[str] = list(APPS)


def app_ports(apps: list[str]) -> list[int]:
    """All forward ports for the given apps, deduped, in order."""
    ports: list[int] = []
    for name in apps:
        spec = APPS.get(name)
        if not spec:
            continue
        for p in spec.ports:
            if p not in ports:
                ports.append(p)
    return ports


def service_apps(apps: list[str]) -> list[str]:
    """The given apps that run as a managed service."""
    return [a for a in apps if a in APPS and APPS[a].is_service]


def scaffold_apps(profile_dir: Path, apps: list[str]) -> list[str]:
    """Vendor each app's template files into the profile folder. Returns names written."""
    written: list[str] = []
    for name in apps:
        spec = APPS.get(name)
        if spec:
            written += spec.scaffold(profile_dir)
    return written
