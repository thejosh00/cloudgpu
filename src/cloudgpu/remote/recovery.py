"""Cross-cutting recovery utilities. Stdlib-only."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .state import State
from .utils import log

if TYPE_CHECKING:
    from .installer import AppInstaller


def setup_path(state: State) -> None:
    """Add cloudgpu/bin to PATH in .bashrc (transient, needs redoing each instance)."""
    bin_dir = state.bin_dir
    bashrc = os.path.expanduser("~/.bashrc")

    marker = "# cloudgpu-path"
    line = f'export PATH="{bin_dir}:$PATH"  {marker}'

    # Check if already present
    if os.path.exists(bashrc):
        with open(bashrc) as f:
            content = f.read()
        if marker in content:
            # Replace existing line
            lines = content.splitlines()
            lines = [l for l in lines if marker not in l]
            lines.append(line)
            with open(bashrc, "w") as f:
                f.write("\n".join(lines) + "\n")
            log("Updated PATH in .bashrc")
            return

    # Append
    with open(bashrc, "a") as f:
        f.write(f"\n{line}\n")
    log("Added cloudgpu/bin to PATH in .bashrc")


def recover_all(state: State, app_installers: dict[str, type[AppInstaller]]) -> None:
    """Recover all installed apps on a new instance."""
    installed = state.list_apps()

    if not installed:
        log("No apps found in state.json - nothing to recover")
        return

    log(f"Found {len(installed)} app(s) to recover: {', '.join(installed.keys())}")

    for app_name, app_state in installed.items():
        if app_name not in app_installers:
            log(f"Skipping {app_name}: no installer available", "warn")
            continue

        log(f"Recovering {app_name}...")
        installer = app_installers[app_name](state)
        try:
            installer.recover()
            log(f"{app_name} recovered successfully")
        except Exception as e:
            log(f"Failed to recover {app_name}: {e}", "error")

    # Setup PATH
    setup_path(state)

    log("Recovery complete!")
