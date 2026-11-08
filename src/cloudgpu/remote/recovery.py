"""Cross-cutting recovery utilities. Stdlib-only."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from . import instance
from .installer import INSTALL_FAILED
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


def recover_all(state: State, app_installers: dict[str, type[AppInstaller]]) -> dict[str, str]:
    """Recover all installed apps on a new instance.

    Keeps going after an app fails, still sets up PATH, and returns the failures
    as {app name: error}. Empty means every recorded app came back. A recorded app with no
    installer is a failure; one whose last install failed is installed again, not recovered.
    """
    instance.require_instance()  # before anything, setup_path included
    installed = state.list_apps()

    if not installed:
        log("No apps found in state.json - nothing to recover")
        return {}

    log(f"Found {len(installed)} app(s) to recover: {', '.join(installed.keys())}")

    failures: dict[str, str] = {}
    for app_name, app_state in installed.items():
        if app_name not in app_installers:
            failures[app_name] = "no installer available"
            log(f"Can't recover {app_name}: no installer available", "error")
            continue

        installer = app_installers[app_name](state)
        try:
            if isinstance(app_state, dict) and app_state.get("status") == INSTALL_FAILED:
                log(f"{app_name}'s last install failed; installing it again...", "warn")
                installer.install()
            else:
                log(f"Recovering {app_name}...")
                installer.recover()
            log(f"{app_name} recovered successfully")
        except Exception as e:
            failures[app_name] = str(e) or type(e).__name__
            log(f"Failed to recover {app_name}: {failures[app_name]}", "error")

    # Setup PATH
    setup_path(state)

    if failures:
        log(f"Recovery failed for {len(failures)} app(s):", "error")
        for app_name, error in failures.items():
            log(f"  {app_name}: {error}", "error")
    else:
        log("Recovery complete!")
    return failures
