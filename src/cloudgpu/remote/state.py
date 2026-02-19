"""State management via state.json on persistent storage. Stdlib-only."""

from __future__ import annotations

import json
import os
import time
from typing import Any


class State:
    """Manages state.json in the persistent cloudgpu directory."""

    def __init__(self, persistent_dir: str):
        self.persistent_dir = persistent_dir
        self.cloudgpu_dir = os.path.join(persistent_dir, "cloudgpu")
        self.state_file = os.path.join(self.cloudgpu_dir, "state.json")

    def _ensure_dirs(self) -> None:
        os.makedirs(self.cloudgpu_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cloudgpu_dir, "bin"), exist_ok=True)
        os.makedirs(os.path.join(self.cloudgpu_dir, "remote"), exist_ok=True)

    def load(self) -> dict[str, Any]:
        """Load state from disk."""
        if not os.path.exists(self.state_file):
            return {"apps": {}, "created_at": time.time()}
        with open(self.state_file) as f:
            return json.load(f)

    def save(self, state: dict[str, Any]) -> None:
        """Save state to disk."""
        self._ensure_dirs()
        state["updated_at"] = time.time()
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)
            f.write("\n")

    def get_app(self, app_name: str) -> dict[str, Any] | None:
        """Get state for a specific app."""
        state = self.load()
        return state.get("apps", {}).get(app_name)

    def set_app(self, app_name: str, app_state: dict[str, Any]) -> None:
        """Set state for a specific app."""
        state = self.load()
        if "apps" not in state:
            state["apps"] = {}
        state["apps"][app_name] = app_state
        self.save(state)

    def remove_app(self, app_name: str) -> None:
        """Remove an app from state."""
        state = self.load()
        state.get("apps", {}).pop(app_name, None)
        self.save(state)

    def list_apps(self) -> dict[str, Any]:
        """List all installed apps."""
        state = self.load()
        return state.get("apps", {})

    @property
    def apps_dir(self) -> str:
        return os.path.join(self.persistent_dir, "apps")

    @property
    def venvs_dir(self) -> str:
        return os.path.join(self.persistent_dir, "venvs")

    @property
    def bin_dir(self) -> str:
        return os.path.join(self.cloudgpu_dir, "bin")
