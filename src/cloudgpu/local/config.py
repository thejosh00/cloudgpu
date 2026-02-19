"""Local config management at ~/.config/cloudgpu/config.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".config" / "cloudgpu"
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_config() -> dict[str, Any]:
    """Load config from disk, returning empty dict if missing."""
    if not CONFIG_FILE.exists():
        return {}
    return json.loads(CONFIG_FILE.read_text())


def save_config(config: dict[str, Any]) -> None:
    """Save config to disk."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2) + "\n")


def get_host(host: str | None = None) -> str:
    """Resolve host from argument or saved config.

    Args:
        host: Explicit host, or None to use saved default.

    Returns:
        The resolved host string.

    Raises:
        click.UsageError: If no host available.
    """
    if host:
        return host
    config = load_config()
    saved = config.get("host")
    if saved:
        return saved
    import click
    raise click.UsageError(
        "No host specified. Run 'cloudgpu setup <host>' first, or pass a host."
    )


def save_host(host: str, persistent_dir: str) -> None:
    """Save host and persistent dir to config."""
    config = load_config()
    config["host"] = host
    config["persistent_dir"] = persistent_dir
    save_config(config)


def get_persistent_dir(host: str | None = None) -> str:
    """Get the persistent directory path from config."""
    config = load_config()
    saved = config.get("persistent_dir")
    if saved:
        return saved
    import click
    raise click.UsageError(
        "Persistent directory not configured. Run 'cloudgpu setup <host>' first."
    )
