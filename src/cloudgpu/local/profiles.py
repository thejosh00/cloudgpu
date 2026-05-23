"""Profile store: declarative desired state (TOML) + tool-owned runtime state (JSON).

A *profile* is the user's desired state for one logical GPU machine — the GPU
preference order, the filesystem that gives it a persistent identity, and the apps
to keep installed. Profiles are per-user and live under ~/.config/cloudgpu/, not in
a repo.

Two stores, deliberately separate so the tool never clobbers the user's edits:

- ``profiles/<name>.toml``  — desired state, hand-edited (we only scaffold it).
- ``runtime/<name>.json``   — runtime state, tool-written (which instance is up now).

The active profile (so commands don't need ``--profile`` every time) is recorded as
``active_profile`` in the existing config.json.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

try:  # py3.11+
    import tomllib
except ModuleNotFoundError:  # py3.10 backport (declared in pyproject)
    import tomli as tomllib  # type: ignore[no-redef]

from . import config

# Defaults applied to every loaded profile. (ssh_key and filesystem/gpu are required.)
_DEFAULTS: dict[str, Any] = {
    "apps": [],
    "region": "",
    "poll_seconds": 20,
    "max_hours": 12,
    "provision_timeout": 3600,  # seconds; cap on the provision script (big downloads)
}


def profiles_dir() -> Path:
    return config.CONFIG_DIR / "profiles"


def runtime_dir() -> Path:
    return config.CONFIG_DIR / "runtime"


def profile_path(name: str) -> Path:
    return profiles_dir() / f"{name}.toml"


def runtime_path(name: str) -> Path:
    return runtime_dir() / f"{name}.json"


def provision_dir(name: str) -> Path:
    """Directory of optional provisioning assets for a profile, run on every ``up``.

    If ``profiles/<name>.provision/`` exists, the whole directory is rsynced to the
    instance and its ``provision.sh`` entry point is executed from inside it (so the
    script can reference sibling files like ComfyUI workflows). It should be idempotent
    (e.g. download a model only if missing) and write persistent data under the
    filesystem so it survives termination.
    """
    return profiles_dir() / f"{name}.provision"


def list_profiles() -> list[str]:
    """Return profile names (sorted), or [] if none exist."""
    d = profiles_dir()
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.toml"))


def load_profile(name: str) -> dict[str, Any]:
    """Load and validate a profile, applying defaults and injecting ``name``.

    Raises:
        config.ConfigError: if the profile is missing, unparseable, or invalid.
    """
    path = profile_path(name)
    if not path.exists():
        raise config.ConfigError(
            f"No profile named '{name}'. Create one with "
            f"'cloudgpu profile create {name} --filesystem <fs> --gpu gh200,a100'."
        )
    try:
        raw = tomllib.loads(path.read_text())
    except (tomllib.TOMLDecodeError, OSError) as e:
        raise config.ConfigError(f"Could not read profile '{name}': {e}") from e

    profile = {**_DEFAULTS, **raw, "name": name}

    # Filesystem is optional; it defaults to the profile name and is auto-created on
    # the first `up` if it doesn't exist yet.
    filesystem = profile.get("filesystem") or name
    if not isinstance(filesystem, str):
        raise config.ConfigError(f"Profile '{name}': 'filesystem' must be a string.")
    profile["filesystem"] = filesystem

    gpu = profile.get("gpu")
    if isinstance(gpu, str):  # be forgiving: "gh200" -> ["gh200"]
        gpu = [gpu]
    if not gpu or not isinstance(gpu, list):
        raise config.ConfigError(
            f"Profile '{name}' must set 'gpu' to a non-empty list, e.g. [\"gh200\", \"a100\"]."
        )
    profile["gpu"] = gpu

    ssh_key = profile.get("ssh_key")
    if not ssh_key or not isinstance(ssh_key, str):
        raise config.ConfigError(
            f"Profile '{name}' must set 'ssh_key' to the Lambda SSH key name to launch "
            "with (a private key matching it must be in ~/.ssh)."
        )

    if isinstance(profile.get("apps"), str):
        profile["apps"] = [profile["apps"]]
    profile.setdefault("instance_name", name)
    return profile


def profile_template(
    name: str,
    *,
    filesystem: str | None = None,
    gpu: list[str],
    apps: list[str],
    ssh_key: str | None = None,
) -> str:
    """Render a profile TOML document (we never use a TOML writer dependency)."""
    def arr(items: list[str]) -> str:
        return "[" + ", ".join(f'"{i}"' for i in items) + "]"

    fs_line = f'filesystem = "{filesystem}"' if filesystem else f'# filesystem = "{name}"'
    lines = [
        f"# cloudgpu profile: {name}",
        f"{fs_line}     # persistent storage; defaults to the profile name, auto-created on first 'up'",
        f"gpu = {arr(gpu)}                # GPU preference order (alias or full type name)",
        f"apps = {arr(apps)}              # apps to keep installed",
    ]
    ssh_line = f'ssh_key = "{ssh_key}"' if ssh_key else 'ssh_key = ""'
    lines += [
        f"{ssh_line}                      # required: Lambda SSH key name (matching key in ~/.ssh)",
        f'instance_name = "{name}"        # optional; Lambda instance name',
        '# region = "us-east-3"           # optional; defaults to the filesystem\'s region',
        "poll_seconds = 20                # capacity poll interval",
        "max_hours = 12                   # give up after this long",
    ]
    return "\n".join(lines) + "\n"


def create_profile(
    name: str,
    *,
    filesystem: str | None = None,
    gpu: list[str],
    apps: list[str],
    ssh_key: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Scaffold profiles/<name>.toml. Raises ConfigError if it exists (unless overwrite)."""
    path = profile_path(name)
    if path.exists() and not overwrite:
        raise config.ConfigError(
            f"Profile '{name}' already exists at {path}. "
            "Edit it with 'cloudgpu profile edit', or pass --force to overwrite."
        )
    profiles_dir().mkdir(parents=True, exist_ok=True)
    path.write_text(profile_template(
        name, filesystem=filesystem, gpu=gpu, apps=apps, ssh_key=ssh_key
    ))
    return path


def delete_profile(name: str) -> None:
    """Remove a profile's TOML, provisioning dir, runtime state, and clear it as active."""
    profile_path(name).unlink(missing_ok=True)
    shutil.rmtree(provision_dir(name), ignore_errors=True)
    clear_runtime(name)
    if get_active() == name:
        cfg = config.load_config()
        cfg.pop("active_profile", None)
        config.save_config(cfg)


# --- active profile pointer (stored in config.json) -----------------------


def get_active() -> str | None:
    return config.load_config().get("active_profile")


def set_active(name: str) -> None:
    """Mark a profile active. Raises ConfigError if it doesn't exist."""
    if not profile_path(name).exists():
        raise config.ConfigError(f"No profile named '{name}'.")
    cfg = config.load_config()
    cfg["active_profile"] = name
    config.save_config(cfg)


def require_profile(name: str | None = None) -> str:
    """Resolve a profile name: explicit arg, else the active profile."""
    if name:
        return name
    active = get_active()
    if active:
        return active
    raise config.ConfigError(
        "No profile selected. Create one with 'cloudgpu profile create ...' "
        "and select it with 'cloudgpu profile use <name>', or pass --profile."
    )


# --- runtime state (tool-written JSON) ------------------------------------


def load_runtime(name: str) -> dict[str, Any]:
    """Return the profile's runtime state, or {} if the instance isn't tracked."""
    path = runtime_path(name)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return {}


def save_runtime(name: str, data: dict[str, Any]) -> None:
    runtime_dir().mkdir(parents=True, exist_ok=True)
    runtime_path(name).write_text(json.dumps({**data, "profile": name}, indent=2) + "\n")


def clear_runtime(name: str) -> None:
    runtime_path(name).unlink(missing_ok=True)
