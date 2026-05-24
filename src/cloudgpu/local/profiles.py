"""Directory-as-profile store.

A *profile* is a folder anywhere on disk that holds its definition, its provisioning, and
its state. You ``cd`` into it and run ``cloudgpu up``.

- ``cloudgpu.toml``        — desired state, hand-edited (``cloudgpu init`` scaffolds it).
- ``provision.py`` / etc.  — optional provisioning payload (the whole folder is rsynced).
- ``cloudgpu.state.json``  — tool-written runtime state (which instance is up); gitignored
                             and never rsynced to the instance.

Secrets stay in the shared ``~/.config/cloudgpu/secrets.env`` (see ``secrets_file``).
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

try:  # py3.11+
    import tomllib
except ModuleNotFoundError:  # py3.10 backport (declared in pyproject)
    import tomli as tomllib  # type: ignore[no-redef]

from . import config

PROFILE_FILE = "cloudgpu.toml"
STATE_FILE = "cloudgpu.state.json"

# Files in the profile folder that must NOT be rsynced to the instance with the
# provisioning payload (tool state, secrets, VCS, build artifacts).
PROVISION_EXCLUDES = [STATE_FILE, "secrets.env", ".git", "__pycache__", "*.pyc"]

# Defaults applied to every loaded profile. (ssh_key and gpu are required; filesystem
# defaults to the folder name.)
_DEFAULTS: dict[str, Any] = {
    "apps": [],
    "region": "",
    "poll_seconds": 20,
    "max_hours": 12,
    "provision_timeout": 3600,  # seconds; cap on the provision script (big downloads)
}


def secrets_file() -> Path:
    """Path to the shared secrets file (KEY=VALUE lines, e.g. CIVITAI_TOKEN).

    If present it's transferred to the instance and sourced into the provision script's
    environment, so scripts can reference $CIVITAI_TOKEN etc. without hardcoding values.
    Keep it mode 600; it is never committed and must not be printed.
    """
    return config.CONFIG_DIR / "secrets.env"


# --- locating + loading a profile -----------------------------------------


def find_profile_dir(explicit: str | Path | None = None) -> Path:
    """Resolve the profile directory: ``explicit`` (``--profile``) or the current dir.

    CWD-only — no walking up to parents. Raises ConfigError if the resolved directory has
    no ``cloudgpu.toml``.
    """
    d = (Path(explicit).expanduser() if explicit else Path.cwd()).resolve()
    if not (d / PROFILE_FILE).exists():
        where = f"'{d}'" if explicit else "the current directory"
        raise config.ConfigError(
            f"No {PROFILE_FILE} in {where}. Run 'cloudgpu init' here, or pass --profile <dir>."
        )
    return d


def has_profile(d: str | Path) -> bool:
    """True if directory ``d`` contains a cloudgpu.toml."""
    return (Path(d).expanduser() / PROFILE_FILE).exists()


def load_profile(profile_dir: str | Path) -> dict[str, Any]:
    """Load and validate the profile in ``profile_dir``.

    Injects ``name`` (the folder name) and ``dir`` (the Path). Raises ConfigError if the
    profile is missing, unparseable, or invalid.
    """
    profile_dir = Path(profile_dir).expanduser().resolve()
    path = profile_dir / PROFILE_FILE
    name = profile_dir.name
    if not path.exists():
        raise config.ConfigError(f"No {PROFILE_FILE} in '{profile_dir}'.")
    try:
        raw = tomllib.loads(path.read_text())
    except (tomllib.TOMLDecodeError, OSError) as e:
        raise config.ConfigError(f"Could not read {path}: {e}") from e

    profile = {**_DEFAULTS, **raw, "name": name, "dir": profile_dir}

    # Filesystem is optional; it defaults to the folder name and is auto-created on the
    # first `up` if it doesn't exist yet.
    filesystem = profile.get("filesystem") or name
    if not isinstance(filesystem, str):
        raise config.ConfigError(f"{path}: 'filesystem' must be a string.")
    profile["filesystem"] = filesystem

    gpu = profile.get("gpu")
    if isinstance(gpu, str):  # be forgiving: "gh200" -> ["gh200"]
        gpu = [gpu]
    if not gpu or not isinstance(gpu, list):
        raise config.ConfigError(
            f"{path} must set 'gpu' to a non-empty list, e.g. [\"gh200\", \"a100\"]."
        )
    profile["gpu"] = gpu

    ssh_key = profile.get("ssh_key")
    if not ssh_key or not isinstance(ssh_key, str):
        raise config.ConfigError(
            f"{path} must set 'ssh_key' to the Lambda SSH key name to launch with "
            "(a private key matching it must be in ~/.ssh)."
        )

    if isinstance(profile.get("apps"), str):
        profile["apps"] = [profile["apps"]]
    profile.setdefault("instance_name", name)
    return profile


# --- runtime state (tool-written, in the profile folder) ------------------


def state_path(profile_dir: str | Path) -> Path:
    return Path(profile_dir).expanduser().resolve() / STATE_FILE


def load_state(profile_dir: str | Path) -> dict[str, Any]:
    """Return the profile's runtime state, or {} if no instance is tracked."""
    path = state_path(profile_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return {}


def save_state(profile_dir: str | Path, data: dict[str, Any]) -> None:
    state_path(profile_dir).write_text(json.dumps(data, indent=2) + "\n")


def clear_state(profile_dir: str | Path) -> None:
    state_path(profile_dir).unlink(missing_ok=True)


# --- scaffolding (`cloudgpu init`) ----------------------------------------


def _toml_template(name: str, *, filesystem: str | None, gpu: list[str],
                   apps: list[str], ssh_key: str) -> str:
    def arr(items: list[str]) -> str:
        return "[" + ", ".join(f'"{i}"' for i in items) + "]"

    fs_line = f'filesystem = "{filesystem}"' if filesystem else f'# filesystem = "{name}"'
    return "\n".join([
        f"# cloudgpu profile: {name}",
        f"{fs_line}     # persistent storage; defaults to the folder name, auto-created on first 'up'",
        f"gpu = {arr(gpu)}                # GPU preference order (alias or full type name)",
        f"apps = {arr(apps)}              # apps to keep installed",
        f'ssh_key = "{ssh_key}"                      # required: Lambda SSH key name (matching key in ~/.ssh)',
        '# region = "us-east-3"           # optional; defaults to the filesystem\'s region',
        "poll_seconds = 20                # capacity poll interval",
        "max_hours = 12                   # give up after this long",
        "provision_timeout = 3600         # seconds; cap on the provision script",
    ]) + "\n"


def _read_template(filename: str) -> str:
    return resources.files("cloudgpu.templates").joinpath(filename).read_text()


def scaffold(
    profile_dir: str | Path,
    *,
    ssh_key: str,
    gpu: list[str],
    apps: list[str],
    filesystem: str | None = None,
    force: bool = False,
) -> Path:
    """Create a profile folder: cloudgpu.toml + vendored comfylib.py + starter provision.py.

    Raises ConfigError if a cloudgpu.toml already exists (unless ``force``).
    """
    profile_dir = Path(profile_dir).expanduser().resolve()
    toml_path = profile_dir / PROFILE_FILE
    if toml_path.exists() and not force:
        raise config.ConfigError(
            f"{toml_path} already exists. Edit it, or pass --force to overwrite."
        )
    profile_dir.mkdir(parents=True, exist_ok=True)
    toml_path.write_text(_toml_template(
        profile_dir.name, filesystem=filesystem, gpu=gpu, apps=apps, ssh_key=ssh_key
    ))
    # Vendor the helper lib + a starter provision script (don't clobber an edited one).
    (profile_dir / "comfylib.py").write_text(_read_template("comfylib.py"))
    provision = profile_dir / "provision.py"
    if force or not provision.exists():
        provision.write_text(_read_template("provision.py"))
    gitignore = profile_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(f"{STATE_FILE}\n")
    return toml_path
