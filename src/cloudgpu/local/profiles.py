"""Directory-as-profile store — the provisioning half of a run.

A *profile* is a folder under a top-level ``profiles/`` directory holding what to
install and provision onto an instance. Hardware (which machine to launch) lives
separately in a *config* (see ``configs.py``); a profile is provisioned onto the
instance a config owns, via ``cloudgpu up --config <c> --profile <p>``.

- ``cloudgpu.toml``        — apps + provisioning settings (``cloudgpu init`` scaffolds it).
- ``provision.py`` / etc.  — optional provisioning payload (the whole folder is rsynced).

Runtime state (which instance is up) is tracked per-config in ``configs.py``, not here.
Secrets stay in the shared ``~/.config/cloudgpu/secrets.env`` (see ``secrets_file``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:  # py3.11+
    import tomllib
except ModuleNotFoundError:  # py3.10 backport (declared in pyproject)
    import tomli as tomllib  # type: ignore[no-redef]

from . import config

PROFILE_FILE = "cloudgpu.toml"
PROFILES_DIR = "profiles"

# Files in the profile folder that must NOT be rsynced to the instance with the
# provisioning payload (secrets, VCS, build artifacts, and any per-config state files).
PROVISION_EXCLUDES = ["*.state.json", "secrets.env", ".git", "__pycache__", "*.pyc"]

# Defaults applied to every loaded profile (provisioning-only; hardware is in a config).
_DEFAULTS: dict[str, Any] = {
    "apps": [],
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


def profile_dir(name_or_path: str | Path) -> Path:
    """Resolve a profile name or path to its directory.

    A bare name resolves against ``./profiles/<name>`` (run from the repo root);
    a value containing a path separator is treated as a path.
    """
    p = Path(name_or_path).expanduser()
    if len(p.parts) > 1:
        return p.resolve()
    return (Path.cwd() / PROFILES_DIR / p.name).resolve()


def find_profile(name_or_path: str | Path) -> Path:
    """Resolve and require a profile directory. Raises ConfigError if it has no toml."""
    d = profile_dir(name_or_path)
    if not (d / PROFILE_FILE).exists():
        raise config.ConfigError(
            f"No {PROFILE_FILE} in '{d}'. Create profiles/{d.name}/, or run "
            "'cloudgpu init <name>'."
        )
    return d


def has_profile(d: str | Path) -> bool:
    """True if directory ``d`` contains a cloudgpu.toml."""
    return (Path(d).expanduser() / PROFILE_FILE).exists()


def load_profile(name_or_path: str | Path) -> dict[str, Any]:
    """Load and validate a provisioning profile.

    Accepts a profile name (resolved under ./profiles) or a directory path. Injects
    ``name`` (folder name) and ``dir`` (the Path). Raises ConfigError if missing,
    unparseable, or invalid.
    """
    d = find_profile(name_or_path)
    path = d / PROFILE_FILE
    name = d.name
    try:
        raw = tomllib.loads(path.read_text())
    except (tomllib.TOMLDecodeError, OSError) as e:
        raise config.ConfigError(f"Could not read {path}: {e}") from e

    profile = {**_DEFAULTS, **raw, "name": name, "dir": d}

    if isinstance(profile.get("apps"), str):
        profile["apps"] = [profile["apps"]]
    if not isinstance(profile["apps"], list):
        raise config.ConfigError(f"{path}: 'apps' must be a list of app names.")
    return profile


# --- scaffolding (`cloudgpu init`) ----------------------------------------


def _toml_template(name: str, *, apps: list[str]) -> str:
    def arr(items: list[str]) -> str:
        return "[" + ", ".join(f'"{i}"' for i in items) + "]"

    return "\n".join([
        f"# cloudgpu profile: {name}  (provisioning)",
        f"apps = {arr(apps)}              # apps to install on the instance",
        "provision_timeout = 3600         # seconds; cap on the provision script",
    ]) + "\n"


def scaffold(
    profile_dir_path: str | Path,
    *,
    apps: list[str],
    force: bool = False,
) -> Path:
    """Create a profile folder: cloudgpu.toml + .gitignore (app-agnostic).

    Any app-specific files (e.g. comfyui's comfylib.py / provision.py) are vendored
    separately by the app registry (``local.apps.scaffold_apps``). Raises ConfigError if a
    cloudgpu.toml already exists (unless ``force``).
    """
    d = Path(profile_dir_path).expanduser().resolve()
    toml_path = d / PROFILE_FILE
    if toml_path.exists() and not force:
        raise config.ConfigError(
            f"{toml_path} already exists. Edit it, or pass --force to overwrite."
        )
    d.mkdir(parents=True, exist_ok=True)
    toml_path.write_text(_toml_template(d.name, apps=apps))
    gitignore = d / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("__pycache__/\n")
    return toml_path
