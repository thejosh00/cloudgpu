"""Hardware configs — the machine half of a run, separate from provisioning.

A *config* is a single TOML file under a top-level ``configs/`` directory describing
the instance to launch: which GPU, which SSH key, which persistent filesystem, region,
and the billing/auto-terminate policy. It owns the instance and its runtime state; one
or more *profiles* (see ``profiles.py``) are provisioned onto that instance. The config
can name them (``profiles = [...]``), and every ``cloudgpu up`` then provisions those.

  configs/<name>.toml         — hardware, hand-edited (``cloudgpu init --config`` scaffolds).
  configs/<name>.state.json   — tool-written runtime state (which instance is up); gitignored.

Note the distinction from ``config.py`` (singular): that module is the global
``~/.config/cloudgpu`` app cache (single-host setup, secrets dir). This module
(``configs.py``, plural) is the per-machine hardware configs under the repo's
``configs/`` folder.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

try:  # py3.11+
    import tomllib
except ModuleNotFoundError:  # py3.10 backport (declared in pyproject)
    import tomli as tomllib  # type: ignore[no-redef]

from . import config

CONFIGS_DIR = "configs"
CONFIG_SUFFIX = ".toml"
STATE_SUFFIX = ".state.json"

# Defaults applied to every loaded config. (ssh_key and gpu are required; filesystem
# and instance_name default to the config name.)
_DEFAULTS: dict[str, Any] = {
    "region": "",
    "poll_seconds": 20,
    "max_hours": 12,
    "auto_terminate_hours": 0,  # > 0: instance self-terminates this long after each `up`
    "profiles": [],  # profile names or paths (as -P takes them) that every `up` provisions
}


# --- locating + loading a config ------------------------------------------


def config_path(name_or_path: str | Path) -> Path:
    """Resolve a config name or path to its TOML file.

    A bare name resolves against ``./configs/<name>.toml`` (run from the repo root);
    a value containing a path separator or ending in .toml is treated as a path.
    """
    p = Path(name_or_path).expanduser()
    if p.suffix == CONFIG_SUFFIX or len(p.parts) > 1:
        return p.resolve()
    return (Path.cwd() / CONFIGS_DIR / f"{p.name}{CONFIG_SUFFIX}").resolve()


def find_config(name_or_path: str | Path) -> Path:
    """Resolve and require a config file. Raises ConfigError if it doesn't exist."""
    path = config_path(name_or_path)
    if not path.exists():
        raise config.ConfigError(
            f"No config at {path}. Create configs/{path.stem}.toml, or run "
            "'cloudgpu init --config <name> ...'."
        )
    return path


def load_config(name_or_path: str | Path) -> dict[str, Any]:
    """Load and validate a hardware config.

    Injects ``name`` (file stem) and ``path`` (the Path). Raises ConfigError if the
    config is missing, unparseable, or invalid.
    """
    path = find_config(name_or_path)
    name = path.stem
    try:
        raw = tomllib.loads(path.read_text())
    except (tomllib.TOMLDecodeError, OSError) as e:
        raise config.ConfigError(f"Could not read {path}: {e}") from e

    cfg = {**_DEFAULTS, **raw, "name": name, "path": path}

    # Filesystem is optional; it defaults to the config name and is auto-created on the
    # first `up` if it doesn't exist yet.
    filesystem = cfg.get("filesystem") or name
    if not isinstance(filesystem, str):
        raise config.ConfigError(f"{path}: 'filesystem' must be a string.")
    cfg["filesystem"] = filesystem

    gpu = cfg.get("gpu")
    if isinstance(gpu, str):  # be forgiving: "gh200" -> ["gh200"]
        gpu = [gpu]
    if not gpu or not isinstance(gpu, list):
        raise config.ConfigError(
            f"{path} must set 'gpu' to a non-empty list, e.g. [\"gh200\", \"a100\"]."
        )
    cfg["gpu"] = gpu

    ssh_key = cfg.get("ssh_key")
    if not ssh_key or not isinstance(ssh_key, str):
        raise config.ConfigError(
            f"{path} must set 'ssh_key' to the Lambda SSH key name to launch with "
            "(a private key matching it must be in ~/.ssh)."
        )

    cfg.setdefault("instance_name", name)

    try:
        cfg["auto_terminate_hours"] = float(cfg.get("auto_terminate_hours") or 0)
    except (TypeError, ValueError):
        raise config.ConfigError(
            f"{path}: 'auto_terminate_hours' must be a number (hours; 0 disables)."
        ) from None

    # Only the shape: whether each profile exists is `up`'s to check, so `down`, `status` and
    # the rest keep working on a config that names a missing one.
    listed = cfg.get("profiles")
    if not isinstance(listed, list) or not all(isinstance(p, str) and p.strip() for p in listed):
        raise config.ConfigError(
            f"{path}: 'profiles' must be a list of profile names or paths, "
            'e.g. ["qwen-image", "krea2"].'
        )
    cfg["profiles"] = list(listed)
    return cfg


# --- runtime state (tool-written, beside the config file) -----------------


def state_path(config_path_or_name: str | Path) -> Path:
    """State file beside the config: configs/<name>.toml -> configs/<name>.state.json."""
    p = config_path(config_path_or_name)
    return p.with_name(f"{p.stem}{STATE_SUFFIX}")


UP_LOCK_SUFFIX = ".up.lock"


class UpLocked(Exception):
    """Another `up` on this computer holds this config's lock."""

    def __init__(self, path: Path):
        super().__init__(str(path))
        self.path = path


def up_lock_path(config_path_or_name: str | Path) -> Path:
    """Lock file beside the config's state: configs/<name>.toml -> configs/<name>.up.lock."""
    p = config_path(config_path_or_name)
    return p.with_name(f"{p.stem}{UP_LOCK_SUFFIX}")


def hold_up_lock(config_path_or_name: str | Path):
    """Take this config's `up` lock without waiting, and return the open file that holds it.

    flock, so a crash or kill releases it with the process. Covers this computer only: two
    machines running `up` on one config aren't stopped by it. The file is left in place.

    Raises:
        UpLocked: If another process holds it.
    """
    import fcntl

    path = up_lock_path(config_path_or_name)
    handle = open(path, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise UpLocked(path) from None
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


class StateFileError(config.ConfigError):
    """The config's state file exists but can't be used: unreadable, not JSON, or not a JSON object.

    The message names the file and says what to do. Nothing has been changed when it is raised.
    """

    def __init__(self, path: Path, config_name: str, why: str):
        super().__init__(
            f"The state file {path} can't be used ({why}), so cloudgpu can't tell which instance "
            f"config '{config_name}' tracks. Stopped without changing anything. Look at the file "
            "and repair it if you can (it is JSON). Or move it aside and run "
            f"'cloudgpu up -C {config_name}', which adopts the instance mounting the config's "
            "filesystem (or launches one if there is none); the list of provisioned profiles is "
            "then gone unless the config lists them (profiles = [...])."
        )
        self.path = path


def load_state(config_path_or_name: str | Path) -> dict[str, Any]:
    """Return the config's runtime state, or {} if there is no state file.

    Raises:
        StateFileError: If the file exists but can't be read, isn't JSON, or isn't a JSON object.
    """
    path = state_path(config_path_or_name)
    try:
        text = path.read_text()
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:  # ValueError: bytes that aren't text
        raise StateFileError(path, str(config_path_or_name), f"couldn't read it: {e}") from e
    try:
        data = json.loads(text)
    except ValueError as e:
        why = "it is empty" if not text.strip() else f"it isn't valid JSON: {e}"
        raise StateFileError(path, str(config_path_or_name), why) from e
    if not isinstance(data, dict):
        raise StateFileError(
            path, str(config_path_or_name), f"it holds a JSON {type(data).__name__}, not an object")
    return data


def save_state(config_path_or_name: str | Path, data: dict[str, Any]) -> None:
    """Write the state file atomically: a temporary file beside it, synced, then renamed over it.

    A failure leaves the previous file as it was and no temporary file. The temporary name ends
    in STATE_SUFFIX so that the same ignore rules cover one left by a kill.
    """
    path = state_path(config_path_or_name)
    text = json.dumps(data, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=f".tmp{STATE_SUFFIX}")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def clear_state(config_path_or_name: str | Path) -> None:
    state_path(config_path_or_name).unlink(missing_ok=True)


# --- scaffolding (`cloudgpu init --config`) -------------------------------


def _toml_template(name: str, *, filesystem: str | None, gpu: list[str], ssh_key: str) -> str:
    def arr(items: list[str]) -> str:
        return "[" + ", ".join(f'"{i}"' for i in items) + "]"

    fs_line = f'filesystem = "{filesystem}"' if filesystem else f'# filesystem = "{name}"'
    return "\n".join([
        f"# cloudgpu config: {name}  (hardware / instance)",
        f"{fs_line}     # persistent storage; defaults to the config name, auto-created on first 'up'",
        f"gpu = {arr(gpu)}                # GPU preference order (alias or full type name)",
        f'ssh_key = "{ssh_key}"                      # required: Lambda SSH key name (matching key in ~/.ssh)',
        '# region = "us-east-3"           # optional; defaults to the filesystem\'s region',
        "poll_seconds = 20                # capacity poll interval",
        "max_hours = 12                   # give up after this long",
        '# profiles = ["qwen-image"]      # profiles every \'up\' provisions (names or paths, as -P takes them)',
        "# auto_terminate_hours = 8       # billing cap: instance self-terminates this long after the last 'up'",
    ]) + "\n"


def scaffold(
    name: str,
    *,
    ssh_key: str,
    gpu: list[str],
    filesystem: str | None = None,
    configs_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Create configs/<name>.toml. Raises ConfigError if it exists (unless force)."""
    base = Path(configs_dir).expanduser() if configs_dir else (Path.cwd() / CONFIGS_DIR)
    base.mkdir(parents=True, exist_ok=True)
    toml_path = (base / f"{name}{CONFIG_SUFFIX}").resolve()
    if toml_path.exists() and not force:
        raise config.ConfigError(
            f"{toml_path} already exists. Edit it, or pass --force to overwrite."
        )
    toml_path.write_text(_toml_template(name, filesystem=filesystem, gpu=gpu, ssh_key=ssh_key))
    gitignore = base / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(f"*{STATE_SUFFIX}\n")
    return toml_path
