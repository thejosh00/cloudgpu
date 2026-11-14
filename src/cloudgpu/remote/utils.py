"""Shared utilities for remote scripts. Stdlib-only."""

from __future__ import annotations

import os
import pwd
import subprocess
import sys

# APP_USER: the unprivileged user that runs app services and app-venv code.
from .run_as_app import APP_USER, CouldNotRun, app_step, run_stoppable

# Printed by every app-venv check before it does anything else, so a check that never got as
# far as running (sudo refused, venv unreadable) can't be mistaken for one that failed.
_CHECK_STARTED = "cloudgpu-venv-check-started"


class VenvCheckUnavailable(RuntimeError):
    """An app-venv check couldn't run to a conclusion, so it says nothing about the venv."""


def run(
    cmd: list[str],
    *,
    cwd: str | None = None,
    env: dict | None = None,
    timeout: int = 600,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    """Run a subprocess with sensible defaults.

    Args:
        cmd: Command and arguments.
        cwd: Working directory.
        env: Environment variables (merged with os.environ).
        timeout: Timeout in seconds.
        check: Raise CalledProcessError on non-zero exit.
        capture: Capture stdout/stderr instead of streaming.
    """
    full_env = None
    if env:
        full_env = {**os.environ, **env}

    if capture:
        return subprocess.run(
            cmd,
            cwd=cwd,
            env=full_env,
            timeout=timeout,
            check=check,
            capture_output=True,
            text=True,
        )
    else:
        return subprocess.run(
            cmd,
            cwd=cwd,
            env=full_env,
            timeout=timeout,
            check=check,
        )


def pip_install(venv_dir: str, *args: str, timeout: int = 600) -> None:
    """Run pip install in a venv as APP_USER, never as the caller.

    The venv's pip is a script run by the venv's Python, which loads every .pth file in
    site-packages before pip starts. APP_USER can write those, so whatever the packages named
    here, running it as the caller would run APP_USER's files as a user with passwordless sudo.
    Raises CouldNotRun if it can't run as APP_USER; there's no fallback to the caller. The venv
    must already be shared with APP_USER (service.prepare_venv_checks). As the ``-r`` installs
    (service.install_requirements_as_app): no caller environment, no tokens, PATH given.
    """
    pip = os.path.join(venv_dir, "bin", "pip")
    app_step([pip, "install", *args],
             env={"PATH": os.environ.get("PATH", os.defpath)}, timeout=timeout)


def requirements_present(path: str) -> bool:
    """Whether there's a requirements file to install. False only if nothing is there.

    cloudgpu-app controls this path. A broken symlink counts as present, so pip fails on it rather
    than the install being skipped quietly, and a directory the caller can't enter raises. It's
    lstat, so the caller doesn't read or follow anything.
    """
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def venv_python(venv_dir: str) -> str:
    """Get the python executable path for a venv."""
    return os.path.join(venv_dir, "bin", "python3")


def run_venv_check(venv_dir: str, code: str, timeout: int = 600) -> bool:
    """Run ``code`` with a venv's Python as APP_USER; True if it succeeded.

    App code can write the venv, so its Python must never run as ``ubuntu``, which has
    passwordless sudo. sudo resets the environment, so secrets don't reach it either.

    False is a verdict: the check ran and failed, or the venv has no Python. Raises
    VenvCheckUnavailable when the check couldn't run to a conclusion, because callers
    rebuild a venv on False and that must not happen to a good one.
    """
    python = venv_python(venv_dir)
    if not os.path.exists(python):
        return False
    try:
        pwd.getpwnam(APP_USER)
    except KeyError:
        raise VenvCheckUnavailable(f"user '{APP_USER}' doesn't exist yet") from None
    try:
        # From / (ubuntu's cwd may be unreadable to APP_USER, or hold planted modules), and on a
        # timeout nothing the check started is left running as APP_USER.
        result = run_stoppable(
            ["sudo", "-n", "-H", "-u", APP_USER, "--", python, "-c",
             f"print({_CHECK_STARTED!r}, flush=True)\n{code}"],
            capture=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        raise VenvCheckUnavailable(f"couldn't run as '{APP_USER}': {e}") from e
    if _CHECK_STARTED not in result.stdout:
        raise VenvCheckUnavailable(
            f"couldn't run as '{APP_USER}': {result.stderr.strip() or 'no output'}")
    return result.returncode == 0


def trust_repo_as_app(repo: str) -> None:
    """Let APP_USER's git use a repo another user cloned, which it otherwise calls dubious.

    Older gits ignore `-c safe.directory`, so this has to be APP_USER's global config. That lives
    in its instance-local HOME, so every new instance needs this again. It names this one repo,
    and it doesn't reach ubuntu's own git.
    """
    app_step(["git", "config", "--global", "--add", "safe.directory", repo], timeout=60)


def git_pull_as_app(repo: str) -> None:
    """Pull as APP_USER, because a hook or config in .git is theirs to write."""
    trust_repo_as_app(repo)
    app_step(["git", "-C", repo, "pull"], timeout=600)


def git_version_as_app(repo: str) -> str | None:
    """The repo's short commit hash, or None if it couldn't be read. Read-only.

    `status` comes through here, so a user that doesn't exist yet means no sudo at all.
    """
    try:
        pwd.getpwnam(APP_USER)
    except KeyError:
        return None
    try:
        result = app_step(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                          capture=True, check=False, timeout=60)
    except (CouldNotRun, subprocess.TimeoutExpired) as e:
        log(f"Couldn't read the version of {repo}: {e}", "warn")
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def check_torch_cuda(venv_dir: str) -> bool:
    """Check if PyTorch with CUDA works in a venv (see run_venv_check)."""
    return run_venv_check(venv_dir, "import torch; assert torch.cuda.is_available()")


def torch_cuda_status(venv_dir: str) -> bool | None:
    """check_torch_cuda for read-only callers: None means it couldn't be checked."""
    try:
        return check_torch_cuda(venv_dir)
    except VenvCheckUnavailable as e:
        log(f"PyTorch CUDA unverified: {e}", "warn")
        return None


def detect_cuda_version() -> str | None:
    """Detect the system CUDA version via nvidia-smi.

    Returns a pip index tag like 'cu124' or None if detection fails.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode != 0:
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    # Use nvcc to get the actual CUDA toolkit version
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode == 0:
            import re
            m = re.search(r"release (\d+)\.(\d+)", result.stdout)
            if m:
                major, minor = m.group(1), m.group(2)
                return f"cu{major}{minor}"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: check /usr/local/cuda/version.txt or similar
    for path in ["/usr/local/cuda/version.txt", "/usr/local/cuda/version.json"]:
        if os.path.exists(path):
            try:
                import re
                with open(path) as f:
                    text = f.read()
                m = re.search(r"(\d+)\.(\d+)", text)
                if m:
                    major, minor = m.group(1), m.group(2)
                    return f"cu{major}{minor}"
            except OSError:
                pass

    # If nvidia-smi works but we can't detect version, default to cu124
    return "cu124"


def torch_index_url() -> str | None:
    """PyTorch wheel index for the system CUDA, or None if undetectable."""
    cuda_tag = detect_cuda_version()
    return f"https://download.pytorch.org/whl/{cuda_tag}" if cuda_tag else None


def pip_install_torch(venv_dir: str, timeout: int = 600) -> None:
    """Install PyTorch with CUDA support into a venv."""
    index_url = torch_index_url()
    if index_url:
        log(f"Detected CUDA: {index_url.rsplit('/', 1)[1]}, "
            f"installing PyTorch from CUDA index...")
        pip_install(
            venv_dir,
            "--force-reinstall",
            "torch", "torchvision", "torchaudio",
            "--index-url", index_url,
            timeout=timeout,
        )
    else:
        log("Could not detect CUDA version, installing PyTorch from default index...", "warn")
        pip_install(venv_dir, "torch", "torchvision", "torchaudio", timeout=timeout)


def log(msg: str, level: str = "info") -> None:
    """Print a log message to stderr."""
    prefix = {"info": ">>>", "warn": "!!!", "error": "ERR"}
    print(f"{prefix.get(level, '>>>')} {msg}", file=sys.stderr, flush=True)
