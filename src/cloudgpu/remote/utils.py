"""Shared utilities for remote scripts. Stdlib-only."""

from __future__ import annotations

import os
import subprocess
import sys


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
    """Run pip install in a venv."""
    pip = os.path.join(venv_dir, "bin", "pip")
    run([pip, "install", *args], timeout=timeout)


def venv_python(venv_dir: str) -> str:
    """Get the python executable path for a venv."""
    return os.path.join(venv_dir, "bin", "python3")


def check_torch_cuda(venv_dir: str) -> bool:
    """Check if PyTorch with CUDA works in a venv."""
    python = venv_python(venv_dir)
    try:
        result = run(
            [python, "-c", "import torch; assert torch.cuda.is_available()"],
            capture=True,
            check=False,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


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


def pip_install_torch(venv_dir: str, timeout: int = 600) -> None:
    """Install PyTorch with CUDA support into a venv."""
    cuda_tag = detect_cuda_version()
    if cuda_tag:
        log(f"Detected CUDA: {cuda_tag}, installing PyTorch from CUDA index...")
        index_url = f"https://download.pytorch.org/whl/{cuda_tag}"
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
