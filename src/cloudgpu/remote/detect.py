"""Detect persistent directory, GPU, and CUDA on Lambda instances. Stdlib-only."""

from __future__ import annotations

import os
import subprocess
import json


def find_persistent_dir() -> str | None:
    """Find the Lambda persistent NFS directory.

    Looks for directories matching /lambda/nfs/*/ pattern.
    Returns the first valid one found, or None.
    """
    nfs_base = "/lambda/nfs"
    if not os.path.isdir(nfs_base):
        return None
    entries = os.listdir(nfs_base)
    for entry in sorted(entries):
        path = os.path.join(nfs_base, entry)
        if os.path.isdir(path) and not entry.startswith("."):
            return path
    return None


def detect_gpu() -> dict:
    """Detect GPU information using nvidia-smi.

    Returns dict with keys: available, gpus (list of {name, memory_mb}), driver_version.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {"available": False, "gpus": [], "driver_version": None}

        gpus = []
        driver_version = None
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({"name": parts[0], "memory_mb": int(float(parts[1]))})
                driver_version = parts[2]

        return {
            "available": len(gpus) > 0,
            "gpus": gpus,
            "driver_version": driver_version,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {"available": False, "gpus": [], "driver_version": None}


def detect_cuda() -> dict:
    """Detect CUDA version from nvcc.

    Returns dict with keys: available, version.
    """
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {"available": False, "version": None}

        for line in result.stdout.splitlines():
            if "release" in line.lower():
                # Parse "Cuda compilation tools, release 12.1, V12.1.105"
                parts = line.split("release")
                if len(parts) >= 2:
                    version = parts[1].strip().split(",")[0].strip()
                    return {"available": True, "version": version}

        return {"available": False, "version": None}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {"available": False, "version": None}


def detect_python_torch() -> dict:
    """Check if PyTorch with CUDA is available in system Python.

    Returns dict with keys: available, torch_version, cuda_available, cuda_version.
    """
    try:
        result = subprocess.run(
            [
                "python3", "-c",
                "import torch; import json; print(json.dumps({"
                "'torch_version': torch.__version__,"
                "'cuda_available': torch.cuda.is_available(),"
                "'cuda_version': torch.version.cuda if torch.cuda.is_available() else None"
                "}))",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            info = json.loads(result.stdout.strip())
            info["available"] = True
            return info
        return {"available": False, "torch_version": None, "cuda_available": False, "cuda_version": None}
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return {"available": False, "torch_version": None, "cuda_available": False, "cuda_version": None}


def detect_all() -> dict:
    """Run all detection and return combined results."""
    persistent_dir = find_persistent_dir()
    return {
        "persistent_dir": persistent_dir,
        "gpu": detect_gpu(),
        "cuda": detect_cuda(),
        "python_torch": detect_python_torch(),
    }


if __name__ == "__main__":
    print(json.dumps(detect_all(), indent=2))
