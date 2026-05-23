"""ComfyUI installer. Stdlib-only."""

from __future__ import annotations

import os
import stat
import time

from ..installer import AppInstaller
from ..utils import run, pip_install, pip_install_torch, venv_python, check_torch_cuda, log

COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
MANAGER_REPO = "https://github.com/ltdrdata/ComfyUI-Manager.git"


class ComfyUIInstaller(AppInstaller):

    @property
    def name(self) -> str:
        return "comfyui"

    @property
    def app_dir(self) -> str:
        return os.path.join(self.state.apps_dir, "comfyui")

    @property
    def venv_dir(self) -> str:
        return os.path.join(self.state.venvs_dir, "comfyui")

    @property
    def launch_script(self) -> str:
        return os.path.join(self.state.bin_dir, "comfyui")

    def install(self) -> None:
        """Install ComfyUI from scratch."""
        log("Installing ComfyUI...")

        # 1. Clone ComfyUI
        if not os.path.exists(self.app_dir):
            log("Cloning ComfyUI...")
            os.makedirs(self.state.apps_dir, exist_ok=True)
            run(["git", "clone", COMFYUI_REPO, self.app_dir])
        else:
            log("ComfyUI repo already exists, pulling latest...")
            run(["git", "-C", self.app_dir, "pull"])

        # 2. Create venv with
        if not os.path.exists(self.venv_dir):
            log("Creating venv (for PyTorch)...")
            os.makedirs(self.state.venvs_dir, exist_ok=True)
            run(["python3", "-m", "venv", self.venv_dir])
        else:
            log("Venv already exists")

        # 3. Verify PyTorch CUDA and install torchvision/torchaudio in venv
        log("Checking PyTorch CUDA availability...")
        if not check_torch_cuda(self.venv_dir):
            log("PyTorch CUDA not available via system packages, installing...", "warn")
            pip_install_torch(self.venv_dir, timeout=600)
            if not check_torch_cuda(self.venv_dir):
                log("PyTorch CUDA still not working after install!", "error")
                raise RuntimeError("Failed to get PyTorch with CUDA support")
        else:
            log("PyTorch with CUDA is available via system packages")

        # Always install torchvision/torchaudio into the venv so they match
        # the venv torch version.
        log("Installing matching torchvision and torchaudio in venv...")
        pip_install(self.venv_dir, "--no-deps",
                    "torchvision", "torchaudio", timeout=600)

        # 4. Install requirements
        log("Installing ComfyUI requirements...")
        requirements = os.path.join(self.app_dir, "requirements.txt")
        if os.path.exists(requirements):
            pip_install(self.venv_dir, "-r", requirements, timeout=600)

        # 5. Clone ComfyUI-Manager
        custom_nodes = os.path.join(self.app_dir, "custom_nodes")
        manager_dir = os.path.join(custom_nodes, "ComfyUI-Manager")
        if not os.path.exists(manager_dir):
            log("Cloning ComfyUI-Manager...")
            os.makedirs(custom_nodes, exist_ok=True)
            run(["git", "clone", MANAGER_REPO, manager_dir])
        else:
            log("ComfyUI-Manager already exists, pulling latest...")
            run(["git", "-C", manager_dir, "pull"])

        # Install manager requirements if they exist
        manager_requirements = os.path.join(manager_dir, "requirements.txt")
        if os.path.exists(manager_requirements):
            pip_install(self.venv_dir, "-r", manager_requirements, timeout=120)

        # 6. Create launch script
        self._create_launch_script()

        # 7. Update state
        self.state.set_app(self.name, {
            "status": "installed",
            "app_dir": self.app_dir,
            "venv_dir": self.venv_dir,
            "installed_at": time.time(),
            "version": self._get_version(),
        })

        log("ComfyUI installed successfully!")

    def verify(self) -> bool:
        """Verify ComfyUI installation."""
        checks = {
            "app_dir": os.path.isdir(self.app_dir),
            "venv": os.path.isdir(self.venv_dir),
            "main_py": os.path.isfile(os.path.join(self.app_dir, "main.py")),
            "torch_cuda": check_torch_cuda(self.venv_dir),
            "launch_script": os.path.isfile(self.launch_script),
        }
        for check, passed in checks.items():
            status = "OK" if passed else "FAIL"
            log(f"  {check}: {status}")
        return all(checks.values())

    def recover(self) -> None:
        """Recover ComfyUI on a new instance."""
        log("Recovering ComfyUI...")

        if not os.path.isdir(self.app_dir):
            log("App directory missing, doing full install", "warn")
            self.install()
            return

        if not os.path.isdir(self.venv_dir):
            log("Venv missing, recreating...", "warn")
            run(["python3", "-m", "venv", self.venv_dir])
            requirements = os.path.join(self.app_dir, "requirements.txt")
            if os.path.exists(requirements):
                pip_install(self.venv_dir, "-r", requirements, timeout=120)

        # Verify PyTorch CUDA (system packages may differ on new instance)
        if not check_torch_cuda(self.venv_dir):
            log("PyTorch CUDA broken, rebuilding venv...", "warn")
            import shutil
            shutil.rmtree(self.venv_dir, ignore_errors=True)
            run(["python3", "-m", "venv", self.venv_dir])
            requirements = os.path.join(self.app_dir, "requirements.txt")
            if os.path.exists(requirements):
                pip_install(self.venv_dir, "-r", requirements, timeout=600)

        # Regenerate launch script (points to transient paths need refreshing)
        self._create_launch_script()

        # Update state
        self.state.set_app(self.name, {
            "status": "installed",
            "app_dir": self.app_dir,
            "venv_dir": self.venv_dir,
            "recovered_at": time.time(),
            "version": self._get_version(),
        })

        log("ComfyUI recovery complete!")

    def get_status(self) -> dict:
        """Get current ComfyUI status."""
        app_exists = os.path.isdir(self.app_dir)
        venv_exists = os.path.isdir(self.venv_dir)
        torch_ok = check_torch_cuda(self.venv_dir) if venv_exists else False

        saved = self.state.get_app(self.name) or {}
        status = "installed" if (app_exists and venv_exists and torch_ok) else "broken"

        return {
            "status": status,
            "app_dir": self.app_dir,
            "venv_dir": self.venv_dir,
            "version": self._get_version() if app_exists else None,
            "torch_cuda": torch_ok,
            **{k: v for k, v in saved.items() if k not in ("status", "app_dir", "venv_dir", "version")},
        }

    def _create_launch_script(self) -> None:
        """Create the launch script at cloudgpu/bin/comfyui."""
        os.makedirs(self.state.bin_dir, exist_ok=True)
        python = venv_python(self.venv_dir)
        main_py = os.path.join(self.app_dir, "main.py")

        script = f"""#!/bin/bash
# ComfyUI launcher - generated by cloudgpu
exec {python} {main_py} --listen 127.0.0.1 --port 8188 "$@"
"""
        with open(self.launch_script, "w") as f:
            f.write(script)
        os.chmod(self.launch_script, os.stat(self.launch_script).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        log(f"Launch script created: {self.launch_script}")

    def _get_version(self) -> str | None:
        """Get the current git commit short hash."""
        if not os.path.isdir(self.app_dir):
            return None
        try:
            result = run(
                ["git", "-C", self.app_dir, "rev-parse", "--short", "HEAD"],
                capture=True,
                check=False,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None
