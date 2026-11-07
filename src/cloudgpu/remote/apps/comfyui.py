"""ComfyUI installer. Stdlib-only."""

from __future__ import annotations

import os
import stat
import time

from ..installer import AppInstaller
from ..run_as_app import app_step
from ..service import (install_requirements_as_app, install_service, prepare_app_dir,
                       prepare_venv_checks, service_active, wipe_venv)
from ..utils import (run, pip_install, pip_install_torch, venv_python, check_torch_cuda,
                     git_pull_as_app, git_version_as_app, requirements_present, torch_cuda_status,
                     trust_repo_as_app, log)

COMFYUI_PORT = 8188

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

        # 1. Clone ComfyUI. The first clone stays ubuntu's: apps_dir isn't cloudgpu-app's to write,
        # and there's nothing planted in a path that doesn't exist yet.
        fresh = not os.path.exists(self.app_dir)
        if fresh:
            log("Cloning ComfyUI...")
            os.makedirs(self.state.apps_dir, exist_ok=True)
            run(["git", "clone", COMFYUI_REPO, self.app_dir])
        prepare_app_dir(self.app_dir)
        if fresh:
            trust_repo_as_app(self.app_dir)
        else:
            log("ComfyUI repo already exists, pulling latest...")
            git_pull_as_app(self.app_dir)

        # 2. Create venv with
        if not os.path.exists(self.venv_dir):
            log("Creating venv (for PyTorch)...")
            os.makedirs(self.state.venvs_dir, exist_ok=True)
            run(["python3", "-m", "venv", self.venv_dir])
        else:
            log("Venv already exists")

        # 3. Verify PyTorch CUDA and install torchvision/torchaudio in venv
        log("Checking PyTorch CUDA availability...")
        prepare_venv_checks(self.venv_dir)
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
        self._install_requirements(self.app_dir, timeout=600)

        # 5. Clone ComfyUI-Manager
        custom_nodes = os.path.join(self.app_dir, "custom_nodes")
        manager_dir = os.path.join(custom_nodes, "ComfyUI-Manager")
        if not os.path.exists(manager_dir):
            log("Cloning ComfyUI-Manager...")
            app_step(["mkdir", "-p", custom_nodes], timeout=60)
            app_step(["git", "clone", MANAGER_REPO, manager_dir], timeout=600)
        else:
            log("ComfyUI-Manager already exists, pulling latest...")
            git_pull_as_app(manager_dir)

        # Install manager requirements if they exist
        self._install_requirements(manager_dir, timeout=120)

        # 6. Create launch script + systemd service
        self._create_launch_script()
        self._setup_service()

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
            "torch_cuda": torch_cuda_status(self.venv_dir) is True,
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
            prepare_app_dir(self.app_dir)
            run(["python3", "-m", "venv", self.venv_dir])
            self._install_requirements(self.app_dir, timeout=120)

        # Verify PyTorch CUDA (system packages may differ on new instance)
        prepare_venv_checks(self.venv_dir)
        trust_repo_as_app(self.app_dir)  # for reading its version, here and in `status`
        if not check_torch_cuda(self.venv_dir):
            log("PyTorch CUDA broken, rebuilding venv...", "warn")
            prepare_app_dir(self.app_dir)
            wipe_venv(self.venv_dir)
            run(["python3", "-m", "venv", self.venv_dir])
            self._install_requirements(self.app_dir, timeout=600)

        # Regenerate launch script (transient paths need refreshing) + service
        self._create_launch_script()
        self._setup_service()

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
        torch_ok = torch_cuda_status(self.venv_dir) if venv_exists else False

        saved = self.state.get_app(self.name) or {}
        if not (app_exists and venv_exists) or torch_ok is False:
            status = "broken"
        else:
            status = "installed" if torch_ok else "unverified"

        return {
            "status": status,
            "app_dir": self.app_dir,
            "venv_dir": self.venv_dir,
            "version": self._get_version() if app_exists else None,
            "torch_cuda": torch_ok,
            "service": service_active(self.name),
            "port": COMFYUI_PORT,
            **{k: v for k, v in saved.items() if k not in ("status", "app_dir", "venv_dir", "version")},
        }

    def service_spec(self) -> dict:
        """systemd service: run ComfyUI (loopback) as a managed, auto-restarting server."""
        return {
            "name": self.name,
            "exec_start": self.launch_script,
            "workdir": self.app_dir,
            "port": COMFYUI_PORT,
            # Only its own dirs are writable (models, outputs, custom nodes live under app_dir).
            "rw_paths": [self.app_dir, self.venv_dir],
        }

    def _install_requirements(self, directory: str, *, timeout: int) -> None:
        """Install ``directory``'s requirements as cloudgpu-app, if it has any.
        prepare_app_dir must have run, so cloudgpu-app can read them."""
        requirements = os.path.join(directory, "requirements.txt")
        if requirements_present(requirements):
            install_requirements_as_app(self.venv_dir, requirements, timeout=timeout)

    def _setup_service(self) -> None:
        """Install/enable the systemd service (best-effort: warn rather than abort)."""
        try:
            install_service(self.service_spec())
        except Exception as e:
            log(f"Could not set up systemd service (run 'cloudgpu ssh -- comfyui' "
                f"manually instead): {e}", "warn")

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
        return git_version_as_app(self.app_dir)
