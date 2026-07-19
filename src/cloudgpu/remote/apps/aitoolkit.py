"""AI Toolkit (ostris) installer — LoRA training with a web UI. Stdlib-only."""

from __future__ import annotations

import os
import platform
import shutil
import stat
import time

from ..installer import AppInstaller
from ..service import install_service, service_active
from ..utils import run, pip_install, pip_install_torch, check_torch_cuda, log

AITOOLKIT_PORT = 8675

AITOOLKIT_REPO = "https://github.com/ostris/ai-toolkit.git"
NODE_VERSION = "22.17.1"  # README requires Node > 20


class AIToolkitInstaller(AppInstaller):

    @property
    def name(self) -> str:
        return "ai-toolkit"

    @property
    def app_dir(self) -> str:
        return os.path.join(self.state.apps_dir, "ai-toolkit")

    @property
    def venv_dir(self) -> str:
        return os.path.join(self.state.venvs_dir, "ai-toolkit")

    @property
    def ui_dir(self) -> str:
        return os.path.join(self.app_dir, "ui")

    @property
    def launch_script(self) -> str:
        return os.path.join(self.state.bin_dir, "ai-toolkit")

    @property
    def node_dir(self) -> str:
        # Arch-suffixed so GH200 (arm64) and A100 (x64) instances can share
        # one NFS filesystem without clobbering each other's Node install.
        return os.path.join(self.state.apps_dir, f"node-{self._arch()}")

    @property
    def node_bin(self) -> str:
        return os.path.join(self.node_dir, "bin")

    @staticmethod
    def _arch() -> str:
        return "arm64" if platform.machine() in ("aarch64", "arm64") else "x64"

    def install(self) -> None:
        """Install AI Toolkit from scratch."""
        log("Installing AI Toolkit...")

        # 1. Clone repo
        if not os.path.exists(self.app_dir):
            log("Cloning AI Toolkit...")
            os.makedirs(self.state.apps_dir, exist_ok=True)
            run(["git", "clone", AITOOLKIT_REPO, self.app_dir])
        else:
            log("AI Toolkit repo already exists, pulling latest...")
            run(["git", "-C", self.app_dir, "pull"])

        # 2. Create venv
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

        log("Installing matching torchvision and torchaudio in venv...")
        pip_install(self.venv_dir, "--no-deps",
                    "torchvision", "torchaudio", timeout=600)

        # 4. Install requirements (large dep set; aarch64 may compile sdists)
        log("Installing AI Toolkit requirements...")
        requirements = os.path.join(self.app_dir, "requirements.txt")
        if os.path.exists(requirements):
            pip_install(self.venv_dir, "-r", requirements, timeout=1800)

        # 5. Wire the UI's training-python resolution to our venv
        self._ensure_venv_symlink()

        # 6. Node.js + web UI build
        self._ensure_node()
        self._build_ui()

        # 7. Create launch script + systemd service
        self._create_launch_script()
        self._setup_service()

        # 8. Update state
        self.state.set_app(self.name, {
            "status": "installed",
            "app_dir": self.app_dir,
            "venv_dir": self.venv_dir,
            "node_arch": self._arch(),
            "installed_at": time.time(),
            "version": self._get_version(),
        })

        log("AI Toolkit installed successfully!")

    def verify(self) -> bool:
        """Verify AI Toolkit installation."""
        node = os.path.join(self.node_bin, "node")
        checks = {
            "app_dir": os.path.isdir(self.app_dir),
            "venv": os.path.isdir(self.venv_dir),
            "run_py": os.path.isfile(os.path.join(self.app_dir, "run.py")),
            "torch_cuda": check_torch_cuda(self.venv_dir),
            "venv_symlink": os.path.islink(os.path.join(self.app_dir, "venv")),
            "node": self._node_works(),
            "ui_built": self._ui_built(),
            "launch_script": os.path.isfile(self.launch_script),
        }
        for check, passed in checks.items():
            status = "OK" if passed else "FAIL"
            log(f"  {check}: {status}")
        return all(checks.values())

    def recover(self) -> None:
        """Recover AI Toolkit on a new instance."""
        log("Recovering AI Toolkit...")

        if not os.path.isdir(self.app_dir):
            log("App directory missing, doing full install", "warn")
            self.install()
            return

        requirements = os.path.join(self.app_dir, "requirements.txt")
        if not os.path.isdir(self.venv_dir):
            log("Venv missing, recreating...", "warn")
            run(["python3", "-m", "venv", self.venv_dir])
            if os.path.exists(requirements):
                pip_install(self.venv_dir, "-r", requirements, timeout=1800)

        # Verify PyTorch CUDA (system packages may differ on new instance;
        # also catches an arch switch, where every .so in the venv is wrong)
        if not check_torch_cuda(self.venv_dir):
            log("PyTorch CUDA broken, rebuilding venv...", "warn")
            shutil.rmtree(self.venv_dir, ignore_errors=True)
            run(["python3", "-m", "venv", self.venv_dir])
            if os.path.exists(requirements):
                pip_install(self.venv_dir, "-r", requirements, timeout=1800)

        self._ensure_venv_symlink()
        self._ensure_node()

        # Rebuild the UI only when needed: native modules (sqlite3, SWC,
        # prisma engines) are per-arch, and a half-built UI must be finished.
        saved = self.state.get_app(self.name) or {}
        if saved.get("node_arch") != self._arch():
            log("Instance architecture changed, rebuilding UI node_modules...", "warn")
            shutil.rmtree(os.path.join(self.ui_dir, "node_modules"), ignore_errors=True)
            self._build_ui()
        elif not self._ui_built() or not os.path.isdir(os.path.join(self.ui_dir, "node_modules")):
            log("UI not fully built, building...", "warn")
            self._build_ui()

        # Regenerate launch script + service (systemd unit is instance-local)
        self._create_launch_script()
        self._setup_service()

        self.state.set_app(self.name, {
            "status": "installed",
            "app_dir": self.app_dir,
            "venv_dir": self.venv_dir,
            "node_arch": self._arch(),
            "recovered_at": time.time(),
            "version": self._get_version(),
        })

        log("AI Toolkit recovery complete!")

    def get_status(self) -> dict:
        """Get current AI Toolkit status."""
        app_exists = os.path.isdir(self.app_dir)
        venv_exists = os.path.isdir(self.venv_dir)
        torch_ok = check_torch_cuda(self.venv_dir) if venv_exists else False
        ui_built = self._ui_built()

        saved = self.state.get_app(self.name) or {}
        status = "installed" if (app_exists and venv_exists and torch_ok and ui_built) else "broken"

        return {
            "status": status,
            "app_dir": self.app_dir,
            "venv_dir": self.venv_dir,
            "version": self._get_version() if app_exists else None,
            "torch_cuda": torch_ok,
            "node_ok": self._node_works(),
            "ui_built": ui_built,
            "service": service_active(self.name),
            "port": AITOOLKIT_PORT,
            **{k: v for k, v in saved.items() if k not in ("status", "app_dir", "venv_dir", "version")},
        }

    def service_spec(self) -> dict:
        """systemd service: run the web UI + job worker (loopback) as a managed server."""
        return {
            "name": self.name,
            "exec_start": self.launch_script,
            "workdir": self.ui_dir,
            "port": AITOOLKIT_PORT,
        }

    def _setup_service(self) -> None:
        """Install/enable the systemd service (best-effort: warn rather than abort)."""
        try:
            install_service(self.service_spec())
        except Exception as e:
            log(f"Could not set up systemd service (run 'cloudgpu ssh -- ai-toolkit' "
                f"manually instead): {e}", "warn")

    def _ensure_venv_symlink(self) -> None:
        """Symlink <app_dir>/venv -> venv_dir so the UI worker spawns training
        jobs with our venv python (ui/cron/pythonPath.ts checks that path)."""
        link = os.path.join(self.app_dir, "venv")
        if os.path.islink(link):
            if os.readlink(link) == self.venv_dir:
                return
            os.remove(link)
        elif os.path.exists(link):
            log(f"{link} exists and is not a symlink; leaving it alone", "warn")
            return
        os.symlink(self.venv_dir, link)

    def _node_works(self) -> bool:
        node = os.path.join(self.node_bin, "node")
        if not os.path.isfile(node):
            return False
        try:
            result = run([node, "--version"], capture=True, check=False, timeout=30)
            return result.returncode == 0
        except Exception:
            return False

    def _ensure_node(self) -> None:
        """Install a standalone Node.js LTS into the persistent dir.

        A tarball on NFS (vs apt/nodesource) survives instance replacement, so
        recover() needs no sudo and no network for Node.
        """
        if self._node_works():
            return
        arch = self._arch()
        url = (f"https://nodejs.org/dist/v{NODE_VERSION}/"
               f"node-v{NODE_VERSION}-linux-{arch}.tar.xz")
        tarball = os.path.join(self.state.apps_dir, f"node-{arch}.tar.xz")
        log(f"Downloading Node.js v{NODE_VERSION} ({arch})...")
        os.makedirs(self.state.apps_dir, exist_ok=True)
        run(["curl", "-fsSL", url, "-o", tarball], timeout=300)
        os.makedirs(self.node_dir, exist_ok=True)
        run(["tar", "-xJf", tarball, "-C", self.node_dir, "--strip-components=1"])
        os.remove(tarball)
        if not self._node_works():
            raise RuntimeError("Node.js install failed")
        result = run([os.path.join(self.node_bin, "node"), "--version"],
                     capture=True, check=False)
        log(f"Node.js installed: {result.stdout.strip()}")

    def _build_ui(self) -> None:
        """npm install + prisma db setup + next build (build once here; the
        service only runs 'start', never rebuilds)."""
        npm = os.path.join(self.node_bin, "npm")
        env = {
            "PATH": self.node_bin + os.pathsep + os.environ.get("PATH", ""),
            "NEXT_TELEMETRY_DISABLED": "1",
        }
        log("Installing UI dependencies (npm install)...")
        run([npm, "install"], cwd=self.ui_dir, env=env, timeout=1800)
        log("Setting up UI database (prisma)...")
        run([npm, "run", "update_db"], cwd=self.ui_dir, env=env, timeout=600)
        log("Building UI (next build)...")
        # next build can exhaust the default Node heap
        build_env = {**env, "NODE_OPTIONS": "--max-old-space-size=4096"}
        run([npm, "run", "build"], cwd=self.ui_dir, env=build_env, timeout=1800)

    def _ui_built(self) -> bool:
        return (os.path.isdir(os.path.join(self.ui_dir, ".next"))
                and os.path.isfile(os.path.join(self.ui_dir, "dist", "cron", "worker.js")))

    def _create_launch_script(self) -> None:
        """Create the launch script at cloudgpu/bin/ai-toolkit."""
        os.makedirs(self.state.bin_dir, exist_ok=True)

        # Replicates the upstream ui 'start' script but pins the hostname to
        # loopback (upstream's 'next start' binds all interfaces). Not
        # 'build_and_start': that rebuilds on every systemd restart.
        # Auth (AI_TOOLKIT_AUTH) intentionally unsupported: the UI is loopback
        # only, reached via 'cloudgpu forward'; the unit template has no env
        # support yet.
        # Training jobs (spawned by the worker) inherit this env; HF_HOME on the
        # persistent fs so the ~40 GB diffusers weights survive instance swaps.
        hf_home = os.path.join(self.state.persistent_dir, "hf_cache")
        script = f"""#!/bin/bash
# AI Toolkit launcher - generated by cloudgpu
# Loopback only; reach via 'cloudgpu forward' (http://localhost:{AITOOLKIT_PORT}).
# venv on PATH so spawned training jobs resolve python3; node on PATH for npm/npx.
export PATH="{self.node_bin}:{self.venv_dir}/bin:$PATH"
export NEXT_TELEMETRY_DISABLED=1
export HF_HOME={hf_home}
cd {self.ui_dir}
exec {self.node_bin}/npx concurrently --restart-tries -1 --restart-after 1000 -n WORKER,UI \\
  "node dist/cron/worker.js" "next start --port {AITOOLKIT_PORT} --hostname 127.0.0.1"
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
