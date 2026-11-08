"""AI Toolkit (ostris) installer — LoRA training with a web UI. Stdlib-only."""

from __future__ import annotations

import os
import platform
import stat
import sys
import time

from ..installer import AppInstaller
from ..run_as_app import app_step
from ..service import (app_status, ensure_torch, install_requirements_as_app, install_service,
                       prepare_app_dir, prepare_venv_checks, service_active, wipe_venv)
from ..utils import (VenvCheckUnavailable, run, pip_install, pip_install_torch,
                     check_torch_cuda, git_pull_as_app, git_version_as_app, requirements_present,
                     run_venv_check, torch_cuda_status, torch_index_url, trust_repo_as_app,
                     venv_python, log)

AITOOLKIT_PORT = 8675

AITOOLKIT_REPO = "https://github.com/ostris/ai-toolkit.git"
NODE_VERSION = "22.17.1"  # README requires Node > 20

# Run as cloudgpu-app (see _patched_requirements). argv: the requirements file, the file to write.
_FLATTEN_REQUIREMENTS = r"""
import os, re, sys

lines = []

def inline(path, parents):
    real = os.path.realpath(path)
    if real in parents:
        sys.exit("requirements include cycle: " + " -> ".join([*parents, real]))
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            include = None
            for flag in ("-r ", "--requirement "):
                if line.startswith(flag):
                    include = line[len(flag):].strip()
            if include:
                inline(os.path.join(os.path.dirname(path), include), [*parents, real])
            elif re.match(r"torchcodec\b", line):
                lines.append(f"# {line}  (pin dropped by cloudgpu)")
            else:
                lines.append(raw.rstrip("\n"))

requirements, patched = sys.argv[1:]
try:
    inline(requirements, [])
    if os.path.lexists(patched):
        os.unlink(patched)
    # O_EXCL: a symlink put here since the unlink fails this rather than redirecting it.
    fd = os.open(patched, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o664)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
except (OSError, UnicodeError) as e:
    sys.exit(str(e))
"""


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
    def hf_home(self) -> str:
        return os.path.join(self.state.persistent_dir, "hf_cache")

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

        # 1. Clone repo. The first clone stays ubuntu's: apps_dir isn't cloudgpu-app's to write,
        # and there's nothing planted in a path that doesn't exist yet.
        fresh = not os.path.exists(self.app_dir)
        if fresh:
            log("Cloning AI Toolkit...")
            os.makedirs(self.state.apps_dir, exist_ok=True)
            run(["git", "clone", AITOOLKIT_REPO, self.app_dir])
        prepare_app_dir(self.app_dir)
        if fresh:
            trust_repo_as_app(self.app_dir)
        else:
            log("AI Toolkit repo already exists, pulling latest...")
            git_pull_as_app(self.app_dir)

        # 2. Create venv
        if not os.path.exists(self.venv_dir):
            log("Creating venv (for PyTorch)...")
            self._create_venv()
        else:
            log("Venv already exists")

        # 3. Verify PyTorch CUDA and install torchvision/torchaudio in venv
        self._ensure_torch()

        # 4. Install requirements (large dep set; aarch64 may compile sdists)
        self._install_requirements()

        # 5. Wire the UI's training-python resolution to our venv
        self._ensure_venv_symlink()

        # 6. Node.js + web UI build
        self._ensure_node()
        self._build_ui()

        # 7. Create launch script + systemd service
        self._create_launch_script()
        self._setup_service()

        # 8. Update state
        self.save_record({
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
            "torch_cuda": torch_cuda_status(self.venv_dir) is True,
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
            # The new install gets a new venv: the profiles must put back what they put in it.
            self.state.mark_reprovision(self.name)
            self.install()
            return

        if not os.path.isdir(self.venv_dir):
            log("Venv missing, recreating...", "warn")
            self._rebuild_venv(wipe=False)

        # Verify PyTorch CUDA (system packages may differ on new instance;
        # also catches an arch switch, where every .so in the venv is wrong)
        prepare_venv_checks(self.venv_dir)
        trust_repo_as_app(self.app_dir)  # for reading its version, here and in `status`
        if not check_torch_cuda(self.venv_dir):
            log("PyTorch CUDA broken, rebuilding venv...", "warn")
            self._rebuild_venv(wipe=True)

        self._ensure_venv_symlink()
        self._ensure_node()

        # Rebuild the UI only when needed: native modules (sqlite3, SWC,
        # prisma engines) are per-arch, and a half-built UI must be finished.
        saved = self.state.get_app(self.name) or {}
        if saved.get("node_arch") != self._arch():
            log("Instance architecture changed, rebuilding UI node_modules...", "warn")
            prepare_app_dir(self.app_dir)
            self._wipe_node_modules()
            self._build_ui()
        elif not self._ui_built() or not os.path.isdir(os.path.join(self.ui_dir, "node_modules")):
            log("UI not fully built, building...", "warn")
            prepare_app_dir(self.app_dir)
            self._build_ui()

        # Regenerate launch script + service (systemd unit is instance-local)
        self._create_launch_script()
        self._setup_service()

        self.save_record({
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
        torch_ok = torch_cuda_status(self.venv_dir) if venv_exists else False
        ui_built = self._ui_built()

        saved = self.state.get_app(self.name) or {}
        if not (app_exists and venv_exists and ui_built) or torch_ok is False:
            status = "broken"
        else:
            status = "installed" if torch_ok else "unverified"
        service = service_active(self.name)

        return {
            "status": app_status(status, service),
            "app_dir": self.app_dir,
            "venv_dir": self.venv_dir,
            "version": self._get_version() if app_exists else None,
            "torch_cuda": torch_ok,
            "node_ok": self._node_works(),
            "ui_built": ui_built,
            "service": service,
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
            # Only its own dirs are writable (UI, datasets and training outputs live under app_dir),
            # plus the HF cache that its jobs and profile provisioning, as this user, both fill.
            "rw_paths": [self.app_dir, self.venv_dir, self.hf_home],
        }

    def _setup_service(self) -> None:
        """Install/enable the systemd service; raises if it can't (so install/recover fail)."""
        # systemd won't start a unit whose ReadWritePaths doesn't exist.
        os.makedirs(self.hf_home, exist_ok=True)
        try:
            install_service(self.service_spec())
        except Exception as e:
            raise RuntimeError(f"Could not set up systemd service '{self.name}': "
                               f"{str(e) or type(e).__name__}") from e

    def _rebuild_venv(self, *, wipe: bool) -> None:
        """Make a new venv in place of a missing or wiped one, as install would.

        Marks the app first: whatever the profiles put in the venv is gone, and the mark has to
        outlive a recover or `up` that dies part way through (as ComfyUI's _rebuild_venv).
        """
        self.state.mark_reprovision(self.name)
        prepare_app_dir(self.app_dir)
        if wipe:
            wipe_venv(self.venv_dir)
        self._create_venv()
        self._ensure_torch()
        self._install_requirements()

    def _create_venv(self) -> None:
        """Create the venv and bring its pip up to date.

        Ubuntu 22.04 seeds fresh venvs with pip 22.0.2, which predates enough
        resolver and wheel-tag work that this dependency set is best installed
        with a current pip.
        """
        os.makedirs(self.state.venvs_dir, exist_ok=True)
        run(["python3", "-m", "venv", self.venv_dir])
        pip_install(self.venv_dir, "--upgrade", "pip", timeout=300)

    def _ensure_torch(self) -> None:
        """Make sure the venv has a working CUDA PyTorch, plus vision/audio (shared with ComfyUI)."""
        ensure_torch(self.venv_dir, prepare=prepare_venv_checks, check=check_torch_cuda,
                     install_torch=pip_install_torch, install=pip_install)

    def _install_requirements(self) -> None:
        """Install ai-toolkit's requirements, then torchcodec separately.
        prepare_app_dir must have run, so cloudgpu-app can read and flatten them."""
        requirements = os.path.join(self.app_dir, "requirements.txt")
        if not requirements_present(requirements):
            return
        log("Installing AI Toolkit requirements...")
        install_requirements_as_app(self.venv_dir, self._patched_requirements(requirements),
                                    timeout=1800)
        self._install_torchcodec()

    def _patched_requirements(self, requirements: str) -> str:
        """Flatten the requirements into one file with the torchcodec pin gone.

        Upstream pins torchcodec to the release built for *its* pinned torch
        (0.9.1 / torch 2.9) and then upgrades it out of the way in its own
        installer. We install the newest torch the instance's driver supports,
        so that pin is wrong for our torch — and on aarch64 (GH200) it is
        unobtainable: torchcodec publishes no Linux arm64 wheel before 0.10, so
        the whole requirements pass dies on it. Flattening is needed because
        the pin lives in requirements_base.txt, pulled in via '-r'.

        Runs as cloudgpu-app, which chooses every path here: as ubuntu, a symlink or an include
        would copy any file ubuntu can read into app_dir, and a symlink at the output would
        overwrite any file ubuntu can write. Raises if it can't flatten them.
        """
        patched = os.path.join(self.app_dir, ".cloudgpu-requirements.txt")
        result = app_step([sys.executable, "-I", "-c", _FLATTEN_REQUIREMENTS, requirements, patched],
                          capture=True, check=False, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"Couldn't read {requirements}: {result.stderr.strip()}")
        return patched

    def _install_torchcodec(self) -> None:
        """Install torchcodec from the same wheel index as torch.

        The PyPI wheels are CUDA 13 builds that fail to dlopen libnvrtc.so.13
        against a cu12x torch; download.pytorch.org publishes a build per CUDA
        tag, and the newest torchcodec on an index pairs with the newest torch
        on that same index — which is exactly what pip_install_torch() puts in
        the venv. Only video/audio datasets use it (via torchaudio), so a
        failure here warns instead of aborting the install.
        """
        index_url = torch_index_url()
        args = ["--no-deps", "torchcodec"]  # its only deps are test extras
        if index_url:
            args += ["--index-url", index_url]
        log("Installing torchcodec (video/audio dataset support)...")
        try:
            pip_install(self.venv_dir, *args, timeout=600)
        except Exception as e:
            log(f"torchcodec install failed ({e}); video and audio datasets "
                f"will not work, image training is unaffected", "warn")
            return
        try:
            loads = run_venv_check(self.venv_dir,
                                   "from torchcodec.decoders import VideoDecoder",
                                   timeout=300)
        except VenvCheckUnavailable as e:
            log(f"Couldn't check that torchcodec loads ({e}); video and audio datasets "
                f"may not work, image training is unaffected", "warn")
            return
        if not loads:
            log("torchcodec installed but fails to load (needs FFmpeg shared "
                "libraries and a matching torch build); video and audio "
                "datasets will not work, image training is unaffected", "warn")

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

    def _wipe_node_modules(self) -> None:
        """Remove node_modules as cloudgpu-app, or raise.

        As ubuntu this could be refused on files cloudgpu-app wrote: `usermod -aG` doesn't reach
        a process that's already running, so ubuntu isn't in the group yet on a new instance.
        """
        node_modules = os.path.join(self.ui_dir, "node_modules")
        app_step(["rm", "-rf", node_modules], timeout=1800)
        if os.path.lexists(node_modules):
            raise RuntimeError(f"Couldn't remove {node_modules}")

    def _build_ui(self) -> None:
        """npm install + prisma db setup + next build (build once here; the
        service only runs 'start', never rebuilds). Every step runs as cloudgpu-app, because
        package.json and node_modules are theirs to write. prepare_app_dir must have run."""
        npm = os.path.join(self.node_bin, "npm")
        env = {
            "PATH": self.node_bin + os.pathsep + os.environ.get("PATH", ""),
            "NEXT_TELEMETRY_DISABLED": "1",
        }
        log("Installing UI dependencies (npm install)...")
        self._npm_install(npm, env)
        self._ensure_sqlite3(npm, env)
        log("Setting up UI database (prisma)...")
        app_step([npm, "run", "update_db"], cwd=self.ui_dir, env=env, timeout=600)
        log("Building UI (next build)...")
        # next build can exhaust the default Node heap
        build_env = {**env, "NODE_OPTIONS": "--max-old-space-size=4096"}
        app_step([npm, "run", "build"], cwd=self.ui_dir, env=build_env, timeout=1800)

    def _npm_install(self, npm: str, env: dict) -> None:
        """npm install, retried once if it leaves node_modules half-written.

        npm downloads and extracts packages in parallel, and on this NFS with a
        cold cache (cloudgpu-app's HOME is instance-local, so it starts empty) the
        extraction races itself: files vanish mid-write with TAR_ENTRY_ERROR
        ENOENT and packages land without their package.json. 'npm install'
        still exits 0, so the damage only surfaces much later — a package that
        lost its metadata reads as MODULE_NOT_FOUND during 'next build'. The
        retry extracts from the now-warm cache, which does not race.
        """
        app_step([npm, "install"], cwd=self.ui_dir, env=env, timeout=1800)
        if self._npm_tree_ok(npm, env):
            return
        log("node_modules came out incomplete (npm's parallel extraction races "
            "on NFS); wiping and reinstalling from the warm cache...", "warn")
        self._wipe_node_modules()
        app_step([npm, "install"], cwd=self.ui_dir, env=env, timeout=1800)
        if not self._npm_tree_ok(npm, env):
            raise RuntimeError("npm install left an incomplete node_modules")

    def _npm_tree_ok(self, npm: str, env: dict) -> bool:
        """True when every installed package matches package.json.

        'npm ls' exits non-zero for invalid or missing entries; unmet
        *optional* deps (the per-platform swc/macstats builds) exit 0, so they
        do not trip this. False wipes node_modules, so a check that couldn't run raises
        CouldNotRun instead.
        """
        result = app_step([npm, "ls", "--all"], cwd=self.ui_dir, env=env,
                          check=False, capture=True, timeout=300)
        return result.returncode == 0

    def _sqlite3_loads(self, env: dict) -> bool:
        """False starts a source rebuild, so a check that couldn't run raises CouldNotRun."""
        node = os.path.join(self.node_bin, "node")
        result = app_step([node, "-e", "require('sqlite3')"], cwd=self.ui_dir,
                          env=env, check=False, capture=True, timeout=120)
        return result.returncode == 0

    def _ensure_sqlite3(self, npm: str, env: dict) -> None:
        """Rebuild sqlite3 from source when its prebuilt binary won't load.

        sqlite3 6.x ships napi prebuilds linked against glibc 2.38, but Lambda
        instances run Ubuntu 22.04 (glibc 2.35), so 'next build' dies with
        ERR_DLOPEN_FAILED while collecting page data. Compiling locally against
        the system glibc fixes it (needs g++/make, present on the image).
        """
        if self._sqlite3_loads(env):
            return
        log("sqlite3 prebuilt binary is incompatible with this system's glibc; "
            "rebuilding from source...", "warn")
        app_step([npm, "rebuild", "sqlite3", "--build-from-source"],
                 cwd=self.ui_dir,
                 env={**env, "npm_config_build_from_source": "true"},
                 timeout=1800)
        if not self._sqlite3_loads(env):
            raise RuntimeError("sqlite3 still fails to load after a source rebuild")
        log("sqlite3 rebuilt from source")

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
        script = f"""#!/bin/bash
# AI Toolkit launcher - generated by cloudgpu
# Loopback only; reach via 'cloudgpu forward' (http://localhost:{AITOOLKIT_PORT}).
# venv on PATH so spawned training jobs resolve python3; node on PATH for npm/npx.
export PATH="{self.node_bin}:{self.venv_dir}/bin:$PATH"
export NEXT_TELEMETRY_DISABLED=1
export HF_HOME={self.hf_home}
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
        return git_version_as_app(self.app_dir)
