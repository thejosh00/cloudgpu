"""Nothing in an app venv runs as ubuntu during install or a venv rebuild (plan item 0.1-fix).

cloudgpu-app can write an app's venv, and the venv's `bin/pip` is a script run by the venv's
Python, which loads every `.pth` file in site-packages. So any pip step run as ubuntu runs
cloudgpu-app's files as a user with passwordless sudo, whatever packages the step names.

The two ways a command can run are faked in-process and recorded: `utils.run` (as the caller,
ubuntu on an instance) and `app_step` / `run_stoppable` (as cloudgpu-app). Nothing real runs:
no sudo, pip, git or venv creation, and any command that isn't faked fails the test.
"""

from __future__ import annotations

import importlib
import os
import pwd
import shutil
import subprocess
from contextlib import ExitStack
from unittest.mock import patch

import pytest

from cloudgpu.remote.apps.aitoolkit import AIToolkitInstaller
from cloudgpu.remote.apps.comfyui import ComfyUIInstaller
from cloudgpu.remote.run_as_app import APP_USER, CouldNotRun
from cloudgpu.remote.state import State

# Every module that looks up a runner by name.
MODULES = ["cloudgpu.remote.utils", "cloudgpu.remote.service",
           "cloudgpu.remote.apps.comfyui", "cloudgpu.remote.apps.aitoolkit"]
INDEX = "https://download.pytorch.org/whl/cu124"
CALLER, APP = "caller", APP_USER


def _no_real_commands(*args, **kwargs):
    raise AssertionError(f"a real command was about to run: {args[0] if args else kwargs}")


def make_venv(venv_dir: str) -> None:
    """What `python3 -m venv` leaves, as far as the installers look: bin/pip and bin/python3."""
    os.makedirs(os.path.join(venv_dir, "bin"), exist_ok=True)
    for name in ("pip", "python3"):
        with open(os.path.join(venv_dir, "bin", name), "w") as f:
            f.write("# never run\n")


class Host:
    """The faked host. ``commands`` is every command in order, as (who ran it, argv)."""

    def __init__(self, installer, torch_answers: list[bool], refuse: str | None = None):
        self.installer = installer
        self.venv = installer.venv_dir
        self.answers = list(torch_answers)
        self.refuse = refuse  # an argument: the venv pip step naming it can't run as cloudgpu-app
        self.commands: list[tuple[str, list[str]]] = []

    # -- as the caller (utils.run) --
    def run(self, cmd, **kwargs):
        self.commands.append((CALLER, list(cmd)))
        if cmd[:3] == ["python3", "-m", "venv"]:
            make_venv(cmd[3])
        elif cmd[:3] == ["sudo", "rm", "-rf"]:
            shutil.rmtree(cmd[3], ignore_errors=True)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    # -- as cloudgpu-app (run_as_app.app_step) --
    def app_step(self, cmd, **kwargs):
        if self.refuse and self._in_venv(cmd[0]) and self.refuse in cmd:
            raise CouldNotRun(f"couldn't run as '{APP_USER}': sudo: a password is required")
        self.commands.append((APP, list(cmd)))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    # -- as cloudgpu-app (the venv checks' `sudo -u cloudgpu-app -- <venv python> -c ...`) --
    def run_stoppable(self, argv, **kwargs):
        assert argv[:6] == ["sudo", "-n", "-H", "-u", APP_USER, "--"], argv
        self.commands.append((APP, list(argv[6:])))
        if "import torch" in argv[-1]:
            ok = self.answers.pop(0) if self.answers else True
        else:
            ok = True
        return subprocess.CompletedProcess(argv, 0 if ok else 1, "cloudgpu-venv-check-started\n", "")

    def _in_venv(self, arg: str) -> bool:
        return arg.startswith(self.venv + os.sep)

    def venv_commands(self) -> list[tuple[str, list[str]]]:
        """Every command whose program is a file in the venv (bin/pip, bin/python3, ...)."""
        return [(who, cmd) for who, cmd in self.commands if self._in_venv(cmd[0])]

    def pip_steps(self) -> list[str]:
        """Each `<venv>/bin/pip install ...` as "who: its arguments"."""
        pip = os.path.join(self.venv, "bin", "pip")
        return [f"{who}: {' '.join(cmd[2:])}" for who, cmd in self.commands if cmd[0] == pip]

    def caller_touching_venv_files(self) -> list[list[str]]:
        """Caller commands with any argument naming a file inside the venv, program or not."""
        return [cmd for who, cmd in self.commands
                if who == CALLER and any(self._in_venv(a) for a in cmd)]

    def stubs(self, stack: ExitStack) -> None:
        fakes = {"run": self.run, "app_step": self.app_step, "run_stoppable": self.run_stoppable,
                 "torch_index_url": lambda: INDEX,
                 # Starts a real service and waits for it to settle; not this item's concern.
                 "install_service": lambda spec: None}
        for module_name in MODULES:
            module = importlib.import_module(module_name)
            for name, fake in fakes.items():
                if hasattr(module, name):
                    stack.enter_context(patch.object(module, name, side_effect=fake))
        real_getpwnam = pwd.getpwnam
        stack.enter_context(patch.object(
            pwd, "getpwnam",
            lambda name: real_getpwnam(pwd.getpwuid(os.getuid()).pw_name if name == APP_USER else name)))
        # Node and the UI build have nothing to do with the venv, and download for real.
        for name in ("_ensure_node", "_build_ui"):
            if hasattr(self.installer, name):
                stack.enter_context(patch.object(type(self.installer), name, lambda self: None))
        stack.enter_context(patch.object(subprocess, "run", _no_real_commands))
        stack.enter_context(patch.object(subprocess, "Popen", _no_real_commands))


def app_on_disk(tmp_path, cls, *, venv: bool):
    """An app on the filesystem from an earlier instance, with requirements files."""
    state = State(str(tmp_path / "nfs"))
    state._ensure_dirs()
    installer = cls(state)
    os.makedirs(installer.app_dir)
    with open(os.path.join(installer.app_dir, "requirements.txt"), "w") as f:
        f.write("pyyaml\n")
    if cls is ComfyUIInstaller:
        manager = os.path.join(installer.app_dir, "custom_nodes", "ComfyUI-Manager")
        os.makedirs(manager)
        with open(os.path.join(manager, "requirements.txt"), "w") as f:
            f.write("gitpython\n")
    else:
        os.makedirs(installer.ui_dir)
    if venv:
        make_venv(installer.venv_dir)
    return installer


# path name -> (is there a venv to start with, the torch check's answers, the method to call).
# Each path's first torch check inside ensure_torch fails, so the CUDA torch install runs too.
PATHS = {
    "install": (False, [False, True], "install"),
    "rebuild-missing-venv": (False, [False, True], "recover"),
    "rebuild-broken-torch": (True, [False, False, True], "recover"),
}
APPS = {"comfyui": ComfyUIInstaller, "ai-toolkit": AIToolkitInstaller}
TORCH = f"--force-reinstall torch torchvision torchaudio --index-url {INDEX}"
VISION = "--no-deps torchvision torchaudio"
CODEC = f"--no-deps torchcodec --index-url {INDEX}"


def drive(tmp_path, app: str, path: str, *, refuse: str | None = None) -> Host:
    venv, answers, method = PATHS[path]
    installer = app_on_disk(tmp_path, APPS[app], venv=venv)
    host = Host(installer, answers, refuse)
    with ExitStack() as stack:
        host.stubs(stack)
        host.exception = None
        try:
            getattr(installer, method)()
        except Exception as e:  # kept for the tests that expect one
            host.exception = e
    return host


every_path = pytest.mark.parametrize("path", list(PATHS))
every_app = pytest.mark.parametrize("app", list(APPS))


class TestEveryVenvCommandRunsAsApp:
    @every_app
    @every_path
    def test_no_venv_program_runs_as_the_caller(self, tmp_path, app, path):
        """Catches: a pip step (torchvision/torchaudio, CUDA torch, `--upgrade pip`, torchcodec)
        run as ubuntu, which runs any `.pth` file cloudgpu-app left in the venv as ubuntu."""
        host = drive(tmp_path, app, path)
        assert host.exception is None, host.exception
        ran = host.venv_commands()
        as_caller = [" ".join(cmd) for who, cmd in ran if who == CALLER]
        assert as_caller == [], "ran as the caller:\n" + "\n".join(as_caller)
        assert host.caller_touching_venv_files() == []
        # And the steps did run, so this isn't passing because they were skipped.
        steps = host.pip_steps()
        expected = [f"{APP}: {TORCH}", f"{APP}: {VISION}"]
        if app == "ai-toolkit":
            expected = [f"{APP}: --upgrade pip", *expected, f"{APP}: {CODEC}"]
        for step in expected:
            assert step in steps, f"missing '{step}' in:\n" + "\n".join(steps)
        assert any(s.startswith(f"{APP}: -r ") for s in steps), steps
        assert any(who == APP and cmd[0].endswith("bin/python3") for who, cmd in ran), ran

    @every_app
    @every_path
    def test_the_venv_is_shared_before_each_pip_step(self, tmp_path, app, path):
        """Catches: pip as cloudgpu-app on a venv ubuntu has just made and not yet shared (the
        `--upgrade pip` straight after `python3 -m venv`), which fails every fresh install."""
        host = drive(tmp_path, app, path)
        pip = os.path.join(host.venv, "bin", "pip")
        shared, pips = False, 0
        for who, cmd in host.commands:
            if cmd[:3] == ["python3", "-m", "venv"]:
                shared = False  # new files, ubuntu's alone
            elif cmd[:4] == ["sudo", "chgrp", "-hR", APP_USER] and cmd[4] == host.venv:
                shared = True
            elif cmd[0] == pip:
                pips += 1
                assert shared, f"pip ran before the venv was shared: {' '.join(cmd)}"
        assert pips >= 3


class TestNoFallbackToTheCaller:
    REFUSED = {"comfyui": ["--force-reinstall", "torchvision"],
               "ai-toolkit": ["--upgrade", "--force-reinstall", "torchvision", "torchcodec"]}

    @pytest.mark.parametrize("app,step", [(a, s) for a, steps in REFUSED.items() for s in steps])
    def test_install_fails_when_a_pip_step_cannot_run_as_app(self, tmp_path, app, step):
        """Catches: a pip step that can't start as cloudgpu-app being run as ubuntu instead, or
        (torchcodec) being logged as a warning while the install carries on as if it were fine."""
        host = drive(tmp_path, app, "install", refuse=step)
        assert isinstance(host.exception, CouldNotRun), host.exception
        assert [c for who, c in host.venv_commands() if who == CALLER] == []
        assert host.caller_touching_venv_files() == []
        assert not any(step in s for s in host.pip_steps()), host.pip_steps()
        assert (host.installer.state.get_app(host.installer.name) or {}).get("status") != "installed"

    def test_torchcodec_that_fails_as_app_is_still_only_a_warning(self, tmp_path, capsys):
        """Catches: the CouldNotRun rule swallowing torchcodec's own failure, so an ordinary
        missing wheel (aarch64) aborts an ai-toolkit install that image training doesn't need."""
        installer = app_on_disk(tmp_path, AIToolkitInstaller, venv=True)
        host = Host(installer, [])
        real = host.app_step

        def codec_fails(cmd, **kwargs):
            result = real(cmd, **kwargs)
            if "torchcodec" in cmd:
                raise subprocess.CalledProcessError(1, cmd)
            return result

        host.app_step = codec_fails
        with ExitStack() as stack:
            host.stubs(stack)
            installer._install_torchcodec()
        assert f"{APP}: {CODEC}" in host.pip_steps()
        assert "torchcodec install failed" in capsys.readouterr().err
