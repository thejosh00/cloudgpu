"""App-venv checks run as cloudgpu-app, and never turn "couldn't check" into "torch is broken".

cloudgpu-app can write the app venvs, and ubuntu has passwordless sudo, so a venv's Python
running as ubuntu hands app code the Lambda key. These tests use no real sudo or instance: the
venv's bin/python3 is a planted script that records who it ran as, and a fake sudo first on PATH
records every call, in order, in the same log.
"""

from __future__ import annotations

import os
import pwd
import stat
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cloudgpu.remote import utils
from cloudgpu.remote.apps.aitoolkit import AIToolkitInstaller
from cloudgpu.remote.apps.comfyui import ComfyUIInstaller
from cloudgpu.remote.state import State
from cloudgpu.remote.utils import check_torch_cuda

APP_USER = "cloudgpu-app"
GOOD_VENV = "planted-by-test"  # not pyvenv.cfg: a rebuild would recreate that

FAKE_SUDO = """#!/bin/bash
# Records its arguments. "sudo -u <user> -- cmd" runs cmd, marking who it was meant to run as;
# anything else is an admin command, which is only recorded (useradd also "creates" the user).
# sudo-refuses makes the run-as-user form fail the way sudo itself fails: exit 1.
echo "sudo $*" >> "$SANDBOX/calls.log"
if [ "$1" = useradd ]; then touch "$SANDBOX/user-exists"; exit 0; fi
target=
while [ $# -gt 0 ]; do
  case "$1" in
    -u) target="$2"; shift 2 ;;
    --) shift; break ;;
    -*) shift ;;
    *) exit 0 ;;
  esac
done
[ -n "$target" ] || exit 0
[ -e "$SANDBOX/sudo-refuses" ] && { echo "sudo: a password is required" >&2; exit 1; }
FAKE_SUDO_TARGET="$target" exec "$@"
"""

PLANTED_PYTHON = """#!/bin/bash
# The file app code would plant in the venv. Records who ran it, then behaves like Python.
echo "planted ran as ${{FAKE_SUDO_TARGET:-the caller, directly}}" >> "$SANDBOX/calls.log"
exec {real_python} "$@"
"""


def executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def plant_venv(venv: Path) -> None:
    executable(venv / "bin" / "python3", PLANTED_PYTHON.format(real_python=sys.executable))
    (venv / GOOD_VENV).write_text("gone if the venv is ever rebuilt\n")


class Sandbox:
    def __init__(self, root: Path):
        self.root = root
        self.log = root / "calls.log"
        self.modules = root / "modules"

    def calls(self) -> list[str]:
        return self.log.read_text().splitlines() if self.log.exists() else []

    def planted_runs(self) -> list[str]:
        return [c for c in self.calls() if c.startswith("planted ran as")]

    def user_exists(self) -> None:
        (self.root / "user-exists").touch()

    def sudo_refuses(self) -> None:
        (self.root / "sudo-refuses").touch()

    def torch(self, cuda: bool) -> None:
        pkg = self.modules / "torch"
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "__init__.py").write_text(
            f"class cuda:\n    is_available = staticmethod(lambda: {cuda})\n")

    def torchcodec(self, loads: bool) -> None:
        pkg = self.modules / "torchcodec"
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "decoders.py").write_text("VideoDecoder = 1\n" if loads else "raise ImportError\n")

    def index(self, *needles: str) -> int:
        """Position of the first call containing every needle."""
        return next(i for i, c in enumerate(self.calls()) if all(n in c for n in needles))


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Fake sudo first on PATH; cloudgpu-app exists only once the sandbox says so."""
    root = tmp_path / "sandbox"
    root.mkdir()
    box = Sandbox(root)
    box.modules.mkdir()
    executable(root / "bin" / "sudo", FAKE_SUDO)
    monkeypatch.setenv("SANDBOX", str(root))
    monkeypatch.setenv("PATH", f"{root / 'bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PYTHONPATH", str(box.modules))

    real_getpwnam = pwd.getpwnam

    def getpwnam(name):
        if name != APP_USER:
            return real_getpwnam(name)
        if (root / "user-exists").exists():
            return MagicMock(pw_name=APP_USER)
        raise KeyError(name)

    monkeypatch.setattr(pwd, "getpwnam", getpwnam)
    return box


@pytest.fixture
def venv(tmp_path):
    path = tmp_path / "venvs" / "app"
    plant_venv(path)
    return path


class TestCheckTorchCuda:
    def test_planted_python_only_reached_through_sudo_as_app_user(self, sandbox, venv):
        """Bug: the venv's Python runs as ubuntu, so app code gets passwordless sudo."""
        sandbox.user_exists()
        sandbox.torch(cuda=True)
        assert check_torch_cuda(str(venv)) is True
        assert sandbox.planted_runs() == [f"planted ran as {APP_USER}"]

    def test_sudo_keeps_its_environment_reset(self, sandbox, venv):
        """Bug: -E or --preserve-env would hand the ubuntu environment to app code."""
        sandbox.user_exists()
        sandbox.torch(cuda=True)
        check_torch_cuda(str(venv))
        sudo = next(c for c in sandbox.calls() if c.startswith("sudo "))
        assert sudo.startswith(f"sudo -n -H -u {APP_USER} -- {venv}/bin/python3 -c ")

    def test_check_that_ran_and_failed_is_still_a_verdict(self, sandbox, venv):
        """Bug: real torch breakage stops being reported, so a broken venv is never rebuilt."""
        sandbox.user_exists()
        sandbox.torch(cuda=False)
        assert check_torch_cuda(str(venv)) is False

    def test_torch_missing_is_a_verdict(self, sandbox, venv):
        """Bug: a venv without torch (its import fails) reads as "couldn't check"."""
        sandbox.user_exists()
        assert check_torch_cuda(str(venv)) is False

    def test_venv_without_python_is_a_verdict(self, sandbox, tmp_path):
        """Bug: a venv whose interpreter is gone is never rebuilt."""
        sandbox.user_exists()
        (tmp_path / "empty").mkdir()
        assert check_torch_cuda(str(tmp_path / "empty")) is False
        assert sandbox.calls() == []

    def test_missing_user_is_not_a_verdict(self, sandbox, venv):
        """Bug: on a new instance a good venv reads as broken, and gets deleted."""
        sandbox.torch(cuda=True)
        with pytest.raises(RuntimeError, match=APP_USER):
            check_torch_cuda(str(venv))
        assert sandbox.planted_runs() == []

    def test_sudo_refusing_is_not_a_verdict(self, sandbox, venv):
        """Bug: sudo failing (exit 1, like a failed assert) reads as broken torch."""
        sandbox.user_exists()
        sandbox.sudo_refuses()
        with pytest.raises(RuntimeError, match="password is required"):
            check_torch_cuda(str(venv))
        assert sandbox.planted_runs() == []

    def test_timeout_is_not_a_verdict(self, sandbox, venv, monkeypatch):
        """Bug: a slow filesystem makes a good venv read as broken."""
        import subprocess
        sandbox.user_exists()

        def too_slow(*args, **kwargs):
            raise subprocess.TimeoutExpired("python3", 600)

        monkeypatch.setattr(utils.subprocess, "run", too_slow)
        with pytest.raises(RuntimeError):
            check_torch_cuda(str(venv))


class TestTorchcodecProbe:
    @pytest.fixture
    def installer(self, tmp_persistent_dir):
        installer = AIToolkitInstaller(State(tmp_persistent_dir))
        plant_venv(Path(installer.venv_dir))
        return installer

    @pytest.fixture(autouse=True)
    def no_real_pip(self):
        with patch("cloudgpu.remote.apps.aitoolkit.pip_install"), \
                patch("cloudgpu.remote.apps.aitoolkit.torch_index_url", return_value=None):
            yield

    def test_planted_python_only_reached_through_sudo_as_app_user(self, sandbox, installer):
        """Bug: the probe runs venv_python directly, bypassing check_torch_cuda's user."""
        sandbox.user_exists()
        sandbox.torchcodec(loads=True)
        installer._install_torchcodec()
        assert sandbox.planted_runs() == [f"planted ran as {APP_USER}"]

    def test_load_failure_warns(self, sandbox, installer, capsys):
        """Bug: a torchcodec that won't load stops being reported."""
        sandbox.user_exists()
        sandbox.torchcodec(loads=False)
        installer._install_torchcodec()
        assert "fails to load" in capsys.readouterr().err

    def test_probe_that_cannot_run_neither_aborts_nor_runs_the_venv(self, sandbox, installer, capsys):
        """Bug: an unrunnable probe aborts the install, or reports torchcodec as broken."""
        sandbox.torchcodec(loads=True)
        installer._install_torchcodec()
        assert sandbox.planted_runs() == []
        err = capsys.readouterr().err
        assert "fails to load" not in err and "Couldn't check" in err


APPS = [
    pytest.param((ComfyUIInstaller, "cloudgpu.remote.apps.comfyui"), id="comfyui"),
    pytest.param((AIToolkitInstaller, "cloudgpu.remote.apps.aitoolkit"), id="ai-toolkit"),
]


@pytest.fixture(params=APPS)
def app(request, tmp_persistent_dir):
    """An app already on the filesystem, with only what recover/install must not touch left real."""
    cls, module = request.param
    installer = cls(State(tmp_persistent_dir))
    Path(installer.app_dir).mkdir(parents=True)
    Path(installer.ui_dir if cls is AIToolkitInstaller else installer.app_dir).mkdir(exist_ok=True)
    plant_venv(Path(installer.venv_dir))
    patches = [patch(f"{module}.install_service"),
               patch(f"{module}.pip_install"),
               patch(f"{module}.pip_install_torch"),
               patch(f"{module}.service_active", return_value="inactive")]
    if cls is AIToolkitInstaller:
        patches += [patch.object(cls, "_ensure_node"),
                    patch.object(cls, "_build_ui"),
                    patch.object(cls, "_ui_built", return_value=True),
                    patch.object(cls, "_node_works", return_value=True)]
        installer.state.set_app(installer.name, {"node_arch": cls._arch()})
    for p in patches:
        p.start()
    yield installer, module
    for p in patches:
        p.stop()


class TestInstallAndRecovery:
    def test_recover_on_new_instance_keeps_a_good_venv(self, sandbox, app):
        """Bug: cloudgpu-app doesn't exist yet, the check "fails", and the venv is rmtree'd."""
        installer, _ = app
        sandbox.torch(cuda=True)
        installer.recover()
        assert (Path(installer.venv_dir) / GOOD_VENV).exists()
        assert sandbox.planted_runs() == [f"planted ran as {APP_USER}"]

    def test_recover_ensures_user_and_access_before_checking(self, sandbox, app):
        """Bug: losing this order deletes good venvs, or checks a venv the user can't read."""
        installer, _ = app
        sandbox.torch(cuda=True)
        installer.recover()
        useradd = sandbox.index("sudo useradd")
        share = sandbox.index("sudo chgrp", installer.venv_dir)
        check = sandbox.index(f"-u {APP_USER} -- {installer.venv_dir}")
        assert useradd < share < check

    def test_recover_keeps_venv_when_the_check_cannot_run(self, sandbox, app):
        """Bug: sudo failing mid-recovery deletes a good venv."""
        installer, _ = app
        sandbox.user_exists()
        sandbox.torch(cuda=True)
        sandbox.sudo_refuses()
        with pytest.raises(RuntimeError):
            installer.recover()
        assert (Path(installer.venv_dir) / GOOD_VENV).exists()

    def test_recover_still_rebuilds_a_venv_with_broken_torch(self, sandbox, app):
        """Bug: a venv that really is broken (say, after an arch switch) is never rebuilt."""
        installer, module = app
        with patch(f"{module}.run", return_value=MagicMock(returncode=1, stdout="")) as run:
            with patch(f"{module}.check_torch_cuda", side_effect=[False, True]):
                installer.recover()
        assert not (Path(installer.venv_dir) / GOOD_VENV).exists()
        assert any(c.args[0][:3] == ["python3", "-m", "venv"] for c in run.call_args_list)

    def test_install_on_new_instance_ensures_user_before_checking(self, sandbox, app):
        """Bug: installing onto a filesystem with a venv fails, as the user isn't there yet."""
        installer, module = app
        sandbox.torch(cuda=True)
        with patch(f"{module}.run", return_value=MagicMock(returncode=1, stdout="")):
            installer.install()
        assert sandbox.index("sudo useradd") < sandbox.index(f"-u {APP_USER} -- {installer.venv_dir}")
        assert (Path(installer.venv_dir) / GOOD_VENV).exists()


class TestStatus:
    def test_missing_user_reports_unverified_and_changes_nothing(self, sandbox, app):
        """Bug: status runs the venv as ubuntu, creates the user, or calls good torch broken."""
        installer, _ = app
        sandbox.torch(cuda=True)
        before = sorted(p for p in Path(installer.state.persistent_dir).rglob("*"))
        status = installer.get_status()
        assert status["torch_cuda"] is None
        assert status["status"] == "unverified"
        assert sandbox.planted_runs() == []
        assert sandbox.calls() == []  # no sudo at all: no useradd, no chgrp
        assert sorted(p for p in Path(installer.state.persistent_dir).rglob("*")) == before

    def test_runs_the_check_as_app_user_when_the_user_exists(self, sandbox, app):
        """Bug: status keeps running the venv as ubuntu on an established instance."""
        installer, _ = app
        sandbox.user_exists()
        sandbox.torch(cuda=True)
        status = installer.get_status()
        assert status["torch_cuda"] is True
        assert status["status"] == "installed"
        assert sandbox.planted_runs() == [f"planted ran as {APP_USER}"]

    def test_broken_torch_still_reports_broken(self, sandbox, app):
        """Bug: real torch breakage shows as merely unverified."""
        installer, _ = app
        sandbox.user_exists()
        sandbox.torch(cuda=False)
        status = installer.get_status()
        assert status["torch_cuda"] is False
        assert status["status"] == "broken"
