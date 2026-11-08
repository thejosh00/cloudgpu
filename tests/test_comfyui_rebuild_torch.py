"""ComfyUI's venv rebuild gets the same torch step as install, and marks its profiles (plan item 0.7).

Everything that would change the host is stubbed in-process and recorded as an event, so the
tests see the order of the steps: no sudo, pip, systemctl or venv creation runs for real.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from cloudgpu.remote import __main__ as remote_main
from cloudgpu.remote import recovery
from cloudgpu.remote.apps.comfyui import ComfyUIInstaller
from cloudgpu.remote.state import State

# Wherever a step may be looked up from: the installer, and the shared venv helpers.
MODULES = ["cloudgpu.remote.apps.comfyui", "cloudgpu.remote.service"]
NOOPS = ("prepare_venv_checks", "prepare_app_dir", "trust_repo_as_app", "git_pull_as_app",
         "install_service", "migrate_ownership", "ensure_app_user")


def _no_real_commands(*args, **kwargs):
    raise AssertionError(f"a real command was about to run: {args[0] if args else kwargs}")


class Host:
    """The stubbed host: records each step as an event, answers the torch check from a script."""

    def __init__(self, installer: ComfyUIInstaller, torch_answers: list[bool]):
        self.installer = installer
        self.answers = list(torch_answers)
        self.events: list[str] = []

    def check(self, venv_dir):
        assert venv_dir == self.installer.venv_dir
        answer = self.answers.pop(0) if self.answers else True
        self.events.append(f"check:{answer}")
        return answer

    def torch(self, venv_dir, timeout=600):
        self.events.append("cuda-torch")

    def pip(self, venv_dir, *args, timeout=600):
        self.events.append("pip " + " ".join(args))

    def requirements(self, venv_dir, requirements, *, timeout):
        rel = os.path.relpath(os.path.dirname(requirements), self.installer.app_dir)
        self.events.append(f"-r {rel}")

    def run(self, cmd, **kwargs):
        if cmd[:3] == ["python3", "-m", "venv"]:
            os.makedirs(cmd[3], exist_ok=True)
            self.events.append("venv")
        else:
            self.events.append("run " + " ".join(cmd))
        return MagicMock(returncode=0, stdout="")

    def wipe(self, venv_dir):
        shutil.rmtree(venv_dir, ignore_errors=True)
        self.events.append("wipe")

    def after(self, event: str) -> list[str]:
        """The events from the last ``event`` on."""
        i = len(self.events) - 1 - self.events[::-1].index(event)
        return self.events[i:]

    def stubs(self, stack: ExitStack) -> None:
        steps = {
            "check_torch_cuda": self.check,
            "pip_install_torch": self.torch,
            "pip_install": self.pip,
            "install_requirements_as_app": self.requirements,
            "run": self.run,
            "wipe_venv": self.wipe,
            "git_version_as_app": lambda *a, **k: "abc1234",
            "service_active": lambda *a, **k: "active",
        }
        for name in NOOPS:
            steps[name] = lambda *a, **k: None
        for module_name in MODULES:
            module = importlib.import_module(module_name)
            for name, fake in steps.items():
                if hasattr(module, name):
                    stack.enter_context(patch.object(module, name, side_effect=fake))
        # Anything not stubbed above must not reach the host.
        stack.enter_context(patch.object(subprocess, "run", _no_real_commands))
        stack.enter_context(patch.object(subprocess, "Popen", _no_real_commands))


@pytest.fixture
def comfy(tmp_path):
    """ComfyUI on the filesystem from an earlier instance: app, Manager, both with requirements."""
    state = State(str(tmp_path / "nfs"))
    state._ensure_dirs()
    installer = ComfyUIInstaller(state)
    manager = os.path.join(installer.app_dir, "custom_nodes", "ComfyUI-Manager")
    os.makedirs(manager)
    with open(os.path.join(installer.app_dir, "requirements.txt"), "w") as f:
        f.write("pyyaml\n")
    with open(os.path.join(manager, "requirements.txt"), "w") as f:
        f.write("gitpython\n")
    with open(os.path.join(installer.app_dir, "main.py"), "w") as f:
        f.write("# main\n")
    state.set_app("comfyui", {"status": "installed", "app_dir": installer.app_dir,
                              "venv_dir": installer.venv_dir, "recovered_at": 1.0, "version": "old"})
    return installer


def host(installer, answers, *, venv: bool):
    if venv:
        os.makedirs(installer.venv_dir, exist_ok=True)
    stack = ExitStack()
    h = Host(installer, answers)
    h.stubs(stack)
    return h, stack


AFTER_TORCH = ["pip --no-deps torchvision torchaudio", "-r .", "-r custom_nodes/ComfyUI-Manager"]


class TestRebuildGetsTheTorchStep:
    def test_venv_missing_installs_cuda_torch_rechecks_then_requirements(self, comfy):
        """Catches: a recreated venv getting only ComfyUI's requirements, so torch is CPU-only."""
        h, stack = host(comfy, [False, True], venv=False)
        with stack:
            comfy.recover()
        assert h.after("venv")[:7] == ["venv", "check:False", "cuda-torch", "check:True", *AFTER_TORCH], h.events

    def test_broken_torch_rebuild_installs_cuda_torch_rechecks_then_requirements(self, comfy):
        """Catches: the 'PyTorch CUDA broken' rebuild reinstalling requirements over a torch never fixed."""
        h, stack = host(comfy, [False, False, True], venv=True)
        with stack:
            comfy.recover()
        assert h.events[0] == "check:False"
        assert h.after("wipe")[:8] == ["wipe", "venv", "check:False", "cuda-torch", "check:True",
                                       *AFTER_TORCH], h.events

    def test_rebuild_with_working_torch_skips_the_cuda_install_but_not_vision_audio(self, comfy):
        """Catches: torchvision/torchaudio skipped when torch already works, so they don't match it."""
        h, stack = host(comfy, [False, True], venv=True)
        with stack:
            comfy.recover()
        assert h.after("wipe")[:6] == ["wipe", "venv", "check:True", *AFTER_TORCH], h.events

    def test_install_keeps_its_order(self, comfy):
        """Catches: sharing the torch step changing what install does, or its order."""
        h, stack = host(comfy, [False, True], venv=True)
        with stack:
            comfy.install()
        events = [e for e in h.events if not e.startswith("run ")]
        assert events == ["check:False", "cuda-torch", "check:True", *AFTER_TORCH], h.events


def _recover_through_main(monkeypatch, installer):
    monkeypatch.setattr(recovery, "setup_path", lambda st: None)
    monkeypatch.setattr(remote_main, "APP_INSTALLERS", {"comfyui": ComfyUIInstaller})
    monkeypatch.setattr(sys, "argv", ["remote", "--persistent-dir", installer.state.persistent_dir, "recover"])
    try:
        remote_main.main()
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    return 0


class TestTorchStillBroken:
    @pytest.mark.parametrize("venv,answers", [(False, [False, False]), (True, [False, False, False])],
                             ids=["venv-missing", "torch-broken"])
    def test_recover_exits_nonzero_naming_comfyui(self, comfy, monkeypatch, capsys, venv, answers):
        """Catches: a rebuild whose torch never works recorded as recovered, and `recover` exiting 0."""
        h, stack = host(comfy, answers, venv=venv)
        with stack:
            code = _recover_through_main(monkeypatch, comfy)
        text = "".join(capsys.readouterr())
        assert code != 0, h.events
        summary = text[text.index("Recovery failed"):]
        assert "comfyui: Failed to get PyTorch with CUDA support" in summary
        assert "Recovery complete!" not in text
        record = comfy.state.get_app("comfyui")
        assert record["recovered_at"] == 1.0, "not recorded as freshly recovered"
        assert not [e for e in h.events if e.startswith("-r ")], "no requirements over a broken torch"

    def test_installer_raises(self, comfy):
        """Catches: the failed re-check only logged, so recover carries on."""
        h, stack = host(comfy, [False, False], venv=False)
        with stack, pytest.raises(RuntimeError, match="PyTorch with CUDA"):
            comfy.recover()


class TestReprovisionMark:
    @pytest.mark.parametrize("venv,answers", [(False, [False, True]), (True, [False, False, True])],
                             ids=["venv-missing", "torch-broken"])
    def test_rebuild_marks_the_app(self, comfy, venv, answers):
        """Catches: a rebuilt venv (custom nodes' requirements gone) that nothing re-provisions."""
        _, stack = host(comfy, answers, venv=venv)
        with stack:
            comfy.recover()
        record = comfy.state.get_app("comfyui")
        assert record["reprovision"] is True
        assert record["status"] == "installed"

    def test_healthy_recover_does_not_mark(self, comfy):
        """Catches: every recover marking the app, so every `up` re-provisions."""
        h, stack = host(comfy, [True], venv=True)
        with stack:
            comfy.recover()
        assert "wipe" not in h.events and "venv" not in h.events
        assert not comfy.state.get_app("comfyui").get("reprovision")

    def test_mark_survives_a_recover_that_dies_after_the_rebuild(self, comfy):
        """Catches: the mark only written at the end, so a recover that fails after wiping the venv
        (then succeeds next time, on the now-healthy venv) never re-provisions."""
        _, stack = host(comfy, [False, True], venv=False)
        with stack, patch.object(ComfyUIInstaller, "_setup_service", side_effect=RuntimeError("no systemd")):
            with pytest.raises(RuntimeError):
                comfy.recover()
        assert comfy.state.get_app("comfyui")["reprovision"] is True
        _, stack = host(comfy, [True], venv=True)
        with stack:
            comfy.recover()
        assert comfy.state.get_app("comfyui")["reprovision"] is True, "a healthy recover keeps it"

    def test_full_reinstall_from_recover_marks_the_app(self, comfy):
        """Catches: recover reinstalling a missing app dir (custom nodes gone) without a mark."""
        shutil.rmtree(comfy.app_dir)

        def clone(cmd, **kwargs):
            if cmd[:2] == ["git", "clone"]:
                os.makedirs(cmd[-1])
            elif cmd[:3] == ["python3", "-m", "venv"]:
                os.makedirs(cmd[3], exist_ok=True)
            return MagicMock(returncode=0, stdout="")

        _, stack = host(comfy, [True], venv=False)
        with stack, patch("cloudgpu.remote.apps.comfyui.run", side_effect=clone), \
                patch("cloudgpu.remote.apps.comfyui.app_step"):
            comfy.recover()
        assert comfy.state.get_app("comfyui")["reprovision"] is True


def test_remote_clear_reprovision_drops_only_the_mark(comfy, monkeypatch):
    """Catches: `up` having no way to clear the mark, or clearing it wiping the app's record."""
    comfy.state.mark_reprovision("comfyui")
    monkeypatch.setattr(sys, "argv", ["remote", "--persistent-dir", comfy.state.persistent_dir,
                                      "clear-reprovision", "--app", "comfyui"])
    remote_main.main()
    record = comfy.state.get_app("comfyui")
    assert "reprovision" not in record
    assert record["status"] == "installed" and record["recovered_at"] == 1.0
