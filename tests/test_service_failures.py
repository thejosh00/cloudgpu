"""A service that won't install fails install/recover, and status shows a service that isn't up (plan item 0.6).

No real sudo, systemctl or instance: each real installer runs in-process with every
host-changing step stubbed, and any subprocess that slips through fails the test.
`install_service` is what fails (or succeeds), as the bug is in what happens after it.
"""

from __future__ import annotations

import io
import os
import re
import json
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from cloudgpu.local import display
from cloudgpu.remote import __main__ as remote_main
from cloudgpu.remote import recovery
from cloudgpu.remote.apps.aitoolkit import AIToolkitInstaller
from cloudgpu.remote.apps.comfyui import ComfyUIInstaller
from cloudgpu.remote.state import State
from tests.fake_lambda import FakeLambda, run_cloudgpu

APPS = [(ComfyUIInstaller, "cloudgpu.remote.apps.comfyui"),
        (AIToolkitInstaller, "cloudgpu.remote.apps.aitoolkit")]
SERVICE_ERROR = "Job for unit.service failed because the control process exited with error code"
GREEN = "\x1b[32m"


def _no_real_commands(*args, **kwargs):
    raise AssertionError(f"a real command ran in a test: {args[:1]}")


class Box:
    """One real installer on a temp filesystem, with every step that would touch the host stubbed."""

    def __init__(self, cls, module, persistent_dir):
        self.cls, self.module, self.persistent_dir = cls, module, persistent_dir
        self.installer = cls(State(persistent_dir))
        self.service_error: str | None = None
        self.service_state = "active"
        self.service_calls: list[str] = []

    def _install_service(self, spec):
        self.service_calls.append(spec["name"])
        self.rw_paths_existed = all(os.path.isdir(p) for p in spec.get("rw_paths", []))
        if self.service_error is not None:
            raise RuntimeError(self.service_error)

    @property
    def name(self):
        return self.installer.name

    def on_disk(self, record=True):
        """An app already on the filesystem (what recover finds); recorded in state.json if asked."""
        inst = self.installer
        Path(inst.app_dir).mkdir(parents=True, exist_ok=True)
        Path(inst.venv_dir, "bin").mkdir(parents=True, exist_ok=True)
        if self.cls is ComfyUIInstaller:
            Path(inst.app_dir, "custom_nodes", "ComfyUI-Manager").mkdir(parents=True, exist_ok=True)
        else:
            Path(inst.ui_dir, "node_modules").mkdir(parents=True, exist_ok=True)
        if record:
            inst.state.set_app(self.name, {"status": "installed", "node_arch": AIToolkitInstaller._arch()})
        return self

    def start(self):
        """Stub the host-changing steps until stop()."""
        self.stack = stack = ExitStack()
        m = self.module
        stubs = [
            patch(f"{m}.install_service", side_effect=self._install_service),
            patch(f"{m}.service_active", side_effect=lambda name: self.service_state),
            patch(f"{m}.run", return_value=MagicMock(returncode=0, stdout="")),
            patch(f"{m}.app_step"),
            patch(f"{m}.pip_install"), patch(f"{m}.pip_install_torch"),
            patch(f"{m}.install_requirements_as_app"),
            patch(f"{m}.prepare_app_dir"), patch(f"{m}.prepare_venv_checks"),
            patch(f"{m}.trust_repo_as_app"), patch(f"{m}.git_pull_as_app"),
            patch(f"{m}.git_version_as_app", return_value="abc1234"),
            patch(f"{m}.check_torch_cuda", return_value=True),
            patch(f"{m}.torch_cuda_status", return_value=True),
            patch(f"{m}.wipe_venv"),
            # Anything not stubbed above must not reach the host.
            patch.object(subprocess, "run", _no_real_commands),
            patch.object(subprocess, "Popen", _no_real_commands),
        ]
        if self.cls is AIToolkitInstaller:
            stubs += [patch.object(AIToolkitInstaller, "_ensure_node"),
                      patch.object(AIToolkitInstaller, "_build_ui"),
                      patch.object(AIToolkitInstaller, "_install_requirements"),
                      patch.object(AIToolkitInstaller, "_ui_built", return_value=True),
                      patch.object(AIToolkitInstaller, "_node_works", return_value=True)]
        for p in stubs:
            stack.enter_context(p)
        return self

    def stop(self):
        self.stack.close()


@pytest.fixture(params=APPS, ids=["comfyui", "ai-toolkit"])
def box(request, tmp_persistent_dir, monkeypatch):
    cls, module = request.param
    # setup_path edits ~/.bashrc: never the real one.
    monkeypatch.setattr(recovery, "setup_path", lambda state: None)
    monkeypatch.setattr(remote_main, "setup_path", lambda state: None)
    b = Box(cls, module, tmp_persistent_dir).start()
    yield b
    b.stop()


def _remote(monkeypatch, persistent_dir, *command):
    """Run the remote tool's main() in-process; return its exit code."""
    monkeypatch.setattr(sys, "argv", ["remote", "--persistent-dir", persistent_dir, *command])
    try:
        remote_main.main()
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    return 0


def _remote_status(monkeypatch, persistent_dir, capsys):
    """The remote `status` JSON's apps, as `cloudgpu status` and `up` read it."""
    monkeypatch.setattr(remote_main, "detect_all", lambda: {})
    capsys.readouterr()
    assert _remote(monkeypatch, persistent_dir, "status") == 0
    return json.loads(capsys.readouterr().out)["apps"]


def _render(apps: dict, monkeypatch) -> str:
    """display.show_status's output, with its ANSI colours kept."""
    out = io.StringIO()
    monkeypatch.setattr(display, "console", Console(file=out, force_terminal=True,
                                                    color_system="standard", width=160))
    display.show_status(apps)
    return out.getvalue()


def _cells(row: str) -> list[str]:
    """A rendered table row's cell texts, colours stripped."""
    return [c.strip() for c in re.sub(r"\x1b\[[0-9;]*m", "", row).split("│")]


def _row(rendered: str, name: str) -> str:
    """The table row for ``name``, colours kept (matched on its text without them)."""
    [line] = [l for l in rendered.splitlines() if f" {name} " in re.sub(r"\x1b\[[0-9;]*m", "", l)]
    return line


# 1. install

def test_install_fails_when_the_service_fails_and_is_not_recorded(box, monkeypatch, capsys):
    """Catches: a service that won't install only warned about, the app recorded installed, `up` skipping it next time."""
    box.on_disk(record=False)
    box.service_error = SERVICE_ERROR
    with pytest.raises(Exception) as raised:
        box.installer.install()
    assert box.service_calls == [box.name]
    assert SERVICE_ERROR in str(raised.value)
    assert box.name in str(raised.value), "the error must name the service"
    assert "installed successfully" not in capsys.readouterr().err
    # `up` installs every profile app the remote `status` doesn't list: it must not be listed.
    assert box.name not in _remote_status(monkeypatch, box.persistent_dir, capsys)


def test_remote_install_exits_nonzero_when_the_service_fails(box, monkeypatch, capsys):
    """Catches: the remote `install` exiting 0, so `cloudgpu install`/`up` report success with no service."""
    box.on_disk(record=False)
    box.service_error = SERVICE_ERROR
    try:
        code = _remote(monkeypatch, box.persistent_dir, "install", "--app", box.name)
    except Exception:
        code = 1  # an uncaught exception: python exits 1
    assert code != 0


# 2. recover

def test_remote_recover_exits_nonzero_and_names_the_app_and_error(box, monkeypatch, capsys):
    """Catches: recovery calling a box whose service won't start recovered, exit 0, 'Recovery complete!'."""
    box.on_disk()
    box.service_error = SERVICE_ERROR
    code = _remote(monkeypatch, box.persistent_dir, "recover")
    err = capsys.readouterr().err
    assert box.service_calls == [box.name]
    assert code != 0
    assert "recovery complete" not in err.lower()
    summary = err[err.index("Recovery failed"):]
    [line] = [l for l in summary.splitlines() if f" {box.name}: " in l]
    assert SERVICE_ERROR in line
    assert "service" in line.lower()


# 3. status

@pytest.mark.parametrize("service", ["inactive", "unknown", "failed"])
def test_status_does_not_call_a_down_service_installed(box, monkeypatch, capsys, service):
    """Catches: `status` showing a green 'installed' for an app whose service isn't running (or can't be checked)."""
    box.on_disk()
    box.service_state = service
    apps = _remote_status(monkeypatch, box.persistent_dir, capsys)
    assert apps[box.name]["service"] == service
    assert apps[box.name]["status"] != "installed"
    row = _row(_render(apps, monkeypatch), box.name)
    assert service in _cells(row), "the table must show the service state in its own column"
    assert GREEN not in row, f"nothing in a {service} app's row may be green"


# 4. the healthy path is unchanged

def test_active_service_still_installs_recovers_and_shows_installed(box, monkeypatch, capsys):
    """Catches: the fix failing or marking down an app whose service did install and is running."""
    box.on_disk(record=False)
    box.installer.install()
    assert box.installer.state.get_app(box.name)["status"] == "installed"
    assert _remote(monkeypatch, box.persistent_dir, "recover") == 0
    assert "Recovery complete!" in capsys.readouterr().err
    assert box.service_calls == [box.name, box.name]
    apps = _remote_status(monkeypatch, box.persistent_dir, capsys)
    assert apps[box.name]["status"] == "installed"
    row = _row(_render(apps, monkeypatch), box.name)
    assert "installed" in row
    assert GREEN in row


def test_ai_toolkit_hf_home_still_made_before_the_unit(box):
    """Catches: making the service failure fatal dropping the HF cache dir systemd needs (ReadWritePaths)."""
    if box.cls is not AIToolkitInstaller:
        pytest.skip("only ai-toolkit's unit has a cache dir outside its app")
    box.on_disk()
    box.installer.recover()
    assert box.service_calls == [box.name]
    assert box.rw_paths_existed, "every ReadWritePaths dir must exist before the unit is installed"


# 5. through `up`

def test_up_fails_when_recover_fails_on_the_service(box, monkeypatch, capsys, tmp_path):
    """Catches: `cloudgpu up` saying the box is up after the service failed to install during recovery.

    The remote `recover` runs for real (in-process, stubbed steps); its exit code is what the
    ssh shim gives `up`, as in tests/test_up_recovery.py.
    """
    box.on_disk()
    box.service_error = SERVICE_ERROR
    code = _remote(monkeypatch, box.persistent_dir, "recover")
    box.stop()  # the CLI below is a real subprocess, against the fake Lambda and ssh shim
    recorded = {box.name: {"status": "installed", "version": "1"}}
    workdir = tmp_path / "up"
    workdir.mkdir()
    with FakeLambda() as fake:
        result = run_cloudgpu(fake, workdir, ["up", "--config", "box"],
                              config_extra="auto_terminate_hours = 8\n",
                              shim_env={"FAKE_SHIM_APPS": json.dumps(recorded),
                                        "FAKE_SHIM_RECOVER_EXIT": str(code)})
    assert "Traceback" not in result.output, result.output
    assert result.returncode != 0, result.output
    assert f"Recovery failed (exit code {code})" in result.output
    assert "is up" not in result.output

