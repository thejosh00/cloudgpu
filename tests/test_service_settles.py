"""A service that dies right after starting fails the install (plan item 0.6-fix).

0.6 made a refused start fail; but with Type=simple and Restart=always, systemd accepts the start
of a unit whose process then exits, and keeps restarting it, so install said "enabled and started".
install_service now waits a settle period and checks the unit is active/running without restarts.

No real sudo, systemctl, journalctl or sleep: systemd is faked at service.run, and any
subprocess that slips through fails the test.
"""

from __future__ import annotations

import subprocess
import time
from types import SimpleNamespace

import pytest

from cloudgpu.remote import service
from cloudgpu.remote.apps.aitoolkit import AIToolkitInstaller
from cloudgpu.remote.apps.comfyui import ComfyUIInstaller
from cloudgpu.remote.state import State

JOURNAL = "comfyui[123]: Traceback (most recent call last)\ncomfyui[123]: ImportError: no module named torch"


def _show(active, sub, restarts):
    return f"ActiveState={active}\nSubState={sub}\nNRestarts={restarts}\n"


class FakeSystemd:
    """Answers `systemctl show` with the given states, in order (the last one repeats)."""

    def __init__(self, *states):
        self.states = list(states)
        self.calls: list[list[str]] = []
        self.slept: list[float] = []

    def run(self, cmd, **kwargs):
        self.calls.append(cmd)
        if cmd[:2] == ["systemctl", "show"]:
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            return SimpleNamespace(returncode=0, stdout=_show(*state))
        if "journalctl" in cmd:
            return SimpleNamespace(returncode=0, stdout=JOURNAL + "\n")
        return SimpleNamespace(returncode=0, stdout="")


def _no_real_commands(*args, **kwargs):
    raise AssertionError(f"a real command ran in a test: {args[:1]}")


@pytest.fixture
def systemd(monkeypatch):
    def install(*states):
        fake = FakeSystemd(*states)
        monkeypatch.setattr(service, "run", fake.run)
        monkeypatch.setattr(time, "sleep", fake.slept.append)
        monkeypatch.setattr(service.pwd, "getpwnam", lambda name: object())
        monkeypatch.setattr(subprocess, "run", _no_real_commands)
        monkeypatch.setattr(subprocess, "Popen", _no_real_commands)
        return fake
    return install


@pytest.fixture
def logs(monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(service, "log", lambda msg, *a, **k: lines.append(msg))
    return lines


def _spec(tmp_path):
    return {"name": "comfyui", "exec_start": "/bin/false", "workdir": str(tmp_path), "port": 8188}


# Just after the start, then after the settle period.
DYING = {
    "auto-restart": [("active", "running", 0), ("activating", "auto-restart", 3)],
    "failed": [("active", "running", 0), ("failed", "failed", 5)],
    "restarting-but-active": [("active", "running", 0), ("active", "running", 2)],
    "inactive": [("active", "running", 0), ("inactive", "dead", 0)],
}


@pytest.mark.parametrize("states", DYING.values(), ids=DYING.keys())
def test_service_that_dies_after_start_fails_install(systemd, logs, tmp_path, states):
    fake = systemd(*states)
    with pytest.raises(RuntimeError) as err:
        service.install_service(_spec(tmp_path))
    msg = str(err.value)
    assert "'comfyui'" in msg
    assert f"{states[-1][0]}/{states[-1][1]}" in msg
    assert "ImportError: no module named torch" in msg  # the journal's last lines
    assert ["sudo", "journalctl", "-u", "comfyui", "-n", "20", "--no-pager"] in fake.calls
    assert not any("enabled and started" in line for line in logs)


def test_waits_at_least_one_restart_cycle(systemd, logs, tmp_path):
    fake = systemd(("active", "running", 0))
    service.install_service(_spec(tmp_path))
    assert fake.slept and fake.slept[0] > 3  # unit_text's RestartSec=3
    assert "RestartSec=3" in service.unit_text(_spec(tmp_path))


def test_service_that_stays_up_passes(systemd, logs, tmp_path):
    fake = systemd(("active", "running", 4))  # restarts from before this install don't count
    service.install_service(_spec(tmp_path))
    assert any("'comfyui' enabled and started" in line for line in logs)
    show = [i for i, c in enumerate(fake.calls) if c[:2] == ["systemctl", "show"]]
    restart = fake.calls.index(["sudo", "systemctl", "restart", "comfyui"])
    assert len(show) == 2 and show[0] > restart  # counted from after the restart
    assert not any("journalctl" in c for c in fake.calls)


def test_unreadable_state_fails_rather_than_passes(systemd, logs, tmp_path, monkeypatch):
    systemd(("active", "running", 0))
    monkeypatch.setattr(service, "run", lambda cmd, **k: SimpleNamespace(returncode=1, stdout=""))
    with pytest.raises(RuntimeError, match="'comfyui' didn't stay up"):
        service.install_service(_spec(tmp_path))


@pytest.mark.parametrize("cls", [ComfyUIInstaller, AIToolkitInstaller], ids=["comfyui", "ai-toolkit"])
def test_installers_setup_service_fails_on_a_dying_service(systemd, logs, tmp_persistent_dir, cls):
    systemd(*DYING["auto-restart"])
    installer = cls(State(tmp_persistent_dir))
    with pytest.raises(RuntimeError) as err:
        installer._setup_service()
    assert f"Could not set up systemd service '{installer.name}'" in str(err.value)
    assert "activating/auto-restart" in str(err.value)


@pytest.mark.parametrize("cls", [ComfyUIInstaller, AIToolkitInstaller], ids=["comfyui", "ai-toolkit"])
def test_installers_setup_service_passes_on_a_service_that_stays_up(systemd, logs, tmp_persistent_dir, cls):
    systemd(("active", "running", 0))
    installer = cls(State(tmp_persistent_dir))
    installer._setup_service()
    assert any(f"'{installer.name}' enabled and started" in line for line in logs)
