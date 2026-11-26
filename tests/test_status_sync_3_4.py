"""`cloudgpu status` shows the last sync and warns when it is stale (3.4 part 3).

Nothing here runs ssh, sudo, systemctl or rclone. The client's ssh is monkeypatched to run the
remote tool's `status` in process; there, sudo is a recorder over temp stand-ins for the
instance's root-owned places (the 3.2 setup tests' Box), answering `systemctl is-active` for the
timer and `cat` of the last-success file.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import time

import pytest
from click.testing import CliRunner

from cloudgpu import cli
from cloudgpu.local import configs, ssh, sync
from cloudgpu.remote import __main__ as remote_main
from cloudgpu.remote import datasync_setup as ds
from cloudgpu.remote.state import State
from tests.test_datasync_setup_3_2 import Box

NOW = 1_795_000_000.0  # a fixed "now" on the instance
TIMER = f"{ds.UNIT_NAME}.timer"
DETECTION = {"persistent_dir": "/lambda/nfs/prof", "gpu": {"available": False}, "cuda": {},
             "python_torch": {}}
BASE = 'gpu = ["gh200"]\nssh_key = "mini"\nfilesystem = "prof"\n'
SYNC = '\n[sync]\nremote = "b2:bkt/small"\n'
STATE = {"instance_id": "i-1", "ip": "1.2.3.4", "host": "ubuntu@1.2.3.4",
         "filesystem": "prof", "persistent_dir": "/lambda/nfs/prof"}
CHANGING = {"start", "stop", "restart", "try-restart", "reload-or-restart", "enable", "disable",
            "kill", "reset-failed", "daemon-reload"}


def stamp(when: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(when))


class StatusBox(Box):
    """An instance with (or without) the sync installed, a timer that is or isn't active, and a
    last-success file that may be there either way (``remove`` leaves it)."""

    def __init__(self, tmp_path, monkeypatch, *, installed=True, interval=5, timer_active=True,
                 last: str | None = None):
        super().__init__(tmp_path, monkeypatch)
        self.timer_active = timer_active
        self.last = last
        os.makedirs(ds.ETC_DIR)
        if installed:
            with open(ds.SPEC_PATH, "w") as f:
                json.dump({"remote": "b2:bkt/small", "persistent_dir": "/lambda/nfs/prof",
                           "paths": ["apps/comfyui/output"], "interval_minutes": interval}, f)
            service, timer = ds.unit_texts(interval)
            for path, text in ((ds.service_path(), service), (ds.timer_path(), timer)):
                with open(path, "w") as f:
                    f.write(text)
        monkeypatch.setattr(ds.time, "time", lambda: NOW)

    def run(self, cmd, **kwargs):
        assert cmd[0] == "sudo", f"only sudo may be run, got {cmd}"
        rest = list(cmd[1:])
        self.calls.append(rest)
        if rest[0] == "cat":
            assert rest[1:] == [ds.LAST_SUCCESS_FILE], rest
            if self.last is None:
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="No such file")
            return subprocess.CompletedProcess(cmd, 0, stdout=self.last + "\n", stderr="")
        if rest[:2] == ["systemctl", "is-active"]:
            assert rest[-1] == TIMER, rest
            return subprocess.CompletedProcess(cmd, 0 if self.timer_active else 3)
        raise AssertionError(f"status may only read, got {rest}")

    def changing(self):
        return [c for c in self.calls if c[0] == "systemctl" and c[1] in CHANGING]


def remote_status(persistent_dir) -> dict:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        remote_main.cmd_status(State(str(persistent_dir)), argparse.Namespace())
    return json.loads(out.getvalue())


@pytest.fixture(autouse=True)
def _no_detection(monkeypatch):
    monkeypatch.setattr(remote_main, "detect_all", lambda: dict(DETECTION))


def _client(tmp_path, monkeypatch, *, with_sync=True, canned: str | None = None):
    """A config with a running instance; ssh runs the remote `status` in process (or answers
    ``canned``, today's remote output). Returns the recorded ssh commands."""
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True, exist_ok=True)
    (repo / "configs" / "c.toml").write_text(BASE + (SYNC if with_sync else ""))
    monkeypatch.chdir(repo)
    configs.save_state("c", dict(STATE))
    fs = tmp_path / "fs"
    fs.mkdir(exist_ok=True)
    commands: list[str] = []

    def ssh_run(host, command, **kw):
        commands.append(command)
        assert command == cli._remote_cmd("/lambda/nfs/prof", "status"), command
        if canned is not None:
            return ssh.SSHResult(0, canned, "")
        return ssh.SSHResult(0, json.dumps(remote_status(fs)), "")

    monkeypatch.setattr(ssh, "ssh_run", ssh_run)
    monkeypatch.setattr(sync, "sync_remote", lambda host, pdir: None)
    return commands


def _status(tmp_path, monkeypatch, **kw):
    commands = _client(tmp_path, monkeypatch, **kw)
    result = CliRunner().invoke(cli.status, ["-C", "c"])
    return result, " ".join(result.output.split()), commands


# --- the instance side -------------------------------------------------------------------


def test_installed_sync_reports_remote_interval_timer_and_last_success(tmp_path, monkeypatch):
    """Catches: the remote status saying nothing about the sync, the age taken from the wrong
    clock, the interval not read from what is installed, or the last-success file read without
    sudo (it is in the sync user's mode-750 directory)."""
    box = StatusBox(tmp_path, monkeypatch, interval=7, last=stamp(NOW - 125))
    data = remote_status(tmp_path)
    assert data["sync"] == {"remote": "b2:bkt/small", "interval_minutes": 7, "timer_active": True,
                            "last_success": stamp(NOW - 125), "age_seconds": 125}
    assert ["cat", ds.LAST_SUCCESS_FILE] in box.calls
    assert box.changing() == []


def test_the_interval_is_the_installed_timers(tmp_path, monkeypatch):
    """Catches: the interval taken from the spec when the timer that actually runs says
    otherwise (the timer is what decides how often uploads happen)."""
    StatusBox(tmp_path, monkeypatch, interval=7, last=stamp(NOW))
    with open(ds.timer_path(), "w") as f:
        f.write(ds.unit_texts(20)[1])
    assert remote_status(tmp_path)["sync"]["interval_minutes"] == 20


def test_never_succeeded_and_inactive_timer_are_reported(tmp_path, monkeypatch):
    """Catches: a missing last-success file reported as some time, or an inactive timer as
    active."""
    box = StatusBox(tmp_path, monkeypatch, timer_active=False, last=None)
    sync_info = remote_status(tmp_path)["sync"]
    assert sync_info["last_success"] is None and sync_info["age_seconds"] is None
    assert sync_info["timer_active"] is False
    assert box.changing() == []


@pytest.mark.parametrize("missing", ["spec", "timer"])
def test_not_installed_reports_nothing_and_ignores_a_leftover_last_success(tmp_path, monkeypatch, missing):
    """Catches: a last-success file left behind by `datasync remove` taken as a working sync; any
    sudo call when sync isn't installed; the remote status changing shape for an instance
    without sync."""
    box = StatusBox(tmp_path, monkeypatch, last=stamp(NOW - 60))
    os.remove(ds.SPEC_PATH if missing == "spec" else ds.timer_path())
    data = remote_status(tmp_path)
    assert set(data) == {"detection", "apps"}
    assert box.calls == []


# --- the client ----------------------------------------------------------------------------


def test_status_shows_remote_and_last_success_without_warning_when_recent(tmp_path, monkeypatch):
    """Catches: `status` not showing the remote, the last sync time or how long ago; a warning
    on a healthy sync."""
    box = StatusBox(tmp_path, monkeypatch, interval=5, last=stamp(NOW - 4 * 60))
    result, out, commands = _status(tmp_path, monkeypatch)
    assert result.exit_code == 0, result.output
    assert "b2:bkt/small" in out
    assert stamp(NOW - 4 * 60) in out and "4m ago" in out
    assert "timer active" in out
    assert "only on the instance" not in out and "no successful sync yet" not in out
    assert box.changing() == []
    assert len(commands) == 1


@pytest.mark.parametrize("interval,age,stale", [
    (5, 14 * 60, False),       # under the 15-minute floor
    (5, 16 * 60, True),        # 3 x 5 min is 15: the floor decides
    (10, 29 * 60, False),      # 3 x 10 min = 30 min is longer than 15
    (10, 31 * 60, True),
    (60, 3 * 3600 + 120, True),
])
def test_an_old_last_success_warns_with_its_age(tmp_path, monkeypatch, interval, age, stale):
    """Catches: no warning when the last success is older than three intervals or 15 minutes
    (whichever is longer), or a warning before that."""
    StatusBox(tmp_path, monkeypatch, interval=interval, last=stamp(NOW - age))
    result, out, _ = _status(tmp_path, monkeypatch)
    assert result.exit_code == 0, result.output
    warned = "newer work is only on the instance" in out
    assert warned == stale, out
    if stale:
        assert f"{cli._overdue_for(age)} ago" in out


def test_an_inactive_timer_warns_even_with_a_recent_success(tmp_path, monkeypatch):
    """Catches: a stopped timer (no more uploads will happen) passing as healthy because the
    last run was recent."""
    StatusBox(tmp_path, monkeypatch, timer_active=False, last=stamp(NOW - 60))
    result, out, _ = _status(tmp_path, monkeypatch)
    assert result.exit_code == 0, result.output
    assert "timer is not active" in out
    assert "1m ago" in out and "newer work is only on the instance" in out


def test_never_succeeded_is_its_own_warning(tmp_path, monkeypatch):
    """Catches: "never synced" shown as no information, or as a stale time."""
    StatusBox(tmp_path, monkeypatch, last=None)
    result, out, _ = _status(tmp_path, monkeypatch)
    assert result.exit_code == 0, result.output
    assert "no successful sync yet" in out.lower()
    assert "only on the instance" in out


@pytest.mark.parametrize("with_sync", [True, False], ids=["config-with-sync", "config-without-sync"])
def test_not_installed_prints_nothing_about_sync_and_is_as_today(tmp_path, monkeypatch, with_sync):
    """Catches: a leftover last-success file shown as a working sync once sync is removed; any
    change to `status` output for an instance without sync."""
    box = StatusBox(tmp_path, monkeypatch, installed=False, last=stamp(NOW - 60))
    result, out, _ = _status(tmp_path, monkeypatch, with_sync=with_sync)
    assert result.exit_code == 0, result.output
    assert "sync" not in out.lower(), out
    assert box.calls == []
    # Today's remote answer, as a6e18e4's `status` prints it: the output is the same.
    today = json.dumps({"detection": DETECTION, "apps": {}})
    _client(tmp_path, monkeypatch, with_sync=with_sync, canned=today)
    old = CliRunner().invoke(cli.status, ["-C", "c"])
    assert old.exit_code == 0 and result.output == old.output


def test_status_runs_nothing_that_starts_stops_or_restarts(tmp_path, monkeypatch):
    """Catches: `status` changing what runs on the instance (for example starting a stopped
    timer it reports on)."""
    for active, last in ((True, stamp(NOW)), (False, None)):
        (tmp_path / str(active)).mkdir()
        box = StatusBox(tmp_path / str(active), monkeypatch, timer_active=active, last=last)
        _status(tmp_path / str(active), monkeypatch)
        assert box.calls, "the instance was asked"
        assert all(c[0] == "cat" or c[:2] == ["systemctl", "is-active"] for c in box.calls), box.calls
        assert box.changing() == []
