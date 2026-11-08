"""The real installers refuse to change a host that isn't a cloudgpu instance (item 0.11).

An instance is recognised by a root-owned marker file the local setup writes over SSH
(``cloudgpu.remote.instance.REAL_MARKER``). Tests never touch that path: the autouse
fixture in conftest points the guard at a temp marker, and the tests here point it at a
temp path that doesn't exist. Every subprocess is recorded and refused, so nothing runs.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from cloudgpu import cli
from cloudgpu.local import config, ssh, sync
try:
    from cloudgpu.remote import instance
except ImportError:  # base code, before the guard existed: let each test fail on its own
    import types
    instance = types.SimpleNamespace(MARKER="", REAL_MARKER="/var/lib/cloudgpu/instance",
                                     NotOnInstance=type("NotOnInstance", (Exception,), {}))
from cloudgpu.remote import __main__ as remote_main
from cloudgpu.remote.apps.aitoolkit import AIToolkitInstaller
from cloudgpu.remote.apps.comfyui import ComfyUIInstaller
from cloudgpu.remote.recovery import recover_all
from cloudgpu.remote.state import State

INSTALLERS = [ComfyUIInstaller, AIToolkitInstaller]
REPO = Path(__file__).resolve().parents[1]


class Ran(Exception):
    """A command reached the (fake) process layer."""


@pytest.fixture
def commands(monkeypatch):
    """Record every subprocess the code tries to start, and stop it there."""
    seen: list = []

    def refuse(cmd, *a, **k):
        seen.append(cmd)
        raise Ran(cmd)

    for name in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, refuse)
    monkeypatch.setattr(os, "system", refuse)
    return seen


@pytest.fixture
def no_marker(monkeypatch, tmp_path):
    missing = tmp_path / "no-such-dir" / "instance"
    monkeypatch.setattr(instance, "MARKER", str(missing))
    return str(missing)


@pytest.fixture
def state(tmp_persistent_dir):
    return State(tmp_persistent_dir)


# --- the guard on each installer -------------------------------------------------------------

@pytest.mark.parametrize("cls", INSTALLERS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("method", ["install", "recover"])
def test_refuses_without_marker_and_runs_nothing(cls, method, state, commands, no_marker):
    installer = cls(state)
    with pytest.raises(instance.NotOnInstance) as e:
        getattr(installer, method)()
    assert no_marker in str(e.value)
    assert "cloudgpu setup" in str(e.value)
    assert commands == []
    assert not os.path.exists(installer.app_dir)
    assert not os.path.exists(state.apps_dir)


@pytest.mark.parametrize("cls", INSTALLERS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("method", ["install", "recover"])
def test_proceeds_to_first_command_with_marker(cls, method, state, commands, monkeypatch, tmp_path):
    marker = tmp_path / "instance"
    marker.write_text("")
    monkeypatch.setattr(instance, "MARKER", str(marker))
    with pytest.raises(Ran):
        getattr(cls(state), method)()
    assert commands, "the installer should have reached its first command"
    assert not isinstance(commands[0], Exception)


def test_guard_reads_no_environment(no_marker, monkeypatch):
    """No environment variable, set by accident or not, can switch the guard off."""
    source = inspect.getsource(instance)
    assert "os.environ" not in source and "getenv" not in source
    for var in ("CLOUDGPU_INSTANCE", "CLOUDGPU_SKIP_GUARD", "CLOUDGPU_INSTANCE_MARKER", "CI"):
        monkeypatch.setenv(var, "1")
    with pytest.raises(instance.NotOnInstance):
        instance.require_instance()


def test_real_marker_is_root_owned_system_path():
    assert instance.REAL_MARKER == "/var/lib/cloudgpu/instance"
    assert not instance.REAL_MARKER.startswith(os.path.expanduser("~"))


# --- the remote entry points -----------------------------------------------------------------

def test_recover_all_refuses_before_anything(state, commands, no_marker, monkeypatch, tmp_path):
    state.set_app("comfyui", {"status": "installed"})
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    with pytest.raises(instance.NotOnInstance):
        recover_all(state, remote_main.APP_INSTALLERS)
    assert commands == []
    assert not (home / ".bashrc").exists()  # setup_path didn't run either


@pytest.mark.parametrize("argv", [["install", "--app", "comfyui"], ["install", "--app", "ai-toolkit"],
                                  ["recover"]])
def test_remote_main_exits_nonzero_naming_marker(argv, tmp_persistent_dir, commands, no_marker,
                                                 monkeypatch, capsys, tmp_path):
    State(tmp_persistent_dir).set_app("comfyui", {"status": "installed"})
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["remote", "--persistent-dir", tmp_persistent_dir, *argv])
    with pytest.raises(SystemExit) as e:
        remote_main.main()
    assert e.value.code not in (0, None)
    assert no_marker in capsys.readouterr().err
    assert commands == []


def test_remote_module_as_process_refuses(tmp_path):
    """`python -m cloudgpu.remote ... install` as a real process, guard pointed at a missing temp
    marker and every subprocess refused, so nothing on this machine can be changed."""
    missing = tmp_path / "absent" / "instance"
    pd = tmp_path / "nfs" / "fs"
    pd.mkdir(parents=True)
    wrapper = textwrap.dedent(f"""
        import os, runpy, subprocess, sys
        def refuse(*a, **k):
            print("COMMAND RAN", a, file=sys.stderr); sys.exit(99)
        for n in ("run", "Popen", "call", "check_call", "check_output"):
            setattr(subprocess, n, refuse)
        os.system = refuse
        import cloudgpu.remote.instance as inst
        inst.MARKER = {str(missing)!r}
        sys.argv = ["cloudgpu.remote", "--persistent-dir", {str(pd)!r}, "install", "--app", "comfyui"]
        runpy.run_module("cloudgpu.remote", run_name="__main__")
    """)
    env = {**os.environ, "HOME": str(tmp_path)}
    proc = subprocess.run([sys.executable, "-c", wrapper], capture_output=True, text=True, env=env,
                          cwd=str(tmp_path), timeout=60)
    assert proc.returncode == 2, proc.stderr
    assert str(missing) in proc.stderr
    assert "COMMAND RAN" not in proc.stderr


# --- the local side writes the marker before any installer runs ------------------------------

@pytest.fixture
def wire(monkeypatch, tmp_config_dir):
    """Fake the SSH layer and rsync; record everything sent, in order."""
    events: list[str] = []

    def fake_subprocess_run(cmd, *a, **k):
        events.append(" ".join(cmd) if isinstance(cmd, list) else cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(sync.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(cli.ssh, "ssh_test", lambda host: True)

    def fake_ssh_run(host, command, **kwargs):
        events.append(command)
        if "python3 -c" in command:
            return ssh.SSHResult(0, json.dumps({"mounts": ["fs"]}), "")
        return ssh.SSHResult(0, json.dumps({}), "")

    monkeypatch.setattr(cli.ssh, "ssh_run", fake_ssh_run)
    config.save_host("ubuntu@1.2.3.4", "/lambda/nfs/fs")
    return events


def _marker_index(events):
    hits = [i for i, e in enumerate(events) if instance.REAL_MARKER in e and "sudo -n" in e]
    assert hits, f"no marker-writing command was sent: {events}"
    return hits[0]


def _remote_index(events, sub):
    hits = [i for i, e in enumerate(events) if "-m remote" in e and sub in e]
    assert hits, f"remote {sub} never ran: {events}"
    return hits[0]


def test_sync_remote_writes_marker():
    events = []
    orig = sync.subprocess.run
    try:
        sync.subprocess.run = lambda cmd, *a, **k: events.append(" ".join(cmd))
        sync.sync_remote("ubuntu@1.2.3.4", "/lambda/nfs/fs")
    finally:
        sync.subprocess.run = orig
    assert _marker_index(events) == 0  # before rsync, and so before any remote tool


def test_setup_host_writes_marker_before_remote_tool(wire):
    cli._setup_host("ubuntu@1.2.3.4", "fs")
    assert _marker_index(wire) < _remote_index(wire, "detect")


@pytest.mark.parametrize("args,sub", [(["install", "ubuntu@1.2.3.4", "--app", "comfyui"], "install"),
                                      (["recover", "ubuntu@1.2.3.4"], "recover")])
def test_cli_writes_marker_before_installer(wire, args, sub):
    result = CliRunner().invoke(cli.cli, args)
    assert result.exit_code == 0, result.output
    assert _marker_index(wire) < _remote_index(wire, sub)
