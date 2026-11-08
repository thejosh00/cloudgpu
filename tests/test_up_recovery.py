"""Every `cloudgpu up` recovers the apps the filesystem records (plan item 0.4).

No real API, instance or key: the real CLI runs in a subprocess against the fake Lambda, with
ssh shimmed. FAKE_SHIM_APPS is what the filesystem's state.json records; the shim answers the
remote `status` with it, and FAKE_SHIM_RECOVER_EXIT is the remote `recover`'s exit code.
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from cloudgpu import cli
from cloudgpu.local import configs
from tests.fake_lambda import FakeLambda, run_cloudgpu

RECORDED = {"comfyui": {"status": "installed", "version": "1", "app_dir": "/lambda/nfs/box-fs/apps/comfyui"}}
CAP = "auto_terminate_hours = 8\n"


def _remote(calls, command):
    """Indexes of the ssh calls that ran the synced tool's `command`."""
    return [i for i, c in enumerate(calls)
            if c["tool"] == "ssh" and "python3 -m remote" in c["command"] and c["command"].endswith(f" {command}")]


def _installs(calls):
    return [(i, c["command"].split("--app ")[1]) for i, c in enumerate(calls)
            if c["tool"] == "ssh" and " install --app " in c["command"]]


def _cap_armed(calls):
    [i] = [i for i, c in enumerate(calls) if c["tool"] == "ssh" and "autoterminate --hours 8" in c["command"]]
    return i


def _profile(workdir, name, apps):
    d = workdir / "profiles" / name
    d.mkdir(parents=True)
    (d / "cloudgpu.toml").write_text(f"apps = {json.dumps(apps)}\n")


def _up(tmp_path, *args, recorded=RECORDED, recover_exit=0):
    with FakeLambda() as fake:
        result = run_cloudgpu(
            fake, tmp_path, ["up", "--config", "box", *args], config_extra=CAP,
            shim_env={"FAKE_SHIM_APPS": json.dumps(recorded), "FAKE_SHIM_RECOVER_EXIT": str(recover_exit)},
        )
    assert "Traceback" not in result.output, result.output
    return result


def test_up_without_profiles_recovers_recorded_apps_once_after_the_cap(tmp_path):
    """Catches: `up -C box` after a termination bringing the box back with none of its apps."""
    result = _up(tmp_path)
    assert result.returncode == 0, result.output
    calls = result.shim_calls
    recovers = _remote(calls, "recover")
    assert len(recovers) == 1, result.output
    assert _cap_armed(calls) < recovers[0]
    assert "is up" in result.output


def test_up_with_an_appless_profile_still_recovers_once(tmp_path):
    """Catches: an empty app union returning before recovery, so `-P <no apps>` recovers nothing."""
    _profile(tmp_path, "bare", [])
    result = _up(tmp_path, "-P", "bare")
    assert result.returncode == 0, result.output
    recovers = _remote(result.shim_calls, "recover")
    assert len(recovers) == 1, result.output
    assert _cap_armed(result.shim_calls) < recovers[0]


def test_up_with_app_profile_recovers_once_then_installs_the_unrecorded(tmp_path):
    """Catches: recovery running twice on one `up`, or the profile's new apps no longer installed."""
    _profile(tmp_path, "gen", ["comfyui", "ai-toolkit"])
    result = _up(tmp_path, "-P", "gen")
    assert result.returncode == 0, result.output
    recovers = _remote(result.shim_calls, "recover")
    assert len(recovers) == 1, result.output
    installs = _installs(result.shim_calls)
    assert [app for _, app in installs] == ["ai-toolkit"], result.output  # comfyui is recorded already
    assert recovers[0] < installs[0][0]


def test_up_with_nothing_recorded_runs_no_recover(tmp_path):
    """Catches: a bare filesystem getting a needless `recover` call."""
    result = _up(tmp_path, recorded={})
    assert result.returncode == 0, result.output
    assert _remote(result.shim_calls, "recover") == []
    assert "is up" in result.output


def test_up_fails_when_recover_fails(tmp_path):
    """Catches: a failed recovery reported as a box that is up."""
    result = _up(tmp_path, recover_exit=3)
    assert result.returncode != 0, result.output
    assert "Recovery failed (exit code 3)" in result.output
    assert "is up" not in result.output


def test_provision_still_recovers_before_installing(tmp_path, tmp_config_dir, monkeypatch):
    """Catches: moving recovery into `up` and leaving `cloudgpu provision` without it."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "c.toml").write_text('gpu = ["gh200"]\nssh_key = "mini"\n')
    _profile(tmp_path, "gen", ["comfyui", "ai-toolkit"])
    monkeypatch.chdir(tmp_path)
    configs.save_state("c", {"host": "ubuntu@h", "persistent_dir": "/lambda/nfs/p"})

    commands = []

    def ssh_run(host, command, **kwargs):
        commands.append(command)
        out = json.dumps({"apps": RECORDED}) if command.endswith(" status") else ""
        return cli.ssh.SSHResult(0, out, "")

    monkeypatch.setattr(cli.sync, "sync_remote", lambda *a, **k: None)
    monkeypatch.setattr(cli.ssh, "ssh_run", ssh_run)
    monkeypatch.setattr(cli, "_run_provision", lambda *a, **k: True)
    monkeypatch.setattr(cli, "_restart_services", lambda *a, **k: None)

    r = CliRunner().invoke(cli.provision, ["-C", "c", "-P", "gen"])
    assert r.exit_code == 0, r.output
    recovers = [i for i, c in enumerate(commands) if c.endswith(" recover")]
    installs = [i for i, c in enumerate(commands) if " install --app " in c]
    assert len(recovers) == 1
    assert [commands[i].split("--app ")[1] for i in installs] == ["ai-toolkit"]
    assert recovers[0] < installs[0]
