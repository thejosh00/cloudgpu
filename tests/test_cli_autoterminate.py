"""Tests for the local auto-terminate wiring in `up` and the `autoterminate`
command (mocked SSH/rsync)."""

from __future__ import annotations

import os
import time

import pytest
from click.testing import CliRunner

from cloudgpu import cli
from cloudgpu.local import config, profiles, ssh


def _profile(tmp_path, hours):
    d = tmp_path / "prof"
    d.mkdir(exist_ok=True)
    line = f"auto_terminate_hours = {hours}\n" if hours is not None else ""
    (d / "cloudgpu.toml").write_text(f'gpu = ["gh200"]\nssh_key = "mini"\n{line}')
    return profiles.load_profile(d)


def test_arm_transfers_key_as_content_and_runs_remote(tmp_path, monkeypatch):
    profile = _profile(tmp_path, 8)
    monkeypatch.setenv("LAMBDA_API_KEY", "secret_abc")

    copied = {}
    def fake_copy_file(local, host, remote):
        copied.update(local=local, remote=remote, content=open(local).read())
    monkeypatch.setattr(cli.sync, "copy_file", fake_copy_file)
    captured = {}
    monkeypatch.setattr(cli.ssh, "ssh_run",
                        lambda host, command, **k: captured.update(command=command) or ssh.SSHResult(0, "", ""))

    before = time.time()
    deadline = cli._configure_auto_terminate(profile, "ubuntu@h", "i-123", "/lambda/nfs/p")

    assert deadline is not None
    assert before + 8 * 3600 - 5 <= deadline <= time.time() + 8 * 3600 + 5
    # Key goes over as file content, never on the command line.
    assert copied["content"] == "LAMBDA_API_KEY=secret_abc\n"
    assert copied["remote"] == ".cloudgpu-lambda.env"
    assert not os.path.exists(copied["local"])  # local temp copy cleaned up
    assert "autoterminate" in captured["command"]
    assert "--hours 8" in captured["command"]
    assert "--instance-id i-123" in captured["command"]
    assert "--key-file .cloudgpu-lambda.env" in captured["command"]
    assert "secret_abc" not in captured["command"]


def test_hours_zero_disarms_without_key_transfer(tmp_path, monkeypatch):
    profile = _profile(tmp_path, 0)
    monkeypatch.setattr(cli.sync, "copy_file", lambda *a, **k: pytest.fail("no key to copy"))
    captured = {}
    monkeypatch.setattr(cli.ssh, "ssh_run",
                        lambda host, command, **k: captured.update(command=command) or ssh.SSHResult(0, "", ""))
    assert cli._configure_auto_terminate(profile, "ubuntu@h", "i-123", "/lambda/nfs/p") is None
    assert "autoterminate --hours 0" in captured["command"]


def test_unknown_instance_id_skips_with_warning(tmp_path, monkeypatch):
    profile = _profile(tmp_path, 8)
    monkeypatch.setenv("LAMBDA_API_KEY", "secret_abc")
    monkeypatch.setattr(cli.sync, "copy_file", lambda *a, **k: pytest.fail("must not copy"))
    monkeypatch.setattr(cli.ssh, "ssh_run", lambda *a, **k: pytest.fail("must not run"))
    assert cli._configure_auto_terminate(profile, "ubuntu@h", None, "/lambda/nfs/p") is None


def test_missing_api_key_skips_with_warning(tmp_path, monkeypatch):
    profile = _profile(tmp_path, 8)
    monkeypatch.delenv("LAMBDA_API_KEY", raising=False)
    monkeypatch.setattr(cli.sync, "copy_file", lambda *a, **k: pytest.fail("must not copy"))
    monkeypatch.setattr(cli.ssh, "ssh_run", lambda *a, **k: pytest.fail("must not run"))
    assert cli._configure_auto_terminate(profile, "ubuntu@h", "i-123", "/lambda/nfs/p") is None


def test_remote_failure_fails_up(tmp_path, monkeypatch):
    profile = _profile(tmp_path, 8)
    monkeypatch.setenv("LAMBDA_API_KEY", "secret_abc")
    monkeypatch.setattr(cli.sync, "copy_file", lambda *a, **k: None)
    def boom(host, command, **k):
        raise ssh.SSHError("exit 1")
    monkeypatch.setattr(cli.ssh, "ssh_run", boom)
    with pytest.raises(SystemExit):
        cli._configure_auto_terminate(profile, "ubuntu@h", "i-123", "/lambda/nfs/p")


def test_profile_default_is_off_and_validated(tmp_path):
    assert _profile(tmp_path, None)["auto_terminate_hours"] == 0
    assert _profile(tmp_path, 8)["auto_terminate_hours"] == 8.0
    with pytest.raises(config.ConfigError):
        _profile(tmp_path, '"soon"')


# --- the `cloudgpu autoterminate` command --------------------------------------


def _profile_dir(tmp_path, hours=8, *, with_state=True):
    d = tmp_path / "prof"
    d.mkdir(exist_ok=True)
    line = f"auto_terminate_hours = {hours}\n" if hours is not None else ""
    (d / "cloudgpu.toml").write_text(f'gpu = ["gh200"]\nssh_key = "mini"\n{line}')
    if with_state:
        profiles.save_state(d, {
            "instance_id": "i-1", "ip": "1.2.3.4", "host": "ubuntu@1.2.3.4",
            "filesystem": "prof", "persistent_dir": "/lambda/nfs/prof",
        })
    return d


def _mock_remote(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli.sync, "copy_file", lambda *a, **k: None)
    monkeypatch.setattr(cli.ssh, "ssh_run",
                        lambda host, command, **k: captured.update(command=command) or ssh.SSHResult(0, "", ""))
    return captured


def test_command_rearms_with_profile_hours(tmp_path, monkeypatch):
    d = _profile_dir(tmp_path, 8)
    monkeypatch.setenv("LAMBDA_API_KEY", "secret_abc")
    captured = _mock_remote(monkeypatch)

    result = CliRunner().invoke(cli.autoterminate, ["--profile", str(d)])
    assert result.exit_code == 0, result.output
    assert "--hours 8" in captured["command"]
    state = profiles.load_state(d)
    assert state["auto_terminate_at"] == pytest.approx(time.time() + 8 * 3600, abs=10)


def test_command_hours_argument_overrides_profile(tmp_path, monkeypatch):
    d = _profile_dir(tmp_path, 8)
    monkeypatch.setenv("LAMBDA_API_KEY", "secret_abc")
    captured = _mock_remote(monkeypatch)

    result = CliRunner().invoke(cli.autoterminate, ["6", "--profile", str(d)])
    assert result.exit_code == 0, result.output
    assert "--hours 6" in captured["command"]


def test_command_off_disarms_and_clears_state(tmp_path, monkeypatch):
    d = _profile_dir(tmp_path, 8)
    state = profiles.load_state(d)
    state["auto_terminate_at"] = 123
    profiles.save_state(d, state)
    captured = _mock_remote(monkeypatch)

    result = CliRunner().invoke(cli.autoterminate, ["--off", "--profile", str(d)])
    assert result.exit_code == 0, result.output
    assert "--hours 0" in captured["command"]
    assert "auto_terminate_at" not in profiles.load_state(d)


def test_command_rejects_hours_and_off_together(tmp_path):
    d = _profile_dir(tmp_path, 8)
    result = CliRunner().invoke(cli.autoterminate, ["6", "--off", "--profile", str(d)])
    assert result.exit_code != 0
    assert "not both" in result.output


def test_command_requires_hours_when_profile_has_none(tmp_path):
    d = _profile_dir(tmp_path, None)
    result = CliRunner().invoke(cli.autoterminate, ["--profile", str(d)])
    assert result.exit_code != 0
    assert "pass HOURS" in result.output


def test_command_requires_tracked_instance(tmp_path):
    d = _profile_dir(tmp_path, 8, with_state=False)
    result = CliRunner().invoke(cli.autoterminate, ["--profile", str(d)])
    assert result.exit_code != 0
    assert "cloudgpu up" in result.output


def test_command_fails_loudly_when_arming_skipped(tmp_path, monkeypatch):
    # hours > 0 but no API key: the helper warns and skips; the command must
    # exit nonzero rather than report a cap that isn't armed.
    d = _profile_dir(tmp_path, 8)
    monkeypatch.delenv("LAMBDA_API_KEY", raising=False)
    result = CliRunner().invoke(cli.autoterminate, ["--profile", str(d)])
    assert result.exit_code != 0
