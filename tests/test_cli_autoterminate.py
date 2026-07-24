"""Tests for the auto-terminate wiring in `_configure_auto_terminate` and the
`autoterminate` command (mocked SSH/rsync)."""

from __future__ import annotations

import os
import time

import pytest
from click.testing import CliRunner

from cloudgpu import cli
from cloudgpu.local import configs, ssh


# --- the _configure_auto_terminate helper (dict with auto_terminate_hours) ------

def test_arm_transfers_key_as_content_and_runs_remote(monkeypatch):
    monkeypatch.setenv("LAMBDA_API_KEY", "secret_abc")
    copied = {}
    def fake_copy_file(local, host, remote):
        copied.update(local=local, remote=remote, content=open(local).read())
    monkeypatch.setattr(cli.sync, "copy_file", fake_copy_file)
    captured = {}
    monkeypatch.setattr(cli.ssh, "ssh_run",
                        lambda host, command, **k: captured.update(command=command) or ssh.SSHResult(0, "", ""))

    before = time.time()
    deadline = cli._configure_auto_terminate({"auto_terminate_hours": 8}, "ubuntu@h", "i-123", "/lambda/nfs/p")

    assert deadline is not None
    assert before + 8 * 3600 - 5 <= deadline <= time.time() + 8 * 3600 + 5
    assert copied["content"] == "LAMBDA_API_KEY=secret_abc\n"
    assert copied["remote"] == ".cloudgpu-lambda.env"
    assert not os.path.exists(copied["local"])  # local temp copy cleaned up
    assert "autoterminate" in captured["command"]
    assert "--hours 8" in captured["command"]
    assert "--instance-id i-123" in captured["command"]
    assert "secret_abc" not in captured["command"]


def test_hours_zero_disarms_without_key_transfer(monkeypatch):
    monkeypatch.setattr(cli.sync, "copy_file", lambda *a, **k: pytest.fail("no key to copy"))
    captured = {}
    monkeypatch.setattr(cli.ssh, "ssh_run",
                        lambda host, command, **k: captured.update(command=command) or ssh.SSHResult(0, "", ""))
    assert cli._configure_auto_terminate({"auto_terminate_hours": 0}, "ubuntu@h", "i-123", "/lambda/nfs/p") is None
    assert "autoterminate --hours 0" in captured["command"]


def test_unknown_instance_id_skips_with_warning(monkeypatch):
    monkeypatch.setenv("LAMBDA_API_KEY", "secret_abc")
    monkeypatch.setattr(cli.sync, "copy_file", lambda *a, **k: pytest.fail("must not copy"))
    monkeypatch.setattr(cli.ssh, "ssh_run", lambda *a, **k: pytest.fail("must not run"))
    assert cli._configure_auto_terminate({"auto_terminate_hours": 8}, "ubuntu@h", None, "/lambda/nfs/p") is None


def test_missing_api_key_skips_with_warning(monkeypatch):
    monkeypatch.delenv("LAMBDA_API_KEY", raising=False)
    monkeypatch.setattr(cli.sync, "copy_file", lambda *a, **k: pytest.fail("must not copy"))
    monkeypatch.setattr(cli.ssh, "ssh_run", lambda *a, **k: pytest.fail("must not run"))
    assert cli._configure_auto_terminate({"auto_terminate_hours": 8}, "ubuntu@h", "i-123", "/lambda/nfs/p") is None


def test_remote_failure_fails_up(monkeypatch):
    monkeypatch.setenv("LAMBDA_API_KEY", "secret_abc")
    monkeypatch.setattr(cli.sync, "copy_file", lambda *a, **k: None)
    def boom(host, command, **k):
        raise ssh.SSHError("exit 1")
    monkeypatch.setattr(cli.ssh, "ssh_run", boom)
    with pytest.raises(SystemExit):
        cli._configure_auto_terminate({"auto_terminate_hours": 8}, "ubuntu@h", "i-123", "/lambda/nfs/p")


# --- the `cloudgpu autoterminate` command --------------------------------------

def _repo(tmp_path, monkeypatch, hours=8, *, with_state=True):
    (tmp_path / "configs").mkdir(exist_ok=True)
    line = f"auto_terminate_hours = {hours}\n" if hours is not None else ""
    (tmp_path / "configs" / "c.toml").write_text(f'gpu = ["gh200"]\nssh_key = "mini"\n{line}')
    monkeypatch.chdir(tmp_path)
    if with_state:
        configs.save_state("c", {
            "instance_id": "i-1", "host": "ubuntu@1.2.3.4", "persistent_dir": "/lambda/nfs/p",
        })


def _mock_remote(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli.sync, "copy_file", lambda *a, **k: None)
    monkeypatch.setattr(cli.ssh, "ssh_run",
                        lambda host, command, **k: captured.update(command=command) or ssh.SSHResult(0, "", ""))
    return captured


def test_command_rearms_with_config_hours(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, 8)
    monkeypatch.setenv("LAMBDA_API_KEY", "secret_abc")
    captured = _mock_remote(monkeypatch)

    result = CliRunner().invoke(cli.autoterminate, ["-C", "c"])
    assert result.exit_code == 0, result.output
    assert "--hours 8" in captured["command"]
    state = configs.load_state("c")
    assert state["auto_terminate_at"] == pytest.approx(time.time() + 8 * 3600, abs=10)


def test_command_hours_argument_overrides_config(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, 8)
    monkeypatch.setenv("LAMBDA_API_KEY", "secret_abc")
    captured = _mock_remote(monkeypatch)

    result = CliRunner().invoke(cli.autoterminate, ["6", "-C", "c"])
    assert result.exit_code == 0, result.output
    assert "--hours 6" in captured["command"]


def test_command_off_disarms_and_clears_state(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, 8)
    state = configs.load_state("c")
    state["auto_terminate_at"] = 123
    configs.save_state("c", state)
    captured = _mock_remote(monkeypatch)

    result = CliRunner().invoke(cli.autoterminate, ["--off", "-C", "c"])
    assert result.exit_code == 0, result.output
    assert "--hours 0" in captured["command"]
    assert "auto_terminate_at" not in configs.load_state("c")


def test_command_rejects_hours_and_off_together(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, 8)
    result = CliRunner().invoke(cli.autoterminate, ["6", "--off", "-C", "c"])
    assert result.exit_code != 0
    assert "not both" in result.output


def test_command_requires_hours_when_config_has_none(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, None)
    result = CliRunner().invoke(cli.autoterminate, ["-C", "c"])
    assert result.exit_code != 0
    assert "pass HOURS" in result.output


def test_command_requires_tracked_instance(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, 8, with_state=False)
    result = CliRunner().invoke(cli.autoterminate, ["-C", "c"])
    assert result.exit_code != 0
    assert "cloudgpu up" in result.output


def test_command_fails_loudly_when_arming_skipped(tmp_path, monkeypatch):
    # hours > 0 but no API key: the helper warns and skips; the command must exit nonzero.
    _repo(tmp_path, monkeypatch, 8)
    monkeypatch.delenv("LAMBDA_API_KEY", raising=False)
    result = CliRunner().invoke(cli.autoterminate, ["-C", "c"])
    assert result.exit_code != 0
