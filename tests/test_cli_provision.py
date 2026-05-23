"""Tests for the per-profile provisioning directory hook (mocked SSH/rsync)."""

from __future__ import annotations

import pytest

from cloudgpu import cli
from cloudgpu.local import profiles, ssh


def _make_profile(name="p"):
    profiles.create_profile(name, gpu=["gh200"], apps=[], ssh_key="mini")
    return profiles.load_profile(name)


def _write_provision(name="p", script="#!/bin/bash\necho hi\n"):
    d = profiles.provision_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "provision.sh").write_text(script)
    return d


def test_no_provision_dir_is_noop(tmp_config_dir, monkeypatch):
    profile = _make_profile()
    monkeypatch.setattr(cli.sync, "copy_dir", lambda *a, **k: pytest.fail("must not copy"))
    monkeypatch.setattr(cli.ssh, "ssh_run", lambda *a, **k: pytest.fail("must not run"))
    cli._run_provision(profile, "ubuntu@h", "/lambda/nfs/p")  # returns silently


def test_runs_dir_with_env_and_streams(tmp_config_dir, monkeypatch):
    profile = _make_profile()
    _write_provision("p")

    copied = {}
    monkeypatch.setattr(cli.sync, "copy_dir",
                        lambda local, host, remote: copied.update(local=local, host=host, remote=remote))
    captured = {}
    def fake_run(host, command, *, capture=True, check=False, timeout=300):
        captured.update(host=host, command=command, capture=capture, timeout=timeout)
        return ssh.SSHResult(0, "", "")
    monkeypatch.setattr(cli.ssh, "ssh_run", fake_run)

    cli._run_provision(profile, "ubuntu@1.2.3.4", "/lambda/nfs/p")

    assert copied["local"] == str(profiles.provision_dir("p"))
    assert copied["remote"] == "/lambda/nfs/p/cloudgpu/provision"
    assert captured["host"] == "ubuntu@1.2.3.4"
    assert captured["capture"] is False                        # streams live
    assert captured["timeout"] == 3600                         # provision_timeout default
    assert "CLOUDGPU_PROVISION_DIR=/lambda/nfs/p/cloudgpu/provision" in captured["command"]
    assert "CLOUDGPU_APPS_DIR=/lambda/nfs/p/apps" in captured["command"]
    assert "cd /lambda/nfs/p/cloudgpu/provision" in captured["command"]
    assert "bash provision.sh" in captured["command"]


def test_dir_without_entrypoint_errors(tmp_config_dir, monkeypatch):
    profile = _make_profile()
    profiles.provision_dir("p").mkdir(parents=True, exist_ok=True)  # no provision.sh inside
    monkeypatch.setattr(cli.sync, "copy_dir", lambda *a, **k: pytest.fail("must not copy"))
    with pytest.raises(SystemExit):
        cli._run_provision(profile, "ubuntu@h", "/lambda/nfs/p")


def test_failure_exits_nonzero(tmp_config_dir, monkeypatch):
    profile = _make_profile()
    _write_provision("p", "exit 1\n")
    monkeypatch.setattr(cli.sync, "copy_dir", lambda *a, **k: None)
    monkeypatch.setattr(cli.ssh, "ssh_run", lambda *a, **k: ssh.SSHResult(1, "", ""))
    with pytest.raises(SystemExit):
        cli._run_provision(profile, "ubuntu@h", "/lambda/nfs/p")


def test_secrets_transferred_and_sourced_not_inlined(tmp_config_dir, monkeypatch):
    profile = _make_profile()
    _write_provision("p")
    # Secret value must never end up in the command string.
    profiles.secrets_file().write_text("CIVITAI_TOKEN=SECRET123\n")

    copied = []
    monkeypatch.setattr(cli.sync, "copy_dir", lambda *a, **k: None)
    monkeypatch.setattr(cli.sync, "copy_file",
                        lambda local, host, remote: copied.append((local, remote)))
    captured = {}
    monkeypatch.setattr(cli.ssh, "ssh_run",
                        lambda host, command, **k: captured.update(command=command) or ssh.SSHResult(0, "", ""))

    cli._run_provision(profile, "ubuntu@h", "/lambda/nfs/p")

    # The secrets file is transferred to an ephemeral home path...
    assert copied == [(str(profiles.secrets_file()), ".cloudgpu-secrets.env")]
    # ...sourced and cleaned up on the instance...
    assert 'set -a; . "$S"' in captured["command"]
    assert "trap" in captured["command"]
    # ...but the value never appears in the command we send.
    assert "SECRET123" not in captured["command"]


def test_no_secrets_means_no_secret_handling(tmp_config_dir, monkeypatch):
    profile = _make_profile()
    _write_provision("p")
    # no secrets.env created
    monkeypatch.setattr(cli.sync, "copy_dir", lambda *a, **k: None)
    monkeypatch.setattr(cli.sync, "copy_file", lambda *a, **k: pytest.fail("no secrets to copy"))
    captured = {}
    monkeypatch.setattr(cli.ssh, "ssh_run",
                        lambda host, command, **k: captured.update(command=command) or ssh.SSHResult(0, "", ""))
    cli._run_provision(profile, "ubuntu@h", "/lambda/nfs/p")
    assert ".cloudgpu-secrets.env" not in captured["command"]


def test_provision_timeout_is_overridable(tmp_config_dir, monkeypatch):
    profiles.profiles_dir().mkdir(parents=True, exist_ok=True)
    profiles.profile_path("p").write_text(
        'gpu = ["gh200"]\nssh_key = "mini"\nprovision_timeout = 7200\n'
    )
    _write_provision("p")
    profile = profiles.load_profile("p")
    monkeypatch.setattr(cli.sync, "copy_dir", lambda *a, **k: None)
    captured = {}
    monkeypatch.setattr(cli.ssh, "ssh_run",
                        lambda host, command, **k: captured.update(k) or ssh.SSHResult(0, "", ""))
    cli._run_provision(profile, "ubuntu@h", "/lambda/nfs/p")
    assert captured["timeout"] == 7200
