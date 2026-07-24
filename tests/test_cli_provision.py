"""Tests for provisioning (the profile folder is the payload). Mocked SSH/rsync."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from cloudgpu import cli
from cloudgpu.local import configs, profiles, ssh


@pytest.fixture(autouse=True)
def _isolate_config(tmp_config_dir):
    """Point CONFIG_DIR at a temp dir so tests never see the real secrets.env."""
    return tmp_config_dir


def _profile(tmp_path, *, entry="py"):
    d = tmp_path / "prof"
    d.mkdir()
    (d / "cloudgpu.toml").write_text('gpu = ["gh200"]\nssh_key = "mini"\n')
    if entry == "py":
        (d / "provision.py").write_text("print('hi')\n")
    elif entry == "sh":
        (d / "provision.sh").write_text("echo hi\n")
    elif entry == "both":
        (d / "provision.py").write_text("print('hi')\n")
        (d / "provision.sh").write_text("echo hi\n")
    return profiles.load_profile(d)


def test_no_entry_is_noop(tmp_path, monkeypatch):
    profile = _profile(tmp_path, entry=None)
    monkeypatch.setattr(cli.sync, "copy_dir", lambda *a, **k: pytest.fail("must not copy"))
    monkeypatch.setattr(cli.ssh, "ssh_run", lambda *a, **k: pytest.fail("must not run"))
    assert cli._run_provision(profile, "ubuntu@h", "/lambda/nfs/p") is False


def test_runs_python_entry_with_excludes(tmp_path, monkeypatch):
    profile = _profile(tmp_path, entry="py")
    copied = {}
    def fake_copy(local, host, remote, exclude=None):
        copied.update(local=local, host=host, remote=remote, exclude=exclude)
    monkeypatch.setattr(cli.sync, "copy_dir", fake_copy)
    captured = {}
    def fake_run(host, command, *, capture=True, check=False, timeout=300):
        captured.update(command=command, capture=capture, timeout=timeout)
        return ssh.SSHResult(0, "", "")
    monkeypatch.setattr(cli.ssh, "ssh_run", fake_run)

    assert cli._run_provision(profile, "ubuntu@1.2.3.4", "/lambda/nfs/p") is True

    assert copied["local"] == str(profile["dir"])
    # per-profile remote dir so multiple profiles on one instance don't collide
    assert copied["remote"] == "/lambda/nfs/p/cloudgpu/provision/prof"
    # state + secrets must never be rsynced onto the persistent filesystem
    assert "*.state.json" in copied["exclude"]
    assert "secrets.env" in copied["exclude"]
    assert captured["capture"] is False
    assert captured["timeout"] == 3600
    assert "CLOUDGPU_PROVISION_DIR=/lambda/nfs/p/cloudgpu/provision/prof" in captured["command"]
    assert "cd /lambda/nfs/p/cloudgpu/provision/prof" in captured["command"]
    assert "python3 provision.py" in captured["command"]


def test_python_preferred_over_sh(tmp_path, monkeypatch):
    profile = _profile(tmp_path, entry="both")
    monkeypatch.setattr(cli.sync, "copy_dir", lambda *a, **k: None)
    captured = {}
    monkeypatch.setattr(cli.ssh, "ssh_run",
                        lambda host, command, **k: captured.update(command=command) or ssh.SSHResult(0, "", ""))
    cli._run_provision(profile, "ubuntu@h", "/lambda/nfs/p")
    assert "python3 provision.py" in captured["command"]
    assert "bash provision.sh" not in captured["command"]


def test_sh_entry(tmp_path, monkeypatch):
    profile = _profile(tmp_path, entry="sh")
    monkeypatch.setattr(cli.sync, "copy_dir", lambda *a, **k: None)
    captured = {}
    monkeypatch.setattr(cli.ssh, "ssh_run",
                        lambda host, command, **k: captured.update(command=command) or ssh.SSHResult(0, "", ""))
    cli._run_provision(profile, "ubuntu@h", "/lambda/nfs/p")
    assert "bash provision.sh" in captured["command"]


def test_failure_exits_nonzero(tmp_path, monkeypatch):
    profile = _profile(tmp_path, entry="py")
    monkeypatch.setattr(cli.sync, "copy_dir", lambda *a, **k: None)
    monkeypatch.setattr(cli.ssh, "ssh_run", lambda *a, **k: ssh.SSHResult(1, "", ""))
    with pytest.raises(SystemExit):
        cli._run_provision(profile, "ubuntu@h", "/lambda/nfs/p")


def test_provision_timeout_override(tmp_path, monkeypatch):
    d = tmp_path / "prof"
    d.mkdir()
    (d / "cloudgpu.toml").write_text('gpu=["gh200"]\nssh_key="mini"\nprovision_timeout=7200\n')
    (d / "provision.py").write_text("print('hi')\n")
    profile = profiles.load_profile(d)
    monkeypatch.setattr(cli.sync, "copy_dir", lambda *a, **k: None)
    captured = {}
    monkeypatch.setattr(cli.ssh, "ssh_run",
                        lambda host, command, **k: captured.update(k) or ssh.SSHResult(0, "", ""))
    cli._run_provision(profile, "ubuntu@h", "/lambda/nfs/p")
    assert captured["timeout"] == 7200


def test_secrets_transferred_sourced_not_inlined(tmp_path, tmp_config_dir, monkeypatch):
    profile = _profile(tmp_path, entry="py")
    profiles.secrets_file().write_text("CIVITAI_TOKEN=SECRET123\n")  # under tmp_config_dir
    copied = []
    monkeypatch.setattr(cli.sync, "copy_dir", lambda *a, **k: None)
    monkeypatch.setattr(cli.sync, "copy_file",
                        lambda local, host, remote: copied.append((local, remote)))
    captured = {}
    monkeypatch.setattr(cli.ssh, "ssh_run",
                        lambda host, command, **k: captured.update(command=command) or ssh.SSHResult(0, "", ""))
    cli._run_provision(profile, "ubuntu@h", "/lambda/nfs/p")
    assert copied == [(str(profiles.secrets_file()), ".cloudgpu-secrets.env")]
    assert 'set -a; . "$S"' in captured["command"]
    assert "SECRET123" not in captured["command"]


def test_no_secrets_no_copy_file(tmp_path, tmp_config_dir, monkeypatch):
    profile = _profile(tmp_path, entry="py")
    monkeypatch.setattr(cli.sync, "copy_dir", lambda *a, **k: None)
    monkeypatch.setattr(cli.sync, "copy_file", lambda *a, **k: pytest.fail("no secrets to copy"))
    captured = {}
    monkeypatch.setattr(cli.ssh, "ssh_run",
                        lambda host, command, **k: captured.update(command=command) or ssh.SSHResult(0, "", ""))
    cli._run_provision(profile, "ubuntu@h", "/lambda/nfs/p")
    assert ".cloudgpu-secrets.env" not in captured["command"]


# --- the `cloudgpu provision` command (config + profile list) ------------------

def _repo_up(tmp_path, monkeypatch, *, profile_apps: dict[str, str]):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "c.toml").write_text('gpu = ["gh200"]\nssh_key = "mini"\n')
    for name, apps_toml in profile_apps.items():
        pd = tmp_path / "profiles" / name
        pd.mkdir(parents=True)
        (pd / "cloudgpu.toml").write_text(f"apps = {apps_toml}\n")
    monkeypatch.chdir(tmp_path)


def test_provision_command_records_profiles(tmp_path, tmp_config_dir, monkeypatch):
    _repo_up(tmp_path, monkeypatch, profile_apps={"generate": '["comfyui"]', "lora": '["ai-toolkit"]'})
    configs.save_state("c", {"host": "ubuntu@h", "persistent_dir": "/lambda/nfs/p"})

    ensured, provisioned = [], []
    monkeypatch.setattr(cli.sync, "sync_remote", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_ensure_apps", lambda host, pd, app_list: ensured.append(app_list))
    monkeypatch.setattr(cli, "_run_provision", lambda prof, host, pd: provisioned.append(prof["name"]) or True)
    monkeypatch.setattr(cli, "_restart_services", lambda host, app_list: None)

    r = CliRunner().invoke(cli.provision, ["-C", "c", "-P", "generate,lora"])
    assert r.exit_code == 0, r.output
    # apps installed once as a union; each profile provisioned in order
    assert ensured == [["comfyui", "ai-toolkit"]]
    assert provisioned == ["generate", "lora"]
    # both profiles recorded in the config's state
    assert configs.load_state("c")["provisioned_profiles"] == ["generate", "lora"]


def test_provision_command_requires_running_instance(tmp_path, tmp_config_dir, monkeypatch):
    _repo_up(tmp_path, monkeypatch, profile_apps={"generate": '["comfyui"]'})
    # no state saved -> no running instance
    r = CliRunner().invoke(cli.provision, ["-C", "c", "-P", "generate"])
    assert r.exit_code != 0
    assert "no running instance" in r.output.lower()
