"""Tests for app-aware `cloudgpu forward` (mocked target + SSH)."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from cloudgpu import cli


def _repo(tmp_path, monkeypatch, *, profiles_apps: dict[str, str], provisioned: list[str],
          extra: dict[str, str] | None = None):
    """Build a repo root with configs/<c> + state + profiles/<p>, then chdir into it.

    ``extra`` appends raw toml lines to a profile (e.g. its own 'ports').
    """
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "c.toml").write_text('gpu = ["gh200"]\nssh_key = "mini"\n')
    (tmp_path / "configs" / "c.state.json").write_text(json.dumps({
        "host": "ubuntu@1.2.3.4", "persistent_dir": "/lambda/nfs/p",
        "provisioned_profiles": provisioned,
    }))
    for pname, apps_toml in profiles_apps.items():
        pd = tmp_path / "profiles" / pname
        pd.mkdir(parents=True)
        (pd / "cloudgpu.toml").write_text(f"apps = {apps_toml}\n" + (extra or {}).get(pname, ""))
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _fake_target(monkeypatch):
    # forward needs a resolved host; skip the real "is it running" check.
    monkeypatch.setattr(cli, "_resolve_target",
                        lambda host=None, config_name=None: ("ubuntu@1.2.3.4", "/lambda/nfs/p"))


def _capture_ssh(monkeypatch):
    captured = {}

    def fake(host, command=None, *, ssh_args=None):
        captured.update(host=host, command=command, ssh_args=ssh_args)
        return 0

    monkeypatch.setattr(cli.ssh, "ssh_interactive", fake)
    return captured


def test_derives_comfyui_port(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, profiles_apps={"generate": '["comfyui"]'}, provisioned=["generate"])
    cap = _capture_ssh(monkeypatch)
    r = CliRunner().invoke(cli.forward, ["-C", "c"])
    assert r.exit_code == 0, r.output
    assert cap["ssh_args"] == ["-L", "8188:localhost:8188", "-N"]


def test_multiple_app_ports_across_profiles(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch,
          profiles_apps={"generate": '["comfyui"]', "lora": '["ai-toolkit"]'},
          provisioned=["generate", "lora"])
    cap = _capture_ssh(monkeypatch)
    r = CliRunner().invoke(cli.forward, ["-C", "c"])
    assert r.exit_code == 0, r.output
    assert cap["ssh_args"] == ["-L", "8188:localhost:8188", "-L", "8675:localhost:8675", "-N"]


def test_explicit_port(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, profiles_apps={"generate": '["comfyui"]'}, provisioned=["generate"])
    cap = _capture_ssh(monkeypatch)
    r = CliRunner().invoke(cli.forward, ["-C", "c", "--port", "9000"])
    assert r.exit_code == 0, r.output
    assert cap["ssh_args"] == ["-L", "9000:localhost:9000", "-N"]


def test_run_mode_single_port(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, profiles_apps={"generate": '["comfyui"]'}, provisioned=["generate"])
    cap = _capture_ssh(monkeypatch)
    r = CliRunner().invoke(cli.forward, ["-C", "c", "--run", "nvidia-smi"])
    assert r.exit_code == 0, r.output
    assert cap["ssh_args"] == ["-L", "8188:localhost:8188"]   # no -N; command runs
    assert "nvidia-smi" in cap["command"]


def test_profile_declared_port_without_an_app(tmp_path, monkeypatch):
    """imgv-style profile: the provision script serves it, so the profile owns the port."""
    _repo(tmp_path, monkeypatch, profiles_apps={"imgv": "[]"}, provisioned=["imgv"],
          extra={"imgv": "ports = [8080]\n"})
    cap = _capture_ssh(monkeypatch)
    r = CliRunner().invoke(cli.forward, ["-C", "c"])
    assert r.exit_code == 0, r.output
    assert cap["ssh_args"] == ["-L", "8080:localhost:8080", "-N"]


def test_profile_ports_join_app_ports(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch,
          profiles_apps={"lora": '["ai-toolkit"]', "imgv": "[]"},
          provisioned=["lora", "imgv"], extra={"imgv": "ports = [8080]\n"})
    cap = _capture_ssh(monkeypatch)
    r = CliRunner().invoke(cli.forward, ["-C", "c"])
    assert r.exit_code == 0, r.output
    assert cap["ssh_args"] == ["-L", "8675:localhost:8675", "-L", "8080:localhost:8080", "-N"]


def test_no_app_ports_errors(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, profiles_apps={}, provisioned=[])
    _capture_ssh(monkeypatch)
    r = CliRunner().invoke(cli.forward, ["-C", "c"])
    assert r.exit_code != 0
    assert "No ports to forward" in r.output
