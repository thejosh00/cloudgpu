"""Tests for app-aware `cloudgpu forward` (mocked target + SSH)."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from cloudgpu import cli


def _profile(tmp_path, apps='["comfyui"]'):
    d = tmp_path / "prof"
    d.mkdir()
    (d / "cloudgpu.toml").write_text(f'gpu = ["gh200"]\nssh_key = "mini"\napps = {apps}\n')
    return d


@pytest.fixture(autouse=True)
def _fake_target(monkeypatch):
    # forward needs a resolved host; skip the real "is it running" check.
    monkeypatch.setattr(cli, "_resolve_target",
                        lambda host=None, profile=None: ("ubuntu@1.2.3.4", "/lambda/nfs/p"))


def _capture_ssh(monkeypatch):
    captured = {}

    def fake(host, command=None, *, ssh_args=None):
        captured.update(host=host, command=command, ssh_args=ssh_args)
        return 0

    monkeypatch.setattr(cli.ssh, "ssh_interactive", fake)
    return captured


def test_derives_comfyui_port(tmp_path, monkeypatch):
    d = _profile(tmp_path)
    cap = _capture_ssh(monkeypatch)
    r = CliRunner().invoke(cli.forward, ["--profile", str(d)])
    assert r.exit_code == 0, r.output
    assert cap["ssh_args"] == ["-L", "8188:localhost:8188", "-N"]


def test_multiple_app_ports(tmp_path, monkeypatch):
    from cloudgpu.local.apps import AppSpec
    monkeypatch.setattr(cli.apps, "APPS", {
        "comfyui": AppSpec("comfyui", ports=(8188,)),
        "foo": AppSpec("foo", ports=(7000,)),
    })
    d = _profile(tmp_path, apps='["comfyui", "foo"]')
    cap = _capture_ssh(monkeypatch)
    r = CliRunner().invoke(cli.forward, ["--profile", str(d)])
    assert r.exit_code == 0, r.output
    assert cap["ssh_args"] == ["-L", "8188:localhost:8188", "-L", "7000:localhost:7000", "-N"]


def test_explicit_port(tmp_path, monkeypatch):
    d = _profile(tmp_path)
    cap = _capture_ssh(monkeypatch)
    r = CliRunner().invoke(cli.forward, ["--profile", str(d), "--port", "9000"])
    assert r.exit_code == 0, r.output
    assert cap["ssh_args"] == ["-L", "9000:localhost:9000", "-N"]


def test_run_mode_single_port(tmp_path, monkeypatch):
    d = _profile(tmp_path)
    cap = _capture_ssh(monkeypatch)
    r = CliRunner().invoke(cli.forward, ["--profile", str(d), "--run", "nvidia-smi"])
    assert r.exit_code == 0, r.output
    assert cap["ssh_args"] == ["-L", "8188:localhost:8188"]   # no -N; command runs
    assert "nvidia-smi" in cap["command"]


def test_no_app_ports_errors(tmp_path, monkeypatch):
    d = _profile(tmp_path, apps='[]')
    _capture_ssh(monkeypatch)
    r = CliRunner().invoke(cli.forward, ["--profile", str(d)])
    assert r.exit_code != 0
    assert "No app ports" in r.output
