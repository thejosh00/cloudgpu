"""Tests for `cloudgpu init` (scaffolding a config or a profile)."""

from __future__ import annotations

from click.testing import CliRunner

from cloudgpu import cli
from cloudgpu.local import configs, profiles


# --- profile scaffolding (default) ---------------------------------------------

def test_init_profile_no_apps(tmp_path):
    d = tmp_path / "bare"
    r = CliRunner().invoke(cli.init, [str(d)])
    assert r.exit_code == 0, r.output
    assert (d / "cloudgpu.toml").exists()
    assert (d / ".gitignore").exists()
    # no app-specific files
    assert not (d / "comfylib.py").exists()
    assert not (d / "provision.py").exists()
    assert profiles.load_profile(d)["apps"] == []


def test_init_profile_with_comfyui_vendors_files(tmp_path):
    d = tmp_path / "generate"
    r = CliRunner().invoke(cli.init, [str(d), "--apps", "comfyui"])
    assert r.exit_code == 0, r.output
    assert (d / "cloudgpu.toml").exists()
    assert (d / "comfylib.py").exists()
    assert (d / "provision.py").exists()
    assert profiles.load_profile(d)["apps"] == ["comfyui"]


def test_init_profile_unknown_app_errors(tmp_path):
    r = CliRunner().invoke(cli.init, [str(tmp_path / "x"), "--apps", "nope"])
    assert r.exit_code != 0
    assert "Unknown app" in r.output


def test_init_profile_refuses_existing(tmp_path):
    d = tmp_path / "x"
    CliRunner().invoke(cli.init, [str(d)])
    r = CliRunner().invoke(cli.init, [str(d)])
    assert r.exit_code != 0
    assert "already exists" in r.output


# --- config scaffolding (--config) ---------------------------------------------

def test_init_config_writes_toml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(cli.init, ["--config", "qwen-image", "--ssh-key", "mini", "--gpu", "gh200,a100"])
    assert r.exit_code == 0, r.output
    c = configs.load_config("qwen-image")
    assert c["gpu"] == ["gh200", "a100"] and c["ssh_key"] == "mini"


def test_init_config_requires_ssh_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(cli.init, ["--config", "c"])
    assert r.exit_code != 0
    assert "ssh-key" in r.output.lower()
