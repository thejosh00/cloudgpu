"""Tests for `cloudgpu init` (scaffolding a profile folder)."""

from __future__ import annotations

from click.testing import CliRunner

from cloudgpu import cli
from cloudgpu.local import profiles


def test_init_bare_no_apps(tmp_path):
    d = tmp_path / "bare"
    r = CliRunner().invoke(cli.init, [str(d), "--ssh-key", "mini"])
    assert r.exit_code == 0, r.output
    assert (d / "cloudgpu.toml").exists()
    assert (d / ".gitignore").exists()
    # bare machine: no app-specific files
    assert not (d / "comfylib.py").exists()
    assert not (d / "provision.py").exists()
    assert profiles.load_profile(d)["apps"] == []


def test_init_with_comfyui_vendors_files(tmp_path):
    d = tmp_path / "newprof"
    r = CliRunner().invoke(
        cli.init, [str(d), "--ssh-key", "mini", "--gpu", "gh200,a100", "--apps", "comfyui"]
    )
    assert r.exit_code == 0, r.output
    assert (d / "cloudgpu.toml").exists()
    assert (d / "comfylib.py").exists()
    assert (d / "provision.py").exists()
    p = profiles.load_profile(d)
    assert p["gpu"] == ["gh200", "a100"] and p["apps"] == ["comfyui"]


def test_init_unknown_app_errors(tmp_path):
    r = CliRunner().invoke(cli.init, [str(tmp_path / "x"), "--ssh-key", "mini", "--apps", "nope"])
    assert r.exit_code != 0
    assert "Unknown app" in r.output


def test_init_requires_ssh_key(tmp_path):
    r = CliRunner().invoke(cli.init, [str(tmp_path / "x")])
    assert r.exit_code != 0
    assert "ssh-key" in r.output.lower()


def test_init_refuses_existing(tmp_path):
    d = tmp_path / "x"
    CliRunner().invoke(cli.init, [str(d), "--ssh-key", "mini"])
    r = CliRunner().invoke(cli.init, [str(d), "--ssh-key", "mini"])
    assert r.exit_code != 0
    assert "already exists" in r.output
