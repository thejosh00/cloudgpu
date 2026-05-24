"""Tests for the directory-as-profile store."""

from __future__ import annotations

import pytest

from cloudgpu.local import config, profiles


def _write_toml(d, body='gpu = ["gh200"]\nssh_key = "mini"\n'):
    d.mkdir(parents=True, exist_ok=True)
    (d / profiles.PROFILE_FILE).write_text(body)
    return d


class TestFindProfileDir:
    def test_finds_cwd(self, tmp_path, monkeypatch):
        _write_toml(tmp_path)
        monkeypatch.chdir(tmp_path)
        assert profiles.find_profile_dir() == tmp_path.resolve()

    def test_explicit_dir(self, tmp_path):
        d = _write_toml(tmp_path / "proj")
        assert profiles.find_profile_dir(str(d)) == d.resolve()

    def test_missing_toml_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(config.ConfigError, match="No cloudgpu.toml"):
            profiles.find_profile_dir()

    def test_has_profile(self, tmp_path):
        assert not profiles.has_profile(tmp_path)
        _write_toml(tmp_path)
        assert profiles.has_profile(tmp_path)


class TestLoadProfile:
    def test_defaults(self, tmp_path):
        d = _write_toml(tmp_path / "my-comfy")
        p = profiles.load_profile(d)
        assert p["name"] == "my-comfy"
        assert p["dir"] == d.resolve()
        assert p["filesystem"] == "my-comfy"   # defaults to folder name
        assert p["gpu"] == ["gh200"]
        assert p["ssh_key"] == "mini"
        assert p["apps"] == []
        assert p["instance_name"] == "my-comfy"
        assert p["poll_seconds"] == 20 and p["max_hours"] == 12
        assert p["provision_timeout"] == 3600

    def test_explicit_filesystem_kept(self, tmp_path):
        d = _write_toml(tmp_path / "p", 'gpu=["gh200"]\nssh_key="mini"\nfilesystem="shared"\n')
        assert profiles.load_profile(d)["filesystem"] == "shared"

    def test_gpu_string_coerced(self, tmp_path):
        d = _write_toml(tmp_path / "p", 'gpu="gh200"\nssh_key="mini"\n')
        assert profiles.load_profile(d)["gpu"] == ["gh200"]

    def test_missing_gpu_invalid(self, tmp_path):
        d = _write_toml(tmp_path / "p", 'ssh_key="mini"\n')
        with pytest.raises(config.ConfigError, match="gpu"):
            profiles.load_profile(d)

    def test_missing_ssh_key_invalid(self, tmp_path):
        d = _write_toml(tmp_path / "p", 'gpu=["gh200"]\n')
        with pytest.raises(config.ConfigError, match="ssh_key"):
            profiles.load_profile(d)

    def test_missing_toml_invalid(self, tmp_path):
        with pytest.raises(config.ConfigError):
            profiles.load_profile(tmp_path)


class TestState:
    def test_save_load_clear(self, tmp_path):
        _write_toml(tmp_path)
        profiles.save_state(tmp_path, {"instance_id": "i-1", "host": "ubuntu@1.2.3.4"})
        assert profiles.state_path(tmp_path).name == "cloudgpu.state.json"
        st = profiles.load_state(tmp_path)
        assert st["instance_id"] == "i-1"
        profiles.clear_state(tmp_path)
        assert profiles.load_state(tmp_path) == {}

    def test_missing_state_is_empty(self, tmp_path):
        assert profiles.load_state(tmp_path) == {}


class TestScaffold:
    def test_writes_all_files_and_loads(self, tmp_path):
        d = tmp_path / "new"
        profiles.scaffold(d, ssh_key="mini", gpu=["gh200", "a100"], apps=["comfyui"])
        assert (d / "cloudgpu.toml").exists()
        assert (d / "comfylib.py").exists()
        assert (d / "provision.py").exists()
        assert (d / ".gitignore").read_text().strip() == "cloudgpu.state.json"
        p = profiles.load_profile(d)
        assert p["gpu"] == ["gh200", "a100"] and p["ssh_key"] == "mini"

    def test_refuses_existing_without_force(self, tmp_path):
        profiles.scaffold(tmp_path / "x", ssh_key="mini", gpu=["gh200"], apps=[])
        with pytest.raises(config.ConfigError, match="already exists"):
            profiles.scaffold(tmp_path / "x", ssh_key="mini", gpu=["a10"], apps=[])

    def test_force_overwrites(self, tmp_path):
        d = tmp_path / "x"
        profiles.scaffold(d, ssh_key="mini", gpu=["gh200"], apps=[])
        profiles.scaffold(d, ssh_key="mini", gpu=["a10"], apps=[], force=True)
        assert profiles.load_profile(d)["gpu"] == ["a10"]

    def test_explicit_filesystem(self, tmp_path):
        d = tmp_path / "x"
        profiles.scaffold(d, ssh_key="mini", gpu=["gh200"], apps=[], filesystem="shared")
        assert profiles.load_profile(d)["filesystem"] == "shared"
