"""Tests for the directory-as-profile store (provisioning-only)."""

from __future__ import annotations

import pytest

from cloudgpu.local import config, profiles


def _write_toml(d, body='apps = ["comfyui"]\n'):
    d.mkdir(parents=True, exist_ok=True)
    (d / profiles.PROFILE_FILE).write_text(body)
    return d


class TestFindProfile:
    def test_explicit_dir(self, tmp_path):
        d = _write_toml(tmp_path / "proj")
        assert profiles.find_profile(str(d)) == d.resolve()

    def test_name_resolves_under_profiles(self, tmp_path, monkeypatch):
        _write_toml(tmp_path / "profiles" / "generate")
        monkeypatch.chdir(tmp_path)
        assert profiles.find_profile("generate") == (tmp_path / "profiles" / "generate").resolve()

    def test_missing_toml_raises(self, tmp_path):
        with pytest.raises(config.ConfigError, match="No cloudgpu.toml"):
            profiles.find_profile(str(tmp_path / "nope"))

    def test_has_profile(self, tmp_path):
        assert not profiles.has_profile(tmp_path)
        _write_toml(tmp_path)
        assert profiles.has_profile(tmp_path)


class TestLoadProfile:
    def test_defaults(self, tmp_path):
        d = _write_toml(tmp_path / "generate", "")  # empty toml -> all defaults
        p = profiles.load_profile(d)
        assert p["name"] == "generate"
        assert p["dir"] == d.resolve()
        assert p["apps"] == []
        assert p["provision_timeout"] == 3600
        # hardware keys are gone from profiles
        assert "gpu" not in p and "ssh_key" not in p and "filesystem" not in p

    def test_apps_and_timeout(self, tmp_path):
        d = _write_toml(tmp_path / "p", 'apps = ["comfyui", "ai-toolkit"]\nprovision_timeout = 9000\n')
        p = profiles.load_profile(d)
        assert p["apps"] == ["comfyui", "ai-toolkit"]
        assert p["provision_timeout"] == 9000

    def test_ports_default_empty(self, tmp_path):
        assert profiles.load_profile(_write_toml(tmp_path / "p", ""))["ports"] == []

    def test_ports_parsed(self, tmp_path):
        d = _write_toml(tmp_path / "p", "apps = []\nports = [8080, 3000]\n")
        assert profiles.load_profile(d)["ports"] == [8080, 3000]

    def test_ports_int_coerced(self, tmp_path):
        d = _write_toml(tmp_path / "p", "apps = []\nports = 8080\n")
        assert profiles.load_profile(d)["ports"] == [8080]

    def test_ports_must_be_numbers(self, tmp_path):
        d = _write_toml(tmp_path / "p", 'apps = []\nports = ["8080"]\n')
        with pytest.raises(config.ConfigError, match="ports"):
            profiles.load_profile(d)

    def test_apps_string_coerced(self, tmp_path):
        d = _write_toml(tmp_path / "p", 'apps = "comfyui"\n')
        assert profiles.load_profile(d)["apps"] == ["comfyui"]

    def test_missing_toml_invalid(self, tmp_path):
        with pytest.raises(config.ConfigError):
            profiles.load_profile(tmp_path / "nope")


class TestScaffold:
    def test_writes_toml_and_gitignore_only(self, tmp_path):
        # scaffold is app-agnostic: just cloudgpu.toml + .gitignore (app files come from
        # the app registry, see test_apps).
        d = tmp_path / "new"
        profiles.scaffold(d, apps=["comfyui"])
        assert (d / "cloudgpu.toml").exists()
        assert (d / ".gitignore").exists()
        assert not (d / "comfylib.py").exists()
        assert not (d / "provision.py").exists()
        assert profiles.load_profile(d)["apps"] == ["comfyui"]

    def test_refuses_existing_without_force(self, tmp_path):
        profiles.scaffold(tmp_path / "x", apps=[])
        with pytest.raises(config.ConfigError, match="already exists"):
            profiles.scaffold(tmp_path / "x", apps=["comfyui"])

    def test_force_overwrites(self, tmp_path):
        d = tmp_path / "x"
        profiles.scaffold(d, apps=[])
        profiles.scaffold(d, apps=["comfyui"], force=True)
        assert profiles.load_profile(d)["apps"] == ["comfyui"]
