"""Tests for the hardware config store (configs/)."""

from __future__ import annotations

import pytest

from cloudgpu.local import config, configs


def _write_config(base, name, body='gpu = ["gh200"]\nssh_key = "mini"\n'):
    d = base / "configs"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.toml").write_text(body)
    return d / f"{name}.toml"


class TestFindConfig:
    def test_name_resolves_under_configs(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "qwen-image")
        monkeypatch.chdir(tmp_path)
        assert configs.find_config("qwen-image") == (tmp_path / "configs" / "qwen-image.toml").resolve()

    def test_explicit_path(self, tmp_path):
        p = _write_config(tmp_path, "sdxl")
        assert configs.find_config(str(p)) == p.resolve()

    def test_missing_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(config.ConfigError, match="No config"):
            configs.find_config("nope")


class TestLoadConfig:
    def test_defaults(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "qwen-image")
        monkeypatch.chdir(tmp_path)
        c = configs.load_config("qwen-image")
        assert c["name"] == "qwen-image"
        assert c["gpu"] == ["gh200"]
        assert c["ssh_key"] == "mini"
        assert c["filesystem"] == "qwen-image"      # defaults to config name
        assert c["instance_name"] == "qwen-image"
        assert c["poll_seconds"] == 20 and c["max_hours"] == 12
        assert c["auto_terminate_hours"] == 0.0

    def test_explicit_filesystem_kept(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "c", 'gpu=["gh200"]\nssh_key="mini"\nfilesystem="shared"\n')
        monkeypatch.chdir(tmp_path)
        assert configs.load_config("c")["filesystem"] == "shared"

    def test_gpu_string_coerced(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "c", 'gpu="gh200"\nssh_key="mini"\n')
        monkeypatch.chdir(tmp_path)
        assert configs.load_config("c")["gpu"] == ["gh200"]

    def test_missing_gpu_invalid(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "c", 'ssh_key="mini"\n')
        monkeypatch.chdir(tmp_path)
        with pytest.raises(config.ConfigError, match="gpu"):
            configs.load_config("c")

    def test_missing_ssh_key_invalid(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "c", 'gpu=["gh200"]\n')
        monkeypatch.chdir(tmp_path)
        with pytest.raises(config.ConfigError, match="ssh_key"):
            configs.load_config("c")

    def test_bad_auto_terminate_invalid(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "c", 'gpu=["gh200"]\nssh_key="mini"\nauto_terminate_hours="soon"\n')
        monkeypatch.chdir(tmp_path)
        with pytest.raises(config.ConfigError, match="auto_terminate_hours"):
            configs.load_config("c")


class TestState:
    def test_save_load_clear(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "qwen-image")
        monkeypatch.chdir(tmp_path)
        configs.save_state("qwen-image", {"instance_id": "i-1", "host": "ubuntu@1.2.3.4"})
        assert configs.state_path("qwen-image").name == "qwen-image.state.json"
        st = configs.load_state("qwen-image")
        assert st["instance_id"] == "i-1"
        configs.clear_state("qwen-image")
        assert configs.load_state("qwen-image") == {}

    def test_missing_state_is_empty(self, tmp_path, monkeypatch):
        _write_config(tmp_path, "qwen-image")
        monkeypatch.chdir(tmp_path)
        assert configs.load_state("qwen-image") == {}


class TestScaffold:
    def test_writes_toml(self, tmp_path):
        p = configs.scaffold("qwen-image", ssh_key="mini", gpu=["gh200", "a100"],
                             configs_dir=tmp_path)
        assert p.exists()
        c = configs.load_config(str(p))
        assert c["gpu"] == ["gh200", "a100"] and c["ssh_key"] == "mini"

    def test_refuses_existing_without_force(self, tmp_path):
        configs.scaffold("c", ssh_key="mini", gpu=["gh200"], configs_dir=tmp_path)
        with pytest.raises(config.ConfigError, match="already exists"):
            configs.scaffold("c", ssh_key="mini", gpu=["a10"], configs_dir=tmp_path)
