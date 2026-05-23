"""Tests for the profile store (desired-state TOML + runtime JSON + active pointer)."""

from __future__ import annotations

import pytest

from cloudgpu.local import config, profiles


@pytest.fixture
def store(tmp_config_dir):
    """Use the temp config dir (monkeypatched by tmp_config_dir) for profiles too."""
    return tmp_config_dir


class TestCreateAndLoad:
    def test_create_then_load_roundtrip_with_defaults(self, store):
        profiles.create_profile(
            "wash", filesystem="washington", gpu=["gh200", "a100"],
            apps=["comfyui"], ssh_key="mini",
        )
        p = profiles.load_profile("wash")
        assert p["name"] == "wash"
        assert p["filesystem"] == "washington"
        assert p["gpu"] == ["gh200", "a100"]
        assert p["apps"] == ["comfyui"]
        assert p["ssh_key"] == "mini"
        # defaults
        assert p["poll_seconds"] == 20
        assert p["max_hours"] == 12
        assert p["instance_name"] == "wash"

    def test_load_missing_raises(self, store):
        with pytest.raises(config.ConfigError, match="No profile named 'nope'"):
            profiles.load_profile("nope")

    def test_create_existing_raises_without_force(self, store):
        profiles.create_profile("p", filesystem="fs", gpu=["a10"], apps=[])
        with pytest.raises(config.ConfigError, match="already exists"):
            profiles.create_profile("p", filesystem="fs2", gpu=["a10"], apps=[])

    def test_create_overwrite(self, store):
        profiles.create_profile("p", filesystem="fs", gpu=["a10"], apps=[], ssh_key="mini")
        profiles.create_profile(
            "p", filesystem="fs2", gpu=["gh200"], apps=[], ssh_key="mini", overwrite=True
        )
        assert profiles.load_profile("p")["filesystem"] == "fs2"

    def test_gpu_string_is_coerced_to_list(self, store):
        profiles.profiles_dir().mkdir(parents=True, exist_ok=True)
        profiles.profile_path("p").write_text(
            'filesystem = "fs"\ngpu = "gh200"\nssh_key = "mini"\n'
        )
        assert profiles.load_profile("p")["gpu"] == ["gh200"]

    def test_filesystem_defaults_to_profile_name(self, store):
        profiles.profiles_dir().mkdir(parents=True, exist_ok=True)
        profiles.profile_path("p").write_text('gpu = ["gh200"]\nssh_key = "mini"\n')
        assert profiles.load_profile("p")["filesystem"] == "p"

    def test_explicit_filesystem_is_kept(self, store):
        profiles.create_profile(
            "p", filesystem="shared", gpu=["a10"], apps=[], ssh_key="mini"
        )
        assert profiles.load_profile("p")["filesystem"] == "shared"

    def test_empty_gpu_is_invalid(self, store):
        profiles.profiles_dir().mkdir(parents=True, exist_ok=True)
        profiles.profile_path("p").write_text(
            'filesystem = "fs"\ngpu = []\nssh_key = "mini"\n'
        )
        with pytest.raises(config.ConfigError, match="gpu"):
            profiles.load_profile("p")

    def test_missing_ssh_key_is_invalid(self, store):
        profiles.profiles_dir().mkdir(parents=True, exist_ok=True)
        profiles.profile_path("p").write_text('filesystem = "fs"\ngpu = ["gh200"]\n')
        with pytest.raises(config.ConfigError, match="ssh_key"):
            profiles.load_profile("p")


class TestActivePointer:
    def test_set_and_get_active(self, store):
        profiles.create_profile("p", filesystem="fs", gpu=["a10"], apps=[])
        profiles.set_active("p")
        assert profiles.get_active() == "p"

    def test_set_active_unknown_raises(self, store):
        with pytest.raises(config.ConfigError):
            profiles.set_active("ghost")

    def test_require_profile_prefers_explicit(self, store):
        profiles.create_profile("a", filesystem="fs", gpu=["a10"], apps=[])
        profiles.set_active("a")
        assert profiles.require_profile("explicit") == "explicit"

    def test_require_profile_falls_back_to_active(self, store):
        profiles.create_profile("a", filesystem="fs", gpu=["a10"], apps=[])
        profiles.set_active("a")
        assert profiles.require_profile(None) == "a"

    def test_require_profile_none_raises(self, store):
        with pytest.raises(config.ConfigError, match="No profile selected"):
            profiles.require_profile(None)


class TestRuntime:
    def test_save_load_clear(self, store):
        profiles.save_runtime("p", {"host": "ubuntu@1.2.3.4", "ip": "1.2.3.4"})
        rt = profiles.load_runtime("p")
        assert rt["host"] == "ubuntu@1.2.3.4"
        assert rt["profile"] == "p"  # injected
        profiles.clear_runtime("p")
        assert profiles.load_runtime("p") == {}

    def test_load_missing_runtime_is_empty(self, store):
        assert profiles.load_runtime("nobody") == {}


class TestDelete:
    def test_delete_removes_toml_runtime_and_active(self, store):
        profiles.create_profile("p", filesystem="fs", gpu=["a10"], apps=[])
        profiles.save_runtime("p", {"host": "h"})
        profiles.set_active("p")
        profiles.delete_profile("p")
        assert not profiles.profile_path("p").exists()
        assert profiles.load_runtime("p") == {}
        assert profiles.get_active() is None
