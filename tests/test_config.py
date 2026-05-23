"""Tests for config module."""

from __future__ import annotations

import json

import pytest

from cloudgpu.local.config import (
    load_config,
    save_config,
    get_host,
    save_host,
    get_persistent_dir,
    save_launch_info,
    get_launch_info,
    ConfigError,
)


class TestConfig:
    def test_load_empty(self, tmp_config_dir):
        assert load_config() == {}

    def test_save_and_load(self, tmp_config_dir):
        save_config({"host": "1.2.3.4", "persistent_dir": "/lambda/nfs/fs1"})
        config = load_config()
        assert config["host"] == "1.2.3.4"
        assert config["persistent_dir"] == "/lambda/nfs/fs1"

    def test_save_host(self, tmp_config_dir):
        save_host("10.0.0.1", "/lambda/nfs/myfs")
        config = load_config()
        assert config["host"] == "10.0.0.1"
        assert config["persistent_dir"] == "/lambda/nfs/myfs"


class TestGetHost:
    def test_explicit_host(self, tmp_config_dir):
        assert get_host("explicit-host") == "explicit-host"

    def test_saved_host(self, tmp_config_dir):
        save_host("saved-host", "/lambda/nfs/fs")
        assert get_host(None) == "saved-host"

    def test_no_host_raises(self, tmp_config_dir):
        with pytest.raises(ConfigError, match="No host specified."):
            get_host(None)


class TestGetPersistentDir:
    def test_saved_dir(self, tmp_config_dir):
        save_host("host", "/lambda/nfs/myfs")
        assert get_persistent_dir() == "/lambda/nfs/myfs"

    def test_no_dir_raises(self, tmp_config_dir):
        with pytest.raises(ConfigError, match="Persistent directory not configured"):
            get_persistent_dir()


class TestLaunchInfo:
    def test_empty_when_unset(self, tmp_config_dir):
        assert get_launch_info() == {}

    def test_save_and_get_full(self, tmp_config_dir):
        save_launch_info(
            "washington", instance_type="gpu_1x_gh200", ssh_key="mini", region="us-east-3"
        )
        assert get_launch_info() == {
            "filesystem": "washington",
            "instance_type": "gpu_1x_gh200",
            "ssh_key": "mini",
            "region": "us-east-3",
        }

    def test_filesystem_only(self, tmp_config_dir):
        save_launch_info("california")
        assert get_launch_info() == {"filesystem": "california"}

    def test_partial_update_preserves_existing(self, tmp_config_dir):
        save_launch_info(
            "washington", instance_type="gpu_1x_gh200", ssh_key="mini", region="us-east-3"
        )
        # Re-save with only some fields; others must survive.
        save_launch_info("washington", instance_type="gpu_1x_a100")
        info = get_launch_info()
        assert info["instance_type"] == "gpu_1x_a100"
        assert info["ssh_key"] == "mini"
        assert info["region"] == "us-east-3"

    def test_does_not_disturb_host(self, tmp_config_dir):
        save_host("10.0.0.1", "/lambda/nfs/washington")
        save_launch_info("washington", instance_type="gpu_1x_gh200")
        config = load_config()
        assert config["host"] == "10.0.0.1"
        assert config["persistent_dir"] == "/lambda/nfs/washington"
        assert config["launch"]["instance_type"] == "gpu_1x_gh200"
