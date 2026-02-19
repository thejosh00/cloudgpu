"""Tests for config module."""

from __future__ import annotations

import json

import pytest

from cloudgpu.local.config import load_config, save_config, get_host, save_host, get_persistent_dir


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
        import click
        with pytest.raises(click.UsageError, match="No host specified"):
            get_host(None)


class TestGetPersistentDir:
    def test_saved_dir(self, tmp_config_dir):
        save_host("host", "/lambda/nfs/myfs")
        assert get_persistent_dir() == "/lambda/nfs/myfs"

    def test_no_dir_raises(self, tmp_config_dir):
        import click
        with pytest.raises(click.UsageError, match="Persistent directory not configured"):
            get_persistent_dir()
