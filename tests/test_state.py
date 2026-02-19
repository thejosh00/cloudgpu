"""Tests for remote state module."""

from __future__ import annotations

import json
import os

from cloudgpu.remote.state import State


class TestState:
    def test_load_empty(self, tmp_persistent_dir):
        state = State(tmp_persistent_dir)
        data = state.load()
        assert "apps" in data
        assert data["apps"] == {}

    def test_save_and_load(self, tmp_persistent_dir):
        state = State(tmp_persistent_dir)
        data = {"apps": {"comfyui": {"status": "installed"}}, "created_at": 1.0}
        state.save(data)
        loaded = state.load()
        assert loaded["apps"]["comfyui"]["status"] == "installed"
        assert "updated_at" in loaded

    def test_set_and_get_app(self, tmp_persistent_dir):
        state = State(tmp_persistent_dir)
        state.set_app("comfyui", {"status": "installed", "version": "abc123"})
        app = state.get_app("comfyui")
        assert app["status"] == "installed"
        assert app["version"] == "abc123"

    def test_get_missing_app(self, tmp_persistent_dir):
        state = State(tmp_persistent_dir)
        assert state.get_app("nonexistent") is None

    def test_remove_app(self, tmp_persistent_dir):
        state = State(tmp_persistent_dir)
        state.set_app("comfyui", {"status": "installed"})
        state.remove_app("comfyui")
        assert state.get_app("comfyui") is None

    def test_list_apps(self, tmp_persistent_dir):
        state = State(tmp_persistent_dir)
        state.set_app("comfyui", {"status": "installed"})
        state.set_app("another", {"status": "installed"})
        apps = state.list_apps()
        assert len(apps) == 2
        assert "comfyui" in apps
        assert "another" in apps

    def test_dirs(self, tmp_persistent_dir):
        state = State(tmp_persistent_dir)
        assert state.apps_dir == os.path.join(tmp_persistent_dir, "apps")
        assert state.venvs_dir == os.path.join(tmp_persistent_dir, "venvs")
        assert state.bin_dir == os.path.join(tmp_persistent_dir, ".cloudgpu", "bin")

    def test_creates_directories_on_save(self, tmp_persistent_dir):
        state = State(tmp_persistent_dir)
        state.set_app("test", {"status": "ok"})
        assert os.path.isdir(os.path.join(tmp_persistent_dir, ".cloudgpu"))
        assert os.path.isdir(os.path.join(tmp_persistent_dir, ".cloudgpu", "bin"))
        assert os.path.isdir(os.path.join(tmp_persistent_dir, ".cloudgpu", "remote"))
