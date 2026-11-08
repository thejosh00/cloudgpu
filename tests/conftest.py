"""Shared test fixtures."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_persistent_dir(tmp_path):
    """Create a temporary persistent directory structure."""
    persistent = tmp_path / "nfs" / "my-filesystem"
    persistent.mkdir(parents=True)
    return str(persistent)


@pytest.fixture
def tmp_config_dir(tmp_path, monkeypatch):
    """Override config directory to a temp location."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setattr("cloudgpu.local.config.CONFIG_DIR", config_dir)
    monkeypatch.setattr("cloudgpu.local.config.CONFIG_FILE", config_dir / "config.json")
    return config_dir


@pytest.fixture(autouse=True)
def _short_launch_look_window(monkeypatch):
    """In-process tests: look for what a doubtful launch made over a tiny window, not a minute."""
    from cloudgpu.local import orchestration

    monkeypatch.setattr(orchestration, "LAUNCH_LOOK_WINDOW_SECONDS", 0.05)
    monkeypatch.setattr(orchestration, "LAUNCH_LOOK_EVERY_SECONDS", 0.01)


@pytest.fixture(autouse=True)
def _instance_marker(tmp_path_factory, monkeypatch):
    """Every test runs "on an instance" whose marker is a temp file, never the real path.

    Tests that check the refusal point the guard at a missing temp path themselves.
    """
    try:
        from cloudgpu.remote import instance
    except ImportError:  # base code without the guard (red runs)
        return
    marker = tmp_path_factory.mktemp("instance") / "instance"
    marker.write_text("")
    monkeypatch.setattr(instance, "MARKER", str(marker))
