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
