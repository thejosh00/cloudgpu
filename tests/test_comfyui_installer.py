"""Tests for ComfyUI installer."""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest

from cloudgpu.remote.apps.comfyui import ComfyUIInstaller
from cloudgpu.remote.state import State


class TestComfyUIInstaller:
    @pytest.fixture
    def state(self, tmp_persistent_dir):
        return State(tmp_persistent_dir)

    @pytest.fixture
    def installer(self, state):
        return ComfyUIInstaller(state)

    def test_name(self, installer):
        assert installer.name == "comfyui"

    def test_paths(self, installer, tmp_persistent_dir):
        assert installer.app_dir == os.path.join(tmp_persistent_dir, "apps", "comfyui")
        assert installer.venv_dir == os.path.join(tmp_persistent_dir, "venvs", "comfyui")
        assert installer.launch_script == os.path.join(
            tmp_persistent_dir, "cloudgpu", "bin", "comfyui"
        )

    def test_service_spec(self, installer):
        spec = installer.service_spec()
        assert spec["name"] == "comfyui"
        assert spec["exec_start"] == installer.launch_script
        assert spec["workdir"] == installer.app_dir
        assert spec["port"] == 8188

    @patch("cloudgpu.remote.apps.comfyui.service_active", return_value="inactive")
    def test_get_status_not_installed(self, mock_active, installer):
        status = installer.get_status()
        assert status["status"] == "broken"
        assert status["torch_cuda"] is False
        assert status["service"] == "inactive"
        assert status["port"] == 8188

    @patch("cloudgpu.remote.apps.comfyui.install_service")
    @patch("cloudgpu.remote.apps.comfyui.run")
    @patch("cloudgpu.remote.apps.comfyui.pip_install")
    @patch("cloudgpu.remote.apps.comfyui.check_torch_cuda", return_value=True)
    def test_install_clones_and_creates_venv(self, mock_torch, mock_pip, mock_run, mock_service, installer, state):
        """Test that install calls git clone and creates venv."""
        # Mock _get_version
        mock_run.return_value = MagicMock(returncode=0, stdout="abc1234\n")

        # Create the dirs that would be created by git clone / venv
        os.makedirs(installer.app_dir)
        os.makedirs(installer.venv_dir)
        os.makedirs(os.path.join(installer.app_dir, "custom_nodes"))
        os.makedirs(os.path.join(installer.app_dir, "custom_nodes", "ComfyUI-Manager"))

        # Create a fake main.py
        with open(os.path.join(installer.app_dir, "main.py"), "w") as f:
            f.write("# main")

        installer.install()

        # Should have updated state
        app_state = state.get_app("comfyui")
        assert app_state is not None
        assert app_state["status"] == "installed"

        # Launch script should exist
        assert os.path.isfile(installer.launch_script)

    def test_create_launch_script(self, installer):
        """Test launch script creation."""
        os.makedirs(installer.state.bin_dir, exist_ok=True)
        # Need venv dir for the path
        os.makedirs(os.path.join(installer.venv_dir, "bin"), exist_ok=True)
        os.makedirs(installer.app_dir, exist_ok=True)

        installer._create_launch_script()

        assert os.path.isfile(installer.launch_script)
        with open(installer.launch_script) as f:
            content = f.read()
        assert "#!/bin/bash" in content
        # Binds to loopback only; reach it via an SSH tunnel (cloudgpu forward).
        assert "--listen 127.0.0.1" in content
        assert "--port 8188" in content
        assert os.access(installer.launch_script, os.X_OK)

    def test_verify_not_installed(self, installer):
        assert installer.verify() is False
