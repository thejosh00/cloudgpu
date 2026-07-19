"""Tests for AI Toolkit installer."""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest

from cloudgpu.remote.apps.aitoolkit import AIToolkitInstaller, AITOOLKIT_PORT
from cloudgpu.remote.state import State


class TestAIToolkitInstaller:
    @pytest.fixture
    def state(self, tmp_persistent_dir):
        return State(tmp_persistent_dir)

    @pytest.fixture
    def installer(self, state):
        return AIToolkitInstaller(state)

    def test_name(self, installer):
        assert installer.name == "ai-toolkit"

    def test_paths(self, installer, tmp_persistent_dir):
        assert installer.app_dir == os.path.join(tmp_persistent_dir, "apps", "ai-toolkit")
        assert installer.venv_dir == os.path.join(tmp_persistent_dir, "venvs", "ai-toolkit")
        assert installer.ui_dir == os.path.join(installer.app_dir, "ui")
        assert installer.launch_script == os.path.join(
            tmp_persistent_dir, "cloudgpu", "bin", "ai-toolkit"
        )
        # Node install is arch-suffixed so both GPU arches can share one NFS fs
        assert os.path.basename(installer.node_dir) == f"node-{installer._arch()}"
        assert installer._arch() in ("arm64", "x64")

    def test_service_spec(self, installer):
        spec = installer.service_spec()
        assert spec["name"] == "ai-toolkit"
        assert spec["exec_start"] == installer.launch_script
        assert spec["workdir"] == installer.ui_dir
        assert spec["port"] == 8675

    @patch("cloudgpu.remote.apps.aitoolkit.service_active", return_value="inactive")
    def test_get_status_not_installed(self, mock_active, installer):
        status = installer.get_status()
        assert status["status"] == "broken"
        assert status["torch_cuda"] is False
        assert status["ui_built"] is False
        assert status["service"] == "inactive"
        assert status["port"] == AITOOLKIT_PORT

    @patch.object(AIToolkitInstaller, "_build_ui")
    @patch.object(AIToolkitInstaller, "_ensure_node")
    @patch("cloudgpu.remote.apps.aitoolkit.install_service")
    @patch("cloudgpu.remote.apps.aitoolkit.run")
    @patch("cloudgpu.remote.apps.aitoolkit.pip_install")
    @patch("cloudgpu.remote.apps.aitoolkit.check_torch_cuda", return_value=True)
    def test_install(self, mock_torch, mock_pip, mock_run, mock_service,
                     mock_node, mock_build, installer, state):
        """Install records state, creates the launch script and venv symlink."""
        mock_run.return_value = MagicMock(returncode=0, stdout="abc1234\n")

        # Dirs that git clone / venv creation would have made
        os.makedirs(os.path.join(installer.app_dir, "ui"))
        os.makedirs(installer.venv_dir)
        with open(os.path.join(installer.app_dir, "run.py"), "w") as f:
            f.write("# run")
        with open(os.path.join(installer.app_dir, "requirements.txt"), "w") as f:
            f.write("pyyaml\n")

        installer.install()

        app_state = state.get_app("ai-toolkit")
        assert app_state is not None
        assert app_state["status"] == "installed"
        assert app_state["node_arch"] == installer._arch()

        assert os.path.isfile(installer.launch_script)
        venv_link = os.path.join(installer.app_dir, "venv")
        assert os.path.islink(venv_link)
        assert os.readlink(venv_link) == installer.venv_dir
        mock_node.assert_called_once()
        mock_build.assert_called_once()

    def test_create_launch_script(self, installer):
        """The launch script must pin the UI to loopback and run the worker."""
        os.makedirs(installer.state.bin_dir, exist_ok=True)
        os.makedirs(os.path.join(installer.venv_dir, "bin"), exist_ok=True)
        os.makedirs(installer.ui_dir, exist_ok=True)

        installer._create_launch_script()

        assert os.path.isfile(installer.launch_script)
        with open(installer.launch_script) as f:
            content = f.read()
        assert "#!/bin/bash" in content
        # Binds to loopback only; reach it via an SSH tunnel (cloudgpu forward).
        assert "--hostname 127.0.0.1" in content
        assert "--port 8675" in content
        assert "dist/cron/worker.js" in content
        assert installer.node_bin in content
        # model downloads must land on the persistent fs, not the ephemeral root
        assert "export HF_HOME=" in content
        assert os.path.join(installer.venv_dir, "bin") in content
        assert os.access(installer.launch_script, os.X_OK)

    @patch.object(AIToolkitInstaller, "_build_ui")
    @patch.object(AIToolkitInstaller, "_ensure_node")
    @patch("cloudgpu.remote.apps.aitoolkit.install_service")
    @patch("cloudgpu.remote.apps.aitoolkit.run")
    @patch("cloudgpu.remote.apps.aitoolkit.pip_install")
    @patch("cloudgpu.remote.apps.aitoolkit.check_torch_cuda", return_value=True)
    def test_recover_arch_change_rebuilds_node_modules(
            self, mock_torch, mock_pip, mock_run, mock_service,
            mock_node, mock_build, installer, state):
        """An arch switch wipes per-arch node_modules and rebuilds the UI."""
        mock_run.return_value = MagicMock(returncode=0, stdout="abc1234\n")
        node_modules = os.path.join(installer.ui_dir, "node_modules")
        os.makedirs(node_modules)
        os.makedirs(installer.venv_dir)
        state.set_app("ai-toolkit", {"status": "installed", "node_arch": "other-arch"})

        installer.recover()

        assert not os.path.isdir(node_modules)
        mock_build.assert_called_once()
        assert state.get_app("ai-toolkit")["node_arch"] == installer._arch()

    @patch.object(AIToolkitInstaller, "_ui_built", return_value=True)
    @patch.object(AIToolkitInstaller, "_build_ui")
    @patch.object(AIToolkitInstaller, "_ensure_node")
    @patch("cloudgpu.remote.apps.aitoolkit.install_service")
    @patch("cloudgpu.remote.apps.aitoolkit.run")
    @patch("cloudgpu.remote.apps.aitoolkit.check_torch_cuda", return_value=True)
    def test_recover_same_arch_skips_ui_build(
            self, mock_torch, mock_run, mock_service,
            mock_node, mock_build, mock_built, installer, state):
        """Normal instance replacement: everything on NFS, no npm rebuild."""
        mock_run.return_value = MagicMock(returncode=0, stdout="abc1234\n")
        os.makedirs(os.path.join(installer.ui_dir, "node_modules"))
        os.makedirs(installer.venv_dir)
        state.set_app("ai-toolkit", {"status": "installed", "node_arch": installer._arch()})

        installer.recover()

        mock_build.assert_not_called()

    def test_verify_not_installed(self, installer):
        assert installer.verify() is False
