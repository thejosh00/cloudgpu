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

    @patch("cloudgpu.remote.apps.aitoolkit.run")
    def test_npm_install_accepts_a_complete_tree(self, mock_run, installer):
        mock_run.return_value = MagicMock(returncode=0)

        installer._npm_install("npm", {"PATH": "/bin"})

        # install + verification only, no reinstall
        assert [c[0][0][:2] for c in mock_run.call_args_list] == [
            ["npm", "install"], ["npm", "ls"],
        ]

    @patch("cloudgpu.remote.apps.aitoolkit.run")
    def test_npm_install_retries_a_half_written_tree(self, mock_run, installer):
        """Cold-cache npm extraction races on NFS and exits 0 anyway."""
        node_modules = os.path.join(installer.ui_dir, "node_modules")
        os.makedirs(node_modules)
        mock_run.side_effect = [
            MagicMock(returncode=0),  # npm install
            MagicMock(returncode=1),  # npm ls: invalid packages
            MagicMock(returncode=0),  # npm install (warm cache)
            MagicMock(returncode=0),  # npm ls: clean
        ]

        installer._npm_install("npm", {"PATH": "/bin"})

        assert not os.path.isdir(node_modules)  # wiped before the retry
        assert mock_run.call_count == 4

    @patch("cloudgpu.remote.apps.aitoolkit.run")
    def test_npm_install_raises_if_still_incomplete(self, mock_run, installer):
        mock_run.return_value = MagicMock(returncode=1)

        with pytest.raises(RuntimeError, match="node_modules"):
            installer._npm_install("npm", {"PATH": "/bin"})

    @patch("cloudgpu.remote.apps.aitoolkit.run")
    def test_ensure_sqlite3_skips_rebuild_when_loadable(self, mock_run, installer):
        mock_run.return_value = MagicMock(returncode=0)

        installer._ensure_sqlite3("npm", {"PATH": "/bin"})

        # Only the probe ran, no rebuild
        assert mock_run.call_count == 1
        assert "-e" in mock_run.call_args[0][0]

    @patch("cloudgpu.remote.apps.aitoolkit.run")
    def test_ensure_sqlite3_rebuilds_from_source(self, mock_run, installer):
        """glibc-incompatible prebuild is recompiled locally, then re-probed."""
        mock_run.side_effect = [
            MagicMock(returncode=1),  # probe fails
            MagicMock(returncode=0),  # npm rebuild
            MagicMock(returncode=0),  # probe passes
        ]

        installer._ensure_sqlite3("npm", {"PATH": "/bin"})

        rebuild = mock_run.call_args_list[1]
        assert rebuild[0][0] == ["npm", "rebuild", "sqlite3", "--build-from-source"]
        assert rebuild[1]["env"]["npm_config_build_from_source"] == "true"

    @patch("cloudgpu.remote.apps.aitoolkit.run")
    def test_ensure_sqlite3_raises_if_still_broken(self, mock_run, installer):
        mock_run.return_value = MagicMock(returncode=1)

        with pytest.raises(RuntimeError, match="sqlite3"):
            installer._ensure_sqlite3("npm", {"PATH": "/bin"})

    def test_patched_requirements_drops_torchcodec_pin(self, installer):
        """The pin lives in an included file and must not reach pip.

        torchcodec ships no Linux arm64 wheel below 0.10, so upstream's
        0.9.1 pin fails the whole requirements pass on a GH200.
        """
        os.makedirs(installer.app_dir)
        with open(os.path.join(installer.app_dir, "requirements_base.txt"), "w") as f:
            f.write("av==16.0.1\ntorchcodec==0.9.1\nlibrosa==0.11.0\n")
        requirements = os.path.join(installer.app_dir, "requirements.txt")
        with open(requirements, "w") as f:
            f.write("-r requirements_base.txt\nscipy==1.12.0\n")

        with open(installer._patched_requirements(requirements)) as f:
            lines = [ln for ln in f.read().splitlines() if ln and not ln.startswith("#")]

        assert lines == ["av==16.0.1", "librosa==0.11.0", "scipy==1.12.0"]

    @patch("cloudgpu.remote.apps.aitoolkit.run")
    @patch("cloudgpu.remote.apps.aitoolkit.pip_install")
    @patch("cloudgpu.remote.apps.aitoolkit.torch_index_url",
           return_value="https://download.pytorch.org/whl/cu128")
    def test_install_torchcodec_uses_torch_index(self, mock_index, mock_pip,
                                                 mock_run, installer):
        """PyPI ships cu13 builds only; torchcodec must match our torch."""
        mock_run.return_value = MagicMock(returncode=0)

        installer._install_torchcodec()

        args = mock_pip.call_args[0][1:]
        assert "torchcodec" in args
        assert args[-2:] == ("--index-url", "https://download.pytorch.org/whl/cu128")

    @patch("cloudgpu.remote.apps.aitoolkit.run")
    @patch("cloudgpu.remote.apps.aitoolkit.pip_install",
           side_effect=RuntimeError("no matching distribution"))
    @patch("cloudgpu.remote.apps.aitoolkit.torch_index_url", return_value=None)
    def test_install_torchcodec_failure_is_not_fatal(self, mock_index, mock_pip,
                                                     mock_run, installer):
        """Only video/audio datasets need it — never fail the install for it."""
        installer._install_torchcodec()

        mock_run.assert_not_called()  # no import probe after a failed install

    def test_verify_not_installed(self, installer):
        assert installer.verify() is False
