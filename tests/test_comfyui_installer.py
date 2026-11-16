"""Tests for ComfyUI installer."""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest

from cloudgpu.remote.apps.comfyui import ComfyUIInstaller
from cloudgpu.remote.state import State


class TestComfyUIInstaller:
    @pytest.fixture(autouse=True)
    def no_real_sudo(self):
        module = "cloudgpu.remote.apps.comfyui"
        with (patch(f"{module}.prepare_venv_checks"), patch(f"{module}.prepare_app_dir"),
              patch(f"{module}.trust_repo_as_app"), patch(f"{module}.git_pull_as_app"),
              patch(f"{module}.git_version_as_app", return_value="abc1234")):
            yield

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


def _fake_clone(files):
    """A stand-in for ``run`` whose ``git clone <repo> <dest>`` leaves a small checkout."""
    clones = []

    def run(cmd, **kwargs):
        if cmd[:2] == ["git", "clone"]:
            dest = cmd[-1]
            clones.append(cmd)
            os.makedirs(os.path.join(dest, ".git"))
            for rel, text in files.items():
                os.makedirs(os.path.dirname(os.path.join(dest, rel)), exist_ok=True)
                with open(os.path.join(dest, rel), "w") as f:
                    f.write(text)
        return MagicMock(returncode=0, stdout="abc1234\n")

    run.clones = clones
    return run


class TestInstallAfterARestore:
    """3.3: a restore puts apps/comfyui/output there before ComfyUI was ever installed."""

    @pytest.fixture(autouse=True)
    def no_real_sudo(self):
        module = "cloudgpu.remote.apps.comfyui"
        with (patch(f"{module}.prepare_venv_checks"), patch(f"{module}.prepare_app_dir") as prepare,
              patch(f"{module}.trust_repo_as_app") as trust, patch(f"{module}.git_pull_as_app") as pull,
              patch(f"{module}.git_version_as_app", return_value="abc1234"),
              patch(f"{module}.install_service"), patch(f"{module}.pip_install"),
              patch(f"{module}.check_torch_cuda", return_value=True),
              patch(f"{module}.app_step"),
              patch.object(ComfyUIInstaller, "_install_requirements"),
              patch.object(ComfyUIInstaller, "_ensure_torch")):
            self.prepare, self.trust, self.pull = prepare, trust, pull
            yield

    @pytest.fixture
    def installer(self, tmp_persistent_dir):
        return ComfyUIInstaller(State(tmp_persistent_dir))

    REPO_FILES = {"main.py": "# main", "requirements.txt": "torch\n",
                  "output/_output_images_will_be_put_here": "",
                  "input/example.png": "the repo's example"}

    def _restored(self, installer):
        out = os.path.join(installer.app_dir, "output")
        os.makedirs(os.path.join(out, "2026-11"))
        os.makedirs(os.path.join(installer.app_dir, "input"))
        files = {os.path.join(out, "2026-11", "render.png"): "my render",
                 os.path.join(out, "a.png"): "my image",
                 os.path.join(installer.app_dir, "input", "example.png"): "my own example"}
        for path, text in files.items():
            with open(path, "w") as f:
                f.write(text)
        return files

    def test_install_over_restored_directories_still_clones_and_keeps_the_files(self, installer):
        restored = self._restored(installer)
        run = _fake_clone(self.REPO_FILES)
        with patch("cloudgpu.remote.apps.comfyui.run", run):
            installer.install()
        assert len(run.clones) == 1  # the clone was not skipped
        assert os.path.isdir(os.path.join(installer.app_dir, ".git"))
        assert os.path.isfile(os.path.join(installer.app_dir, "main.py"))
        for path, text in restored.items():  # every restored file, byte for byte
            with open(path) as f:
                assert f.read() == text
        # the repo's own file in a restored directory arrives where nothing was in its way
        assert os.path.isfile(os.path.join(installer.app_dir, "output", "_output_images_will_be_put_here"))
        # treated as a first install, not an update of an existing one
        self.trust.assert_called_once_with(installer.app_dir)
        self.pull.assert_not_called()
        self.prepare.assert_called_once_with(installer.app_dir)
        assert installer.state.get_app("comfyui")["status"] == "installed"
        assert not os.path.exists(installer.app_dir + ".clone")  # nothing left beside it

    def test_install_on_an_empty_filesystem_clones_straight_into_place(self, installer):
        run = _fake_clone(self.REPO_FILES)
        with patch("cloudgpu.remote.apps.comfyui.run", run):
            installer.install()
        assert [c[-1] for c in run.clones] == [installer.app_dir]

    def test_an_existing_install_is_updated_not_cloned_again(self, installer):
        self._restored(installer)
        with open(os.path.join(installer.app_dir, "main.py"), "w") as f:
            f.write("# main")
        run = _fake_clone(self.REPO_FILES)
        with patch("cloudgpu.remote.apps.comfyui.run", run):
            installer.install()
        assert run.clones == []
        self.pull.assert_any_call(installer.app_dir)

    def test_a_clone_interrupted_while_moving_in_is_done_again(self, installer):
        self._restored(installer)
        os.makedirs(installer.app_dir + ".clone/.git")  # left by a killed install
        with open(os.path.join(installer.app_dir, "main.py"), "w") as f:
            f.write("# moved in before the kill")
        run = _fake_clone(self.REPO_FILES)
        with patch("cloudgpu.remote.apps.comfyui.run", run):
            installer.install()
        assert len(run.clones) == 1 and os.path.isdir(os.path.join(installer.app_dir, ".git"))
        assert not os.path.exists(installer.app_dir + ".clone")

    def test_recover_with_only_restored_directories_does_a_full_install(self, installer):
        restored = self._restored(installer)
        run = _fake_clone(self.REPO_FILES)
        with patch("cloudgpu.remote.apps.comfyui.run", run):
            installer.recover()
        assert len(run.clones) == 1
        assert all(os.path.isfile(p) for p in restored)

    @patch("cloudgpu.remote.apps.comfyui.service_active", return_value="inactive")
    def test_status_with_only_restored_directories_is_not_installed(self, _active, installer):
        self._restored(installer)
        os.makedirs(installer.venv_dir)
        with patch("cloudgpu.remote.apps.comfyui.torch_cuda_status", return_value=True):
            status = installer.get_status()
        assert status["status"] == "broken" and status["version"] is None
