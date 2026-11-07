"""Provisioning runs venv binaries and custom-node git as cloudgpu-app, with only named tokens.

cloudgpu-app can write the venvs and app_dir, and ubuntu has passwordless sudo and every secret
during provisioning. These tests use no real sudo or instance and made-up token values: a fake
sudo first on PATH records every call, and the planted pip, python and git hook record who ran
them. The fake sudo deliberately doesn't clean the environment, so a token only stays out of a
step if the code keeps it out. It does apply sudo's default umask of 022.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

import cloudgpu.remote
from cloudgpu import cli
from cloudgpu.local import profiles, ssh, sync
from cloudgpu.remote import utils
from cloudgpu.remote.apps.aitoolkit import AIToolkitInstaller
from cloudgpu.remote.service import unit_text
from cloudgpu.remote.state import State
from cloudgpu.templates.comfyui import comfylib

APP_USER = "cloudgpu-app"
HELPER = Path(cloudgpu.remote.__file__).parent / "run_as_app.py"
COULD_NOT_RUN = 125

HF = "made-up-hf-value"
CIVITAI = "made-up-civitai-value"

FAKE_SUDO = """#!/bin/bash
S={sandbox}
echo "sudo $*" >> "$S/calls.log"
[ -e "$S/user-missing" ] && {{ echo "sudo: unknown user {user}" >&2; exit 1; }}
[ -e "$S/sudo-refuses" ] && {{ echo "sudo: a password is required" >&2; exit 1; }}
target=
while [ $# -gt 0 ]; do
  case "$1" in
    -u) target="$2"; shift 2 ;;
    --) shift; break ;;
    -*) shift ;;
    *) break ;;
  esac
done
umask 022
[ -e "$S/drop-stdin" ] && exec </dev/null
HOME="$S/app-home" FAKE_SUDO_TARGET="$target" exec "$@"
"""

# What app code would plant. Records who ran it and what it was given, then does nothing else.
PLANTED = """#!/bin/bash
S={sandbox}
echo "{label} ran as ${{FAKE_SUDO_TARGET:-the caller, directly}} in $PWD" >> "$S/calls.log"
echo "{label} argv: $*" >> "$S/calls.log"
env > "$S/{label}.env"
cat > "$S/{label}.stdin" 2>/dev/null
touch "$S/{label}.wrote"
exit {exit_code}
"""


# First on PATH: records who ran git, then runs the real one.
GIT_SHIM = """#!/bin/bash
echo "git ran as ${{FAKE_SUDO_TARGET:-the caller, directly}}: $*" >> {sandbox}/calls.log
exec {real_git} "$@"
"""
REAL_GIT = shutil.which("git")


def executable(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


class Sandbox:
    def __init__(self, root: Path):
        self.root = root
        (root / "app-home").mkdir()
        executable(root / "fake-bin" / "sudo", FAKE_SUDO.format(sandbox=root, user=APP_USER))
        executable(root / "fake-bin" / "git", GIT_SHIM.format(sandbox=root, real_git=REAL_GIT))
        self.path = f"{root / 'fake-bin'}:{os.environ['PATH']}"

    @property
    def log(self) -> str:
        path = self.root / "calls.log"
        return path.read_text() if path.exists() else ""

    def plant(self, path: Path, label: str, exit_code: int = 0) -> Path:
        return executable(path, PLANTED.format(sandbox=self.root, label=label, exit_code=exit_code))

    def flag(self, name: str) -> None:
        (self.root / name).touch()

    def assert_only_as_app(self, label: str) -> None:
        ran = [line for line in self.log.splitlines() if f"{label} ran as" in line]
        assert ran, f"{label} never ran:\n{self.log}"
        assert all(f"ran as {APP_USER}" in line for line in ran), self.log

    def assert_never_ran(self, label: str) -> None:
        assert f"{label} ran as" not in self.log, self.log


@pytest.fixture
def sandbox(tmp_path):
    root = tmp_path / "sandbox"
    root.mkdir()
    return Sandbox(root)


# --- the helper profile scripts use ----------------------------------------------------------


@pytest.fixture
def step(sandbox, tmp_path):
    return sandbox.plant(tmp_path / "venv" / "bin" / "python3", "planted")


def helper(sandbox, *args, env=None, path=None):
    return subprocess.run(
        [sys.executable, str(HELPER), *args],
        env={"PATH": path or sandbox.path, **(env or {})},
        capture_output=True, text=True, cwd=sandbox.root,
    )


class TestHelper:
    def test_step_reached_only_through_sudo_as_app(self, sandbox, step):
        result = helper(sandbox, "--", str(step), "-c", "code")
        assert result.returncode == 0, result.stderr
        sandbox.assert_only_as_app("planted")
        assert f"sudo -n -H -u {APP_USER} -- " in sandbox.log
        assert "planted argv: -c code" in sandbox.log

    def test_runs_from_root_not_the_callers_directory(self, sandbox, step):
        helper(sandbox, "--", str(step))
        assert "ran as cloudgpu-app in /\n" in sandbox.log

    def test_named_token_reaches_step(self, sandbox, step):
        helper(sandbox, "--pass", "HF_TOKEN", "--", str(step), env={"HF_TOKEN": HF})
        assert f"HF_TOKEN={HF}\n" in (sandbox.root / "planted.env").read_text()

    def test_unnamed_token_does_not_reach_step(self, sandbox, step):
        helper(sandbox, "--pass", "HF_TOKEN", "--", str(step),
               env={"HF_TOKEN": HF, "CIVITAI_TOKEN": CIVITAI})
        assert CIVITAI not in (sandbox.root / "planted.env").read_text()

    def test_no_token_value_in_any_argv(self, sandbox, step):
        result = helper(sandbox, "--pass", "HF_TOKEN", "--", str(step),
                        env={"HF_TOKEN": HF, "CIVITAI_TOKEN": CIVITAI})
        assert result.returncode == 0, result.stderr
        assert sandbox.log  # the log holds every argv sudo and the step were given
        assert HF not in sandbox.log and CIVITAI not in sandbox.log

    def test_step_cant_read_more_from_stdin(self, sandbox, step):
        helper(sandbox, "--pass", "HF_TOKEN", "--", str(step), env={"HF_TOKEN": HF})
        assert (sandbox.root / "planted.stdin").read_text() == ""

    def test_value_arrives_literally(self, sandbox, step):
        value = "made up $HOME `id` 'quoted' \"too\" = value"
        result = helper(sandbox, "--pass", "ODD", "--", str(step), env={"ODD": value})
        assert result.returncode == 0, result.stderr
        assert f"ODD={value}\n" in (sandbox.root / "planted.env").read_text()

    def test_several_names_including_non_secrets(self, sandbox, step):
        helper(sandbox, "--pass", "HF_TOKEN", "--pass", "HF_HOME", "--", str(step),
               env={"HF_TOKEN": HF, "HF_HOME": "/cache"})
        seen = (sandbox.root / "planted.env").read_text()
        assert f"HF_TOKEN={HF}\n" in seen and "HF_HOME=/cache\n" in seen

    @pytest.mark.parametrize("env", [{}, {"HF_TOKEN": ""}])
    def test_named_token_not_set_stops_the_step(self, sandbox, step, env):
        result = helper(sandbox, "--pass", "HF_TOKEN", "--", str(step), env=env)
        assert result.returncode == COULD_NOT_RUN
        assert "HF_TOKEN" in result.stderr
        sandbox.assert_never_ran("planted")

    def test_token_with_newline_stops_the_step_without_printing_it(self, sandbox, step):
        result = helper(sandbox, "--pass", "BAD", "--", str(step), env={"BAD": "made\nup"})
        assert result.returncode == COULD_NOT_RUN
        assert "made" not in result.stderr
        sandbox.assert_never_ran("planted")

    def test_tokens_that_dont_arrive_stop_the_step(self, sandbox, step):
        sandbox.flag("drop-stdin")
        result = helper(sandbox, "--pass", "HF_TOKEN", "--", str(step), env={"HF_TOKEN": HF})
        assert result.returncode == COULD_NOT_RUN
        assert "didn't arrive" in result.stderr
        sandbox.assert_never_ran("planted")

    def test_user_missing_fails_clearly_and_never_runs_it(self, sandbox, step):
        sandbox.flag("user-missing")
        result = helper(sandbox, "--", str(step))
        assert result.returncode == COULD_NOT_RUN
        assert APP_USER in result.stderr and "unknown user" in result.stderr
        sandbox.assert_never_ran("planted")

    def test_sudo_refusing_fails_clearly_and_never_runs_it(self, sandbox, step):
        sandbox.flag("sudo-refuses")
        result = helper(sandbox, "--", str(step))
        assert result.returncode == COULD_NOT_RUN
        assert "a password is required" in result.stderr
        sandbox.assert_never_ran("planted")

    def test_no_sudo_at_all_fails_clearly_and_never_runs_it(self, sandbox, step, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = helper(sandbox, "--", str(step), path=str(empty))
        assert result.returncode == COULD_NOT_RUN
        assert APP_USER in result.stderr
        sandbox.assert_never_ran("planted")

    def test_steps_own_exit_code_comes_back(self, sandbox, tmp_path):
        failing = sandbox.plant(tmp_path / "bin" / "failing", "failing", exit_code=3)
        assert helper(sandbox, "--", str(failing)).returncode == 3

    def test_files_stay_group_writable_despite_sudos_umask(self, sandbox, step):
        helper(sandbox, "--", str(step))
        mode = (sandbox.root / "planted.wrote").stat().st_mode
        assert mode & stat.S_IWGRP, "a later pip_install as ubuntu couldn't change this file"

    @pytest.mark.parametrize("args", [[], ["--pass", "X"], ["--"]])
    def test_no_step_is_a_usage_error(self, sandbox, args):
        assert helper(sandbox, *args).returncode != 0
        assert sandbox.log == ""


# --- custom_node -----------------------------------------------------------------------------


def git(*args, cwd=None):
    subprocess.run([REAL_GIT, "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                   cwd=cwd, check=True, capture_output=True,
                   env={"PATH": os.environ["PATH"], "HOME": "/nonexistent",
                        "GIT_CONFIG_NOSYSTEM": "1"})


class Node:
    """A ComfyUI install with a custom node's origin, and a planted venv pip."""

    def __init__(self, sandbox: Sandbox, tmp_path: Path, monkeypatch):
        self.sandbox = sandbox
        apps, venvs = tmp_path / "apps", tmp_path / "venvs"
        self.origin = tmp_path / "origin" / "node"
        self.origin.mkdir(parents=True)
        git("init", "-q", "-b", "main", cwd=self.origin)
        (self.origin / "requirements.txt").write_text("somepackage\n")
        git("add", ".", cwd=self.origin)
        git("commit", "-qm", "first", cwd=self.origin)
        self.url = f"file://{self.origin}"
        self.dest = apps / "comfyui" / "custom_nodes" / "node"
        self.pip = venvs / "comfyui" / "bin" / "pip"
        sandbox.plant(self.pip, "pip")

        monkeypatch.delenv("COMFYUI_DIR", raising=False)
        monkeypatch.setenv("CLOUDGPU_APPS_DIR", str(apps))
        monkeypatch.setenv("CLOUDGPU_VENVS_DIR", str(venvs))
        monkeypatch.setenv("CLOUDGPU_RUN_AS_APP", str(HELPER))
        monkeypatch.setenv("PATH", sandbox.path)
        monkeypatch.setenv("HF_TOKEN", HF)
        monkeypatch.setenv("CIVITAI_TOKEN", CIVITAI)

    def clone_with_planted_hook(self) -> None:
        """An existing clone, a planted hook that fires on the next pull, and a commit to pull."""
        self.dest.parent.mkdir(parents=True)
        git("clone", "-q", self.url, str(self.dest))
        self.sandbox.plant(self.dest / ".git" / "hooks" / "post-merge", "hook")
        (self.origin / "new.txt").write_text("new\n")
        git("add", ".", cwd=self.origin)
        git("commit", "-qm", "second", cwd=self.origin)


@pytest.fixture
def node(sandbox, tmp_path, monkeypatch):
    return Node(sandbox, tmp_path, monkeypatch)


class TestCustomNode:
    def test_clone_and_pip_only_through_sudo_as_app(self, node, sandbox):
        assert comfylib.custom_node(node.url) == node.dest
        assert (node.dest / "requirements.txt").exists()
        sandbox.assert_only_as_app("git")
        assert f"git clone --depth 1 {node.url} {node.dest}" in sandbox.log
        sandbox.assert_only_as_app("pip")
        assert f"pip argv: install -r {node.dest}/requirements.txt" in sandbox.log

    def test_planted_hook_and_pip_on_update_only_as_app(self, node, sandbox):
        node.clone_with_planted_hook()
        comfylib.custom_node(node.url)
        assert (node.dest / "new.txt").exists(), "the pull didn't happen"
        sandbox.assert_only_as_app("git")
        sandbox.assert_only_as_app("hook")
        sandbox.assert_only_as_app("pip")

    def test_existing_clone_is_trusted_before_the_pull(self, node, sandbox):
        # A clone made as ubuntu is "dubious ownership" to cloudgpu-app, and git would refuse it.
        node.clone_with_planted_hook()
        comfylib.custom_node(node.url)
        trust = f"git config --global --add safe.directory {node.dest}"
        pull = f"git -C {node.dest} pull --ff-only"
        assert sandbox.log.index(trust) < sandbox.log.index(pull)
        config = (sandbox.root / "app-home" / ".gitconfig").read_text()
        assert str(node.dest) in config

    def test_no_token_reaches_pip_or_a_hook(self, node, sandbox):
        node.clone_with_planted_hook()
        comfylib.custom_node(node.url)
        for label in ("pip", "hook"):
            seen = (sandbox.root / f"{label}.env").read_text()
            assert HF not in seen and CIVITAI not in seen, label

    @pytest.mark.parametrize("flag", ["user-missing", "sudo-refuses"])
    def test_clone_cant_run_as_app(self, node, sandbox, capfd, flag):
        sandbox.flag(flag)
        with pytest.raises(SystemExit) as e:
            comfylib.custom_node(node.url)
        assert e.value.code != 0
        assert APP_USER in capfd.readouterr().err
        assert not node.dest.exists()

    @pytest.mark.parametrize("flag", ["user-missing", "sudo-refuses"])
    def test_update_cant_run_as_app(self, node, sandbox, capfd, flag):
        node.clone_with_planted_hook()
        sandbox.flag(flag)
        with pytest.raises(SystemExit) as e:
            comfylib.custom_node(node.url)
        assert e.value.code != 0
        assert APP_USER in capfd.readouterr().err
        sandbox.assert_never_ran("hook")
        sandbox.assert_never_ran("pip")

    def test_helper_not_available_never_runs_it_directly(self, node, sandbox, monkeypatch, capfd):
        node.clone_with_planted_hook()
        monkeypatch.delenv("CLOUDGPU_RUN_AS_APP")
        with pytest.raises(SystemExit) as e:
            comfylib.custom_node(node.url)
        assert e.value.code != 0
        assert "CLOUDGPU_RUN_AS_APP" in capfd.readouterr().err
        sandbox.assert_never_ran("hook")
        sandbox.assert_never_ran("pip")

    def test_failed_pull_exits_non_zero(self, node, sandbox):
        node.clone_with_planted_hook()
        subprocess.run(["rm", "-rf", str(node.origin)], check=True)
        with pytest.raises(SystemExit) as e:
            comfylib.custom_node(node.url)
        assert e.value.code != 0
        sandbox.assert_never_ran("pip")

    def test_failed_clone_exits_non_zero(self, node):
        with pytest.raises(SystemExit) as e:
            comfylib.custom_node(f"{node.url}-not-there", name="node")
        assert e.value.code != 0

    def test_failed_pip_exits_non_zero(self, node, sandbox):
        sandbox.plant(node.pip, "pip", exit_code=3)
        with pytest.raises(SystemExit) as e:
            comfylib.custom_node(node.url)
        assert e.value.code != 0


# --- how provisioning reaches the helper -----------------------------------------------------


class TestProvisioningPath:
    def test_entry_point_is_told_where_the_helper_is(self, tmp_path, tmp_config_dir, monkeypatch):
        d = tmp_path / "prof"
        d.mkdir()
        (d / "cloudgpu.toml").write_text('gpu = ["gh200"]\nssh_key = "mini"\n')
        (d / "provision.py").write_text("print('hi')\n")
        monkeypatch.setattr(cli.sync, "copy_dir", lambda *a, **k: None)
        captured = {}
        monkeypatch.setattr(cli.ssh, "ssh_run",
                            lambda host, command, **k: captured.update(command=command)
                            or ssh.SSHResult(0, "", ""))
        cli._run_provision(profiles.load_profile(d), "ubuntu@h", "/lambda/nfs/p")
        assert "CLOUDGPU_RUN_AS_APP=/lambda/nfs/p/cloudgpu/remote/run_as_app.py " in captured["command"]

    def test_helper_is_in_the_directory_that_gets_synced(self):
        assert (sync.get_remote_dir() / "run_as_app.py").is_file()

    def test_helper_targets_the_user_the_services_run_as(self):
        assert utils.APP_USER == APP_USER

    def test_apps_are_ensured_before_any_provisioning(self, monkeypatch):
        # Installing or recovering an app is what creates cloudgpu-app, on `up` and `provision`.
        order = []
        monkeypatch.setattr(cli, "_ensure_apps", lambda *a: order.append("ensure apps"))
        monkeypatch.setattr(cli, "_run_provision", lambda *a: order.append("provision"))
        monkeypatch.setattr(cli, "_restart_services", lambda *a: None)
        monkeypatch.setattr(cli.configs, "save_state", lambda *a: None)
        cli._provision_profiles("c", {}, "ubuntu@h", "/p", [{"name": "lora", "apps": ["ai-toolkit"]}])
        assert order == ["ensure apps", "provision"]


class TestHfCache:
    """lora's snapshot writes <persistent>/hf_cache, now as cloudgpu-app."""

    @pytest.fixture
    def installer(self, tmp_persistent_dir):
        return AIToolkitInstaller(State(tmp_persistent_dir))

    def test_cache_is_in_the_services_writable_paths(self, installer, tmp_persistent_dir):
        hf_cache = os.path.join(tmp_persistent_dir, "hf_cache")
        assert hf_cache in installer.service_spec()["rw_paths"]
        assert hf_cache in unit_text(installer.service_spec())

    def test_cache_exists_before_the_unit_needs_it(self, installer, monkeypatch):
        # install_service shares rw_paths with the group, and systemd won't start a unit whose
        # ReadWritePaths doesn't exist.
        seen = {}
        monkeypatch.setattr("cloudgpu.remote.apps.aitoolkit.install_service",
                            lambda spec: seen.update(exists=os.path.isdir(installer.hf_home)))
        installer._setup_service()
        assert seen == {"exists": True}

    def test_launcher_uses_the_same_cache(self, installer):
        os.makedirs(installer.ui_dir, exist_ok=True)
        installer._create_launch_script()
        with open(installer.launch_script) as f:
            assert f"export HF_HOME={installer.hf_home}\n" in f.read()
