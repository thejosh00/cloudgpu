"""App records and re-provision marks stay true (plan item 0.13).

Four paths left state.json saying the wrong thing, so the next `up` skipped work it should
redo or reported a broken app as fine:
  1. a failed re-install of an already-recorded app kept its old record;
  2. a recorded app with no installer was only a warning on recovery;
  3. only `up` cleared re-provision marks (`recover` and `provision` ignored them);
  4. an ai-toolkit venv rebuild set no mark.

Fakes only: every host-changing step is stubbed in-process (and subprocess refuses to run),
the CLI tests stub ssh/rsync in-process, and the `up` test uses the fake Lambda with ssh shimmed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from contextlib import ExitStack
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from cloudgpu import cli
from cloudgpu.local import configs
from cloudgpu.remote import __main__ as remote_main
from cloudgpu.remote import recovery
from cloudgpu.remote.apps.aitoolkit import AIToolkitInstaller
from cloudgpu.remote.apps.comfyui import ComfyUIInstaller
from cloudgpu.remote.state import State
from tests.test_comfyui_rebuild_torch import comfy, host  # noqa: F401  (comfy is a fixture)
from tests.test_up_recovery import BOTH, _clears, _provisions, _remote, _up_after_rebuild


def _no_real_commands(*args, **kwargs):
    raise AssertionError(f"a real command was about to run: {args[0] if args else kwargs}")


# --- 1. a failed re-install of a recorded app makes the next recover install it again ------------

class Spy(ComfyUIInstaller):
    """ComfyUI, noting whether recover_all chose install or recover for it."""
    calls: list[str] = []

    def install(self):
        Spy.calls.append("install")
        return super().install()

    def recover(self):
        Spy.calls.append("recover")
        return super().recover()


def _fail_service():
    return patch.object(ComfyUIInstaller, "_setup_service", side_effect=RuntimeError("no systemd"))


def _fail_requirements():
    return patch("cloudgpu.remote.apps.comfyui.install_requirements_as_app",
                 side_effect=RuntimeError("pip failed"))


@pytest.mark.parametrize("failure", [_fail_service, _fail_requirements], ids=["service", "requirements"])
def test_failed_reinstall_of_a_recorded_app_is_installed_again_by_the_next_recover(comfy, failure, monkeypatch):
    """Catches: a re-install failing after the old record exists, so the next `up` sees an installed
    app and only recovers it; and a successful re-install dropping the app's re-provision mark."""
    comfy.state.mark_reprovision("comfyui")
    _, stack = host(comfy, [True], venv=True)
    with stack, failure(), pytest.raises(RuntimeError):
        comfy.install()
    record = comfy.state.get_app("comfyui")
    assert record["status"] != "installed", record
    assert record["reprovision"] is True, "the failed install keeps the mark"

    # The next `up` runs the remote recover: it must install ComfyUI again, not only recover it.
    Spy.calls = []
    monkeypatch.setattr(recovery, "setup_path", lambda st: None)
    _, stack = host(comfy, [True], venv=True)
    with stack:
        failures = recovery.recover_all(comfy.state, {"comfyui": Spy})
    assert Spy.calls == ["install"], Spy.calls
    assert failures == {}
    record = comfy.state.get_app("comfyui")
    assert record["status"] == "installed" and "installed_at" in record, record
    assert record["reprovision"] is True, "a successful install keeps the mark"


def test_failed_reinstall_of_ai_toolkit_is_installed_again(toolkit, monkeypatch):
    """Catches: ai-toolkit's re-install (same shape as ComfyUI's) keeping its old record on failure."""
    calls = []

    class ToolkitSpy(AIToolkitInstaller):
        def install(self):
            calls.append("install")
            return super().install()

        def recover(self):
            calls.append("recover")
            return super().recover()

    stub = ToolkitHost(toolkit, [True])
    with stub.stubs(), patch.object(AIToolkitInstaller, "_setup_service",
                                    side_effect=RuntimeError("no systemd")), pytest.raises(RuntimeError):
        toolkit.install()
    assert toolkit.state.get_app("ai-toolkit")["status"] != "installed"

    monkeypatch.setattr(recovery, "setup_path", lambda st: None)
    with ToolkitHost(toolkit, [True]).stubs():
        assert recovery.recover_all(toolkit.state, {"ai-toolkit": ToolkitSpy}) == {}
    assert calls == ["install"], calls
    assert toolkit.state.get_app("ai-toolkit")["status"] == "installed"


def test_failed_first_install_still_records_nothing(comfy):
    """Guard (0.6): an app installed for the first time that fails stays unrecorded."""
    comfy.state.remove_app("comfyui")
    _, stack = host(comfy, [True], venv=True)
    with stack, _fail_service(), pytest.raises(RuntimeError):
        comfy.install()
    assert comfy.state.get_app("comfyui") is None


# --- 2. a recorded app with no installer fails recovery -----------------------------------------

class _Recovers:
    calls: list[str] = []

    def __init__(self, state):
        self.state = state

    def recover(self):
        _Recovers.calls.append("comfyui")


def _recorded(tmp_persistent_dir, *apps):
    state = State(tmp_persistent_dir)
    for app in apps:
        state.set_app(app, {"status": "installed"})
    return state


def test_recover_all_counts_an_app_without_installer_as_a_failure(tmp_persistent_dir, monkeypatch):
    """Catches: 'Skipping <app>: no installer available' while recover_all reports nothing failed."""
    state = _recorded(tmp_persistent_dir, "mystery", "comfyui")
    paths = []
    monkeypatch.setattr(recovery, "setup_path", lambda st: paths.append(st))
    _Recovers.calls = []
    failures = recovery.recover_all(state, {"comfyui": _Recovers})
    assert list(failures) == ["mystery"], failures
    assert "no installer" in failures["mystery"]
    assert _Recovers.calls == ["comfyui"], "the other apps still recover"
    assert len(paths) == 1, "PATH is still set up"


def test_remote_recover_exits_nonzero_for_an_app_without_installer(tmp_persistent_dir, monkeypatch, capsys):
    """Catches: remote `recover` exiting 0 with a recorded app it didn't bring back."""
    _recorded(tmp_persistent_dir, "comfyui", "mystery")
    monkeypatch.setattr(recovery, "setup_path", lambda st: None)
    monkeypatch.setattr(remote_main, "APP_INSTALLERS", {"comfyui": _Recovers})
    monkeypatch.setattr(sys, "argv", ["remote", "--persistent-dir", tmp_persistent_dir, "recover"])
    with pytest.raises(SystemExit) as e:
        remote_main.main()
    assert e.value.code not in (0, None)
    text = "".join(capsys.readouterr())
    assert "mystery" in text[text.index("Recovery failed"):]


# --- 3. `recover` and `provision` keep the marks true -------------------------------------------

INSTALLED = {"status": "installed", "version": "1"}
UNMARKED = {"comfyui": INSTALLED, "ai-toolkit": INSTALLED}
MARKED = {**UNMARKED, "comfyui": {**INSTALLED, "reprovision": True}}


@pytest.fixture
def box(tmp_path, tmp_config_dir, monkeypatch):
    """Config 'c' with a running instance, recorded profiles a and b (ComfyUI) and t (ai-toolkit).

    ssh is stubbed: `status` answers UNMARKED until a `recover` ran, then ``box.after``.
    """
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "c.toml").write_text('gpu = ["gh200"]\nssh_key = "mini"\n')
    for name, apps in {"a": ["comfyui"], "b": ["comfyui"], "t": ["ai-toolkit"]}.items():
        d = tmp_path / "profiles" / name
        d.mkdir(parents=True)
        (d / "cloudgpu.toml").write_text(f"apps = {json.dumps(apps)}\n")
    monkeypatch.chdir(tmp_path)

    class Box:
        commands: list[str] = []
        provisioned: list[str] = []
        after = MARKED
        fail_provision: set[str] = set()

        def recorded(self, *names):
            configs.save_state("c", {"host": "ubuntu@h", "persistent_dir": "/lambda/nfs/p",
                                     "provisioned_profiles": list(names)})

        def clears(self):
            return [c.split("--app ")[1] for c in self.commands if " clear-reprovision " in c]

    b = Box()
    b.commands, b.provisioned, b.fail_provision = [], [], set()

    def ssh_run(host, command, **kwargs):
        b.commands.append(command)
        if command.endswith(" status"):
            apps = b.after if any(c.endswith(" recover") for c in b.commands) else UNMARKED
            return cli.ssh.SSHResult(0, json.dumps({"apps": apps}), "")
        return cli.ssh.SSHResult(0, "", "")

    def run_provision(prof, host, pd):
        b.commands.append(f"provision {prof['name']}")
        if prof["name"] in b.fail_provision:
            sys.exit(5)
        b.provisioned.append(prof["name"])
        return True

    monkeypatch.setattr(cli.sync, "sync_remote", lambda *a, **k: None)
    monkeypatch.setattr(cli.ssh, "ssh_run", ssh_run)
    monkeypatch.setattr(cli, "_run_provision", run_provision)
    monkeypatch.setattr(cli, "_restart_services", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_record_launch_info", lambda *a, **k: {})
    return b


def _flat(output: str) -> str:
    return " ".join(output.split())


def test_recover_that_leaves_a_mark_names_the_app_and_the_command(box):
    """Catches: `cloudgpu recover` setting a ComfyUI mark (venv rebuilt) and saying nothing."""
    box.recorded("a", "b")
    r = CliRunner().invoke(cli.cli, ["recover", "-C", "c"])
    assert r.exit_code == 0, r.output
    out = _flat(r.output)
    assert "comfyui" in out and "cloudgpu up -C c" in out, r.output
    assert "marked" in out
    assert box.clears() == [], "recover provisions nothing, so it mustn't clear the mark"


def test_recover_without_a_mark_says_nothing_about_one(box):
    """Guard: a healthy recover doesn't ask for re-provisioning."""
    box.recorded("a", "b")
    box.after = UNMARKED
    r = CliRunner().invoke(cli.cli, ["recover", "-C", "c"])
    assert r.exit_code == 0, r.output
    assert "marked" not in r.output and "cloudgpu up" not in r.output


def test_provision_one_profile_also_provisions_the_others_using_the_marked_app_then_clears(box):
    """Catches: `provision -P a` leaving the mark (so the next `up` provisions again), or clearing it
    while b, which also uses ComfyUI, never got its custom nodes back."""
    box.recorded("a", "b", "t")
    r = CliRunner().invoke(cli.provision, ["-C", "c", "-P", "a"])
    assert r.exit_code == 0, r.output
    assert box.provisioned == ["a", "b"], "b uses the marked app; t doesn't"
    assert box.clears() == ["comfyui"]
    clear = next(i for i, c in enumerate(box.commands) if " clear-reprovision " in c)
    assert clear > box.commands.index("provision b"), "cleared only after every profile was provisioned"


def test_provision_covering_every_profile_clears_the_mark(box):
    """Catches: `provision` never clearing a mark even when it re-provisioned every profile using it."""
    box.recorded("a", "b")
    r = CliRunner().invoke(cli.provision, ["-C", "c", "-P", "a,b"])
    assert r.exit_code == 0, r.output
    assert box.provisioned == ["a", "b"]
    assert box.clears() == ["comfyui"]


def test_failed_provision_keeps_the_mark(box):
    """Guard: a provision that fails (here b's) mustn't clear the mark."""
    box.recorded("a", "b")
    box.fail_provision = {"b"}
    r = CliRunner().invoke(cli.provision, ["-C", "c", "-P", "a,b"])
    assert r.exit_code != 0, r.output
    assert box.clears() == []


def test_provision_with_an_unloadable_recorded_profile_keeps_the_mark_and_says_how_to_clear_it(box):
    """Catches: `provision -P a` clearing (or silently keeping) the mark while recorded profile 'gone',
    which may use ComfyUI, can't be provisioned again."""
    box.recorded("a", "gone")
    r = CliRunner().invoke(cli.provision, ["-C", "c", "-P", "a"])
    assert r.exit_code == 0, r.output
    assert box.provisioned == ["a"]
    assert box.clears() == []
    out = _flat(r.output)
    assert "comfyui" in out and "cloudgpu up -C c -P gone" in out, r.output


# --- 4. an ai-toolkit venv rebuild sets the mark first ------------------------------------------

AIT = "cloudgpu.remote.apps.aitoolkit"


class ToolkitHost:
    """The stubbed host for ai-toolkit: each rebuild step is an event that notes whether the app
    was already marked when it ran."""

    def __init__(self, installer: AIToolkitInstaller, torch_answers: list[bool]):
        self.installer = installer
        self.answers = list(torch_answers)
        self.events: list[tuple[str, bool]] = []

    def marked(self) -> bool:
        return bool((self.installer.state.get_app("ai-toolkit") or {}).get("reprovision"))

    def event(self, name):
        def step(*args, **kwargs):
            self.events.append((name, self.marked()))
            if name == "create-venv":
                os.makedirs(self.installer.venv_dir, exist_ok=True)
            if name == "wipe":
                shutil.rmtree(self.installer.venv_dir, ignore_errors=True)
        return step

    def check(self, venv_dir):
        return self.answers.pop(0) if self.answers else True

    def stubs(self, **overrides) -> ExitStack:
        stack = ExitStack()
        for name in ("prepare_app_dir", "prepare_venv_checks", "trust_repo_as_app", "git_pull_as_app"):
            stack.enter_context(patch(f"{AIT}.{name}"))
        stack.enter_context(patch(f"{AIT}.git_version_as_app", return_value="abc1234"))
        stack.enter_context(patch(f"{AIT}.check_torch_cuda", side_effect=self.check))
        stack.enter_context(patch(f"{AIT}.wipe_venv", side_effect=self.event("wipe")))
        stack.enter_context(patch(f"{AIT}.run", side_effect=_no_real_commands))
        steps = {"_create_venv": "create-venv", "_ensure_torch": "torch", "_install_requirements": "requirements",
                 "_ensure_venv_symlink": None, "_ensure_node": None, "_build_ui": None,
                 "_create_launch_script": None, "_setup_service": None}
        for method, name in steps.items():
            fake = overrides.get(method) or (self.event(name) if name else (lambda *a, **k: None))
            stack.enter_context(patch.object(AIToolkitInstaller, method, side_effect=fake, autospec=False))
        stack.enter_context(patch.object(AIToolkitInstaller, "_ui_built", return_value=True))
        stack.enter_context(patch.object(subprocess, "run", _no_real_commands))
        stack.enter_context(patch.object(subprocess, "Popen", _no_real_commands))
        return stack


@pytest.fixture
def toolkit(tmp_path):
    """ai-toolkit on the filesystem from an earlier instance, built for this arch, with its venv."""
    state = State(str(tmp_path / "nfs"))
    state._ensure_dirs()
    installer = AIToolkitInstaller(state)
    os.makedirs(os.path.join(installer.ui_dir, "node_modules"))
    os.makedirs(installer.venv_dir)
    state.set_app("ai-toolkit", {"status": "installed", "app_dir": installer.app_dir,
                                 "venv_dir": installer.venv_dir, "node_arch": installer._arch(),
                                 "recovered_at": 1.0, "version": "old"})
    return installer


def _first_rebuild_step(events):
    return next(e for e in events if e[0] in ("wipe", "create-venv"))


@pytest.mark.parametrize("case", ["venv-missing", "torch-broken"])
def test_ai_toolkit_venv_rebuild_marks_first_and_keeps_the_mark(toolkit, case):
    """Catches: ai-toolkit rebuilding its venv (a profile's packages gone) with nothing marked, so
    nothing re-provisions; and the mark dropped by the record recover writes at the end."""
    if case == "venv-missing":
        shutil.rmtree(toolkit.venv_dir)
        h = ToolkitHost(toolkit, [True])
    else:
        h = ToolkitHost(toolkit, [False])
    with h.stubs():
        toolkit.recover()
    assert _first_rebuild_step(h.events)[1] is True, f"marked before the first rebuild step: {h.events}"
    record = toolkit.state.get_app("ai-toolkit")
    assert record["reprovision"] is True and record["status"] == "installed", record


@pytest.mark.parametrize("case", ["venv-missing", "torch-broken"])
def test_ai_toolkit_mark_outlives_a_rebuild_that_dies(toolkit, case):
    """Catches: the mark written only once the rebuild finished, so a recover that dies part way
    (then succeeds next time on the new venv) never re-provisions."""
    if case == "venv-missing":
        shutil.rmtree(toolkit.venv_dir)
    h = ToolkitHost(toolkit, [case == "venv-missing"])
    with h.stubs(_ensure_torch=RuntimeError("no CUDA torch")), pytest.raises(RuntimeError):
        toolkit.recover()
    assert toolkit.state.get_app("ai-toolkit").get("reprovision") is True


def test_ai_toolkit_missing_app_dir_marks_before_the_full_install(toolkit):
    """Catches: recover reinstalling a missing ai-toolkit (new venv) without a mark, and the mark
    lost when that install dies part way."""
    shutil.rmtree(toolkit.app_dir)
    seen = []

    def install(self):
        seen.append(bool(self.state.get_app("ai-toolkit").get("reprovision")))
        raise RuntimeError("clone failed")

    with ToolkitHost(toolkit, [True]).stubs(), patch.object(AIToolkitInstaller, "install", install), \
            pytest.raises(RuntimeError):
        toolkit.recover()
    assert seen == [True]
    assert toolkit.state.get_app("ai-toolkit")["reprovision"] is True


def test_ai_toolkit_full_install_from_recover_keeps_the_mark(toolkit):
    """Catches: ai-toolkit's install writing its record over the mark recover just set."""
    shutil.rmtree(toolkit.app_dir)

    def clone(cmd, **kwargs):
        assert cmd[:2] == ["git", "clone"], cmd
        os.makedirs(cmd[-1])

    with ToolkitHost(toolkit, [True]).stubs(), patch(f"{AIT}.run", side_effect=clone):
        toolkit.recover()
    record = toolkit.state.get_app("ai-toolkit")
    assert record["reprovision"] is True and record["status"] == "installed", record


def test_ai_toolkit_healthy_recover_sets_no_mark(toolkit):
    """Guard: a healthy recover (venv there, torch fine) rebuilds nothing and marks nothing."""
    h = ToolkitHost(toolkit, [True])
    with h.stubs():
        toolkit.recover()
    assert not [e for e in h.events if e[0] in ("wipe", "create-venv")], h.events
    assert not toolkit.state.get_app("ai-toolkit").get("reprovision")


def test_up_reprovisions_the_ai_toolkit_profiles_after_its_rebuild_and_clears_the_mark(tmp_path):
    """Guard: once ai-toolkit carries the mark, `up` re-provisions its recorded profile ('trainer',
    not ComfyUI's 'nodes') and clears the mark after."""
    after = {**BOTH, "ai-toolkit": {**BOTH["ai-toolkit"], "reprovision": True}}
    result = _up_after_rebuild(tmp_path, after=after)
    assert result.returncode == 0, result.output
    calls = result.shim_calls
    [recover] = _remote(calls, "recover")
    provisions = _provisions(calls)
    assert [name for _, name in provisions] == ["trainer"], result.output
    assert recover < provisions[0][0]
    clears = _clears(calls)
    assert [app for _, app in clears] == ["ai-toolkit"], result.output
    assert provisions[-1][0] < clears[0][0]
