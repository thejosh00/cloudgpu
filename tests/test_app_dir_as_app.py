"""git, npm and node in app directories run as cloudgpu-app, and never fall back to ubuntu.

cloudgpu-app can write all of app_dir, including .git and the UI's package.json and node_modules,
and ubuntu has passwordless sudo. These tests use no real sudo or instance: a fake sudo first on
PATH records every call, git is real behind a recording shim, and a planted hook, npm, node and
package.json script record who ran them. The fake sudo doesn't clean the environment, so a token
only stays out of a step if the code keeps it out. All of these run as one real user, so they show
who a step was meant to run as, not what the instance's permissions allow.
"""

from __future__ import annotations

import json
import os
import pwd
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from cloudgpu.remote import run_as_app as helper
from cloudgpu.remote.apps.aitoolkit import AIToolkitInstaller
from cloudgpu.remote.apps.comfyui import ComfyUIInstaller
from cloudgpu.remote.state import State

APP_USER = "cloudgpu-app"
TOKEN = "made-up-token-value"
REAL_GIT = shutil.which("git")

FAKE_SUDO = """#!/bin/bash
# "sudo -u <user> -- cmd" runs cmd, marking who it was meant to run as. Anything else is an admin
# command, which is only recorded. useradd "creates" the user.
S={root}
echo "sudo ${{*//$'\n'/ }}" >> "$S/calls.log"  # one line per call: run_as_app's shell has several
if [ "$1" = useradd ]; then
  [ -e "$S/useradd-fails" ] && {{ echo "useradd: refused" >&2; exit 1; }}
  touch "$S/user-exists"; exit 0
fi
target=
while [ $# -gt 0 ]; do
  case "$1" in
    -u) target="$2"; shift 2 ;;
    --) shift; break ;;
    -*) shift ;;
    *) exit 0 ;;
  esac
done
[ -n "$target" ] || exit 0
[ -e "$S/sudo-refuses" ] && {{ echo "sudo: a password is required" >&2; exit 1; }}
[ -e "$S/user-exists" ] || {{ echo "sudo: unknown user $target" >&2; exit 1; }}
[ -e "$S/secure-path" ] && PATH=/usr/bin:/bin
umask 022
HOME="$S/app-home" FAKE_SUDO_TARGET="$target" exec "$@"
"""

GIT_SHIM = """#!/bin/bash
echo "git ran as ${{FAKE_SUDO_TARGET:-the caller, directly}}: $*" >> {root}/calls.log
exec {real_git} "$@"
"""

# Starts with env, as the real npm does, so it only runs if node's directory reached PATH: it gets
# here through the planted node below. Runs package.json's scripts, as the real npm does.
PLANTED_NPM = """#!/usr/bin/env node
"""

# `node --version` from node_dir stays ubuntu's (see TestNodeDirStaysOutOfScope), so it isn't recorded.
PLANTED_NODE = """#!/bin/bash
S={root}
if [ "$1" = --version ]; then echo v0; exit 0; fi
if [ "$1" = -e ]; then
  echo "node -e ran as ${{FAKE_SUDO_TARGET:-the caller, directly}} in $PWD" >> "$S/calls.log"
  env > "$S/node.env"
  if [ -e "$S/sqlite3-rebuilt" ] || [ ! -e "$S/sqlite3-broken" ]; then exit 0; fi
  exit 1
fi
shift  # the npm script's own path: from here on this is npm
echo "npm $* ran as ${{FAKE_SUDO_TARGET:-the caller, directly}} in $PWD" >> "$S/calls.log"
env > "$S/npm-$1-$2.env"
run_script() {{
  body=$("{python}" -I -c 'import json,sys; print(json.load(open("package.json")).get("scripts",{{}}).get(sys.argv[1],""))' "$1")
  [ -z "$body" ] || sh -c "$body"
}}
case "$1" in
  install) mkdir -p node_modules; run_script postinstall ;;
  ls) [ -e "$S/tree-bad" ] && exit 1; exit 0 ;;
  run) run_script "$2" ;;
  rebuild) touch "$S/sqlite3-rebuilt" ;;
esac
"""

PLANTED_SCRIPT = """#!/bin/bash
echo "{label} ran as ${{FAKE_SUDO_TARGET:-the caller, directly}}" >> {root}/calls.log
"""


def executable(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


class Sandbox:
    def __init__(self, root: Path):
        self.root = root
        (root / "app-home").mkdir()
        executable(root / "bin" / "sudo", FAKE_SUDO.format(root=root))
        executable(root / "bin" / "git", GIT_SHIM.format(root=root, real_git=REAL_GIT))

    def calls(self) -> list[str]:
        log = self.root / "calls.log"
        return log.read_text().splitlines() if log.exists() else []

    def flag(self, name: str) -> None:
        (self.root / name).touch()

    def user_exists(self) -> None:
        self.flag("user-exists")

    def index(self, *needles: str) -> int:
        return next(i for i, c in enumerate(self.calls()) if all(n in c for n in needles))

    def ran(self, label: str) -> list[str]:
        return [c for c in self.calls() if c.startswith(f"{label} ran as ")
                or (label == "git" and c.startswith("git ran as "))]

    def assert_only_as_app(self, label: str) -> None:
        ran = self.ran(label)
        assert ran, f"{label} never ran:\n" + "\n".join(self.calls())
        assert all(f"ran as {APP_USER}" in c for c in ran), "\n".join(ran)

    def assert_never_ran(self, label: str) -> None:
        assert not self.ran(label), "\n".join(self.calls())

    def direct(self) -> list[str]:
        return [c for c in self.calls() if "the caller, directly" in c]

    def env_of(self, name: str) -> dict[str, str]:
        lines = (self.root / f"{name}.env").read_text().splitlines()
        return dict(line.split("=", 1) for line in lines if "=" in line)


def git(*args: str, cwd: Path | None = None) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
           "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    return subprocess.run([REAL_GIT, *args], cwd=cwd, env=env, check=True,
                          capture_output=True, text=True).stdout.strip()


def upstream(path: Path) -> Path:
    path.mkdir(parents=True)
    git("init", "-q", "-b", "main", cwd=path)
    (path / "main.py").write_text("one\n")
    git("add", ".", cwd=path)
    git("commit", "-q", "-m", "one", cwd=path)
    return path


def new_upstream_commit(path: Path) -> None:
    (path / "main.py").write_text("two\n")
    git("commit", "-qam", "two", cwd=path)


def clone_with_hook(box: Sandbox, origin: Path, dest: Path, label: str) -> None:
    """A repo already on the filesystem, with a hook app code planted. The next pull fires it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    git("clone", "-q", str(origin), str(dest))
    git("config", "pull.ff", "only", cwd=dest)
    executable(dest / ".git" / "hooks" / "post-merge", PLANTED_SCRIPT.format(label=label, root=box.root))
    new_upstream_commit(origin)


@pytest.fixture
def box(tmp_path, monkeypatch):
    """Fake sudo first on PATH; cloudgpu-app exists only once the sandbox says so."""
    root = (tmp_path / "sandbox").resolve()
    root.mkdir()
    sandbox = Sandbox(root)
    caller_home = root / "caller-home"
    caller_home.mkdir()
    monkeypatch.setenv("HOME", str(caller_home))
    monkeypatch.setenv("PATH", f"{root / 'bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.delenv("FAKE_SUDO_TARGET", raising=False)

    real_getpwnam = pwd.getpwnam

    def getpwnam(name):
        if name != APP_USER:
            return real_getpwnam(name)
        if (root / "user-exists").exists():
            return real_getpwnam(os.environ.get("USER") or pwd.getpwuid(os.getuid()).pw_name)
        raise KeyError(name)

    monkeypatch.setattr(pwd, "getpwnam", getpwnam)
    return sandbox


def make_installer(tmp_path: Path, cls):
    persistent = (tmp_path / "nfs").resolve()
    state = State(str(persistent))
    state._ensure_dirs()
    installer = cls(state)
    Path(installer.venv_dir).mkdir(parents=True)  # exists, so install() doesn't create one
    return installer


class ComfyUIApp:
    cls = ComfyUIInstaller
    module = "cloudgpu.remote.apps.comfyui"
    repo_const = "COMFYUI_REPO"
    stubbed = ("pip_install", "pip_install_torch", "install_service")
    methods: tuple[str, ...] = ()


class AIToolkitApp:
    cls = AIToolkitInstaller
    module = "cloudgpu.remote.apps.aitoolkit"
    repo_const = "AITOOLKIT_REPO"
    stubbed = ("pip_install", "pip_install_torch", "install_service")
    methods = ("_ensure_torch", "_install_requirements", "_ensure_node")


def plant_ui(box: Sandbox, installer: AIToolkitInstaller) -> None:
    """The planted node and npm, and a package.json with a planted script for install and build."""
    ui = Path(installer.ui_dir)
    ui.mkdir(parents=True, exist_ok=True)
    executable(ui / "planted-script.sh", PLANTED_SCRIPT.format(label="planted script", root=box.root))
    (ui / "package.json").write_text(json.dumps({"scripts": {
        "postinstall": "./planted-script.sh",
        "update_db": "./planted-script.sh",
        "build": "./planted-script.sh",
    }}))
    node_bin = Path(installer.node_bin)
    executable(node_bin / "node", PLANTED_NODE.format(root=box.root, python=sys.executable))
    executable(node_bin / "npm", PLANTED_NPM)


@pytest.fixture(params=[ComfyUIApp, AIToolkitApp], ids=["comfyui", "ai-toolkit"])
def app(request, box, tmp_path):
    """An installer whose app_dir is a real clone, on a filesystem from an earlier instance."""
    spec = request.param
    installer = make_installer(tmp_path, spec.cls)
    origin = upstream(tmp_path / "origin")
    clone_with_hook(box, origin, Path(installer.app_dir), "planted hook")
    manager_origin = upstream(tmp_path / "manager-origin")
    clone_with_hook(box, manager_origin, Path(installer.app_dir) / "custom_nodes" / "ComfyUI-Manager",
                    "planted manager hook")
    if spec is AIToolkitApp:
        plant_ui(box, installer)
    patches = [patch(f"{spec.module}.{name}") for name in spec.stubbed]
    patches.append(patch(f"{spec.module}.check_torch_cuda", return_value=True))
    patches += [patch.object(spec.cls, name) for name in spec.methods]
    for p in patches:
        p.start()
    yield installer
    for p in patches:
        p.stop()


@pytest.fixture
def toolkit(box, tmp_path):
    installer = make_installer(tmp_path, AIToolkitInstaller)
    plant_ui(box, installer)
    box.user_exists()
    return installer


class TestGitAsApp:
    def test_pull_and_its_hook_run_only_as_app(self, box, app):
        """Bug: a hook planted in .git runs as ubuntu on the next install, with passwordless sudo."""
        app.install()
        pulls = [c for c in box.calls() if c.startswith("git ran as") and " pull" in c]
        assert pulls
        assert all(f"ran as {APP_USER}" in c for c in pulls), pulls
        box.assert_only_as_app("planted hook")
        assert "two" in (Path(app.app_dir) / "main.py").read_text()

    def test_no_git_step_runs_directly(self, box, app):
        """Bug: one git step (a version read, say) is left running as ubuntu on planted config."""
        app.install()
        box.assert_only_as_app("git")
        assert box.direct() == []

    def test_comfyui_manager_pull_runs_as_app(self, box, tmp_path):
        """Bug: custom_nodes/ComfyUI-Manager is missed, and its planted hook runs as ubuntu."""
        installer = make_installer(tmp_path, ComfyUIInstaller)
        origin = upstream(tmp_path / "origin")
        clone_with_hook(box, origin, Path(installer.app_dir), "planted hook")
        manager = Path(installer.app_dir) / "custom_nodes" / "ComfyUI-Manager"
        clone_with_hook(box, upstream(tmp_path / "manager-origin"), manager, "planted manager hook")
        module = ComfyUIApp.module
        with (patch(f"{module}.pip_install"), patch(f"{module}.install_service"),
              patch(f"{module}.check_torch_cuda", return_value=True)):
            installer.install()
        box.assert_only_as_app("planted manager hook")

    def test_comfyui_manager_clone_runs_as_app(self, box, tmp_path):
        """Bug: a new manager clone is ubuntu's, inside a directory cloudgpu-app can write."""
        installer = make_installer(tmp_path, ComfyUIInstaller)
        clone_with_hook(box, upstream(tmp_path / "origin"), Path(installer.app_dir), "planted hook")
        manager_origin = upstream(tmp_path / "manager-origin")
        module = ComfyUIApp.module
        with (patch(f"{module}.MANAGER_REPO", str(manager_origin)), patch(f"{module}.pip_install"),
              patch(f"{module}.install_service"), patch(f"{module}.check_torch_cuda", return_value=True)):
            installer.install()
        box.assert_only_as_app("git")
        assert (Path(installer.app_dir) / "custom_nodes" / "ComfyUI-Manager" / "main.py").exists()

    def test_status_reads_the_version_as_app(self, box, app):
        """Bug: `status` runs git as ubuntu, reading a .git/config cloudgpu-app can write."""
        box.user_exists()
        with patch(f"{type(app).__module__}.torch_cuda_status", return_value=True):
            status = app.get_status()
        box.assert_only_as_app("git")
        assert status["version"] == git("rev-parse", "--short", "HEAD", cwd=Path(app.app_dir))

    def test_status_with_no_user_reads_no_version_and_uses_no_sudo(self, box, app):
        """Bug: `status` on a new instance runs git as ubuntu, or changes the instance."""
        with patch(f"{type(app).__module__}.torch_cuda_status", return_value=True):
            status = app.get_status()
        assert status["version"] is None
        assert box.calls() == []

    def test_status_with_sudo_refused_reads_no_version(self, box, app):
        """Bug: a refused sudo makes `status` crash, or fall back to git as ubuntu."""
        box.user_exists()
        box.flag("sudo-refuses")
        with patch(f"{type(app).__module__}.torch_cuda_status", return_value=True):
            status = app.get_status()
        assert status["version"] is None
        assert box.direct() == []


class TestWhenItCannotRun:
    def test_sudo_refused_fails_the_install_and_nothing_planted_runs(self, box, app):
        """Bug: a refused sudo falls back to ubuntu, or is ignored and install carries on."""
        box.flag("sudo-refuses")
        with pytest.raises(helper.CouldNotRun):
            app.install()
        box.assert_never_ran("planted hook")
        assert box.direct() == []
        assert "one" in (Path(app.app_dir) / "main.py").read_text()  # never pulled

    def test_user_that_cannot_be_created_fails_the_install(self, box, app):
        """Bug: with no cloudgpu-app the pull goes ahead as ubuntu."""
        box.flag("useradd-fails")
        with pytest.raises(subprocess.CalledProcessError):
            app.install()
        box.assert_never_ran("planted hook")
        assert box.direct() == []

    def test_failed_pull_fails_the_install(self, box, app):
        """Bug: a pull that really failed is ignored, and the install goes on with old code."""
        git("remote", "set-url", "origin", "/nonexistent", cwd=Path(app.app_dir))
        with pytest.raises(subprocess.CalledProcessError):
            app.install()
        box.assert_never_ran("planted hook")


class TestNewInstance:
    def test_user_and_sharing_come_before_the_first_pull(self, box, app):
        """Bug: on a new instance the first pull comes before the user or the group exists."""
        app.install()
        useradd = box.index("sudo useradd")
        share = box.index("sudo chgrp", app.app_dir)
        writable = box.index("sudo chmod -R g+rwX", app.app_dir)
        pull = box.index("git ran as", " pull")
        assert useradd < share < pull
        assert useradd < writable < pull

    def test_repo_cloned_by_ubuntu_is_trusted_by_name_for_app_only(self, box, app):
        """Bug: git refuses a repo ubuntu cloned, or a wildcard weakens ubuntu's own git."""
        app.install()
        config = (box.root / "app-home" / ".gitconfig").read_text()
        assert f"directory = {app.app_dir}" in config
        assert "directory = *" not in config
        assert not (box.root / "caller-home" / ".gitconfig").exists()
        assert box.index("safe.directory", app.app_dir) < box.index("git ran as", " pull")

    @pytest.mark.parametrize("spec", [ComfyUIApp, AIToolkitApp], ids=["comfyui", "ai-toolkit"])
    def test_trust_is_redone_on_recovery(self, box, tmp_path, spec):
        """Bug: APP_HOME is instance-local, so a recovered instance can't read the version."""
        installer = make_installer(tmp_path, spec.cls)
        clone_with_hook(box, upstream(tmp_path / "origin"), Path(installer.app_dir), "planted hook")
        if spec is AIToolkitApp:
            plant_ui(box, installer)
        patches = [patch(f"{spec.module}.check_torch_cuda", return_value=True),
                   patch(f"{spec.module}.install_service")]
        patches += [patch.object(spec.cls, m) for m in spec.methods]
        for p in patches:
            p.start()
        try:
            installer.recover()
        finally:
            for p in patches:
                p.stop()
        config = (box.root / "app-home" / ".gitconfig").read_text()
        assert f"directory = {installer.app_dir}" in config

    @pytest.mark.parametrize("spec", [ComfyUIApp, AIToolkitApp], ids=["comfyui", "ai-toolkit"])
    def test_fresh_clone_is_shared_and_trusted_so_the_next_pull_works(self, box, tmp_path, spec):
        """Bug: a fresh clone stays ubuntu's alone, and cloudgpu-app can't pull it next time."""
        installer = make_installer(tmp_path, spec.cls)
        origin = upstream(tmp_path / "origin")
        manager_origin = upstream(tmp_path / "manager-origin")
        names = [patch(f"{spec.module}.{n}") for n in spec.stubbed]
        names.append(patch(f"{spec.module}.check_torch_cuda", return_value=True))
        names.append(patch(f"{spec.module}.{spec.repo_const}", str(origin)))
        if spec is ComfyUIApp:
            names.append(patch(f"{spec.module}.MANAGER_REPO", str(manager_origin)))
        else:
            names += [patch.object(spec.cls, m) for m in spec.methods]
            names.append(patch.object(spec.cls, "_build_ui"))
        for p in names:
            p.start()
        try:
            installer.install()
        finally:
            for p in names:
                p.stop()
        clone = box.index("git ran as the caller, directly: clone", installer.app_dir)
        assert clone < box.index("sudo chgrp", installer.app_dir)
        assert box.index("sudo chgrp", installer.app_dir) < box.index("safe.directory", installer.app_dir)
        # The clone is the one git step that stays ubuntu's: nothing is planted in a path that
        # doesn't exist yet, and apps_dir isn't cloudgpu-app's to write.
        directly = [c for c in box.direct() if c.startswith("git ran as")]
        assert len(directly) == 1, directly


class TestBuildUi:
    def test_every_step_runs_as_app_in_the_ui_dir(self, box, toolkit):
        """Bug: an npm or node step is left running as ubuntu, or from the wrong directory."""
        box.flag("sqlite3-broken")
        toolkit._build_ui()
        steps = [c for c in box.calls() if c.startswith(("npm ", "node "))]
        expected = ["npm install", "npm ls --all", "node -e", "npm rebuild sqlite3 --build-from-source",
                    "node -e", "npm run update_db", "npm run build"]
        assert [c.split(" ran as ")[0] for c in steps] == expected
        assert all(c.endswith(f"ran as {APP_USER} in {toolkit.ui_dir}") for c in steps), steps

    def test_planted_package_json_script_runs_only_as_app(self, box, toolkit):
        """Bug: a build script planted in package.json runs as ubuntu on the next install."""
        toolkit._build_ui()
        box.assert_only_as_app("planted script")

    def test_install_builds_the_ui_as_app(self, box, tmp_path):
        """Bug: _build_ui is right, but install() reaches npm another way."""
        installer = make_installer(tmp_path, AIToolkitInstaller)
        clone_with_hook(box, upstream(tmp_path / "origin"), Path(installer.app_dir), "planted hook")
        plant_ui(box, installer)
        spec = AIToolkitApp
        patches = [patch(f"{spec.module}.{n}") for n in spec.stubbed]
        patches += [patch.object(spec.cls, m) for m in spec.methods]
        for p in patches:
            p.start()
        try:
            installer.install()
        finally:
            for p in patches:
                p.stop()
        assert box.direct() == []
        box.assert_only_as_app("planted script")

    def test_recover_shares_the_dir_before_building(self, box, tmp_path):
        """Bug: an earlier ubuntu build's files aren't writable by cloudgpu-app when it rebuilds."""
        installer = make_installer(tmp_path, AIToolkitInstaller)
        clone_with_hook(box, upstream(tmp_path / "origin"), Path(installer.app_dir), "planted hook")
        plant_ui(box, installer)
        installer.state.set_app(installer.name, {"node_arch": installer._arch()})
        with (patch(f"{AIToolkitApp.module}.check_torch_cuda", return_value=True),
              patch(f"{AIToolkitApp.module}.install_service"),
              patch.object(AIToolkitInstaller, "_ensure_node")):
            installer.recover()
        assert box.index("sudo chgrp", installer.app_dir) < box.index("npm install ran as")
        assert box.direct() == []

    def test_arch_change_wipes_as_app_after_sharing(self, box, tmp_path):
        """Bug: ubuntu's wipe of node_modules is refused on files cloudgpu-app wrote, and hidden."""
        installer = make_installer(tmp_path, AIToolkitInstaller)
        clone_with_hook(box, upstream(tmp_path / "origin"), Path(installer.app_dir), "planted hook")
        plant_ui(box, installer)
        stale = Path(installer.ui_dir) / "node_modules" / "stale"
        stale.parent.mkdir()
        stale.write_text("from the other architecture")
        installer.state.set_app(installer.name, {"node_arch": "another-arch"})
        with (patch(f"{AIToolkitApp.module}.check_torch_cuda", return_value=True),
              patch(f"{AIToolkitApp.module}.install_service"),
              patch.object(AIToolkitInstaller, "_ensure_node")):
            installer.recover()
        assert not stale.exists()
        share = box.index("sudo chgrp", installer.app_dir)
        wipe = box.index(f"-u {APP_USER}", "rm -rf", "node_modules")
        assert share < wipe < box.index("npm install ran as")


class TestCheckThatCouldNotRun:
    def plant_tree(self, toolkit) -> Path:
        node_modules = Path(toolkit.ui_dir) / "node_modules"
        node_modules.mkdir()
        (node_modules / "good").write_text("still here\n")
        return node_modules

    def test_refused_sudo_does_not_wipe_node_modules(self, box, toolkit):
        """Bug: a refused sudo reads as "the tree is bad", and a good node_modules is wiped."""
        node_modules = self.plant_tree(toolkit)
        box.flag("sudo-refuses")
        with pytest.raises(helper.CouldNotRun):
            toolkit._build_ui()
        assert (node_modules / "good").exists()
        assert box.direct() == []

    def test_tree_check_refused_is_not_a_verdict(self, box, toolkit):
        """Bug: _npm_tree_ok returns False on a refused sudo, which wipes node_modules."""
        self.plant_tree(toolkit)
        npm = os.path.join(toolkit.node_bin, "npm")
        box.flag("sudo-refuses")
        with pytest.raises(helper.CouldNotRun):
            toolkit._npm_tree_ok(npm, {"PATH": toolkit.node_bin + os.pathsep + os.environ["PATH"]})

    def test_sqlite3_check_refused_starts_no_rebuild(self, box, toolkit):
        """Bug: _sqlite3_loads returns False on a refused sudo, which starts a source rebuild."""
        npm = os.path.join(toolkit.node_bin, "npm")
        env = {"PATH": toolkit.node_bin + os.pathsep + os.environ["PATH"]}
        box.flag("sudo-refuses")
        with pytest.raises(helper.CouldNotRun):
            toolkit._sqlite3_loads(env)  # False here is what starts a rebuild
        with pytest.raises(helper.CouldNotRun):
            toolkit._ensure_sqlite3(npm, env)
        assert not (box.root / "sqlite3-rebuilt").exists()
        assert not [c for c in box.calls() if c.startswith("npm rebuild")]

    def test_a_check_that_ran_and_failed_is_still_a_verdict(self, box, toolkit):
        """Bug: real damage stops being noticed, so a half-written tree reaches next build."""
        self.plant_tree(toolkit)
        box.flag("tree-bad")
        with pytest.raises(RuntimeError, match="incomplete"):
            toolkit._build_ui()
        assert len([c for c in box.calls() if c.startswith("npm install")]) == 2  # the NFS retry


class TestEnvironment:
    def test_what_npm_needs_arrives_even_when_sudo_resets_path(self, box, toolkit):
        """Bug: `#!/usr/bin/env node` can't find node, or the build settings don't survive sudo."""
        box.flag("secure-path")
        box.flag("sqlite3-broken")
        toolkit._build_ui()
        build = box.env_of("npm-run-build")
        assert build["NODE_OPTIONS"] == "--max-old-space-size=4096"
        assert build["NEXT_TELEMETRY_DISABLED"] == "1"
        assert build["PATH"].split(os.pathsep)[0] == toolkit.node_bin
        assert box.env_of("npm-rebuild-sqlite3")["npm_config_build_from_source"] == "true"
        assert box.env_of("npm-install-")["NEXT_TELEMETRY_DISABLED"] == "1"

    def test_a_token_in_the_callers_environment_does_not_reach_a_step(self, box, toolkit):
        """Bug: `run`'s {**os.environ, **env} hands every secret to a planted build script."""
        box.flag("sqlite3-broken")
        toolkit._build_ui()
        for name in ("npm-install-", "npm-ls---all", "npm-run-update_db", "npm-run-build",
                     "npm-rebuild-sqlite3", "node"):
            env = box.env_of(name)
            assert "HF_TOKEN" not in env, name
            assert TOKEN not in env.values(), name

    def test_a_token_does_not_reach_a_planted_git_hook(self, box, tmp_path):
        """Bug: the pull as cloudgpu-app still carries ubuntu's environment to a planted hook."""
        installer = make_installer(tmp_path, ComfyUIInstaller)
        origin = upstream(tmp_path / "origin")
        clone_with_hook(box, origin, Path(installer.app_dir), "planted hook")
        hook = Path(installer.app_dir) / ".git" / "hooks" / "post-merge"
        hook.write_text(hook.read_text() + f"env > {box.root}/hook.env\n")
        manager = Path(installer.app_dir) / "custom_nodes" / "ComfyUI-Manager"
        clone_with_hook(box, upstream(tmp_path / "manager-origin"), manager, "planted manager hook")
        module = ComfyUIApp.module
        with (patch(f"{module}.pip_install"), patch(f"{module}.install_service"),
              patch(f"{module}.check_torch_cuda", return_value=True)):
            installer.install()
        assert "HF_TOKEN" not in box.env_of("hook")


class TestWipe:
    def undeletable(self, toolkit) -> Path:
        node_modules = Path(toolkit.ui_dir) / "node_modules"
        locked = node_modules / "locked"
        locked.mkdir(parents=True)
        (locked / "file").write_text("can't be removed\n")
        locked.chmod(0o555)
        return locked

    def test_wipe_that_fails_does_not_go_on(self, box, toolkit):
        """Bug: ignore_errors hides a refused wipe, so the "wiped" tree is reinstalled over."""
        locked = self.undeletable(toolkit)
        box.flag("tree-bad")
        try:
            with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
                toolkit._build_ui()
            installs = [c for c in box.calls() if c.startswith("npm install")]
            assert len(installs) == 1, installs
            assert not [c for c in box.calls() if c.startswith("npm run")]
        finally:
            locked.chmod(0o755)

    def test_wipe_that_reports_success_but_leaves_the_tree_raises(self, box, toolkit):
        """Bug: a wipe whose exit code lies (NFS, say) goes on as if the tree were gone."""
        node_modules = Path(toolkit.ui_dir) / "node_modules"
        node_modules.mkdir()
        executable(box.root / "bin" / "rm", "#!/bin/bash\nexit 0\n")
        with pytest.raises(RuntimeError, match="node_modules"):
            toolkit._wipe_node_modules()

    def test_wipe_removes_the_tree_as_app(self, box, toolkit):
        """Bug: the wipe runs as ubuntu, which may not be in cloudgpu-app's group yet."""
        node_modules = Path(toolkit.ui_dir) / "node_modules"
        node_modules.mkdir()
        (node_modules / "stale").write_text("x")
        toolkit._wipe_node_modules()
        assert not node_modules.exists()
        assert box.index(f"-u {APP_USER}", "rm -rf", str(node_modules)) >= 0

    def test_wipe_refused_leaves_the_tree_and_raises(self, box, toolkit):
        """Bug: a refused wipe reads as done."""
        node_modules = Path(toolkit.ui_dir) / "node_modules"
        node_modules.mkdir()
        box.flag("sudo-refuses")
        with pytest.raises(helper.CouldNotRun):
            toolkit._wipe_node_modules()
        assert node_modules.exists()


class TestNodeDirStaysOutOfScope:
    def test_node_dir_is_never_shared_with_cloudgpu_app(self, box, tmp_path):
        """Bug: node_dir becomes writable by cloudgpu-app, while ubuntu still runs node from it."""
        installer = make_installer(tmp_path, AIToolkitInstaller)
        clone_with_hook(box, upstream(tmp_path / "origin"), Path(installer.app_dir), "planted hook")
        plant_ui(box, installer)
        module = AIToolkitApp.module
        with (patch(f"{module}.check_torch_cuda", return_value=True), patch(f"{module}.pip_install"),
              patch.object(AIToolkitInstaller, "_ensure_torch"),
              patch.object(AIToolkitInstaller, "_install_requirements"),
              patch.object(AIToolkitInstaller, "_ensure_node"), patch(f"{module}.install_service")):
            installer.install()
        admin = [c for c in box.calls() if c.startswith(("sudo chgrp", "sudo chmod", "sudo find"))]
        assert admin
        assert not [c for c in admin if installer.node_dir in c
                    or c.rstrip().endswith(installer.state.apps_dir)], admin
        assert installer.node_dir not in installer.service_spec()["rw_paths"]


class TestAppStep:
    def test_timeout_stops_the_step(self, box):
        """Bug: `npm install` that hangs stops the whole `up`, as the helper had no timeout."""
        box.user_exists()
        with pytest.raises(subprocess.TimeoutExpired):
            helper.app_step(["sleep", "30"], timeout=0.5)

    def test_capture_returns_the_steps_own_output_and_exit(self, box):
        """Bug: the started marker leaks into a step's output, or a failed step reads as refused."""
        box.user_exists()
        result = helper.app_step(["/bin/sh", "-c", "echo out; echo err >&2; exit 3"],
                                 capture=True, check=False)
        assert (result.returncode, result.stdout, result.stderr) == (3, "out\n", "err\n")

    def test_capture_tells_a_refused_sudo_from_a_failed_step(self, box):
        """Bug: with check=False a refused sudo is just another non-zero exit."""
        box.user_exists()
        box.flag("sudo-refuses")
        with pytest.raises(helper.CouldNotRun):
            helper.app_step(["/bin/sh", "-c", "exit 0"], capture=True, check=False)

    def test_a_directory_it_cannot_enter_means_the_step_never_ran(self, box, tmp_path):
        """Bug: when `cd` fails the step runs in / anyway, or reads as the step's own failure."""
        box.user_exists()
        marker = box.root / "ran"
        with pytest.raises(helper.CouldNotRun):
            helper.app_step(["/usr/bin/touch", str(marker)], cwd=str(tmp_path / "missing"),
                            capture=True, check=False)
        assert not marker.exists()

    def test_runs_in_the_directory_it_is_given(self, box, tmp_path):
        """Bug: 0.2's helper always ran in /, and npm needs the UI directory."""
        box.user_exists()
        result = helper.app_step(["/bin/pwd"], cwd=str(tmp_path.resolve()), capture=True)
        assert result.stdout.strip() == str(tmp_path.resolve())

    def test_settings_given_here_arrive_and_the_callers_do_not(self, box):
        """Bug: PATH or NODE_OPTIONS don't reach a step, or the caller's secrets do."""
        box.user_exists()
        result = helper.app_step(["/usr/bin/env"], env={"NODE_OPTIONS": "x"}, capture=True)
        assert "NODE_OPTIONS=x" in result.stdout
        assert TOKEN not in result.stdout

    def test_group_write_survives_sudos_umask(self, box, tmp_path):
        """Bug: files the step writes aren't group-writable, which breaks ubuntu's next write."""
        box.user_exists()
        result = helper.app_step(["/bin/sh", "-c", "umask"], capture=True)
        assert result.stdout.strip().endswith("002")
