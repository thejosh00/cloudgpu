"""`pip install -r` from app directories runs as cloudgpu-app, and never falls back to ubuntu.

cloudgpu-app can write every requirements file in app_dir, pip builds whatever one names, and ubuntu
has passwordless sudo. These tests use no real sudo or instance: a fake sudo first on PATH records
every call, and a planted pip and build backend record who ran them. The fake sudo doesn't clean the
environment, so a token only stays out if the code keeps it out. Everything runs as one real user, so
to show what cloudgpu-app "can't" read, the fake sudo hides the marker file while a step runs as it.
"""

from __future__ import annotations

import grp
import os
import pwd
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from cloudgpu.remote.apps import aitoolkit
from cloudgpu.remote.apps.aitoolkit import AIToolkitInstaller
from cloudgpu.remote.apps.comfyui import ComfyUIInstaller
from cloudgpu.remote.state import State
from cloudgpu.remote.utils import pip_install

APP_USER = "cloudgpu-app"
TOKEN = "made-up-token-value"
SECRET = "made-up-contents-of-a-file-only-ubuntu-can-read"
ME = pwd.getpwuid(os.getuid()).pw_name

FAKE_SUDO = """#!/bin/bash
# "sudo -u <user> -- cmd" runs cmd, marking who it was meant to run as. rm is real, as far as this
# user's permissions go, unless told otherwise. Anything else is an admin command, only recorded.
S={root}
echo "sudo ${{*//$'\n'/ }}" >> "$S/calls.log"
if [ "$1" = useradd ]; then
  [ -e "$S/useradd-fails" ] && {{ echo "useradd: refused" >&2; exit 1; }}
  touch "$S/user-exists"; exit 0
fi
if [ "$1" = rm ]; then
  [ -e "$S/rm-does-nothing" ] && exit 0
  exec "$@"
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
export HOME="$S/app-home" FAKE_SUDO_TARGET="$target"
if [ "$target" = {app_user} ] && [ -e "$S/marker.txt" ]; then
  chmod 000 "$S/marker.txt"
  "$@"; rc=$?
  chmod 600 "$S/marker.txt"
  exit $rc
fi
exec "$@"
"""

# What a venv's pip looks like: it records who ran it, what with, and what it saw.
PLANTED_PIP = """#!/bin/bash
S={root}
echo "pip ran as ${{FAKE_SUDO_TARGET:-the caller, directly}} in $PWD umask $(umask): $*" >> "$S/calls.log"
case "$*" in *" -r "*)
  env > "$S/pip-r.env"
  [ -e "$S/pip-fails" ] && {{ echo "pip: no matching distribution" >&2; exit 1; }}
  [ -e "$S/pip-hangs" ] && exec sleep 60
  [ -e "$3" ] || {{ echo "pip: could not open requirements file $3" >&2; exit 1; }}
  cp "$3" "$S/pip-was-given.txt"
  ;;
esac
exit 0
"""

# `python3 -m venv <dir>` makes a venv holding the planted pip.
PYTHON3_SHIM = """#!/bin/bash
S={root}
if [ "$1" = -m ] && [ "$2" = venv ]; then
  echo "venv created at $3" >> "$S/calls.log"
  mkdir -p "$3/bin"; cp "$S/planted-pip" "$3/bin/pip"; exit 0
fi
exec {python} "$@"
"""

BACKEND = """
import os
def _record():
    with open({marker!r}, "a") as f:
        f.write("planted build backend ran as " + os.environ.get("FAKE_SUDO_TARGET", "the caller, directly") + "\\n")
    with open({marker!r} + ".env", "w") as f:
        f.write("\\n".join(f"{{k}}={{v}}" for k, v in os.environ.items()))
    raise SystemExit(1)
def get_requires_for_build_wheel(config_settings=None):
    _record()
def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _record()
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
        executable(root / "bin" / "sudo", FAKE_SUDO.format(root=root, app_user=APP_USER))
        executable(root / "bin" / "python3", PYTHON3_SHIM.format(root=root, python=sys.executable))
        executable(root / "planted-pip", PLANTED_PIP.format(root=root))

    def calls(self) -> list[str]:
        log = self.root / "calls.log"
        return log.read_text().splitlines() if log.exists() else []

    def flag(self, name: str) -> None:
        (self.root / name).touch()

    def requirement_installs(self) -> list[str]:
        return [c for c in self.calls() if c.startswith("pip ran as") and " -r " in c]

    def assert_requirements_only_as_app(self, count: int) -> None:
        installs = self.requirement_installs()
        assert len(installs) == count, "\n".join(self.calls())
        assert all(c.startswith(f"pip ran as {APP_USER} ") for c in installs), "\n".join(installs)

    def index(self, *needles: str) -> int:
        return next(i for i, c in enumerate(self.calls()) if all(n in c for n in needles))

    def first_install(self) -> int:
        return next(i for i, c in enumerate(self.calls()) if c.startswith("pip ran as") and " -r " in c)

    def plant_pip(self, venv: Path) -> None:
        executable(venv / "bin" / "pip", (self.root / "planted-pip").read_text())


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
            return real_getpwnam(ME)
        raise KeyError(name)

    monkeypatch.setattr(pwd, "getpwnam", getpwnam)
    return sandbox


class Spec:
    def __init__(self, cls, module, methods):
        self.cls, self.module, self.methods = cls, module, methods


COMFY = Spec(ComfyUIInstaller, "cloudgpu.remote.apps.comfyui", ())
TOOLKIT = Spec(AIToolkitInstaller, "cloudgpu.remote.apps.aitoolkit",
               ("_ensure_node", "_build_ui", "_ensure_venv_symlink"))


def make(box: Sandbox, tmp_path: Path, spec: Spec, *, venv: bool = True):
    """An app on the filesystem from an earlier instance, with a real requirements file."""
    state = State(str((tmp_path / "nfs").resolve()))
    state._ensure_dirs()
    installer = spec.cls(state)
    app = Path(installer.app_dir)
    app.mkdir(parents=True)
    (app / "requirements.txt").write_text("pyyaml\n")
    if spec is TOOLKIT:
        Path(installer.ui_dir).mkdir()
        state.set_app(installer.name, {"node_arch": spec.cls._arch()})
    else:
        manager = app / "custom_nodes" / "ComfyUI-Manager"
        manager.mkdir(parents=True)
        (manager / "requirements.txt").write_text("gitpython\n")
    if venv:
        box.plant_pip(Path(installer.venv_dir))
    return installer


def healthy(*_):
    return True


@pytest.fixture
def stubs():
    """Everything else install() and recover() do, so only the requirements steps stay real."""
    active = []

    def start(spec: Spec, torch=healthy):
        names = ["git_pull_as_app", "trust_repo_as_app", "install_service"]
        ps = [patch(f"{spec.module}.{n}") for n in names]
        ps.append(patch(f"{spec.module}.git_version_as_app", return_value="abc"))
        ps.append(patch(f"{spec.module}.check_torch_cuda", side_effect=torch))
        if spec is TOOLKIT:
            ps += [patch.object(spec.cls, m) for m in spec.methods]
            ps.append(patch.object(spec.cls, "_ui_built", return_value=True))
            ps.append(patch.object(spec.cls, "_install_torchcodec"))
            ps.append(patch(f"{spec.module}.torch_index_url", return_value=None))
        for p in ps:
            p.start()
            active.append(p)

    yield start
    for p in active:
        p.stop()


def broken_then_healthy():
    answers = iter([False])
    return lambda *_: next(answers, True)


# Each way into the -r install, and how many -r installs it does (ComfyUI's are the app's and
# ComfyUI-Manager's, on recovery's rebuilds too since 0.7).
PATHS = [
    pytest.param(COMFY, "install", 2, id="comfyui-install"),
    pytest.param(COMFY, "venv-missing", 2, id="comfyui-recover-venv-missing"),
    pytest.param(COMFY, "rebuild", 2, id="comfyui-recover-rebuild"),
    pytest.param(TOOLKIT, "install", 1, id="ai-toolkit-install"),
    pytest.param(TOOLKIT, "venv-missing", 1, id="ai-toolkit-recover-venv-missing"),
    pytest.param(TOOLKIT, "rebuild", 1, id="ai-toolkit-recover-rebuild"),
]


def go(box, tmp_path, stubs, spec, path):
    """Set the app up for this path and return (installer, the call that takes it)."""
    installer = make(box, tmp_path, spec, venv=path != "venv-missing")
    stubs(spec, torch=broken_then_healthy() if path == "rebuild" else healthy)
    return installer, (installer.install if path == "install" else installer.recover)


class TestEveryInstallRunsAsApp:
    @pytest.mark.parametrize("spec,path,count", PATHS)
    def test_reaches_pip_only_through_sudo_as_app(self, box, tmp_path, stubs, spec, path, count):
        """Bug: pip builds what a planted requirements file names, as ubuntu, with sudo."""
        _, call = go(box, tmp_path, stubs, spec, path)
        call()
        box.assert_requirements_only_as_app(count)

    @pytest.mark.parametrize("spec,path,count", PATHS)
    def test_token_does_not_reach_pip(self, box, tmp_path, stubs, spec, path, count):
        """Bug: `run` merges os.environ, so HF_TOKEN reaches pip and what it builds."""
        _, call = go(box, tmp_path, stubs, spec, path)
        call()
        assert TOKEN not in (box.root / "pip-r.env").read_text()

    def test_comfyui_timeouts_kept(self, box, tmp_path, stubs):
        """Bug: moving the install loses a timeout, and a hung build hangs `up` for ever."""
        installer = make(box, tmp_path, COMFY)
        stubs(COMFY)
        with patch("cloudgpu.remote.service.app_step") as step:
            installer.install()
        assert [c.kwargs["timeout"] for c in step.call_args_list] == [600, 120]
        assert all(not c.kwargs.get("capture") for c in step.call_args_list), "pip must stream"

    @pytest.mark.parametrize("path,timeout", [("venv-missing", 120), ("rebuild", 600)])
    def test_comfyui_recover_timeouts_kept(self, box, tmp_path, stubs, path, timeout):
        """Bug: as above, on the recovery paths (the app's, then ComfyUI-Manager's as in install)."""
        _, call = go(box, tmp_path, stubs, COMFY, path)
        with patch("cloudgpu.remote.service.app_step") as step:
            call()
        assert [c.kwargs["timeout"] for c in step.call_args_list] == [timeout, 120]

    def test_ai_toolkit_timeout_kept(self, box, tmp_path, stubs):
        """Bug: as above: ai-toolkit's set can compile sdists for half an hour on aarch64."""
        installer = make(box, tmp_path, TOOLKIT)
        stubs(TOOLKIT)
        with patch("cloudgpu.remote.service.app_step") as step:
            installer.install()
        assert [c.kwargs["timeout"] for c in step.call_args_list] == [1800]
        assert not step.call_args.kwargs.get("capture"), "pip must stream"

    def test_pip_gets_a_compilers_path_and_group_write(self, box, tmp_path, stubs):
        """Bug: sudo resets PATH so sdists find no compiler; or files lack group write for ubuntu."""
        installer = make(box, tmp_path, COMFY)
        box.flag("secure-path")
        stubs(COMFY)
        installer.install()
        seen = dict(l.split("=", 1) for l in (box.root / "pip-r.env").read_text().splitlines() if "=" in l)
        assert seen["PATH"].startswith(str(box.root / "bin"))
        assert seen["HOME"] == str(box.root / "app-home"), "pip's cache must be cloudgpu-app's"
        assert "umask 0002" in box.requirement_installs()[0]
        assert " in / " in box.requirement_installs()[0], "a planted file in the cwd mustn't count"


class TestRealPipRealPackage:
    """A real clean venv and real pip, with a requirements file naming a local planted package."""

    @pytest.fixture
    def planted(self, box, tmp_path):
        installer = make(box, tmp_path, COMFY, venv=False)
        subprocess.run([sys.executable, "-m", "venv", installer.venv_dir], check=True, env={"PATH": os.defpath})
        pkg = Path(installer.app_dir) / "planted"
        marker = box.root / "backend-ran"
        pkg.mkdir()
        (pkg / "pyproject.toml").write_text(
            '[build-system]\nrequires = []\nbuild-backend = "backend"\nbackend-path = ["."]\n'
            '[project]\nname = "planted"\nversion = "0"\n')
        (pkg / "backend.py").write_text(BACKEND.format(marker=str(marker)))
        (Path(installer.app_dir) / "requirements.txt").write_text(f"{pkg}\n")
        return installer, marker

    def test_build_backend_runs_only_as_app_without_the_token(self, box, stubs, planted):
        """Bug, the driver's own demo: a planted package's backend runs as ubuntu, with its tokens."""
        installer, marker = planted
        stubs(COMFY)
        with pytest.raises(subprocess.CalledProcessError):
            installer.install()
        lines = marker.read_text().splitlines()
        assert lines and all(l == f"planted build backend ran as {APP_USER}" for l in lines), lines
        assert TOKEN not in Path(f"{marker}.env").read_text()

    def test_refused_sudo_runs_nothing_planted(self, box, stubs, planted):
        """Bug: a refused sudo falls back to running pip as ubuntu."""
        installer, marker = planted
        box.flag("sudo-refuses")
        stubs(COMFY)
        with pytest.raises(RuntimeError, match="couldn't run as"):
            installer.install()
        assert not marker.exists()


class TestFailuresLookLikeFailures:
    @pytest.mark.parametrize("spec,path,count", PATHS)
    def test_sudo_refused_fails_and_pip_never_runs(self, box, tmp_path, stubs, spec, path, count):
        """Bug: "couldn't run" is swallowed, or falls back to pip as ubuntu."""
        _, call = go(box, tmp_path, stubs, spec, path)
        box.flag("sudo-refuses")
        with pytest.raises(RuntimeError, match="couldn't run as"):
            call()
        assert box.requirement_installs() == []

    @pytest.mark.parametrize("spec,path,count", PATHS)
    def test_user_cannot_be_created_fails(self, box, tmp_path, stubs, spec, path, count):
        """Bug: useradd fails and the step goes ahead, or carries on as ubuntu."""
        _, call = go(box, tmp_path, stubs, spec, path)
        box.flag("useradd-fails")
        with pytest.raises(subprocess.CalledProcessError):
            call()
        assert box.requirement_installs() == []

    @pytest.mark.parametrize("spec,path,count", PATHS)
    def test_pip_failure_fails(self, box, tmp_path, stubs, spec, path, count):
        """Bug: pip's own failure no longer stops the install, leaving a half-built venv "installed"."""
        installer, call = go(box, tmp_path, stubs, spec, path)
        box.flag("pip-fails")
        with pytest.raises(subprocess.CalledProcessError):
            call()
        assert (installer.state.get_app(installer.name) or {}).get("status") != "installed"

    @pytest.mark.parametrize("spec", [COMFY, TOOLKIT], ids=["comfyui", "ai-toolkit"])
    def test_timeout_fails(self, box, tmp_path, stubs, spec):
        """Bug: a hung pip isn't stopped, or its timeout reads as success."""
        import cloudgpu.remote.service as service
        real = service.app_step
        installer = make(box, tmp_path, spec)
        stubs(spec)
        box.flag("pip-hangs")
        with patch.object(service, "app_step",
                          side_effect=lambda cmd, **kw: real(cmd, **{**kw, "timeout": 1})):
            with pytest.raises(subprocess.TimeoutExpired):
                installer.install()

    def test_torchcodec_warning_does_not_swallow_a_requirements_failure(self, box, tmp_path):
        """Bug: the warn-and-carry-on for torchcodec catches the -r install's failure too."""
        installer = make(box, tmp_path, TOOLKIT)
        box.flag("user-exists")
        box.flag("pip-fails")
        with pytest.raises(subprocess.CalledProcessError):
            installer._install_requirements()

    def test_torchcodec_failure_still_only_warns(self, box, tmp_path, capsys):
        """Bug: making -r failures fatal makes torchcodec's fatal too."""
        installer = make(box, tmp_path, TOOLKIT)
        box.flag("user-exists")
        real = Path(installer.venv_dir) / "bin" / "pip"
        real.write_text(real.read_text().replace('case "$*" in', '[ "$2" = --no-deps ] && exit 1\ncase "$*" in'))
        with patch("cloudgpu.remote.apps.aitoolkit.torch_index_url", return_value=None):
            installer._install_requirements()
        assert "torchcodec install failed" in capsys.readouterr().err


class TestPresenceCheck:
    def test_no_file_skips_the_install(self, box, tmp_path, stubs):
        """Bug: an app with no requirements file now fails."""
        installer = make(box, tmp_path, COMFY)
        (Path(installer.app_dir) / "requirements.txt").unlink()
        (Path(installer.app_dir) / "custom_nodes" / "ComfyUI-Manager" / "requirements.txt").unlink()
        stubs(COMFY)
        installer.install()
        assert box.requirement_installs() == []

    @pytest.mark.parametrize("spec", [COMFY, TOOLKIT], ids=["comfyui", "ai-toolkit"])
    def test_broken_symlink_fails_rather_than_skipping(self, box, tmp_path, stubs, spec):
        """Bug: os.path.exists reads a broken symlink as "no requirements": the install is skipped."""
        installer = make(box, tmp_path, spec)
        requirements = Path(installer.app_dir) / "requirements.txt"
        requirements.unlink()
        requirements.symlink_to(box.root / "nothing-here")
        stubs(spec)
        with pytest.raises((subprocess.CalledProcessError, RuntimeError)):
            installer.install()
        assert (installer.state.get_app(installer.name) or {}).get("status") != "installed"


class TestFlatten:
    @pytest.fixture
    def toolkit(self, box, tmp_path):
        installer = make(box, tmp_path, TOOLKIT)
        box.flag("user-exists")
        return installer

    def written_to_app_dir(self, installer) -> str:
        out = []
        for p in Path(installer.app_dir).rglob("*"):
            if p.is_file() and not p.is_symlink():
                out.append(p.read_text(errors="replace"))
        return "\n".join(out)

    def secret(self, box) -> Path:
        marker = box.root / "marker.txt"
        marker.write_text(SECRET + "\n")
        marker.chmod(0o600)
        return marker

    def test_torchcodec_pin_dropped_and_the_result_is_what_gets_installed(self, box, toolkit):
        """Bug: the pin reaches pip (no arm64 wheel below 0.10, so a GH200 install dies on it)."""
        app = Path(toolkit.app_dir)
        (app / "requirements_base.txt").write_text("av==16.0.1\ntorchcodec==0.9.1\nlibrosa==0.11.0\n")
        (app / "requirements.txt").write_text("-r requirements_base.txt\nscipy==1.12.0\n")
        with patch.object(AIToolkitInstaller, "_install_torchcodec"):
            toolkit._install_requirements()
        box.assert_requirements_only_as_app(1)
        assert f"-r {app / '.cloudgpu-requirements.txt'}" in box.requirement_installs()[0]
        given = (box.root / "pip-was-given.txt").read_text().splitlines()
        assert [l for l in given if l and not l.startswith("#")] == ["av==16.0.1", "librosa==0.11.0", "scipy==1.12.0"]
        assert "# torchcodec==0.9.1  (pin dropped by cloudgpu)" in given

    def test_flatten_runs_through_sudo_as_app(self, box, toolkit):
        """Bug: flattening stays ubuntu's, following paths cloudgpu-app chose."""
        toolkit._patched_requirements(os.path.join(toolkit.app_dir, "requirements.txt"))
        assert any(c.startswith("sudo ") and f"-u {APP_USER}" in c and "cloudgpu-requirements" in c
                   for c in box.calls()), "\n".join(box.calls())

    def test_symlink_at_the_output_does_not_redirect_the_write(self, box, toolkit):
        """Bug: open(patched, "w") follows a planted symlink and overwrites what ubuntu can write."""
        victim = box.root / "victim.txt"
        victim.write_text("untouched\n")
        patched = Path(toolkit.app_dir) / ".cloudgpu-requirements.txt"
        patched.symlink_to(victim)
        toolkit._patched_requirements(os.path.join(toolkit.app_dir, "requirements.txt"))
        assert victim.read_text() == "untouched\n"
        assert not patched.is_symlink()
        assert "pyyaml" in patched.read_text()

    def test_requirements_symlink_is_not_read_as_ubuntu(self, box, toolkit):
        """Bug: open(path) follows a symlink as ubuntu, copying a secret to where app code reads it."""
        marker = self.secret(box)
        requirements = Path(toolkit.app_dir) / "requirements.txt"
        requirements.unlink()
        requirements.symlink_to(marker)
        with pytest.raises(RuntimeError, match="Couldn't read"):
            toolkit._patched_requirements(str(requirements))
        assert SECRET not in self.written_to_app_dir(toolkit)

    def test_include_is_not_read_as_ubuntu(self, box, toolkit):
        """Bug: an `-r` include can name any path, and ubuntu copies that file into app_dir."""
        marker = self.secret(box)
        requirements = Path(toolkit.app_dir) / "requirements.txt"
        requirements.write_text(f"-r {marker}\n")
        with pytest.raises(RuntimeError, match="Couldn't read"):
            toolkit._patched_requirements(str(requirements))
        assert SECRET not in self.written_to_app_dir(toolkit)

    def test_the_failure_message_carries_no_contents(self, box, toolkit):
        """Bug: the error echoes a requirements file, and any token in it, into the log."""
        marker = self.secret(box)
        requirements = Path(toolkit.app_dir) / "requirements.txt"
        requirements.write_text(f"--extra-index-url https://{SECRET}@example.invalid/simple\n-r {marker}\n")
        with pytest.raises(RuntimeError) as raised:
            toolkit._patched_requirements(str(requirements))
        assert SECRET not in str(raised.value)

    def test_include_cycle_fails_clearly(self, box, toolkit):
        """Bug: a cycle recurses until it crashes, or hangs the install."""
        app = Path(toolkit.app_dir)
        (app / "requirements.txt").write_text("-r other.txt\n")
        (app / "other.txt").write_text("-r requirements.txt\n")
        with pytest.raises(RuntimeError, match="include cycle"):
            toolkit._patched_requirements(str(app / "requirements.txt"))

    def test_self_include_through_a_symlink_is_a_cycle(self, box, toolkit):
        """Bug: a cycle through a symlink recurses until it crashes."""
        app = Path(toolkit.app_dir)
        (app / "alias.txt").symlink_to(app / "requirements.txt")
        (app / "requirements.txt").write_text("-r alias.txt\n")
        with pytest.raises(RuntimeError, match="include cycle"):
            toolkit._patched_requirements(str(app / "requirements.txt"))

    def test_self_include_under_a_new_spelling_each_time_is_a_cycle(self, box, toolkit):
        """Bug: the cycle check compares names, and `./` makes every name different, so it never fires."""
        app = Path(toolkit.app_dir)
        (app / "requirements.txt").write_text("-r ./requirements.txt\n")
        with pytest.raises(RuntimeError, match="include cycle"):
            toolkit._patched_requirements(str(app / "requirements.txt"))

    def test_symlink_planted_after_the_unlink_does_not_redirect_the_write(self, tmp_path, monkeypatch):
        """Bug: without O_EXCL, a symlink put at the output just after the unlink gets written through."""
        victim = tmp_path / "victim.txt"
        victim.write_text("untouched\n")
        requirements = tmp_path / "requirements.txt"
        requirements.write_text("av\n")
        patched = tmp_path / ".cloudgpu-requirements.txt"
        patched.write_text("old\n")
        real_unlink = os.unlink

        def unlink_then_plant(path, *args, **kwargs):
            real_unlink(path, *args, **kwargs)
            os.symlink(victim, path)

        monkeypatch.setattr(os, "unlink", unlink_then_plant)
        monkeypatch.setattr(sys, "argv", ["flatten", str(requirements), str(patched)])
        with pytest.raises(SystemExit) as stopped:
            exec(aitoolkit._FLATTEN_REQUIREMENTS, {})
        assert stopped.value.code  # a message, not a clean exit
        assert victim.read_text() == "untouched\n"

    def test_same_file_included_twice_is_not_a_cycle(self, box, toolkit):
        """Bug: the cycle check refuses a harmless repeat, failing a good install."""
        app = Path(toolkit.app_dir)
        (app / "base.txt").write_text("av\n")
        (app / "requirements.txt").write_text("-r base.txt\n-r base.txt\n")
        patched = toolkit._patched_requirements(str(app / "requirements.txt"))
        assert Path(patched).read_text().splitlines() == ["av", "av"]

    def test_flatten_that_cannot_run_fails_and_writes_nothing(self, box, toolkit):
        """Bug: with sudo refused, flattening falls back to ubuntu."""
        box.flag("sudo-refuses")
        with pytest.raises(RuntimeError, match="couldn't run as"):
            toolkit._patched_requirements(os.path.join(toolkit.app_dir, "requirements.txt"))
        assert not (Path(toolkit.app_dir) / ".cloudgpu-requirements.txt").exists()


class TestWriteAccessComesFirst:
    @pytest.mark.parametrize("spec,path,count", PATHS)
    def test_user_and_sharing_come_before_the_first_install(self, box, tmp_path, stubs, spec, path, count):
        """Bug: in the "venv missing" paths the first -r install ran before the user or sharing existed."""
        installer, call = go(box, tmp_path, stubs, spec, path)
        call()
        first = box.first_install()
        assert box.index("sudo useradd") < first
        assert box.index("sudo chgrp", installer.venv_dir) < first
        assert box.index("sudo chgrp", installer.app_dir) < first

    @pytest.mark.parametrize("spec", [COMFY, TOOLKIT], ids=["comfyui", "ai-toolkit"])
    def test_venv_shared_again_after_ubuntus_own_installs(self, box, tmp_path, stubs, spec):
        """Bug: files from ubuntu's --no-deps install have no group write, so pip as app can't replace them."""
        installer = make(box, tmp_path, spec)
        stubs(spec)
        installer.install()
        no_deps = box.index("pip ran as", "--no-deps")
        first = box.first_install()
        assert no_deps < first
        assert any("sudo chgrp" in c and installer.venv_dir in c for c in box.calls()[no_deps:first])


class TestVenvWipe:
    @pytest.mark.parametrize("spec", [COMFY, TOOLKIT], ids=["comfyui", "ai-toolkit"])
    def test_rebuild_really_removes_the_old_venv(self, box, tmp_path, stubs, spec):
        """Bug: a half-removed venv is rebuilt over, and carries on looking fine."""
        installer, call = go(box, tmp_path, stubs, spec, "rebuild")
        leftover = Path(installer.venv_dir) / "leftover"
        leftover.touch()
        call()
        assert not leftover.exists()
        assert box.index("sudo rm -rf", installer.venv_dir) < box.index("venv created at", installer.venv_dir)

    @pytest.mark.parametrize("spec", [COMFY, TOOLKIT], ids=["comfyui", "ai-toolkit"])
    def test_wipe_that_is_refused_does_not_go_on(self, box, tmp_path, stubs, spec):
        """Bug: rmtree(ignore_errors=True) hides a refused wipe, and the rebuild goes on over it."""
        installer, call = go(box, tmp_path, stubs, spec, "rebuild")
        locked = Path(installer.venv_dir) / "locked"
        locked.mkdir()
        (locked / "file").touch()
        locked.chmod(0o555)
        try:
            with pytest.raises((subprocess.CalledProcessError, RuntimeError)):
                call()
            assert not any("venv created at" in c for c in box.calls())
            assert box.requirement_installs() == []
        finally:
            locked.chmod(0o755)

    @pytest.mark.parametrize("spec", [COMFY, TOOLKIT], ids=["comfyui", "ai-toolkit"])
    def test_wipe_that_leaves_the_venv_does_not_go_on(self, box, tmp_path, stubs, spec):
        """Bug: rm says it worked but the venv is still there (NFS can do this)."""
        installer, call = go(box, tmp_path, stubs, spec, "rebuild")
        box.flag("rm-does-nothing")
        with pytest.raises(RuntimeError, match="Couldn't remove"):
            call()
        assert not any("venv created at" in c for c in box.calls())

    def test_healthy_venv_is_not_wiped(self, box, tmp_path, stubs):
        """Bug: the wipe runs on every recovery, rebuilding a good venv."""
        installer, _ = go(box, tmp_path, stubs, COMFY, "install")
        installer.recover()
        assert not any("sudo rm" in c for c in box.calls())


class TestUbuntusOwnInstalls:
    """They stay ubuntu's, because a planted file can't change what they name, but must still work."""

    def no_group_yet(self, monkeypatch):
        monkeypatch.setattr(grp, "getgrnam", lambda name: type("G", (), {"gr_gid": 54321})())
        monkeypatch.setattr(os, "getgroups", lambda: [20])

    def venv(self, box, tmp_path) -> Path:
        venv = tmp_path / "venv"
        box.plant_pip(venv)
        box.flag("user-exists")
        return venv

    def test_new_session_when_this_process_lacks_the_group(self, box, tmp_path, monkeypatch):
        """Bug: on a new instance ubuntu isn't in the group yet, so it can't replace cloudgpu-app's files."""
        self.no_group_yet(monkeypatch)
        pip_install(str(self.venv(box, tmp_path)), "--no-deps", "torchvision")
        assert box.calls()[0].startswith(f"sudo -n -u {ME} -- ")
        assert box.calls()[1].startswith(f"pip ran as {ME} ")
        assert not any(f"-u {APP_USER}" in c for c in box.calls())

    def test_directly_when_this_process_has_the_group(self, box, tmp_path, monkeypatch):
        """Bug: a sudo session is added when nothing needs it, so every caller changes by accident."""
        self.no_group_yet(monkeypatch)
        monkeypatch.setattr(os, "getgroups", lambda: [20, 54321])
        pip_install(str(self.venv(box, tmp_path)), "--no-deps", "torchvision")
        assert [c.split(" in ")[0] for c in box.calls()] == ["pip ran as the caller, directly"]

    def test_directly_when_the_group_is_this_process_primary(self, box, tmp_path, monkeypatch):
        """Bug: as above, for a primary group, which getgroups may leave out."""
        self.no_group_yet(monkeypatch)
        monkeypatch.setattr(os, "getgid", lambda: 54321)
        pip_install(str(self.venv(box, tmp_path)), "--no-deps", "torchvision")
        assert [c.split(" in ")[0] for c in box.calls()] == ["pip ran as the caller, directly"]

    def test_directly_when_there_is_no_group(self, box, tmp_path, monkeypatch):
        """Bug: `status`-style callers on a new instance fail, or sudo for a group that isn't there."""
        def no_group(name):
            raise KeyError(name)
        monkeypatch.setattr(grp, "getgrnam", no_group)
        pip_install(str(self.venv(box, tmp_path)), "--upgrade", "pip")
        assert [c.split(" in ")[0] for c in box.calls()] == ["pip ran as the caller, directly"]

    def test_failure_through_the_session_still_fails(self, box, tmp_path, monkeypatch):
        """Bug: going through sudo loses pip's non-zero exit, or a refused sudo reads as success."""
        self.no_group_yet(monkeypatch)
        box.flag("sudo-refuses")
        with pytest.raises(subprocess.CalledProcessError):
            pip_install(str(self.venv(box, tmp_path)), "--no-deps", "torchvision")

    @pytest.mark.parametrize("spec", [COMFY, TOOLKIT], ids=["comfyui", "ai-toolkit"])
    def test_ubuntus_named_installs_stay_ubuntus(self, box, tmp_path, stubs, spec):
        """Bug: installs the plan says stay ubuntu's are moved too, which phase 4 relies on not happening."""
        installer = make(box, tmp_path, spec)
        stubs(spec)
        installer.install()
        no_deps = [c for c in box.calls() if c.startswith("pip ran as") and "--no-deps" in c]
        assert no_deps and all(c.startswith("pip ran as the caller, directly") for c in no_deps)

    def test_force_reinstall_of_torch_stays_ubuntus(self, box, tmp_path, stubs):
        """Bug: as above, for the torch install, in the path where torch is broken."""
        installer = make(box, tmp_path, COMFY)
        answers = iter([False, True])
        stubs(COMFY, torch=lambda *_: next(answers))
        with patch("cloudgpu.remote.utils.torch_index_url", return_value="https://example.invalid/cu124"):
            installer.install()
        torch = [c for c in box.calls() if "--force-reinstall" in c and c.startswith("pip ran as")]
        assert torch and torch[0].startswith("pip ran as the caller, directly")
