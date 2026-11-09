"""A venv-missing rebuild gets the full install timeout, and a timed-out step is really stopped (0.7-fix).

On an instance, recovery rebuilt ComfyUI's venv with 120 s for `pip install -r`, timed out, and pip
kept running as cloudgpu-app after cloudgpu gave up on it. The first tests fake the installer's
steps. The others run app_step's timeout handling for real, with sudo stood in for by a script
that runs its command locally as the current user: no real sudo and no cloudgpu-app.
"""

from __future__ import annotations

import os
import shutil
import signal
import stat
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

import pytest

from cloudgpu.remote import run_as_app as helper
from cloudgpu.remote import utils
from tests.test_comfyui_rebuild_torch import Host, comfy  # noqa: F401  (comfy is a fixture)


class TimeoutHost(Host):
    """The stubbed host, also recording the timeout each requirements install is given."""

    def __init__(self, installer, torch_answers):
        super().__init__(installer, torch_answers)
        self.timeouts: dict[str, int] = {}

    def requirements(self, venv_dir, requirements, *, timeout):
        super().requirements(venv_dir, requirements, timeout=timeout)
        self.timeouts[self.events[-1]] = timeout


def _run(installer, step, *, answers):
    h = TimeoutHost(installer, answers)
    with ExitStack() as stack:
        h.stubs(stack)
        step()
    return h.timeouts


class TestRebuildTimeout:
    def test_venv_missing_gets_the_fresh_install_requirements_timeout(self, comfy):  # noqa: F811
        """Catches: the venv-missing recovery giving `pip install -r` 120 s, too little for a new venv."""
        os.makedirs(comfy.venv_dir)
        fresh = _run(comfy, comfy.install, answers=[False, True])
        shutil.rmtree(comfy.venv_dir)
        rebuilt = _run(comfy, comfy.recover, answers=[False, True])
        assert fresh["-r ."] == 600
        assert rebuilt["-r ."] == fresh["-r ."], rebuilt


# sudo stand-ins: run what follows `--` as the current user. EXEC_SUDO becomes the command, as a
# sudo that execs would. RELAY_SUDO keeps running and passes SIGTERM on to the command only, never
# to the command's own children, which is what real sudo does when it relays at all.
EXEC_SUDO = """#!/bin/sh
while [ $# -gt 0 ]; do
  case "$1" in --) shift; break ;; -u) shift 2 ;; *) shift ;; esac
done
exec "$@"
"""

RELAY_SUDO = """#!/bin/bash
while [ $# -gt 0 ]; do
  case "$1" in --) shift; break ;; -u) shift 2 ;; *) shift ;; esac
done
exec 3<&0
"$@" <&3 3<&- &
cmd=$!
exec 3<&-
trap 'kill -TERM $cmd 2>/dev/null' TERM
while kill -0 $cmd 2>/dev/null; do wait $cmd; done
wait $cmd
"""

# A step that leaves a child running, a grandchild whose parent has gone (so it's init's), and,
# in the stubborn variant, ignores SIGTERM (as do its children, which inherit that). Each writes
# its PID before the step sleeps, so the test knows what to look for.
STEP = """{trap}
sleep 60 & echo $! > "$1/child.pid"
( sleep 60 & echo $! > "$1/grandchild.pid" )
echo $$ > "$1/step.pid"
wait
"""


def _alive(pid: int) -> bool:
    """Running, as opposed to gone or a zombie waiting to be reaped."""
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout
    return bool(out.strip()) and not out.strip().startswith("Z")


@pytest.fixture(params=["exec-sudo", "relaying-sudo"])
def fake_sudo(request, tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sudo = bin_dir / "sudo"
    sudo.write_text(EXEC_SUDO if request.param == "exec-sudo" else RELAY_SUDO)
    sudo.chmod(sudo.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(helper, "_STOP_GRACE", 0.5, raising=False)
    return sudo


@pytest.fixture
def pids_dir(tmp_path):
    d = tmp_path / "pids"
    d.mkdir()
    yield d
    for f in d.glob("*.pid"):  # whatever a failing run left behind
        try:
            os.kill(int(f.read_text()), signal.SIGKILL)
        except (ValueError, ProcessLookupError, PermissionError):
            pass


def _step_pids(pids_dir: Path) -> dict[str, int]:
    names = ("step", "child", "grandchild")
    pids = {n: int((pids_dir / f"{n}.pid").read_text()) for n in names
            if (pids_dir / f"{n}.pid").exists()}
    assert set(pids) == set(names), f"the step didn't get going in time: {pids}"
    return pids


class TestTimedOutStepIsStopped:
    @pytest.mark.parametrize("trap", ["", "trap '' TERM"], ids=["obeys-sigterm", "ignores-sigterm"])
    def test_nothing_the_step_started_outlives_app_step(self, fake_sudo, pids_dir, trap):
        """Catches: on timeout only sudo is signalled, so the step or its children keep running."""
        started = time.monotonic()
        with pytest.raises(subprocess.TimeoutExpired):
            helper.app_step(["/bin/sh", "-c", STEP.format(trap=trap), "step", str(pids_dir)],
                            timeout=1.5)
        pids = _step_pids(pids_dir)
        deadline = time.monotonic() + 2  # SIGKILL is immediate, the exit a moment later
        while any(_alive(p) for p in pids.values()) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not {n: p for n, p in pids.items() if _alive(p)}, "still running after TimeoutExpired"
        assert time.monotonic() - started < 15

    def test_a_step_that_finishes_in_time_is_untouched(self, fake_sudo, pids_dir):
        """Catches: the stop machinery firing, or leaving a mark, when there was no timeout."""
        result = helper.app_step(["/bin/sh", "-c", "echo done"], capture=True, timeout=10)
        assert result.stdout == "done\n"


class TestTimedOutVenvCheckIsStopped:
    def test_nothing_the_check_started_outlives_it(self, fake_sudo, pids_dir, tmp_path, monkeypatch):
        """Catches: run_venv_check's sudo -u cloudgpu-app python, killed on timeout as sudo only."""
        venv = tmp_path / "venv"
        python = venv / "bin" / "python3"
        python.parent.mkdir(parents=True)
        # Stands in for a venv Python whose check hangs, ignoring SIGTERM, with children.
        python.write_text("#!/bin/sh\n" + STEP.format(trap="trap '' TERM").replace("$1", str(pids_dir)))
        python.chmod(0o755)
        monkeypatch.setattr(utils.pwd, "getpwnam", lambda name: object())
        with pytest.raises(utils.VenvCheckUnavailable):
            utils.run_venv_check(str(venv), "pass", timeout=1.5)
        pids = _step_pids(pids_dir)
        deadline = time.monotonic() + 2
        while any(_alive(p) for p in pids.values()) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not {n: p for n, p in pids.items() if _alive(p)}, "still running after the timeout"
