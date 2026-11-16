"""Item 3.4 part 1: the instance-side final sync (``datasync final``).

Nothing here runs sudo, systemctl or rclone: sudo is the recorder from the 3.2 setup
tests, with a made-up service whose runs succeed or fail and write a last-success time.
"""

from __future__ import annotations

import json
import os
import subprocess
import time

import pytest

from cloudgpu.remote import __main__ as remote_main
from cloudgpu.remote import datasync_setup as ds
from tests.test_datasync_setup_3_2 import KEY_TEXT, KEY_VALUE, Box

OLD = "2026-11-01T08:00:00Z"


def stamp(when: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(when))


class FinalBox(Box):
    """The 3.2 recorder, plus a timer, a service that may be mid-run, and its results."""

    def __init__(self, tmp_path, monkeypatch, installed=True):
        super().__init__(tmp_path, monkeypatch)
        self.timer_active = True
        self.busy_polls = 0          # how many times the service is still seen running
        self.fresh_ok = True         # does the run we start succeed
        self.fresh_code: int | None = None   # exit of `systemctl start`, if not from fresh_ok
        self.fresh_writes: str | None = None  # last-success a good run writes (default: now)
        self.hang = False            # `systemctl start <service>` never returns
        self.last: str | None = None  # contents of the last-success file
        self.result = "success"
        self.timer_during_run: list[bool] = []
        self.slept: list[float] = []
        self.kwargs: list[dict] = []
        monkeypatch.setattr(ds.time, "sleep", self.slept.append)
        os.makedirs(ds.ETC_DIR)
        with open(ds.KEY_PATH, "w") as f:
            f.write(KEY_TEXT)
        if installed:
            for path in (ds.SPEC_PATH, ds.service_path(), ds.timer_path()):
                with open(path, "w") as f:
                    f.write("{}")

    def run(self, cmd, **kwargs):
        assert cmd[0] == "sudo", f"only sudo may be run, got {cmd}"
        rest = list(cmd[1:])
        self.calls.append(rest)
        self.envs.append(kwargs.get("env"))
        self.kwargs.append(kwargs)
        timer, service = f"{ds.UNIT_NAME}.timer", f"{ds.UNIT_NAME}.service"
        if rest[0] == "cat":
            assert rest[1] == ds.LAST_SUCCESS_FILE
            if self.last is None:
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="No such file")
            return subprocess.CompletedProcess(cmd, 0, stdout=self.last + "\n", stderr="")
        assert rest[0] == "systemctl", rest
        if rest[1] == "show":
            assert rest[-1] == service
            if "--property=ActiveState" in rest:
                if self.busy_polls > 0:
                    self.busy_polls -= 1
                    return subprocess.CompletedProcess(cmd, 0, stdout="activating\n", stderr="")
                return subprocess.CompletedProcess(cmd, 0, stdout="inactive\n", stderr="")
            assert "--property=Result" in rest
            return subprocess.CompletedProcess(cmd, 0, stdout=self.result + "\n", stderr="")
        if rest[1:] == ["stop", timer]:
            self.timer_active = False
        elif rest[1:] == ["start", timer]:
            self.timer_active = True
        elif rest[1:] == ["start", service]:
            self.timer_during_run.append(self.timer_active)
            if self.hang:
                raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))
            if self.fresh_ok:
                self.result = "success"
                self.last = self.fresh_writes or stamp(time.time())
            else:
                self.result = "exit-code"
            code = self.fresh_code if self.fresh_code is not None else (0 if self.fresh_ok else 1)
            return subprocess.CompletedProcess(cmd, code)
        else:
            raise AssertionError(f"unexpected command {rest}")
        return subprocess.CompletedProcess(cmd, 0)

    def short(self):
        """The recorded commands that change something, in order."""
        return [" ".join(c[1:]) for c in self.calls if c[0] == "systemctl" and c[1] != "show"]

    def starts(self):
        return [c for c in self.calls if c[1:] == ["start", f"{ds.UNIT_NAME}.service"]]


TIMER = f"{ds.UNIT_NAME}.timer"
SERVICE = f"{ds.UNIT_NAME}.service"


@pytest.fixture
def box(tmp_path, monkeypatch):
    return FinalBox(tmp_path, monkeypatch)


def test_timer_is_stopped_first_then_one_fresh_run_is_started_and_waited_for(box, capsys):
    assert ds.main("final", "/fs") == 0
    assert box.calls[0] == ["systemctl", "stop", TIMER]
    assert box.short() == [f"stop {TIMER}", f"start {SERVICE}"]
    assert box.timer_during_run == [False]  # no later run could start beside it
    # waited for: a blocking start bounded by the unit's own limit, and its result read after
    start = box.calls.index(["systemctl", "start", SERVICE])
    assert box.kwargs[start]["timeout"] == ds.FINAL_WAIT_SECONDS >= 6 * 60 * 60
    assert "--no-block" not in box.calls[start]
    assert any("--property=Result" in c for c in box.calls[start + 1:])
    err = capsys.readouterr().err
    assert "Final sync finished at " + box.last in err
    assert "timer stays stopped" in err
    assert not box.timer_active


def test_a_run_in_progress_is_waited_for_before_the_fresh_run(box, capsys):
    box.busy_polls = 3
    assert ds.main("final", "/fs") == 0
    kinds = ["poll" if "--property=ActiveState" in c else " ".join(c[1:3])
             for c in box.calls if c[0] == "systemctl" and "--property=Result" not in c]
    assert kinds == ["stop " + TIMER, "poll", "poll", "poll", "poll", "start " + SERVICE]
    assert len(box.slept) == 3
    assert len(box.starts()) == 1  # the run in progress is not counted as the final one
    assert "in progress" in capsys.readouterr().err


def test_a_run_in_progress_that_never_ends_is_a_failure_and_starts_nothing(box, capsys, monkeypatch):
    box.busy_polls = 10 ** 6
    box.last = OLD
    clock = iter(range(0, 10 ** 9, 3600))
    monkeypatch.setattr(ds.time, "monotonic", lambda: next(clock))
    assert ds.main("final", "/fs") == 1
    assert box.starts() == []
    assert box.timer_active
    err = capsys.readouterr().err
    assert "did not finish" in err and OLD in err


def test_an_earlier_last_success_with_a_failing_fresh_run_is_a_failure(box, capsys):
    box.last = OLD
    box.fresh_ok = False
    assert ds.main("final", "/fs") == 1
    err = capsys.readouterr().err
    assert "error: The final sync run failed" in err
    assert f"The last successful sync finished at {OLD}." in err
    assert "Traceback" not in err


def test_a_good_service_result_with_only_an_earlier_last_success_is_a_failure(box, capsys):
    """Judged by both: the result alone is not enough."""
    box.fresh_writes = OLD
    assert ds.main("final", "/fs") == 1
    err = capsys.readouterr().err
    assert "without recording a success" in err and OLD in err
    assert box.timer_active


def test_a_fresh_last_success_with_a_failed_service_result_is_a_failure(box, capsys):
    """Judged by both: a new time alone is not enough."""
    box.fresh_code = 1
    assert ds.main("final", "/fs") == 1
    assert "final sync run failed" in capsys.readouterr().err
    assert box.timer_active


def test_a_failure_with_no_success_ever_says_so(box, capsys):
    box.fresh_ok = False
    assert ds.main("final", "/fs") == 1
    assert "no successful sync" in capsys.readouterr().err


def test_after_a_failed_run_the_timer_is_running_again(box, capsys):
    box.fresh_ok = False
    assert ds.main("final", "/fs") == 1
    assert box.short() == [f"stop {TIMER}", f"start {SERVICE}", f"start {TIMER}"]
    assert box.timer_active
    assert "timer is running again" in capsys.readouterr().err


def test_after_a_run_that_times_out_the_timer_is_running_again(box, capsys):
    box.hang = True
    box.last = OLD
    assert ds.main("final", "/fs") == 1
    assert box.short()[-1] == f"start {TIMER}" and box.timer_active
    err = capsys.readouterr().err
    assert "did not finish" in err and OLD in err and "Traceback" not in err


def test_sync_not_installed_is_a_failure_that_says_so_and_runs_nothing(tmp_path, monkeypatch, capsys):
    box = FinalBox(tmp_path, monkeypatch, installed=False)
    assert ds.main("final", "/fs") == 1
    assert box.calls == []
    assert "Sync is not set up on this instance" in capsys.readouterr().err


@pytest.mark.parametrize("missing", ["spec", "service", "timer"])
def test_a_half_removed_sync_is_not_installed(box, capsys, missing):
    os.remove({"spec": ds.SPEC_PATH, "service": ds.service_path(), "timer": ds.timer_path()}[missing])
    assert ds.main("final", "/fs") == 1
    assert box.calls == []
    assert "not set up" in capsys.readouterr().err


@pytest.mark.parametrize("outcome", ["good", "bad", "hang"])
def test_the_key_is_in_no_command_environment_or_output(box, capsys, outcome):
    box.last = OLD
    box.fresh_ok = outcome == "good"
    box.hang = outcome == "hang"
    ds.main("final", "/fs")
    out = capsys.readouterr()
    assert KEY_VALUE not in out.out and KEY_VALUE not in out.err
    assert box.calls
    for cmd in box.calls:
        assert all(KEY_VALUE not in arg for arg in cmd), cmd
        assert ds.KEY_PATH not in cmd  # the key file is never read or passed here
    assert all(env is None or KEY_VALUE not in json.dumps(env) for env in box.envs)
    assert all(KEY_VALUE not in str(v) for k, v in os.environ.items())


def test_remote_tool_has_a_datasync_final_command(box, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["cloudgpu-remote", "--persistent-dir", "/fs",
                                     "datasync", "final"])
    remote_main.main()
    assert len(box.starts()) == 1

    box.fresh_ok = False
    with pytest.raises(SystemExit) as stop:
        remote_main.main()
    assert stop.value.code == 1
    assert "error: " in capsys.readouterr().err
