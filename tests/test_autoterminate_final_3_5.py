"""Item 3.5: the billing guard's final sync, limited to 10 minutes.

Nothing here runs systemctl, touches the Lambda API or a bucket: ``subprocess.run`` is a
recorder with a made-up sync service, ``terminate`` is recorded, and time is a fake clock
that only moves when the code sleeps or a recorded command takes time.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time as real_time
import urllib.error
from pathlib import Path

import pytest

from cloudgpu.remote import __main__ as remote_main
from cloudgpu.remote import autoterminate as at
from cloudgpu.remote import datasync_setup as ds

START = 1_800_000_000.0          # the fake clock's start (seconds since the epoch)
LIMIT = 10 * 60                  # the plan's limit, from deciding to sync to the request
MARGIN = 2                       # what the request may come after the limit, at most
OLD = "2026-11-01T08:00:00Z"
LAMBDA_KEY = "lambda-KEY-must-not-leak-1234"
SYNC_KEY_TEXT = "[b2]\ntype = b2\naccount = acct\nkey = sync-KEY-must-not-leak-5678\n"


def stamp(when: float) -> str:
    return real_time.strftime("%Y-%m-%dT%H:%M:%SZ", real_time.gmtime(when))


class FakeTime:
    """The parts of ``time`` the guard uses, on a clock tests move."""

    def __init__(self):
        self.now = START
        self.slept: list[float] = []
        self.strftime, self.gmtime, self.strptime = real_time.strftime, real_time.gmtime, real_time.strptime
        self.localtime = real_time.localtime

    def time(self):
        return self.now

    def monotonic(self):
        return self.now - START + 1000.0

    def sleep(self, seconds):
        assert seconds > 0
        self.slept.append(seconds)
        self.now += seconds


class Guard:
    """An armed instance: config, sync files, a sync service and a recorded Lambda API."""

    def __init__(self, tmp_path, monkeypatch, *, installed=True, deadline=None):
        self.clock = FakeTime()
        monkeypatch.setattr(at, "time", self.clock)
        etc = tmp_path / "etc"
        etc.mkdir()
        self.persistent = tmp_path / "nfs" / "fs1"
        (self.persistent / "cloudgpu").mkdir(parents=True)
        self.state = tmp_path / "var-lib-cloudgpu-sync"
        self.state.mkdir()
        paths = {
            "CONF_PATH": etc / "autoterminate.json",
            "FINAL_RECORD_PATH": etc / "autoterminate-final.json",
            "SYNC_SPEC_PATH": etc / "datasync.json",
            "SYNC_SERVICE_PATH": etc / "cloudgpu-datasync.service",
            "SYNC_TIMER_PATH": etc / "cloudgpu-datasync.timer",
            "SYNC_LAST_SUCCESS": self.state / "last-success",
        }
        for name, path in paths.items():
            monkeypatch.setattr(at, name, str(path), raising=False)
        self.paths = paths
        (etc / "rclone.conf").write_text(SYNC_KEY_TEXT)
        if installed:
            for name in ("SYNC_SPEC_PATH", "SYNC_SERVICE_PATH", "SYNC_TIMER_PATH"):
                paths[name].write_text("{}")
        self.arm(deadline if deadline is not None else START - 60)

        self.calls: list[list[str]] = []
        self.call_times: list[float] = []
        self.kwargs: list[dict] = []
        self.events: list[tuple[str, float]] = []
        self.busy_until = 0.0        # a run in progress until this time
        self.run_seconds = 30.0      # how long a fresh run takes
        self.fresh_ok = True
        self.fresh_writes: str | None = None   # last-success a good run writes (default: now)
        self.hang = False            # the fresh run never ends
        self.result = "success"
        self.terminate_fails = False
        self.last = None
        monkeypatch.setattr(at.subprocess, "run", self.run)
        monkeypatch.setattr(at, "terminate", self.terminate)

    def arm(self, deadline):
        conf = {"instance_id": "i-abc", "deadline_epoch": deadline, "api_key": LAMBDA_KEY,
                "persistent_dir": str(self.persistent)}
        self.paths["CONF_PATH"].write_text(json.dumps(conf))

    @property
    def last(self):
        path = self.paths["SYNC_LAST_SUCCESS"]
        return path.read_text().strip() if path.exists() else None

    @last.setter
    def last(self, value):
        path = self.paths["SYNC_LAST_SUCCESS"]
        if value is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(value + "\n")

    def run(self, cmd, **kwargs):
        assert cmd[0] == "systemctl", f"only systemctl may be run, got {cmd}"
        self.calls.append(list(cmd))
        self.kwargs.append(kwargs)
        self.call_times.append(self.clock.now)
        timeout = kwargs.get("timeout")
        assert timeout is not None and timeout > 0, "every call must be bounded"
        timer, service = f"{at.SYNC_UNIT}.timer", f"{at.SYNC_UNIT}.service"
        rest = cmd[1:]
        if rest[0] == "show":
            assert rest[-1] == service
            self.clock.now += 0.1
            if "--property=ActiveState" in rest:
                state = "activating" if self.clock.now < self.busy_until else "inactive"
                return subprocess.CompletedProcess(cmd, 0, stdout=state + "\n", stderr="")
            assert "--property=Result" in rest
            return subprocess.CompletedProcess(cmd, 0, stdout=self.result + "\n", stderr="")
        if rest == ["stop", timer]:
            self.events.append(("stop-timer", self.clock.now))
            return subprocess.CompletedProcess(cmd, 0)
        if rest == ["start", service]:
            self.events.append(("start-sync", self.clock.now))
            if self.hang or self.run_seconds > timeout:
                self.clock.now += timeout
                raise subprocess.TimeoutExpired(cmd, timeout)
            self.clock.now += self.run_seconds
            if self.fresh_ok:
                self.result = "success"
                self.last = self.fresh_writes or stamp(self.clock.now)
                return subprocess.CompletedProcess(cmd, 0)
            self.result = "exit-code"
            return subprocess.CompletedProcess(cmd, 1)
        raise AssertionError(f"unexpected command {cmd}")

    def terminate(self, conf):
        self.events.append(("terminate", self.clock.now))
        if self.terminate_fails:
            raise urllib.error.URLError("api down")

    def sync_calls(self):
        return [c for c in self.calls if at.SYNC_UNIT in c[-1]]

    def terminated_at(self) -> list[float]:
        return [t for name, t in self.events if name == "terminate"]

    def guard_log(self) -> list[str]:
        path = self.persistent / "cloudgpu" / "guard.log"
        return path.read_text().splitlines() if path.exists() else []


@pytest.fixture
def guard(tmp_path, monkeypatch):
    return Guard(tmp_path, monkeypatch)


# --- (1) and (2): when ---------------------------------------------------------------


def test_before_the_deadline_nothing_new_happens(tmp_path, monkeypatch):
    # Catches: a sync started (or the terminate requested) while the deadline is ahead.
    g = Guard(tmp_path, monkeypatch, deadline=START + 3600)
    assert at.check() == 0
    assert g.calls == [] and g.events == [] and g.guard_log() == []


def test_no_sync_installed_terminates_as_before_and_says_so(tmp_path, monkeypatch, capsys):
    # Catches: a guard that touches the sync unit, or waits, when sync isn't installed.
    g = Guard(tmp_path, monkeypatch, installed=False)
    assert at.check() == 0
    assert g.calls == []
    assert g.terminated_at() == [START]
    assert g.guard_log() == [f"{stamp(START)} guard terminating instance i-abc: "
                             "final sync no sync installed"]
    assert "no sync installed" in capsys.readouterr().err


# --- (3): a good final sync ----------------------------------------------------------


def test_a_good_final_sync_runs_before_the_request_and_leaves_the_timer_stopped(guard):
    # Catches: the request sent before the sync, a sync run without stopping the timer
    # first, or the timer started again after a good sync.
    assert at.check() == 0
    service, timer = f"{at.SYNC_UNIT}.service", f"{at.SYNC_UNIT}.timer"
    assert guard.calls == [
        ["systemctl", "stop", timer],
        ["systemctl", "show", "--property=ActiveState", "--value", service],
        ["systemctl", "start", service],
        ["systemctl", "show", "--property=Result", "--value", service],
    ]
    assert [name for name, _ in guard.events] == ["stop-timer", "start-sync", "terminate"]
    synced = guard.last
    assert guard.guard_log() == [
        f"{stamp(guard.clock.now)} guard terminating instance i-abc: final sync synced at {synced}"]


# --- (4): a run in progress ----------------------------------------------------------


def test_a_run_in_progress_is_waited_for_and_counts_against_the_limit(guard):
    # Catches: joining the run in progress instead of a fresh one, or a wait that resets
    # the limit (the fresh run here would end past it if the wait were not counted).
    guard.busy_until = START + 400
    guard.run_seconds = 300
    assert at.check() == 0
    starts = [t for name, t in guard.events if name == "start-sync"]
    assert starts and starts[0] >= START + 400
    assert guard.terminated_at()[0] <= START + LIMIT + MARGIN
    assert "cut off at the time limit" in guard.guard_log()[0]


# --- (5): failures -------------------------------------------------------------------


def test_a_failed_fresh_run_still_terminates_and_logs_the_last_success(guard):
    # Catches: termination skipped after a failed sync, or the timer restarted (as down does).
    guard.last = OLD
    guard.fresh_ok = False
    assert at.check() == 0
    assert len(guard.terminated_at()) == 1
    assert ["systemctl", "start", f"{at.SYNC_UNIT}.timer"] not in guard.calls
    line = guard.guard_log()[0]
    assert "final sync failed: the run's result was exit-code" in line
    assert f"last success {OLD}" in line


def test_a_run_that_only_has_an_earlier_success_is_a_failure(guard):
    # Catches: counting a success the fresh run did not write.
    guard.fresh_writes = OLD
    assert at.check() == 0
    assert len(guard.terminated_at()) == 1
    line = guard.guard_log()[0]
    assert "failed: the run recorded no new success" in line and OLD in line


# --- (6): the limit ------------------------------------------------------------------


def test_a_run_in_progress_past_the_limit_is_cut_off(guard):
    # Catches: a wait for the run in progress that is not bounded by the 10 minutes.
    guard.last = OLD
    guard.busy_until = START + 6 * 3600
    assert at.check() == 0
    assert ["systemctl", "start", f"{at.SYNC_UNIT}.service"] not in guard.calls
    assert guard.terminated_at()[0] <= START + LIMIT + MARGIN
    assert f"final sync cut off at the time limit; last success {OLD}" in guard.guard_log()[0]


def test_a_fresh_run_still_going_at_the_limit_is_cut_off(guard):
    # Catches: the fresh run given down's 6-hour wait instead of what is left of the limit.
    guard.hang = True
    assert at.check() == 0
    assert guard.terminated_at()[0] <= START + LIMIT + MARGIN
    assert "cut off at the time limit; last success never" in guard.guard_log()[0]


def test_no_systemctl_call_may_outlast_the_limit(guard):
    # Catches: a call given a fixed timeout that reaches past what is left of the limit.
    guard.busy_until = START + LIMIT - 20
    guard.hang = True
    at.check()
    end = START + LIMIT
    for kwargs, cmd, at_time in zip(guard.kwargs, guard.calls, guard.call_times):
        assert at_time + kwargs["timeout"] <= end, cmd
    assert guard.terminated_at()[0] <= end + MARGIN


# --- (7): once per deadline ----------------------------------------------------------


def test_a_retried_request_does_not_sync_again_until_rearmed(guard):
    # Catches: every 5-minute retry running (and waiting on) another final sync.
    guard.terminate_fails = True
    assert at.check() == 1
    first = len(guard.sync_calls())
    assert first > 0
    record = json.loads(guard.paths["FINAL_RECORD_PATH"].read_text())
    assert record["outcome"].startswith("synced at")
    assert os.stat(guard.paths["FINAL_RECORD_PATH"]).st_mode & 0o077 == 0  # root-only

    guard.clock.now += 300
    assert at.check() == 1
    assert len(guard.sync_calls()) == first
    assert len(guard.terminated_at()) == 2
    assert len(guard.guard_log()) == 1

    guard.arm(guard.clock.now - 1)   # re-armed with a new deadline, which has passed
    guard.terminate_fails = False
    assert at.check() == 0
    assert len(guard.sync_calls()) == 2 * first
    assert len(guard.guard_log()) == 2


# --- (8): the persistent log ---------------------------------------------------------


def test_an_unwritable_guard_log_never_blocks_termination(guard):
    # Catches: an OSError from the log write escaping and skipping the request.
    os.rmdir(guard.persistent / "cloudgpu")
    assert at.check() == 0
    assert len(guard.terminated_at()) == 1


def test_the_guard_log_follows_no_symlink(guard, tmp_path):
    # Catches: root appending through a link the app user planted on the shared filesystem.
    target = tmp_path / "elsewhere"
    target.write_text("untouched\n")
    (guard.persistent / "cloudgpu" / "guard.log").symlink_to(target)
    assert at.check() == 0
    assert target.read_text() == "untouched\n"
    assert len(guard.terminated_at()) == 1


def test_a_fifo_in_place_of_the_guard_log_does_not_hold_termination(guard):
    # Catches: an open that blocks for ever on a FIFO the app user made.
    os.mkfifo(guard.persistent / "cloudgpu" / "guard.log")
    assert at.check() == 0
    assert len(guard.terminated_at()) == 1


def test_a_hung_guard_log_write_is_given_up_on(guard, monkeypatch):
    # Catches: a network filesystem that hangs the write also hanging the request.
    monkeypatch.setattr(at, "GUARD_LOG_WAIT_SECONDS", 0.2)
    monkeypatch.setattr(at, "append_guard_log", lambda *a: real_time.sleep(5))
    began = real_time.monotonic()
    assert at.check() == 0
    assert real_time.monotonic() - began < 3
    assert len(guard.terminated_at()) == 1


# --- (9): no key anywhere -------------------------------------------------------------


def test_neither_key_appears_in_a_command_environment_or_log(guard, capsys):
    guard.terminate_fails = True
    at.check()
    guard.fresh_ok = False
    guard.arm(guard.clock.now - 1)
    at.check()
    err = capsys.readouterr().err
    seen = [json.dumps(guard.calls), json.dumps([k.get("env") for k in guard.kwargs]), err,
            "\n".join(guard.guard_log()), guard.paths["FINAL_RECORD_PATH"].read_text()]
    for text in seen:
        assert LAMBDA_KEY not in text
        assert "sync-KEY-must-not-leak" not in text
        assert "rclone.conf" not in text


# --- arming ---------------------------------------------------------------------------


def test_the_remote_tool_arms_with_the_persistent_dir(tmp_path, monkeypatch):
    # Catches: the guard never learning where the persistent log goes.
    seen = {}
    monkeypatch.setattr(at, "arm", lambda *a: seen.setdefault("args", a))
    state = remote_main.State(str(tmp_path))
    args = remote_main.argparse.Namespace(hours=2.0, instance_id="i-abc", key_file="k.env")
    remote_main.cmd_autoterminate(state, args)
    assert seen["args"] == (2.0, "i-abc", "k.env", str(tmp_path))


def test_arm_stores_the_persistent_dir_and_disarm_removes_the_record(tmp_path, monkeypatch):
    installs, sudo = {}, []
    monkeypatch.setattr(at, "_sudo", lambda *cmd: sudo.append(list(cmd)))
    monkeypatch.setattr(at, "_sudo_install", lambda c, dest, mode: installs.update({dest: c}))
    key = tmp_path / "k.env"
    key.write_text("LAMBDA_API_KEY=x\n")
    at.arm(1.0, "i-abc", str(key), "/lambda/nfs/fs1")
    assert json.loads(installs[at.CONF_PATH])["persistent_dir"] == "/lambda/nfs/fs1"

    conf = tmp_path / "autoterminate.json"
    conf.write_text("{}")
    monkeypatch.setattr(at, "CONF_PATH", str(conf))
    monkeypatch.setattr(at.subprocess, "run", lambda cmd, **k: None)
    sudo.clear()
    at.disarm()
    assert at.FINAL_RECORD_PATH in next(c for c in sudo if c[:2] == ["rm", "-f"])


def test_a_config_armed_before_this_change_still_terminates(guard):
    # Catches: a KeyError on configs written without persistent_dir.
    conf = json.loads(guard.paths["CONF_PATH"].read_text())
    del conf["persistent_dir"]
    guard.paths["CONF_PATH"].write_text(json.dumps(conf))
    assert at.check() == 0
    assert len(guard.terminated_at()) == 1


# --- (10) and (11): standalone and in step with the sync ---------------------------------


def test_the_limit_is_the_plans_ten_minutes():
    assert at.FINAL_LIMIT_SECONDS == LIMIT


def test_the_sync_names_match_what_datasync_setup_installs():
    # Catches: the guard's copy of the paths drifting from the sync's, so it never syncs.
    assert at.SYNC_UNIT == ds.UNIT_NAME
    assert at.SYNC_SPEC_PATH == ds.SPEC_PATH
    assert at.SYNC_SERVICE_PATH == ds.service_path()
    assert at.SYNC_TIMER_PATH == ds.timer_path()
    assert at.SYNC_LAST_SUCCESS == ds.LAST_SUCCESS_FILE
    assert at.ETC_DIR == ds.ETC_DIR


def test_the_etc_copy_runs_check_standalone(tmp_path):
    # Catches: an import of cloudgpu creeping into the file the /etc copy runs from.
    source = Path(at.__file__).read_text()
    copy = tmp_path / "etc" / "autoterminate.py"
    copy.parent.mkdir()
    copy.write_text(source)
    persistent = tmp_path / "fs"
    (persistent / "cloudgpu").mkdir(parents=True)
    for name in ("datasync.json", "svc", "tmr"):
        (tmp_path / "etc" / name).write_text("{}")
    conf = tmp_path / "etc" / "autoterminate.json"
    conf.write_text(json.dumps({"instance_id": "i-abc", "deadline_epoch": 1, "api_key": "k",
                                "persistent_dir": str(persistent)}))
    driver = tmp_path / "driver.py"
    driver.write_text(textwrap.dedent(f"""
        import importlib.util, json, subprocess, sys, time

        class Block:
            def find_spec(self, name, path=None, target=None):
                if name == "cloudgpu" or name.startswith("cloudgpu."):
                    raise ImportError("the package must not be imported")
                return None
        sys.meta_path.insert(0, Block())
        spec = importlib.util.spec_from_file_location("autoterminate", {str(copy)!r})
        at = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(at)
        etc = {str(tmp_path / "etc")!r}
        at.CONF_PATH = etc + "/autoterminate.json"
        at.FINAL_RECORD_PATH = etc + "/final.json"
        at.SYNC_SPEC_PATH, at.SYNC_SERVICE_PATH, at.SYNC_TIMER_PATH = (
            etc + "/datasync.json", etc + "/svc", etc + "/tmr")
        at.SYNC_LAST_SUCCESS = etc + "/last-success"
        calls, asked = [], []
        def run(cmd, **kw):
            calls.append(cmd)
            if cmd[1] == "show":
                out = "success" if "--property=Result" in cmd else "inactive"
                return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
            if cmd[1] == "start":
                open(at.SYNC_LAST_SUCCESS, "w").write(
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 1)))
            return subprocess.CompletedProcess(cmd, 0)
        at.subprocess.run = run
        at.terminate = lambda conf: asked.append(conf["instance_id"])
        code = at.check()
        print(json.dumps({{"code": code, "calls": calls, "asked": asked,
                          "loaded": sorted(m for m in sys.modules if m.startswith("cloudgpu"))}}))
    """))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    done = subprocess.run([sys.executable, "-I", str(driver)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["code"] == 0 and out["asked"] == ["i-abc"] and out["loaded"] == []
    assert ["systemctl", "start", "cloudgpu-datasync.service"] in out["calls"]
    assert "final sync synced at" in (persistent / "cloudgpu" / "guard.log").read_text()
