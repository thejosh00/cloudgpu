"""An instance past its auto-terminate deadline is said loudly, and `down` confirms it worked (R29).

No real API, instance or key. Most tests run the real CLI in a subprocess against the fake
Lambda; PACE divides its waits, so the 60s confirmation takes about two seconds. The last
section is in-process on a fake clock, to pin the real 60 seconds.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from cloudgpu import cli as cli_module
from cloudgpu.local import configs, lambda_api
from tests.fake_lambda import FakeLambda, run_cloudgpu

PACE = 30
TRACKED_ID = "fake-instance-1"
STATUS = ["status", "--config", "box"]
DOWN = ["down", "--config", "box", "--yes"]


def state_with_deadline(seconds_from_now: float) -> dict:
    return {
        "instance_id": TRACKED_ID,
        "ip": "192.0.2.1",
        "host": "ubuntu@192.0.2.1",
        "filesystem": "box-fs",
        "persistent_dir": "/lambda/nfs/box-fs",
        "auto_terminate_at": int(time.time() + seconds_from_now),
    }


OVERDUE_BY = 2 * 3600 + 5 * 60 + 20  # reads as "2h 05m"


def assert_overdue_error(output: str) -> None:
    flat = " ".join(output.split())
    assert f"Instance {TRACKED_ID} is OVERDUE" in flat, output
    assert "2h 05m ago" in flat, output
    assert "cloudgpu down -C box" in flat, output


# --- the overdue check -----------------------------------------------------------------------------


def test_status_exits_1_and_names_the_overdue_instance(tmp_path):
    """Catches: an instance billing past its cap with nothing on this machine saying so."""
    state = state_with_deadline(-OVERDUE_BY)
    with FakeLambda() as fake:
        fake.add_instance(status="active")
        result = run_cloudgpu(fake, tmp_path, STATUS, state=state, pace=PACE)
        alive = fake.alive()

    assert "Traceback" not in result.output, result.output
    assert result.returncode == 1, result.output
    assert_overdue_error(result.output)
    # It only reports: nothing terminated, nothing written.
    assert fake.count("POST", "/terminate") == 0
    assert alive == [TRACKED_ID]
    assert result.state_bytes == json.dumps(state).encode()


@pytest.mark.parametrize("status", ["terminated", "terminating", None], ids=["terminated", "terminating", "gone"])
def test_status_is_quiet_when_the_overdue_instance_is_going_or_gone(tmp_path, status):
    """Catches: crying overdue (and failing `status`) about an instance that is no longer billing."""
    with FakeLambda() as fake:
        if status:
            fake.add_instance(status=status, ip=False)
        result = run_cloudgpu(fake, tmp_path, STATUS, state=state_with_deadline(-OVERDUE_BY), pace=PACE)

    assert fake.count("GET", f"/instances/{TRACKED_ID}") == 1  # it did look
    assert "OVERDUE" not in result.output, result.output
    assert "deadline" not in result.output, result.output
    assert result.returncode == 0, result.output


def test_a_future_deadline_makes_no_lambda_read(tmp_path):
    """Catches: `status` starting to call (and depend on) the Lambda API on every run."""
    with FakeLambda() as fake:
        fake.add_instance(status="active")
        result = run_cloudgpu(fake, tmp_path, STATUS, state=state_with_deadline(3600), pace=PACE)

    assert result.returncode == 0, result.output
    assert fake.requests == []
    assert "OVERDUE" not in result.output


def test_no_deadline_makes_no_lambda_read(tmp_path):
    """Catches: a config with no cap being checked anyway."""
    state = state_with_deadline(0)
    del state["auto_terminate_at"]
    with FakeLambda() as fake:
        fake.add_instance(status="active")
        result = run_cloudgpu(fake, tmp_path, STATUS, state=state, pace=PACE)

    assert result.returncode == 0, result.output
    assert fake.requests == []


def test_an_unclear_read_says_the_deadline_passed_and_the_state_is_unknown(tmp_path):
    """Catches: a passed deadline going unmentioned because Lambda couldn't be read."""
    with FakeLambda(instance_reads_then="http_500") as fake:
        fake.add_instance(status="active")
        result = run_cloudgpu(fake, tmp_path, STATUS, state=state_with_deadline(-OVERDUE_BY), pace=PACE)

    flat = " ".join(result.output.split())
    assert "Traceback" not in result.output, result.output
    assert "deadline for config 'box' passed 2h 05m ago" in flat, result.output
    assert f"the state of instance {TRACKED_ID} couldn't be read" in flat, result.output
    assert "OVERDUE" not in result.output


@pytest.mark.parametrize(
    "command",
    [["up", "--config", "box"], ["autoterminate", "6", "--config", "box"], DOWN],
    ids=["up", "autoterminate", "down"],
)
def test_other_commands_print_the_overdue_error_and_keep_their_exit_code(tmp_path, command):
    """Catches: the overdue instance only being mentioned by `status`, or the report breaking
    the command that is about to fix it."""
    with FakeLambda() as fake:
        fake.add_instance(status="active")
        result = run_cloudgpu(fake, tmp_path, command, state=state_with_deadline(-OVERDUE_BY), pace=PACE)

    assert "Traceback" not in result.output, result.output
    assert_overdue_error(result.output)
    assert result.returncode == 0, result.output
    assert fake.launches == []


# --- `down` confirms ---------------------------------------------------------------------------------


def test_down_keeps_the_state_when_the_termination_does_not_take(tmp_path):
    """Catches: `down` forgetting an instance that is still running and billing."""
    state = state_with_deadline(3600)
    started = time.monotonic()
    with FakeLambda(terminate_takes=False) as fake:
        fake.add_instance(status="active")
        result = run_cloudgpu(fake, tmp_path, DOWN, state=state, pace=PACE, timeout=30)
        alive = fake.alive()
    took = time.monotonic() - started

    flat = " ".join(result.output.split())
    assert "Traceback" not in result.output, result.output
    assert result.returncode == 1, result.output
    assert fake.count("POST", "/terminate") == 1
    assert alive == [TRACKED_ID]
    assert f"instance {TRACKED_ID} is still listed as running" in flat, result.output
    assert "cloudgpu down -C box" in flat, result.output
    assert result.state_bytes == json.dumps(state).encode()
    assert took < 20  # 60s at PACE 30 is 2s; it must give up, not wait forever


def test_down_delete_filesystem_stops_before_the_filesystem_when_unconfirmed(tmp_path):
    """Catches: the filesystem step running (or state being cleared) after a termination that didn't take."""
    state = state_with_deadline(3600)
    with FakeLambda(terminate_takes=False) as fake:
        fake.add_instance(status="active")
        result = run_cloudgpu(fake, tmp_path, [*DOWN, "--delete-filesystem"], state=state, pace=PACE, timeout=30)

    assert result.returncode == 1, result.output
    assert result.state_bytes == json.dumps(state).encode()
    assert fake.deleted_filesystems == []
    assert "Waiting for the instance to terminate" not in result.output


def test_down_clears_the_state_once_the_instance_is_terminating(tmp_path, monkeypatch):
    """Catches: `down` waiting for fully gone (minutes) before it lets go, or not polling at all."""
    _in_process_repo(tmp_path, monkeypatch)
    clock = _Clock(monkeypatch)
    answers = iter([{"status": "active", "ip": "1.2.3.4"}, {"status": "active", "ip": "1.2.3.4"},
                    {"status": "terminating"}])
    monkeypatch.setattr(lambda_api, "get_instance", lambda _: next(answers))
    monkeypatch.setattr(lambda_api, "terminate_instances", lambda ids: {})

    result = CliRunner().invoke(cli_module.down, ["--yes", "-C", "c"])

    assert result.exit_code == 0, result.output
    assert configs.load_state("c") == {}
    assert clock.slept == [5]  # one poll saw it still active, the next saw it terminating


def test_down_gives_up_after_60_seconds_and_keeps_the_state(tmp_path, monkeypatch):
    """Catches: the confirmation waiting forever, or far less than the 60 seconds it promises."""
    _in_process_repo(tmp_path, monkeypatch)
    before = (tmp_path / "configs" / "c.state.json").read_bytes()
    clock = _Clock(monkeypatch)
    monkeypatch.setattr(lambda_api, "get_instance", lambda _: {"status": "active", "ip": "1.2.3.4"})
    monkeypatch.setattr(lambda_api, "terminate_instances", lambda ids: {})

    result = CliRunner().invoke(cli_module.down, ["--yes", "-C", "c"])

    assert result.exit_code == 1, result.output
    assert sum(clock.slept) == 60
    assert (tmp_path / "configs" / "c.state.json").read_bytes() == before


def test_wait_terminated_without_going_still_needs_gone(monkeypatch):
    """Catches: the filesystem wait starting to accept 'terminating', when the filesystem is still mounted."""
    clock = _Clock(monkeypatch)
    monkeypatch.setattr(lambda_api, "get_instance", lambda _: {"status": "terminating"})
    assert cli_module._wait_terminated("i-1", timeout=30, poll=10) is False
    assert cli_module._wait_terminated("i-1", timeout=30, poll=10, going=True) is True
    assert clock.slept == [10, 10, 10]


# --- the fake, and its scripted run --------------------------------------------------------------


def test_the_scripted_fake_run_does_up_then_down_and_leaves_nothing_alive():
    """Catches: the one-command run no longer covering `down`, or `down` leaving the instance alive."""
    proc = subprocess.run(
        [sys.executable, "-m", "tests.fake_lambda"], cwd=Path(__file__).parent.parent,
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "cloudgpu up exit code: 0" in proc.stdout
    assert "cloudgpu down exit code: 0" in proc.stdout
    assert "fake's live instances:   []" in proc.stdout
    assert "local state tracks:      None" in proc.stdout


# --- helpers ---------------------------------------------------------------------------------------


class _Clock:
    """time.time and time.sleep with nothing really slept."""

    def __init__(self, monkeypatch):
        self.now = 1000.0
        self.slept = []
        monkeypatch.setattr("time.time", lambda: self.now)
        monkeypatch.setattr("time.monotonic", lambda: self.now)
        monkeypatch.setattr("time.sleep", self.sleep)

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def _in_process_repo(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "c.toml").write_text('gpu = ["gh200"]\nssh_key = "mini"\nfilesystem = "prof"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CLOUDGPU_TEST_LAMBDA_API_BASE", raising=False)
    configs.save_state("c", {
        "instance_id": "i-1", "ip": "1.2.3.4", "host": "ubuntu@1.2.3.4",
        "filesystem": "prof", "persistent_dir": "/lambda/nfs/prof",
    })
