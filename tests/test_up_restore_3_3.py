"""`cloudgpu up` restores from the bucket between the sync install and the timer (3.3 part 3).

No real API, instance, bucket, key or rclone: the real CLI runs in a subprocess against the fake
Lambda, as tests/test_datasync_up_3_2.py does (its helpers are used here). One more ssh shim sits
in front: for the remote `datasync restore` call it records the call, writes FAKE_RESTORE_STDERR
to stderr and exits FAKE_RESTORE_EXIT; every other call goes on to the usual shims.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

from tests.fake_lambda import FakeLambda, run_cloudgpu
from tests.test_datasync_up_3_2 import (
    PLAN_ORDER, SYNC, WAYS, _assert_no_secret_anywhere, _assert_stopped_still_billing, _datasync,
    _rclone_conf, _shim_env, _steps,
)

RESTORE_SHIM = '''#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
while args and args[0].startswith("-"):
    args = args[2:]
command = " ".join(args[1:])
if " datasync restore" in command:
    with open(os.environ["FAKE_SHIM_LOG"], "a") as log:
        log.write(json.dumps({"tool": "ssh", "host": args[0], "command": command}) + "\\n")
    sys.stderr.write(os.environ.get("FAKE_RESTORE_STDERR", ""))
    sys.exit(int(os.environ.get("FAKE_RESTORE_EXIT") or 0))
os.execv(sys.executable, [sys.executable, os.environ["FAKE_NEXT_SSH"], *sys.argv[1:]])
'''
RESTORED = (">>> Restoring 2 paths from b2:bkt/small...\n"
            ">>> restore copied from b2:bkt/small: 1 restored, 1 with nothing in the bucket; the caller records it\n"
            ">>> Restored from b2:bkt/small; recorded 2026-11-16T10:00:00Z.\n")
ALREADY = ">>> This filesystem was already restored from b2:bkt/small at 2026-11-16T10:00:00Z; nothing downloaded.\n"
FAILED = (">>> failed apps/comfyui/output: rclone exited 1\n"
          ">>> restore failed for: apps/comfyui/output (1 restored). Nothing is recorded, so the next "
          "run restores again; what was restored stays.\n"
          "error: The restore from b2:bkt/small failed (the paths are named above). Nothing is "
          "recorded, so the next up restores again.\n")
STARTS = (" recover", " install", "systemctl", "provision.", " verify ")


def _up(fake, workdir, way="fresh", *, config_extra=SYNC, stderr=RESTORED, exit_code=0):
    state = None
    if way == "reused":
        state = {"instance_id": fake.add_instance()}
    elif way == "adopted":
        fake.add_instance()
    for name in ("shim-calls.jsonl", "ssh-seen.jsonl"):
        (workdir / name).unlink(missing_ok=True)
    env = _shim_env(workdir)
    front = workdir / "restore-shim"
    front.mkdir(exist_ok=True)
    (front / "ssh").write_text(RESTORE_SHIM.replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
    (front / "ssh").chmod(0o755)
    first = env["PATH"].split(os.pathsep)[0]
    env.update(PATH=os.pathsep.join([str(front), env["PATH"]]), FAKE_NEXT_SSH=os.path.join(first, "ssh"),
               FAKE_RESTORE_STDERR=stderr, FAKE_RESTORE_EXIT=str(exit_code))
    result = run_cloudgpu(fake, workdir, ["up", "--config", "box"], config_extra=config_extra,
                          shim_env=env, state=state)
    assert "Traceback" not in result.output, result.output
    return result


def _ssh(result):
    return [c["command"] for c in result.shim_calls if c["tool"] == "ssh"]


def _flat(result):
    return " ".join(result.output.split())


# (2)
@pytest.mark.parametrize("way", WAYS)
def test_restore_runs_after_install_and_cap_before_timer_and_any_service(tmp_path, way):
    """Catches: no restore call; the restore after the timer is running (half-downloaded files
    uploaded), before the cap is armed (an unbounded download on a billing machine), or after
    recovery, installs or restarts have started a service."""
    _rclone_conf(tmp_path)
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, way)
    assert result.returncode == 0, result.output
    assert _datasync(result) == ["setup", "restore", "start"], result.output
    assert _steps(result) == PLAN_ORDER, result.output  # no new named step
    calls = _ssh(result)
    [at] = [i for i, c in enumerate(calls) if " datasync restore" in c]
    assert calls[at].endswith(" datasync restore"), calls[at]
    assert any(" autoterminate" in c for c in calls[:at]), calls
    assert not [c for c in calls[:at] if any(s in c for s in STARTS)], calls[:at]
    assert calls[at + 1].endswith(" datasync start"), calls[at + 1]
    assert [c for c in calls[at + 2:] if c.endswith(" status")], "the run never got to 'make apps match'"
    _assert_no_secret_anywhere(result, tmp_path)


# (2) what the person sees
def test_up_says_it_is_restoring_and_from_where_and_shows_what_the_instance_said(tmp_path):
    """Catches: a silent restore (a long download looking like a hang), the remote unnamed, or
    the instance's own account of what was restored dropped with the captured stderr."""
    _rclone_conf(tmp_path)
    with FakeLambda() as fake:
        result = _up(fake, tmp_path)
    out = _flat(result)
    assert "Restoring from b2:bkt/small" in out, result.output
    assert "Restored from b2:bkt/small; recorded 2026-11-16T10:00:00Z." in out, result.output
    assert out.index("Restoring from b2:bkt/small") < out.index("Sync timer running."), result.output
    assert ">>>" not in result.output, result.output


# (5)
def test_already_restored_line_reaches_the_person(tmp_path):
    """Catches: a second `up` giving no sign that nothing was downloaded, and why."""
    _rclone_conf(tmp_path)
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, "reused", stderr=ALREADY)
    assert result.returncode == 0, result.output
    assert ("This filesystem was already restored from b2:bkt/small at 2026-11-16T10:00:00Z; "
            "nothing downloaded.") in _flat(result), result.output
    assert _datasync(result) == ["setup", "restore", "start"]


# (9)
@pytest.mark.parametrize("way", WAYS)
def test_a_failed_restore_fails_up_in_a_sentence_and_starts_nothing(tmp_path, way):
    """Catches: `up` going on to start services on a filesystem that is missing the person's
    files; the timer started by `up` after a failed restore; the failed paths not named; a
    traceback; the instance dropped from the state file."""
    _rclone_conf(tmp_path)
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, way, stderr=FAILED, exit_code=1)
        alive = fake.alive()
    _assert_stopped_still_billing(result, "Could not restore from b2:bkt/small")
    out = _flat(result)
    assert "restore failed for: apps/comfyui/output" in out, result.output
    assert "The restore from b2:bkt/small failed" in out, result.output
    assert "SSH command failed" not in out and "PYTHONPATH" not in out, result.output
    assert _datasync(result) == ["setup", "restore"], result.output
    calls = _ssh(result)
    at = max(i for i, c in enumerate(calls) if " datasync restore" in c)
    assert calls[at + 1:] == [], calls[at + 1:]
    assert not [c for c in calls if any(s in c for s in STARTS)], calls
    assert alive == ["fake-instance-1"]
    _assert_no_secret_anywhere(result, tmp_path)


def test_an_unreachable_remote_is_said_as_the_instance_said_it(tmp_path):
    """Catches: the reason (the remote could not be reached) lost on the way to the person."""
    _rclone_conf(tmp_path)
    said = (">>> restore failed: the remote could not be reached (rclone lsf b2:bkt exited 1); "
            "nothing was downloaded and nothing is recorded\n"
            "error: The restore from b2:bkt/small failed (the paths are named above). Nothing is "
            "recorded, so the next up restores again.\n")
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, stderr=said, exit_code=1)
    _assert_stopped_still_billing(result, "Could not restore from b2:bkt/small")
    assert "the remote could not be reached (rclone lsf b2:bkt exited 1)" in _flat(result), result.output


# (1)
@pytest.mark.parametrize("way", WAYS)
def test_without_sync_no_restore_call_and_nothing_said(tmp_path, way):
    """Catches: a restore attempted (and failing, with no spec on the instance) on a config that
    never asked for sync, or `up` talking about restoring."""
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, way, config_extra="", stderr=FAILED, exit_code=1)
    assert result.returncode == 0, result.output
    assert _datasync(result) == ["remove"], result.output
    assert "estor" not in result.output, result.output
    assert _steps(result) == PLAN_ORDER
