"""API errors `down` and `up` must surface, not swallow (plan item 0.12).

The real CLI in a subprocess against the fake Lambda: no real API, instance or key. PACE divides
timeouts and retry delays, so nothing really sleeps through them.
"""

from __future__ import annotations

import json

import pytest

from tests.fake_lambda import FakeLambda, run_cloudgpu

PACE = 30
DOWN = ["down", "--config", "box", "--yes"]
DOWN_DELETE = [*DOWN, "--delete-filesystem"]
TRACKED_ID = "fake-instance-1"


def tracked_state(instance_id=TRACKED_ID):
    return {
        "instance_id": instance_id,
        "ip": "192.0.2.1",
        "host": "ubuntu@192.0.2.1",
        "filesystem": "box-fs",
        "provisioned_profiles": ["alpha"],
    }


def no_traceback(result):
    assert "Traceback" not in result.output, result.output


# --- 1. a failed filesystem lookup is not "no filesystem" -----------------------------------------


@pytest.mark.parametrize("mode", ["http_503", "http_401", "drop"])
def test_down_delete_filesystem_with_a_failed_lookup_says_it_was_not_deleted(tmp_path, mode):
    """Catches: `_find_filesystem` swallowing the error, so `down` says "nothing to delete" and exits 0."""
    with FakeLambda(filesystem_reads_then=mode) as fake:
        result = run_cloudgpu(fake, tmp_path, DOWN_DELETE, pace=PACE)

    no_traceback(result)
    assert result.returncode != 0, result.output
    assert "nothing to delete" not in result.output
    assert "box-fs" in result.output
    assert "not deleted" in result.output
    assert fake.count("DELETE", "/filesystems/fs-box-fs") == 0
    assert fake.filesystems == ["box-fs"]


def test_down_delete_filesystem_when_the_list_lacks_it_is_nothing_to_delete(tmp_path):
    """Catches: a clean lookup that doesn't find the filesystem turning into a failure."""
    with FakeLambda(filesystem=None) as fake:
        result = run_cloudgpu(fake, tmp_path, DOWN_DELETE, pace=PACE)

    no_traceback(result)
    assert result.returncode == 0, result.output
    assert "nothing to delete" in result.output
    assert fake.count("DELETE", "/filesystems/") == 0


# --- 2. down reads the tracked id before terminating it -------------------------------------------


def test_down_with_a_tracked_id_lambda_no_longer_has_clears_it(tmp_path):
    """Catches: terminating an id Lambda has lost, failing, and keeping the stale id forever."""
    with FakeLambda() as fake:
        result = run_cloudgpu(fake, tmp_path, DOWN, state=tracked_state("lost-id"), pace=PACE)

    no_traceback(result)
    assert result.returncode == 0, result.output
    assert fake.count("POST", "/terminate") == 0
    assert "lost-id" in result.output
    assert "already gone" in result.output
    assert "instance_id" not in result.state


def test_down_with_a_tracked_id_whose_read_fails_stops_and_keeps_it(tmp_path):
    """Catches: terminating, or forgetting, an instance whose state Lambda couldn't tell us."""
    state = tracked_state()
    with FakeLambda(instance_reads_then="http_503") as fake:
        fake.add_instance(status="active")
        result = run_cloudgpu(fake, tmp_path, DOWN, state=state, pace=PACE)
        alive = fake.alive()

    no_traceback(result)
    assert result.returncode != 0, result.output
    assert fake.count("POST", "/terminate") == 0
    assert TRACKED_ID in result.output
    assert "Couldn't tell" in result.output
    assert result.state_bytes == json.dumps(state).encode()
    assert alive == [TRACKED_ID]


def test_down_with_a_live_tracked_id_terminates_it(tmp_path):
    """Catches: the new read getting in the way of terminating what is really there."""
    with FakeLambda() as fake:
        fake.add_instance(status="active")
        result = run_cloudgpu(fake, tmp_path, DOWN, state=tracked_state(), pace=PACE)
        alive = fake.alive()

    no_traceback(result)
    assert result.returncode == 0, result.output
    assert fake.count("POST", "/terminate") == 1
    assert f"Termination requested for {TRACKED_ID}" in result.output
    assert alive == []
    assert result.state == {}


def test_down_delete_filesystem_with_a_gone_tracked_id_deletes_without_waiting(tmp_path):
    """Catches: an already-gone instance failing the terminate, or being polled before the delete."""
    with FakeLambda() as fake:
        result = run_cloudgpu(fake, tmp_path, DOWN_DELETE, state=tracked_state("lost-id"), pace=PACE)

    no_traceback(result)
    assert result.returncode == 0, result.output
    assert fake.count("POST", "/terminate") == 0
    assert fake.count("GET", "/instances/lost-id") == 1  # the one read; no wait for termination
    assert "Waiting for the instance to terminate" not in result.output
    assert fake.deleted_filesystems == ["fs-box-fs"]
    assert result.state == {}


# --- 3. the "not usable" stop says how to terminate -----------------------------------------------


@pytest.mark.parametrize("tracked", [True, False], ids=["saved", "mounting"])
def test_up_with_an_unhealthy_instance_names_up_and_down(tmp_path, tracked):
    """Catches: "Run this again once it's active" with no way named to stop an unhealthy instance billing."""
    with FakeLambda() as fake:
        fake.add_instance(status="unhealthy")
        result = run_cloudgpu(
            fake, tmp_path, ["up", "--config", "box"], state=tracked_state() if tracked else None, pace=PACE,
        )

    no_traceback(result)
    assert result.returncode != 0, result.output
    assert fake.launches == []
    assert TRACKED_ID in result.output
    assert "cloudgpu up -C box" in result.output
    assert "cloudgpu down -C box" in result.output
