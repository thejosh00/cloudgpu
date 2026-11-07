"""Gone, alive or unknown: `cloudgpu up` and `down` against the fake Lambda.

No real API, instance or key. Everything here but the last section runs the real CLI in a
subprocess. PACE divides its timeouts and retry delays, so nothing really sleeps through them.
"""

from __future__ import annotations

import json

import pytest

from cloudgpu import cli as cli_module
from cloudgpu.local import lambda_api
from tests.fake_lambda import FakeLambda, run_cloudgpu, run_up

PACE = 30  # a 2s request timeout, and retries a fraction of a second apart
DOWN = ["down", "--config", "box", "--yes"]
TRACKED_ID = "fake-instance-1"
ATTEMPTS = 3


def tracked_state(instance_id=TRACKED_ID):
    return {
        "instance_id": instance_id,
        "ip": "192.0.2.1",
        "host": "ubuntu@192.0.2.1",
        "filesystem": "box-fs",
        "provisioned_profiles": ["alpha", "beta"],
    }


def assert_stopped_unchanged(fake, result, state):
    """Non-zero, nothing launched or created, and the state file exactly as it was."""
    assert result.returncode != 0, result.output
    assert fake.launches == [], result.output
    assert ("POST", "/api/v1/filesystems") not in fake.requests
    assert result.state_bytes == json.dumps(state).encode()
    assert "Traceback" not in result.output, result.output
    assert "Nothing was changed" in result.output, result.output
    assert "cloudgpu lambda instances" in result.output, result.output


# --- a tracked instance, and a read that doesn't say ------------------------------------------


@pytest.mark.parametrize(
    "mode",
    ["http_500", "http_500_not_found_code", "http_502", "http_503", "http_429", "http_401", "http_403", "404_no_code", "404_other_code",
     "200_html", "200_no_status", "200_unlisted_status", "drop", "stall"],
)
def test_up_stops_unchanged_when_the_tracked_read_is_unclear(tmp_path, mode):
    """Catches: any failed read counting as gone, which loses state and launches a second instance."""
    state = tracked_state()
    with FakeLambda(instance_reads_then=mode) as fake:
        fake.add_instance()
        result = run_up(fake, tmp_path, state=state, pace=PACE)

    assert_stopped_unchanged(fake, result, state)


@pytest.mark.parametrize("mode", ["http_500", "http_502", "http_503", "http_429", "404_no_code", "drop", "200_html"])
def test_a_read_that_might_clear_is_tried_three_times_and_says_so(tmp_path, mode):
    """Catches: giving up on the first blip, or a message that doesn't say what was tried."""
    state = tracked_state()
    with FakeLambda(instance_reads_then=mode) as fake:
        fake.add_instance()
        result = run_up(fake, tmp_path, state=state, pace=PACE)

    assert fake.count("GET", f"/instances/{TRACKED_ID}") == ATTEMPTS
    assert f"Tried {ATTEMPTS} times over" in result.output, result.output


@pytest.mark.parametrize("mode", ["http_401", "http_403"])
def test_a_refused_key_is_not_retried(tmp_path, mode):
    """Catches: hammering the API with a key it has already refused."""
    state = tracked_state()
    with FakeLambda(instance_reads_then=mode) as fake:
        fake.add_instance()
        result = run_up(fake, tmp_path, state=state, pace=PACE)

    assert fake.count("GET", f"/instances/{TRACKED_ID}") == 1
    assert "Tried once" in result.output, result.output
    assert_stopped_unchanged(fake, result, state)


def test_no_key_stops_unchanged_with_no_retries(tmp_path):
    """Catches: a missing key reading as "the instance is gone"."""
    state = tracked_state()
    with FakeLambda() as fake:
        fake.add_instance()
        result = run_up(fake, tmp_path, state=state, api_key=None, pace=PACE)

    assert fake.requests == []  # nothing can be sent without a key
    assert "retrying" not in result.output
    assert "Tried once" in result.output, result.output
    assert "LAMBDA_API_KEY" in result.output
    assert_stopped_unchanged(fake, result, state)


# --- alive, but not ready ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,ip", [("booting", False), ("active", False), ("unhealthy", True), ("terminating", True), ("preempted", False)]
)
def test_up_stops_on_a_tracked_instance_that_is_not_ready(tmp_path, status, ip):
    """Catches: reading anything but active-with-an-ip as gone, and launching beside it."""
    state = tracked_state()
    with FakeLambda() as fake:
        fake.add_instance(status=status, ip=ip)
        result = run_up(fake, tmp_path, state=state, pace=PACE)

    assert_stopped_unchanged(fake, result, state)
    assert TRACKED_ID in result.output
    assert status in result.output
    assert fake.alive() == [TRACKED_ID]


def test_up_reuses_a_tracked_active_instance(tmp_path):
    """Catches: a reuse path broken by the rewrite, launching when it has no need to."""
    with FakeLambda() as fake:
        fake.add_instance()
        result = run_up(fake, tmp_path, state=tracked_state(), pace=PACE)

    assert result.returncode == 0, result.output
    assert fake.launches == []
    assert "Reusing running instance" in result.output
    assert result.state["provisioned_profiles"] == ["alpha", "beta"]


# --- really gone --------------------------------------------------------------------------------


def test_published_not_found_is_gone_and_launches_once(tmp_path):
    """Catches: stopping forever on an instance that really is gone, or losing what it recorded."""
    with FakeLambda(instance_reads=["not_found"]) as fake:
        result = run_up(fake, tmp_path, state=tracked_state("gone-id"), pace=PACE)

    assert result.returncode == 0, result.output
    assert len(fake.launches) == 1
    assert fake.count("GET", "/instances/gone-id") == 1
    assert result.state["instance_id"] == "fake-instance-1"
    assert result.state["provisioned_profiles"] == ["alpha", "beta"]


def test_status_terminated_is_gone_and_launches_once(tmp_path):
    """Catches: a terminated instance, still on the list, counting as mounting the filesystem."""
    with FakeLambda() as fake:
        fake.add_instance(status="terminated", ip=False)
        result = run_up(fake, tmp_path, state=tracked_state(), pace=PACE)

    assert result.returncode == 0, result.output
    assert len(fake.launches) == 1
    assert result.state["instance_id"] == "fake-instance-2"
    assert result.state["provisioned_profiles"] == ["alpha", "beta"]


def test_an_error_that_clears_reuses_the_instance(tmp_path):
    """Catches: no retry at all, so one blip stops a healthy `up`."""
    with FakeLambda(instance_reads=["http_503", "drop"]) as fake:
        fake.add_instance()
        result = run_up(fake, tmp_path, state=tracked_state(), pace=PACE)

    assert result.returncode == 0, result.output
    assert fake.launches == []
    assert "Reusing running instance" in result.output
    assert "retrying" in result.output


# --- gone, and then what the listing says -------------------------------------------------------


@pytest.mark.parametrize(
    "mode", ["http_503", "http_401", "drop", "200_html", "200_not_a_list", "200_no_status", "200_unlisted_status"]
)
def test_gone_then_a_failed_list_does_not_launch(tmp_path, mode):
    """Catches: "couldn't list" reading as "nothing mounts it", and clearing state before asking."""
    state = tracked_state("gone-id")
    with FakeLambda(instance_reads=["not_found"], list_reads_then=mode) as fake:
        result = run_up(fake, tmp_path, state=state, pace=PACE)

    assert_stopped_unchanged(fake, result, state)


@pytest.mark.parametrize(
    "status,ip", [("booting", False), ("active", False), ("unhealthy", True), ("terminating", True), ("preempted", False)]
)
def test_nothing_tracked_and_a_not_ready_instance_on_the_filesystem_does_not_launch(tmp_path, status, ip):
    """Catches: the old filter and its no-ip fall-through, which launch beside a billing instance."""
    with FakeLambda() as fake:
        fake.add_instance(status=status, ip=ip)
        result = run_up(fake, tmp_path, pace=PACE)

    assert result.returncode != 0, result.output
    assert fake.launches == []
    assert result.state_bytes is None
    assert TRACKED_ID in result.output


def test_nothing_tracked_prefers_the_usable_instance_on_the_filesystem(tmp_path):
    """Catches: stopping on a booting instance when an active one is there to adopt."""
    with FakeLambda() as fake:
        fake.add_instance(status="booting", ip=False)
        usable = fake.add_instance()
        result = run_up(fake, tmp_path, pace=PACE)

    assert result.returncode == 0, result.output
    assert fake.launches == []
    assert result.state["instance_id"] == usable


def test_an_instance_on_another_filesystem_does_not_stop_a_launch(tmp_path):
    """Catches: any not-ready instance on the account blocking every config's `up`."""
    with FakeLambda() as fake:
        fake.add_instance(status="booting", ip=False)
        fake.instances[0]["file_system_names"] = ["someone-elses"]
        result = run_up(fake, tmp_path, pace=PACE)

    assert result.returncode == 0, result.output
    assert len(fake.launches) == 1


# --- a launch after a real gone -------------------------------------------------------------------


def test_a_launch_that_fails_keeps_provisioned_profiles(tmp_path):
    """Catches: clearing state before anything has replaced it."""
    state = tracked_state("gone-id")
    with FakeLambda(instance_reads=["not_found"], then="bad_request") as fake:
        result = run_up(fake, tmp_path, state=state, max_hours=0.0003, pace=PACE)

    assert result.returncode != 0, result.output
    assert len(fake.launches) >= 1
    assert result.state["provisioned_profiles"] == ["alpha", "beta"]
    assert result.state_bytes == json.dumps(state).encode()


def test_a_launch_that_times_out_keeps_provisioned_profiles(tmp_path):
    """Catches: as above, with no launch even attempted before the time ran out."""
    state = tracked_state("gone-id")
    with FakeLambda(instance_reads=["not_found"], then="insufficient_capacity") as fake:
        result = run_up(fake, tmp_path, state=state, max_hours=0.0001, pace=PACE)

    assert result.returncode != 0, result.output
    assert "Timed out" in result.output
    assert result.state["provisioned_profiles"] == ["alpha", "beta"]


# --- a launch reply that doesn't say what happened ------------------------------------------------


@pytest.mark.parametrize("mode", ["start_then_drop", "start_then_200_html"])
def test_a_launch_reply_that_says_nothing_launches_exactly_once(tmp_path, mode):
    """Catches: a lost launch reply becoming a LambdaAPIError, so `up` launches a second instance."""
    with FakeLambda(launches=[mode]) as fake:
        result = run_up(fake, tmp_path, pace=PACE)

    assert result.returncode != 0, result.output
    assert len(fake.launches) == 1
    assert fake.alive() == ["fake-instance-1"]


def test_a_lost_launch_reply_after_a_gone_keeps_the_state(tmp_path):
    """Catches: the same, losing what the gone instance recorded as well."""
    state = tracked_state("gone-id")
    with FakeLambda(instance_reads=["not_found"], launches=["start_then_drop"]) as fake:
        result = run_up(fake, tmp_path, state=state, pace=PACE)

    assert result.returncode != 0, result.output
    assert len(fake.launches) == 1
    assert result.state_bytes == json.dumps(state).encode()


# --- down ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode", ["http_503", "http_401", "drop", "200_html", "200_not_a_list", "200_no_status", "200_unlisted_status"]
)
def test_down_with_nothing_tracked_and_a_failed_list_stops(tmp_path, mode):
    """Catches: "No running instance" and exit 0 during an outage, while an instance goes on billing."""
    with FakeLambda(list_reads_then=mode) as fake:
        fake.add_instance()
        result = run_cloudgpu(fake, tmp_path, DOWN, pace=PACE)

    assert result.returncode != 0, result.output
    assert "No running instance" not in result.output
    assert "Traceback" not in result.output, result.output
    assert fake.count("POST", "/terminate") == 0
    assert fake.alive() == [TRACKED_ID]


def test_down_failed_list_leaves_a_stale_state_file_alone(tmp_path):
    """Catches: clearing state on the way out of a failed list. The state has no id, to reach the list."""
    state = {"filesystem": "box-fs", "provisioned_profiles": ["alpha"]}
    with FakeLambda(list_reads_then="http_503") as fake:
        result = run_cloudgpu(fake, tmp_path, DOWN, state=state, pace=PACE)

    assert result.returncode != 0, result.output
    assert result.state_bytes == json.dumps(state).encode()


@pytest.mark.parametrize("status", ["booting", "unhealthy", "active"])
def test_down_finds_and_terminates_an_untracked_instance(tmp_path, status):
    """Catches: `down` leaving an unhealthy instance billing, because the old filter hid it."""
    with FakeLambda() as fake:
        fake.add_instance(status=status)
        result = run_cloudgpu(fake, tmp_path, DOWN, pace=PACE)

    assert result.returncode == 0, result.output
    assert fake.count("POST", "/terminate") == 1
    assert fake.alive() == []


def test_down_does_not_terminate_a_terminating_instance_twice(tmp_path):
    """Catches: asking to terminate what is already going, and failing on the answer."""
    with FakeLambda() as fake:
        fake.add_instance(status="terminating")
        result = run_cloudgpu(fake, tmp_path, DOWN, pace=PACE)

    assert result.returncode == 0, result.output
    assert fake.count("POST", "/terminate") == 0
    assert "already terminating" in result.output


def test_down_with_a_clear_empty_list_is_still_a_noop(tmp_path):
    """Catches: `down` refusing when there's truly nothing to do."""
    with FakeLambda() as fake:
        result = run_cloudgpu(fake, tmp_path, DOWN, pace=PACE)

    assert result.returncode == 0, result.output
    assert "No running instance" in result.output


# --- _wait_terminated: in-process, on a fake clock ----------------------------------------------


class Clock:
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


def test_wait_terminated_on_errors_alone_does_not_return_true(monkeypatch):
    """Catches: any error counting as gone, the step before deleting a filesystem."""
    Clock(monkeypatch)

    def fail(_):
        raise lambda_api.LambdaAPIError("Lambda API request failed (HTTP 503)")

    monkeypatch.setattr(lambda_api, "get_instance", fail)
    assert cli_module._wait_terminated("i-1", timeout=60, poll=10) is False


def test_wait_terminated_true_on_terminated(monkeypatch):
    """Catches: never recognising the end of the wait."""
    Clock(monkeypatch)
    monkeypatch.setattr(lambda_api, "get_instance", lambda _: {"status": "terminated"})
    assert cli_module._wait_terminated("i-1", timeout=60, poll=10) is True


def test_wait_terminated_true_on_published_not_found(tmp_path, monkeypatch):
    """Catches: a read of a really gone instance polling until the timeout."""
    Clock(monkeypatch)
    with FakeLambda(instance_reads_then="not_found") as fake:
        monkeypatch.setenv("CLOUDGPU_TEST_LAMBDA_API_BASE", fake.base_url)
        monkeypatch.setenv("LAMBDA_API_KEY", "dummy-key-not-real")
        assert cli_module._wait_terminated("i-1", timeout=60, poll=10) is True


def test_wait_terminated_keeps_polling_past_a_status_that_is_not_gone(monkeypatch):
    """Catches: `terminating` being taken for finished, then deleting a filesystem still in use."""
    Clock(monkeypatch)
    answers = iter([{"status": "terminating"}, {"status": "terminating"}, {"status": "terminated"}])
    monkeypatch.setattr(lambda_api, "get_instance", lambda _: next(answers))
    assert cli_module._wait_terminated("i-1", timeout=60, poll=10) is True
