"""R7: never leave an untracked or second instance when a launch is in doubt.

One test per acceptance criterion, running the real `cloudgpu up` against the fake Lambda.
"""

from __future__ import annotations

import pytest

from cloudgpu.local import lambda_api, orchestration
from tests.fake_lambda import FakeLambda, run_up

PACE = 30  # request timeout 2s, look window 2s
LAUNCHED = "fake-instance-1"


def assert_one_tracked(fake, result):
    assert result.returncode == 0, result.output
    assert "Traceback" not in result.output, result.output
    assert len(fake.launches) == 1
    assert fake.alive() == [LAUNCHED]
    assert result.state["instance_id"] == LAUNCHED


def test_502_on_launch_ends_with_exactly_one_tracked_instance(tmp_path):
    with FakeLambda(launches=["start_then_502"]) as fake:
        result = run_up(fake, tmp_path, pace=PACE)
        assert_one_tracked(fake, result)


def test_dropped_launch_reply_ends_with_one_tracked_instance(tmp_path):
    with FakeLambda(launches=["start_then_drop"]) as fake:
        result = run_up(fake, tmp_path, pace=PACE)
        assert_one_tracked(fake, result)


def test_launch_read_timeout_ends_with_one_tracked_instance(tmp_path):
    with FakeLambda(launches=["start_then_hang"]) as fake:
        result = run_up(fake, tmp_path, pace=PACE)
        assert_one_tracked(fake, result)


def test_list_that_lags_then_shows_the_instance_adopts_it(tmp_path):
    # The first list is up's own look before launching; the next two lag behind the launch.
    with FakeLambda(launches=["start_then_502"], list_reads=["ok", "200_empty", "200_empty"]) as fake:
        result = run_up(fake, tmp_path, pace=PACE)
        assert_one_tracked(fake, result)
        assert result.output.count("looking again") == 2, result.output


def test_up_during_boot_waits_and_launches_nothing(tmp_path):
    with FakeLambda() as fake:
        booting = fake.add_instance(boot_polls=3)
        result = run_up(fake, tmp_path, pace=PACE)

    assert result.returncode == 0, result.output
    assert len(fake.launches) == 0
    assert "waiting up to" in result.output
    assert result.state["instance_id"] == booting


def test_up_during_boot_that_never_finishes_stops_and_says_what_bills(tmp_path):
    with FakeLambda() as fake:
        booting = fake.add_instance(status="booting", ip=False)
        result = run_up(fake, tmp_path, pace=PACE, config_extra="boot_minutes = 0.002\n")

    assert result.returncode != 0
    assert len(fake.launches) == 0
    assert booting in result.output
    assert "billing" in result.output


def test_two_concurrent_ups_send_one_launch_and_one_refuses(tmp_path):
    second = {}
    with FakeLambda(boot_polls=1) as fake:
        def start_second_while_first_boots(path):
            if path.endswith(f"/instances/{LAUNCHED}") and not second:
                second["result"] = run_up(fake, tmp_path, pace=PACE)

        fake.on_get = start_second_while_first_boots
        first = run_up(fake, tmp_path, pace=PACE)

    refused = second["result"]
    assert first.returncode == 0, first.output
    assert refused.returncode != 0
    assert "already running" in refused.output, refused.output
    assert "Traceback" not in refused.output
    assert len(fake.launches) == 1
    assert fake.alive() == [LAUNCHED]


def test_400_stops_after_one_attempt_with_lambdas_message(tmp_path):
    with FakeLambda(then="bad_request") as fake:
        result = run_up(fake, tmp_path, pace=PACE, max_hours=1.0)

    assert result.returncode != 0
    assert len(fake.launches) == 1
    assert "global/invalid-parameters" in result.output
    assert "Invalid request." in result.output
    assert "Traceback" not in result.output


def test_insufficient_capacity_polls_then_succeeds(tmp_path):
    with FakeLambda(launches=["insufficient_capacity", "insufficient_capacity"]) as fake:
        result = run_up(fake, tmp_path, pace=PACE)

    assert result.returncode == 0, result.output
    assert len(fake.launches) == 3
    assert fake.alive() == [LAUNCHED]


def test_429_backs_off_honouring_retry_after(monkeypatch):
    monkeypatch.setattr(orchestration, "ensure_ssh_key", lambda name: "k")
    monkeypatch.setattr(lambda_api, "list_filesystems", lambda: [{"name": "fs", "id": "fs-1", "region": {"name": "r1"}}])
    monkeypatch.setattr(lambda_api, "list_instance_types",
                        lambda: {"gpu_1x_a10": {"regions_with_capacity_available": [{"name": "r1"}]}})
    launches, sleeps = [], []

    def launch(**kw):
        launches.append(kw)
        if len(launches) == 1:
            raise lambda_api.LambdaAPIError("HTTP 429", status=429, code="global/rate-limited", retry_after=7)
        return {"instance_ids": ["i-2"]}

    monkeypatch.setattr(lambda_api, "launch_instance", launch)
    monkeypatch.setattr(lambda_api, "get_instance", lambda i: {"status": "active", "ip": "1.2.3.4"})
    monkeypatch.setattr(orchestration.time, "sleep", sleeps.append)
    profile = {"name": "box", "filesystem": "fs", "gpu": ["a10"], "poll_seconds": 1, "max_hours": 1}
    runtime = orchestration.acquire_instance(profile)

    assert runtime["instance_id"] == "i-2"
    assert len(launches) == 2
    assert 7 in sleeps


def test_429_with_no_retry_after_backs_off_at_least_the_floor(monkeypatch):
    e = lambda_api.LambdaAPIError("HTTP 429", status=429, code="global/rate-limited")
    assert lambda_api.classify_launch_error(e) is lambda_api.LaunchOutcome.RATE_LIMITED
    assert orchestration.RATE_LIMIT_BACKOFF_SECONDS >= 30


def test_listing_fails_after_doubtful_launch_no_second_launch_and_names_billing(tmp_path):
    with FakeLambda(launches=["start_then_502"], list_reads=["ok"], list_reads_then="http_503") as fake:
        result = run_up(fake, tmp_path, pace=PACE)

    assert result.returncode != 0
    assert "Traceback" not in result.output, result.output
    assert len(fake.launches) == 1
    assert "billing" in result.output
    assert "box-fs" in result.output
    assert "cloudgpu lambda instances" in result.output


@pytest.mark.parametrize("status,outcome", [
    (500, lambda_api.LaunchOutcome.IN_DOUBT), (502, lambda_api.LaunchOutcome.IN_DOUBT),
    (400, lambda_api.LaunchOutcome.REJECTED), (404, lambda_api.LaunchOutcome.REJECTED),
    (429, lambda_api.LaunchOutcome.RATE_LIMITED), (None, lambda_api.LaunchOutcome.NOT_SENT),
])
def test_launch_errors_are_classified(status, outcome):
    assert lambda_api.classify_launch_error(lambda_api.LambdaAPIError("x", status=status)) is outcome
    capacity = lambda_api.LambdaAPIError("x", status=400, code=lambda_api.INSUFFICIENT_CAPACITY_CODE)
    assert lambda_api.classify_launch_error(capacity) is lambda_api.LaunchOutcome.NO_CAPACITY
    assert lambda_api.classify_launch_error(lambda_api.LambdaReplyLost("x")) is lambda_api.LaunchOutcome.IN_DOUBT
