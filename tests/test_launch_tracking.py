"""A launched instance is written down before `up` waits for it, and never left untracked.

`cloudgpu up` against the fake Lambda, in a subprocess, with no real API, instance or key. A
Ctrl-C is a SIGINT sent once the fake has seen the request it should land in.
"""

from __future__ import annotations

import json

import pytest

from cloudgpu.local import lambda_api, liveness, orchestration
from tests.fake_lambda import FakeLambda, run_cloudgpu, run_up

PACE = 30
LAUNCHED = "fake-instance-1"
GONE_STATE = {"instance_id": "gone-id", "provisioned_profiles": ["alpha", "beta"]}
FOREVER = 10**6  # boot polls: still booting for as long as any test runs


def instance_reads(fake, instance_id=LAUNCHED) -> int:
    return fake.count("GET", f"/instances/{instance_id}")


def assert_names_what_is_running(result, instance_id=LAUNCHED):
    assert "Traceback" not in result.output, result.output
    assert instance_id in result.output, result.output
    assert "still running" in result.output, result.output
    assert "cloudgpu up -C box" in result.output, result.output
    assert "cloudgpu down -C box" in result.output, result.output


# --- 1. the id is saved before the first boot poll is answered ------------------------------------


def test_state_holds_the_id_and_profiles_before_the_first_boot_poll(tmp_path):
    """Catches: saving state only once the instance is active."""
    state_file = tmp_path / "configs" / "box.state.json"
    seen = []

    with FakeLambda(instance_reads=["not_found"], boot_polls=2) as fake:
        def on_get(path):
            if path.endswith(f"/instances/{LAUNCHED}"):
                seen.append(json.loads(state_file.read_text()))
        fake.on_get = on_get
        result = run_up(fake, tmp_path, state=GONE_STATE, pace=PACE)

    assert result.returncode == 0, result.output
    assert seen, "the launched instance was never polled"
    first = seen[0]
    assert first["instance_id"] == LAUNCHED
    assert first["provisioned_profiles"] == ["alpha", "beta"]
    assert first["instance_type"] == "gpu_1x_a10"
    assert first["region"] == "us-fake-1"
    assert "host" not in first  # not active yet: nothing may treat it as usable
    assert result.state["instance_id"] == LAUNCHED
    assert result.state["provisioned_profiles"] == ["alpha", "beta"]


# --- 2. Ctrl-C while booting, and while the launch request is out --------------------------------


def interrupt_while_booting(fake, tmp_path, **kwargs):
    return run_up(fake, tmp_path, poll_seconds=1, pace=PACE,
                  interrupt_when=lambda: instance_reads(fake) >= 1, **kwargs)


def test_ctrl_c_while_booting_names_the_instance_and_keeps_it_tracked(tmp_path):
    """Catches: a Ctrl-C during the boot wait leaving a billed instance nobody wrote down."""
    with FakeLambda(boot_polls=FOREVER) as fake:
        result = interrupt_while_booting(fake, tmp_path, state=GONE_STATE)
        alive = fake.alive()

    assert result.returncode != 0
    assert_names_what_is_running(result)
    assert "gpu_1x_a10" in result.output and "us-fake-1" in result.output
    assert result.state["instance_id"] == LAUNCHED
    assert result.state["provisioned_profiles"] == ["alpha", "beta"]
    assert alive == [LAUNCHED]


def test_ctrl_c_during_the_launch_request_says_a_launch_may_have_gone_through(tmp_path):
    """Catches: a Ctrl-C mid-launch ending in a bare 'Aborted!' with no word of what may be running."""
    with FakeLambda(launches=["start_then_stall"]) as fake:
        result = run_up(fake, tmp_path, pace=PACE, interrupt_when=lambda: len(fake.launches) >= 1)

    assert result.returncode != 0
    assert "Traceback" not in result.output, result.output
    assert "may have gone through" in result.output, result.output
    assert "cloudgpu lambda instances" in result.output, result.output
    assert len(fake.launches) == 1


# --- 3. a boot that times out or dies -------------------------------------------------------------


def test_a_boot_timeout_keeps_the_id_and_says_what_is_running(tmp_path):
    """Catches: the boot-wait timeout exiting with a plain error and no saved id."""
    with FakeLambda(boot_polls=FOREVER) as fake:
        result = run_up(fake, tmp_path, pace=PACE, config_extra="boot_minutes = 0.005\n")
        alive = fake.alive()

    assert result.returncode != 0
    assert "timed out" in result.output.lower()
    assert_names_what_is_running(result)
    assert result.state["instance_id"] == LAUNCHED
    assert alive == [LAUNCHED]
    assert len(fake.launches) == 1


def test_an_instance_in_error_while_booting_stays_tracked(tmp_path):
    """Catches: forgetting an instance whose status isn't one that says it's gone."""
    with FakeLambda(boot_polls=1, boot_ends="error") as fake:
        result = run_up(fake, tmp_path, state=GONE_STATE, pace=PACE)

    assert result.returncode != 0
    assert "'error'" in result.output
    assert_names_what_is_running(result)
    assert result.state["instance_id"] == LAUNCHED
    assert result.state["provisioned_profiles"] == ["alpha", "beta"]
    assert len(fake.launches) == 1


def test_an_instance_terminated_while_booting_is_cleared_and_not_relaunched(tmp_path):
    """Catches: tracking an instance that's gone, or launching again after it died."""
    with FakeLambda(instance_reads=["not_found"], boot_polls=1, boot_ends="terminated") as fake:
        result = run_up(fake, tmp_path, state=GONE_STATE, pace=PACE)
        alive = fake.alive()

    assert result.returncode != 0
    assert "Traceback" not in result.output, result.output
    assert "terminated" in result.output
    assert "nothing is left running" in result.output.lower(), result.output
    assert result.state.get("instance_id") != LAUNCHED
    assert result.state["provisioned_profiles"] == ["alpha", "beta"]
    assert alive == []
    assert len(fake.launches) == 1


# --- 4. what the next command does with the instance left booting ---------------------------------


def test_up_after_a_ctrl_c_does_not_launch_again(tmp_path):
    """Catches: a second `up` that doesn't see the saved, still-booting id and launches another."""
    with FakeLambda(boot_polls=FOREVER) as fake:
        interrupt_while_booting(fake, tmp_path)
        again = run_up(fake, tmp_path, pace=PACE, config_extra="boot_minutes = 0.005\n")

    assert again.returncode != 0
    assert "Traceback" not in again.output, again.output
    assert LAUNCHED in again.output
    assert len(fake.launches) == 1
    assert again.state["instance_id"] == LAUNCHED


def test_down_after_a_ctrl_c_terminates_the_saved_id(tmp_path):
    """Catches: `down` missing an id saved with no host.

    Every list after `up`'s first fails, so `down` can't fall back to what mounts the filesystem:
    only the saved id can find the instance.
    """
    with FakeLambda(boot_polls=FOREVER, list_reads=["ok"], list_reads_then="http_503") as fake:
        interrupt_while_booting(fake, tmp_path)
        down = run_cloudgpu(fake, tmp_path, ["down", "--config", "box", "--yes"], pace=PACE)
        alive = fake.alive()

    assert down.returncode == 0, down.output
    assert f"Termination requested for {LAUNCHED}" in down.output
    assert alive == []
    assert down.state == {}


@pytest.mark.parametrize("args", [
    ["status", "-C", "box"],
    ["ssh", "-C", "box"],
    ["forward", "-C", "box"],
    ["provision", "-C", "box", "-P", "alpha"],
    ["autoterminate", "-C", "box"],
])
def test_commands_reading_state_handle_an_id_with_no_host(tmp_path, args):
    """Catches: a command assuming a saved id means a host is saved too."""
    state = {"instance_id": LAUNCHED, "filesystem": "box-fs", "instance_type": "gpu_1x_a10", "region": "us-fake-1"}
    with FakeLambda() as fake:
        result = run_cloudgpu(fake, tmp_path, args, state=state, pace=PACE)

    assert result.returncode != 0
    assert "Traceback" not in result.output, result.output
    assert result.shim_calls == []
    assert fake.launches == []


@pytest.mark.parametrize("args", [["status", "-C", "box"], ["ssh", "-C", "box"]])
def test_commands_needing_a_host_name_the_unready_instance(tmp_path, args):
    """Catches: telling the person nothing runs when an instance is tracked and billing."""
    state = {"instance_id": LAUNCHED, "filesystem": "box-fs"}
    with FakeLambda() as fake:
        result = run_cloudgpu(fake, tmp_path, args, state=state, pace=PACE)

    assert result.returncode != 0
    assert LAUNCHED in result.output, result.output
    assert "cloudgpu down -C box" in result.output, result.output


# --- 5. a gateway error after a launch that took effect -------------------------------------------


def test_start_then_502_tracks_the_instance_it_made_and_launches_once(tmp_path):
    """Catches: reading every launch error as 'capacity vanished' and launching a second instance."""
    with FakeLambda(launches=["start_then_502"]) as fake:
        result = run_up(fake, tmp_path, pace=PACE)
        alive = fake.alive()

    assert result.returncode == 0, result.output
    assert len(fake.launches) == 1
    assert alive == [LAUNCHED]
    assert result.state["instance_id"] == LAUNCHED
    assert "HTTP 502" in result.output


def test_start_then_502_with_an_unclear_list_stops_without_launching_again(tmp_path):
    """Catches: launching again when the follow-up lookup can't say what the launch made."""
    with FakeLambda(launches=["start_then_502"], list_reads=["ok", "http_503", "http_503", "http_503"]) as fake:
        result = run_up(fake, tmp_path, pace=PACE)

    assert result.returncode != 0
    assert "Traceback" not in result.output, result.output
    assert "may have gone through" in result.output
    assert len(fake.launches) == 1


@pytest.mark.parametrize("mode", ["insufficient_capacity", "rate_limited"])
def test_known_launch_rejections_still_poll_and_launch_once(tmp_path, mode):
    """Catches: treating a published rejection as a launch that might have gone through."""
    with FakeLambda(launches=[mode]) as fake:
        result = run_up(fake, tmp_path, pace=PACE)

    assert result.returncode == 0, result.output
    assert len(fake.launches) == 2
    assert fake.alive() == [LAUNCHED]
    assert result.state["instance_id"] == LAUNCHED


# --- the launch-error rule, in-process ---------------------------------------------------------


@pytest.fixture
def capacity_everywhere(monkeypatch):
    """acquire_instance with capacity in one region, and every launch failing as `errors` say."""
    monkeypatch.setattr(orchestration, "ensure_ssh_key", lambda name: "k")
    monkeypatch.setattr(lambda_api, "list_filesystems",
                        lambda: [{"name": "fs", "id": "fs-1", "region": {"name": "r1"}}])
    monkeypatch.setattr(lambda_api, "list_instance_types",
                        lambda: {"gpu_1x_a10": {"regions_with_capacity_available": [{"name": "r1"}]}})
    profile = {"name": "box", "filesystem": "fs", "gpu": ["a10"], "poll_seconds": 0, "max_hours": 0.0003}
    return profile


def test_an_unrecognised_launch_error_with_nothing_mounting_stops(capacity_everywhere, monkeypatch):
    """Catches: a 5xx with nothing found yet read as safe to launch again."""
    launches = []

    def launch(**kw):
        launches.append(kw)
        raise lambda_api.LambdaAPIError("Lambda API request failed (HTTP 500)", status=500)

    monkeypatch.setattr(lambda_api, "launch_instance", launch)
    monkeypatch.setattr(liveness, "find_mounting", lambda fs: liveness.Answer(liveness.Kind.GONE))
    with pytest.raises(lambda_api.LambdaReplyLost, match="HTTP 500"):
        orchestration.acquire_instance(capacity_everywhere)
    assert len(launches) == 1


def test_an_error_with_no_http_answer_polls_again(capacity_everywhere, monkeypatch):
    """Catches: a request that never reached Lambda (no HTTP status) stopping the poll."""
    launches = []

    def launch(**kw):
        launches.append(kw)
        if len(launches) == 1:
            raise lambda_api.LambdaAPIError("Could not reach Lambda Cloud API: refused")
        return {"instance_ids": ["i-2"]}

    monkeypatch.setattr(lambda_api, "launch_instance", launch)
    monkeypatch.setattr(liveness, "find_mounting", lambda fs: pytest.fail("no lookup for a request never sent"))
    monkeypatch.setattr(lambda_api, "get_instance", lambda i: {"status": "active", "ip": "1.2.3.4"})
    runtime = orchestration.acquire_instance(capacity_everywhere)
    assert runtime["instance_id"] == "i-2"
    assert len(launches) == 2


def test_on_launched_gets_the_id_before_the_first_poll(capacity_everywhere, monkeypatch):
    """Catches: handing the id back only once the wait is over."""
    events = []
    monkeypatch.setattr(lambda_api, "launch_instance", lambda **kw: {"instance_ids": ["i-1"]})
    monkeypatch.setattr(lambda_api, "get_instance",
                        lambda i: events.append("poll") or {"status": "active", "ip": "1.2.3.4"})
    orchestration.acquire_instance(capacity_everywhere,
                                   on_launched=lambda rt: events.append(("launched", rt["instance_id"])))
    assert events[0] == ("launched", "i-1")
    assert "poll" in events
