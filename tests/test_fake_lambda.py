"""`cloudgpu up`, end to end, against the fake Lambda: one test per thing the fake can do.

The tests named `..._today` document launch behaviour that's wrong and is due to change
(R7 part 2). They prove the fake reproduces the failure, not that the behaviour is right.
"""

from __future__ import annotations

import pytest

from tests.fake_lambda import FakeLambda, run_up


def test_up_runs_to_exit_0_with_one_tracked_instance(tmp_path):
    with FakeLambda() as fake:
        result = run_up(fake, tmp_path)

    assert result.returncode == 0, result.output
    assert len(fake.launches) == 1
    assert fake.launches[0]["file_system_names"] == ["box-fs"]
    assert fake.alive() == ["fake-instance-1"]
    assert result.state["instance_id"] == "fake-instance-1"
    assert result.state["persistent_dir"] == "/lambda/nfs/box-fs"
    # Setup ran after the launch, against the launched instance, and only through the shims.
    assert {c["host"].split(":")[0] for c in result.shim_calls} == {"ubuntu@192.0.2.1"}
    assert {"ssh", "rsync"} == {c["tool"] for c in result.shim_calls}


def test_up_creates_a_missing_filesystem(tmp_path):
    with FakeLambda(filesystem=None) as fake:
        result = run_up(fake, tmp_path)

    assert result.returncode == 0, result.output
    assert ("POST", "/api/v1/filesystems") in fake.requests
    assert fake.alive() == ["fake-instance-1"]


def test_mode_start_then_502_tracks_the_one_instance_it_launched(tmp_path):
    with FakeLambda(launches=["start_then_502"]) as fake:
        result = run_up(fake, tmp_path)

    assert result.returncode == 0, result.output
    assert len(fake.launches) == 1
    assert fake.alive() == ["fake-instance-1"]
    assert result.state["instance_id"] == "fake-instance-1"


def test_mode_start_then_drop_adopts_the_instance_without_a_second_launch(tmp_path):
    with FakeLambda(launches=["start_then_drop"]) as fake:
        result = run_up(fake, tmp_path)

    assert result.returncode == 0, result.output
    assert "Traceback" not in result.output
    assert "it may have gone through" in result.output
    assert len(fake.launches) == 1
    assert fake.alive() == ["fake-instance-1"]
    assert result.state["instance_id"] == "fake-instance-1"


def test_mode_insufficient_capacity_polls_then_launches(tmp_path):
    with FakeLambda(launches=["insufficient_capacity"]) as fake:
        result = run_up(fake, tmp_path)

    assert result.returncode == 0, result.output
    assert "insufficient-capacity" in result.output
    assert len(fake.launches) == 2
    assert fake.alive() == ["fake-instance-1"]


def test_mode_rate_limited_polls_then_launches(tmp_path):
    with FakeLambda(launches=["rate_limited"]) as fake:
        result = run_up(fake, tmp_path)

    assert result.returncode == 0, result.output
    assert "HTTP 429" in result.output
    assert len(fake.launches) == 2
    assert fake.alive() == ["fake-instance-1"]


def test_mode_bad_request_stops_after_one_attempt(tmp_path):
    with FakeLambda(then="bad_request") as fake:
        result = run_up(fake, tmp_path, max_hours=0.0003)  # about a second

    assert result.returncode == 1
    assert "Invalid request." in result.output
    assert len(fake.launches) == 1
    assert fake.alive() == []


def test_mode_boot_polls_waits_for_the_ip(tmp_path):
    with FakeLambda(boot_polls=3) as fake:
        result = run_up(fake, tmp_path)

    assert result.returncode == 0, result.output
    assert result.output.count("status=booting") == 3
    assert result.state["ip"] == "192.0.2.1"
    assert len(fake.launches) == 1


def test_up_during_boot_does_not_launch_a_second_instance(tmp_path):
    with FakeLambda() as fake:
        fake.add_instance(boot_polls=1000)
        result = run_up(fake, tmp_path, config_extra="boot_minutes = 0.0001\n")

    assert result.returncode != 0
    assert fake.launches == []
    assert fake.alive() == ["fake-instance-1"]


def test_fake_refuses_an_unknown_mode():
    with pytest.raises(ValueError, match="start_then_503"):
        FakeLambda(launches=["start_then_503"])
