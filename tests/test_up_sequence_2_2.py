"""Every `cloudgpu up` runs the same steps in the same order (plan item 2.2, part 2).

No real API, instance or key: the real CLI runs in a subprocess against the fake Lambda, with ssh
and rsync shimmed. A step is made to fail, or to hang for a Ctrl-C, with the ssh shim's
FAKE_SHIM_FAIL_ON. What `up` did is read from what it printed and from the ssh calls it made.
"""

from __future__ import annotations

import json

import pytest

from tests.fake_lambda import FakeLambda, run_cloudgpu

PLAN_ORDER = [
    "find or launch the instance",
    "set up the instance",
    "arm the cap",
    "make apps match",
    "provision",
    "restart services",
    "report",
]
COMFY = {"comfyui": {"status": "installed", "version": "1", "app_dir": "/lambda/nfs/box-fs/apps/comfyui"}}
MARKED = {"comfyui": {**COMFY["comfyui"], "reprovision": True}}
RECORDED_ENV = {"FAKE_SHIM_APPS": json.dumps(COMFY)}
GEN = 'profiles = ["gen"]\n'
WAYS = ["fresh", "reused", "adopted"]


def _profile(workdir, name, apps=()):
    d = workdir / "profiles" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "cloudgpu.toml").write_text(f"apps = {json.dumps(list(apps))}\n")
    (d / "provision.py").write_text("print('provisioned')\n")


def _steps(result):
    """The steps `up` announced, in order."""
    return [line.split(": ", 1)[1].strip() for line in result.output.splitlines()
            if line.startswith("Step ") and ": " in line]


def _event(call):
    """What one ssh call was, in words; None for the ones that don't matter here."""
    c = call["command"]
    if call["tool"] != "ssh":
        return None
    if c.endswith(" detect"):
        return "detect"
    if "autoterminate --hours" in c:
        return "arm cap"
    if c.endswith(" status"):
        return "read recorded apps"
    if c.endswith(" recover"):
        return "recover"
    if " install --app " in c:
        return "install " + c.split("--app ")[1]
    if "provision.py" in c:
        return "provision " + c.split("/cloudgpu/provision/")[1].split(" ")[0]
    if "systemctl restart" in c:
        return "restart " + ",".join(part.split("restart ")[1] for part in c.split(" && "))
    if " clear-reprovision " in c:
        return "clear mark " + c.split("--app ")[1]
    return None


def _events(result):
    return [e for e in map(_event, result.shim_calls) if e]


def _up(fake, workdir, way, *, config_extra="", shim_env=None, state=None, interrupt_when=None,
        tidy=True):
    """One `up -C box`; `way` puts the fake in the shape that makes `up` launch, reuse or adopt."""
    if way == "reused":
        state = {**(state or {}), "instance_id": fake.add_instance()}
    elif way == "adopted":
        fake.add_instance()
    log = workdir / "shim-calls.jsonl"
    log.unlink(missing_ok=True)  # the shim reads it to know what already ran: one run at a time
    result = run_cloudgpu(fake, workdir, ["up", "--config", "box"], config_extra=config_extra,
                          shim_env=shim_env, state=state, interrupt_when=interrupt_when)
    if tidy:  # not asked of a run stopped mid-step: how each stop is reported isn't this item's
        assert "Traceback" not in result.output, result.output
    return result


# --- (1) the same steps, in the plan's order, however the machine was found ----------------------


@pytest.mark.parametrize("way", WAYS)
@pytest.mark.parametrize("with_profiles", [False, True], ids=["bare", "profiles"])
def test_up_runs_the_plans_steps_in_order(tmp_path, way, with_profiles):
    """Catches: a step skipped or moved on one path (reuse, adopt, launch; profiles or none)."""
    _profile(tmp_path, "gen", ["comfyui", "ai-toolkit"])
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, way, config_extra=GEN if with_profiles else "", shim_env=RECORDED_ENV)
        launches = len(fake.launches)
    assert result.returncode == 0, result.output
    assert launches == (1 if way == "fresh" else 0)
    assert _steps(result) == PLAN_ORDER, result.output
    events = _events(result)
    middle = (["install ai-toolkit", "provision gen", "restart comfyui,ai-toolkit"] if with_profiles else [])
    assert events == ["detect", "arm cap", "read recorded apps", "recover", "read recorded apps",
                      *middle, "read recorded apps"], result.output


def test_each_step_is_announced_before_its_work(tmp_path):
    """Catches: the step lines printed as a list up front, not where each step actually starts."""
    _profile(tmp_path, "gen", ["comfyui", "ai-toolkit"])
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, "fresh", config_extra=GEN, shim_env=RECORDED_ENV)
    out = result.output
    marks = ["Step 1/7", "Launched", "Step 2/7", "Remote tool synced", "Step 3/7", "Step 4/7",
             "Recovering existing apps", "Installing ai-toolkit", "Step 5/7", "Provisioning 'gen'",
             "Step 6/7", "Restarting service(s)", "Step 7/7", "is up"]
    at = [out.find(m) for m in marks]
    assert -1 not in at and at == sorted(at), (list(zip(marks, at)), out)


# --- (2) nothing to provision: apps still match, services restarted by recovery, not twice -------


@pytest.mark.parametrize("way", WAYS)
def test_up_with_nothing_to_provision_recovers_and_does_not_restart_twice(tmp_path, way):
    """Catches: a bare `up` skipping recovery (whose remote restarts services), or restarting again."""
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, way, shim_env=RECORDED_ENV)
    assert result.returncode == 0, result.output
    events = _events(result)
    assert events.count("recover") == 1, result.output
    assert not [e for e in events if e.startswith("restart")], result.output
    assert "Nothing to provision." in result.output
    assert ("Nothing was provisioned, so not restarting comfyui again: "
            "recovery already restarted and confirmed them.") in result.output, result.output


def test_restart_after_provisioning_covers_recorded_apps_too(tmp_path):
    """Catches: restarting only this `up`'s profile apps, leaving a recorded service on old files."""
    _profile(tmp_path, "tools", ["ai-toolkit"])
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, "fresh", config_extra='profiles = ["tools"]\n', shim_env=RECORDED_ENV)
    assert result.returncode == 0, result.output
    assert [e for e in _events(result) if e.startswith("restart")] == ["restart comfyui,ai-toolkit"]


def test_all_missing_apps_are_installed_before_any_profile_is_provisioned(tmp_path):
    """Catches: "make apps match" folded back into provisioning, one profile at a time."""
    _profile(tmp_path, "a", ["comfyui"])
    _profile(tmp_path, "b", ["ai-toolkit"])
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, "fresh", config_extra='profiles = ["a", "b"]\n')
    assert result.returncode == 0, result.output
    assert [e for e in _events(result) if e.split()[0] in ("install", "provision")] == [
        "install comfyui", "install ai-toolkit", "provision a", "provision b"]


# --- (3) stopped at any step, then run again ------------------------------------------------------

# step -> the ssh call it stops at (and, for the last, the earlier call that must have happened)
STOP_AT = {
    "set up the instance": {"FAKE_SHIM_FAIL_ON": " detect"},
    "arm the cap": {"FAKE_SHIM_FAIL_ON": "autoterminate --hours"},
    "make apps match (recover)": {"FAKE_SHIM_FAIL_ON": " recover"},
    "make apps match (install)": {"FAKE_SHIM_FAIL_ON": " install --app "},
    "provision": {"FAKE_SHIM_FAIL_ON": "provision.py"},
    "restart services": {"FAKE_SHIM_FAIL_ON": "systemctl restart"},
    "report": {"FAKE_SHIM_FAIL_ON": " status", "FAKE_SHIM_FAIL_AFTER": "systemctl restart"},
}


@pytest.mark.parametrize("mode", ["fails", "interrupted"])
@pytest.mark.parametrize("step", list(STOP_AT))
def test_up_stopped_at_a_step_then_run_again_finishes_on_the_same_instance(tmp_path, step, mode):
    """Catches: a re-run launching a second instance, skipping a step, or ending in another state."""
    whole, stopped = tmp_path / "whole", tmp_path / "stopped"
    for d in (whole, stopped):
        d.mkdir()
        _profile(d, "gen", ["comfyui", "ai-toolkit"])
    with FakeLambda() as fake:
        reference = _up(fake, whole, "fresh", config_extra=GEN, shim_env=RECORDED_ENV)
    assert reference.returncode == 0, reference.output

    stalled = stopped / "shim-calls.jsonl.stalled"
    env = {**RECORDED_ENV, **STOP_AT[step]}
    with FakeLambda() as fake:
        if mode == "fails":
            first = _up(fake, stopped, "fresh", config_extra=GEN, shim_env=env, tidy=False)
        else:
            first = _up(fake, stopped, "fresh", config_extra=GEN, interrupt_when=stalled.exists, tidy=False,
                        shim_env={**env, "FAKE_SHIM_FAIL_MODE": "stall"})
            assert stalled.exists(), first.output
        if (step, mode) != ("restart services", "fails"):  # a failed restart is a warning, as before
            assert first.returncode != 0, first.output
            assert "is up" not in first.output, first.output
            assert _steps(first)[-1] == step.split(" (")[0], first.output
        assert first.state["instance_id"] == "fake-instance-1"  # recorded before the stop

        # The same command again (the harness rewrites the same config), without the fault.
        again = _up(fake, stopped, "as the state file says", config_extra=GEN, shim_env=RECORDED_ENV)
        launches, alive = len(fake.launches), fake.alive()
    assert again.returncode == 0, again.output
    assert launches == 1 and alive == ["fake-instance-1"], again.output
    assert "Reusing running instance fake-instance-1" in again.output, again.output
    assert _steps(again) == PLAN_ORDER, again.output
    assert _events(again) == _events(reference), again.output
    # launched_at is the clock at the launch call: the re-run must keep the first run's, and it is
    # the one value two separate launches can't share.
    assert again.state["launched_at"] == first.state["launched_at"]
    assert {**again.state, "launched_at": 0} == {**reference.state, "launched_at": 0}


def test_a_profile_is_recorded_once_its_own_provisioning_succeeded(tmp_path):
    """Catches: a later profile's failure forgetting the earlier one is already on the filesystem."""
    _profile(tmp_path, "a")
    _profile(tmp_path, "b")
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, "fresh", config_extra='profiles = ["a", "b"]\n',
                     shim_env={"FAKE_SHIM_FAIL_ON": "/cloudgpu/provision/b "})
    assert result.returncode != 0, result.output
    assert _events(result)[-2:] == ["provision a", "provision b"], result.output
    assert result.state["provisioned_profiles"] == ["a"]


# --- (4) recovery once; a mark cleared only after provisioning and the restart ------------------


@pytest.mark.parametrize("way", WAYS)
def test_a_mark_is_cleared_after_its_profile_was_provisioned_and_services_restarted(tmp_path, way):
    """Catches: the mark dropped before the restart, or recovery run a second time for the redo."""
    _profile(tmp_path, "gen", ["comfyui"])
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, way, state={"provisioned_profiles": ["gen"]},
                     shim_env={"FAKE_SHIM_APPS": json.dumps(MARKED)})
    assert result.returncode == 0, result.output
    assert _steps(result) == PLAN_ORDER, result.output
    assert _events(result) == ["detect", "arm cap", "read recorded apps", "recover", "read recorded apps",
                               "provision gen", "restart comfyui", "clear mark comfyui",
                               "read recorded apps"], result.output


@pytest.mark.parametrize("stop", ["provision.py", "systemctl restart"])
def test_a_mark_stays_when_up_stops_before_the_restart_finished(tmp_path, stop):
    """Catches: a mark cleared though its profile wasn't put back, so no later `up` ever does it."""
    _profile(tmp_path, "gen", ["comfyui"])
    stalled = tmp_path / "shim-calls.jsonl.stalled"
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, "fresh", state={"provisioned_profiles": ["gen"]},
                     interrupt_when=stalled.exists, tidy=False,
                     shim_env={"FAKE_SHIM_APPS": json.dumps(MARKED), "FAKE_SHIM_FAIL_ON": stop,
                               "FAKE_SHIM_FAIL_MODE": "stall"})
    assert result.returncode != 0 and stalled.exists(), result.output
    assert not [e for e in _events(result) if e.startswith("clear mark")], result.output
