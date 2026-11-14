"""`cloudgpu up` succeeds only if the expected services are running (plan item 2.3).

No real API, instance, key, systemctl or sleep. Two halves, joined by the report between them:

- the remote `verify` command runs in-process with systemd faked at service.run (as
  tests/test_service_settles.py does);
- the real CLI runs in a subprocess against the fake Lambda with ssh and rsync shimmed (as
  tests/test_up_sequence_2_2.py does). The ssh shim answers `verify` with every asked service
  running, or with FAKE_SHIM_VERIFY, which these tests fill with what the real remote code
  printed for a faked systemd, so the CLI is tested against the remote's own words.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from cloudgpu.remote import __main__ as remote_main
from cloudgpu.remote import service
from tests.fake_lambda import FakeLambda, run_cloudgpu

PLAN_ORDER = [
    "find or launch the instance",
    "set up the instance",
    "arm the cap",
    "make apps match",
    "provision",
    "restart services",
    "verify",
    "report",
]
COMFY = {"comfyui": {"status": "installed", "version": "1", "app_dir": "/lambda/nfs/box-fs/apps/comfyui"}}
MARKED = {"comfyui": {**COMFY["comfyui"], "reprovision": True}}
RECORDED_ENV = {"FAKE_SHIM_APPS": json.dumps(COMFY)}
GEN = 'profiles = ["gen"]\n'
WAYS = ["fresh", "reused", "adopted"]
JOURNAL = "comfyui[123]: ImportError: no module named broken_node"

RUNNING = [("active", "running", 0), ("active", "running", 0)]
# Just after the restart, then after the settle period.
DEAD = {
    "inactive": [("inactive", "dead", 0), ("inactive", "dead", 0)],
    "failed": [("active", "running", 0), ("failed", "failed", 0)],
    "restart loop": [("active", "running", 2), ("active", "running", 5)],
}
# What the output must say systemd reports, for each.
SAYS = {"inactive": "inactive/dead", "failed": "failed/failed", "restart loop": "restarts: 3 in 10s"}


# --- the remote half: systemd faked at service.run ------------------------------------------------


def _no_real_commands(*args, **kwargs):
    raise AssertionError(f"a real command ran in a test: {args[:1]}")


class FakeSystemd:
    """Answers `systemctl show <name>` with that service's states, in order (the last repeats)."""

    def __init__(self, states):
        self.states = {name: list(s) for name, s in states.items()}
        self.calls: list[list[str]] = []
        self.slept: list[float] = []

    def run(self, cmd, **kwargs):
        self.calls.append(cmd)
        if cmd[:2] == ["systemctl", "show"]:
            states = self.states.get(cmd[-1])
            if not states:
                return SimpleNamespace(returncode=1, stdout="")
            active, sub, restarts = states.pop(0) if len(states) > 1 else states[0]
            return SimpleNamespace(returncode=0, stdout=f"ActiveState={active}\nSubState={sub}\nNRestarts={restarts}\n")
        if "journalctl" in cmd:
            return SimpleNamespace(returncode=0, stdout=JOURNAL + "\n")
        raise AssertionError(f"verify ran something other than a read: {cmd}")


@pytest.fixture
def systemd(monkeypatch):
    def install(states):
        fake = FakeSystemd(states)
        monkeypatch.setattr(service, "run", fake.run)
        monkeypatch.setattr(time, "sleep", fake.slept.append)
        monkeypatch.setattr(subprocess, "run", _no_real_commands)
        monkeypatch.setattr(subprocess, "Popen", _no_real_commands)
        return fake
    return install


def _remote_verify(monkeypatch, capsys, tmp_path, states, *args):
    """Run the remote tool's `verify` in-process over a faked systemd; return (fake, its stdout)."""
    fake = FakeSystemd(states)
    with monkeypatch.context() as m:
        m.setattr(service, "run", fake.run)
        m.setattr(time, "sleep", fake.slept.append)
        m.setattr(subprocess, "run", _no_real_commands)
        m.setattr(subprocess, "Popen", _no_real_commands)
        m.setattr(sys, "argv", ["remote", "--persistent-dir", str(tmp_path / "persistent"), "verify",
                                *[a for name in states for a in ("--service", name)], *args])
        capsys.readouterr()
        remote_main.main()
    return fake, capsys.readouterr().out


def _by_name(stdout):
    return {s["name"]: s for s in json.loads(stdout)["services"]}


def test_remote_verify_passes_running_services_after_one_settle(monkeypatch, capsys, tmp_path):
    """Catches: a healthy service reported as down, or the settle wait paid once per service."""
    fake, out = _remote_verify(monkeypatch, capsys, tmp_path, {"comfyui": RUNNING, "ai-toolkit": RUNNING})
    report = _by_name(out)
    assert [report[n]["ok"] for n in ("comfyui", "ai-toolkit")] == [True, True], out
    assert fake.slept == [service.SETTLE_SECONDS]
    assert not [c for c in fake.calls if "journalctl" in c]


@pytest.mark.parametrize("how", list(DEAD))
def test_remote_verify_fails_a_service_that_is_not_running_after_the_settle(monkeypatch, capsys, tmp_path, how):
    """Catches: a service that is down, died after its start, or is crash-looping, passed as running."""
    fake, out = _remote_verify(monkeypatch, capsys, tmp_path, {"comfyui": DEAD[how], "ai-toolkit": RUNNING})
    report = _by_name(out)
    assert report["comfyui"]["ok"] is False and report["ai-toolkit"]["ok"] is True, out
    said = "{active}/{sub}, restarts: {restarts}".format(**report["comfyui"])
    assert SAYS[how] in said, out
    assert JOURNAL in report["comfyui"]["journal"]
    assert report["ai-toolkit"]["journal"] == ""


def test_remote_verify_fails_a_service_systemd_cannot_describe(monkeypatch, capsys, tmp_path):
    """Catches: unreadable systemd state treated as running (unknown is not active)."""
    _, out = _remote_verify(monkeypatch, capsys, tmp_path, {"comfyui": []})
    report = _by_name(out)
    assert report["comfyui"]["ok"] is False
    assert (report["comfyui"]["active"], report["comfyui"]["sub"]) == ("unknown", "unknown")


def test_remote_verify_without_a_settle_still_reads_systemd(monkeypatch, capsys, tmp_path):
    """Catches: `--settle 0` (after recovery's own confirm) skipping the read, not just the wait."""
    fake, out = _remote_verify(monkeypatch, capsys, tmp_path,
                               {"comfyui": DEAD["inactive"], "ai-toolkit": RUNNING}, "--settle", "0")
    report = _by_name(out)
    assert report["comfyui"]["ok"] is False and report["ai-toolkit"]["ok"] is True, out
    assert sum(fake.slept) == 0


def test_confirm_running_still_raises_with_state_and_journal(systemd):
    """Catches: install/recover's check weakened by sharing its test with verify."""
    systemd({"comfyui": DEAD["failed"]})
    with pytest.raises(RuntimeError) as e:
        service.confirm_running("comfyui")
    assert "failed/failed" in str(e.value) and JOURNAL in str(e.value)
    systemd({"comfyui": RUNNING})
    service.confirm_running("comfyui")


# --- the local half: the real CLI against the fake Lambda and the ssh shim ------------------------


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
    c = call["command"]
    if call["tool"] != "ssh":
        return None
    if "systemctl restart" in c:
        return "restart"
    if " verify " in c:
        return "verify"
    if " clear-reprovision " in c:
        return "clear mark"
    if c.endswith(" status"):
        return "status"
    return None


def _events(result):
    return [e for e in map(_event, result.shim_calls) if e]


def _verify_calls(result):
    return [c["command"] for c in result.shim_calls if c["tool"] == "ssh" and " verify " in c["command"]]


def _asked(command):
    """The services one remote `verify` call was asked about."""
    return [part.split()[0] for part in command.split("--service ")[1:]]


def _up(fake, workdir, way, *, config_extra="", shim_env=None, state=None):
    """One `up -C box`; `way` puts the fake in the shape that makes `up` launch, reuse or adopt."""
    if way == "reused":
        state = {**(state or {}), "instance_id": fake.add_instance()}
    elif way == "adopted":
        fake.add_instance()
    (workdir / "shim-calls.jsonl").unlink(missing_ok=True)
    result = run_cloudgpu(fake, workdir, ["up", "--config", "box"], config_extra=config_extra,
                          shim_env=shim_env, state=state)
    assert "Traceback" not in result.output, result.output
    return result


@pytest.fixture
def report(monkeypatch, capsys, tmp_path):
    """What the real remote `verify` prints for the given systemd states, to hand to the ssh shim."""
    def make(states, *args):
        return _remote_verify(monkeypatch, capsys, tmp_path, states, *args)[1]
    return make


def _assert_failed_and_still_billing(result, service_name):
    out = " ".join(result.output.split())
    assert result.returncode != 0, result.output
    assert "is up" not in result.output, result.output
    assert f"Service '{service_name}' is not running" in out, result.output
    assert "still running and billing" in out, result.output
    assert "cloudgpu down -C box" in out and "cloudgpu up -C box" in out, result.output
    assert _steps(result)[-1] == "verify", result.output  # stopped there: no report step


# (1) + (2)
@pytest.mark.parametrize("way", WAYS)
@pytest.mark.parametrize("with_profiles", [False, True], ids=["bare", "profiles"])
def test_up_runs_the_plans_eight_steps_and_succeeds_when_services_run(tmp_path, way, with_profiles):
    """Catches: verify missing or out of place on one path (launch, reuse, adopt; profiles or none)."""
    _profile(tmp_path, "gen", ["comfyui", "ai-toolkit"])
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, way, config_extra=GEN if with_profiles else "", shim_env=RECORDED_ENV)
        launches = len(fake.launches)
    assert result.returncode == 0, result.output
    assert launches == (1 if way == "fresh" else 0)
    assert _steps(result) == PLAN_ORDER, result.output
    assert [f"Step {n}/8" in result.output for n in range(1, 9)] == [True] * 8, result.output
    assert "Config 'box' is up" in result.output
    # After the restart (when there is one) and before the report's status read.
    events = _events(result)
    assert events[events.index("verify") - 1:] == [
        "restart" if with_profiles else "status", "verify", "status"], result.output
    [verify] = _verify_calls(result)
    assert _asked(verify) == (["comfyui", "ai-toolkit"] if with_profiles else ["comfyui"])
    out = result.output
    assert out.find("Step 6/8") < out.find("Step 7/8") < out.find("running: ") < out.find("Step 8/8") < out.find("is up")


# (3)
@pytest.mark.parametrize("how", list(DEAD))
def test_up_fails_when_a_service_is_not_running_after_the_restart(tmp_path, report, how):
    """Catches: `up` saying "is up" and exiting 0 over a service that died after its restart."""
    _profile(tmp_path, "gen", ["comfyui", "ai-toolkit"])
    dead = report({"comfyui": DEAD[how], "ai-toolkit": RUNNING})
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, "fresh", config_extra=GEN, shim_env={**RECORDED_ENV, "FAKE_SHIM_VERIFY": dead})
        alive = fake.alive()
    _assert_failed_and_still_billing(result, "comfyui")
    out = " ".join(result.output.split())
    assert SAYS[how] in out, result.output
    assert JOURNAL in out, result.output
    assert "Service 'ai-toolkit' is not running" not in out, result.output
    assert result.state["instance_id"] == "fake-instance-1" and alive == ["fake-instance-1"]
    assert result.state["provisioned_profiles"] == ["gen"]
    assert "fake-instance-1 is still running and billing" in out, result.output


# (4)
@pytest.mark.parametrize("way", WAYS)
def test_up_that_provisioned_nothing_fails_over_a_dead_recorded_service(tmp_path, report, way):
    """Catches: verify skipped when nothing was provisioned, so a dead recorded service passes."""
    dead = report({"comfyui": DEAD["inactive"]}, "--settle", "0")
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, way, state={"provisioned_profiles": ["gen"]},
                     shim_env={**RECORDED_ENV, "FAKE_SHIM_VERIFY": dead})
    _assert_failed_and_still_billing(result, "comfyui")
    assert "inactive/dead" in result.output
    assert "Nothing to provision." in result.output
    assert result.state["instance_id"] == "fake-instance-1"
    assert result.state["provisioned_profiles"] == ["gen"]


def test_verify_waits_the_settle_only_after_ups_own_restart(tmp_path):
    """Catches: the settle doubled right after recovery's confirm, or dropped after `up`'s restart."""
    _profile(tmp_path, "gen", ["comfyui"])
    with FakeLambda() as fake:
        bare = _up(fake, tmp_path, "fresh", shim_env=RECORDED_ENV)
        restarted = _up(fake, tmp_path, "as the state file says", config_extra=GEN, shim_env=RECORDED_ENV)
    assert bare.returncode == 0 and restarted.returncode == 0, bare.output + restarted.output
    assert "restart" not in _events(bare) and "restart" in _events(restarted)
    [after_recovery], [after_restart] = _verify_calls(bare), _verify_calls(restarted)
    assert after_recovery.endswith(" --settle 0"), after_recovery
    assert "--settle" not in after_restart, after_restart  # the remote's own SETTLE_SECONDS


# (5)
def test_verify_covers_a_recorded_service_this_ups_profiles_do_not_name(tmp_path, report):
    """Catches: verifying only this `up`'s profile apps, so a dead recorded service passes."""
    _profile(tmp_path, "tools", ["ai-toolkit"])
    dead = report({"comfyui": DEAD["failed"], "ai-toolkit": RUNNING})
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, "fresh", config_extra='profiles = ["tools"]\n',
                     shim_env={**RECORDED_ENV, "FAKE_SHIM_VERIFY": dead})
    [verify] = _verify_calls(result)
    assert _asked(verify) == ["comfyui", "ai-toolkit"]
    _assert_failed_and_still_billing(result, "comfyui")


def test_an_expected_service_missing_from_the_report_fails(tmp_path, report):
    """Catches: `up` passing because the report only lists healthy services (unknown is not active)."""
    _profile(tmp_path, "tools", ["ai-toolkit"])
    only_one = report({"ai-toolkit": RUNNING})
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, "fresh", config_extra='profiles = ["tools"]\n',
                     shim_env={**RECORDED_ENV, "FAKE_SHIM_VERIFY": only_one})
    _assert_failed_and_still_billing(result, "comfyui")
    assert "unknown" in result.output


# (6)
def test_a_failed_restart_with_services_down_fails_up(tmp_path, report):
    """Catches: the old ending: a failed restart warned about, then "is up" and exit 0."""
    _profile(tmp_path, "gen", ["comfyui"])
    dead = report({"comfyui": DEAD["inactive"]})
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, "fresh", config_extra=GEN,
                     shim_env={**RECORDED_ENV, "FAKE_SHIM_FAIL_ON": "systemctl restart", "FAKE_SHIM_VERIFY": dead})
    assert "Could not restart services" in result.output, result.output
    _assert_failed_and_still_billing(result, "comfyui")


def test_a_failed_restart_with_services_found_running_is_a_warning_and_exit_0(tmp_path):
    """Catches: the chosen ending changing unnoticed: the warning stays, verify decides, and it
    waits the full settle because `up` tried a restart."""
    _profile(tmp_path, "gen", ["comfyui"])
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, "fresh", config_extra=GEN,
                     shim_env={**RECORDED_ENV, "FAKE_SHIM_FAIL_ON": "systemctl restart"})
    assert result.returncode == 0, result.output
    assert "Could not restart services" in result.output, result.output
    [verify] = _verify_calls(result)
    assert "--settle" not in verify, verify
    assert result.output.find("Could not restart services") < result.output.find("Step 7/8") < result.output.find("is up")


# (7)
@pytest.mark.parametrize("fault", [
    {"FAKE_SHIM_FAIL_ON": " verify "},
    {"FAKE_SHIM_VERIFY": "this is not json"},
    {"FAKE_SHIM_VERIFY": '{"services": "none"}'},
    {"FAKE_SHIM_VERIFY": '[{"name": "comfyui", "ok": true}]'},
], ids=["ssh fails", "not json", "wrong shape", "not an object"])
def test_up_fails_in_a_sentence_when_the_check_cannot_be_made(tmp_path, fault):
    """Catches: an unreadable check passing as healthy, or surfacing as a traceback."""
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, "fresh", shim_env={**RECORDED_ENV, **fault})
    assert result.returncode != 0, result.output
    assert "Traceback" not in result.output, result.output
    assert "is up" not in result.output, result.output
    out = " ".join(result.output.split())
    assert "Couldn't check whether comfyui" in out, result.output
    assert "still running and billing" in out, result.output
    assert result.state["instance_id"] == "fake-instance-1"


# (8)
@pytest.mark.parametrize("with_profile", [False, True], ids=["bare", "appless profile"])
def test_no_service_apps_means_nothing_to_verify_and_up_passes(tmp_path, with_profile):
    """Catches: a machine with no services failing verify, or making a pointless remote call."""
    _profile(tmp_path, "plain", [])
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, "fresh", config_extra='profiles = ["plain"]\n' if with_profile else "",
                     shim_env={"FAKE_SHIM_FAIL_ON": " verify "})
    assert result.returncode == 0, result.output
    assert _steps(result) == PLAN_ORDER, result.output
    assert "nothing to verify" in result.output, result.output
    assert _verify_calls(result) == []
    assert "is up" in result.output


# (9)
def test_up_again_after_a_failed_verify_succeeds_on_the_same_instance(tmp_path, report):
    """Catches: a failed verify leaving state that makes the next `up` launch again, or fail."""
    _profile(tmp_path, "gen", ["comfyui"])
    dead = report({"comfyui": DEAD["restart loop"]})
    with FakeLambda() as fake:
        first = _up(fake, tmp_path, "fresh", config_extra=GEN, shim_env={**RECORDED_ENV, "FAKE_SHIM_VERIFY": dead})
        _assert_failed_and_still_billing(first, "comfyui")
        again = _up(fake, tmp_path, "as the state file says", config_extra=GEN, shim_env=RECORDED_ENV)
        launches, alive = len(fake.launches), fake.alive()
    assert again.returncode == 0, again.output
    assert launches == 1 and alive == ["fake-instance-1"], again.output
    assert "Reusing running instance fake-instance-1" in again.output, again.output
    assert _steps(again) == PLAN_ORDER and "is up" in again.output, again.output
    assert again.state["launched_at"] == first.state["launched_at"]


# The mark: unchanged by this item, and pinned so a change to it is a decision, not an accident.
def test_a_reprovision_mark_is_still_cleared_before_verify_even_when_verify_fails(tmp_path, report):
    """Catches: the mark's clearing moved by this item without anyone deciding it."""
    _profile(tmp_path, "gen", ["comfyui"])
    dead = report({"comfyui": DEAD["failed"]})
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, "fresh", state={"provisioned_profiles": ["gen"]},
                     shim_env={"FAKE_SHIM_APPS": json.dumps(MARKED), "FAKE_SHIM_VERIFY": dead})
    _assert_failed_and_still_billing(result, "comfyui")
    assert _events(result)[-3:] == ["restart", "clear mark", "verify"], result.output
