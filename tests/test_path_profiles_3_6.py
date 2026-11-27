"""A profile given by path is remembered by its path; a recorded profile that won't load is named
(plan item 3.6).

No real API, instance, bucket or key. `up` runs as the real CLI in a subprocess against the fake
Lambda, with ssh and rsync shimmed (as tests/test_up_recovery.py and test_datasync_up_3_2.py do).
`forward` and `status` run in process with ssh monkeypatched (as tests/test_cli_forward.py does).

The profile under test lives outside any `profiles/` folder and is only ever given by path.
"""

from __future__ import annotations

import json
import shutil

from click.testing import CliRunner

from cloudgpu import cli
from cloudgpu.local import apps
from tests.fake_lambda import FakeLambda, run_cloudgpu
from tests.test_datasync_up_3_2 import GOOD_CONF, _sent_text, _shim_env

CAP = "auto_terminate_hours = 8\n"
SYNC = '[sync]\nremote = "b2:bkt/small"\n'
COMFY = {"comfyui": {"status": "installed", "version": "1", "app_dir": "/lambda/nfs/box-fs/apps/comfyui"}}
REBUILT = {"comfyui": {**COMFY["comfyui"], "reprovision": True}}
STATUS = {"detection": {"persistent_dir": "/lambda/nfs/p", "gpu": {"available": False}}, "apps": {}}


def _make_profile(d, app_list, extra=""):
    d.mkdir(parents=True, exist_ok=True)
    (d / "cloudgpu.toml").write_text(f"apps = {json.dumps(app_list)}\n{extra}")
    (d / "provision.py").write_text("print('provisioned')\n")
    return d


def _outside(tmp_path, name="pathprof", app_list=("comfyui",), extra='ports = [7777]\n'
             '[sync]\nextra_paths = ["pathprof-notes"]\n'):
    """A profile in a folder that is not under any profiles/ folder."""
    return _make_profile(tmp_path / "outside" / name, list(app_list), extra)


def _repo(tmp_path, name="repo"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _flat(output):
    return " ".join(output.split())


def _says(output, text):
    """``text`` is in the output, ignoring where the in-process console wrapped lines (inside a path)."""
    return "".join(text.split()) in "".join(output.split())


def _up(workdir, *args, state=None, config_extra=CAP, shim=None):
    for name in ("shim-calls.jsonl", "ssh-seen.jsonl"):  # each run's calls only
        (workdir / name).unlink(missing_ok=True)
    with FakeLambda() as fake:
        if state and state.get("instance_id") == "reuse":
            state = {**state, "instance_id": fake.add_instance()}
        result = run_cloudgpu(fake, workdir, ["up", "--config", "box", *args], config_extra=config_extra,
                              state=state, shim_env=shim)
    assert "Traceback" not in result.output, result.output
    return result


def _provisions(calls):
    """The remote provision folder of each ssh call that ran a profile's provision entry point."""
    return [c["command"].split("/cloudgpu/provision/")[1].split(" ")[0] for c in calls
            if c["tool"] == "ssh" and "provision.py" in c["command"]]


def _first_up_by_path(tmp_path):
    """`up -P ../outside/pathprof` from repo/: the only time the profile is ever named."""
    prof = _outside(tmp_path)
    repo = _repo(tmp_path)
    first = _up(repo, "-P", "../outside/pathprof")
    assert first.returncode == 0, first.output
    return prof, repo, first


# --- recording ------------------------------------------------------------------------------------


def test_up_records_a_profile_given_by_path_by_its_resolved_path(tmp_path):
    """Catches: a profile given by path recorded as its bare folder name, which no later step can find."""
    prof, _, first = _first_up_by_path(tmp_path)
    assert first.state["provisioned_profiles"] == [str(prof.resolve())]


def test_a_profile_under_profiles_is_still_recorded_by_name_even_given_by_path(tmp_path):
    """Catches: names turned into paths for profiles under profiles/ (state would stop being portable)."""
    repo = _repo(tmp_path)
    _make_profile(repo / "profiles" / "gen", ["comfyui"])
    _make_profile(repo / "profiles" / "tool", [])
    result = _up(repo, "-P", "gen,./profiles/tool")
    assert result.returncode == 0, result.output
    assert result.state["provisioned_profiles"] == ["gen", "tool"]


def test_provision_records_a_profile_given_by_path_by_its_resolved_path(tmp_path, monkeypatch):
    """Catches: `cloudgpu provision -P <path>` still recording only the folder name."""
    prof = _outside(tmp_path)
    repo = _repo(tmp_path)
    (repo / "configs").mkdir()
    (repo / "configs" / "c.toml").write_text('gpu = ["gh200"]\nssh_key = "mini"\n')
    (repo / "configs" / "c.state.json").write_text(json.dumps(
        {"host": "ubuntu@h", "persistent_dir": "/lambda/nfs/p", "provisioned_profiles": ["old"]}))
    monkeypatch.chdir(repo)

    def ssh_run(host, command, **kwargs):
        out = json.dumps({"apps": COMFY}) if command.endswith(" status") else ""
        return cli.ssh.SSHResult(0, out, "")

    monkeypatch.setattr(cli.sync, "sync_remote", lambda *a, **k: None)
    monkeypatch.setattr(cli.ssh, "ssh_run", ssh_run)
    monkeypatch.setattr(cli, "_run_provision", lambda *a, **k: True)
    monkeypatch.setattr(cli, "_restart_services", lambda *a, **k: None)
    r = CliRunner().invoke(cli.provision, ["-C", "c", "-P", "../outside/pathprof"])
    assert r.exit_code == 0, r.output
    state = json.loads((repo / "configs" / "c.state.json").read_text())
    assert state["provisioned_profiles"] == ["old", str(prof.resolve())]


# --- a later `up` with no -P ------------------------------------------------------------------------


def test_a_later_up_without_P_keeps_the_path_profile_and_reports_it(tmp_path):
    """Catches: the next `up` reporting "Bare machine" and no profile, as the Phase 3 session saw."""
    prof, repo, first = _first_up_by_path(tmp_path)
    second = _up(repo, state={**first.state, "instance_id": "reuse"}, shim={"FAKE_SHIM_APPS": json.dumps(COMFY)})
    assert second.returncode == 0, second.output
    out = _flat(second.output)
    assert f"is up (profiles: {prof.resolve()})" in out, out
    assert "Bare machine" not in out, out
    assert "http://localhost:8188" in out, out
    assert "Profile ports forwarded on demand (7777)" in out, out
    assert "Can't load" not in out and "won't load" not in out, out
    assert second.state["provisioned_profiles"] == [str(prof.resolve())]


def test_a_later_up_from_another_directory_still_finds_the_path_profile(tmp_path):
    """Catches: a path recorded as given (relative), which means something else from another folder."""
    prof, _, first = _first_up_by_path(tmp_path)
    elsewhere = _repo(tmp_path, "somewhere-else/deeper")
    second = _up(elsewhere, state={**first.state, "instance_id": "reuse"},
                 shim={"FAKE_SHIM_APPS": json.dumps(COMFY)})
    assert second.returncode == 0, second.output
    out = _flat(second.output)
    assert "http://localhost:8188" in out and "Bare machine" not in out, out
    assert str(prof.resolve()) in out


def test_recovery_reprovisions_the_path_profile_when_its_app_was_rebuilt(tmp_path):
    """Catches: a rebuilt ComfyUI venv never getting a path profile's nodes back (it was forgotten)."""
    prof, repo, first = _first_up_by_path(tmp_path)
    second = _up(repo, state={**first.state, "instance_id": "reuse"},
                 shim={"FAKE_SHIM_APPS": json.dumps(COMFY), "FAKE_SHIM_APPS_AFTER_RECOVER": json.dumps(REBUILT)})
    assert second.returncode == 0, second.output
    assert _provisions(second.shim_calls) == ["pathprof"], second.output
    clears = [c for c in second.shim_calls if c["tool"] == "ssh" and " clear-reprovision " in c["command"]]
    assert len(clears) == 1, second.output
    assert second.state["provisioned_profiles"] == [str(prof.resolve())]


def test_reprovision_does_not_run_a_path_profile_twice_when_it_is_also_given(tmp_path):
    """Catches: the recorded path and the -P path compared by different keys, so it provisions twice."""
    prof, repo, first = _first_up_by_path(tmp_path)
    second = _up(repo, "-P", "../outside/pathprof", state={**first.state, "instance_id": "reuse"},
                 shim={"FAKE_SHIM_APPS": json.dumps(COMFY), "FAKE_SHIM_APPS_AFTER_RECOVER": json.dumps(REBUILT)})
    assert second.returncode == 0, second.output
    assert _provisions(second.shim_calls) == ["pathprof"], second.output
    assert second.state["provisioned_profiles"] == [str(prof.resolve())]


def test_sync_paths_include_the_recorded_path_profile(tmp_path):
    """Catches: sync's path list leaving out a path profile's apps' folders and its extra paths."""
    prof, repo, first = _first_up_by_path(tmp_path)
    conf = repo / "home" / ".config" / "cloudgpu" / "rclone.conf"
    conf.parent.mkdir(parents=True, exist_ok=True)
    conf.write_text(GOOD_CONF)
    conf.chmod(0o600)
    second = _up(repo, state={**first.state, "instance_id": "reuse"}, config_extra=CAP + SYNC,
                 shim=_shim_env(repo, FAKE_SHIM_APPS=json.dumps(COMFY)))
    assert second.returncode == 0, second.output
    request = json.loads(_sent_text(second, cli.SYNC_REQUEST_TRANSFER))
    assert request["paths"] == [*apps.APPS["comfyui"].sync_paths, "pathprof-notes"], request


# --- a recorded profile that won't load -------------------------------------------------------------


def test_up_warns_naming_a_recorded_profile_whose_folder_is_gone_and_carries_on(tmp_path):
    """Catches: a recorded profile silently dropped by `up` (its apps quietly lose their ports)."""
    prof, repo, first = _first_up_by_path(tmp_path)
    _make_profile(repo / "profiles" / "gen", ["ai-toolkit"])
    shutil.rmtree(prof)
    state = {**first.state, "instance_id": "reuse",
             "provisioned_profiles": [*first.state["provisioned_profiles"], "gen", "gone-by-name"]}
    second = _up(repo, state=state, shim={"FAKE_SHIM_APPS": json.dumps(COMFY)})
    assert second.returncode == 0, second.output
    out = _flat(second.output)
    assert f"Recorded profile '{prof.resolve()}' can't be loaded" in out, out
    assert "Recorded profile 'gone-by-name' can't be loaded" in out, out
    assert "http://localhost:8675" in out, out  # gen's ai-toolkit still reported
    # Nothing is forgotten: fixing the folder brings it back.
    assert second.state["provisioned_profiles"] == state["provisioned_profiles"]


def test_sync_up_warns_once_per_missing_profile(tmp_path):
    """Catches: the warning repeated by every step that reloads the recorded profiles."""
    prof, repo, first = _first_up_by_path(tmp_path)
    shutil.rmtree(prof)
    conf = repo / "home" / ".config" / "cloudgpu" / "rclone.conf"
    conf.parent.mkdir(parents=True, exist_ok=True)
    conf.write_text(GOOD_CONF)
    conf.chmod(0o600)
    second = _up(repo, state={**first.state, "instance_id": "reuse"}, config_extra=CAP + SYNC,
                 shim=_shim_env(repo, FAKE_SHIM_APPS=json.dumps(COMFY)))
    assert second.returncode == 0, second.output
    assert _flat(second.output).count(f"Recorded profile '{prof.resolve()}' can't be loaded") == 1, second.output


def _forward_repo(tmp_path, monkeypatch, provisioned):
    repo = _repo(tmp_path, "fwd")
    (repo / "configs").mkdir()
    (repo / "configs" / "c.toml").write_text('gpu = ["gh200"]\nssh_key = "mini"\n')
    (repo / "configs" / "c.state.json").write_text(json.dumps({
        "host": "ubuntu@1.2.3.4", "persistent_dir": "/lambda/nfs/p", "provisioned_profiles": provisioned}))
    monkeypatch.chdir(repo)
    monkeypatch.setattr(cli, "_resolve_target", lambda host=None, config_name=None: ("ubuntu@1.2.3.4", "/lambda/nfs/p"))
    captured = {}

    def fake(host, command=None, *, ssh_args=None):
        captured.update(host=host, command=command, ssh_args=ssh_args)
        return 0

    monkeypatch.setattr(cli.ssh, "ssh_interactive", fake)
    return repo, captured


def test_forward_without_p_forwards_the_path_profiles_ports_from_another_directory(tmp_path, monkeypatch):
    """Catches: `forward -C` finding no ports after `up -P <path>` (state written by a real `up`)."""
    prof, _, first = _first_up_by_path(tmp_path)
    _, cap = _forward_repo(tmp_path, monkeypatch, first.state["provisioned_profiles"])
    r = CliRunner().invoke(cli.forward, ["-C", "c"])
    assert r.exit_code == 0, r.output
    assert cap["ssh_args"] == ["-L", "8188:localhost:8188", "-L", "7777:localhost:7777", "-N"]
    assert "can't be loaded" not in r.output


def test_forward_warns_naming_a_missing_recorded_profile_and_forwards_the_rest(tmp_path, monkeypatch):
    """Catches: `forward` silently skipping a recorded profile it can't load."""
    gone = tmp_path / "outside" / "gone"
    repo, cap = _forward_repo(tmp_path, monkeypatch, [str(gone), "gen", "gone-by-name"])
    _make_profile(repo / "profiles" / "gen", ["ai-toolkit"])
    r = CliRunner().invoke(cli.forward, ["-C", "c"])
    assert r.exit_code == 0, r.output
    out = _flat(r.output)
    assert _says(out, f"Recorded profile '{gone}' can't be loaded"), out
    assert "Recorded profile 'gone-by-name' can't be loaded" in out, out
    assert cap["ssh_args"] == ["-L", "8675:localhost:8675", "-N"]


def _status(tmp_path, monkeypatch, provisioned):
    repo, _ = _forward_repo(tmp_path, monkeypatch, provisioned)
    monkeypatch.setattr(cli.sync, "sync_remote", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_remote_run", lambda *a, **k: json.dumps(STATUS))
    return repo, CliRunner().invoke(cli.status, ["-C", "c"])


def test_status_warns_naming_a_missing_recorded_profile(tmp_path, monkeypatch):
    """Catches: `status` saying nothing about a recorded profile no command can load any more."""
    gone = tmp_path / "outside" / "gone"
    _, r = _status(tmp_path, monkeypatch, [str(gone), "gone-by-name"])
    assert r.exit_code == 0, r.output
    out = _flat(r.output)
    assert _says(out, f"Recorded profile '{gone}' can't be loaded"), out
    assert "Recorded profile 'gone-by-name' can't be loaded" in out, out


def test_status_is_quiet_when_every_recorded_profile_loads(tmp_path, monkeypatch):
    """Catches: a warning for profiles that load fine (a path one included)."""
    prof = _outside(tmp_path)
    repo = _repo(tmp_path, "fwd")
    _make_profile(repo / "profiles" / "gen", ["ai-toolkit"])
    _, r = _status(tmp_path, monkeypatch, [str(prof.resolve()), "gen"])
    assert r.exit_code == 0, r.output
    assert "can't be loaded" not in r.output


# --- state written before the fix ---------------------------------------------------------------------


def test_a_plain_names_state_file_loads_and_behaves_as_before(tmp_path, monkeypatch):
    """Catches: state from before the fix (plain names) rewritten, rejected, or no longer found."""
    repo = _repo(tmp_path)
    _make_profile(repo / "profiles" / "gen", ["comfyui"], '[sync]\nextra_paths = ["gen-notes"]\n')
    _make_profile(repo / "profiles" / "lora", ["ai-toolkit"])
    conf = repo / "home" / ".config" / "cloudgpu" / "rclone.conf"
    conf.parent.mkdir(parents=True, exist_ok=True)
    conf.write_text(GOOD_CONF)
    conf.chmod(0o600)
    old = {"provisioned_profiles": ["gen", "lora"], "instance_id": "reuse"}
    result = _up(repo, state=old, config_extra=CAP + SYNC,
                 shim=_shim_env(repo, FAKE_SHIM_APPS=json.dumps(COMFY)))
    assert result.returncode == 0, result.output
    out = _flat(result.output)
    assert "is up (profiles: gen, lora)" in out, out
    assert "http://localhost:8188" in out and "http://localhost:8675" in out, out
    assert "can't be loaded" not in out, out
    assert result.state["provisioned_profiles"] == ["gen", "lora"]
    request = json.loads(_sent_text(result, cli.SYNC_REQUEST_TRANSFER))
    assert "gen-notes" in request["paths"]
    assert all(p in request["paths"] for p in apps.APPS["ai-toolkit"].sync_paths), request

    monkeypatch.chdir(repo)
    monkeypatch.setattr(cli, "_resolve_target", lambda host=None, config_name=None: ("h", "/p"))
    cap = {}
    monkeypatch.setattr(cli.ssh, "ssh_interactive", lambda host, command=None, *, ssh_args=None: cap.update(a=ssh_args) or 0)
    r = CliRunner().invoke(cli.forward, ["-C", "box"])
    assert r.exit_code == 0, r.output
    assert cap["a"] == ["-L", "8188:localhost:8188", "-L", "8675:localhost:8675", "-N"]
