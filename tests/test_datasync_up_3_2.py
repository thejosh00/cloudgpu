"""`cloudgpu up` puts the bucket sync on the instance, or takes it away (plan item 3.2, part 3).

No real API, instance, bucket, key or rclone. The real CLI runs in a subprocess against the fake
Lambda (as tests/test_up_verify_2_3.py does), with a throwaway HOME holding a made-up
rclone.conf. Two extra shims sit in front of the usual ones: an rsync that also keeps a copy of
each file it was asked to send (so a test can read what would have reached the instance), and an
ssh that also records its whole argv and environment before handing over to the usual shim.
"""

from __future__ import annotations

import json
import os
import stat
import sys

import pytest

from cloudgpu import cli
from cloudgpu.local import apps, config, configs, profiles
from tests.fake_lambda import SHIMS, FakeLambda, run_cloudgpu

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
WAYS = ["fresh", "reused", "adopted"]
SYNC = '[sync]\nremote = "b2:bkt/small"\n'
KEY_ID = "0041aaaabbbbcccc0000000001"
KEY = "K004SecretApplicationKeyNeverShown"
OTHER_KEY = "OtherRemoteSecretLeftBehind"
GOOD_CONF = (
    f"[b2]\ntype = b2\naccount = {KEY_ID}\nkey = {KEY}\nHardDelete = false\n\n"
    f"[other]\ntype = s3\nsecret_access_key = {OTHER_KEY}\n"
)
SECRETS = (KEY_ID, KEY, OTHER_KEY)

RSYNC_SHIM = '''#!/usr/bin/env python3
"""rsync for these tests: records the whole command line and keeps a copy of a file it sends."""
import json, os, shutil, stat, sys
args = sys.argv[1:]
source, destination = args[-2], args[-1]
keep = os.environ["FAKE_KEEP_DIR"]
record = {"tool": "rsync", "host": destination.split(":", 1)[0], "command": " ".join(args),
          "source": source, "destination": destination.split(":", 1)[1], "env": dict(os.environ)}
if os.path.isfile(source):
    record["mode"] = oct(stat.S_IMODE(os.stat(source).st_mode))
    record["dir_mode"] = oct(stat.S_IMODE(os.stat(os.path.dirname(source)).st_mode))
    record["kept"] = os.path.join(keep, "%d-%s" % (len(os.listdir(keep)), os.path.basename(destination)))
    shutil.copyfile(source, record["kept"])
with open(os.environ["FAKE_SHIM_LOG"], "a") as log:
    log.write(json.dumps(record) + "\\n")
fail_on = os.environ.get("FAKE_RSYNC_FAIL_ON")
sys.exit(1 if fail_on and fail_on in destination else 0)
'''
SSH_SHIM = '''#!/usr/bin/env python3
"""ssh for these tests: records argv and environment, then runs the usual shim."""
import json, os, sys
with open(os.environ["FAKE_SSH_SEEN"], "a") as seen:
    seen.write(json.dumps({"argv": sys.argv[1:], "env": dict(os.environ)}) + "\\n")
os.execv(sys.executable, [sys.executable, os.environ["FAKE_REAL_SSH"], *sys.argv[1:]])
'''


def _shim_env(workdir, **extra):
    """PATH with the recording shims first; where they keep what they saw."""
    shims = workdir / "extra-shims"
    shims.mkdir(exist_ok=True)
    for name, text in (("rsync", RSYNC_SHIM), ("ssh", SSH_SHIM)):
        (shims / name).write_text(text.replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
        (shims / name).chmod(0o755)
    (workdir / "kept").mkdir(exist_ok=True)
    return {
        "PATH": os.pathsep.join([str(shims), str(SHIMS), os.environ["PATH"]]),
        "FAKE_KEEP_DIR": str(workdir / "kept"),
        "FAKE_SSH_SEEN": str(workdir / "ssh-seen.jsonl"),
        "FAKE_REAL_SSH": str(SHIMS / "ssh"),
        **extra,
    }


def _rclone_conf(workdir, text=GOOD_CONF):
    path = workdir / "home" / ".config" / "cloudgpu" / "rclone.conf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o600)
    return path


def _profile(workdir, name, app_list=(), extra=""):
    d = workdir / "profiles" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "cloudgpu.toml").write_text(f"apps = {json.dumps(list(app_list))}\n{extra}")
    (d / "provision.py").write_text("print('provisioned')\n")


def _up(fake, workdir, way="fresh", *, config_extra=SYNC, state=None, **shim_extra):
    """One `up -C box`; `way` makes it launch, reuse, adopt, or go by the state file as it is."""
    if way == "reused":
        state = {**(state or {}), "instance_id": fake.add_instance()}
    elif way == "adopted":
        fake.add_instance()
    for name in ("shim-calls.jsonl", "ssh-seen.jsonl"):
        (workdir / name).unlink(missing_ok=True)
    result = run_cloudgpu(fake, workdir, ["up", "--config", "box"], config_extra=config_extra,
                          shim_env=_shim_env(workdir, **shim_extra), state=state)
    assert "Traceback" not in result.output, result.output
    return result


def _steps(result):
    return [line.split(": ", 1)[1].strip() for line in result.output.splitlines()
            if line.startswith("Step ") and ": " in line]


def _datasync(result):
    """The remote `datasync` actions `up` ran over ssh, in order."""
    return [c["command"].split(" datasync ", 1)[1].split()[0] for c in result.shim_calls
            if c["tool"] == "ssh" and " datasync " in c["command"]]


def _sent(result, destination):
    """The rsync calls that sent a file to this home-relative name on the instance."""
    return [c for c in result.shim_calls if c["tool"] == "rsync" and c.get("destination") == destination]


def _sent_text(result, destination):
    [call] = _sent(result, destination)
    with open(call["kept"]) as f:
        return f.read()


def _assert_no_secret_anywhere(result, workdir):
    """No key value in any recorded ssh or rsync command line, any ssh argv or environment, or output."""
    seen = (workdir / "ssh-seen.jsonl").read_text() if (workdir / "ssh-seen.jsonl").exists() else ""
    assert result.shim_calls, "nothing was recorded, so this would prove nothing"
    for secret in SECRETS:
        assert secret not in result.output, result.output
        for call in result.shim_calls:
            assert secret not in json.dumps(call), call
        assert secret not in seen


def _assert_stopped_still_billing(result, sentence):
    out = " ".join(result.output.split())
    assert result.returncode != 0, result.output
    assert "Traceback" not in result.output, result.output
    assert sentence in out, result.output
    assert "fake-instance-1 is still running and billing" in out, result.output
    assert "cloudgpu down -C box" in out and "cloudgpu up -C box" in out, result.output
    assert "is up" not in result.output, result.output
    assert _steps(result)[-1] == "arm the cap", result.output  # before the slow steps
    assert result.state["instance_id"] == "fake-instance-1"


# (1)
@pytest.mark.parametrize("way", WAYS)
def test_up_without_sync_removes_it_and_installs_nothing(tmp_path, way):
    """Catches: a config with no [sync] leaving an old timer and key on the instance, sending a
    key, needing rclone.conf, or gaining a step."""
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, way, config_extra="")
    assert result.returncode == 0, result.output
    assert _datasync(result) == ["remove"], result.output
    assert not _sent(result, cli.SYNC_KEY_TRANSFER) and not _sent(result, cli.SYNC_REQUEST_TRANSFER)
    assert _steps(result) == PLAN_ORDER, result.output
    assert "Config 'box' is up" in result.output
    assert "bucket sync" not in result.output and "Sync timer" not in result.output, result.output  # quiet


# (2)
@pytest.mark.parametrize("conf, why", [
    (None, "does not exist"),
    (f"[other]\ntype = s3\nsecret_access_key = {OTHER_KEY}\n", "has no [b2] section"),
    (f"key = {KEY}\n[b2\n", "could not be read"),
], ids=["no file", "no section", "not an rclone config"])
@pytest.mark.parametrize("way", ["fresh", "reused"])
def test_up_with_sync_stops_before_any_launch_or_ssh_without_the_key(tmp_path, conf, why, way):
    """Catches: a machine launched (and billing) or changed before `up` notices the bucket key is
    missing; or the refusal quoting a line of the key file."""
    if conf is not None:
        _rclone_conf(tmp_path, conf)
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, way)
        launches, requests = len(fake.launches), list(fake.requests)
    out = " ".join(result.output.split())
    assert result.returncode != 0, result.output
    assert launches == 0 and requests == [], requests
    assert result.shim_calls == [], result.shim_calls
    assert str(tmp_path / "home" / ".config" / "cloudgpu" / "rclone.conf") in out, result.output
    assert "[b2]" in out and why in out, result.output
    assert "Step " not in result.output, result.output
    for secret in SECRETS:
        assert secret not in result.output, result.output
    assert result.state == ({} if way == "fresh" else {"instance_id": "fake-instance-1"})


# (3)
@pytest.mark.parametrize("way", WAYS)
def test_only_the_remotes_section_travels_and_the_key_is_in_no_command(tmp_path, way):
    """Catches: the whole rclone.conf (other remotes' keys) sent to the instance; the key put on
    an ssh command line, in an ssh environment or in the output; a temp copy left on this machine."""
    _rclone_conf(tmp_path)
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, way)
    assert result.returncode == 0, result.output
    assert _steps(result) == PLAN_ORDER, result.output
    sent = _sent_text(result, cli.SYNC_KEY_TRANSFER)
    assert sent == f"[b2]\ntype = b2\naccount = {KEY_ID}\nkey = {KEY}\nHardDelete = false\n"
    assert "other" not in sent and OTHER_KEY not in sent
    [call] = _sent(result, cli.SYNC_KEY_TRANSFER)
    assert call["mode"] == "0o600" and call["dir_mode"] == "0o700", call
    assert not os.path.exists(call["source"]) and not os.path.exists(os.path.dirname(call["source"]))
    # Nothing else that was sent holds a key either, and no call names one.
    for other in (c for c in result.shim_calls if c.get("kept") and c is not call):
        with open(other["kept"]) as f:
            text = f.read()
        assert not any(s in text for s in SECRETS), other
    _assert_no_secret_anywhere(result, tmp_path)
    # The setup names the transferred files, and runs after both arrived.
    commands = [c.get("destination") or c["command"] for c in result.shim_calls]
    [setup] = [c for c in commands if " datasync setup" in c]
    assert f"--key-file {cli.SYNC_KEY_TRANSFER}" in setup and f"--spec-file {cli.SYNC_REQUEST_TRANSFER}" in setup
    assert commands.index(cli.SYNC_KEY_TRANSFER) < commands.index(setup)
    assert commands.index(cli.SYNC_REQUEST_TRANSFER) < commands.index(setup)


def test_the_local_copy_of_the_key_is_removed_when_sending_it_fails(tmp_path):
    """Catches: the temp key file left on this machine when the transfer fails; the failure
    shown as a traceback or without the billing reminder."""
    _rclone_conf(tmp_path)
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, FAKE_RSYNC_FAIL_ON=cli.SYNC_KEY_TRANSFER)
    _assert_stopped_still_billing(result, "Could not set up the bucket sync")
    [call] = _sent(result, cli.SYNC_KEY_TRANSFER)
    assert not os.path.exists(call["source"]) and not os.path.exists(os.path.dirname(call["source"]))
    assert _datasync(result) == [], result.output
    _assert_no_secret_anywhere(result, tmp_path)


def test_install_and_start_are_two_remote_calls_in_that_order(tmp_path):
    """Catches: the timer started by the setup call itself (`--start`), leaving item 3.3 nowhere
    to put a restore between "in place" and "running"; or the timer never started."""
    _rclone_conf(tmp_path)
    with FakeLambda() as fake:
        result = _up(fake, tmp_path)
    assert result.returncode == 0, result.output
    assert _datasync(result) == ["setup", "restore", "start"], result.output
    [setup] = [c["command"] for c in result.shim_calls if " datasync setup" in c["command"]]
    assert "--start" not in setup, setup
    # Inside "arm the cap": after the cap's own call, before recovery.
    remote = [c["command"] for c in result.shim_calls if c["tool"] == "ssh" and "python3 -m remote" in c["command"]]
    kinds = [c.split("--persistent-dir ", 1)[1].split(" ", 1)[1] for c in remote]
    at = kinds.index("autoterminate --hours 0")
    assert [k.split(" --")[0] for k in kinds[at:at + 5]] == [
        "autoterminate", "datasync setup", "datasync restore", "datasync start", "status"], kinds


# (8)
def test_the_instance_gets_exactly_sync_paths_for_the_config_apps_and_profiles(tmp_path, monkeypatch):
    """Catches: the path list recomputed differently from apps.sync_paths, or dropping the
    profile's or the config's extra paths, the interval, or the remote."""
    _rclone_conf(tmp_path)
    _profile(tmp_path, "gen", ["comfyui"], '[sync]\nextra_paths = ["notebooks", "apps/comfyui/output/keep"]\n')
    extra = 'profiles = ["gen"]\n[sync]\nremote = "b2:bkt/small"\ninterval_minutes = 15\nextra_paths = ["datasets/mine"]\n'
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, config_extra=extra)
    assert result.returncode == 0, result.output
    request = json.loads(_sent_text(result, cli.SYNC_REQUEST_TRANSFER))
    monkeypatch.chdir(tmp_path)
    cfg, prof = configs.load_config("box"), profiles.load_profile("gen")
    expected = apps.sync_paths(cfg, ["comfyui"], [prof])
    assert request == {"remote": "b2:bkt/small", "paths": expected, "interval_minutes": 15}
    assert "notebooks" in expected and "datasets/mine" in expected and len(expected) > 2


def test_paths_cover_profiles_already_on_the_filesystem_and_default_interval(tmp_path):
    """Catches: an `up` without -P syncing nothing of a profile an earlier `up` provisioned
    (its outputs would never reach the bucket); the default interval not being 5."""
    _rclone_conf(tmp_path)
    _profile(tmp_path, "train", ["ai-toolkit"], '[sync]\nextra_paths = ["notes"]\n')
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, state={"provisioned_profiles": ["train"]})
    assert result.returncode == 0, result.output
    request = json.loads(_sent_text(result, cli.SYNC_REQUEST_TRANSFER))
    assert request["interval_minutes"] == 5
    assert request["paths"] == [*apps.APPS["ai-toolkit"].sync_paths, "notes"]


def test_a_path_that_leaves_the_persistent_dir_stops_up_before_launch(monkeypatch, tmp_path, capsys):
    """Catches: an absolute or '..' path sent to the instance after a launch, to be refused there."""
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    (tmp_path / "rclone.conf").write_text(GOOD_CONF)
    cfg = {"sync": {"remote": "b2:bkt", "interval_minutes": 5, "extra_paths": []}}
    for bad in ("/etc", "a/../../b", "~/x"):
        monkeypatch.setattr(apps, "sync_paths", lambda *a, bad=bad: ["ok/path", bad])
        with pytest.raises(SystemExit) as stop:
            cli._plan_sync(cfg, {}, [])
        assert stop.value.code == 1
        err = " ".join(capsys.readouterr().err.split())
        assert f"'{bad}'" in err and "Nothing was launched" in err, err
    monkeypatch.setattr(apps, "sync_paths", lambda *a: ["ok/path"])
    assert cli._plan_sync(cfg, {}, []) == {"remote": "b2:bkt", "paths": ["ok/path"], "interval_minutes": 5}
    assert cli._plan_sync({"sync": None}, {}, []) is None


# (9)
def test_a_second_up_converges_again_and_removing_sync_takes_it_away(tmp_path):
    """Catches: sync set up only on a fresh launch (a reused instance left without it, or with a
    stale path list), or left running after [sync] is deleted from the config."""
    _rclone_conf(tmp_path)
    with FakeLambda() as fake:
        first = _up(fake, tmp_path)
        second = _up(fake, tmp_path, "as the state file says")
        (tmp_path / "home" / ".config" / "cloudgpu" / "rclone.conf").unlink()  # not needed without [sync]
        third = _up(fake, tmp_path, "as the state file says", config_extra="")
        launches = len(fake.launches)
    for result in (first, second, third):
        assert result.returncode == 0, result.output
        assert _steps(result) == PLAN_ORDER, result.output
    assert launches == 1
    assert _datasync(first) == ["setup", "restore", "start"] and _datasync(second) == ["setup", "restore", "start"]
    assert _sent_text(second, cli.SYNC_KEY_TRANSFER) == _sent_text(first, cli.SYNC_KEY_TRANSFER)
    assert _sent_text(second, cli.SYNC_REQUEST_TRANSFER) == _sent_text(first, cli.SYNC_REQUEST_TRANSFER)
    assert _datasync(third) == ["remove"], third.output
    assert not _sent(third, cli.SYNC_KEY_TRANSFER)


# (10)
@pytest.mark.parametrize("fail_on, config_extra, sentence, ran", [
    (" datasync setup", SYNC, "Could not set up the bucket sync", ["setup"]),
    (" datasync start", SYNC, "Could not start the bucket sync timer", ["setup", "restore", "start"]),
    (" datasync remove", "", "Could not remove the bucket sync", ["remove"]),
], ids=["setup", "start", "remove"])
def test_a_failed_remote_sync_step_fails_up_in_a_sentence_and_keeps_the_instance(
        tmp_path, fail_on, config_extra, sentence, ran):
    """Catches: `up` carrying on to "is up" with the sync the person asked for absent (or an old
    one still running); the failure as a traceback; the instance dropped from the state file."""
    _rclone_conf(tmp_path)
    with FakeLambda() as fake:
        result = _up(fake, tmp_path, config_extra=config_extra, FAKE_SHIM_FAIL_ON=fail_on)
        alive = fake.alive()
    _assert_stopped_still_billing(result, sentence)
    assert _datasync(result) == ran, result.output
    assert alive == ["fake-instance-1"]
    assert not [c for c in result.shim_calls if c["command"].endswith(" recover")]
    _assert_no_secret_anywhere(result, tmp_path)


# (11)
@pytest.mark.parametrize("command", [["status", "--config", "box"], ["down", "--config", "box", "--yes"]],
                         ids=["status", "down"])
def test_down_and_status_do_not_need_the_key_file(tmp_path, command):
    """Catches: `down` refusing to terminate a billing instance (or `status` to report) because
    rclone.conf is missing on this machine."""
    with FakeLambda() as fake:
        inst = fake.add_instance()
        state = {"instance_id": inst, "host": "ubuntu@192.0.2.1", "ip": "192.0.2.1",
                 "persistent_dir": "/lambda/nfs/box-fs", "filesystem": "box-fs"}
        result = run_cloudgpu(fake, tmp_path, command, config_extra=SYNC, state=state,
                              shim_env=_shim_env(tmp_path))
        alive = fake.alive()
    assert result.returncode == 0, result.output
    assert "Traceback" not in result.output and "rclone.conf" not in result.output, result.output
    assert not (tmp_path / "home" / ".config" / "cloudgpu" / "rclone.conf").exists()
    assert alive == ([] if command[0] == "down" else [inst])
    assert not [c for c in result.shim_calls if " datasync " in c["command"]]


# --- the key file reader itself ---------------------------------------------------------------


def test_the_section_reader_keeps_keys_as_written_and_never_quotes_the_file(monkeypatch, tmp_path):
    """Catches: option names lower-cased or values interpolated on the way through; a section
    from [DEFAULT] or another remote leaking in; a parse error printing a line of the key file."""
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    assert profiles.rclone_config_file() == tmp_path / "rclone.conf"
    (tmp_path / "rclone.conf").write_text(
        f"[DEFAULT]\nshared = {OTHER_KEY}\n[b2]\ntype = b2\nKey = 100%{KEY}\n[b2x]\nkey = {OTHER_KEY}\n")
    assert profiles.rclone_remote_section("b2:bkt/small") == f"[b2]\ntype = b2\nKey = 100%{KEY}\n"
    for remote, text in (("b2:bkt", f"{KEY}\n[b2]\n"), ("gone:bkt", GOOD_CONF)):
        (tmp_path / "rclone.conf").write_text(text)
        with pytest.raises(config.ConfigError) as e:
            profiles.rclone_remote_section(remote)
        name = remote.split(":")[0]
        assert str(tmp_path / "rclone.conf") in str(e.value) and f"[{name}]" in str(e.value)
        assert not any(s in str(e.value) for s in SECRETS), str(e.value)
