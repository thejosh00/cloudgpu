"""Item 3.2 part 2: the instance-side sync setup, start and removal.

Nothing here runs sudo, systemd, useradd, rclone or the network: sudo is replaced by a
recorder that acts on temp directories, and the download by a local zip.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile

import pytest

from cloudgpu.remote import __main__ as remote_main
from cloudgpu.remote import datasync_setup as ds

KEY_VALUE = "K004-very-secret-application-key"
KEY_TEXT = f"[b2]\ntype = b2\naccount = 004abc\nkey = {KEY_VALUE}\n"
PERSISTENT = "/lambda/nfs/fs"


def _zip(arch: str, body: bytes = b"#!fake rclone\n") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"rclone-v{ds.RCLONE_VERSION}-linux-{arch}/rclone", body)
        z.writestr(f"rclone-v{ds.RCLONE_VERSION}-linux-{arch}/README.txt", "x")
    return buf.getvalue()


class Box:
    """Temp stand-ins for the instance's root-owned places, and a record of every sudo call."""

    def __init__(self, tmp_path, monkeypatch):
        self.root = tmp_path / "root"
        self.home = tmp_path / "home"
        self.home.mkdir()
        self.calls: list[list[str]] = []
        self.envs: list[dict | None] = []
        self.fetched: list[str] = []
        self.fail_on: str | None = None
        self.users: set[str] = set()
        self.zip_bytes: bytes | None = None
        etc = self.root / "etc/cloudgpu"
        systemd = self.root / "etc/systemd/system"
        systemd.mkdir(parents=True)
        monkeypatch.setattr(ds, "ETC_DIR", str(etc))
        monkeypatch.setattr(ds, "KEY_PATH", str(etc / "rclone.conf"))
        monkeypatch.setattr(ds, "RUNNER_PATH", str(etc / "datasync.py"))
        monkeypatch.setattr(ds, "SPEC_PATH", str(etc / "datasync.json"))
        monkeypatch.setattr(ds, "SYSTEMD_DIR", str(systemd))
        monkeypatch.setattr(ds, "RCLONE_DIR", str(self.root / "usr/local/lib/cloudgpu"))
        monkeypatch.setattr(ds.subprocess, "run", self.run)
        monkeypatch.setattr(ds, "_fetch", self.fetch)
        monkeypatch.setattr(ds.platform, "machine", lambda: "aarch64")
        monkeypatch.setattr(ds.pwd, "getpwnam", self.getpwnam)
        from cloudgpu.remote import service
        monkeypatch.setattr(service, "ensure_app_user", lambda: self.calls.append(["<ensure_app_user>"]))
        self.use_zip(_zip("arm64"), "arm64", monkeypatch)

    def use_zip(self, data, arch, monkeypatch):
        self.zip_bytes = data
        monkeypatch.setitem(ds.RCLONE_SHA256, arch, hashlib.sha256(data).hexdigest())

    def getpwnam(self, name):
        if name not in self.users:
            raise KeyError(name)
        return object()

    def fetch(self, url, dest):
        self.fetched.append(url)
        with open(dest, "wb") as f:
            f.write(self.zip_bytes)

    def run(self, cmd, **kwargs):
        assert cmd[0] == "sudo", f"only sudo may be run, got {cmd}"
        cmd = list(cmd[1:])
        self.calls.append(cmd)
        self.envs.append(kwargs.get("env"))
        if self.fail_on and self.fail_on in cmd:
            if kwargs.get("check"):
                raise subprocess.CalledProcessError(1, ["sudo", *cmd])
            return subprocess.CompletedProcess(cmd, 1)
        if cmd[0] == "install" and "-d" in cmd:
            for d in cmd[cmd.index("755") + 1:]:
                os.makedirs(d, exist_ok=True)
        elif cmd[0] == "install":
            shutil.copyfile(cmd[-2], cmd[-1])
        elif cmd[0] == "mv":
            os.replace(cmd[-2], cmd[-1])
        elif cmd[0] == "rm":
            for p in cmd[2:]:
                if os.path.lexists(p):
                    os.remove(p)
        elif cmd[0] == "useradd":
            self.users.add(cmd[-1])
        return subprocess.CompletedProcess(cmd, 0)

    def transfer(self, request=None, key_text=KEY_TEXT):
        key = self.home / ".cloudgpu-rclone.conf"
        key.write_text(key_text)
        req = self.home / ".cloudgpu-sync.json"
        req.write_text(json.dumps(request if request is not None else {
            "remote": "b2:bkt/small", "paths": ["apps/comfyui/output", "apps/comfyui/user"],
            "interval_minutes": 5}))
        return str(key), str(req)

    def install_cmd(self, dest):
        return [c for c in self.calls if c[0] == "install" and c[-1] == dest]

    def systemctl(self):
        return [c for c in self.calls if c[0] == "systemctl"]


@pytest.fixture
def box(tmp_path, monkeypatch):
    return Box(tmp_path, monkeypatch)


# --- the rendered units --------------------------------------------------------------

def _settings(unit: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for line in unit.splitlines():
        if "=" in line and not line.startswith("["):
            k, v = line.split("=", 1)
            out.setdefault(k, []).append(v)
    return out


def test_service_runs_as_the_sync_user_not_root_or_app():
    s = _settings(ds.unit_texts(5)[0])
    assert s["User"] == [ds.SYNC_USER] and s["Group"] == [ds.SYNC_USER]
    assert ds.SYNC_USER not in ("root", "cloudgpu-app", "ubuntu")
    assert s["Type"] == ["oneshot"]  # a run still going at the next tick is not started twice
    assert s["NoNewPrivileges"] == ["yes"]


def test_service_cannot_write_the_persistent_dir():
    service, _ = ds.unit_texts(5)
    s = _settings(service)
    assert s["ProtectSystem"] == ["strict"]
    assert "ReadWritePaths" not in s and "ReadWritePaths" not in service
    assert "BindPaths" not in s
    # the only writable place is systemd's state directory, on the instance's own disk
    assert s["StateDirectory"] == [ds.STATE_DIR_NAME]
    assert ds.LAST_SUCCESS_FILE.startswith("/var/lib/")


def test_service_gets_the_key_as_a_credential_not_a_path_or_env():
    service, _ = ds.unit_texts(5)
    s = _settings(service)
    assert s["LoadCredential"] == [f"{ds.CREDENTIAL_NAME}:{ds.KEY_PATH}"]
    (exec_start,) = s["ExecStart"]
    assert f'--rclone-config "$${{CREDENTIALS_DIRECTORY}}/{ds.CREDENTIAL_NAME}"' in exec_start
    assert f"--rclone-config {ds.KEY_PATH}" not in exec_start  # it could not read that anyway
    assert f"{ds.RUNNER_PATH} run --spec {ds.SPEC_PATH}" in exec_start
    assert "EnvironmentFile" not in s
    assert "%d" not in service  # that specifier needs systemd 250; Ubuntu 22.04 has 249


def test_service_run_is_bounded():
    assert _settings(ds.unit_texts(5)[0])["TimeoutStartSec"] == [ds.RUN_TIMEOUT]
    assert ds.RUN_TIMEOUT not in ("", "0", "infinity")


@pytest.mark.parametrize("minutes", [5, 15])
def test_timer_fires_every_interval_minutes(minutes):
    t = _settings(ds.unit_texts(minutes)[1])
    assert t["OnUnitActiveSec"] == [f"{minutes}min"]
    assert t["WantedBy"] == ["timers.target"]


# --- the key -------------------------------------------------------------------------

def test_key_is_installed_root_only_under_etc_and_outside_the_persistent_dir(box):
    assert ds.main("setup", PERSISTENT, *box.transfer()) == 0
    (cmd,) = box.install_cmd(ds.KEY_PATH)
    assert cmd[:7] == ["install", "-o", "root", "-g", "root", "-m", "600"]
    assert open(ds.KEY_PATH).read() == KEY_TEXT


def test_real_key_and_program_locations_are_not_on_the_persistent_filesystem():
    import importlib
    fresh = importlib.reload(importlib.import_module("cloudgpu.remote.datasync_setup"))
    try:
        assert fresh.KEY_PATH == "/etc/cloudgpu/rclone.conf"
        for p in (fresh.KEY_PATH, fresh.RUNNER_PATH, fresh.SPEC_PATH, fresh.rclone_path(),
                  fresh.LAST_SUCCESS_FILE, fresh.service_path(), fresh.timer_path()):
            assert p.startswith(("/etc/", "/usr/local/lib/", "/var/lib/")), p
            assert not p.startswith(("/lambda", "/home"))
    finally:
        importlib.reload(fresh)


def test_transferred_copy_is_removed_when_setup_succeeds(box):
    key, req = box.transfer()
    assert ds.main("setup", PERSISTENT, key, req) == 0
    assert not os.path.exists(key) and not os.path.exists(req)


@pytest.mark.parametrize("fail_on", ["600", "useradd", "daemon-reload"])
def test_transferred_copy_is_removed_when_setup_fails(box, fail_on):
    box.fail_on = fail_on
    key, req = box.transfer()
    assert ds.main("setup", PERSISTENT, key, req) == 1
    assert not os.path.exists(key) and not os.path.exists(req)


def test_transferred_copy_is_removed_when_the_request_is_bad_or_missing(box):
    key, req = box.transfer(request={"remote": "nocolon", "paths": [], "interval_minutes": 5})
    assert ds.main("setup", PERSISTENT, key, req) == 1
    assert not os.path.exists(key)
    assert box.calls == []  # nothing was touched
    key, _ = box.transfer()
    assert ds.main("setup", PERSISTENT, key, None) == 1
    assert not os.path.exists(key)


@pytest.mark.parametrize("fail_on", [None, "600", "useradd", "enable"])
def test_key_value_is_in_no_command_env_or_output(box, capsys, fail_on):
    box.fail_on = fail_on
    ds.main("setup", PERSISTENT, *box.transfer(), start_timer=True)
    ds.main("remove", PERSISTENT)
    out = capsys.readouterr()
    assert KEY_VALUE not in out.out and KEY_VALUE not in out.err
    for cmd in box.calls:
        assert all(KEY_VALUE not in arg for arg in cmd), cmd
    assert all(env is None for env in box.envs)  # no call is handed a built environment
    for name in (ds.SPEC_PATH, ds.RUNNER_PATH, ds.service_path(), ds.timer_path()):
        if os.path.exists(name):
            assert KEY_VALUE not in open(name).read()


# --- rclone --------------------------------------------------------------------------

def test_real_pins_are_two_sha256_for_the_official_download():
    import importlib
    fresh = importlib.reload(importlib.import_module("cloudgpu.remote.datasync_setup"))
    assert set(fresh.RCLONE_SHA256) == {"amd64", "arm64"}
    for digest in fresh.RCLONE_SHA256.values():
        assert len(digest) == 64 and int(digest, 16) >= 0
    assert fresh.rclone_url("arm64") == (
        f"https://downloads.rclone.org/v{fresh.RCLONE_VERSION}/"
        f"rclone-v{fresh.RCLONE_VERSION}-linux-arm64.zip")


@pytest.mark.parametrize("machine,arch", [("x86_64", "amd64"), ("aarch64", "arm64")])
def test_rclone_is_fetched_for_the_instances_arch_and_installed_root_owned(
        box, monkeypatch, machine, arch):
    monkeypatch.setattr(ds.platform, "machine", lambda: machine)
    box.use_zip(_zip(arch, b"binary-" + arch.encode()), arch, monkeypatch)
    assert ds.ensure_rclone() is True
    assert box.fetched == [ds.rclone_url(arch)]
    assert f"linux-{arch}.zip" in box.fetched[0]
    assert open(ds.rclone_path(), "rb").read() == b"binary-" + arch.encode()
    (cmd,) = [c for c in box.calls if c[0] == "install" and "-d" not in c]
    assert cmd[:7] == ["install", "-o", "root", "-g", "root", "-m", "755"]
    assert ds.rclone_path().startswith(ds.RCLONE_DIR) and ds.RCLONE_VERSION in ds.rclone_path()


def test_rclone_with_the_wrong_checksum_is_refused(box, monkeypatch):
    monkeypatch.setitem(ds.RCLONE_SHA256, "arm64", "0" * 64)
    with pytest.raises(ds.SetupError, match="SHA-256"):
        ds.ensure_rclone()
    assert not os.path.exists(ds.rclone_path())
    assert box.calls == []  # nothing was installed, not even a directory


def test_unknown_machine_type_is_refused_before_any_download(box, monkeypatch):
    monkeypatch.setattr(ds.platform, "machine", lambda: "riscv64")
    with pytest.raises(ds.SetupError, match="riscv64"):
        ds.ensure_rclone()
    assert box.fetched == []


def test_rclone_is_skipped_when_the_pinned_version_is_installed(box):
    assert ds.ensure_rclone() is True
    box.calls.clear()
    assert ds.ensure_rclone() is False
    assert box.fetched == [ds.rclone_url("arm64")]  # only the first time
    assert box.calls == []


# --- the sync user -------------------------------------------------------------------

def test_sync_user_has_own_group_no_login_shell_reads_through_app_group_and_no_sudo(box):
    ds.ensure_sync_user()
    (add,) = [c for c in box.calls if c[0] == "useradd"]
    assert add[-1] == ds.SYNC_USER
    assert "--system" in add and "--user-group" in add and "--no-create-home" in add
    assert add[add.index("--shell") + 1] == "/usr/sbin/nologin"
    mods = [c for c in box.calls if c[0] == "usermod"]
    assert mods == [["usermod", "-aG", "cloudgpu-app", ds.SYNC_USER]]
    assert not any("sudo" in arg or "admin" in arg for c in box.calls for arg in c)
    assert box.calls[0] == ["<ensure_app_user>"]  # the group exists before it is joined


def test_sync_user_is_not_created_twice(box):
    ds.ensure_sync_user()
    ds.ensure_sync_user()
    assert len([c for c in box.calls if c[0] == "useradd"]) == 1


# --- setup, start, remove ------------------------------------------------------------

def test_setup_writes_runner_spec_and_units_and_does_not_start_the_timer(box):
    assert ds.main("setup", PERSISTENT, *box.transfer()) == 0
    spec = json.load(open(ds.SPEC_PATH))
    assert spec == {
        "remote": "b2:bkt/small", "persistent_dir": PERSISTENT,
        "paths": ["apps/comfyui/output", "apps/comfyui/user"], "interval_minutes": 5,
        "rclone": ds.rclone_path(), "last_success_file": ds.LAST_SUCCESS_FILE,
    }
    assert "rclone_config" not in spec  # only the credential path, given by the unit
    from cloudgpu.remote import datasync
    assert open(ds.RUNNER_PATH).read() == open(datasync.__file__).read()
    assert (open(ds.service_path()).read(), open(ds.timer_path()).read()) == ds.unit_texts(5)
    for dest in (ds.RUNNER_PATH, ds.SPEC_PATH, ds.service_path(), ds.timer_path()):
        (cmd,) = box.install_cmd(dest)
        assert cmd[:7] == ["install", "-o", "root", "-g", "root", "-m", "644"]
    assert box.systemctl() == [["systemctl", "daemon-reload"]]


def test_the_installed_spec_is_one_the_runner_accepts(box):
    from cloudgpu.remote import datasync
    assert ds.main("setup", PERSISTENT, *box.transfer()) == 0
    assert datasync.load_spec(ds.SPEC_PATH)["paths"] == ["apps/comfyui/output", "apps/comfyui/user"]


def test_interval_from_the_request_reaches_the_timer(box):
    req = {"remote": "b2:bkt/small", "paths": ["apps/x"], "interval_minutes": 15}
    assert ds.main("setup", PERSISTENT, *box.transfer(request=req)) == 0
    assert "OnUnitActiveSec=15min" in open(ds.timer_path()).read()


def test_start_enables_the_timer(box):
    assert ds.main("setup", PERSISTENT, *box.transfer()) == 0
    assert ds.main("start", PERSISTENT) == 0
    assert ["systemctl", "enable", "--now", f"{ds.UNIT_NAME}.timer"] in box.systemctl()


def test_setup_with_start_enables_the_timer_after_everything_is_in_place(box):
    assert ds.main("setup", PERSISTENT, *box.transfer(), start_timer=True) == 0
    enable = box.calls.index(["systemctl", "enable", "--now", f"{ds.UNIT_NAME}.timer"])
    assert box.calls.index(["systemctl", "daemon-reload"]) < enable
    assert all(c[0] != "install" for c in box.calls[enable:])


def test_start_without_setup_fails_with_a_sentence(box, capsys):
    assert ds.main("start", PERSISTENT) == 1
    assert "not set up" in capsys.readouterr().err
    assert box.calls == []


def test_second_setup_changes_nothing_and_leaves_one_timer(box):
    assert ds.main("setup", PERSISTENT, *box.transfer(), start_timer=True) == 0
    before = {p: open(p).read() for p in
              (ds.KEY_PATH, ds.SPEC_PATH, ds.RUNNER_PATH, ds.service_path(), ds.timer_path())}
    box.calls.clear()
    assert ds.main("setup", PERSISTENT, *box.transfer(), start_timer=True) == 0
    assert {p: open(p).read() for p in before} == before
    timers = [f for f in os.listdir(ds.SYSTEMD_DIR) if f.endswith(".timer")]
    assert timers == [f"{ds.UNIT_NAME}.timer"]
    # nothing but the key (which can't be compared without reading it) is rewritten
    installs = [c for c in box.calls if c[0] == "install" and "-d" not in c]
    assert [c[-1] for c in installs] == [ds.KEY_PATH]
    assert ["systemctl", "daemon-reload"] not in box.calls
    assert not any(c[0] == "useradd" for c in box.calls)
    assert len(box.fetched) == 1
    assert ["systemctl", "enable", "--now", f"{ds.UNIT_NAME}.timer"] in box.calls  # still running


def test_remove_takes_timer_units_key_spec_and_runner_away(box):
    assert ds.main("setup", PERSISTENT, *box.transfer(), start_timer=True) == 0
    box.calls.clear()
    assert ds.main("remove", PERSISTENT) == 0
    for p in (ds.KEY_PATH, ds.SPEC_PATH, ds.RUNNER_PATH, ds.service_path(), ds.timer_path()):
        assert not os.path.exists(p), p
    assert box.calls[0] == ["systemctl", "disable", "--now", f"{ds.UNIT_NAME}.timer"]
    assert ["systemctl", "stop", f"{ds.UNIT_NAME}.service"] in box.calls
    assert box.calls[-1] == ["systemctl", "daemon-reload"]


def test_remove_is_silent_when_nothing_was_there(box, capsys):
    assert ds.main("remove", PERSISTENT) == 0
    assert box.calls == []
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""


def test_remove_takes_a_leftover_key_away_even_without_units(box):
    os.makedirs(ds.ETC_DIR)
    open(ds.KEY_PATH, "w").write(KEY_TEXT)
    assert ds.main("remove", PERSISTENT) == 0
    assert not os.path.exists(ds.KEY_PATH)


@pytest.mark.parametrize("request_", [
    {"remote": "b2:bkt", "paths": ["/etc"], "interval_minutes": 5},
    {"remote": "b2:bkt", "paths": ["a/../../etc"], "interval_minutes": 5},
    {"remote": "b2:bkt", "paths": ["a"], "interval_minutes": 0},
    {"remote": "b2:bkt", "paths": ["a"], "interval_minutes": "5\nExecStartPre=/bin/evil"},
    {"remote": "b2:bkt", "paths": "a", "interval_minutes": 5},
    ["not", "an", "object"],
])
def test_a_bad_request_is_refused_before_anything_is_installed(box, capsys, request_):
    assert ds.main("setup", PERSISTENT, *box.transfer(request=request_)) == 1
    assert capsys.readouterr().err.startswith("error: ")
    assert box.calls == []


# --- the remote tool's command -------------------------------------------------------

def _remote(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["remote", "--persistent-dir", PERSISTENT, "datasync", *argv])
    remote_main.main()


def test_remote_command_sets_up_then_starts_then_removes(box, monkeypatch):
    key, req = box.transfer()
    _remote(monkeypatch, "setup", "--key-file", key, "--spec-file", req)
    assert json.load(open(ds.SPEC_PATH))["persistent_dir"] == PERSISTENT
    assert not any(c[:2] == ["systemctl", "enable"] for c in box.calls)
    _remote(monkeypatch, "start")
    assert ["systemctl", "enable", "--now", f"{ds.UNIT_NAME}.timer"] in box.calls
    _remote(monkeypatch, "remove")
    assert not os.path.exists(ds.KEY_PATH) and not os.path.exists(ds.timer_path())


def test_remote_command_failure_is_nonzero_with_a_sentence_not_a_traceback(box, monkeypatch, capsys):
    box.fail_on = "useradd"
    key, req = box.transfer()
    with pytest.raises(SystemExit) as e:
        _remote(monkeypatch, "setup", "--key-file", key, "--spec-file", req, "--start")
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert "error: Sync setup failed" in err and "useradd" in err
    assert "Traceback" not in err and KEY_VALUE not in err
    assert not os.path.exists(key)
    assert not any(c[:2] == ["systemctl", "enable"] for c in box.calls)  # never started half-built
