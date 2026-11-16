"""Item 3.3 part 2: the restricted restore run, ownership, and the timer around it.

Nothing here runs sudo, systemd, systemd-run or rclone: sudo is the recorder from the
3.2 setup tests, and the restricted run is a function that writes files into a temp tree.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess

import pytest

from cloudgpu.remote import __main__ as remote_main
from cloudgpu.remote import datasync
from cloudgpu.remote import datasync_setup as ds
from tests.test_datasync_setup_3_2 import KEY_VALUE, Box

REMOTE = "b2:bkt/small"
PATHS = ["apps/comfyui/output", "apps/comfyui/user"]


class RestoreBox(Box):
    """The 3.2 recorder, plus a timer that is running or not and a made-up restricted run."""

    def __init__(self, tmp_path, monkeypatch):
        super().__init__(tmp_path, monkeypatch)
        self.fs = tmp_path / "fs"
        self.fs.mkdir()
        self.timer_active = False
        self.run_code = 0
        self.during_run: list[str] = []
        monkeypatch.setattr(ds, "_app_gid", lambda: os.getgid())
        os.makedirs(ds.ETC_DIR)
        self.write_spec()

    def write_spec(self, remote=REMOTE, paths=PATHS):
        with open(ds.SPEC_PATH, "w") as f:
            json.dump({"remote": remote, "persistent_dir": str(self.fs), "paths": paths,
                       "interval_minutes": 5, "rclone": "/x/rclone",
                       "last_success_file": "/x/last"}, f)

    def run(self, cmd, **kwargs):
        rest = list(cmd[1:])
        if rest[:2] == ["systemctl", "is-active"]:
            assert cmd[0] == "sudo"
            self.calls.append(rest)
            return subprocess.CompletedProcess(cmd, 0 if self.timer_active else 3)
        if rest[0] == "systemd-run":
            assert cmd[0] == "sudo"
            self.calls.append(rest)
            self.envs.append(kwargs.get("env"))
            # what the timer was doing while files arrived
            self.during_run.append("active" if self.timer_active else "stopped")
            for path in self.writable(rest):
                with open(os.path.join(path, "restored.png"), "w") as f:
                    f.write("from the bucket")
            return subprocess.CompletedProcess(cmd, self.run_code)
        if rest[0] == "systemctl" and rest[-1] == f"{ds.UNIT_NAME}.timer":
            if rest[1] == "stop":
                self.timer_active = False
            elif rest[1] == "start":
                self.timer_active = True
        return super().run(cmd, **kwargs)

    @staticmethod
    def props(cmd):
        return [cmd[i + 1] for i, a in enumerate(cmd) if a == "-p"]

    def writable(self, cmd):
        return [p.split("=", 1)[1] for p in self.props(cmd) if p.startswith("ReadWritePaths=")]

    def restore_runs(self):
        return [c for c in self.calls if c[0] == "systemd-run"]

    def index(self, *head):
        return next(i for i, c in enumerate(self.calls) if c[:len(head)] == list(head))


@pytest.fixture
def box(tmp_path, monkeypatch):
    return RestoreBox(tmp_path, monkeypatch)


# --- who runs it and what it can touch -------------------------------------------------

def test_restore_runs_once_as_the_sync_user_restricted_like_the_upload(box):
    ds.restore(str(box.fs))
    (cmd,) = box.restore_runs()
    props = box.props(cmd)
    assert f"User={ds.SYNC_USER}" in props and f"Group={ds.SYNC_USER}" in props
    assert ds.SYNC_USER not in ("root", "cloudgpu-app", "ubuntu")
    assert f"SupplementaryGroups={ds.APP_USER}" in props
    for needed in ("NoNewPrivileges=yes", "ProtectSystem=strict", "ProtectHome=yes",
                   "PrivateTmp=yes", "UMask=0002", f"RuntimeMaxSec={ds.RUN_TIMEOUT}"):
        assert needed in props
    assert "--wait" in cmd  # the caller learns whether it worked


def test_restore_can_write_the_declared_paths_and_nothing_else(box):
    ds.restore(str(box.fs))
    (cmd,) = box.restore_runs()
    assert box.writable(cmd) == [str(box.fs / p) for p in PATHS]
    joined = " ".join(cmd)
    assert "BindPaths" not in joined
    # not the persistent dir itself, not the state directory of records
    assert str(box.fs) not in box.writable(cmd)
    assert str(box.fs / "cloudgpu") not in box.writable(cmd)


def test_the_key_reaches_the_run_as_a_credential_and_is_in_no_command_or_env(box):
    with open(ds.KEY_PATH, "w") as f:
        f.write(f"[b2]\nkey = {KEY_VALUE}\n")
    ds.restore(str(box.fs))
    (cmd,) = box.restore_runs()
    assert f"LoadCredential={ds.CREDENTIAL_NAME}:{ds.KEY_PATH}" in box.props(cmd)
    assert cmd[-3:-1] == ["/bin/sh", "-c"]
    assert f'--rclone-config "${{CREDENTIALS_DIRECTORY}}/{ds.CREDENTIAL_NAME}"' in cmd[-1]
    assert f"{ds.RUNNER_PATH} restore --no-record --spec {ds.SPEC_PATH}" in cmd[-1]
    assert KEY_VALUE not in json.dumps(box.calls)
    assert all(env is None for env in box.envs)


def test_the_upload_service_still_cannot_write_the_persistent_dir():
    service, _ = ds.unit_texts(5)
    assert "ReadWritePaths" not in service and "BindPaths" not in service


# --- the directories it restores into --------------------------------------------------

def test_missing_declared_paths_are_made_group_writable_setgid_and_parents_plain(box):
    ds.restore(str(box.fs))
    for p in PATHS:
        mode = os.stat(box.fs / p).st_mode
        assert stat.S_IMODE(mode) == 0o2775, p
    parent = os.stat(box.fs / "apps/comfyui").st_mode
    assert not parent & stat.S_ISGID and not parent & stat.S_IWGRP
    # made by this user, not by root: no sudo mkdir or install -d under the persistent dir
    assert not [c for c in box.calls if c[0] in ("mkdir", "install") and str(box.fs) in " ".join(c)]


def test_a_symlinked_path_is_not_writable_not_created_through_and_is_said(box, tmp_path, capsys):
    outside = tmp_path / "outside"
    outside.mkdir()
    (box.fs / "apps").mkdir()
    os.symlink(outside, box.fs / "apps/comfyui")
    ds.restore(str(box.fs))
    (cmd,) = box.restore_runs()
    assert box.writable(cmd) == []
    assert list(outside.iterdir()) == []
    assert "apps/comfyui is a symlink" in capsys.readouterr().err
    assert not [c for c in box.calls if c[0] in ("chgrp", "chmod", "find")]


def test_a_path_with_a_space_is_refused_rather_than_widening_the_write_access(box):
    box.write_spec(paths=["apps/my output"])
    with pytest.raises(ds.SetupError, match="space"):
        ds.restore(str(box.fs))
    assert box.restore_runs() == []


# --- group and modes, by the commands recorded -----------------------------------------

def test_restored_paths_get_the_apps_group_and_modes_after_the_run(box):
    ds.restore(str(box.fs))
    run_at = box.index("systemd-run")
    for p in PATHS:
        full = str(box.fs / p)
        expected = [["chgrp", "-hR", ds.APP_USER, full],
                    ["chmod", "-R", "g+rwX", full],
                    ["find", full, "-type", "d", "-exec", "chmod", "g+s", "{}", "+"]]
        at = box.calls.index(expected[0])
        assert box.calls[at:at + 3] == expected
        assert at > run_at
    # only the declared paths: nothing is run over the app dir or the persistent dir
    targets = {c[-1] if c[0] != "find" else c[1] for c in box.calls if c[0] in ("chgrp", "chmod", "find")}
    assert targets == {str(box.fs / p) for p in PATHS}


def test_paths_that_restored_are_shared_even_when_the_restore_failed(box):
    box.run_code = 1
    with pytest.raises(ds.SetupError, match="failed"):
        ds.restore(str(box.fs))
    assert ["chgrp", "-hR", ds.APP_USER, str(box.fs / PATHS[0])] in box.calls


# --- the record -------------------------------------------------------------------------

def test_the_record_is_written_by_the_caller_after_a_good_run_only(box):
    ds.restore(str(box.fs))
    record = datasync.read_restore_record(str(box.fs))
    assert record["remote"] == REMOTE and record["at"].endswith("Z")
    # by this process: no sudo command touches state.json
    assert "state.json" not in json.dumps(box.calls)


def test_a_failed_run_writes_no_record_and_says_so_in_a_sentence(box, capsys):
    box.run_code = 1
    code = ds.main("restore", str(box.fs))
    assert code == 1
    assert datasync.read_restore_record(str(box.fs)) is None
    err = capsys.readouterr().err
    assert f"error: The restore from {REMOTE} failed" in err and "Traceback" not in err


def test_an_already_restored_filesystem_runs_nothing_and_says_when(box, capsys):
    ds.restore(str(box.fs))
    box.calls.clear()
    box.timer_active = True
    ds.restore(str(box.fs))
    assert box.calls == []  # no run, and the timer is not touched
    assert "already restored" in capsys.readouterr().err


def test_a_record_for_another_remote_restores_again(box):
    ds.restore(str(box.fs))
    box.write_spec(remote="b2:other/place")
    ds.restore(str(box.fs))
    assert len(box.restore_runs()) == 2
    assert datasync.read_restore_record(str(box.fs))["remote"] == "b2:other/place"


def test_restore_without_setup_fails_in_a_sentence(box, capsys):
    os.remove(ds.SPEC_PATH)
    assert ds.main("restore", str(box.fs)) == 1
    assert "not set up" in capsys.readouterr().err
    assert box.restore_runs() == []


# --- the timer ---------------------------------------------------------------------------

def test_a_running_timer_and_run_are_stopped_before_and_the_timer_started_after(box):
    box.timer_active = True
    ds.restore(str(box.fs))
    timer, service = f"{ds.UNIT_NAME}.timer", f"{ds.UNIT_NAME}.service"
    stop_timer = box.index("systemctl", "stop", timer)
    stop_run = box.index("systemctl", "stop", service)
    run_at = box.index("systemd-run")
    start = box.index("systemctl", "start", timer)
    assert stop_timer < stop_run < run_at < start
    assert box.during_run == ["stopped"]
    assert start > box.calls.index(["chgrp", "-hR", ds.APP_USER, str(box.fs / PATHS[-1])])
    assert box.timer_active


def test_a_timer_that_was_not_running_is_not_started(box):
    ds.restore(str(box.fs))
    assert not [c for c in box.systemctl() if c[1] in ("start", "enable", "try-restart", "restart")]
    assert not box.timer_active


def test_a_running_timer_is_started_again_after_a_failed_restore(box):
    box.timer_active = True
    box.run_code = 1
    with pytest.raises(ds.SetupError):
        ds.restore(str(box.fs))
    assert box.during_run == ["stopped"] and box.timer_active


# --- the runner inside the restricted run, and the entry point ---------------------------

def test_no_record_runner_neither_reads_nor_writes_state_json(tmp_path, monkeypatch):
    fs = tmp_path / "fs"
    (fs / "cloudgpu").mkdir(parents=True)
    (fs / "cloudgpu/state.json").write_text("{ not json")  # unreadable to the run: never opened
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"remote": REMOTE, "persistent_dir": str(fs),
                                "paths": ["apps/comfyui/output"], "interval_minutes": 5}))
    seen = []
    monkeypatch.setattr(datasync.subprocess, "run",
                        lambda cmd, **kw: seen.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    code = datasync.main(["restore", "--no-record", "--spec", str(spec),
                          "--rclone", "/x/rclone", "--rclone-config", "/x/conf"])
    assert code == 0 and [c[1] for c in seen] == ["lsf", "copy"]  # part 3: the reachable check first
    assert (fs / "cloudgpu/state.json").read_text() == "{ not json"
    assert [p.name for p in (fs / "cloudgpu").iterdir()] == ["state.json"]


def test_remote_tool_has_a_datasync_restore_command(box, monkeypatch):
    monkeypatch.setattr("sys.argv", ["cloudgpu-remote", "--persistent-dir", str(box.fs),
                                     "datasync", "restore"])
    remote_main.main()
    assert len(box.restore_runs()) == 1
    assert datasync.read_restore_record(str(box.fs))["remote"] == REMOTE
