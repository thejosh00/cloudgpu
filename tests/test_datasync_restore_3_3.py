"""The instance-side restore runner (3.3 part 1): what one restore copies, skips and records.

rclone is a fake executable in a temp directory: it records its arguments and
environment and copies from a temp "bucket" directory the way `rclone copy
--ignore-existing` would. Nothing here runs a real rclone or touches the network.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

SECRET = "K004-not-a-real-key"
REMOTE = "b2:bkt/small"

# The fake: "b2:bkt/small/<path>" is <tmp>/bucket/<path>. A missing source directory
# exits 3 (rclone's "Directory not found"). <bin>/exit.json maps a path to a forced exit
# code; <bin>/sleep holds seconds to sleep first.
FAKE_RCLONE = """#!{python}
import json, os, shutil, sys, time
here = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]
with open(os.path.join(here, "calls.jsonl"), "a") as f:
    f.write(json.dumps({{"argv": argv, "env": dict(os.environ)}}) + "\\n")
if os.path.exists(os.path.join(here, "sleep")):
    time.sleep(float(open(os.path.join(here, "sleep")).read()))
source, dest = argv[-2], argv[-1]
if ":" not in source.split("/")[0]:
    sys.exit(0)  # an upload: recorded only
rel = source[len("{remote}/"):]
forced = {{}}
if os.path.exists(os.path.join(here, "exit.json")):
    forced = json.load(open(os.path.join(here, "exit.json")))
if rel in forced:
    sys.exit(forced[rel])
src = os.path.join(os.path.dirname(here), "bucket", rel)
if not os.path.isdir(src):
    sys.exit(3)
for root, dirs, files in os.walk(src):
    out = os.path.join(dest, os.path.relpath(root, src))
    os.makedirs(out, exist_ok=True)
    for name in files:
        target = os.path.join(out, name)
        if "--ignore-existing" in argv and os.path.lexists(target):
            continue
        shutil.copyfile(os.path.join(root, name), target)
sys.exit(0)
"""


@pytest.fixture
def datasync():
    from cloudgpu.remote import datasync

    return datasync


class Box:
    """A temp persistent dir, a temp bucket, a fake rclone, a config holding a secret, a spec."""

    def __init__(self, tmp: Path) -> None:
        self.root = tmp / "fs"
        self.root.mkdir()
        self.bucket = tmp / "bucket"
        self.bucket.mkdir()
        self.bin = tmp / "bin"
        self.bin.mkdir()
        self.rclone = self.bin / "rclone"
        self.rclone.write_text(FAKE_RCLONE.format(python=sys.executable, remote=REMOTE))
        self.rclone.chmod(0o755)
        self.config = tmp / "rclone.conf"
        self.config.write_text(f"[b2]\ntype = b2\naccount = id\nkey = {SECRET}\n")
        self.last = tmp / "last-success"
        self.spec = tmp / "spec.json"
        self.state = self.root / "cloudgpu" / "state.json"

    def in_bucket(self, path: str, name: str = "a.png", text: str = "from the bucket") -> None:
        (self.bucket / path).mkdir(parents=True, exist_ok=True)
        (self.bucket / path / name).write_text(text)

    def write_spec(self, paths: list, **extra) -> None:
        spec = {
            "remote": REMOTE,
            "persistent_dir": str(self.root),
            "paths": paths,
            "rclone": str(self.rclone),
            "rclone_config": str(self.config),
            "last_success_file": str(self.last),
        }
        spec.update(extra)
        self.spec.write_text(json.dumps(spec))

    def force_exit(self, **codes: int) -> None:
        (self.bin / "exit.json").write_text(json.dumps(codes))

    def calls(self) -> list[dict]:
        log = self.bin / "calls.jsonl"
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines()]

    def record(self) -> dict | None:
        if not self.state.exists():
            return None
        return json.loads(self.state.read_text()).get("restore")


@pytest.fixture
def box(tmp_path):
    return Box(tmp_path)


def restore(datasync, box, capfd, **kw) -> tuple[int, str]:
    code = datasync.restore(str(box.spec), **kw)
    out = capfd.readouterr()
    return code, out.out + out.err


# --- the rclone calls --------------------------------------------------------------------


def test_exact_argv_one_copy_per_path_from_remote_to_persistent_dir(datasync, box, capfd):
    """Catches: source and destination swapped (an upload), a wrong layout, a missing
    skip-existing flag, or any unreviewed flag (the whole argv is pinned)."""
    box.in_bucket("apps/comfyui/output")
    box.in_bucket("datasets")
    box.write_spec(["apps/comfyui/output", "datasets"])
    code, _ = restore(datasync, box, capfd)
    assert code == 0
    assert [c["argv"] for c in box.calls()] == [
        ["copy", "--config", str(box.config), "--ignore-existing", "--exclude", "*.partial",
         f"{REMOTE}/apps/comfyui/output", f"{box.root}/apps/comfyui/output"],
        ["copy", "--config", str(box.config), "--ignore-existing", "--exclude", "*.partial",
         f"{REMOTE}/datasets", f"{box.root}/datasets"],
    ]
    assert (box.root / "datasets/a.png").read_text() == "from the bucket"


def test_no_sync_delete_link_or_inplace_flag(datasync, box, capfd):
    """Catches: `sync` or a delete flag removing local files, -L/--copy-links writing
    through links, or --inplace leaving a short file at the real name."""
    box.in_bucket("a")
    box.write_spec(["a"])
    restore(datasync, box, capfd)
    (call,) = box.calls()
    argv = call["argv"]
    assert argv[0] == "copy" and "sync" not in argv and "move" not in argv
    flags = [a for a in argv if a.startswith("-")]
    assert not [f for f in flags if "delete" in f or "link" in f or "inplace" in f]
    assert not [f for f in flags if re.fullmatch(r"-[A-Za-z]*[Ll][A-Za-z]*", f)]


def test_key_is_in_no_argument_environment_or_output(datasync, box, capfd, monkeypatch):
    """Catches: the runner opening the rclone config and leaking the key into argv (ps),
    the child's environment or the log; or RCLONE_* flags-by-environment passed on."""
    monkeypatch.setenv("RCLONE_COPY_LINKS", "true")
    box.in_bucket("a")
    box.write_spec(["a"])
    _, out = restore(datasync, box, capfd)
    (call,) = box.calls()
    assert SECRET not in out
    assert SECRET not in json.dumps(call)
    assert not [k for k in call["env"] if k.startswith("RCLONE_")]
    assert SECRET not in box.state.read_text()


# --- never replaces a local file -----------------------------------------------------------


def test_existing_local_file_is_left_byte_for_byte(datasync, box, capfd):
    """Catches: a restore overwriting newer local work with the bucket's older copy."""
    box.in_bucket("out", "a.png", "old bucket copy, longer than local")
    box.in_bucket("out", "b.png", "only in bucket")
    (box.root / "out").mkdir()
    (box.root / "out/a.png").write_bytes(b"local\x00work")
    box.write_spec(["out"])
    code, _ = restore(datasync, box, capfd)
    assert code == 0
    assert (box.root / "out/a.png").read_bytes() == b"local\x00work"
    assert (box.root / "out/b.png").read_text() == "only in bucket"


# --- empty versus unreachable --------------------------------------------------------------


def test_source_missing_from_bucket_is_nothing_to_restore_and_record_is_written(datasync, box, capfd):
    """Catches: the first `up` ever with [sync] (empty bucket, rclone exit 3) failing, or
    ending without the record so every later `up` restores again."""
    box.in_bucket("datasets")
    box.write_spec(["apps/comfyui/output", "datasets"])
    code, out = restore(datasync, box, capfd)
    assert code == 0
    assert len(box.calls()) == 2
    assert "1 restored, 1 with nothing in the bucket" in out
    record = box.record()
    assert record["remote"] == REMOTE
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", record["at"])


@pytest.mark.parametrize("code", [1, 2, 4, 5, 6, 7, 8, 9, 10])
def test_any_other_exit_code_fails_names_the_path_and_writes_no_record(datasync, box, capfd, code):
    """Catches: a wrong key (1), no network (5) or a fatal error (7) being read as "empty
    bucket", which would record a restore that never happened and lose it for good."""
    box.in_bucket("good")
    box.write_spec(["good", "apps/bad"])
    box.force_exit(**{"apps/bad": code})
    rc, out = restore(datasync, box, capfd)
    assert rc == 1
    assert "restore failed for: apps/bad" in out and f"rclone exited {code}" in out
    assert box.record() is None
    assert (box.root / "good/a.png").exists()  # what restored stays; no rollback


def test_rclone_that_cannot_be_run_fails_and_writes_no_record(datasync, box, capfd):
    """Catches: a missing rclone binary raising a traceback or being recorded as restored."""
    box.write_spec(["a"], rclone="/nowhere/rclone")
    rc, out = restore(datasync, box, capfd)
    assert rc == 1 and "restore failed for: a" in out
    assert box.record() is None


# --- once per filesystem -------------------------------------------------------------------


def test_second_run_makes_no_rclone_call_and_says_already_restored(datasync, box, capfd):
    """Catches: every `up` copying the bucket down again, bringing back files the person
    deleted on the instance."""
    box.in_bucket("a")
    box.write_spec(["a"])
    assert restore(datasync, box, capfd)[0] == 0
    at = box.record()["at"]
    (box.root / "a/a.png").unlink()
    code, out = restore(datasync, box, capfd)
    assert code == 0
    assert len(box.calls()) == 1
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) == 1
    assert "already restored" in lines[0] and REMOTE in lines[0] and at in lines[0]
    assert not (box.root / "a/a.png").exists()


def test_record_naming_another_remote_restores_again(datasync, box, capfd):
    """Catches: a filesystem restored from an old bucket never getting the new bucket's files."""
    box.state.parent.mkdir()
    box.state.write_text(json.dumps(
        {"apps": {"comfyui": {"v": 1}}, "restore": {"remote": "b2:other/x", "at": "2026-01-01T00:00:00Z"}}
    ))
    box.in_bucket("a")
    box.write_spec(["a"])
    code, _ = restore(datasync, box, capfd)
    assert code == 0 and len(box.calls()) == 1
    state = json.loads(box.state.read_text())
    assert state["restore"]["remote"] == REMOTE
    assert state["apps"] == {"comfyui": {"v": 1}}  # the app records are kept


def test_record_is_the_one_state_py_reads(datasync, box, capfd):
    """Catches: the standalone runner and remote/state.py disagreeing on the record's
    key or shape, so `up` would never see a finished restore."""
    from cloudgpu.remote.state import RESTORE_KEY, State

    assert datasync.RESTORE_KEY == RESTORE_KEY
    state = State(str(box.root))
    assert state.get_restore() is None and state.needs_restore(REMOTE)
    box.write_spec([])
    assert restore(datasync, box, capfd)[0] == 0
    assert state.get_restore()["remote"] == REMOTE
    assert not state.needs_restore(REMOTE) and state.needs_restore("b2:other/x")
    state.set_app("comfyui", {"v": 1})  # an app write keeps the record
    assert datasync.read_restore_record(str(box.root))["remote"] == REMOTE


def test_unreadable_state_json_fails_without_overwriting_it(datasync, box, capfd):
    """Catches: a damaged state.json being replaced by one holding only the restore
    record, losing the app records."""
    box.state.parent.mkdir()
    box.state.write_text("{not json")
    box.in_bucket("a")
    box.write_spec(["a"])
    rc, out = restore(datasync, box, capfd)
    assert rc == 1 and "state.json" in out
    assert box.calls() == []
    assert box.state.read_text() == "{not json"


# --- symlinks ------------------------------------------------------------------------------


def test_symlinked_path_and_path_under_a_symlink_are_skipped_named_and_not_written(
    datasync, box, capfd, tmp_path
):
    """Catches: a planted link turning the restore into a write somewhere outside the
    declared path."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (box.root / "apps/comfyui").mkdir(parents=True)
    (box.root / "apps/comfyui/output").symlink_to(outside)
    (box.root / "apps/linked").symlink_to(outside)
    for p in ("apps/comfyui/output", "apps/linked/datasets", "real"):
        box.in_bucket(p)
    box.write_spec(["apps/comfyui/output", "apps/linked/datasets", "real"])
    code, out = restore(datasync, box, capfd)
    assert "skipped apps/comfyui/output: apps/comfyui/output is a symlink" in out
    assert "skipped apps/linked/datasets: apps/linked is a symlink" in out
    assert [c["argv"][-2] for c in box.calls()] == [f"{REMOTE}/real"]
    assert list(outside.iterdir()) == []
    # Decision: a symlink skip still completes the restore (the upload skips them too).
    assert code == 0 and box.record()["remote"] == REMOTE
    assert "apps/comfyui/output" in out.splitlines()[-1]


def test_symlinked_cloudgpu_dir_gets_no_record_written_through_it(datasync, box, capfd, tmp_path):
    """Catches: the record being written through a planted `cloudgpu` link to a
    state.json somewhere else."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (box.root / "cloudgpu").symlink_to(outside)
    box.write_spec([])
    rc, out = restore(datasync, box, capfd)
    assert rc == 1 and "symlink" in out
    assert list(outside.iterdir()) == []


def test_refused_and_non_directory_paths_fail(datasync, box, capfd):
    """Catches: '../x' or an absolute path restoring outside the persistent dir, or a
    file where a directory should be being recorded as restored."""
    (box.root / "afile").write_text("x")
    box.write_spec(["../x", "/etc", "afile"])
    rc, out = restore(datasync, box, capfd)
    assert rc == 1 and box.calls() == [] and box.record() is None
    assert "restore failed for: ../x, /etc, afile" in out


# --- the time bound ------------------------------------------------------------------------


def test_run_is_bounded_and_a_timeout_fails_without_a_record(datasync, box, capfd):
    """Catches: a hung rclone blocking `up` for ever with the instance billing, or a
    timed-out restore being recorded as done."""
    box.in_bucket("a")
    box.in_bucket("b")
    (box.bin / "sleep").write_text("30")
    box.write_spec(["a", "b"])
    started = time.monotonic()
    rc, out = restore(datasync, box, capfd, timeout=1)
    assert time.monotonic() - started < 15
    assert rc == 1 and "restore failed for: a, b" in out and "out of time" in out
    assert len(box.calls()) == 1  # no call for b once the time is gone
    assert box.record() is None
    assert 0 < datasync.RESTORE_TIMEOUT <= 6 * 60 * 60


# --- partial downloads ---------------------------------------------------------------------


def test_upload_runner_excludes_partial_leftovers(datasync, box, capfd):
    """Catches: the timer uploading half a file that a killed restore left behind."""
    (box.root / "a").mkdir()
    (box.root / "a/model.safetensors.1a2b3c4d.partial").write_text("half")
    (box.root / "a/done.png").write_text("x")
    box.write_spec(["a"])
    assert datasync.run(str(box.spec)) == 0
    capfd.readouterr()
    (call,) = box.calls()
    argv = call["argv"]
    assert argv[0] == "copy" and argv[-1] == f"{REMOTE}/a"
    i = argv.index("--exclude")
    assert argv[i + 1] == "*.partial" and i + 1 < len(argv) - 2


def test_restore_removes_leftovers_of_a_killed_download_and_keeps_everything_else(
    datasync, box, capfd, tmp_path
):
    """Catches: half-downloaded files piling up on the filesystem as if they were real,
    or the cleanup deleting real files or reaching through a link."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.1a2b3c4d.partial").write_text("not ours")
    (box.root / "a/sub").mkdir(parents=True)
    (box.root / "a/sub/big.bin.0f9e8d7c.partial").write_text("half")
    (box.root / "a/notes.partial").write_text("a real file")
    (box.root / "a/link").symlink_to(outside)
    box.in_bucket("a/sub", "big.bin", "whole")
    box.write_spec(["a"])
    code, out = restore(datasync, box, capfd)
    assert code == 0
    assert not (box.root / "a/sub/big.bin.0f9e8d7c.partial").exists()
    assert "big.bin.0f9e8d7c.partial" in out
    assert (box.root / "a/sub/big.bin").read_text() == "whole"
    assert (box.root / "a/notes.partial").read_text() == "a real file"
    assert (outside / "x.1a2b3c4d.partial").read_text() == "not ours"


# --- the command line ----------------------------------------------------------------------


def test_the_file_restores_as_a_standalone_script(datasync, box, tmp_path):
    """Catches: the `restore --spec` command line part 2's unit will use not working from
    a root-owned copy outside the package."""
    copy = tmp_path / "etc" / "datasync.py"
    copy.parent.mkdir()
    copy.write_text(Path(datasync.__file__).read_text())
    box.in_bucket("a")
    box.write_spec(["a"], rclone_config="/nowhere/else.conf")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    done = subprocess.run(
        [sys.executable, "-I", str(copy), "restore", "--spec", str(box.spec),
         "--rclone-config", str(box.config)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert (box.root / "a/a.png").read_text() == "from the bucket"
    assert box.record()["remote"] == REMOTE
    assert SECRET not in done.stdout + done.stderr
