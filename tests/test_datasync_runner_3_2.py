"""The instance-side sync runner (3.2 part 1): what one run copies, skips and records.

rclone is a fake executable in a temp directory that records its arguments and
environment; nothing here runs a real rclone or touches the network.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

SECRET = "K004-not-a-real-key"

FAKE_RCLONE = """#!{python}
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "calls.jsonl"), "a") as f:
    f.write(json.dumps({{"argv": sys.argv[1:], "env": dict(os.environ)}}) + "\\n")
source = sys.argv[-2]
sys.exit(3 if os.path.exists(os.path.join(source, "FAIL")) else 0)
"""


@pytest.fixture
def datasync():
    from cloudgpu.remote import datasync

    return datasync


class Box:
    """A temp persistent dir, a fake rclone, a config holding a secret, and a spec."""

    def __init__(self, tmp: Path) -> None:
        self.root = tmp / "fs"
        self.root.mkdir()
        self.bin = tmp / "bin"
        self.bin.mkdir()
        self.rclone = self.bin / "rclone"
        self.rclone.write_text(FAKE_RCLONE.format(python=sys.executable))
        self.rclone.chmod(0o755)
        self.config = tmp / "rclone.conf"
        self.config.write_text(f"[b2]\ntype = b2\naccount = id\nkey = {SECRET}\n")
        self.last = tmp / "last-success"
        self.spec = tmp / "spec.json"

    def data(self, *paths: str, files: bool = True) -> None:
        for p in paths:
            (self.root / p).mkdir(parents=True, exist_ok=True)
            if files:
                (self.root / p / "a.png").write_text("x")

    def write_spec(self, paths: list, **extra) -> None:
        spec = {
            "remote": "b2:bkt/small",
            "persistent_dir": str(self.root),
            "paths": paths,
            "rclone": str(self.rclone),
            "rclone_config": str(self.config),
            "last_success_file": str(self.last),
        }
        spec.update(extra)
        self.spec.write_text(json.dumps(spec))

    def calls(self) -> list[dict]:
        log = self.bin / "calls.jsonl"
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines()]

    def pairs(self) -> list[tuple[str, str]]:
        return [(c["argv"][-2], c["argv"][-1]) for c in self.calls()]


@pytest.fixture
def box(tmp_path):
    return Box(tmp_path)


def run(datasync, box, capfd) -> tuple[int, str]:
    code = datasync.run(str(box.spec))
    out = capfd.readouterr()
    return code, out.out + out.err


# --- acceptance (6): the rclone calls ---------------------------------------------------


def test_one_copy_per_path_with_point_5_source_and_destination(datasync, box, capfd):
    # Catches: a wrong destination layout (files landing at the remote's root or under
    # the wrong prefix), or paths merged into one call or dropped.
    box.data("apps/comfyui/output", "apps/comfyui/user", "datasets")
    box.write_spec(["apps/comfyui/output", "apps/comfyui/user", "datasets"])
    code, _ = run(datasync, box, capfd)
    assert code == 0
    assert box.pairs() == [
        (f"{box.root}/apps/comfyui/output", "b2:bkt/small/apps/comfyui/output"),
        (f"{box.root}/apps/comfyui/user", "b2:bkt/small/apps/comfyui/user"),
        (f"{box.root}/datasets", "b2:bkt/small/datasets"),
    ]


def test_calls_are_copy_never_sync_and_delete_nothing(datasync, box, capfd):
    # Catches: `rclone sync` or a deleting flag, which would remove bucket files when
    # they are deleted on the instance.
    box.data("a", "b")
    box.write_spec(["a", "b"])
    run(datasync, box, capfd)
    calls = box.calls()
    assert len(calls) == 2
    for call in calls:
        argv = call["argv"]
        assert argv[0] == "copy"
        assert "sync" not in argv and "move" not in argv
        assert not [a for a in argv if "delete" in a or a.startswith("--backup-dir")]


def test_no_link_flag_in_any_call(datasync, box, capfd):
    # Catches: -L/--copy-links (uploads whatever a planted link points at) or
    # -l/--links (uploads link files).
    box.data("a")
    box.write_spec(["a"])
    run(datasync, box, capfd)
    (call,) = box.calls()
    flags = [a for a in call["argv"] if a.startswith("-")]
    assert not [f for f in flags if "link" in f]
    assert not [f for f in flags if re.fullmatch(r"-[A-Za-z]*[Ll][A-Za-z]*", f)]


def test_only_known_flags_are_passed(datasync, box, capfd):
    # Catches: any new flag slipping in unreviewed (the whole flag list is pinned here).
    box.data("a")
    box.write_spec(["a"])
    run(datasync, box, capfd)
    (call,) = box.calls()
    assert call["argv"] == [
        "copy", "--config", str(box.config), "--fast-list", f"{box.root}/a", "b2:bkt/small/a",
    ]


def test_key_is_not_in_arguments_environment_or_output(datasync, box, capfd):
    # Catches: the runner reading the rclone config and leaking the key into a command
    # line (visible in ps), the child's environment, or the log.
    box.data("a")
    box.write_spec(["a"])
    _, out = run(datasync, box, capfd)
    (call,) = box.calls()
    assert SECRET not in out
    assert SECRET not in json.dumps(call)
    assert call["argv"][1:3] == ["--config", str(box.config)]


def test_rclone_environment_variables_are_not_passed_on(datasync, box, capfd, monkeypatch):
    # Catches: RCLONE_COPY_LINKS=true (or any RCLONE_* flag-by-environment) reaching
    # rclone and turning link following or deletion back on.
    monkeypatch.setenv("RCLONE_COPY_LINKS", "true")
    box.data("a")
    box.write_spec(["a"])
    run(datasync, box, capfd)
    (call,) = box.calls()
    assert not [k for k in call["env"] if k.startswith("RCLONE_")]


def test_arguments_override_the_spec(datasync, box, capfd, tmp_path):
    # Catches: part 2 being unable to pass the config path (a systemd credential path)
    # and the last-success path on the command line.
    box.data("a")
    box.write_spec(["a"], rclone_config="/nowhere/else.conf", last_success_file="/nowhere/last")
    other_last = tmp_path / "other-last"
    code = datasync.main([
        "run", "--spec", str(box.spec), "--rclone-config", str(box.config),
        "--last-success-file", str(other_last),
    ])
    capfd.readouterr()
    assert code == 0
    assert box.calls()[0]["argv"][2] == str(box.config)
    assert other_last.exists()


# --- acceptance (7): skips, failures, the success time -----------------------------------


def test_missing_path_is_skipped_without_failing(datasync, box, capfd):
    # Catches: a fresh install (no `user/` yet) making every run fail, or rclone being
    # run on a source that isn't there.
    box.data("apps/comfyui/output")
    box.write_spec(["apps/comfyui/output", "apps/comfyui/user"])
    code, out = run(datasync, box, capfd)
    assert code == 0
    assert [p[0] for p in box.pairs()] == [f"{box.root}/apps/comfyui/output"]
    assert "apps/comfyui/user" not in out
    assert box.last.exists()


def test_symlinked_path_is_skipped_named_and_not_copied(datasync, box, capfd, tmp_path):
    # Catches: following a declared path that app code replaced with a link to
    # somewhere else the sync user can read.
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("x")
    box.data("apps/comfyui")
    (box.root / "apps/comfyui/output").symlink_to(outside)
    box.data("datasets")
    box.write_spec(["apps/comfyui/output", "datasets"])
    code, out = run(datasync, box, capfd)
    assert "apps/comfyui/output" in out and "symlink" in out
    assert box.pairs() == [(f"{box.root}/datasets", "b2:bkt/small/datasets")]
    assert code == 0


def test_path_under_a_symlinked_directory_is_skipped_and_named(datasync, box, capfd, tmp_path):
    # Catches: checking only the last component, so a link higher up
    # (apps/comfyui -> elsewhere) is followed.
    outside = tmp_path / "outside"
    (outside / "output").mkdir(parents=True)
    (outside / "output" / "secret.txt").write_text("x")
    box.data("apps")
    (box.root / "apps/comfyui").symlink_to(outside)
    box.write_spec(["apps/comfyui/output"])
    code, out = run(datasync, box, capfd)
    assert box.calls() == []
    assert "skipped apps/comfyui/output" in out and "apps/comfyui is a symlink" in out
    assert code == 0


def test_symlink_skip_does_not_record_success(datasync, box, capfd, tmp_path):
    # Catches: a run that left declared data out being reported as a complete sync
    # (the last-sync time would look fresh while a path is not backed up).
    box.data("a")
    (box.root / "b").symlink_to(tmp_path)
    box.write_spec(["a", "b"])
    code, _ = run(datasync, box, capfd)
    assert code == 0
    assert not box.last.exists()


def test_one_failure_still_runs_the_rest_and_exits_non_zero_naming_it(datasync, box, capfd):
    # Catches: stopping at the first failed path (later paths never backed up), or a
    # failed run exiting 0 or not saying which path failed.
    box.data("a", "b", "c")
    (box.root / "b" / "FAIL").write_text("")
    box.write_spec(["a", "b", "c"])
    code, out = run(datasync, box, capfd)
    assert code == 1
    assert [p[1] for p in box.pairs()] == ["b2:bkt/small/a", "b2:bkt/small/b", "b2:bkt/small/c"]
    summary = [line for line in out.splitlines() if "sync failed for" in line]
    assert len(summary) == 1 and summary[0].split("sync failed for: ")[1].startswith("b ")


def test_success_time_written_when_everything_copied(datasync, box, capfd):
    # Catches: no record of a good run, or a time that isn't UTC ISO-8601 for `status`.
    box.data("a", "b")
    box.write_spec(["a", "b"])
    code, _ = run(datasync, box, capfd)
    assert code == 0
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ\n", box.last.read_text())
    assert [p.name for p in box.last.parent.iterdir() if p.name.startswith(".last-success")] == []


def test_success_time_untouched_by_a_failed_run(datasync, box, capfd):
    # Catches: a failed run refreshing (or erasing) the last good time.
    box.data("a", "b")
    (box.root / "b" / "FAIL").write_text("")
    box.last.write_text("2026-01-01T00:00:00Z\n")
    box.write_spec(["a", "b"])
    code, _ = run(datasync, box, capfd)
    assert code == 1
    assert box.last.read_text() == "2026-01-01T00:00:00Z\n"


def test_rclone_that_cannot_be_run_is_a_named_failure(datasync, box, capfd):
    # Catches: a missing rclone binary raising a traceback, or passing as success.
    box.data("a")
    box.write_spec(["a"], rclone=str(box.bin / "absent"))
    code, out = run(datasync, box, capfd)
    assert code == 1
    assert "failed a" in out and "Traceback" not in out
    assert not box.last.exists()


@pytest.mark.parametrize("bad", ["/etc", "../outside", "apps/../../etc", "a/..", ""])
def test_absolute_and_dotdot_paths_are_refused(datasync, box, capfd, bad):
    # Catches: a path that leaves the persistent dir being handed to rclone.
    box.data("ok")
    box.write_spec([bad, "ok"])
    code, out = run(datasync, box, capfd)
    assert code == 1
    assert box.pairs() == [(f"{box.root}/ok", "b2:bkt/small/ok")]
    assert f"refused {bad!r}" in out
    assert not box.last.exists()


def test_path_that_is_a_file_is_a_failure(datasync, box, capfd):
    # Catches: `rclone copy` of a file source, which would land it at <path>/<name>
    # and not at the destination of point 5.
    (box.root / "notes.txt").write_text("x")
    box.write_spec(["notes.txt"])
    code, out = run(datasync, box, capfd)
    assert code == 1 and box.calls() == []
    assert "notes.txt is not a directory" in out


def test_directory_with_no_files_makes_no_rclone_call(datasync, box, capfd, tmp_path):
    # Catches: spending bucket listing calls every run on paths with nothing to upload.
    box.data("empty/sub", files=False)
    (box.root / "empty" / "link").symlink_to(tmp_path)
    box.write_spec(["empty"])
    code, _ = run(datasync, box, capfd)
    assert code == 0 and box.calls() == []
    assert box.last.exists()


def test_bad_spec_fails_with_a_sentence(datasync, box, capfd):
    # Catches: a missing or broken spec file ending in a traceback or exit 0.
    box.spec.write_text("{not json")
    code, out = run(datasync, box, capfd)
    assert code == 1 and "no valid sync spec" in out and "Traceback" not in out


# --- standalone ---------------------------------------------------------------------------


def _source() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "cloudgpu" / "remote" / "datasync.py"


def test_runner_imports_only_the_standard_library():
    # Catches: an import from cloudgpu (or a third-party package), which would break the
    # root-owned copy that runs outside the package.
    tree = ast.parse(_source().read_text())
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import"
            modules.add((node.module or "").split(".")[0])
    assert modules - {"__future__"} <= set(sys.stdlib_module_names)
    assert "cloudgpu" not in modules


def test_runs_as_a_script_from_a_copy_outside_the_package(box, tmp_path):
    # Catches: the copied file depending on its package (sys.path, __package__) or the
    # `run --spec` command line part 2's service will use not working.
    copy = tmp_path / "etc" / "datasync.py"
    copy.parent.mkdir()
    copy.write_text(_source().read_text())
    box.data("a")
    box.write_spec(["a"])
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    done = subprocess.run(
        [sys.executable, "-I", str(copy), "run", "--spec", str(box.spec)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert box.pairs() == [(f"{box.root}/a", "b2:bkt/small/a")]
    assert box.last.exists()
