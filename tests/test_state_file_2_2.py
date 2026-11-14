"""The local state file is written atomically, and a corrupt one stops every command (plan item
2.2, part B).

No real API, instance or key: everything runs in-process, with the Lambda client's one request
function and every subprocess (ssh, rsync) replaced by something that records the call and fails.
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest
from click.testing import CliRunner

from cloudgpu import cli
from cloudgpu.local import config, configs, lambda_api

OLD = {"instance_id": "i-old", "host": "ubuntu@1.2.3.4", "persistent_dir": "/lambda/nfs/box-fs",
       "provisioned_profiles": ["a"]}
NEW = {"instance_id": "i-new", "host": "ubuntu@5.6.7.8", "persistent_dir": "/lambda/nfs/box-fs",
       "provisioned_profiles": ["a", "b"]}

TRUNCATED = b'{\n  "instance_id": "i-old",\n  "host": "ubu'
CORRUPT = {"empty": b"", "truncated": TRUNCATED, "a list": b"[]\n", "not utf-8": b"\xff\xfe{}"}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A directory with configs/box.toml and a profile `a`, as the current directory."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "box.toml").write_text('gpu = ["gh200"]\nssh_key = "mini"\nfilesystem = "box-fs"\n')
    profile = tmp_path / "profiles" / "a"
    profile.mkdir(parents=True)
    (profile / "cloudgpu.toml").write_text("apps = []\n")
    (profile / "provision.py").write_text("print('provisioned')\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _state_file(repo):
    return repo / "configs" / "box.state.json"


def _others(repo):
    """Every file in configs/ but the config and its state file: a stray temporary file shows here."""
    return sorted(p.name for p in (repo / "configs").iterdir() if p.name not in ("box.toml", "box.state.json"))


# --- (5) save_state is atomic ---------------------------------------------------------------------


class TestSaveState:
    def test_a_save_writes_complete_json_and_leaves_nothing_else(self, repo):
        """Catches: a save that leaves its temporary file behind, or writes something `load_state` can't read."""
        configs.save_state("box", OLD)
        configs.save_state("box", NEW)
        assert json.loads(_state_file(repo).read_text()) == NEW
        assert configs.load_state("box") == NEW
        assert _others(repo) == []

    def test_a_write_that_fails_partway_keeps_the_old_file(self, repo, monkeypatch):
        """Catches: a full disk mid-write leaving a truncated state file (the tracked id lost)."""
        configs.save_state("box", OLD)
        before = _state_file(repo).read_bytes()
        real_fdopen = os.fdopen

        class HalfThenFull:
            def __init__(self, handle):
                self.handle = handle

            def write(self, text):
                self.handle.write(text[: len(text) // 2])
                self.handle.flush()
                raise OSError(28, "No space left on device")

            def __getattr__(self, name):
                return getattr(self.handle, name)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.handle.close()

        monkeypatch.setattr(configs.os, "fdopen", lambda *a, **k: HalfThenFull(real_fdopen(*a, **k)))
        with pytest.raises(OSError, match="No space left"):
            configs.save_state("box", NEW)
        assert _state_file(repo).read_bytes() == before
        assert _others(repo) == []

    def test_a_failed_fsync_keeps_the_old_file(self, repo, monkeypatch):
        """Catches: a save that replaces the old file with one the disk never confirmed it holds."""
        configs.save_state("box", OLD)
        before = _state_file(repo).read_bytes()

        def no_fsync(fd):
            raise OSError(5, "Input/output error")

        monkeypatch.setattr(configs.os, "fsync", no_fsync)
        with pytest.raises(OSError, match="Input/output error"):
            configs.save_state("box", NEW)
        assert _state_file(repo).read_bytes() == before
        assert _others(repo) == []

    @pytest.mark.parametrize("failure", [OSError(13, "Permission denied"), KeyboardInterrupt()],
                             ids=["oserror", "ctrl-c"])
    def test_a_failed_replace_keeps_the_old_file(self, repo, monkeypatch, failure):
        """Catches: a failed or interrupted rename leaving the old file changed, or a temporary file behind."""
        configs.save_state("box", OLD)
        before = _state_file(repo).read_bytes()

        def no_replace(src, dst):
            raise failure

        monkeypatch.setattr(configs.os, "replace", no_replace)
        with pytest.raises(type(failure)):
            configs.save_state("box", NEW)
        assert _state_file(repo).read_bytes() == before
        assert _others(repo) == []

    def test_a_failed_first_save_leaves_no_file_at_all(self, repo, monkeypatch):
        """Catches: a failed first save leaving a partial state file, which the next command would trip on."""
        monkeypatch.setattr(configs.os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError(28, "full")))
        with pytest.raises(OSError):
            configs.save_state("box", NEW)
        assert not _state_file(repo).exists()
        assert _others(repo) == []
        assert configs.load_state("box") == {}


# --- (6) load_state says when the file is corrupt -------------------------------------------------


class TestLoadState:
    def test_a_missing_file_is_empty(self, repo):
        """Catches: a config that never had an instance being reported as corrupt."""
        assert configs.load_state("box") == {}

    def test_the_error_is_a_config_error(self):
        """Catches: the error escaping the callers that already report ConfigError."""
        assert issubclass(configs.StateFileError, config.ConfigError)

    @pytest.mark.parametrize("content", CORRUPT.values(), ids=CORRUPT.keys())
    def test_a_corrupt_file_raises_naming_the_file(self, repo, content):
        """Catches: a truncated, empty or wrong-shaped file read as "no instance tracked"."""
        _state_file(repo).write_bytes(content)
        with pytest.raises(configs.StateFileError) as raised:
            configs.load_state("box")
        assert str(_state_file(repo)) in str(raised.value)
        assert raised.value.path == _state_file(repo)
        assert _state_file(repo).read_bytes() == content

    @pytest.mark.skipif(os.geteuid() == 0, reason="root can read any file")
    def test_an_unreadable_file_raises_naming_the_file(self, repo):
        """Catches: a file the user can't read being treated as no file."""
        configs.save_state("box", OLD)
        _state_file(repo).chmod(0)
        try:
            with pytest.raises(configs.StateFileError) as raised:
                configs.load_state("box")
        finally:
            _state_file(repo).chmod(0o600)
        assert str(_state_file(repo)) in str(raised.value)

    def test_the_message_says_what_to_do(self, repo):
        """Catches: an error that names the file but leaves the person guessing what to do next."""
        _state_file(repo).write_bytes(TRUNCATED)
        with pytest.raises(configs.StateFileError) as raised:
            configs.load_state("box")
        message = str(raised.value)
        assert "Look at the file" in message
        assert "move it aside" in message
        assert "cloudgpu up -C box" in message
        assert "adopts the instance mounting the config's filesystem" in message
        assert "provisioned profiles" in message and "unless the config lists them" in message


# --- (7) every reader stops ------------------------------------------------------------------------

COMMANDS = {
    "up": ["up", "-C", "box"],
    "up -P": ["up", "-C", "box", "-P", "a"],
    "down": ["down", "-C", "box", "--yes"],
    "down --delete-filesystem": ["down", "-C", "box", "--yes", "--delete-filesystem"],
    "status -C": ["status", "-C", "box"],
    "provision": ["provision", "-C", "box", "-P", "a"],
    "autoterminate": ["autoterminate", "-C", "box"],
    "autoterminate --off": ["autoterminate", "--off", "-C", "box"],
    "forward -C": ["forward", "-C", "box"],
    "forward -C -p": ["forward", "-C", "box", "-p", "8188"],
    "install -C": ["install", "-C", "box", "--app", "comfyui"],
    "recover -C": ["recover", "-C", "box"],
    "ssh -C": ["ssh", "-C", "box"],
}


@pytest.fixture
def outside(monkeypatch):
    """Every call that would leave this process: Lambda requests and subprocesses. Each is
    recorded and fails, so nothing is reached even on code that tries."""
    calls: list[tuple] = []

    def request(method, path, body=None):
        calls.append(("lambda", method, path))
        raise lambda_api.LambdaAPIError("no Lambda call may be made here")

    def process(*args, **kwargs):
        calls.append(("subprocess", args))
        raise OSError("no subprocess may be started here")

    monkeypatch.setattr(lambda_api, "_request", request)
    monkeypatch.setattr(subprocess, "run", process)
    monkeypatch.setattr(subprocess, "Popen", process)
    monkeypatch.setenv("LAMBDA_API_KEY", "dummy-key-not-real")
    return calls


def _flat(text):
    """Without whitespace: the console wraps long lines, and may split a path."""
    return "".join(text.split())


@pytest.mark.parametrize("content", [CORRUPT["truncated"], CORRUPT["empty"], CORRUPT["a list"]],
                         ids=["truncated", "empty", "a list"])
@pytest.mark.parametrize("args", COMMANDS.values(), ids=COMMANDS.keys())
def test_a_corrupt_state_file_stops_the_command(repo, outside, capfd, args, content):
    """Catches: a command treating a corrupt state file as "no instance": `up` launching a second
    machine or saving over the file, `down` saying there is nothing to tear down, the others
    saying to run `up` first."""
    _state_file(repo).write_bytes(content)
    result = CliRunner().invoke(cli.cli, args)
    captured = capfd.readouterr()
    output = result.output + captured.out + captured.err

    assert result.exit_code not in (0, None), output
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert "Traceback" not in output
    flat = _flat(output)
    assert _flat(str(_state_file(repo))) in flat, output
    assert _flat("move it aside") in flat, output
    assert _flat("cloudgpu up -C box") in flat, output
    assert _flat("unless the config lists them") in flat, output
    assert _flat("nothing to tear down") not in flat.lower(), output
    assert _flat("no running instance") not in flat.lower(), output
    assert outside == []
    assert _state_file(repo).read_bytes() == content
    assert [n for n in _others(repo) if n != "box.up.lock"] == []


def test_up_holds_nothing_after_stopping_on_a_corrupt_file(repo, outside):
    """Catches: `up` stopping on the corrupt file while still holding the config's lock."""
    _state_file(repo).write_bytes(TRUNCATED)
    assert CliRunner().invoke(cli.cli, ["up", "-C", "box"]).exit_code != 0
    configs.hold_up_lock("box").close()


def test_a_good_state_file_still_reaches_the_instance(repo, outside):
    """Catches: the check refusing a healthy state file (every test above would still pass)."""
    configs.save_state("box", OLD)
    result = CliRunner().invoke(cli.cli, ["status", "-C", "box"])
    assert result.exit_code != 0
    assert outside and outside[0][0] == "subprocess"
