"""`cloudgpu down` asks the instance for a final sync before terminating (3.4 part 2).

No real API, instance, ssh or bucket: Lambda and ssh are monkeypatched in process, as
tests/test_cli_down.py does. Every Lambda and ssh call is recorded in one list so the order
(confirmation, final sync, terminate, filesystem delete) can be asserted.
"""

from __future__ import annotations

import subprocess

import pytest
from click.testing import CliRunner

from cloudgpu import cli
from cloudgpu.local import configs, lambda_api, ssh
from cloudgpu.remote import datasync_setup

BASE = 'gpu = ["gh200"]\nssh_key = "mini"\nfilesystem = "prof"\n'
SYNC = '\n[sync]\nremote = "b2:bkt"\n'
STATE = {"instance_id": "i-1", "ip": "1.2.3.4", "host": "ubuntu@1.2.3.4",
         "filesystem": "prof", "persistent_dir": "/lambda/nfs/prof"}
OK_SAID = (">>> Running the final sync...\n"
           ">>> Final sync finished at 2026-11-26T20:00:00Z. The sync timer stays stopped.\n")
FAIL_SAID = ("error: The final sync run failed (result: exit-code). The last successful sync finished at "
             "2026-11-26T19:40:00Z.\n>>> The sync timer is running again.\n")


def _repo(tmp_path, monkeypatch, *, with_sync=True, with_state=True):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "c.toml").write_text(BASE + (SYNC if with_sync else ""))
    monkeypatch.chdir(tmp_path)
    if with_state:
        configs.save_state("c", dict(STATE))
    return tmp_path / "configs" / "c.state.json"


def _fakes(monkeypatch, *, status="active", ssh_result=None, ssh_raises=None, instances=None):
    """Record every call; the instance reads `status` until terminate is called."""
    calls: list[tuple] = []
    terminated: list = []

    def get_instance(i):
        calls.append(("get_instance", i))
        if status == "gone" or terminated:
            return {"id": i, "status": "terminated"}
        return {"id": i, "status": status, "ip": "1.2.3.4"}

    def terminate(ids):
        calls.append(("terminate", ids))
        terminated.append(ids)

    def ssh_run(host, command, **kw):
        calls.append(("ssh", host, command, kw.get("timeout")))
        if ssh_raises is not None:
            raise ssh_raises
        return ssh_result or ssh.SSHResult(0, "", OK_SAID)

    monkeypatch.setattr(lambda_api, "get_instance", get_instance)
    monkeypatch.setattr(lambda_api, "list_instances", lambda: calls.append(("list_instances",)) or (instances or []))
    monkeypatch.setattr(lambda_api, "terminate_instances", terminate)
    monkeypatch.setattr(lambda_api, "list_filesystems", lambda: [{"name": "prof", "id": "fs-1"}])
    monkeypatch.setattr(lambda_api, "delete_filesystem", lambda fid: calls.append(("delete_filesystem", fid)))
    monkeypatch.setattr(ssh, "ssh_run", ssh_run)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    return calls


def _flat(result):
    return " ".join(result.output.split())


def _kinds(calls):
    return [c[0] for c in calls if c[0] in ("ssh", "terminate", "delete_filesystem")]


# (1) no [sync]: no ssh, output as today
def test_without_sync_down_makes_no_ssh_call(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, with_sync=False)
    calls = _fakes(monkeypatch)
    result = CliRunner().invoke(cli.down, ["-y", "-C", "c"])
    assert result.exit_code == 0, result.output
    assert _kinds(calls) == ["terminate"]
    assert "sync" not in result.output.lower()


def test_without_sync_skip_sync_is_not_an_error(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, with_sync=False)
    calls = _fakes(monkeypatch)
    result = CliRunner().invoke(cli.down, ["-y", "--skip-sync", "-C", "c"])
    assert result.exit_code == 0, result.output
    assert _kinds(calls) == ["terminate"]


# (2) success: one final-sync call after the confirmation, before terminate
def test_final_sync_runs_once_before_terminate_and_after_confirmation(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch)
    calls = _fakes(monkeypatch)
    seen_at_confirm = []
    real_confirm = cli.click.confirm

    def confirm(text, **kw):
        seen_at_confirm.append(list(_kinds(calls)))
        return real_confirm(text, **kw)

    monkeypatch.setattr(cli.click, "confirm", confirm)
    result = CliRunner().invoke(cli.down, ["-C", "c"], input="y\n")
    assert result.exit_code == 0, result.output
    assert seen_at_confirm == [[]]  # nothing synced or terminated before the question
    assert _kinds(calls) == ["ssh", "terminate"]
    ssh_call = next(c for c in calls if c[0] == "ssh")
    assert ssh_call[1] == "ubuntu@1.2.3.4"
    assert ssh_call[2] == cli._remote_cmd("/lambda/nfs/prof", "datasync", ["final"])
    out = _flat(result)
    assert "Final sync to b2:bkt" in out
    assert "Final sync finished at 2026-11-26T20:00:00Z" in out
    assert configs.load_state("c") == {}


def test_declining_the_confirmation_does_nothing(tmp_path, monkeypatch):
    state_file = _repo(tmp_path, monkeypatch)
    before = state_file.read_bytes()
    calls = _fakes(monkeypatch)
    result = CliRunner().invoke(cli.down, ["-C", "c"], input="n\n")
    assert result.exit_code != 0
    assert _kinds(calls) == []
    assert state_file.read_bytes() == before


def test_ssh_wait_covers_the_instance_wait_twice():
    assert cli.FINAL_SYNC_SSH_TIMEOUT > 2 * datasync_setup.FINAL_WAIT_SECONDS


def test_ssh_wait_passed_to_ssh(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch)
    calls = _fakes(monkeypatch)
    CliRunner().invoke(cli.down, ["-y", "-C", "c"])
    assert next(c for c in calls if c[0] == "ssh")[3] == cli.FINAL_SYNC_SSH_TIMEOUT


FAILURES = {
    "remote-nonzero": dict(ssh_result=ssh.SSHResult(1, "", FAIL_SAID)),
    "ssh-unreachable": dict(ssh_result=ssh.SSHResult(255, "", "ssh: connect to host 1.2.3.4 port 22: "
                                                              "Connection timed out\n")),
    "ssh-oserror": dict(ssh_raises=OSError("ssh not found")),
    "timeout": dict(ssh_raises=subprocess.TimeoutExpired("ssh", 1)),
    "not-installed": dict(ssh_result=ssh.SSHResult(1, "", "error: Sync is not set up on this instance, so "
                                                          "there is no final sync to do.\n")),
}


# (3) failure without -y, answered no
@pytest.mark.parametrize("how", sorted(FAILURES))
def test_failed_sync_answered_no_leaves_everything(tmp_path, monkeypatch, how):
    state_file = _repo(tmp_path, monkeypatch)
    before = state_file.read_bytes()
    calls = _fakes(monkeypatch, **FAILURES[how])
    result = CliRunner().invoke(cli.down, ["-C", "c"], input="y\nn\n")
    assert result.exit_code == 1, result.output
    assert _kinds(calls) == ["ssh"]
    assert state_file.read_bytes() == before
    out = _flat(result)
    assert "The final sync to b2:bkt failed" in out
    assert "still running and billing" in out
    assert "Terminate anyway" in out
    assert "Traceback" not in result.output
    if how == "remote-nonzero":
        assert "2026-11-26T19:40:00Z" in out
    if how == "not-installed":
        assert "Sync is not set up on this instance" in out


def test_failed_sync_default_answer_is_no(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch)
    calls = _fakes(monkeypatch, **FAILURES["remote-nonzero"])
    result = CliRunner().invoke(cli.down, ["-C", "c"], input="y\n\n")
    assert result.exit_code == 1, result.output
    assert _kinds(calls) == ["ssh"]


@pytest.mark.parametrize("how", sorted(FAILURES))
def test_failed_sync_answered_yes_terminates(tmp_path, monkeypatch, how):
    _repo(tmp_path, monkeypatch)
    calls = _fakes(monkeypatch, **FAILURES[how])
    result = CliRunner().invoke(cli.down, ["-C", "c"], input="y\ny\n")
    assert result.exit_code == 0, result.output
    assert _kinds(calls) == ["ssh", "terminate"]
    assert configs.load_state("c") == {}


# (4) failure with -y: refuse, name --skip-sync
@pytest.mark.parametrize("how", sorted(FAILURES))
def test_failed_sync_with_yes_refuses_and_names_skip_sync(tmp_path, monkeypatch, how):
    state_file = _repo(tmp_path, monkeypatch)
    before = state_file.read_bytes()
    calls = _fakes(monkeypatch, **FAILURES[how])
    result = CliRunner().invoke(cli.down, ["-y", "-C", "c"])
    assert result.exit_code == 1, result.output
    assert _kinds(calls) == ["ssh"]
    assert "--skip-sync" in result.output
    assert "Terminate anyway" not in result.output
    assert state_file.read_bytes() == before


# (5) --skip-sync, with and without -y
@pytest.mark.parametrize("args,inp", [(["-y"], None), ([], "y\n")])
def test_skip_sync_terminates_without_ssh(tmp_path, monkeypatch, args, inp):
    _repo(tmp_path, monkeypatch)
    calls = _fakes(monkeypatch)
    result = CliRunner().invoke(cli.down, [*args, "--skip-sync", "-C", "c"], input=inp)
    assert result.exit_code == 0, result.output
    assert _kinds(calls) == ["terminate"]
    out = _flat(result)
    assert "Final sync skipped" in out and "not in the bucket" in out


# (6) nothing to sync from: no ssh, no sync prompt, ends as today, also with -y and no --skip-sync
@pytest.mark.parametrize("args", [["-y"], []])
def test_already_gone_no_ssh(tmp_path, monkeypatch, args):
    _repo(tmp_path, monkeypatch)
    calls = _fakes(monkeypatch, status="gone")
    result = CliRunner().invoke(cli.down, [*args, "-C", "c"])
    assert result.exit_code == 0, result.output
    assert _kinds(calls) == []
    assert "already gone" in result.output
    assert "no final sync was possible" in result.output
    assert configs.load_state("c") == {}


@pytest.mark.parametrize("args", [["-y"], []])
def test_already_terminating_no_ssh(tmp_path, monkeypatch, args):
    _repo(tmp_path, monkeypatch, with_state=False)
    calls = _fakes(monkeypatch, instances=[{"id": "i-9", "status": "terminating", "ip": "1.2.3.4",
                                            "file_system_names": ["prof"]}])
    result = CliRunner().invoke(cli.down, [*args, "-C", "c"])
    assert result.exit_code == 0, result.output
    assert _kinds(calls) == []
    assert "already terminating" in result.output
    assert "no final sync was possible" in result.output


@pytest.mark.parametrize("args", [["-y"], []])
def test_no_instance_no_ssh(tmp_path, monkeypatch, args):
    _repo(tmp_path, monkeypatch, with_state=False)
    calls = _fakes(monkeypatch, instances=[])
    result = CliRunner().invoke(cli.down, [*args, "-C", "c"])
    assert result.exit_code == 0, result.output
    assert _kinds(calls) == []
    assert "No running instance" in result.output
    assert "no final sync was possible" in result.output


def test_found_by_filesystem_syncs_through_its_ip(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, with_state=False)
    calls = _fakes(monkeypatch, instances=[{"id": "i-9", "status": "active", "ip": "5.6.7.8",
                                            "file_system_names": ["prof"]}])
    result = CliRunner().invoke(cli.down, ["-y", "-C", "c"])
    assert result.exit_code == 0, result.output
    assert _kinds(calls) == ["ssh", "terminate"]
    assert next(c for c in calls if c[0] == "ssh")[1] == "ubuntu@5.6.7.8"


# (7) --delete-filesystem with a failed sync and a no: nothing deleted
def test_delete_filesystem_failed_sync_no_deletes_nothing(tmp_path, monkeypatch):
    state_file = _repo(tmp_path, monkeypatch)
    before = state_file.read_bytes()
    calls = _fakes(monkeypatch, **FAILURES["remote-nonzero"])
    result = CliRunner().invoke(cli.down, ["--delete-filesystem", "-C", "c"], input="y\nn\n")
    assert result.exit_code == 1, result.output
    assert _kinds(calls) == ["ssh"]
    assert state_file.read_bytes() == before


def test_delete_filesystem_syncs_first(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch)
    calls = _fakes(monkeypatch)
    result = CliRunner().invoke(cli.down, ["-y", "--delete-filesystem", "-C", "c"])
    assert result.exit_code == 0, result.output
    assert _kinds(calls) == ["ssh", "terminate", "delete_filesystem"]


def test_delete_filesystem_with_yes_and_failed_sync_refuses(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch)
    calls = _fakes(monkeypatch, **FAILURES["ssh-unreachable"])
    result = CliRunner().invoke(cli.down, ["-y", "--delete-filesystem", "-C", "c"])
    assert result.exit_code == 1
    assert _kinds(calls) == ["ssh"]


# Ctrl-C during the wait
def test_ctrl_c_during_final_sync_leaves_instance_and_state(tmp_path, monkeypatch):
    state_file = _repo(tmp_path, monkeypatch)
    before = state_file.read_bytes()
    calls = _fakes(monkeypatch, ssh_raises=KeyboardInterrupt())
    result = CliRunner().invoke(cli.down, ["-y", "-C", "c"])
    assert result.exit_code == 130, result.output
    assert _kinds(calls) == ["ssh"]
    assert state_file.read_bytes() == before
    assert "Interrupted during the final sync" in _flat(result)
    assert "Traceback" not in result.output


def test_lambda_terminate_has_no_final_sync(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch)
    calls = _fakes(monkeypatch)
    result = CliRunner().invoke(cli.cli, ["lambda", "terminate", "-y", "i-1"])
    assert _kinds(calls) in (["terminate"], [])
    assert "ssh" not in _kinds(calls)
