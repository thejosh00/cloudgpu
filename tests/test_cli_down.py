"""Tests for `cloudgpu down` (directory profile; mocked Lambda API)."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from cloudgpu import cli
from cloudgpu.local import lambda_api, profiles


def _profile_dir(tmp_path, *, with_state=True):
    d = tmp_path / "prof"
    d.mkdir()
    (d / "cloudgpu.toml").write_text('gpu = ["gh200"]\nssh_key = "mini"\nfilesystem = "prof"\n')
    if with_state:
        profiles.save_state(d, {
            "instance_id": "i-1", "ip": "1.2.3.4", "host": "ubuntu@1.2.3.4",
            "filesystem": "prof", "persistent_dir": "/lambda/nfs/prof",
        })
    return d


def test_down_terminates_and_keeps_filesystem(tmp_path, monkeypatch):
    d = _profile_dir(tmp_path)
    terminated = []
    monkeypatch.setattr(lambda_api, "terminate_instances", lambda ids: terminated.append(ids))
    monkeypatch.setattr(lambda_api, "delete_filesystem",
                        lambda *a, **k: pytest.fail("must not delete filesystem without the flag"))

    result = CliRunner().invoke(cli.down, ["--yes", "--profile", str(d)])
    assert result.exit_code == 0, result.output
    assert terminated == [["i-1"]]
    assert profiles.load_state(d) == {}  # state cleared


def test_down_delete_filesystem(tmp_path, monkeypatch):
    d = _profile_dir(tmp_path)
    terminated, deleted = [], []
    monkeypatch.setattr(lambda_api, "terminate_instances", lambda ids: terminated.append(ids))
    monkeypatch.setattr(lambda_api, "get_instance", lambda i: {"status": "terminated"})
    monkeypatch.setattr(lambda_api, "list_filesystems", lambda: [{"name": "prof", "id": "fs-1"}])
    monkeypatch.setattr(lambda_api, "delete_filesystem", lambda fid: deleted.append(fid))

    result = CliRunner().invoke(cli.down, ["--yes", "--delete-filesystem", "--profile", str(d)])
    assert result.exit_code == 0, result.output
    assert terminated == [["i-1"]]
    assert deleted == ["fs-1"]
    assert profiles.load_state(d) == {}


def test_down_without_instance_is_noop(tmp_path, monkeypatch):
    d = _profile_dir(tmp_path, with_state=False)
    monkeypatch.setattr(lambda_api, "list_instances", lambda: [])
    monkeypatch.setattr(lambda_api, "terminate_instances",
                        lambda ids: pytest.fail("nothing to terminate"))
    result = CliRunner().invoke(cli.down, ["--yes", "--profile", str(d)])
    assert result.exit_code == 0, result.output
    assert "No running instance" in result.output


def test_down_outside_profile_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no cloudgpu.toml here
    result = CliRunner().invoke(cli.down, ["--yes"])
    assert result.exit_code != 0
    assert "cloudgpu.toml" in result.output
