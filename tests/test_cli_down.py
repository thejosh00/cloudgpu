"""Tests for `cloudgpu down` (config-driven; mocked Lambda API)."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from cloudgpu import cli
from cloudgpu.local import configs, lambda_api


def _repo(tmp_path, monkeypatch, *, with_state=True):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "c.toml").write_text(
        'gpu = ["gh200"]\nssh_key = "mini"\nfilesystem = "prof"\n')
    monkeypatch.chdir(tmp_path)
    if with_state:
        configs.save_state("c", {
            "instance_id": "i-1", "ip": "1.2.3.4", "host": "ubuntu@1.2.3.4",
            "filesystem": "prof", "persistent_dir": "/lambda/nfs/prof",
        })


def test_down_terminates_and_keeps_filesystem(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch)
    terminated = []
    monkeypatch.setattr(lambda_api, "terminate_instances", lambda ids: terminated.append(ids))
    monkeypatch.setattr(lambda_api, "delete_filesystem",
                        lambda *a, **k: pytest.fail("must not delete filesystem without the flag"))

    result = CliRunner().invoke(cli.down, ["--yes", "-C", "c"])
    assert result.exit_code == 0, result.output
    assert terminated == [["i-1"]]
    assert configs.load_state("c") == {}  # state cleared


def test_down_delete_filesystem(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch)
    terminated, deleted = [], []
    monkeypatch.setattr(lambda_api, "terminate_instances", lambda ids: terminated.append(ids))
    monkeypatch.setattr(lambda_api, "get_instance", lambda i: {"status": "terminated"})
    monkeypatch.setattr(lambda_api, "list_filesystems", lambda: [{"name": "prof", "id": "fs-1"}])
    monkeypatch.setattr(lambda_api, "delete_filesystem", lambda fid: deleted.append(fid))

    result = CliRunner().invoke(cli.down, ["--yes", "--delete-filesystem", "-C", "c"])
    assert result.exit_code == 0, result.output
    assert terminated == [["i-1"]]
    assert deleted == ["fs-1"]
    assert configs.load_state("c") == {}


def test_down_without_instance_is_noop(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, with_state=False)
    monkeypatch.setattr(lambda_api, "list_instances", lambda: [])
    monkeypatch.setattr(lambda_api, "terminate_instances",
                        lambda ids: pytest.fail("nothing to terminate"))
    result = CliRunner().invoke(cli.down, ["--yes", "-C", "c"])
    assert result.exit_code == 0, result.output
    assert "No running instance" in result.output


def test_down_unknown_config_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no configs/ here
    result = CliRunner().invoke(cli.down, ["--yes", "-C", "nope"])
    assert result.exit_code != 0
    assert "No config" in result.output
