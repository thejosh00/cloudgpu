"""Tests for `cloudgpu down` (mocked Lambda API; never touches a real instance)."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from cloudgpu import cli
from cloudgpu.local import lambda_api, profiles


@pytest.fixture
def profile_with_instance(tmp_config_dir):
    profiles.create_profile("p", filesystem="p", gpu=["gh200"], apps=[], ssh_key="mini")
    profiles.set_active("p")
    profiles.save_runtime("p", {
        "instance_id": "i-1", "ip": "1.2.3.4", "host": "ubuntu@1.2.3.4",
        "filesystem": "p", "persistent_dir": "/lambda/nfs/p",
    })
    return "p"


def test_down_terminates_and_keeps_filesystem(profile_with_instance, monkeypatch):
    terminated = []
    monkeypatch.setattr(lambda_api, "terminate_instances", lambda ids: terminated.append(ids))
    monkeypatch.setattr(lambda_api, "delete_filesystem",
                        lambda *a, **k: pytest.fail("must not delete filesystem without the flag"))

    result = CliRunner().invoke(cli.down, ["--yes"])
    assert result.exit_code == 0, result.output
    assert terminated == [["i-1"]]
    assert profiles.load_runtime("p") == {}  # runtime cleared


def test_down_delete_filesystem_terminates_then_deletes(profile_with_instance, monkeypatch):
    terminated, deleted = [], []
    monkeypatch.setattr(lambda_api, "terminate_instances", lambda ids: terminated.append(ids))
    monkeypatch.setattr(lambda_api, "get_instance", lambda i: {"status": "terminated"})
    monkeypatch.setattr(lambda_api, "list_filesystems",
                        lambda: [{"name": "p", "id": "fs-1"}])
    monkeypatch.setattr(lambda_api, "delete_filesystem", lambda fid: deleted.append(fid))

    result = CliRunner().invoke(cli.down, ["--yes", "--delete-filesystem"])
    assert result.exit_code == 0, result.output
    assert terminated == [["i-1"]]
    assert deleted == ["fs-1"]
    assert profiles.load_runtime("p") == {}


def test_down_without_running_instance_is_noop_terminate(tmp_config_dir, monkeypatch):
    profiles.create_profile("idle", filesystem="idle", gpu=["gh200"], apps=[], ssh_key="mini")
    profiles.set_active("idle")
    # no runtime, and nothing mounting the filesystem
    monkeypatch.setattr(lambda_api, "list_instances", lambda: [])
    monkeypatch.setattr(lambda_api, "terminate_instances",
                        lambda ids: pytest.fail("nothing to terminate"))

    result = CliRunner().invoke(cli.down, ["--yes"])
    assert result.exit_code == 0, result.output
    assert "No running instance" in result.output
