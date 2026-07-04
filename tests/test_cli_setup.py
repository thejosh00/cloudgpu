"""Tests for host setup / persistent-dir detection. Mocked SSH/rsync."""

from __future__ import annotations

import json

import pytest

from cloudgpu import cli
from cloudgpu.local import ssh


class TestPickPersistentDir:
    def test_profile_filesystem_selected(self):
        assert cli._pick_persistent_dir(["aaa", "my-comfy"], "my-comfy") == "/lambda/nfs/my-comfy"

    def test_profile_filesystem_not_mounted_exits(self):
        with pytest.raises(SystemExit):
            cli._pick_persistent_dir(["other-fs"], "my-comfy")

    def test_profile_filesystem_nothing_mounted_exits(self):
        with pytest.raises(SystemExit):
            cli._pick_persistent_dir([], "my-comfy")

    def test_no_filesystem_single_mount(self):
        assert cli._pick_persistent_dir(["only-fs"], None) == "/lambda/nfs/only-fs"

    def test_no_filesystem_no_mounts_exits(self):
        with pytest.raises(SystemExit):
            cli._pick_persistent_dir([], None)

    def test_no_filesystem_multiple_mounts_is_ambiguous(self):
        with pytest.raises(SystemExit):
            cli._pick_persistent_dir(["aaa", "bbb"], None)


class TestSetupHost:
    """The detect one-liner's mounts must flow into _pick_persistent_dir."""

    def _wire(self, monkeypatch, mounts):
        monkeypatch.setattr(cli.ssh, "ssh_test", lambda host: True)

        def fake_run(host, command, **kwargs):
            if "python3 -c" in command:  # the quick-detect one-liner
                return ssh.SSHResult(0, json.dumps({"mounts": mounts}), "")
            return ssh.SSHResult(0, json.dumps({}), "")  # full detection via _remote_run

        monkeypatch.setattr(cli.ssh, "ssh_run", fake_run)
        monkeypatch.setattr(cli.sync, "sync_remote", lambda host, pd: None)

    def test_uses_profile_filesystem_over_sort_order(self, monkeypatch):
        # 'zz-other' sorts last; the old detection would have picked it.
        self._wire(monkeypatch, ["my-comfy", "zz-other"])
        persistent_dir, _ = cli._setup_host("ubuntu@1.2.3.4", "my-comfy")
        assert persistent_dir == "/lambda/nfs/my-comfy"

    def test_missing_profile_filesystem_fails_loudly(self, monkeypatch):
        self._wire(monkeypatch, ["zz-other"])
        with pytest.raises(SystemExit):
            cli._setup_host("ubuntu@1.2.3.4", "my-comfy")

    def test_manual_flow_single_mount(self, monkeypatch):
        self._wire(monkeypatch, ["only-fs"])
        persistent_dir, _ = cli._setup_host("ubuntu@1.2.3.4")
        assert persistent_dir == "/lambda/nfs/only-fs"
