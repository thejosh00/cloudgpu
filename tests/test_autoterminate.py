"""Tests for the remote auto-terminate module (mocked sudo/systemd/network)."""

from __future__ import annotations

import json
import time
import urllib.error

import pytest

from cloudgpu.remote import autoterminate as at


class TestParseKeyFile:
    def test_extracts_key(self):
        assert at.parse_key_file("LAMBDA_API_KEY=secret_abc\n") == "secret_abc"

    def test_ignores_other_lines(self):
        text = "# comment\nOTHER=x\nLAMBDA_API_KEY=secret_abc\n"
        assert at.parse_key_file(text) == "secret_abc"

    def test_missing_key_raises(self):
        with pytest.raises(ValueError):
            at.parse_key_file("OTHER=x\n")

    def test_empty_value_raises(self):
        with pytest.raises(ValueError):
            at.parse_key_file("LAMBDA_API_KEY=\n")


class TestUnitTexts:
    def test_service_runs_check_from_etc_copy(self):
        service, _ = at.unit_texts()
        assert f"ExecStart=/usr/bin/python3 {at.SCRIPT_PATH} --check" in service
        assert "Type=oneshot" in service

    def test_timer_repeats_and_enables_on_boot(self):
        _, timer = at.unit_texts()
        assert "OnBootSec=" in timer
        assert "OnUnitActiveSec=" in timer
        assert "WantedBy=timers.target" in timer


class TestDue:
    def test_before_deadline(self):
        assert at.due({"deadline_epoch": 1000}, 999) is False

    def test_at_and_after_deadline(self):
        assert at.due({"deadline_epoch": 1000}, 1000) is True
        assert at.due({"deadline_epoch": 1000}, 5000) is True


class TestCheck:
    def _conf(self, tmp_path, monkeypatch, deadline):
        conf = {"instance_id": "i-123", "deadline_epoch": deadline, "api_key": "k"}
        path = tmp_path / "autoterminate.json"
        path.write_text(json.dumps(conf))
        monkeypatch.setattr(at, "CONF_PATH", str(path))
        return conf

    def test_not_due_does_nothing(self, tmp_path, monkeypatch):
        self._conf(tmp_path, monkeypatch, time.time() + 3600)
        monkeypatch.setattr(at, "terminate", lambda conf: pytest.fail("must not terminate"))
        assert at.check() == 0

    def test_due_terminates(self, tmp_path, monkeypatch):
        self._conf(tmp_path, monkeypatch, time.time() - 60)
        killed = []
        monkeypatch.setattr(at, "terminate", lambda conf: killed.append(conf["instance_id"]))
        assert at.check() == 0
        assert killed == ["i-123"]

    def test_api_failure_returns_nonzero_for_retry(self, tmp_path, monkeypatch):
        self._conf(tmp_path, monkeypatch, time.time() - 60)
        def boom(conf):
            raise urllib.error.URLError("down")
        monkeypatch.setattr(at, "terminate", boom)
        assert at.check() == 1

    def test_missing_conf_returns_nonzero(self, tmp_path, monkeypatch):
        monkeypatch.setattr(at, "CONF_PATH", str(tmp_path / "nope.json"))
        assert at.check() == 1


class TestArm:
    @pytest.fixture
    def sudo_log(self, monkeypatch):
        """Capture _sudo commands and _sudo_install writes instead of touching the system."""
        log = {"sudo": [], "installs": {}}
        monkeypatch.setattr(at, "_sudo", lambda *cmd: log["sudo"].append(list(cmd)))
        monkeypatch.setattr(
            at, "_sudo_install",
            lambda content, dest, mode: log["installs"].update({dest: (content, mode)}),
        )
        return log

    def test_arm_writes_conf_units_and_enables_timer(self, tmp_path, sudo_log):
        key_file = tmp_path / "key.env"
        key_file.write_text("LAMBDA_API_KEY=secret_abc\n")

        before = time.time()
        deadline = at.arm(2.0, "i-123", str(key_file))

        assert not key_file.exists()  # handoff copy must be deleted
        conf_content, conf_mode = sudo_log["installs"][at.CONF_PATH]
        assert conf_mode == "600"  # root-only: it holds the API key
        conf = json.loads(conf_content)
        assert conf["instance_id"] == "i-123"
        assert conf["api_key"] == "secret_abc"
        assert conf["deadline_epoch"] == deadline
        assert before + 2 * 3600 - 5 <= deadline <= time.time() + 2 * 3600 + 5

        # Self-copy + both units land in place, then the timer is enabled.
        assert at.SCRIPT_PATH in sudo_log["installs"]
        assert "def check()" in sudo_log["installs"][at.SCRIPT_PATH][0]
        assert f"/etc/systemd/system/{at.UNIT_NAME}.service" in sudo_log["installs"]
        assert f"/etc/systemd/system/{at.UNIT_NAME}.timer" in sudo_log["installs"]
        assert ["systemctl", "enable", "--now", f"{at.UNIT_NAME}.timer"] in sudo_log["sudo"]

    def test_arm_bad_key_file_raises(self, tmp_path, sudo_log):
        key_file = tmp_path / "key.env"
        key_file.write_text("nothing here\n")
        with pytest.raises(ValueError):
            at.arm(2.0, "i-123", str(key_file))
        assert sudo_log["installs"] == {}


class TestDisarm:
    def test_never_armed_is_silent_noop(self, tmp_path, monkeypatch):
        monkeypatch.setattr(at, "CONF_PATH", str(tmp_path / "nope.json"))
        monkeypatch.setattr(at, "_sudo", lambda *cmd: pytest.fail("must not sudo"))
        at.disarm()  # no error, no sudo

    def test_armed_removes_everything(self, tmp_path, monkeypatch):
        conf = tmp_path / "autoterminate.json"
        conf.write_text("{}")
        monkeypatch.setattr(at, "CONF_PATH", str(conf))
        sudo_calls = []
        monkeypatch.setattr(at, "_sudo", lambda *cmd: sudo_calls.append(list(cmd)))
        disabled = []
        monkeypatch.setattr(
            at.subprocess, "run",
            lambda cmd, **k: disabled.append(cmd),
        )
        at.disarm()
        assert any("disable" in cmd for cmd in disabled)
        rm = next(c for c in sudo_calls if c[:2] == ["rm", "-f"])
        assert str(conf) in rm and at.SCRIPT_PATH in rm
        assert ["systemctl", "daemon-reload"] in sudo_calls
