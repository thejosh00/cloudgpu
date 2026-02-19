"""Tests for SSH module."""

from __future__ import annotations

from unittest.mock import patch, MagicMock
import subprocess

from cloudgpu.local.ssh import ssh_run, ssh_test, SSHResult, SSHError, SSH_OPTIONS


class TestSSHResult:
    def test_ok_true(self):
        r = SSHResult(returncode=0, stdout="output", stderr="")
        assert r.ok is True

    def test_ok_false(self):
        r = SSHResult(returncode=1, stdout="", stderr="error")
        assert r.ok is False


class TestSSHRun:
    @patch("cloudgpu.local.ssh.subprocess.run")
    def test_captures_output(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="hello\n", stderr=""
        )
        result = ssh_run("user@host", "echo hello")
        assert result.ok
        assert result.stdout == "hello\n"

        call_args = mock_run.call_args
        assert call_args[0][0] == ["ssh", *SSH_OPTIONS, "user@host", "echo hello"]
        assert call_args[1]["capture_output"] is True

    @patch("cloudgpu.local.ssh.subprocess.run")
    def test_check_raises_on_failure(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="permission denied"
        )
        import pytest
        with pytest.raises(SSHError, match="permission denied"):
            ssh_run("user@host", "bad command", check=True)

    @patch("cloudgpu.local.ssh.subprocess.run")
    def test_no_capture_streams_to_terminal(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        result = ssh_run("user@host", "echo hello", capture=False)
        assert result.ok
        assert result.stdout == ""

        call_args = mock_run.call_args
        assert "capture_output" not in call_args[1]


class TestSSHTest:
    @patch("cloudgpu.local.ssh.ssh_run")
    def test_success(self, mock_ssh_run):
        mock_ssh_run.return_value = SSHResult(returncode=0, stdout="ok\n", stderr="")
        assert ssh_test("user@host") is True

    @patch("cloudgpu.local.ssh.ssh_run")
    def test_failure(self, mock_ssh_run):
        mock_ssh_run.return_value = SSHResult(returncode=1, stdout="", stderr="err")
        assert ssh_test("user@host") is False

    @patch("cloudgpu.local.ssh.ssh_run")
    def test_timeout(self, mock_ssh_run):
        mock_ssh_run.side_effect = subprocess.TimeoutExpired("ssh", 15)
        assert ssh_test("user@host") is False
