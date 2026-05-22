"""SSH execution via subprocess - leverages user's existing SSH config/keys."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class SSHResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


SSH_OPTIONS = [
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=10",
    "-o", "BatchMode=yes",
]


def ssh_run(
    host: str,
    command: str,
    *,
    capture: bool = True,
    check: bool = False,
    timeout: int = 300,
) -> SSHResult:
    """Run a command on a remote host via SSH.

    Args:
        host: SSH host (user@ip or hostname from ssh config).
        command: Shell command to execute remotely.
        capture: If True, capture stdout/stderr. If False, stream to terminal.
        check: If True, raise on non-zero exit code.
        timeout: Command timeout in seconds.
    """
    cmd = ["ssh", *SSH_OPTIONS, host, command]

    if capture:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        ssh_result = SSHResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    else:
        result = subprocess.run(cmd, timeout=timeout)
        ssh_result = SSHResult(
            returncode=result.returncode,
            stdout="",
            stderr="",
        )

    if check and not ssh_result.ok:
        raise SSHError(
            f"SSH command failed (exit {ssh_result.returncode}): {command}\n"
            f"stderr: {ssh_result.stderr.strip()}"
        )
    return ssh_result


def ssh_test(host: str) -> bool:
    """Test SSH connectivity to a host."""
    try:
        result = ssh_run(host, "echo ok", timeout=15)
        return result.ok and "ok" in result.stdout
    except (subprocess.TimeoutExpired, OSError):
        return False


def ssh_interactive(host: str, command: str | None = None) -> int:
    """Open an interactive SSH session, optionally running a command."""
    cmd = ["ssh", *SSH_OPTIONS, host]
    if command:
        cmd.append(command)
    result = subprocess.run(cmd)
    return result.returncode


class SSHError(Exception):
    pass
