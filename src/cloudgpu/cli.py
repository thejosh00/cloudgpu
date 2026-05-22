"""Click CLI: setup, install, recover, status, ssh."""

from __future__ import annotations

import json
import sys

import click
from rich.console import Console

from cloudgpu.local import config, display, ssh, sync

console = Console()

AVAILABLE_APPS = ["comfyui"]


def _remote_cmd(persistent_dir: str, command: str, extra_args: list[str] | None = None) -> str:
    """Build the SSH command string to invoke the remote tool."""
    cloudgpu_dir = f"{persistent_dir}/cloudgpu"
    cmd_parts = [
        f"PYTHONPATH={cloudgpu_dir}",
        "python3 -m remote",
        f"--persistent-dir {persistent_dir}",
        command,
    ]
    if extra_args:
        cmd_parts.extend(extra_args)
    return " ".join(cmd_parts)


def _remote_run(host: str, persistent_dir: str, command: str, extra_args: list[str] | None = None) -> str:
    """Run a remote command via the synced tool and return stdout."""
    full_cmd = _remote_cmd(persistent_dir, command, extra_args)
    result = ssh.ssh_run(host, full_cmd, check=True, timeout=1200)
    return result.stdout


def _resolve_target(host: str | None) -> tuple[str, str]:
    """Resolve the host and persistent dir, surfacing config errors to the user."""
    try:
        host = config.get_host(host)
        persistent_dir = config.get_persistent_dir()
    except config.ConfigError as e:
        raise click.UsageError(str(e)) from e
    return host, persistent_dir


@click.group()
def cli() -> None:
    """CloudGPU - Install GPU Python apps on Lambda Labs instances."""
    pass


@cli.command()
@click.argument("host")
def setup(host: str) -> None:
    """Test SSH, detect persistent dir, sync tool, save config."""
    # 1. Test SSH
    with console.status("Testing SSH connection..."):
        if not ssh.ssh_test(host):
            display.error(f"Cannot connect to {host} via SSH.")
            display.info("Make sure you can run: ssh " + host)
            sys.exit(1)
    display.success(f"SSH connection to {host} OK")

    # 2. Sync remote scripts (need to detect persistent dir first)
    with console.status("Detecting instance environment..."):
        # Run detect directly before syncing (it's a standalone script)
        detect_cmd = "python3 -c \"" + (
            "import os, subprocess, json; "
            "nfs='/lambda/nfs'; "
            "pd=None; "
            "[pd:=os.path.join(nfs,e) for e in sorted(os.listdir(nfs)) "
            "if os.path.isdir(os.path.join(nfs,e)) and not e.startswith('.')] "
            "if os.path.isdir(nfs) else None; "
            "print(json.dumps({'persistent_dir': pd}))"
        ) + "\""
        result = ssh.ssh_run(host, detect_cmd, check=True)
        quick_detect = json.loads(result.stdout)

    persistent_dir = quick_detect.get("persistent_dir")
    if not persistent_dir:
        display.error("No persistent directory found at /lambda/nfs/")
        display.info("Make sure a filesystem is attached to this instance.")
        sys.exit(1)

    display.success(f"Persistent directory: {persistent_dir}")

    # 3. Sync remote tool
    with console.status("Syncing remote tool..."):
        sync.sync_remote(host, persistent_dir)
    display.success("Remote tool synced")

    # 4. Full detection via synced tool
    with console.status("Running full detection..."):
        output = _remote_run(host, persistent_dir, "detect")
        detection = json.loads(output)

    display.show_detection(detection)

    # 5. Save config
    config.save_host(host, persistent_dir)
    display.success(f"Config saved. You can now run commands without specifying the host.")


@cli.command()
@click.argument("host", required=False)
@click.option("--app", type=click.Choice(AVAILABLE_APPS), help="App to install")
def install(host: str | None, app: str | None) -> None:
    """Install a GPU app on the instance."""
    host, persistent_dir = _resolve_target(host)

    # If no app specified, prompt interactively
    if not app:
        app = click.prompt(
            "Which app to install?",
            type=click.Choice(AVAILABLE_APPS),
            default=AVAILABLE_APPS[0],
        )

    # Sync latest remote scripts
    with console.status("Syncing remote tool..."):
        sync.sync_remote(host, persistent_dir)

    # Run install
    display.info(f"Installing {app}...")
    result = ssh.ssh_run(
        host,
        _remote_cmd(persistent_dir, "install", [f"--app {app}"]),
        capture=False,
        timeout=1200,
    )
    if result.ok:
        display.success(f"{app} installed successfully!")
    else:
        display.error(f"Installation failed (exit code {result.returncode})")
        sys.exit(1)


@cli.command()
@click.argument("host", required=False)
def recover(host: str | None) -> None:
    """Restore everything on a new instance from persistent storage."""
    host, persistent_dir = _resolve_target(host)

    # Sync remote tool
    with console.status("Syncing remote tool..."):
        sync.sync_remote(host, persistent_dir)

    # Run recovery
    display.info("Running recovery...")
    result = ssh.ssh_run(
        host,
        _remote_cmd(persistent_dir, "recover"),
        capture=False,
        timeout=1200,
    )

    if result.ok:
        # Update saved host to new instance
        config.save_host(host, persistent_dir)
        display.success("Recovery complete!")
    else:
        display.error(f"Recovery failed (exit code {result.returncode})")
        sys.exit(1)


@cli.command()
@click.argument("host", required=False)
def status(host: str | None) -> None:
    """Show what's installed and their health."""
    host, persistent_dir = _resolve_target(host)

    with console.status("Syncing remote tool..."):
        sync.sync_remote(host, persistent_dir)

    with console.status("Checking status..."):
        output = _remote_run(host, persistent_dir, "status")
        status_data = json.loads(output)

    display.show_detection(status_data.get("detection", {}))
    display.show_status(status_data.get("apps", {}))


@cli.command(name="ssh", context_settings={"ignore_unknown_options": True})
@click.option("--host", "-H", default=None, help="SSH host (uses saved host if omitted)")
@click.argument("command", nargs=-1, type=click.UNPROCESSED)
def ssh_cmd(host: str | None, command: tuple[str, ...]) -> None:
    """Open SSH to instance, optionally running a command.

    Examples:
        cloudgpu ssh
        cloudgpu ssh -- comfyui
        cloudgpu ssh -H my-host -- nvidia-smi
    """
    host, persistent_dir = _resolve_target(host)

    cmd = None
    if command:
        # Prepend the cloudgpu/bin to PATH so launch scripts are available
        bin_dir = f"{persistent_dir}/cloudgpu/bin"
        cmd = f"export PATH={bin_dir}:$PATH && " + " ".join(command)

    exit_code = ssh.ssh_interactive(host, cmd)
    sys.exit(exit_code)
