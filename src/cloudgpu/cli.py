"""Click CLI: setup, install, recover, status, ssh."""

from __future__ import annotations

import json
import sys

import click
from rich.console import Console

from cloudgpu.local import config, display, lambda_api, ssh, sync

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


def _api_call(fn, *args, **kwargs):
    """Run a Lambda API call, surfacing errors as a clean message and exit."""
    try:
        return fn(*args, **kwargs)
    except lambda_api.LambdaAPIError as e:
        display.error(str(e))
        sys.exit(1)


@cli.group(name="lambda")
def lambda_group() -> None:
    """Manage Lambda Cloud resources via the API (needs LAMBDA_API_KEY)."""
    pass


@lambda_group.command(name="instances")
def lambda_instances() -> None:
    """List running instances."""
    display.show_instances(_api_call(lambda_api.list_instances))


@lambda_group.command(name="instance-types")
@click.option("--available", is_flag=True, help="Only show types with capacity available.")
def lambda_instance_types(available: bool) -> None:
    """List available instance types and their pricing/capacity."""
    display.show_instance_types(_api_call(lambda_api.list_instance_types), available_only=available)


@lambda_group.command(name="launch")
@click.option("--region", "region_name", required=True, help="Region, e.g. us-tx-1")
@click.option("--type", "instance_type_name", required=True, help="Instance type, e.g. gpu_1x_a10")
@click.option("--ssh-key", "ssh_keys", multiple=True, required=True, help="SSH key name (repeatable)")
@click.option("--filesystem", "filesystems", multiple=True, help="Filesystem to mount (repeatable)")
@click.option("--name", "name", default=None, help="Instance name")
def lambda_launch(
    region_name: str,
    instance_type_name: str,
    ssh_keys: tuple[str, ...],
    filesystems: tuple[str, ...],
    name: str | None,
) -> None:
    """Launch a new instance."""
    result = _api_call(
        lambda_api.launch_instance,
        region_name=region_name,
        instance_type_name=instance_type_name,
        ssh_key_names=list(ssh_keys),
        file_system_names=list(filesystems) or None,
        name=name,
    )
    ids = _extract_instance_ids(result)
    if ids:
        display.success("Launched instance(s): " + ", ".join(ids))
    else:
        display.success("Launch request accepted.")
        display.info(json.dumps(result, indent=2))


@lambda_group.command(name="restart")
@click.argument("instance_ids", nargs=-1, required=True)
def lambda_restart(instance_ids: tuple[str, ...]) -> None:
    """Restart one or more instances by id."""
    _api_call(lambda_api.restart_instances, list(instance_ids))
    display.success("Restart requested for: " + ", ".join(instance_ids))


@lambda_group.command(name="terminate")
@click.argument("instance_ids", nargs=-1, required=True)
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
def lambda_terminate(instance_ids: tuple[str, ...], yes: bool) -> None:
    """Terminate one or more instances by id."""
    if not yes:
        click.confirm(
            f"Terminate {len(instance_ids)} instance(s): {', '.join(instance_ids)}?",
            abort=True,
        )
    _api_call(lambda_api.terminate_instances, list(instance_ids))
    display.success("Termination requested for: " + ", ".join(instance_ids))


@lambda_group.command(name="filesystems")
def lambda_filesystems() -> None:
    """List filesystems."""
    display.show_filesystems(_api_call(lambda_api.list_filesystems))


@lambda_group.command(name="create-filesystem")
@click.argument("name")
@click.option("--region", "region", required=True, help="Region, e.g. us-tx-1")
def lambda_create_filesystem(name: str, region: str) -> None:
    """Create a filesystem named NAME in REGION."""
    fs = _api_call(lambda_api.create_filesystem, name, region)
    fs_id = fs.get("id") if isinstance(fs, dict) else None
    display.success(f"Created filesystem '{name}'" + (f" (id: {fs_id})" if fs_id else ""))


@lambda_group.command(name="delete-filesystem")
@click.argument("filesystem_id")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
def lambda_delete_filesystem(filesystem_id: str, yes: bool) -> None:
    """Delete a filesystem by FILESYSTEM_ID."""
    if not yes:
        click.confirm(f"Delete filesystem {filesystem_id}?", abort=True)
    _api_call(lambda_api.delete_filesystem, filesystem_id)
    display.success(f"Deleted filesystem {filesystem_id}")


def _extract_instance_ids(launch_result) -> list[str]:
    """Pull instance ids out of a launch response (shape varies by API version)."""
    if isinstance(launch_result, dict):
        if isinstance(launch_result.get("instance_ids"), list):
            return [str(i) for i in launch_result["instance_ids"]]
        if launch_result.get("id"):
            return [str(launch_result["id"])]
    elif isinstance(launch_result, list):
        return [str(i.get("id")) for i in launch_result if isinstance(i, dict) and i.get("id")]
    return []
