"""Click CLI: init, up, down, setup, install, recover, status, ssh, forward."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click
from rich.console import Console

from cloudgpu.local import apps, config, display, lambda_api, orchestration, profiles, ssh, sync

console = Console()

# Installable apps come from the app registry (cloudgpu.local.apps); core stays generic.
AVAILABLE_APPS = apps.AVAILABLE_APPS


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


def _resolve_target(host: str | None = None, profile_dir: str | None = None) -> tuple[str, str]:
    """Resolve the (host, persistent_dir) to operate on.

    Precedence:
      1. An explicit ``host`` argument/flag (persistent_dir from config.json).
      2. A profile directory (``--profile``, else the current dir if it has a
         cloudgpu.toml) — uses its cloudgpu.state.json.
      3. The global single-host config saved by ``cloudgpu setup``.
    """
    if not host:
        d = None
        if profile_dir:
            try:
                d = profiles.find_profile_dir(profile_dir)
            except config.ConfigError as e:
                raise click.UsageError(str(e)) from e
        elif profiles.has_profile(Path.cwd()):
            d = Path.cwd()
        if d is not None:
            state = profiles.load_state(d)
            if state.get("host") and state.get("persistent_dir"):
                return state["host"], state["persistent_dir"]
            raise click.UsageError(
                f"Profile at '{d}' has no running instance yet. Run 'cloudgpu up' first."
            )
    try:
        host = config.get_host(host)
        persistent_dir = config.get_persistent_dir()
    except config.ConfigError as e:
        raise click.UsageError(str(e)) from e
    return host, persistent_dir


def _record_launch_info(host: str, persistent_dir: str) -> dict:
    """Save how to recover this deployment, enriching via the Lambda API if possible.

    The filesystem (tail of persistent_dir) is always recorded. region,
    instance_type, and ssh_key are best-effort: they need LAMBDA_API_KEY and the
    instance to be findable, so any API failure is swallowed silently.
    """
    filesystem = persistent_dir.rstrip("/").split("/")[-1]
    instance_type = region = ssh_key = None
    try:
        ip = host.split("@")[-1]
        for inst in lambda_api.list_instances():
            if inst.get("ip") == ip:
                instance_type = (inst.get("instance_type") or {}).get("name")
                region = (inst.get("region") or {}).get("name")
                names = inst.get("ssh_key_names") or []
                ssh_key = names[0] if names else None
                break
        if region is None:  # instance not matched; derive region from the filesystem
            for fs in lambda_api.list_filesystems():
                if fs.get("name") == filesystem:
                    r = fs.get("region") or {}
                    region = r.get("name") if isinstance(r, dict) else r
                    break
    except Exception:
        pass  # best-effort enrichment: still record the filesystem below

    config.save_launch_info(
        filesystem, instance_type=instance_type, ssh_key=ssh_key, region=region
    )
    return {
        "filesystem": filesystem,
        "instance_type": instance_type,
        "region": region,
        "ssh_key": ssh_key,
    }


@click.group()
def cli() -> None:
    """CloudGPU - Install GPU Python apps on Lambda Labs instances."""
    pass


def _setup_host(host: str) -> tuple[str, dict]:
    """SSH-test, detect the persistent dir, sync the remote tool, run full detection.

    Returns (persistent_dir, detection). Exits the process with a clear message on
    SSH failure or a missing persistent directory. Shared by `setup` and `up`.
    """
    # 1. Test SSH
    with console.status("Testing SSH connection..."):
        if not ssh.ssh_test(host):
            display.error(f"Cannot connect to {host} via SSH.")
            display.info("Make sure you can run: ssh " + host)
            sys.exit(1)
    display.success(f"SSH connection to {host} OK")

    # 2. Detect persistent dir (standalone one-liner, before the tool is synced)
    with console.status("Detecting instance environment..."):
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

    return persistent_dir, detection


@cli.command()
@click.argument("host")
def setup(host: str) -> None:
    """Test SSH, detect persistent dir, sync tool, save config."""
    persistent_dir, _ = _setup_host(host)

    # Save config
    config.save_host(host, persistent_dir)
    display.success("Config saved. You can now run commands without specifying the host.")

    # Record how to recover this deployment (filesystem + launch params)
    info = _record_launch_info(host, persistent_dir)
    detail = info.get("instance_type") or "?"
    region = info.get("region") or "?"
    display.info(f"Launch info saved for recovery: {info['filesystem']} ({detail}, {region})")


@cli.command()
@click.argument("host", required=False)
@click.option("--app", type=click.Choice(AVAILABLE_APPS), help="App to install")
@click.option("--profile", "-P", "profile", default=None, help="Profile directory (default: current dir)")
def install(host: str | None, app: str | None, profile: str | None) -> None:
    """Install a GPU app on the instance."""
    host, persistent_dir = _resolve_target(host, profile)

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
@click.option("--profile", "-P", "profile", default=None, help="Profile directory (default: current dir)")
def recover(host: str | None, profile: str | None) -> None:
    """Restore everything on a new instance from persistent storage."""
    host, persistent_dir = _resolve_target(host, profile)

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
        # Update saved host (and launch info) to the new instance
        config.save_host(host, persistent_dir)
        _record_launch_info(host, persistent_dir)
        display.success("Recovery complete!")
    else:
        display.error(f"Recovery failed (exit code {result.returncode})")
        sys.exit(1)


@cli.command()
@click.argument("host", required=False)
@click.option("--profile", "-P", "profile", default=None, help="Profile directory (default: current dir)")
def status(host: str | None, profile: str | None) -> None:
    """Show what's installed and their health."""
    host, persistent_dir = _resolve_target(host, profile)

    with console.status("Syncing remote tool..."):
        sync.sync_remote(host, persistent_dir)

    with console.status("Checking status..."):
        output = _remote_run(host, persistent_dir, "status")
        status_data = json.loads(output)

    display.show_detection(status_data.get("detection", {}))
    display.show_status(status_data.get("apps", {}))


@cli.command(name="ssh", context_settings={"ignore_unknown_options": True})
@click.option("--host", "-H", default=None, help="SSH host (uses saved host if omitted)")
@click.option("--profile", "-P", "profile", default=None, help="Profile directory (default: current dir)")
@click.argument("command", nargs=-1, type=click.UNPROCESSED)
def ssh_cmd(host: str | None, profile: str | None, command: tuple[str, ...]) -> None:
    """Open SSH to instance, optionally running a command.

    Examples:
        cloudgpu ssh
        cloudgpu ssh -- comfyui
        cloudgpu ssh -H my-host -- nvidia-smi
    """
    host, persistent_dir = _resolve_target(host, profile)

    cmd = None
    if command:
        # Prepend the cloudgpu/bin to PATH so launch scripts are available
        bin_dir = f"{persistent_dir}/cloudgpu/bin"
        cmd = f"export PATH={bin_dir}:$PATH && " + " ".join(command)

    exit_code = ssh.ssh_interactive(host, cmd)
    sys.exit(exit_code)


def _profile_ports(profile_dir: str | None) -> list[int]:
    """Ports to forward for the profile's apps (deduped, in order)."""
    try:
        profile = profiles.load_profile(profiles.find_profile_dir(profile_dir))
    except config.ConfigError:
        return []
    return apps.app_ports(profile.get("apps", []))


@cli.command(name="forward")
@click.option("--host", "-H", default=None, help="SSH host (bypasses the profile; requires --port)")
@click.option("--port", "-p", "port", default=None, type=int, help="Forward a specific remote port (default: the profile's app ports)")
@click.option("--local-port", "local_port", default=None, type=int, help="Local port (single-port mode; defaults to --port)")
@click.option(
    "--run", "run_cmd", default=None,
    help="Run a command on the instance over the tunnel (single port). Tunnel lives as "
         "long as the command; Ctrl-C stops both.",
)
@click.option("--profile", "-P", "profile", default=None, help="Profile directory (default: current dir)")
def forward(host: str | None, port: int | None, local_port: int | None, run_cmd: str | None, profile: str | None) -> None:
    """Forward the profile's app port(s) to localhost over SSH.

    With no --port, the ports come from the profile's apps (services run on the instance,
    so just `cloudgpu forward` then open the URL). Use --port for an explicit port, or
    --run to start a command over the tunnel.

    Examples:
        cloudgpu forward                    # forward the profile's app ports
        cloudgpu forward -p 8888            # forward a specific port
        cloudgpu forward --run nvidia-smi   # run a command over the tunnel
    """
    host, persistent_dir = _resolve_target(host, profile)

    # Single-port mode: explicit --port, or --run (a command needs one port).
    if port is not None or run_cmd:
        if port is None:
            derived = _profile_ports(profile)
            if not derived:
                raise click.UsageError("Pass --port <n> (no app port to infer from this profile).")
            port = derived[0]
        local_port = local_port or port
        spec = f"{local_port}:localhost:{port}"
        display.info(f"Forwarding http://localhost:{local_port} -> {host} port {port}")
        if run_cmd:
            bin_dir = f"{persistent_dir}/cloudgpu/bin"
            command = f"export PATH={bin_dir}:$PATH && {run_cmd}"
            display.info(f"Starting '{run_cmd}' (Ctrl-C stops it and closes the tunnel)...")
            sys.exit(ssh.ssh_interactive(host, command, ssh_args=["-L", spec]))
        display.info("Holding the tunnel; press Ctrl-C to close it.")
        sys.exit(ssh.ssh_interactive(host, ssh_args=["-L", spec, "-N"]))

    # Profile mode: forward every app port in one tunnel.
    ports = _profile_ports(profile)
    if not ports:
        raise click.UsageError(
            "No app ports to forward for this profile. Pass --port <n>, or add an app "
            f"with a known port ({', '.join(AVAILABLE_APPS)}) to cloudgpu.toml."
        )
    ssh_args: list[str] = []
    for p in ports:
        ssh_args += ["-L", f"{p}:localhost:{p}"]
        display.info(f"Forwarding http://localhost:{p} -> {host} port {p}")
    ssh_args.append("-N")
    display.info("Holding the tunnel; press Ctrl-C to close it.")
    sys.exit(ssh.ssh_interactive(host, ssh_args=ssh_args))


# --- profile-driven machine management ------------------------------------


def _live_instance(instance_id: str | None) -> dict | None:
    """Return the instance dict if it's active with an IP, else None."""
    if not instance_id:
        return None
    try:
        inst = lambda_api.get_instance(instance_id)
    except lambda_api.LambdaAPIError:
        return None  # gone / terminated
    if inst.get("status") == "active" and inst.get("ip"):
        return inst
    return None


def _instance_mounting(filesystem: str) -> dict | None:
    """Find a live instance already mounting ``filesystem`` (or None)."""
    try:
        for inst in lambda_api.list_instances():
            names = inst.get("file_system_names") or []
            if filesystem in names and inst.get("status") in ("active", "booting"):
                return inst
    except lambda_api.LambdaAPIError:
        pass
    return None


def _runtime_from_instance(inst: dict, filesystem: str) -> dict:
    """Build a runtime dict from a Lambda instance object (used when adopting)."""
    itype = inst.get("instance_type") or {}
    region = inst.get("region") or {}
    ip = inst.get("ip")
    return {
        "instance_id": inst.get("id"),
        "ip": ip,
        "host": f"ubuntu@{ip}",
        "filesystem": filesystem,
        "instance_type": itype.get("name") if isinstance(itype, dict) else None,
        "region": region.get("name") if isinstance(region, dict) else region,
        "ssh_key": (inst.get("ssh_key_names") or [None])[0],
        "price_cents_per_hour": itype.get("price_cents_per_hour") if isinstance(itype, dict) else None,
    }


def _ensure_apps(host: str, persistent_dir: str, app_list: list[str]) -> None:
    """Converge installed apps to the profile: recover what exists, install what's missing."""
    if not app_list:
        return
    output = _remote_run(host, persistent_dir, "status")
    known = json.loads(output).get("apps", {})

    # Recover restores everything already recorded on the filesystem (no re-download).
    if known:
        display.info("Recovering existing apps from persistent storage...")
        result = ssh.ssh_run(host, _remote_cmd(persistent_dir, "recover"), capture=False, timeout=1200)
        if not result.ok:
            display.error(f"Recovery failed (exit code {result.returncode})")
            sys.exit(1)

    # Install any profile app not yet present in state.json.
    for app in [a for a in app_list if a not in known]:
        if app not in AVAILABLE_APPS:
            display.warn(f"Unknown app '{app}' in profile; skipping. Known: {', '.join(AVAILABLE_APPS)}")
            continue
        display.info(f"Installing {app}...")
        result = ssh.ssh_run(
            host, _remote_cmd(persistent_dir, "install", [f"--app {app}"]),
            capture=False, timeout=1200,
        )
        if not result.ok:
            display.error(f"Installing {app} failed (exit code {result.returncode})")
            sys.exit(1)


def _run_provision(profile: dict, host: str, persistent_dir: str) -> bool:
    """Run the profile's provisioning (if any) on the instance.

    The profile folder itself is the payload: it's rsynced to the instance (excluding
    tool state / secrets / VCS) and its entry point (``provision.py`` preferred, else
    ``provision.sh``) runs from inside it on every `up`, with the persistent-dir paths +
    CLOUDGPU_PROVISION_DIR in the environment and cloudgpu/bin on PATH. Output streams
    live; a non-zero exit fails `up`. A profile with no entry point skips provisioning.
    """
    pdir = profile["dir"]
    if (pdir / "provision.py").exists():
        entry = "python3 provision.py"
    elif (pdir / "provision.sh").exists():
        entry = "chmod +x provision.sh && bash provision.sh"
    else:
        return False  # no provisioning in this profile

    display.info("Provisioning...")
    bin_dir = f"{persistent_dir}/cloudgpu/bin"
    remote_dir = f"{persistent_dir}/cloudgpu/provision"
    sync.copy_dir(str(pdir), host, remote_dir, exclude=profiles.PROVISION_EXCLUDES)

    # Secrets (e.g. CIVITAI_TOKEN): transfer the file as content (never on the command
    # line), source it into the environment on the instance, and remove it afterward.
    # It goes to an ephemeral home path (not the persistent filesystem) so it doesn't
    # linger on shared storage.
    secrets_prefix = ""
    if profiles.secrets_file().exists():
        sync.copy_file(str(profiles.secrets_file()), host, ".cloudgpu-secrets.env")
        secrets_prefix = (
            'S="$HOME/.cloudgpu-secrets.env"; chmod 600 "$S"; '
            'trap \'rm -f "$S"\' EXIT; set -a; . "$S"; set +a; '
        )
        display.info("Loaded secrets for provisioning.")

    env = (
        f"CLOUDGPU_PERSISTENT_DIR={persistent_dir} "
        f"CLOUDGPU_APPS_DIR={persistent_dir}/apps "
        f"CLOUDGPU_VENVS_DIR={persistent_dir}/venvs "
        f"CLOUDGPU_BIN_DIR={bin_dir} "
        f"CLOUDGPU_PROVISION_DIR={remote_dir} "
        f"PATH={bin_dir}:$PATH"
    )
    command = f"{secrets_prefix}cd {remote_dir} && export {env} && {entry}"
    result = ssh.ssh_run(host, command, capture=False, timeout=int(profile.get("provision_timeout", 3600)))
    if not result.ok:
        display.error(f"Provision failed (exit code {result.returncode})")
        sys.exit(1)
    display.success("Provision complete.")
    return True


def _restart_services(host: str, app_list: list[str]) -> None:
    """Restart the profile's service apps (so provision changes take effect)."""
    services = apps.service_apps(app_list)
    if not services:
        return
    display.info(f"Restarting service(s): {', '.join(services)} ...")
    cmd = " && ".join(f"sudo systemctl restart {s}" for s in services)
    result = ssh.ssh_run(host, cmd, capture=False, timeout=120)
    if not result.ok:
        display.warn(f"Could not restart services (exit {result.returncode}); they may still be starting.")


def _report_up(name: str, profile: dict, runtime: dict, persistent_dir: str) -> None:
    """Print final status + how to reach the machine + billing reminder."""
    output = _remote_run(runtime["host"], persistent_dir, "status")
    data = json.loads(output)
    display.show_detection(data.get("detection", {}))
    display.show_status(data.get("apps", {}))

    display.success(f"Profile '{name}' is up.")
    display.info(
        f"  GPU: {runtime.get('instance_type') or '?'}   Region: {runtime.get('region') or '?'}"
        f"   IP: {runtime.get('ip') or '?'}   Price: {display.price(runtime.get('price_cents_per_hour'))}"
    )
    display.info(f"  Instance: {runtime.get('instance_id') or '?'}   Filesystem: {runtime.get('filesystem')}")
    ports = apps.app_ports(profile.get("apps", []))
    if ports:
        display.info("Apps run as services on the instance. Reach them:  cloudgpu forward")
        for p in ports:
            display.info(f"  http://localhost:{p}")
    else:
        display.info("Bare machine (no apps). SSH in with:  cloudgpu ssh")
    if runtime.get("instance_id"):
        display.warn(
            "Billing per hour while running. Tear down with 'cloudgpu down' (keeps the "
            "filesystem; 'cloudgpu up' brings it back), or 'cloudgpu down --delete-filesystem' "
            "to remove the data too."
        )


@cli.command()
@click.option("--profile", "-P", "profile_dir", default=None, help="Profile directory (default: current dir)")
def up(profile_dir: str | None) -> None:
    """Converge a profile's machine to its desired state (launch/recover + apps).

    Run from inside a profile folder (one with a cloudgpu.toml). Idempotent: reuses a
    running instance if one exists, otherwise polls for GPU capacity, launches
    (auto-creating the filesystem if needed), sets up, ensures the profile's apps are
    installed, and runs provisioning. Re-run after a termination to recover.
    """
    try:
        d = profiles.find_profile_dir(profile_dir)
        profile = profiles.load_profile(d)
    except config.ConfigError as e:
        raise click.UsageError(str(e)) from e

    name = profile["name"]
    filesystem = profile["filesystem"]

    # Reuse a live instance if we already track one for this profile.
    state = profiles.load_state(d)
    inst = _live_instance(state.get("instance_id"))
    if inst:
        display.info(f"Reusing running instance {state['instance_id']} ({inst.get('ip')}).")
        state["host"] = f"ubuntu@{inst.get('ip')}"
        state["ip"] = inst.get("ip")
    else:
        if state.get("instance_id"):
            display.info("Tracked instance is gone; reconciling...")
            profiles.clear_state(d)
        # Adopt an instance already mounting this filesystem (recovery / lost state).
        # The filesystem is the profile's identity, so it's ours.
        existing = _instance_mounting(filesystem)
        if existing and existing.get("ip"):
            display.info(
                f"Adopting running instance {existing.get('id')} ({existing.get('ip')}) "
                f"already mounting '{filesystem}'."
            )
            state = _runtime_from_instance(existing, filesystem)
            profiles.save_state(d, state)
        else:
            try:
                state = orchestration.acquire_instance(profile)
            except (orchestration.OrchestrationError, lambda_api.LambdaAPIError) as e:
                display.error(str(e))
                sys.exit(1)
            profiles.save_state(d, state)
            if state.get("created_filesystem"):
                display.success(f"Created filesystem '{filesystem}' in {state['region']}.")
            display.success(
                f"Launched {state['instance_type']} in {state['region']} at {state['ip']} "
                f"({display.price(state.get('price_cents_per_hour'))})."
            )

    # Set up the host (SSH, detect, sync) and persist where it landed.
    persistent_dir, _ = _setup_host(state["host"])
    state["persistent_dir"] = persistent_dir
    profiles.save_state(d, state)

    # Ensure the profile's apps are present, run provisioning, then report.
    _ensure_apps(state["host"], persistent_dir, profile.get("apps", []))
    if _run_provision(profile, state["host"], persistent_dir):
        # Restart services so provisioning changes (e.g. custom nodes) take effect.
        _restart_services(state["host"], profile.get("apps", []))
    _report_up(name, profile, state, persistent_dir)


def _find_filesystem(name: str) -> dict | None:
    """Return the account filesystem with this name, or None."""
    try:
        for fs in lambda_api.list_filesystems():
            if fs.get("name") == name:
                return fs
    except lambda_api.LambdaAPIError:
        pass
    return None


def _wait_terminated(instance_id: str, timeout: int = 300, poll: int = 10) -> bool:
    """Poll until the instance is terminated/gone (so its filesystem frees up)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            inst = lambda_api.get_instance(instance_id)
        except lambda_api.LambdaAPIError:
            return True  # gone
        if inst.get("status") == "terminated":
            return True
        time.sleep(poll)
    return False


@cli.command()
@click.option("--profile", "-P", "profile_dir", default=None, help="Profile directory (default: current dir)")
@click.option("--delete-filesystem", is_flag=True, help="Also delete the persistent filesystem (DESTROYS its data)")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompts")
def down(profile_dir: str | None, delete_filesystem: bool, yes: bool) -> None:
    """Terminate a profile's instance. Keeps the filesystem unless --delete-filesystem.

    Run from inside a profile folder. Without --delete-filesystem the data persists, so
    'cloudgpu up' brings the machine back later. With it, the instance is terminated and
    the filesystem (and all its data) is permanently deleted.
    """
    try:
        d = profiles.find_profile_dir(profile_dir)
        profile = profiles.load_profile(d)
    except config.ConfigError as e:
        raise click.UsageError(str(e)) from e

    name = profile["name"]
    filesystem = profile["filesystem"]
    state = profiles.load_state(d)

    # Find the instance: tracked id, or whatever is mounting the filesystem.
    instance_id = state.get("instance_id")
    if not instance_id:
        inst = _instance_mounting(filesystem)
        instance_id = inst.get("id") if inst else None

    # Confirm (a single, appropriately-scoped prompt).
    if not yes:
        if delete_filesystem:
            click.confirm(
                f"Terminate profile '{name}' AND permanently delete filesystem "
                f"'{filesystem}' and all its data?",
                abort=True,
            )
        elif instance_id:
            click.confirm(
                f"Terminate instance {instance_id} for profile '{name}'? "
                f"(filesystem '{filesystem}' is kept)",
                abort=True,
            )

    if instance_id:
        _api_call(lambda_api.terminate_instances, [instance_id])
        display.success(f"Termination requested for {instance_id}.")
    else:
        display.info("No running instance for this profile.")
    profiles.clear_state(d)

    if not delete_filesystem:
        return

    fs = _find_filesystem(filesystem)
    if not fs:
        display.info(f"Filesystem '{filesystem}' not found; nothing to delete.")
        return
    # The filesystem can't be deleted while an instance still mounts it.
    if instance_id:
        display.info("Waiting for the instance to terminate before deleting the filesystem...")
        if not _wait_terminated(instance_id):
            display.error(
                "Instance did not terminate in time. Re-run 'cloudgpu down --delete-filesystem' "
                "once it's gone, or delete it with 'cloudgpu lambda delete-filesystem'."
            )
            sys.exit(1)
    for attempt in range(6):
        try:
            lambda_api.delete_filesystem(fs.get("id"))
            display.success(f"Deleted filesystem '{filesystem}'.")
            return
        except lambda_api.LambdaAPIError as e:
            if attempt == 5:
                display.error(f"Could not delete filesystem '{filesystem}': {e}")
                sys.exit(1)
            display.warn(f"Filesystem still busy ({e}); retrying in 10s...")
            time.sleep(10)


@cli.command()
@click.argument("directory", required=False, default=".")
@click.option("--ssh-key", "ssh_key", required=True, help="Lambda SSH key name (a matching private key must be in ~/.ssh)")
@click.option("--gpu", default="gh200,a100", help="GPU preference order, comma-separated (e.g. gh200,a100)")
@click.option("--apps", "apps_csv", default="", help=f"Comma-separated apps to install (default: none). Known: {', '.join(AVAILABLE_APPS)}")
@click.option("--filesystem", default=None, help="Persistent filesystem name (default: the folder name)")
@click.option("--force", is_flag=True, help="Overwrite an existing cloudgpu.toml")
def init(directory: str, ssh_key: str, gpu: str, apps_csv: str, filesystem: str | None, force: bool) -> None:
    """Scaffold a profile in DIRECTORY (default: current dir).

    Writes cloudgpu.toml + .gitignore. With --apps, also vendors each app's files (e.g.
    comfyui drops comfylib.py + a starter provision.py). With no --apps you get a bare GPU
    machine. Then `cd` into the folder, edit the files, and run `cloudgpu up`.
    """
    gpu_list = [g.strip() for g in gpu.split(",") if g.strip()]
    apps_list = [a.strip() for a in apps_csv.split(",") if a.strip()]
    unknown = [a for a in apps_list if a not in AVAILABLE_APPS]
    if unknown:
        raise click.UsageError(
            f"Unknown app(s): {', '.join(unknown)}. Known: {', '.join(AVAILABLE_APPS)}"
        )
    try:
        toml_path = profiles.scaffold(
            directory, ssh_key=ssh_key, gpu=gpu_list, apps=apps_list,
            filesystem=filesystem, force=force,
        )
    except config.ConfigError as e:
        raise click.UsageError(str(e)) from e

    d = toml_path.parent
    vendored = apps.scaffold_apps(d, apps_list)
    display.success(f"Initialized cloudgpu profile in {d}")
    written = ["cloudgpu.toml", ".gitignore", *vendored]
    display.info("  " + ", ".join(written))
    if not apps_list:
        display.info("  (bare machine — no apps; edit cloudgpu.toml to add some)")
    cd_hint = "" if directory in (".", "") else f"cd {d} && "
    display.info(f"Next: {cd_hint}cloudgpu up")


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
