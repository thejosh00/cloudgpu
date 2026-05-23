"""Click CLI: profile, up, down, setup, install, recover, status, ssh, forward."""

from __future__ import annotations

import json
import sys
import time

import click
from rich.console import Console

from cloudgpu.local import config, display, lambda_api, orchestration, profiles, ssh, sync

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


def _resolve_target(host: str | None = None, profile: str | None = None) -> tuple[str, str]:
    """Resolve the (host, persistent_dir) to operate on.

    Precedence:
      1. An explicit ``host`` argument/flag (persistent_dir from config.json).
      2. A profile (``--profile`` or the active profile) with saved runtime state.
      3. The global single-host config saved by ``cloudgpu setup``.
    """
    if not host:
        name = profile or profiles.get_active()
        if name:
            runtime = profiles.load_runtime(name)
            if runtime.get("host") and runtime.get("persistent_dir"):
                return runtime["host"], runtime["persistent_dir"]
            raise click.UsageError(
                f"Profile '{name}' has no running instance yet. Run 'cloudgpu up' first."
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
@click.option("--profile", "-P", "profile", default=None, help="Profile to target (default: active)")
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
@click.option("--profile", "-P", "profile", default=None, help="Profile to target (default: active)")
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
@click.option("--profile", "-P", "profile", default=None, help="Profile to target (default: active)")
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
@click.option("--profile", "-P", "profile", default=None, help="Profile to target (default: active)")
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


@cli.command(name="forward")
@click.option("--host", "-H", default=None, help="SSH host (uses saved host if omitted)")
@click.option("--port", "-p", default=8188, type=int, help="Remote port to forward (default 8188, ComfyUI)")
@click.option("--local-port", "local_port", default=None, type=int, help="Local port (defaults to --port)")
@click.option(
    "--run", "run_cmd", is_flag=False, flag_value="comfyui",
    help="Start a command on the instance over the same tunnel (default: comfyui). "
         "The tunnel lives as long as the command; Ctrl-C stops both. "
         "Omitting --run holds the tunnel only.",
)
@click.option("--profile", "-P", "profile", default=None, help="Profile to target (default: active)")
def forward(host: str | None, port: int, local_port: int | None, run_cmd: str | None, profile: str | None) -> None:
    """Forward a remote port to localhost over SSH (default: ComfyUI on 8188).

    By default this just holds the tunnel (Ctrl-C to close); start the server
    separately with `cloudgpu ssh -- comfyui`. With --run, a single command both
    starts the server and tunnels it.

    Examples:
        cloudgpu forward                    # hold localhost:8188 -> instance:8188
        cloudgpu forward --run              # start comfyui AND tunnel it
        cloudgpu forward --run nvidia-smi   # run something else over the tunnel
        cloudgpu forward -p 8888            # forward a different port
        cloudgpu forward --local-port 9000  # serve locally on 9000
    """
    host, persistent_dir = _resolve_target(host, profile)
    local_port = local_port or port
    spec = f"{local_port}:localhost:{port}"

    display.info(f"Forwarding http://localhost:{local_port} -> {host} port {port}")
    if run_cmd:
        # Run the command over the forwarded connection: PATH includes the
        # generated launch scripts (comfyui, etc.). Tunnel == command lifetime.
        bin_dir = f"{persistent_dir}/cloudgpu/bin"
        command = f"export PATH={bin_dir}:$PATH && {run_cmd}"
        display.info(f"Starting '{run_cmd}' on the instance (Ctrl-C stops it and closes the tunnel)...")
        exit_code = ssh.ssh_interactive(host, command, ssh_args=["-L", spec])
    else:
        display.info("Holding the tunnel; press Ctrl-C to close it.")
        exit_code = ssh.ssh_interactive(host, ssh_args=["-L", spec, "-N"])
    sys.exit(exit_code)


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


def _ensure_apps(host: str, persistent_dir: str, apps: list[str]) -> None:
    """Converge installed apps to the profile: recover what exists, install what's missing."""
    if not apps:
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
    for app in [a for a in apps if a not in known]:
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


def _run_provision(profile: dict, host: str, persistent_dir: str) -> None:
    """Run the profile's provisioning (if any) on the instance.

    Convention: ~/.config/cloudgpu/profiles/<name>.provision/ — a directory holding a
    ``provision.sh`` entry point plus any files it needs (workflows, configs, ...). The
    whole directory is rsynced to the instance and ``provision.sh`` runs from inside it
    on every `up`, with the persistent-dir paths + CLOUDGPU_PROVISION_DIR in the
    environment and cloudgpu/bin on PATH. Output streams live; a non-zero exit fails `up`.
    """
    pdir = profiles.provision_dir(profile["name"])
    if not pdir.is_dir():
        return
    if not (pdir / "provision.sh").exists():
        display.error(f"Provision dir {pdir} is missing its 'provision.sh' entry point.")
        sys.exit(1)

    display.info(f"Provisioning from {pdir.name}/ ...")
    bin_dir = f"{persistent_dir}/cloudgpu/bin"
    remote_dir = f"{persistent_dir}/cloudgpu/provision"
    sync.copy_dir(str(pdir), host, remote_dir)

    env = (
        f"CLOUDGPU_PERSISTENT_DIR={persistent_dir} "
        f"CLOUDGPU_APPS_DIR={persistent_dir}/apps "
        f"CLOUDGPU_VENVS_DIR={persistent_dir}/venvs "
        f"CLOUDGPU_BIN_DIR={bin_dir} "
        f"CLOUDGPU_PROVISION_DIR={remote_dir} "
        f"PATH={bin_dir}:$PATH"
    )
    command = f"cd {remote_dir} && export {env} && chmod +x provision.sh && bash provision.sh"
    result = ssh.ssh_run(host, command, capture=False, timeout=int(profile.get("provision_timeout", 3600)))
    if not result.ok:
        display.error(f"Provision failed (exit code {result.returncode})")
        sys.exit(1)
    display.success("Provision complete.")


def _report_up(name: str, runtime: dict, persistent_dir: str) -> None:
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
    display.info("Reach a UI app over SSH, e.g.:  cloudgpu forward --run comfyui  (then open http://localhost:8188)")
    if runtime.get("instance_id"):
        display.warn(
            "Billing per hour while running. Tear down with 'cloudgpu down' (keeps the "
            "filesystem; 'cloudgpu up' brings it back), or 'cloudgpu down --delete-filesystem' "
            "to remove the data too."
        )


@cli.command()
@click.option("--profile", "-P", "profile_name", default=None, help="Profile to bring up (default: active)")
def up(profile_name: str | None) -> None:
    """Converge a profile's machine to its desired state (launch/recover + apps).

    Idempotent: reuses a running instance if one exists, otherwise polls for GPU
    capacity, launches (auto-creating the filesystem if needed), sets up, and ensures
    the profile's apps are installed. Re-run after a termination to recover.
    """
    try:
        name = profiles.require_profile(profile_name)
        profile = profiles.load_profile(name)
    except config.ConfigError as e:
        raise click.UsageError(str(e)) from e

    filesystem = profile["filesystem"]

    # Reuse a live instance if we already track one for this profile.
    runtime = profiles.load_runtime(name)
    inst = _live_instance(runtime.get("instance_id"))
    if inst:
        display.info(f"Reusing running instance {runtime['instance_id']} ({inst.get('ip')}).")
        runtime["host"] = f"ubuntu@{inst.get('ip')}"
        runtime["ip"] = inst.get("ip")
    else:
        if runtime.get("instance_id"):
            display.info("Tracked instance is gone; reconciling...")
            profiles.clear_runtime(name)
        # Adopt an instance already mounting this filesystem (recovery / migration /
        # lost runtime state). The filesystem is the profile's identity, so it's ours.
        existing = _instance_mounting(filesystem)
        if existing and existing.get("ip"):
            display.info(
                f"Adopting running instance {existing.get('id')} ({existing.get('ip')}) "
                f"already mounting '{filesystem}'."
            )
            runtime = _runtime_from_instance(existing, filesystem)
            profiles.save_runtime(name, runtime)
        else:
            try:
                runtime = orchestration.acquire_instance(profile)
            except (orchestration.OrchestrationError, lambda_api.LambdaAPIError) as e:
                display.error(str(e))
                sys.exit(1)
            profiles.save_runtime(name, runtime)
            if runtime.get("created_filesystem"):
                display.success(f"Created filesystem '{filesystem}' in {runtime['region']}.")
            display.success(
                f"Launched {runtime['instance_type']} in {runtime['region']} at {runtime['ip']} "
                f"({display.price(runtime.get('price_cents_per_hour'))})."
            )

    # Set up the host (SSH, detect, sync) and persist where it landed.
    persistent_dir, _ = _setup_host(runtime["host"])
    runtime["persistent_dir"] = persistent_dir
    profiles.save_runtime(name, runtime)
    profiles.set_active(name)

    # Ensure the profile's apps are present, run provisioning, then report.
    _ensure_apps(runtime["host"], persistent_dir, profile.get("apps", []))
    _run_provision(profile, runtime["host"], persistent_dir)
    _report_up(name, runtime, persistent_dir)


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
@click.option("--profile", "-P", "profile_name", default=None, help="Profile to tear down (default: active)")
@click.option("--delete-filesystem", is_flag=True, help="Also delete the persistent filesystem (DESTROYS its data)")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompts")
def down(profile_name: str | None, delete_filesystem: bool, yes: bool) -> None:
    """Terminate a profile's instance. Keeps the filesystem unless --delete-filesystem.

    Without --delete-filesystem the data persists, so 'cloudgpu up' brings the machine
    back later. With it, the instance is terminated and the filesystem (and all its data)
    is permanently deleted.
    """
    try:
        name = profiles.require_profile(profile_name)
        profile = profiles.load_profile(name)
    except config.ConfigError as e:
        raise click.UsageError(str(e)) from e

    filesystem = profile["filesystem"]
    runtime = profiles.load_runtime(name)

    # Find the instance: tracked id, or whatever is mounting the filesystem.
    instance_id = runtime.get("instance_id")
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
    profiles.clear_runtime(name)

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


def _profile_or_active(name: str | None) -> str:
    try:
        return profiles.require_profile(name)
    except config.ConfigError as e:
        raise click.UsageError(str(e)) from e


@cli.group(name="profile")
def profile_group() -> None:
    """Manage machine profiles (declarative desired state)."""


@profile_group.command(name="create")
@click.argument("name")
@click.option("--filesystem", default=None, help="Persistent filesystem name (default: the profile name; auto-created on first 'up')")
@click.option("--gpu", default="gh200,a100", help="GPU preference order, comma-separated (e.g. gh200,a100)")
@click.option("--apps", default="comfyui", help="Comma-separated apps to keep installed")
@click.option("--ssh-key", "ssh_key", required=True, help="Lambda SSH key name (a matching private key must be in ~/.ssh)")
@click.option("--force", is_flag=True, help="Overwrite an existing profile")
@click.option("--activate/--no-activate", default=True, help="Select this profile as active")
def profile_create(name, filesystem, gpu, apps, ssh_key, force, activate):
    """Scaffold a new profile TOML."""
    gpu_list = [g.strip() for g in gpu.split(",") if g.strip()]
    apps_list = [a.strip() for a in apps.split(",") if a.strip()]
    try:
        path = profiles.create_profile(
            name, filesystem=filesystem, gpu=gpu_list, apps=apps_list,
            ssh_key=ssh_key, overwrite=force,
        )
    except config.ConfigError as e:
        raise click.UsageError(str(e)) from e
    display.success(f"Created profile '{name}' at {path}")
    if activate:
        profiles.set_active(name)
        display.info(f"Selected '{name}' as the active profile.")
    display.info("Edit it with 'cloudgpu profile edit', then run 'cloudgpu up'.")


@profile_group.command(name="list")
def profile_list():
    """List profiles (the active one is marked *)."""
    rows = []
    for n in profiles.list_profiles():
        try:
            p = profiles.load_profile(n)
            fs, gpu = p["filesystem"], p["gpu"]
        except config.ConfigError:
            fs, gpu = "[invalid]", []
        rows.append({
            "name": n,
            "filesystem": fs,
            "gpu": gpu,
            "last_ip": profiles.load_runtime(n).get("ip"),
        })
    display.show_profiles(rows, profiles.get_active())


@profile_group.command(name="show")
@click.argument("name", required=False)
def profile_show(name):
    """Print a profile's TOML and its runtime state."""
    name = _profile_or_active(name)
    try:
        display.info(profiles.profile_path(name).read_text().rstrip())
    except OSError as e:
        raise click.UsageError(str(e)) from e
    runtime = profiles.load_runtime(name)
    if runtime:
        display.info("\n[bold]runtime[/bold]:")
        display.info(json.dumps(runtime, indent=2))


@profile_group.command(name="use")
@click.argument("name")
def profile_use(name):
    """Select NAME as the active profile."""
    try:
        profiles.set_active(name)
    except config.ConfigError as e:
        raise click.UsageError(str(e)) from e
    display.success(f"Active profile: {name}")


@profile_group.command(name="edit")
@click.argument("name", required=False)
def profile_edit(name):
    """Open a profile's TOML in $EDITOR."""
    name = _profile_or_active(name)
    path = profiles.profile_path(name)
    if not path.exists():
        raise click.UsageError(f"No profile named '{name}'.")
    click.edit(filename=str(path))


@profile_group.command(name="delete")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
def profile_delete(name, yes):
    """Delete a profile and its runtime state (does NOT touch the filesystem/instance)."""
    if not profiles.profile_path(name).exists():
        raise click.UsageError(f"No profile named '{name}'.")
    if not yes:
        click.confirm(f"Delete profile '{name}' (and its runtime state)?", abort=True)
    profiles.delete_profile(name)
    display.success(f"Deleted profile '{name}'.")


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
