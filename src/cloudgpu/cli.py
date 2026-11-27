"""Click CLI: init, up, provision, down, setup, install, recover, status, ssh, forward, autoterminate.

A run is composed of a *config* (hardware — which machine to launch; ``configs/<name>.toml``,
loaded via ``cloudgpu.local.configs``) and zero or more *profiles* (provisioning — what to
install; ``profiles/<name>/``, loaded via ``cloudgpu.local.profiles``). The config owns the
instance and its state; profiles are provisioned onto it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import NoReturn

import click
from rich.console import Console
from rich.markup import escape

from cloudgpu.local import apps, capacity, config, configs, display, lambda_api, liveness, orchestration, profiles, ssh, sync

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


def _load_state(config_name: str) -> dict:
    """The config's state, or stop: a state file that can't be used is reported, never read as
    "no instance" (which would let `up` launch a second machine and `down` find nothing)."""
    try:
        return configs.load_state(config_name)
    except configs.StateFileError as e:
        display.error(escape(str(e)))
        sys.exit(1)


def _overdue_for(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes < 1:
        return "under a minute"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m"


def _report_overdue(config_name: str, state: dict) -> bool:
    """Say so on stderr if the instance has outlived its auto-terminate deadline. True if it has.

    The cap is enforced only by a timer on the instance; when that fails, nothing else notices.
    This only reports: it terminates nothing and changes no state. No Lambda call is made unless
    the saved deadline has passed.
    """
    deadline = state.get("auto_terminate_at")
    instance_id = state.get("instance_id")
    if not instance_id or isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
        return False
    late = time.time() - deadline
    if late <= 0:
        return False
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(deadline))
    read = liveness.read_instance(instance_id)
    if read.kind is liveness.Kind.UNKNOWN:
        display.error(
            f"The auto-terminate deadline for config '{escape(config_name)}' passed {_overdue_for(late)} ago "
            f"({when}), and the state of instance {escape(str(instance_id))} couldn't be read: it may still be "
            f"running and billing.\n{escape(read.detail)}\n"
            f"Check it with 'cloudgpu lambda instances', or run 'cloudgpu down -C {escape(config_name)}'."
        )
        return False
    if read.kind is liveness.Kind.GONE or read.instance.get("status") == "terminating":
        return False
    status = str(read.instance.get("status"))
    display.error(
        f"Instance {escape(str(instance_id))} is OVERDUE: its auto-terminate deadline passed "
        f"{_overdue_for(late)} ago ({when}) and it is still '{escape(status)}', "
        f"so the cap on the instance did not work and it is still billing. "
        f"Run 'cloudgpu down -C {escape(config_name)}' to terminate it."
    )
    return True


def _resolve_target(host: str | None = None, config_name: str | None = None) -> tuple[str, str]:
    """Resolve the (host, persistent_dir) to operate on.

    Precedence:
      1. An explicit ``host`` argument/flag (persistent_dir from config.json).
      2. A config (``--config``) — uses its configs/<name>.state.json.
      3. The global single-host config saved by ``cloudgpu setup``.
    """
    if not host and config_name:
        try:
            configs.find_config(config_name)  # validate it exists
        except config.ConfigError as e:
            raise click.UsageError(str(e)) from e
        state = _load_state(config_name)
        if state.get("host") and state.get("persistent_dir"):
            return state["host"], state["persistent_dir"]
        if state.get("instance_id") and not state.get("host"):
            raise click.UsageError(
                f"Config '{config_name}' tracks instance {state['instance_id']}, which isn't set up yet "
                "(it may still be booting, and is billed while it runs). Run "
                f"'cloudgpu up -C {config_name}' to finish, or 'cloudgpu down -C {config_name}' to terminate it."
            )
        raise click.UsageError(
            f"Config '{config_name}' has no running instance yet. Run 'cloudgpu up --config {config_name}' first."
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
    # Refuse a bad test override here, before any command can read its refusal as a failed
    # API call (`up` would take that to mean its tracked instance is gone).
    try:
        lambda_api.api_base_override()
    except lambda_api.LambdaAPIError as e:
        raise click.UsageError(str(e)) from e


def _pick_persistent_dir(mounts: list[str], filesystem: str | None) -> str:
    """Choose the persistent dir from the filesystems mounted under /lambda/nfs.

    With ``filesystem`` (the profile flow) the matching mount is required — installing
    onto any other filesystem would silently strand the profile's data. Without it (the
    manual ``setup`` flow) a single mount is unambiguous; anything else needs
    --filesystem. Exits with a clear message when no safe choice exists.
    """
    if filesystem:
        if filesystem in mounts:
            return f"/lambda/nfs/{filesystem}"
        display.error(f"Filesystem '{filesystem}' is not mounted on this instance.")
        display.info(f"Mounted under /lambda/nfs: {', '.join(mounts) or '(none)'}")
        sys.exit(1)
    if not mounts:
        display.error("No persistent directory found at /lambda/nfs/")
        display.info("Make sure a filesystem is attached to this instance.")
        sys.exit(1)
    if len(mounts) > 1:
        display.error(f"Multiple filesystems mounted: {', '.join(mounts)}.")
        display.info("Pass --filesystem <name> to pick the one to use.")
        sys.exit(1)
    return f"/lambda/nfs/{mounts[0]}"


def _setup_host(host: str, filesystem: str | None = None) -> tuple[str, dict]:
    """SSH-test, detect the persistent dir, sync the remote tool, run full detection.

    ``filesystem`` (when known, e.g. from the profile) names the mount that must back
    the persistent dir. Returns (persistent_dir, detection). Exits the process with a
    clear message on SSH failure or a missing persistent directory. Shared by `setup`
    and `up`.
    """
    # 1. Test SSH
    with console.status("Testing SSH connection..."):
        if not ssh.ssh_test(host):
            display.error(f"Cannot connect to {host} via SSH.")
            display.info("Make sure you can run: ssh " + host)
            sys.exit(1)
    display.success(f"SSH connection to {host} OK")

    # 2. List mounted filesystems (standalone one-liner, before the tool is synced)
    with console.status("Detecting instance environment..."):
        detect_cmd = "python3 -c \"" + (
            "import os, json; "
            "nfs='/lambda/nfs'; "
            "m=sorted(e for e in os.listdir(nfs) "
            "if os.path.isdir(os.path.join(nfs,e)) and not e.startswith('.')) "
            "if os.path.isdir(nfs) else []; "
            "print(json.dumps({'mounts': m}))"
        ) + "\""
        result = ssh.ssh_run(host, detect_cmd, check=True)
        mounts = json.loads(result.stdout).get("mounts") or []

    persistent_dir = _pick_persistent_dir(mounts, filesystem)
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
@click.option("--filesystem", default=None,
              help="Filesystem to use (required if the instance mounts more than one)")
def setup(host: str, filesystem: str | None) -> None:
    """Test SSH, detect persistent dir, sync tool, save config."""
    persistent_dir, _ = _setup_host(host, filesystem)

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
@click.option("--config", "-C", "config_name", default=None, help="Config name (resolves the instance to act on)")
def install(host: str | None, app: str | None, config_name: str | None) -> None:
    """Install a GPU app on the instance."""
    host, persistent_dir = _resolve_target(host, config_name)

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
@click.option("--config", "-C", "config_name", default=None, help="Config name (resolves the instance to act on)")
def recover(host: str | None, config_name: str | None) -> None:
    """Restore everything on a new instance from persistent storage."""
    host, persistent_dir = _resolve_target(host, config_name)

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
        _warn_marks_left(host, persistent_dir, config_name)
    else:
        display.error(f"Recovery failed (exit code {result.returncode})")
        sys.exit(1)


def _warn_marks_left(host: str, persistent_dir: str, config_name: str | None) -> None:
    """After `recover`: name the apps whose venv a rebuild emptied, and the command that
    re-provisions their profiles and clears the mark. `recover` provisions nothing itself."""
    try:
        known = json.loads(_remote_run(host, persistent_dir, "status")).get("apps", {})
    except (ssh.SSHError, ValueError) as e:
        display.warn(f"Couldn't check for apps whose profiles must be provisioned again: {e}")
        return
    marked = _marked_apps(known)
    if not marked:
        return
    command = f"cloudgpu up -C {config_name}" if config_name else "cloudgpu up -C <this instance's config>"
    apps_list = ", ".join(marked)
    display.warn(
        f"Recovery rebuilt the venv of {apps_list}, so what its profiles installed there is gone. "
        f"{apps_list} stays marked for re-provisioning until you run '{command}', which provisions "
        "those profiles again and clears the mark.")


@cli.command()
@click.argument("host", required=False)
@click.option("--config", "-C", "config_name", default=None, help="Config name (resolves the instance to act on)")
def status(host: str | None, config_name: str | None) -> None:
    """Show what's installed and their health. Exits 1 if the instance is past its auto-terminate deadline."""
    explicit_host = host
    host, persistent_dir = _resolve_target(host, config_name)
    # Before the ssh, so it is said even if the instance can't be reached.
    overdue = False
    if config_name and not explicit_host:
        state = _load_state(config_name)
        overdue = bool(_report_overdue(config_name, state))
        _recorded_profiles(state, warn=True)

    with console.status("Syncing remote tool..."):
        sync.sync_remote(host, persistent_dir)

    with console.status("Checking status..."):
        output = _remote_run(host, persistent_dir, "status")
        status_data = json.loads(output)

    display.show_detection(status_data.get("detection", {}))
    display.show_status(status_data.get("apps", {}))
    if "sync" in status_data:
        _show_sync(status_data["sync"])
    if overdue:
        display.error(f"Instance is past its auto-terminate deadline (see above); run 'cloudgpu down -C {config_name}'.")
        sys.exit(1)


# A last success older than this many intervals (or STALE_FLOOR_SECONDS, if longer) is stale.
STALE_INTERVALS = 3
STALE_FLOOR_SECONDS = 15 * 60


def _sync_stale_after(interval_minutes: object) -> int:
    """Seconds after which the last successful sync is stale. An unknown interval: the floor."""
    if isinstance(interval_minutes, bool) or not isinstance(interval_minutes, int) or interval_minutes < 1:
        return STALE_FLOOR_SECONDS
    return max(STALE_INTERVALS * interval_minutes * 60, STALE_FLOOR_SECONDS)


def _show_sync(info: dict) -> None:
    """The sync lines of `status`, from what the instance reported (only sent when installed).
    Stale (older than the threshold, or the timer not active) and never-succeeded each warn."""
    remote = escape(str(info.get("remote") or "an unknown remote"))
    interval = info.get("interval_minutes")
    every = f" every {interval} min" if isinstance(interval, int) else ""
    active = info.get("timer_active") is True
    display.info(f"Sync: to {remote}{every}, timer {'active' if active else '[red]not active[/red]'}")
    last, age = info.get("last_success"), info.get("age_seconds")
    if not isinstance(age, (int, float)) or not last:
        display.info("  Last successful sync: none yet")
        not_active = "The sync timer is not active and there is " if not active else "There is "
        display.warn(f"{not_active}no successful sync yet: work on the instance is only on the instance.")
        return
    ago = f"{_overdue_for(age)} ago"
    display.info(f"  Last successful sync: {escape(str(last))} ({ago})")
    if not active:
        display.warn(f"The sync timer is not active, so nothing more is uploaded; the last successful "
                     f"sync was {ago}: newer work is only on the instance.")
    elif age > _sync_stale_after(interval):
        display.warn(f"The last successful sync was {ago} (stale after "
                     f"{_overdue_for(_sync_stale_after(interval))}): newer work is only on the instance.")


@cli.command(name="ssh", context_settings={"ignore_unknown_options": True})
@click.option("--host", "-H", default=None, help="SSH host (uses saved host if omitted)")
@click.option("--config", "-C", "config_name", default=None, help="Config name (resolves the instance to act on)")
@click.argument("command", nargs=-1, type=click.UNPROCESSED)
def ssh_cmd(host: str | None, config_name: str | None, command: tuple[str, ...]) -> None:
    """Open SSH to instance, optionally running a command.

    Examples:
        cloudgpu ssh -C qwen-image
        cloudgpu ssh -C qwen-image -- comfyui
        cloudgpu ssh -H my-host -- nvidia-smi
    """
    host, persistent_dir = _resolve_target(host, config_name)

    cmd = None
    if command:
        # Prepend the cloudgpu/bin to PATH so launch scripts are available
        bin_dir = f"{persistent_dir}/cloudgpu/bin"
        cmd = f"export PATH={bin_dir}:$PATH && " + " ".join(command)

    exit_code = ssh.ssh_interactive(host, cmd)
    sys.exit(exit_code)


def _recorded_profiles(state: dict, *, warn: bool) -> list[dict]:
    """The profiles the state file records as provisioned, loaded from what was recorded (a name
    under ./profiles, or a path). One that won't load is left out, and with ``warn`` named in a
    warning; it stays recorded, so putting its folder back brings it back."""
    loaded: list[dict] = []
    for record in state.get("provisioned_profiles") or []:
        try:
            loaded.append(profiles.load_profile(record))
        except config.ConfigError as e:
            if warn:
                display.warn(escape(
                    f"Recorded profile '{record}' can't be loaded, so it is left out (its apps' ports, "
                    f"sync paths and re-provisioning): {e}"))
    return loaded


def _provisioned_profiles(config_name: str | None, *, warn: bool = False) -> list[dict]:
    """The profiles provisioned onto this config's instance (unloadable ones left out, and
    named in a warning with ``warn``)."""
    if not config_name:
        return []
    return _recorded_profiles(_load_state(config_name), warn=warn)


def _apps_for_config(config_name: str | None) -> list[str]:
    """Union of apps across the profiles provisioned onto this config's instance."""
    result: list[str] = []
    for profile in _provisioned_profiles(config_name):
        for a in profile.get("apps", []):
            if a not in result:
                result.append(a)
    return result


def _config_ports(config_name: str | None, *, warn: bool = False) -> list[int]:
    """Ports to forward for the config: its apps' ports, plus profile-declared ones.

    A profile can serve something its provision script installed rather than an app
    (imgv, say), so its 'ports' are forwarded alongside the app ports. With ``warn``, a
    recorded profile that won't load is named.
    """
    provisioned = _provisioned_profiles(config_name, warn=warn)
    ports = apps.app_ports([a for p in provisioned for a in p.get("apps", [])])
    for profile in provisioned:
        for port in profile.get("ports", []):
            if port not in ports:
                ports.append(port)
    return ports


@cli.command(name="forward")
@click.option("--host", "-H", default=None, help="SSH host (bypasses the config; requires --port)")
@click.option("--port", "-p", "port", default=None, type=int, help="Forward a specific remote port (default: the config's app ports)")
@click.option("--local-port", "local_port", default=None, type=int, help="Local port (single-port mode; defaults to --port)")
@click.option(
    "--run", "run_cmd", default=None,
    help="Run a command on the instance over the tunnel (single port). Tunnel lives as "
         "long as the command; Ctrl-C stops both.",
)
@click.option("--config", "-C", "config_name", default=None, help="Config name (resolves the instance + app ports)")
def forward(host: str | None, port: int | None, local_port: int | None, run_cmd: str | None, config_name: str | None) -> None:
    """Forward the config's app port(s) to localhost over SSH.

    With no --port, the ports come from the apps provisioned onto the config's instance
    (services run on the instance, so just `cloudgpu forward -C <config>` then open the
    URL). Use --port for an explicit port, or --run to start a command over the tunnel.

    Examples:
        cloudgpu forward -C qwen-image      # forward the config's app ports
        cloudgpu forward -C qwen-image -p 8888
        cloudgpu forward -C qwen-image --run nvidia-smi
    """
    host, persistent_dir = _resolve_target(host, config_name)

    # Single-port mode: explicit --port, or --run (a command needs one port).
    if port is not None or run_cmd:
        if port is None:
            derived = _config_ports(config_name, warn=True)
            if not derived:
                raise click.UsageError("Pass --port <n> (no app port to infer from this config).")
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

    # Config mode: forward every app port in one tunnel.
    ports = _config_ports(config_name, warn=True)
    if not ports:
        raise click.UsageError(
            "No ports to forward for this config. Pass --port <n>, provision a profile "
            f"with a known-port app ({', '.join(AVAILABLE_APPS)}) via 'cloudgpu up', or "
            "declare 'ports' in the profile's cloudgpu.toml."
        )
    ssh_args: list[str] = []
    for p in ports:
        ssh_args += ["-L", f"{p}:localhost:{p}"]
        display.info(f"Forwarding http://localhost:{p} -> {host} port {p}")
    ssh_args.append("-N")
    display.info("Holding the tunnel; press Ctrl-C to close it.")
    sys.exit(ssh.ssh_interactive(host, ssh_args=ssh_args))


# --- profile-driven machine management ------------------------------------


def _stop_unchanged(message: str) -> NoReturn:
    """Stop without touching anything, and say where to look."""
    display.error(
        f"{escape(message)}\nNothing was changed. Look with 'cloudgpu lambda instances'."
    )
    sys.exit(1)


def _stop_not_ready(config_name: str, inst: dict) -> NoReturn:
    """An instance exists but can't be used yet: another launch would bill twice."""
    status = liveness.describe(inst)
    _stop_unchanged(
        f"Instance {inst.get('id')} is {status}, so not launching another. It may still be billing.\n"
        f"'cloudgpu up -C {config_name}' uses it once it's active; 'cloudgpu down -C {config_name}' terminates it."
    )


def _stop_launched(config_name: str, runtime: dict, why: str, code: int = 1) -> NoReturn:
    """A launch went through and `up` can't go on: say what's running and billed, and what to run."""
    display.error(
        f"{escape(why)}\nInstance {runtime.get('instance_id')} ({runtime.get('instance_type')}, "
        f"{runtime.get('region')}) is still running and billed, and is saved for this config. "
        f"'cloudgpu up -C {config_name}' picks it up again; 'cloudgpu down -C {config_name}' terminates it."
    )
    sys.exit(code)


def _forget_gone_launch(config_name: str, before: dict, runtime: dict, why: str) -> NoReturn:
    """The launched instance terminated while booting: put the state back as it was before the launch.

    Lambda said terminated, so nothing bills and there's nothing for `down` to do; keeping the id
    would only make the next `up` read it to learn the same thing.
    """
    if before:
        configs.save_state(config_name, before)
    else:
        configs.clear_state(config_name)
    display.error(
        f"{escape(why)}\nInstance {runtime.get('instance_id')} ({runtime.get('instance_type')}, "
        f"{runtime.get('region')}) is terminated, so nothing is left running from this launch. "
        f"Run 'cloudgpu up -C {config_name}' to try again."
    )
    sys.exit(1)


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


def _recover_recorded(host: str, persistent_dir: str) -> dict:
    """Recover every app the filesystem's state.json records; return what it records.

    Recover restores the recorded apps (no re-download). Nothing recorded: no recover.
    A failed recover fails the command. What's returned is read after the recover, so it carries
    the re-provision marks a venv rebuild left (see _reprovision).
    """
    output = _remote_run(host, persistent_dir, "status")
    known = json.loads(output).get("apps", {})
    if known:
        display.info("Recovering existing apps from persistent storage...")
        result = ssh.ssh_run(host, _remote_cmd(persistent_dir, "recover"), capture=False, timeout=3600)
        if not result.ok:
            display.error(f"Recovery failed (exit code {result.returncode})")
            sys.exit(1)
        known = json.loads(_remote_run(host, persistent_dir, "status")).get("apps", {})
    return known


def _marked_apps(known: dict) -> list[str]:
    """The recorded apps carrying a re-provision mark (their venv was rebuilt)."""
    return [app for app, rec in known.items() if isinstance(rec, dict) and rec.get("reprovision")]


def _reprovision(config_name: str, state: dict, known: dict,
                 profile_list: list[dict]) -> tuple[list[dict], list[str]]:
    """The recorded profiles to provision again because recovery rebuilt an app's venv.

    A rebuild loses what provisioning put into the venv (custom nodes' requirements), so the
    remote records ``reprovision`` on the app until it's cleared. Returns (the recorded profiles
    using a marked app that aren't in ``profile_list`` already, the marked apps whose marks can be
    cleared once those are provisioned). A recorded profile that won't load might use a marked
    app, so then nothing is cleared, and the warning says what to fix.
    """
    marked = _marked_apps(known)
    if not marked:
        return [], []
    requested = {p["dir"] for p in profile_list}
    redo: list[dict] = []
    unloadable: list[str] = []
    for record in state.get("provisioned_profiles") or []:
        try:
            prof = profiles.load_profile(record)
        except config.ConfigError:
            unloadable.append(record)  # named, with the reason, by `up` and `provision` already
            continue
        if prof["dir"] not in requested and any(a in marked for a in prof.get("apps", [])):
            requested.add(prof["dir"])
            redo.append(prof)
    apps_list = ", ".join(marked)
    if redo:
        display.warn(f"Recovery rebuilt the venv of {apps_list}; provisioning "
                     f"{', '.join(p['name'] for p in redo)} again to put back what it installed.")
    if unloadable:
        display.warn(
            f"Recovery rebuilt the venv of {apps_list}, and recorded profile(s) "
            f"{escape(', '.join(unloadable))} can't be loaded, so they can't be provisioned again. "
            f"Fix them and run 'cloudgpu up -C {config_name} -P {escape(','.join(unloadable))}'; "
            "until then every 'up' asks again.")
        return redo, []
    return redo, marked


def _clear_reprovision(host: str, persistent_dir: str, marked: list[str]) -> None:
    """Drop the marks, now the profiles using those apps were provisioned again."""
    for app in marked:
        result = ssh.ssh_run(host, _remote_cmd(persistent_dir, "clear-reprovision", [f"--app {app}"]),
                             timeout=120)
        if not result.ok:
            display.warn(f"Couldn't clear {app}'s re-provision mark (exit {result.returncode}); "
                         "the next 'up' provisions its profiles again.")


def _ensure_apps(host: str, persistent_dir: str, app_list: list[str], known: dict | None = None) -> None:
    """Converge installed apps to the profile: recover what exists, install what's missing.

    ``known`` is what an earlier ``_recover_recorded`` returned this run (``up`` recovers on
    every run); given it, recovery isn't repeated.
    """
    if not app_list:
        return
    if known is None:
        known = _recover_recorded(host, persistent_dir)
    _install_missing(host, persistent_dir, app_list, known)


def _install_missing(host: str, persistent_dir: str, app_list: list[str], known: dict) -> None:
    """Install each app in ``app_list`` that the filesystem (``known``) doesn't record yet.

    Safe to run again: an app whose install finished is recorded, so it isn't installed twice.
    """
    for app in [a for a in app_list if a not in known]:
        if app not in AVAILABLE_APPS:
            display.warn(f"Unknown app '{app}' in profile; skipping. Known: {', '.join(AVAILABLE_APPS)}")
            continue
        display.info(f"Installing {app}...")
        result = ssh.ssh_run(
            host, _remote_cmd(persistent_dir, "install", [f"--app {app}"]),
            capture=False, timeout=3600,
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

    The entry point stays ubuntu, with every secret: it's the person's own script. Anything
    it runs from a venv or app dir must go through CLOUDGPU_RUN_AS_APP (remote/run_as_app.py).
    """
    pdir = profile["dir"]
    if (pdir / "provision.py").exists():
        entry = "python3 provision.py"
    elif (pdir / "provision.sh").exists():
        entry = "chmod +x provision.sh && bash provision.sh"
    else:
        return False  # no provisioning in this profile

    display.info(f"Provisioning '{profile['name']}'...")
    bin_dir = f"{persistent_dir}/cloudgpu/bin"
    # Per-profile remote dir so multiple profiles on one instance don't collide.
    remote_dir = f"{persistent_dir}/cloudgpu/provision/{profile['name']}"
    sync.copy_dir(str(pdir), host, remote_dir, exclude=profiles.PROVISION_EXCLUDES)

    # Secrets (e.g. CIVITAI_TOKEN): transfer the file as content (never on the command
    # line), source it into the environment on the instance, and delete it right after
    # sourcing — before the provision entry point runs (and so before it, or the
    # service restart that follows, can start any app service). The EXIT trap is only
    # a backstop if sourcing itself fails. It goes to an ephemeral home path (not the
    # persistent filesystem) so it doesn't linger on shared storage.
    secrets_prefix = ""
    if profiles.secrets_file().exists():
        sync.copy_file(str(profiles.secrets_file()), host, ".cloudgpu-secrets.env")
        secrets_prefix = (
            'S="$HOME/.cloudgpu-secrets.env"; chmod 600 "$S"; '
            'trap \'rm -f "$S"\' EXIT; set -a; . "$S"; set +a; rm -f "$S"; '
        )
        display.info("Loaded secrets for provisioning.")

    env = (
        f"CLOUDGPU_PERSISTENT_DIR={persistent_dir} "
        f"CLOUDGPU_APPS_DIR={persistent_dir}/apps "
        f"CLOUDGPU_VENVS_DIR={persistent_dir}/venvs "
        f"CLOUDGPU_BIN_DIR={bin_dir} "
        f"CLOUDGPU_PROVISION_DIR={remote_dir} "
        f"CLOUDGPU_RUN_AS_APP={persistent_dir}/cloudgpu/remote/run_as_app.py "
        f"PATH={bin_dir}:$PATH"
    )
    # umask 002: files provisioning creates in the (setgid, cloudgpu-app group) app dirs
    # stay group-writable, so the unprivileged app services can still write them.
    command = f"{secrets_prefix}umask 002 && cd {remote_dir} && export {env} && {entry}"
    result = ssh.ssh_run(host, command, capture=False, timeout=int(profile.get("provision_timeout", 3600)))
    if not result.ok:
        display.error(f"Provision failed (exit code {result.returncode})")
        sys.exit(1)
    display.success("Provision complete.")
    return True


def _configure_auto_terminate(cfg: dict, host: str, instance_id: str | None, persistent_dir: str) -> int | None:
    """Converge the instance's self-terminate timer to the config's auto_terminate_hours.

    With hours > 0 the instance gets a root-only systemd timer that terminates it via
    the Lambda API once the deadline passes — a billing cap that holds even if this
    machine never runs 'cloudgpu down'. Each 'up' re-arms, so the deadline is always
    auto_terminate_hours after the latest 'up'; hours <= 0 disarms. Returns the deadline
    epoch when armed, else None. A failure fails 'up' loudly: a cap the user asked for
    must never be silently absent.
    """
    hours = float(cfg.get("auto_terminate_hours") or 0)
    try:
        if hours <= 0:
            _remote_run(host, persistent_dir, "autoterminate", ["--hours 0"])
            return None
        if not instance_id:
            display.warn(
                "auto_terminate_hours is set but this instance's id is unknown; "
                "skipping the auto-terminate guard."
            )
            return None
        api_key = os.environ.get(lambda_api.API_KEY_ENV)
        if not api_key:
            display.warn(
                f"auto_terminate_hours is set but {lambda_api.API_KEY_ENV} is not; "
                "skipping the auto-terminate guard."
            )
            return None
        # The key travels as file content (over the rsync stream), never on a command
        # line; the remote side moves it to root-only storage and deletes this copy.
        fd, tmp = tempfile.mkstemp(prefix="cloudgpu-lambda-", suffix=".env")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(f"LAMBDA_API_KEY={api_key}\n")
            sync.copy_file(tmp, host, ".cloudgpu-lambda.env")
        finally:
            os.remove(tmp)
        display.info(f"Arming auto-terminate ({hours:g}h)...")
        _remote_run(
            host, persistent_dir, "autoterminate",
            [f"--hours {hours:g}", f"--instance-id {instance_id}",
             "--key-file .cloudgpu-lambda.env"],
        )
        return int(time.time() + hours * 3600)
    except ssh.SSHError as e:
        display.error(f"Could not configure auto-terminate: {e}")
        sys.exit(1)


# Where the two files of a sync setup land in ubuntu's home; the remote setup deletes both.
SYNC_KEY_TRANSFER = ".cloudgpu-rclone.conf"
SYNC_REQUEST_TRANSFER = ".cloudgpu-datasync.json"


def _plan_sync(cfg: dict, state: dict, profile_list: list[dict]) -> dict | None:
    """What this `up` will ask the instance to sync, or None without ``[sync]``. Local only.

    Run before `up` launches or changes anything: with ``[sync]``, the bucket key must be in the
    person's rclone config under the remote's name, and every path must stay inside the
    persistent dir (the instance would refuse it anyway, after the launch). The paths are
    ``apps.sync_paths`` for the config, this `up`'s profiles and the ones the state file records
    as already provisioned, and those profiles' apps. Nothing from the key file is kept or shown.
    """
    sync_cfg = cfg.get("sync")
    if not sync_cfg:
        return None
    try:
        profiles.rclone_remote_section(sync_cfg["remote"])
    except config.ConfigError as e:
        display.error(escape(f"{e} Nothing was launched or changed."))
        sys.exit(1)
    # A recorded profile that won't load was named at the start of `up`.
    in_play = _each_once(list(profile_list) + _recorded_profiles(state, warn=False))
    paths = apps.sync_paths(cfg, _app_union(in_play), in_play)
    for path in paths:
        if path.startswith(("/", "~")) or ".." in path.split("/"):
            display.error(escape(
                f"The sync path '{path}' must be a relative path inside the persistent dir "
                "(no leading '/', no '~', no '..'). Nothing was launched or changed."
            ))
            sys.exit(1)
    return {"remote": sync_cfg["remote"], "paths": paths, "interval_minutes": sync_cfg["interval_minutes"]}


def _stop_sync_failed(config_name: str, runtime: dict, what: str, e: Exception) -> NoReturn:
    """Sync the person asked for must never be silently absent: stop, and say what is billing."""
    display.error(
        f"{what}: {escape(str(e))}\n"
        f"Instance {runtime.get('instance_id') or '?'} is still running and billing. "
        f"'cloudgpu down -C {config_name}' stops it; 'cloudgpu up -C {config_name}' tries again."
    )
    sys.exit(1)


def _install_sync(config_name: str, runtime: dict, persistent_dir: str, plan: dict) -> None:
    """Put the bucket key, the path list and the sync units in place on the instance. Starts nothing.

    Only the remote's own section of the person's rclone config travels, as file content over
    the rsync stream (never a command line or an environment variable); the remote setup moves
    it to root-only storage and deletes the transferred copy whether or not it worked. The local
    temporary copy is mode 600 in a private temp dir and is removed whatever happens.
    """
    host = runtime["host"]
    display.info(f"Setting up sync to {escape(plan['remote'])} ({len(plan['paths'])} paths, "
                 f"every {plan['interval_minutes']} minutes)...")
    try:
        with tempfile.TemporaryDirectory(prefix="cloudgpu-sync-") as tmp:
            key_path = os.path.join(tmp, "rclone.conf")
            fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(profiles.rclone_remote_section(plan["remote"]))
            request_path = os.path.join(tmp, "datasync.json")
            with open(request_path, "w") as f:
                json.dump(plan, f)
            sync.copy_file(request_path, host, SYNC_REQUEST_TRANSFER)
            sync.copy_file(key_path, host, SYNC_KEY_TRANSFER)
        _remote_run(host, persistent_dir, "datasync",
                    ["setup", f"--key-file {SYNC_KEY_TRANSFER}", f"--spec-file {SYNC_REQUEST_TRANSFER}"])
    except (ssh.SSHError, subprocess.SubprocessError, OSError, config.ConfigError) as e:
        _stop_sync_failed(config_name, runtime, "Could not set up the bucket sync", e)


# A restore of real datasets can take hours; the instance bounds it (datasync_setup's
# RESTORE_WAIT_SECONDS: the six-hour run limit plus five minutes), and this waits a little longer.
RESTORE_SSH_TIMEOUT = 6 * 60 * 60 + 600


def _restore_from_bucket(config_name: str, runtime: dict, persistent_dir: str, plan: dict) -> None:
    """Bring the declared paths down from the bucket if this filesystem has not been restored
    (the instance decides, and says so). Before the timer and before any service starts.

    What the remote tool says goes to stderr; it is shown here, on success and on failure,
    in place of the ssh command line. The bucket key is never in it: the remote tool passes
    rclone the key file's path only.
    """
    remote = escape(plan["remote"])
    display.info(f"Restoring from {remote} if this filesystem has not been restored yet "
                 "(a first restore of large files can take a long time)...")
    try:
        result = ssh.ssh_run(runtime["host"], _remote_cmd(persistent_dir, "datasync", ["restore"]),
                             timeout=RESTORE_SSH_TIMEOUT)
    except (ssh.SSHError, subprocess.SubprocessError, OSError) as e:
        _stop_sync_failed(config_name, runtime, f"Could not restore from {plan['remote']}", e)
    said = [line.removeprefix(">>> ").removeprefix("error: ").strip()
            for line in result.stderr.splitlines() if line.strip()]
    if not result.ok:
        why = "\n".join(said) or f"the remote restore exited {result.returncode} and said nothing"
        _stop_sync_failed(config_name, runtime, f"Could not restore from {plan['remote']}",
                          RuntimeError(f"{why}\nNo service was started and the sync timer was not started."))
    for line in said:
        display.info(escape(line))


def _start_sync_timer(config_name: str, runtime: dict, persistent_dir: str) -> None:
    """Start (or keep running) the sync timer that `_install_sync` put in place."""
    try:
        _remote_run(runtime["host"], persistent_dir, "datasync", ["start"])
    except (ssh.SSHError, subprocess.SubprocessError, OSError) as e:
        _stop_sync_failed(config_name, runtime, "Could not start the bucket sync timer", e)
    display.info("Sync timer running.")


def _remove_sync(config_name: str, runtime: dict, persistent_dir: str) -> None:
    """The config has no ``[sync]``: take timer, units and key off the instance (quiet if none)."""
    try:
        _remote_run(runtime["host"], persistent_dir, "datasync", ["remove"])
    except (ssh.SSHError, subprocess.SubprocessError, OSError) as e:
        _stop_sync_failed(config_name, runtime, "Could not remove the bucket sync", e)


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


def _verify_services(config_name: str, runtime: dict, persistent_dir: str, services: list[str],
                     *, settled: bool) -> None:
    """Stop `up` unless every expected service is running, as systemd on the instance reports it.

    The remote `verify` reads systemd (it never goes by the restart's exit code) with the test
    install and recover use: active, running, and not restarted across a settle. ``settled``:
    recovery confirmed these same services a moment ago and `up` hasn't restarted them since, so
    one look is enough; after `up`'s own restart (even a failed one) they get the full settle.
    A check that can't be made is a failure. Changes nothing: not the state file, not the instance.
    """
    if not services:
        display.info("No service apps on this machine, so nothing to verify.")
        return
    names = ", ".join(services)
    display.info(f"Checking that {names} {'is' if len(services) == 1 else 'are'} running...")
    args = [f"--service {s}" for s in services] + (["--settle 0"] if settled else [])
    try:
        result = ssh.ssh_run(runtime["host"], _remote_cmd(persistent_dir, "verify", args), timeout=300)
        if not result.ok:
            raise ValueError(f"the check exited {result.returncode}: {result.stderr.strip()[-500:]}")
        reported = {s["name"]: s for s in json.loads(result.stdout)["services"]}
    except (ValueError, KeyError, TypeError, OSError, ssh.subprocess.SubprocessError) as e:
        display.error(f"Couldn't check whether {names} {'is' if len(services) == 1 else 'are'} "
                      f"running, and unknown is not running: {escape(str(e) or type(e).__name__)}")
        _stop_not_verified(config_name, runtime)
    unknown = {"ok": False, "active": "unknown", "sub": "unknown", "restarts": "unknown",
               "journal": "(the check didn't report this service)"}
    failed = [s for s in services if reported.get(s, unknown).get("ok") is not True]
    if not failed:
        display.info(f"  running: {names}")
        return
    for s in failed:
        r = {**unknown, **reported.get(s, {})}
        display.error(
            f"Service '{s}' is not running: {escape(str(r['active']))}/{escape(str(r['sub']))}, "
            f"restarts: {escape(str(r['restarts']))}. Last journal lines:\n{escape(str(r['journal']))}"
        )
    display.error(f"Config '{escape(config_name)}' is not up: {len(failed)} of {len(services)} "
                  f"expected service(s) not running ({', '.join(failed)}).")
    _stop_not_verified(config_name, runtime)


def _stop_not_verified(config_name: str, runtime: dict) -> NoReturn:
    """Verify failed: nothing is forgotten or terminated, so say what is still billing."""
    display.error(
        f"Instance {runtime.get('instance_id') or '?'} is still running and billing. "
        f"'cloudgpu down -C {config_name}' stops it; 'cloudgpu up -C {config_name}' tries again."
    )
    sys.exit(1)


def _report_up(config_name: str, runtime: dict, persistent_dir: str, app_list: list[str]) -> None:
    """Print final status + how to reach the machine + billing reminder."""
    output = _remote_run(runtime["host"], persistent_dir, "status")
    data = json.loads(output)
    display.show_detection(data.get("detection", {}))
    display.show_status(data.get("apps", {}))

    provisioned = runtime.get("provisioned_profiles") or []
    prof_note = f" (profiles: {escape(', '.join(provisioned))})" if provisioned else ""
    display.success(f"Config '{config_name}' is up{prof_note}.")
    display.info(
        f"  GPU: {runtime.get('instance_type') or '?'}   Region: {runtime.get('region') or '?'}"
        f"   IP: {runtime.get('ip') or '?'}   Price: {display.price(runtime.get('price_cents_per_hour'))}"
    )
    display.info(f"  Instance: {runtime.get('instance_id') or '?'}   Filesystem: {runtime.get('filesystem')}")
    ports = apps.app_ports(app_list)
    if ports:
        display.info(f"Apps run as services on the instance. Reach them:  cloudgpu forward -C {config_name}")
        for p in ports:
            display.info(f"  http://localhost:{p}")
    else:
        display.info(f"Bare machine (no apps). SSH in with:  cloudgpu ssh -C {config_name}")
    # Profile-declared ports have no service behind them — they're forwarded for
    # whatever the profile installs and you start yourself (e.g. imgv).
    extra = [p for p in _config_ports(config_name) if p not in ports]
    if extra:
        display.info(
            f"Profile ports forwarded on demand ({', '.join(str(p) for p in extra)}):  "
            f"cloudgpu forward -C {config_name}"
        )
    if runtime.get("instance_id"):
        display.warn(
            "Billing per hour while running. Tear down with 'cloudgpu down' (keeps the "
            "filesystem; 'cloudgpu up' brings it back), or 'cloudgpu down --delete-filesystem' "
            "to remove the data too."
        )
        if runtime.get("auto_terminate_at"):
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(runtime["auto_terminate_at"]))
            display.info(f"  Auto-terminate: {when} ('cloudgpu autoterminate -C {config_name}' to reset/extend)")
        else:
            display.info(
                "  Tip: set auto_terminate_hours in the config to cap billing "
                "if you forget 'cloudgpu down'."
            )


def _parse_profiles(profile_csv: str | None) -> list[dict]:
    """Load each profile in a comma-separated list; UsageError on any bad name."""
    names = [p.strip() for p in (profile_csv or "").split(",") if p.strip()]
    loaded = []
    for n in names:
        try:
            loaded.append(profiles.load_profile(n))
        except config.ConfigError as e:
            raise click.UsageError(str(e)) from e
    return loaded


def _config_profiles(cfg: dict) -> list[dict]:
    """Load each profile the config lists; UsageError naming the profile and the config file."""
    loaded = []
    for n in cfg.get("profiles") or []:
        try:
            loaded.append(profiles.load_profile(n))
        except config.ConfigError as e:
            raise click.UsageError(
                f"{cfg['path']}: can't load profile '{n}' listed in 'profiles': {e}"
            ) from e
    return loaded


def _each_once(profile_list: list[dict]) -> list[dict]:
    """The profiles in order, without a later one that is the same folder as an earlier one."""
    seen: set = set()
    once = []
    for prof in profile_list:
        if prof["dir"] not in seen:
            seen.add(prof["dir"])
            once.append(prof)
    return once


def _app_union(profile_list: list[dict]) -> list[str]:
    """Every app the profiles name, each once, in the order first named."""
    app_union: list[str] = []
    for prof in profile_list:
        for a in prof.get("apps", []):
            if a not in app_union:
                app_union.append(a)
    return app_union


def _provision_profiles(config_name: str, state: dict, host: str, persistent_dir: str,
                        profile_list: list[dict], known: dict | None = None) -> list[str]:
    """Ensure apps + run provisioning for each profile, record them in state.

    Returns the union of the profiles' apps (for reporting / port forwarding). The union
    is installed once (so `recover` runs a single time), then each profile's payload is
    provisioned into its own remote dir. Each profile's record (its name, or its resolved path
    when given by a path outside ./profiles) is merged into state['provisioned_profiles']. ``known``: the apps already recovered this run, if any.
    """
    app_union = _app_union(profile_list)

    _ensure_apps(host, persistent_dir, app_union, known)
    for prof in profile_list:
        _run_provision(prof, host, persistent_dir)
    _restart_services(host, app_union)

    provisioned = list(state.get("provisioned_profiles") or [])
    for prof in profile_list:
        if profiles.record_of(prof) not in provisioned:
            provisioned.append(profiles.record_of(prof))
    state["provisioned_profiles"] = provisioned
    configs.save_state(config_name, state)
    return app_union


@cli.command()
@click.option("--config", "-C", "config_name", required=True, help="Config name (hardware: configs/<name>.toml)")
@click.option("--profile", "-P", "profile_csv", default=None, help="Profile(s) to provision (comma-separated; optional)")
def up(config_name: str, profile_csv: str | None) -> None:
    """Launch/recover the config's machine, then provision its profile(s).

    The config (hardware) is required. Every `up` provisions the profiles the config lists
    (`profiles = [...]`), then any given with -P (comma-separated) that aren't among them;
    with neither you get a bare instance. Idempotent: reuses a running
    instance if one exists, otherwise polls for GPU capacity, launches (auto-creating the
    filesystem if needed), sets up, and provisions. Re-run after a termination to recover.
    """
    try:
        cfg = configs.load_config(config_name)
    except config.ConfigError as e:
        raise click.UsageError(str(e)) from e
    # All loaded here, before the lock and any Lambda call: a typo must not launch a machine.
    profile_list = _each_once(_config_profiles(cfg) + _parse_profiles(profile_csv))

    # One `up` per config at a time on this computer, so two can't both launch. Refuse rather
    # than wait: the other may be polling for capacity for hours.
    try:
        lock = configs.hold_up_lock(config_name)
    except configs.UpLocked as e:
        display.error(
            f"Another 'cloudgpu up -C {escape(config_name)}' is already running on this computer "
            f"(it holds {escape(str(e.path))}), so not starting a second one. "
            "Let it finish, or stop it first."
        )
        sys.exit(1)
    with lock:
        _up_locked(config_name, cfg, profile_list)


def _waits_for_boot(inst: dict) -> bool:
    """Still coming up: booting, or active with no ip yet. Worth waiting on, not launching beside."""
    return inst.get("status") == "booting" or (inst.get("status") == "active" and not inst.get("ip"))


def _wait_for_booting(config_name: str, cfg: dict, inst: dict) -> dict:
    """Wait, bounded by the config's boot_minutes, for an instance on this filesystem to boot.

    Never launches. On timeout or failure stop, and say what is still billing.
    """
    inst_id = inst.get("id")
    poll = int(cfg.get("poll_seconds", 20))
    minutes = float(cfg.get("boot_minutes", 20))
    display.info(
        f"Instance {inst_id} on '{escape(cfg['filesystem'])}' is {liveness.describe(inst)}; "
        f"waiting up to {minutes:g} minutes for it rather than launching another..."
    )
    try:
        return {"id": inst_id, **orchestration.wait_active(inst_id, poll=poll, max_minutes=minutes)}
    except orchestration.BootFailed as e:
        if e.gone:
            _stop_unchanged(f"Instance {inst_id} {e}, so it is no longer billing. Run 'cloudgpu up -C {config_name}' to launch.")
        why = str(e)
    except (lambda_api.LambdaAPIError, lambda_api.LambdaReplyLost) as e:
        why = f"couldn't keep checking it: {e}"
    _stop_unchanged(
        f"Waiting for instance {inst_id} ({liveness.describe(inst)}) to boot: {why}. Not launching another.\n"
        f"Instance {inst_id} may still be running and billing. 'cloudgpu up -C {config_name}' waits for it "
        f"again; 'cloudgpu down -C {config_name}' terminates it."
    )


# The steps every `up` runs, in this order, whichever way it found its machine (plans/v2.md,
# Phase 2). Each is safe to run again, so an `up` stopped anywhere is finished by running it
# again.
UP_STEPS = (
    "find or launch the instance",
    "set up the instance",
    "arm the cap",
    "make apps match",
    "provision",
    "restart services",
    "verify",
    "report",
)


def _step(name: str) -> None:
    """Say which of UP_STEPS starts now."""
    display.info(f"[bold]Step {UP_STEPS.index(name) + 1}/{len(UP_STEPS)}: {name}[/bold]")


def _up_locked(config_name: str, cfg: dict, profile_list: list[dict]) -> None:
    filesystem = cfg["filesystem"]
    # Read before anything else: a state file that can't be used stops `up` here.
    state = _load_state(config_name)
    _report_overdue(config_name, state)  # reports only; this `up` is about to set a new deadline
    _recorded_profiles(state, warn=True)  # named once here; later steps leave them out quietly
    # Local only, and before any launch or ssh: a missing bucket key stops `up` here.
    sync_plan = _plan_sync(cfg, state, profile_list)

    _step("find or launch the instance")
    state = _find_or_launch(config_name, cfg, state)
    host = state["host"]

    # The config's filesystem must back the persistent dir, never whatever else is mounted.
    _step("set up the instance")
    persistent_dir, _ = _setup_host(host, filesystem)
    state["persistent_dir"] = persistent_dir
    configs.save_state(config_name, state)

    # Before the slow steps, so the cap holds even if an install or provisioning hangs.
    _step("arm the cap")
    deadline = _configure_auto_terminate(cfg, host, state.get("instance_id"), persistent_dir)
    if deadline is not None:
        state["auto_terminate_at"] = deadline
    else:
        state.pop("auto_terminate_at", None)
    configs.save_state(config_name, state)
    # Same step, after the cap: like it, the sync lives on the instance's own disk, so every
    # `up` puts it back (or takes it away). The restore goes between install and start: no
    # upload runs while files arrive, and no service has started yet (recovery is the next step).
    if sync_plan:
        _install_sync(config_name, state, persistent_dir, sync_plan)
        _restore_from_bucket(config_name, state, persistent_dir, sync_plan)
        _start_sync_timer(config_name, state, persistent_dir)
    else:
        _remove_sync(config_name, state, persistent_dir)

    # Recover what the filesystem records (once per `up`; the remote restarts and confirms each
    # recorded service), then install what this `up`'s profiles need and it lacks. This `up`'s
    # profiles are the requested ones plus the recorded ones a venv rebuild emptied.
    _step("make apps match")
    known = _recover_recorded(host, persistent_dir)
    redo, marked = _reprovision(config_name, state, known, profile_list)
    to_provision = profile_list + redo
    app_union = _app_union(to_provision)
    _install_missing(host, persistent_dir, app_union, known)

    # The only step that may have nothing to do. Each profile is recorded as soon as its own
    # provisioning succeeded, so an `up` stopped later still knows it is on the filesystem.
    _step("provision")
    if not to_provision:
        display.info("Nothing to provision.")
    for prof in to_provision:
        _run_provision(prof, host, persistent_dir)
        record = profiles.record_of(prof)
        if record not in (state.get("provisioned_profiles") or []):
            state["provisioned_profiles"] = [*(state.get("provisioned_profiles") or []), record]
        configs.save_state(config_name, state)

    # Everything on the machine (the recorded apps and this `up`'s), but only when something was
    # provisioned: recovery restarted and confirmed the recorded services a moment ago, and a
    # second restart would be unchecked. A re-provision mark goes only after all of this.
    _step("restart services")
    services = apps.service_apps(list(dict.fromkeys([*known, *app_union])))
    if not services:
        display.info("No service apps on this machine, so nothing to restart.")
    elif to_provision:
        _restart_services(host, services)
    else:
        display.info(
            f"Nothing was provisioned, so not restarting {', '.join(services)} again: "
            "recovery already restarted and confirmed them."
        )
    _clear_reprovision(host, persistent_dir, marked)

    # Success is what systemd says now, not what the restart's exit code said. Only an `up`
    # that restarted nothing skips the settle wait: recovery's own confirm just waited it.
    _step("verify")
    _verify_services(config_name, state, persistent_dir, services, settled=not to_provision)

    _step("report")
    _report_up(config_name, state, persistent_dir, _apps_for_config(config_name))


def _find_or_launch(config_name: str, cfg: dict, state: dict) -> dict:
    """The config's instance, as the state to carry on with: reused, adopted or launched.

    Reuse a live instance if one is tracked for this config. The state file stays as it is until
    something replaces it: every way of stopping here keeps it, and what it records. Run again
    after an `up` that stopped later, this finds the instance that `up` recorded and launches
    nothing.
    """
    filesystem = cfg["filesystem"]
    tracked = state.get("instance_id")
    read = liveness.read_instance(tracked) if tracked else None
    if read and read.kind is liveness.Kind.UNKNOWN:
        _stop_unchanged(read.detail)
    if read and read.kind is liveness.Kind.ALIVE:
        inst = {"id": tracked, **read.instance}
        if not liveness.usable(inst):
            if not _waits_for_boot(inst):
                _stop_not_ready(config_name, inst)
            inst = _wait_for_booting(config_name, cfg, inst)
        ip = inst.get("ip")
        display.info(f"Reusing running instance {tracked} ({ip}).")
        state["host"] = f"ubuntu@{ip}"
        state["ip"] = ip
    else:
        provisioned = state.get("provisioned_profiles")
        if tracked:
            display.info("Tracked instance is gone; reconciling...")
        # Adopt an instance already mounting this filesystem (recovery / lost state).
        # The filesystem is the config's identity, so it's ours.
        mounting = liveness.find_mounting(filesystem)
        if mounting.kind is liveness.Kind.UNKNOWN:
            _stop_unchanged(mounting.detail)
        if mounting.kind is liveness.Kind.ALIVE:
            existing = mounting.instance
            if not liveness.usable(existing):
                if not _waits_for_boot(existing):
                    _stop_not_ready(config_name, existing)
                existing = _wait_for_booting(config_name, cfg, existing)
            display.info(
                f"Adopting running instance {existing.get('id')} ({existing.get('ip')}) "
                f"already mounting '{filesystem}'."
            )
            state = _runtime_from_instance(existing, filesystem)
        else:
            launched: dict = {}

            def save_launched(runtime: dict) -> None:
                # Written down before the boot wait, so nothing can leave it billing untracked.
                launched.update(runtime)
                early = dict(runtime)
                if provisioned:
                    early["provisioned_profiles"] = provisioned
                configs.save_state(config_name, early)

            try:
                state = orchestration.acquire_instance(cfg, on_launched=save_launched)
            except KeyboardInterrupt:
                if launched:
                    _stop_launched(config_name, launched, "Interrupted while waiting for the instance to boot.", 130)
                raise
            except orchestration.BootFailed as e:
                if e.gone:
                    _forget_gone_launch(config_name, state, launched, str(e))
                _stop_launched(config_name, launched, str(e))
            except lambda_api.LambdaReplyLost as e:
                if launched:
                    _stop_launched(config_name, launched, str(e))
                display.error(
                    f"{escape(str(e))}\nNot trying again: if that was a launch, it may have gone through, and "
                    f"an instance may be running and billing on filesystem '{escape(filesystem)}'. Check with "
                    f"'cloudgpu lambda instances'. Once it shows, 'cloudgpu up -C {config_name}' adopts it and "
                    f"'cloudgpu down -C {config_name}' terminates it."
                )
                sys.exit(1)
            except (orchestration.OrchestrationError, lambda_api.LambdaAPIError) as e:
                if launched:
                    _stop_launched(config_name, launched, str(e))
                display.error(str(e))
                sys.exit(1)
            if state.get("created_filesystem"):
                display.success(f"Created filesystem '{filesystem}' in {state['region']}.")
            display.success(
                f"Launched {state['instance_type']} in {state['region']} at {state['ip']} "
                f"({display.price(state.get('price_cents_per_hour'))})."
            )
        # Carry forward the set of profiles provisioned onto this filesystem.
        if provisioned:
            state["provisioned_profiles"] = provisioned
        configs.save_state(config_name, state)

    return state


@cli.command()
@click.option("--config", "-C", "config_name", required=True, help="Config name (the running instance to provision)")
@click.option("--profile", "-P", "profile_csv", required=True, help="Profile(s) to provision (comma-separated)")
def provision(config_name: str, profile_csv: str) -> None:
    """Provision profile(s) onto a config's already-running instance.

    Installs each profile's apps (recovering existing ones first) and runs its
    provisioning payload, without relaunching the machine. Use this to add a profile to
    a box that's already up, or to re-run a profile's provisioning.
    """
    try:
        configs.load_config(config_name)  # validate the config exists
    except config.ConfigError as e:
        raise click.UsageError(str(e)) from e
    profile_list = _parse_profiles(profile_csv)
    if not profile_list:
        raise click.UsageError("Pass at least one profile with --profile.")

    state = _load_state(config_name)
    host, persistent_dir = state.get("host"), state.get("persistent_dir")
    if not host or not persistent_dir:
        raise click.UsageError(
            f"Config '{config_name}' has no running instance. Run "
            f"'cloudgpu up --config {config_name}' first."
        )

    with console.status("Syncing remote tool..."):
        sync.sync_remote(host, persistent_dir)
    # As `up` does: also provision the recorded profiles that use an app a venv rebuild marked,
    # and clear the marks only once all of them were provisioned again.
    _recorded_profiles(state, warn=True)
    known = _recover_recorded(host, persistent_dir)
    redo, marked = _reprovision(config_name, state, known, profile_list)
    app_list = _provision_profiles(config_name, state, host, persistent_dir, profile_list + redo, known)
    _clear_reprovision(host, persistent_dir, marked)
    names = ", ".join(p["name"] for p in profile_list + redo)
    display.success(f"Provisioned {names} onto '{config_name}'.")
    ports = apps.app_ports(app_list)
    for p in ports:
        display.info(f"  http://localhost:{p}   (cloudgpu forward -C {config_name})")


@cli.command()
@click.argument("hours", required=False, default=None, type=float)
@click.option("--off", is_flag=True, help="Disarm the timer entirely")
@click.option("--config", "-C", "config_name", required=True, help="Config name (the running instance to re-arm)")
def autoterminate(hours: float | None, off: bool, config_name: str) -> None:
    """Reset the instance's auto-terminate timer without a full 'up'.

    With no HOURS, re-arms to the config's auto_terminate_hours from now — a quick
    way to buy more time before the deadline (e.g. mid LoRA training). Pass HOURS
    to override this once, or --off to disarm the cap.

    \b
    Examples:
        cloudgpu autoterminate -C qwen-lora        # reset: config hours from now
        cloudgpu autoterminate 6 -C qwen-lora      # 6 hours from now, just this once
        cloudgpu autoterminate --off -C qwen-lora  # disarm the cap
    """
    if off and hours is not None:
        raise click.UsageError("Pass either HOURS or --off, not both.")
    try:
        cfg = configs.load_config(config_name)
    except config.ConfigError as e:
        raise click.UsageError(str(e)) from e
    state = _load_state(config_name)
    if not state.get("host") or not state.get("persistent_dir"):
        raise click.UsageError(f"No tracked instance for config '{config_name}'; run 'cloudgpu up' first.")
    _report_overdue(config_name, state)  # reports only; a new deadline is about to be set

    if off:
        hours = 0.0
    elif hours is None:
        hours = float(cfg.get("auto_terminate_hours") or 0)
        if hours <= 0:
            raise click.UsageError(
                "auto_terminate_hours is not set in the config; pass HOURS "
                "explicitly (or --off to disarm)."
            )

    deadline = _configure_auto_terminate(
        {**cfg, "auto_terminate_hours": hours},
        state["host"], state.get("instance_id"), state["persistent_dir"],
    )
    if deadline is not None:
        state["auto_terminate_at"] = deadline
        configs.save_state(config_name, state)
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(deadline))
        display.success(f"Auto-terminate armed: instance terminates at {when} ({hours:g}h from now).")
    elif hours > 0:
        # _configure_auto_terminate already explained why arming was skipped
        # (missing instance id or API key); an asked-for cap must not fail quietly.
        sys.exit(1)
    else:
        state.pop("auto_terminate_at", None)
        configs.save_state(config_name, state)
        display.success("Auto-terminate disarmed.")


# How long plain `down` waits for Lambda to show the instance terminating or gone.
DOWN_CONFIRM_SECONDS = 60
DOWN_CONFIRM_POLL_SECONDS = 5


def _find_filesystem(config_name: str, name: str) -> dict | None:
    """Return the account filesystem with this name, or None if the list doesn't have it.

    A lookup that fails stops: it must not read as "no filesystem", which would leave one billing.
    """
    try:
        return orchestration.filesystem_record(name)
    except (lambda_api.LambdaAPIError, lambda_api.LambdaReplyLost) as e:
        display.error(
            f"Couldn't look up filesystem '{escape(name)}': {escape(str(e))}\n"
            f"Filesystem '{escape(name)}' was not deleted and may still be billing. "
            f"Run 'cloudgpu down -C {config_name} --delete-filesystem' again to delete it."
        )
        sys.exit(1)


def _wait_terminated(instance_id: str, timeout: float = 300, poll: float = 10, *, going: bool = False) -> bool:
    """Poll until the instance is terminated/gone (so its filesystem frees up).

    This is the step before deleting a filesystem, so a read that can't tell keeps polling.
    With `going`, 'terminating' is enough: that is what `down` needs before it forgets the instance.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        read = liveness.read_instance(instance_id)
        if read.kind is liveness.Kind.GONE:
            return True
        if going and read.kind is liveness.Kind.ALIVE and read.instance.get("status") == "terminating":
            return True
        if read.kind is liveness.Kind.UNKNOWN:
            display.warn(escape(read.detail))
        time.sleep(poll)
    return False


# The instance bounds a final sync (datasync_setup.FINAL_WAIT_SECONDS, the six-hour run limit
# plus five minutes, per wait; a run in progress is waited out first, so two waits worst case);
# this waits a little longer, so the instance's own answer arrives first.
FINAL_SYNC_SSH_TIMEOUT = 2 * (6 * 60 * 60 + 300) + 600


def _final_sync(config_name: str, sync_cfg: dict, host: str | None,
                persistent_dir: str | None) -> tuple[bool, str]:
    """Ask the instance for one fresh upload run before termination (`datasync final`).

    Returns (True, "") on success, else (False, one sentence on why, with the last successful
    sync time when the instance gave one). Nothing is re-sent: what the last `up` installed is
    synced. The bucket key is in no command or output here; the instance names only its time.
    Ctrl-C is not caught here (the caller says what was left).
    """
    remote = escape(sync_cfg["remote"])
    display.info(f"Final sync to {remote} before terminating "
                 "(waits for a run in progress and one more full run)...")
    if not host or not persistent_dir:
        return False, "This config's state has no address for the instance, so it could not be reached."
    try:
        result = ssh.ssh_run(host, _remote_cmd(persistent_dir, "datasync", ["final"]),
                             timeout=FINAL_SYNC_SSH_TIMEOUT)
    except subprocess.TimeoutExpired:
        return False, (f"The final sync did not answer within {FINAL_SYNC_SSH_TIMEOUT // 3600} hours; "
                       "the sync timer on the instance may be stopped.")
    except (ssh.SSHError, subprocess.SubprocessError, OSError) as e:
        return False, f"Could not reach the instance over ssh: {e}"
    said = [line.removeprefix(">>> ").removeprefix("error: ").strip()
            for line in result.stderr.splitlines() if line.strip()]
    if result.ok:
        for line in said:
            display.info(escape(line))
        display.success(f"Final sync to {remote} done.")
        return True, ""
    if result.returncode == 255:
        return False, "Could not reach the instance over ssh" + (f": {said[-1]}" if said else ".")
    return False, " ".join(said) or f"The remote final sync exited {result.returncode} and said nothing."


@cli.command()
@click.option("--config", "-C", "config_name", required=True, help="Config name (the instance to tear down)")
@click.option("--delete-filesystem", is_flag=True, help="Also delete the persistent filesystem (DESTROYS its data)")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompts")
@click.option("--skip-sync", is_flag=True,
              help="With [sync]: terminate without the final sync (work since the last sync is lost)")
def down(config_name: str, delete_filesystem: bool, yes: bool, skip_sync: bool) -> None:
    """Terminate a config's instance. Keeps the filesystem unless --delete-filesystem.

    Without --delete-filesystem the data persists, so 'cloudgpu up' brings the machine
    back later. With it, the instance is terminated and the filesystem (and all its data)
    is permanently deleted. With [sync] in the config, a final sync to the bucket runs
    first; if it fails, nothing is terminated unless you say so (with -y: --skip-sync).
    """
    try:
        cfg = configs.load_config(config_name)
    except config.ConfigError as e:
        raise click.UsageError(str(e)) from e

    name = cfg["name"]
    filesystem = cfg["filesystem"]
    state = _load_state(config_name)
    _report_overdue(config_name, state)  # reports only; this `down` is about to fix it

    # Find the instance: tracked id, or whatever is mounting the filesystem.
    instance_id = state.get("instance_id")
    already_terminating = False
    already_gone = False
    found: dict = {}
    if instance_id:
        # Read it first: terminating an id Lambda has lost fails, and would keep the stale id forever.
        read = liveness.read_instance(instance_id)
        if read.kind is liveness.Kind.UNKNOWN:
            _stop_unchanged(read.detail)
        already_gone = read.kind is liveness.Kind.GONE
        if read.kind is liveness.Kind.ALIVE:
            found = read.instance or {}
    else:
        mounting = liveness.find_mounting(filesystem)
        if mounting.kind is liveness.Kind.UNKNOWN:
            _stop_unchanged(mounting.detail)
        if mounting.kind is liveness.Kind.ALIVE:
            found = mounting.instance or {}
            instance_id = found.get("id")
            already_terminating = found.get("status") == "terminating"

    # Confirm (a single, appropriately-scoped prompt).
    if not yes:
        if delete_filesystem:
            click.confirm(
                f"Terminate config '{name}' AND permanently delete filesystem "
                f"'{filesystem}' and all its data?",
                abort=True,
            )
        elif instance_id and not already_terminating and not already_gone:
            click.confirm(
                f"Terminate instance {instance_id} for config '{name}'? "
                f"(filesystem '{filesystem}' is kept)",
                abort=True,
            )

    # The final sync: after the confirmation, before anything is terminated or cleared.
    sync_cfg = cfg.get("sync")
    if sync_cfg:
        if not instance_id or already_gone or already_terminating:
            display.info("No instance to sync from, so no final sync was possible.")
        elif skip_sync:
            display.warn("Final sync skipped (--skip-sync): work since the last sync is not in the bucket.")
        else:
            host = state.get("host") or (f"ubuntu@{found['ip']}" if found.get("ip") else None)
            persistent_dir = state.get("persistent_dir") or f"/lambda/nfs/{filesystem}"
            try:
                ok, why = _final_sync(config_name, sync_cfg, host, persistent_dir)
            except KeyboardInterrupt:
                display.error(
                    f"Interrupted during the final sync: nothing was terminated or deleted, instance "
                    f"{instance_id} is still running and billing, and config '{config_name}' still tracks it "
                    f"(its sync timer may be stopped until the next 'cloudgpu up -C {config_name}')."
                )
                sys.exit(130)
            if not ok:
                display.error(escape(f"The final sync to {sync_cfg['remote']} failed: {why}"))
                still = (f"Nothing was terminated or deleted. Instance {instance_id} is still running and billing; "
                         f"config '{config_name}' still tracks it.")
                if yes:
                    display.error(f"{still} 'cloudgpu down -C {config_name} -y --skip-sync' terminates it "
                                  "without a final sync.")
                    sys.exit(1)
                if not click.confirm("Terminate anyway, without the work since the last successful sync?",
                                     default=False):
                    display.error(f"{still} 'cloudgpu down -C {config_name}' tries again; "
                                  f"'--skip-sync' terminates it without a final sync.")
                    sys.exit(1)

    if already_gone:
        display.info(f"Instance {instance_id} is already gone; clearing it from this config.")
    elif already_terminating:
        display.info(f"Instance {instance_id} is already terminating.")
    elif instance_id:
        _api_call(lambda_api.terminate_instances, [instance_id])
        display.success(f"Termination requested for {instance_id}.")
        # Forget the instance only once Lambda shows it going: a request that didn't take
        # would otherwise leave a billing instance this config no longer knows about.
        pace = lambda_api.pace()
        if not _wait_terminated(instance_id, timeout=DOWN_CONFIRM_SECONDS / pace, poll=DOWN_CONFIRM_POLL_SECONDS / pace,
                                going=True):
            display.error(
                f"Lambda accepted the request, but instance {instance_id} is still listed as running after "
                f"{DOWN_CONFIRM_SECONDS}s and may still be billing. Nothing was cleared: config '{config_name}' "
                f"still tracks it. Run 'cloudgpu down -C {config_name}' again, or check it with "
                "'cloudgpu lambda instances'."
            )
            sys.exit(1)
    else:
        display.info("No running instance for this config.")
    configs.clear_state(config_name)

    if not delete_filesystem:
        return

    fs = _find_filesystem(config_name, filesystem)
    if not fs:
        display.info(f"Filesystem '{filesystem}' not found; nothing to delete.")
        return
    # The filesystem can't be deleted while an instance still mounts it.
    if instance_id and not already_gone:
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
@click.argument("name")
@click.option("--config", "as_config", is_flag=True, help="Scaffold a hardware config (configs/<name>.toml) instead of a profile")
@click.option("--ssh-key", "ssh_key", default=None, help="[config] Lambda SSH key name (a matching private key must be in ~/.ssh)")
@click.option("--gpu", default="gh200,a100", help="[config] GPU preference order, comma-separated (e.g. gh200,a100)")
@click.option("--filesystem", default=None, help="[config] Persistent filesystem name (default: the config name)")
@click.option("--apps", "apps_csv", default="", help=f"[profile] Comma-separated apps to install. Known: {', '.join(AVAILABLE_APPS)}")
@click.option("--force", is_flag=True, help="Overwrite an existing file")
def init(name: str, as_config: bool, ssh_key: str | None, gpu: str, filesystem: str | None,
         apps_csv: str, force: bool) -> None:
    """Scaffold a config (hardware) or a profile (provisioning).

    \b
    Config:   cloudgpu init --config qwen-image --ssh-key mini --gpu gh200,a100
    Profile:  cloudgpu init generate --apps comfyui

    A config writes configs/<name>.toml (the machine to launch). A profile writes
    profiles/<name>/cloudgpu.toml (+ vendored app files, e.g. comfyui's comfylib.py).
    Then: cloudgpu up --config <config> --profile <profile>.
    """
    if as_config:
        if not ssh_key:
            raise click.UsageError("--ssh-key is required for --config.")
        gpu_list = [g.strip() for g in gpu.split(",") if g.strip()]
        try:
            toml_path = configs.scaffold(
                name, ssh_key=ssh_key, gpu=gpu_list, filesystem=filesystem, force=force,
            )
        except config.ConfigError as e:
            raise click.UsageError(str(e)) from e
        display.success(f"Initialized cloudgpu config: {toml_path}")
        display.info(f"Next: cloudgpu up --config {name} --profile <profile>")
        return

    apps_list = [a.strip() for a in apps_csv.split(",") if a.strip()]
    unknown = [a for a in apps_list if a not in AVAILABLE_APPS]
    if unknown:
        raise click.UsageError(
            f"Unknown app(s): {', '.join(unknown)}. Known: {', '.join(AVAILABLE_APPS)}"
        )
    d = profiles.profile_dir(name)
    try:
        profiles.scaffold(d, apps=apps_list, force=force)
    except config.ConfigError as e:
        raise click.UsageError(str(e)) from e

    vendored = apps.scaffold_apps(d, apps_list)
    display.success(f"Initialized cloudgpu profile in {d}")
    written = ["cloudgpu.toml", ".gitignore", *vendored]
    display.info("  " + ", ".join(written))
    if not apps_list:
        display.info("  (no apps; edit cloudgpu.toml to add some)")
    display.info(f"Next: cloudgpu up --config <config> --profile {name}")


def _api_call(fn, *args, **kwargs):
    """Run a Lambda API call, surfacing errors as a clean message and exit."""
    try:
        return fn(*args, **kwargs)
    except (lambda_api.LambdaAPIError, lambda_api.LambdaReplyLost) as e:
        display.error(str(e))
        sys.exit(1)


@cli.group(name="capacity")
def capacity_group() -> None:
    """Record GPU capacity over time (needs LAMBDA_API_KEY)."""
    pass


@capacity_group.command(name="log")
def capacity_log() -> None:
    """Take one capacity sample and append it to the log. A failure records nothing."""
    try:
        path = capacity.append_sample(_api_call(lambda_api.list_instance_types))
    except (ValueError, OSError) as e:
        display.error(f"No capacity sample recorded: {e}")
        sys.exit(1)
    display.info(f"Recorded a capacity sample in {path}")


def _filesystems_or_exit() -> list[dict]:
    """The account's filesystems. A failed lookup must stop the report, not read as no home region."""
    try:
        filesystems = _api_call(lambda_api.list_filesystems)
        if not isinstance(filesystems, list) or not all(isinstance(f, dict) for f in filesystems):
            raise ValueError("the response is not a list of filesystems")
    except (ValueError, OSError) as e:
        display.error(f"No capacity report: could not look up filesystems: {e}")
        sys.exit(1)
    return filesystems


@capacity_group.command(name="report")
def capacity_report() -> None:
    """Show how often each config in ./configs had capacity, from the logged samples."""
    path = capacity.log_path()
    try:
        samples, skipped = capacity.read_samples(path)
    except OSError as e:
        display.error(f"No capacity report: could not read {path}: {e}")
        sys.exit(1)
    if not samples:
        display.error(f"No capacity samples in {path} ({skipped} unreadable lines skipped).")
        sys.exit(1)

    config_dir = Path.cwd() / configs.CONFIGS_DIR
    config_paths = sorted(config_dir.glob(f"*{configs.CONFIG_SUFFIX}"))
    if not config_paths:
        display.error(f"No capacity report: no configs in {config_dir}.")
        sys.exit(1)

    reports: list[dict] = []
    loaded: list[dict] = []
    for config_path in config_paths:
        try:
            loaded.append(configs.load_config(config_path))
        except config.ConfigError as e:
            reports.append({"name": config_path.stem, "gpu": [], "targets": [], "problem": str(e)})
    # Only a config with no region of its own needs the API, and so the key.
    filesystems = _filesystems_or_exit() if any(not c.get("region") for c in loaded) else []

    for cfg in loaded:
        targets = orchestration.expand_targets(cfg["gpu"])
        report = {"name": cfg["name"], "gpu": cfg["gpu"], "targets": targets}
        home = orchestration.home_region(cfg, filesystems)
        if home is None:
            report["problem"] = (
                f"it sets no region and no filesystem named '{cfg['filesystem']}' has one, "
                "so it has no home region yet."
            )
        else:
            report["home"] = home
            report["home_source"] = (
                "set in the config" if cfg.get("region") else f"its filesystem '{cfg['filesystem']}'"
            )
            report.update(capacity.summarize(samples, targets, home))
        reports.append(report)
    reports.sort(key=lambda r: r["name"])

    display.show_capacity_report(path, samples, skipped, reports)
    if all("problem" in r for r in reports):
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
