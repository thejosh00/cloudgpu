"""Click CLI: init, up, provision, down, setup, install, recover, status, ssh, forward, autoterminate.

A run is composed of a *config* (hardware — which machine to launch; ``configs/<name>.toml``,
loaded via ``cloudgpu.local.configs``) and zero or more *profiles* (provisioning — what to
install; ``profiles/<name>/``, loaded via ``cloudgpu.local.profiles``). The config owns the
instance and its state; profiles are provisioned onto it.
"""

from __future__ import annotations

import json
import os
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
        state = configs.load_state(config_name)
        if state.get("host") and state.get("persistent_dir"):
            return state["host"], state["persistent_dir"]
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
    else:
        display.error(f"Recovery failed (exit code {result.returncode})")
        sys.exit(1)


@cli.command()
@click.argument("host", required=False)
@click.option("--config", "-C", "config_name", default=None, help="Config name (resolves the instance to act on)")
def status(host: str | None, config_name: str | None) -> None:
    """Show what's installed and their health."""
    host, persistent_dir = _resolve_target(host, config_name)

    with console.status("Syncing remote tool..."):
        sync.sync_remote(host, persistent_dir)

    with console.status("Checking status..."):
        output = _remote_run(host, persistent_dir, "status")
        status_data = json.loads(output)

    display.show_detection(status_data.get("detection", {}))
    display.show_status(status_data.get("apps", {}))


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


def _provisioned_profiles(config_name: str | None) -> list[dict]:
    """The profiles provisioned onto this config's instance (unloadable ones skipped)."""
    if not config_name:
        return []
    state = configs.load_state(config_name)
    loaded: list[dict] = []
    for pname in state.get("provisioned_profiles", []):
        try:
            loaded.append(profiles.load_profile(pname))
        except config.ConfigError:
            continue
    return loaded


def _apps_for_config(config_name: str | None) -> list[str]:
    """Union of apps across the profiles provisioned onto this config's instance."""
    result: list[str] = []
    for profile in _provisioned_profiles(config_name):
        for a in profile.get("apps", []):
            if a not in result:
                result.append(a)
    return result


def _config_ports(config_name: str | None) -> list[int]:
    """Ports to forward for the config: its apps' ports, plus profile-declared ones.

    A profile can serve something its provision script installed rather than an app
    (imgv, say), so its 'ports' are forwarded alongside the app ports.
    """
    provisioned = _provisioned_profiles(config_name)
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
            derived = _config_ports(config_name)
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
    ports = _config_ports(config_name)
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


def _stop_not_ready(inst: dict) -> NoReturn:
    """An instance exists but can't be used yet: another launch would bill twice."""
    status = liveness.describe(inst)
    _stop_unchanged(f"Instance {inst.get('id')} is {status}, so not launching another. Run this again once it's active.")


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
        result = ssh.ssh_run(host, _remote_cmd(persistent_dir, "recover"), capture=False, timeout=3600)
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


def _report_up(config_name: str, runtime: dict, persistent_dir: str, app_list: list[str]) -> None:
    """Print final status + how to reach the machine + billing reminder."""
    output = _remote_run(runtime["host"], persistent_dir, "status")
    data = json.loads(output)
    display.show_detection(data.get("detection", {}))
    display.show_status(data.get("apps", {}))

    provisioned = runtime.get("provisioned_profiles") or []
    prof_note = f" (profiles: {', '.join(provisioned)})" if provisioned else ""
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


def _provision_profiles(config_name: str, state: dict, host: str, persistent_dir: str,
                        profile_list: list[dict]) -> list[str]:
    """Ensure apps + run provisioning for each profile, record them in state.

    Returns the union of the profiles' apps (for reporting / port forwarding). The union
    is installed once (so `recover` runs a single time), then each profile's payload is
    provisioned into its own remote dir. Profile names are merged into
    state['provisioned_profiles'].
    """
    app_union: list[str] = []
    for prof in profile_list:
        for a in prof.get("apps", []):
            if a not in app_union:
                app_union.append(a)

    _ensure_apps(host, persistent_dir, app_union)
    for prof in profile_list:
        _run_provision(prof, host, persistent_dir)
    _restart_services(host, app_union)

    provisioned = list(state.get("provisioned_profiles") or [])
    for prof in profile_list:
        if prof["name"] not in provisioned:
            provisioned.append(prof["name"])
    state["provisioned_profiles"] = provisioned
    configs.save_state(config_name, state)
    return app_union


@cli.command()
@click.option("--config", "-C", "config_name", required=True, help="Config name (hardware: configs/<name>.toml)")
@click.option("--profile", "-P", "profile_csv", default=None, help="Profile(s) to provision (comma-separated; optional)")
def up(config_name: str, profile_csv: str | None) -> None:
    """Launch/recover the config's machine, then provision the given profile(s).

    The config (hardware) is required; profiles (provisioning) are optional and
    comma-separated — with none you get a bare instance. Idempotent: reuses a running
    instance if one exists, otherwise polls for GPU capacity, launches (auto-creating the
    filesystem if needed), sets up, and provisions. Re-run after a termination to recover.
    """
    try:
        cfg = configs.load_config(config_name)
    except config.ConfigError as e:
        raise click.UsageError(str(e)) from e
    profile_list = _parse_profiles(profile_csv)

    filesystem = cfg["filesystem"]

    # Reuse a live instance if we already track one for this config. The state file stays as it
    # is until something replaces it: every way of stopping below keeps it, and what it records.
    state = configs.load_state(config_name)
    tracked = state.get("instance_id")
    read = liveness.read_instance(tracked) if tracked else None
    if read and read.kind is liveness.Kind.UNKNOWN:
        _stop_unchanged(read.detail)
    if read and read.kind is liveness.Kind.ALIVE:
        if not liveness.usable(read.instance):
            _stop_not_ready({"id": tracked, **read.instance})
        ip = read.instance.get("ip")
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
                _stop_not_ready(existing)
            display.info(
                f"Adopting running instance {existing.get('id')} ({existing.get('ip')}) "
                f"already mounting '{filesystem}'."
            )
            state = _runtime_from_instance(existing, filesystem)
        else:
            try:
                state = orchestration.acquire_instance(cfg)
            except lambda_api.LambdaReplyLost as e:
                display.error(
                    f"{escape(str(e))}\nNot trying again: if that was a launch, it may have gone through. "
                    "Check 'cloudgpu lambda instances' before running this again."
                )
                sys.exit(1)
            except (orchestration.OrchestrationError, lambda_api.LambdaAPIError) as e:
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

    # Set up the host (SSH, detect, sync) and persist where it landed. The config's
    # filesystem must back the persistent dir — never whatever else happens to be mounted.
    persistent_dir, _ = _setup_host(state["host"], filesystem)
    state["persistent_dir"] = persistent_dir
    configs.save_state(config_name, state)

    # Arm the billing cap before the slow steps, so it holds even if install/provision hang.
    deadline = _configure_auto_terminate(cfg, state["host"], state.get("instance_id"), persistent_dir)
    if deadline is not None:
        state["auto_terminate_at"] = deadline
    else:
        state.pop("auto_terminate_at", None)
    configs.save_state(config_name, state)

    # Provision the requested profiles (if any), then report.
    if profile_list:
        _provision_profiles(config_name, state, state["host"], persistent_dir, profile_list)
    app_list = _apps_for_config(config_name)
    _report_up(config_name, state, persistent_dir, app_list)


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

    state = configs.load_state(config_name)
    host, persistent_dir = state.get("host"), state.get("persistent_dir")
    if not host or not persistent_dir:
        raise click.UsageError(
            f"Config '{config_name}' has no running instance. Run "
            f"'cloudgpu up --config {config_name}' first."
        )

    with console.status("Syncing remote tool..."):
        sync.sync_remote(host, persistent_dir)
    app_list = _provision_profiles(config_name, state, host, persistent_dir, profile_list)
    names = ", ".join(p["name"] for p in profile_list)
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
    state = configs.load_state(config_name)
    if not state.get("host") or not state.get("persistent_dir"):
        raise click.UsageError(f"No tracked instance for config '{config_name}'; run 'cloudgpu up' first.")

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
    """Poll until the instance is terminated/gone (so its filesystem frees up).

    This is the step before deleting a filesystem, so a read that can't tell keeps polling.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        read = liveness.read_instance(instance_id)
        if read.kind is liveness.Kind.GONE:
            return True
        if read.kind is liveness.Kind.UNKNOWN:
            display.warn(escape(read.detail))
        time.sleep(poll)
    return False


@cli.command()
@click.option("--config", "-C", "config_name", required=True, help="Config name (the instance to tear down)")
@click.option("--delete-filesystem", is_flag=True, help="Also delete the persistent filesystem (DESTROYS its data)")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompts")
def down(config_name: str, delete_filesystem: bool, yes: bool) -> None:
    """Terminate a config's instance. Keeps the filesystem unless --delete-filesystem.

    Without --delete-filesystem the data persists, so 'cloudgpu up' brings the machine
    back later. With it, the instance is terminated and the filesystem (and all its data)
    is permanently deleted.
    """
    try:
        cfg = configs.load_config(config_name)
    except config.ConfigError as e:
        raise click.UsageError(str(e)) from e

    name = cfg["name"]
    filesystem = cfg["filesystem"]
    state = configs.load_state(config_name)

    # Find the instance: tracked id, or whatever is mounting the filesystem.
    instance_id = state.get("instance_id")
    already_terminating = False
    if not instance_id:
        mounting = liveness.find_mounting(filesystem)
        if mounting.kind is liveness.Kind.UNKNOWN:
            _stop_unchanged(mounting.detail)
        if mounting.kind is liveness.Kind.ALIVE:
            instance_id = mounting.instance.get("id")
            already_terminating = mounting.instance.get("status") == "terminating"

    # Confirm (a single, appropriately-scoped prompt).
    if not yes:
        if delete_filesystem:
            click.confirm(
                f"Terminate config '{name}' AND permanently delete filesystem "
                f"'{filesystem}' and all its data?",
                abort=True,
            )
        elif instance_id and not already_terminating:
            click.confirm(
                f"Terminate instance {instance_id} for config '{name}'? "
                f"(filesystem '{filesystem}' is kept)",
                abort=True,
            )

    if already_terminating:
        display.info(f"Instance {instance_id} is already terminating.")
    elif instance_id:
        _api_call(lambda_api.terminate_instances, [instance_id])
        display.success(f"Termination requested for {instance_id}.")
    else:
        display.info("No running instance for this config.")
    configs.clear_state(config_name)

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
