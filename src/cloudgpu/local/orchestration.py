"""Find capacity and launch a GPU instance for a profile.

Ported from the launch-comfyui-instance skill's poll_launch.py, but built on the
``lambda_api`` client instead of raw urllib. Polls Lambda for a GPU in the profile's
GPU-preference order, restricted to the region where the profile's filesystem lives,
launches it, and waits for it to become active.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from . import display, lambda_api

# Friendly aliases -> Lambda instance-type names, in preference order.
ALIASES = {
    "gh200": ["gpu_1x_gh200"],
    "a100": ["gpu_1x_a100", "gpu_1x_a100_sxm4"],
    "a10": ["gpu_1x_a10"],
    "h100": ["gpu_1x_h100_pcie", "gpu_1x_h100_sxm5"],
}

SKIP_SSH_NAMES = {"known_hosts", "known_hosts2", "config", "authorized_keys", "environment"}


class OrchestrationError(Exception):
    """Raised when an instance can't be acquired (no capacity, no SSH key, boot failure)."""


def expand_targets(gpu: list[str]) -> list[str]:
    """Expand GPU aliases into Lambda instance-type names, preserving order, deduped."""
    out: list[str] = []
    for tok in gpu:
        tok = tok.strip()
        if not tok:
            continue
        out.extend(ALIASES.get(tok, [tok if tok.startswith("gpu_") else f"gpu_1x_{tok}"]))
    seen: set[str] = set()
    ordered: list[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def _key_body(pubkey_line: str) -> str:
    parts = pubkey_line.split()
    return parts[1] if len(parts) >= 2 else ""


def _iter_local_private_keys():
    """Yield (private_key_path, public_key_body) for each usable key in ~/.ssh."""
    ssh_dir = Path.home() / ".ssh"
    if not ssh_dir.is_dir():
        return
    for priv in sorted(ssh_dir.iterdir()):
        if priv.is_dir() or priv.suffix == ".pub" or priv.name in SKIP_SSH_NAMES:
            continue
        try:
            res = subprocess.run(
                ["ssh-keygen", "-y", "-f", str(priv)],
                capture_output=True, text=True, timeout=10,
                stdin=subprocess.DEVNULL,  # never block on a passphrase prompt
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if res.returncode == 0:
            yield priv, _key_body(res.stdout.strip())


def _local_key_for_account_key(name: str) -> Path | None:
    """Find the local ~/.ssh private key matching the named Lambda account key."""
    target = None
    for k in lambda_api.list_ssh_keys():
        if k.get("name") == name:
            target = _key_body(k.get("public_key", ""))
            break
    if not target:
        return None
    for priv, body in _iter_local_private_keys():
        if body and body == target:
            return priv
    return None


def ensure_ssh_key(name: str) -> str:
    """Load the local private key matching the named account key into the ssh-agent.

    The profile names the Lambda SSH key to launch with; we find the matching local
    private key in ~/.ssh and ``ssh-add`` it so plain ``ssh`` works afterwards. If no
    local match is found we warn but proceed (it may already be in the agent / config).
    """
    try:
        local_priv = _local_key_for_account_key(name)
    except lambda_api.LambdaAPIError:
        local_priv = None
    if local_priv:
        subprocess.run(["ssh-add", str(local_priv)], check=False)
    else:
        display.warn(
            f"No local private key in ~/.ssh matches the Lambda SSH key '{name}'. "
            "Ensure it's loaded in your ssh-agent, or SSH will fail."
        )
    return name


def _fs_region(fs: dict) -> str | None:
    r = fs.get("region") or {}
    return r.get("name") if isinstance(r, dict) else r


def filesystem_record(name: str) -> dict | None:
    """Return the account filesystem with this name, or None."""
    for fs in lambda_api.list_filesystems():
        if fs.get("name") == name:
            return fs
    return None


def region_for_filesystem(filesystem: str) -> str | None:
    """Return the region a filesystem lives in, or None if not found."""
    fs = filesystem_record(filesystem)
    return _fs_region(fs) if fs else None


def pick_target(types: dict, targets: list[str], region: str) -> str | None:
    """Return the first target type (in preference order) with capacity in ``region``."""
    for tname in targets:
        entry = types.get(tname)
        if not entry:
            continue
        available = [r.get("name") for r in entry.get("regions_with_capacity_available", [])]
        if region in available:
            return tname
    return None


def pick_target_any(types: dict, targets: list[str], allowed: list[str] | None = None) -> tuple[str, str] | None:
    """Return the first (type, region) with capacity, honoring preference order.

    ``allowed`` optionally restricts to a set of region names; None means any region.
    """
    for tname in targets:
        entry = types.get(tname)
        if not entry:
            continue
        for r in entry.get("regions_with_capacity_available", []):
            region = r.get("name")
            if region and (allowed is None or region in allowed):
                return tname, region
    return None


def wait_active(instance_id: str, poll: int = 20, max_minutes: int = 20) -> dict:
    """Poll until the instance is active with an IP, or raise on failure/timeout."""
    deadline = time.time() + max_minutes * 60
    while time.time() < deadline:
        inst = lambda_api.get_instance(instance_id)
        status, ip = inst.get("status"), inst.get("ip")
        display.info(f"  status={status} ip={ip or '-'}")
        if status == "active" and ip:
            return inst
        if status in ("terminated", "error"):
            raise OrchestrationError(f"instance entered '{status}' before becoming active")
        time.sleep(poll)
    raise OrchestrationError("timed out waiting for instance to become active")


def acquire_instance(profile: dict[str, Any]) -> dict[str, Any]:
    """Poll for capacity and launch an instance for the profile.

    The filesystem is auto-managed: if one named ``profile['filesystem']`` already
    exists, the launch is pinned to its region; otherwise a filesystem is created in
    the first region that has capacity (honoring an optional ``profile['region']``).

    Returns a runtime dict (instance_id, ip, host, filesystem, filesystem_id,
    created_filesystem, instance_type, region, ssh_key, price_cents_per_hour,
    launched_at). Raises OrchestrationError on timeout/boot failure, LambdaAPIError on
    API failure.
    """
    filesystem = profile.get("filesystem") or profile["name"]
    targets = expand_targets(profile["gpu"])
    ssh_key = ensure_ssh_key(profile.get("ssh_key"))
    poll = int(profile.get("poll_seconds", 20))
    max_hours = float(profile.get("max_hours", 12))
    pinned_region = profile.get("region") or None

    # If the filesystem already exists, its region is fixed; otherwise we'll create it
    # in whichever region first has capacity (pinned region wins if set).
    fs = filesystem_record(filesystem)
    fs_id = fs.get("id") if fs else None
    region = pinned_region or (_fs_region(fs) if fs else None)
    created = False

    where = region or (pinned_region or "any region with capacity")
    display.info(
        f"Looking for {targets} in {where} (filesystem '{filesystem}'"
        f"{'' if fs else ', will create'}), polling every {poll}s for up to {max_hours}h..."
    )

    deadline = time.time() + max_hours * 3600
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            types = lambda_api.list_instance_types()
        except lambda_api.LambdaAPIError as e:  # transient; keep polling
            display.warn(f"capacity check failed: {e}; retrying")
            time.sleep(poll)
            continue

        if region:  # filesystem exists or region is pinned
            tname = pick_target(types, targets, region)
            hit = (tname, region) if tname else None
        else:       # creating a new filesystem: any region with capacity
            hit = pick_target_any(types, targets, None)
        if not hit:
            if attempt % 15 == 0:
                display.info(f"  no capacity in {region or 'any eligible region'} yet (attempt {attempt})")
            time.sleep(poll)
            continue

        tname, chosen_region = hit

        # Create the filesystem the first time we land on a region (then pin to it).
        if fs_id is None:
            display.info(f"Creating filesystem '{filesystem}' in {chosen_region}...")
            new_fs = lambda_api.create_filesystem(filesystem, chosen_region)
            fs_id = (new_fs or {}).get("id")
            region = chosen_region
            created = True

        display.success(f"Capacity found: {tname} in {chosen_region} -- launching")
        try:
            res = lambda_api.launch_instance(
                region_name=chosen_region,
                instance_type_name=tname,
                ssh_key_names=[ssh_key],
                file_system_names=[filesystem],
                name=profile.get("instance_name", profile["name"]),
            )
        except lambda_api.LambdaAPIError as e:  # capacity may have vanished in the race
            display.warn(f"launch failed ({e}); continuing to poll")
            time.sleep(poll)
            continue

        instance_id = _first_instance_id(res)
        if not instance_id:
            raise OrchestrationError(f"launch returned no instance id: {res}")
        display.info(f"Launched {instance_id}; waiting for it to become active...")
        inst = wait_active(instance_id, poll=poll)

        ip = inst.get("ip")
        return {
            "instance_id": instance_id,
            "ip": ip,
            "host": f"ubuntu@{ip}",
            "filesystem": filesystem,
            "filesystem_id": fs_id,
            "created_filesystem": created,
            "instance_type": tname,
            "region": chosen_region,
            "ssh_key": ssh_key,
            "price_cents_per_hour": (inst.get("instance_type") or {}).get("price_cents_per_hour"),
            "launched_at": int(time.time()),
        }

    raise OrchestrationError(
        f"Timed out after {max_hours}h without finding capacity for {targets}"
        + (f" in {region}." if region else ".")
    )


def _first_instance_id(launch_result: Any) -> str | None:
    """Pull the first instance id out of a launch response (shape varies)."""
    if isinstance(launch_result, dict):
        ids = launch_result.get("instance_ids")
        if isinstance(ids, list) and ids:
            return str(ids[0])
        if launch_result.get("id"):
            return str(launch_result["id"])
    return None
