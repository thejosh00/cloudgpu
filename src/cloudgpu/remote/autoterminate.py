"""Self-terminating lifetime cap. Stdlib-only; runs on the instance.

``cloudgpu up`` arms this when the profile sets ``auto_terminate_hours``: the module
copies itself to /etc/cloudgpu/autoterminate.py, writes a root-only config (Lambda API
key, instance id, deadline), and installs a systemd timer that re-runs the check every
few minutes. Past the deadline the instance terminates itself via the Lambda API, so a
forgotten ``cloudgpu down`` stops billing even with no machine watching. Each ``up``
re-arms, so the deadline is always ``auto_terminate_hours`` after the latest ``up``.

Everything lives on the instance-local disk (root-owned; the config is mode 600) —
never on the shared persistent filesystem — and dies with the instance. This file must
stay standalone (stdlib imports only, no package-relative imports): the /etc copy runs
outside the package via ``python3 /etc/cloudgpu/autoterminate.py --check``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

ETC_DIR = "/etc/cloudgpu"
SCRIPT_PATH = f"{ETC_DIR}/autoterminate.py"
CONF_PATH = f"{ETC_DIR}/autoterminate.json"
UNIT_NAME = "cloudgpu-autoterminate"
API_BASE = "https://cloud.lambda.ai/api/v1"
USER_AGENT = "cloudgpu/0.1.0"


def log(msg: str) -> None:
    print(f">>> {msg}", file=sys.stderr, flush=True)


def _sudo(*cmd: str) -> None:
    subprocess.run(["sudo", *cmd], check=True, timeout=60)


def _sudo_install(content: str, dest: str, mode: str) -> None:
    """Write ``content`` to a root-owned file via a private temp file + sudo install."""
    tmp = f"/tmp/cloudgpu-{os.path.basename(dest)}.{os.getpid()}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(content)
    try:
        _sudo("install", "-o", "root", "-g", "root", "-m", mode, tmp, dest)
    finally:
        os.remove(tmp)


def unit_texts() -> tuple[str, str]:
    """(service, timer) unit file contents (pure; testable without sudo)."""
    service = f"""[Unit]
Description=cloudgpu auto-terminate check

[Service]
Type=oneshot
ExecStart=/usr/bin/python3 {SCRIPT_PATH} --check
"""
    timer = """[Unit]
Description=cloudgpu auto-terminate: cap instance lifetime

[Timer]
OnBootSec=2min
OnUnitActiveSec=5min

[Install]
WantedBy=timers.target
"""
    return service, timer


def parse_key_file(text: str) -> str:
    """Extract the API key from LAMBDA_API_KEY=... lines."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("LAMBDA_API_KEY="):
            key = line.split("=", 1)[1].strip()
            if key:
                return key
    raise ValueError("key file has no LAMBDA_API_KEY=... line")


def arm(hours: float, instance_id: str, key_file: str) -> int:
    """Install the root-only config + self-copy + systemd timer. Returns the deadline epoch."""
    with open(key_file) as f:
        api_key = parse_key_file(f.read())
    os.remove(key_file)  # transferred for this handoff only; the root copy is canonical

    now = int(time.time())
    deadline = now + int(hours * 3600)
    conf = {
        "instance_id": instance_id,
        "deadline_epoch": deadline,
        "api_key": api_key,
        "armed_at": now,
        "hours": hours,
    }
    _sudo("install", "-d", "-o", "root", "-g", "root", "-m", "755", ETC_DIR)
    _sudo_install(json.dumps(conf, indent=2) + "\n", CONF_PATH, "600")
    with open(os.path.abspath(__file__)) as f:
        _sudo_install(f.read(), SCRIPT_PATH, "644")
    service, timer = unit_texts()
    _sudo_install(service, f"/etc/systemd/system/{UNIT_NAME}.service", "644")
    _sudo_install(timer, f"/etc/systemd/system/{UNIT_NAME}.timer", "644")
    _sudo("systemctl", "daemon-reload")
    _sudo("systemctl", "enable", "--now", f"{UNIT_NAME}.timer")
    log(
        f"Auto-terminate armed: instance {instance_id} terminates at "
        f"{time.strftime('%Y-%m-%d %H:%M %Z', time.localtime(deadline))} "
        f"({hours:g}h from now). Re-run 'cloudgpu up' to extend."
    )
    return deadline


def disarm() -> None:
    """Remove the timer + config (idempotent; silent when never armed)."""
    timer_unit = f"/etc/systemd/system/{UNIT_NAME}.timer"
    if not (os.path.exists(CONF_PATH) or os.path.exists(timer_unit)):
        return
    subprocess.run(
        ["sudo", "systemctl", "disable", "--now", f"{UNIT_NAME}.timer"],
        check=False, timeout=60,
    )
    _sudo("rm", "-f", CONF_PATH, SCRIPT_PATH,
          f"/etc/systemd/system/{UNIT_NAME}.service", timer_unit)
    _sudo("systemctl", "daemon-reload")
    log("Auto-terminate disarmed.")


def due(conf: dict, now: float) -> bool:
    return now >= float(conf["deadline_epoch"])


def terminate(conf: dict) -> None:
    """Ask the Lambda API to terminate this instance."""
    body = json.dumps({"instance_ids": [conf["instance_id"]]}).encode()
    req = urllib.request.Request(
        f"{API_BASE}/instance-operations/terminate",
        data=body,
        headers={
            "Authorization": f"Bearer {conf['api_key']}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60):
        pass


def check() -> int:
    """Timer entry point (runs as root from the /etc copy every few minutes)."""
    try:
        with open(CONF_PATH) as f:
            conf = json.load(f)
    except (OSError, ValueError) as e:
        log(f"no valid config at {CONF_PATH}: {e}")
        return 1
    now = time.time()
    if not due(conf, now):
        log(f"deadline in {(float(conf['deadline_epoch']) - now) / 3600:.1f}h; nothing to do")
        return 0
    log(f"deadline passed; terminating instance {conf['instance_id']} via the Lambda API...")
    try:
        terminate(conf)
    except (urllib.error.URLError, OSError) as e:
        log(f"terminate request failed ({e}); the timer retries in a few minutes")
        return 1
    log("termination requested")
    return 0


if __name__ == "__main__":
    sys.exit(check() if "--check" in sys.argv else 0)
