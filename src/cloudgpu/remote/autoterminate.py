"""Self-terminating lifetime cap. Stdlib-only; runs on the instance.

``cloudgpu up`` arms this when the profile sets ``auto_terminate_hours``: the module
copies itself to /etc/cloudgpu/autoterminate.py, writes a root-only config (Lambda API
key, instance id, deadline), and installs a systemd timer that re-runs the check every
few minutes. Past the deadline the instance terminates itself via the Lambda API, so a
forgotten ``cloudgpu down`` stops billing even with no machine watching. Each ``up``
re-arms, so the deadline is always ``auto_terminate_hours`` after the latest ``up``.

Before the request, when the bucket sync is installed, the guard runs one final sync
(``final_sync``) with the same rules as ``cloudgpu down``'s (``datasync_setup.final``):
timer stopped, a run in progress waited for, one fresh run of the existing unit that must
succeed and record a success at or after its start. It is limited to
``FINAL_LIMIT_SECONDS`` in total; stopping the bill wins, so past the limit or after a
failure the guard requests termination anyway. The attempt is recorded once per deadline
(root-only, beside the config) so a retried request does not sync again, and one line per
guard termination is appended to ``<persistent dir>/cloudgpu/guard.log``, the only thing
the guard writes off the instance's own disk; a failure to write either never blocks
termination.

Everything else lives on the instance-local disk (root-owned; the config is mode 600) and
dies with the instance. This file must
stay standalone (stdlib imports only, no package-relative imports): the /etc copy runs
outside the package via ``python3 /etc/cloudgpu/autoterminate.py --check``.
"""

from __future__ import annotations

import calendar
import json
import os
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ETC_DIR = "/etc/cloudgpu"
SCRIPT_PATH = f"{ETC_DIR}/autoterminate.py"
CONF_PATH = f"{ETC_DIR}/autoterminate.json"
UNIT_NAME = "cloudgpu-autoterminate"
API_BASE = "https://cloud.lambda.ai/api/v1"
USER_AGENT = "cloudgpu/0.1.0"
# Where the final sync's outcome for the current deadline is recorded (root-only).
FINAL_RECORD_PATH = f"{ETC_DIR}/autoterminate-final.json"
# The whole final sync, from deciding to sync to sending the request (plan v2, Phase 3).
FINAL_LIMIT_SECONDS = 10 * 60
FINAL_POLL_SECONDS = 5
# Kept out of the sync's share of the limit for the guard log write, which is on a network
# filesystem that can hang; the guard stops waiting for it after this long.
GUARD_LOG_WAIT_SECONDS = 15
# Appended to on the persistent filesystem, beside cloudgpu/state.json.
GUARD_LOG_NAME = "guard.log"

# The bucket sync, as datasync_setup.py installs it. Repeated because this file must stay
# standalone; tests/test_autoterminate_final_3_5.py fails if they drift apart.
SYNC_UNIT = "cloudgpu-datasync"
SYNC_SPEC_PATH = f"{ETC_DIR}/datasync.json"
SYNC_SERVICE_PATH = f"/etc/systemd/system/{SYNC_UNIT}.service"
SYNC_TIMER_PATH = f"/etc/systemd/system/{SYNC_UNIT}.timer"
SYNC_LAST_SUCCESS = "/var/lib/cloudgpu-sync/last-success"


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


def arm(hours: float, instance_id: str, key_file: str, persistent_dir: str | None = None) -> int:
    """Install the root-only config + self-copy + systemd timer. Returns the deadline epoch."""
    try:
        with open(key_file) as f:
            api_key = parse_key_file(f.read())
    finally:
        # Transferred for this handoff only (the root copy is canonical): remove it even
        # when parsing fails, so a bad file never lingers in ubuntu's home.
        os.remove(key_file)

    now = int(time.time())
    deadline = now + int(hours * 3600)
    conf = {
        "instance_id": instance_id,
        "deadline_epoch": deadline,
        "api_key": api_key,
        "armed_at": now,
        "hours": hours,
        "persistent_dir": persistent_dir,
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
    _sudo("rm", "-f", CONF_PATH, SCRIPT_PATH, FINAL_RECORD_PATH,
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


def _stamp(when: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(when))


def _last_success() -> tuple[float, str] | None:
    """(seconds, text) of the sync's last good run, or None (read directly: we are root)."""
    try:
        with open(SYNC_LAST_SUCCESS) as f:
            text = f.read().strip()
        return calendar.timegm(time.strptime(text, "%Y-%m-%dT%H:%M:%SZ")), text
    except (OSError, ValueError):
        return None


class _CutOff(Exception):
    """The final sync's time limit was reached."""


def _left(end: float) -> float:
    left = end - time.monotonic()
    if left <= 0:
        raise _CutOff
    return left


def _systemctl(end: float, *args: str) -> subprocess.CompletedProcess:
    """One systemctl call, never allowed to run past the limit."""
    return subprocess.run(["systemctl", *args], check=False, capture_output=True, text=True,
                          timeout=min(60.0, _left(end)))


def _sync_state(end: float, prop: str) -> str:
    done = _systemctl(end, "show", f"--property={prop}", "--value", f"{SYNC_UNIT}.service")
    return (done.stdout or "").strip()


def final_sync(limit: float = FINAL_LIMIT_SECONDS - GUARD_LOG_WAIT_SECONDS) -> str:
    """The guard's final sync, within ``limit`` seconds. Returns the outcome as one phrase.

    Same rules as ``datasync_setup.final``: stop the timer, wait for a run in progress
    rather than join it, run the service once more, and count it only on the unit's
    result and a last-success time at or after its start. Unlike ``down``'s, the timer is
    not started again after a failure (termination follows at once), and nothing here
    raises: every way it can end is an outcome, and termination is requested after it.
    Only systemctl is run; the bucket key is never read or passed.
    """
    if not all(os.path.exists(p) for p in (SYNC_SPEC_PATH, SYNC_SERVICE_PATH, SYNC_TIMER_PATH)):
        return "no sync installed"
    end = time.monotonic() + limit

    def failed(why: str) -> str:
        last = _last_success()
        return f"failed: {why}; last success {last[1] if last else 'never'}"

    try:
        if _systemctl(end, "stop", f"{SYNC_UNIT}.timer").returncode != 0:
            return failed("the sync timer could not be stopped")
        waited = False
        while _sync_state(end, "ActiveState") in ("activating", "deactivating"):
            if not waited:
                log("A sync run is in progress; waiting for it to finish...")
                waited = True
            time.sleep(min(FINAL_POLL_SECONDS, _left(end)))
        log("Running the final sync...")
        started = int(time.time())
        try:
            # Starting a oneshot returns when its run has ended, non-zero if it failed.
            code = subprocess.run(["systemctl", "start", f"{SYNC_UNIT}.service"], check=False,
                                  timeout=_left(end)).returncode
        except subprocess.TimeoutExpired:
            raise _CutOff from None
        result = _sync_state(end, "Result")
        if code != 0 or result != "success":
            return failed(f"the run's result was {result or 'unknown'}")
        last = _last_success()
        if last is None or last[0] < started:
            return failed("the run recorded no new success")
        return f"synced at {last[1]}"
    except _CutOff:
        last = _last_success()
        return f"cut off at the time limit; last success {last[1] if last else 'never'}"
    except subprocess.TimeoutExpired:
        if time.monotonic() >= end:
            last = _last_success()
            return f"cut off at the time limit; last success {last[1] if last else 'never'}"
        return failed("systemctl did not answer")
    except (OSError, subprocess.SubprocessError) as e:
        return failed(str(e))


def _attempted(conf: dict) -> str | None:
    """The recorded outcome of this deadline's final sync, or None if not attempted."""
    try:
        with open(FINAL_RECORD_PATH) as f:
            record = json.load(f)
        if record.get("deadline_epoch") == conf["deadline_epoch"]:
            return str(record.get("outcome"))
    except (OSError, ValueError, AttributeError):
        pass
    return None


def _record_attempt(conf: dict, outcome: str, now: float) -> None:
    try:
        fd = os.open(FINAL_RECORD_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"deadline_epoch": conf["deadline_epoch"], "outcome": outcome,
                       "at": _stamp(now)}, f)
    except OSError as e:
        log(f"could not record the final sync attempt ({e}); a retry will sync again")


def append_guard_log(conf: dict, outcome: str, now: float) -> None:
    """One line on the persistent filesystem that outlives the instance. Never raises.

    That filesystem is writable by the app user, so as root: no symlink is followed (for
    the cloudgpu directory or the file), nothing is created but the file, and anything but
    a regular file is left alone (O_NONBLOCK, so a FIFO can't hold termination up).
    """
    persistent_dir = conf.get("persistent_dir")
    if not persistent_dir:
        log("not writing the guard log: the persistent filesystem was not given when armed")
        return
    line = (f"{_stamp(now)} guard terminating instance {conf['instance_id']}: "
            f"final sync {outcome}\n")
    directory = os.path.join(persistent_dir, "cloudgpu")
    try:
        dir_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            fd = os.open(GUARD_LOG_NAME, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW
                         | os.O_NONBLOCK, 0o644, dir_fd=dir_fd)
        finally:
            os.close(dir_fd)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError(f"{GUARD_LOG_NAME} is not a regular file")
            os.write(fd, line.encode())
        finally:
            os.close(fd)
    except OSError as e:
        log(f"could not write {os.path.join(directory, GUARD_LOG_NAME)}: {e}")


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
    earlier = _attempted(conf)
    if earlier is not None:
        log(f"deadline passed; the final sync was already attempted ({earlier})")
    else:
        log(f"deadline passed; final sync first (at most {FINAL_LIMIT_SECONDS / 60:g} minutes)")
        outcome = final_sync()
        log(f"final sync: {outcome}")
        _record_attempt(conf, outcome, time.time())
        # In a thread with a bound, so a hung network filesystem can't hold termination up.
        writer = threading.Thread(target=append_guard_log, args=(conf, outcome, time.time()),
                                  daemon=True)
        writer.start()
        writer.join(GUARD_LOG_WAIT_SECONDS)
        if writer.is_alive():
            log(f"gave up waiting for the guard log after {GUARD_LOG_WAIT_SECONDS}s")
    log(f"terminating instance {conf['instance_id']} via the Lambda API...")
    try:
        terminate(conf)
    except (urllib.error.URLError, OSError) as e:
        log(f"terminate request failed ({e}); the timer retries in a few minutes")
        return 1
    log("termination requested")
    return 0


if __name__ == "__main__":
    sys.exit(check() if "--check" in sys.argv else 0)
