"""Put the bucket sync in place on the instance, start it, or take it away. Stdlib-only.

``setup`` installs, on the instance's own disk and all root-owned:

- the sync user (``cloudgpu-sync``: own group, no login shell, not in sudo, a member of
  the ``cloudgpu-app`` group so it reads the synced directories through the group);
- the bucket key: the transferred rclone config, moved to /etc/cloudgpu/rclone.conf
  (root, mode 600). The transferred copy is removed whether or not that worked. This
  module never opens the file: it only passes its path to ``sudo install``, so the key
  is never in a command line, an environment variable, a log line or an exception;
- rclone, one pinned version, checked against a SHA-256 pinned here;
- a copy of the runner (datasync.py), the spec it reads, a oneshot service and a timer.

``setup`` does not start anything; ``start`` enables the timer. They are separate so a
restore can run in between. ``remove`` stops the timer and takes the units, key, spec
and runner copy away, quietly when there was nothing.

Nothing is put on the persistent filesystem, where ``cloudgpu-app`` can write.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import pwd
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

from . import datasync
from .utils import APP_USER

SYNC_USER = "cloudgpu-sync"
ETC_DIR = "/etc/cloudgpu"
KEY_PATH = f"{ETC_DIR}/rclone.conf"
RUNNER_PATH = f"{ETC_DIR}/datasync.py"
SPEC_PATH = f"{ETC_DIR}/datasync.json"
SYSTEMD_DIR = "/etc/systemd/system"
UNIT_NAME = "cloudgpu-datasync"
CREDENTIAL_NAME = "rclone.conf"
# systemd creates /var/lib/<STATE_DIR_NAME> owned by the sync user (StateDirectory=).
STATE_DIR_NAME = "cloudgpu-sync"
STATE_DIR = f"/var/lib/{STATE_DIR_NAME}"
LAST_SUCCESS_FILE = f"{STATE_DIR}/last-success"
# A run has no rclone timeout of its own. This only stops a hung run from blocking every
# later one for ever; a first upload of large datasets must still fit inside it.
RUN_TIMEOUT = "6h"

# One pinned rclone. The hashes are those in
# https://downloads.rclone.org/v1.76.0/SHA256SUMS (fetched 2026-11-15).
RCLONE_VERSION = "1.76.0"
RCLONE_SHA256 = {
    "amd64": "b00f00b3266c3764413afe599d3ec316f6f43b3e0ea8a0bc1d186d2133e46df7",
    "arm64": "ee169016675e23a69d77240565133ff7dbc9b33d16fa8f7053d4af39f1eaeb78",
}
RCLONE_DIR = "/usr/local/lib/cloudgpu"  # root-owned; cloudgpu-app cannot write here
ARCHES = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}


class SetupError(Exception):
    """Sync could not be put in place; the message is a sentence for the person."""


def log(msg: str) -> None:
    print(f">>> {msg}", file=sys.stderr, flush=True)


def _sudo(*cmd: str) -> None:
    subprocess.run(["sudo", *cmd], check=True, timeout=120)


def _sudo_quiet(*cmd: str) -> None:
    """For steps that may fail because there was nothing to act on."""
    subprocess.run(["sudo", *cmd], check=False, timeout=120,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def rclone_path() -> str:
    """The version is in the path, so a new pin never reuses an old binary."""
    return f"{RCLONE_DIR}/rclone-v{RCLONE_VERSION}/rclone"


def service_path() -> str:
    return f"{SYSTEMD_DIR}/{UNIT_NAME}.service"


def timer_path() -> str:
    return f"{SYSTEMD_DIR}/{UNIT_NAME}.timer"


def _install_text(content: str, dest: str, mode: str = "644") -> bool:
    """Write a root-owned, world-readable file. False (and nothing done) if it is already so."""
    try:
        with open(dest) as f:
            if f.read() == content:
                return False
    except OSError:
        pass
    fd, tmp = tempfile.mkstemp(prefix="cloudgpu-")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        _sudo("install", "-o", "root", "-g", "root", "-m", mode, tmp, dest)
    finally:
        os.remove(tmp)
    return True


# --- the sync user -----------------------------------------------------------------

def ensure_sync_user() -> None:
    """Create ``cloudgpu-sync`` (idempotent): system user, own group, no login shell, no
    home of its own, not in sudo; in the ``cloudgpu-app`` group, read access only."""
    from . import service
    service.ensure_app_user()  # the group it joins has to exist; idempotent
    try:
        pwd.getpwnam(SYNC_USER)
    except KeyError:
        _sudo("useradd", "--system", "--user-group", "--home-dir", STATE_DIR,
              "--no-create-home", "--shell", "/usr/sbin/nologin", SYNC_USER)
        log(f"Created the sync user '{SYNC_USER}'.")
    _sudo("usermod", "-aG", APP_USER, SYNC_USER)


# --- the key -----------------------------------------------------------------------

def install_key(key_file: str) -> None:
    """Copy the transferred rclone config to the root-only key file, without opening it."""
    if not os.path.isfile(key_file):
        raise SetupError(f"The bucket key file {key_file} did not arrive on the instance.")
    _sudo("install", "-o", "root", "-g", "root", "-m", "600", key_file, KEY_PATH)


# --- rclone ------------------------------------------------------------------------

def rclone_arch(machine: str | None = None) -> str:
    machine = platform.machine() if machine is None else machine
    arch = ARCHES.get(machine.lower())
    if arch is None:
        raise SetupError(f"There is no pinned rclone for this machine type ({machine}); "
                         "only linux amd64 and arm64 are.")
    return arch


def rclone_url(arch: str) -> str:
    return (f"https://downloads.rclone.org/v{RCLONE_VERSION}/"
            f"rclone-v{RCLONE_VERSION}-linux-{arch}.zip")


def _fetch(url: str, dest: str) -> None:
    with urllib.request.urlopen(url, timeout=120) as response, open(dest, "wb") as f:
        shutil.copyfileobj(response, f)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_rclone() -> bool:
    """Install the pinned rclone, root-owned. False if it was already there."""
    dest = rclone_path()
    if os.path.isfile(dest):
        return False
    arch = rclone_arch()
    url = rclone_url(arch)
    work = tempfile.mkdtemp(prefix="cloudgpu-rclone-")
    try:
        archive = os.path.join(work, "rclone.zip")
        try:
            _fetch(url, archive)
        except OSError as e:
            raise SetupError(f"Could not download rclone from {url}: {e}") from None
        got = _sha256(archive)
        if got != RCLONE_SHA256[arch]:
            raise SetupError(
                f"The rclone download from {url} does not match the pinned SHA-256 "
                f"(got {got}); it was not installed.")
        member = f"rclone-v{RCLONE_VERSION}-linux-{arch}/rclone"
        binary = os.path.join(work, "rclone")
        try:
            with zipfile.ZipFile(archive) as z, z.open(member) as src, open(binary, "wb") as out:
                shutil.copyfileobj(src, out)
        except (KeyError, zipfile.BadZipFile) as e:
            raise SetupError(f"The rclone download from {url} has no {member}: {e}") from None
        _sudo("install", "-d", "-o", "root", "-g", "root", "-m", "755",
              RCLONE_DIR, os.path.dirname(dest))
        # Under another name first, then renamed: an interrupted copy never sits at the
        # path that "already installed" looks at.
        _sudo("install", "-o", "root", "-g", "root", "-m", "755", binary, dest + ".new")
        _sudo("mv", "-f", dest + ".new", dest)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    log(f"Installed rclone v{RCLONE_VERSION} ({arch}) at {dest}.")
    return True


# --- spec and units ----------------------------------------------------------------

def build_spec(request: object, persistent_dir: str) -> dict:
    """The runner's spec from what the caller sent (remote, paths, interval_minutes)."""
    if not isinstance(request, dict):
        raise SetupError("The sync request is not a JSON object.")
    remote = request.get("remote")
    if not isinstance(remote, str) or ":" not in remote or remote.startswith("-"):
        raise SetupError("The sync request's 'remote' must look like name:bucket/path.")
    paths = request.get("paths")
    if not isinstance(paths, list):
        raise SetupError("The sync request's 'paths' must be a list.")
    for path in paths:
        problem = datasync.path_problem(path)
        if problem:
            raise SetupError(f"The sync request's path {path!r} is refused: {problem}.")
    interval = request.get("interval_minutes")
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise SetupError("The sync request's 'interval_minutes' must be a whole number, 1 or more.")
    if not persistent_dir.startswith("/"):
        raise SetupError("The persistent dir must be an absolute path.")
    return {
        "remote": remote,
        "persistent_dir": persistent_dir,
        "paths": paths,
        "interval_minutes": interval,
        "rclone": rclone_path(),
        "last_success_file": LAST_SUCCESS_FILE,
    }


def unit_texts(interval_minutes: int) -> tuple[str, str]:
    """(service, timer) unit file contents (pure; testable without sudo)."""
    # LoadCredential= (systemd 247 or newer; Ubuntu 22.04 has 249): systemd, as root,
    # copies the root-only key file into a directory only this service's user can read,
    # for the length of one run, and names it in $CREDENTIALS_DIRECTORY. The runner gets
    # the path through the shell rather than the %d specifier, which needs systemd 250.
    # "$$" is systemd's escape for a literal "$", so the shell expands the variable.
    #
    # No ReadWritePaths: with ProtectSystem=strict the whole filesystem, the persistent
    # dir included, is read-only to a run. Only the state directory can be written.
    service = f"""[Unit]
Description=cloudgpu data sync: upload the synced paths to the bucket
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User={SYNC_USER}
Group={SYNC_USER}
SupplementaryGroups={APP_USER}
LoadCredential={CREDENTIAL_NAME}:{KEY_PATH}
StateDirectory={STATE_DIR_NAME}
StateDirectoryMode=0750
Environment=HOME={STATE_DIR}
ExecStart=/bin/sh -c 'exec /usr/bin/python3 {RUNNER_PATH} run --spec {SPEC_PATH} --rclone-config "$${{CREDENTIALS_DIRECTORY}}/{CREDENTIAL_NAME}"'
TimeoutStartSec={RUN_TIMEOUT}
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
"""
    timer = f"""[Unit]
Description=cloudgpu data sync: every {interval_minutes} minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec={interval_minutes}min

[Install]
WantedBy=timers.target
"""
    return service, timer


# --- setup, start, remove ----------------------------------------------------------

def _discard(path: str | None) -> None:
    if path:
        try:
            os.remove(path)
        except OSError:
            pass


def setup(persistent_dir: str, key_file: str, request_file: str) -> None:
    """Put key, user, rclone, runner, spec and units in place. Starts nothing."""
    try:
        try:
            with open(request_file) as f:
                request = json.load(f)
        except (OSError, ValueError) as e:
            raise SetupError(f"The sync request {request_file} could not be read: {e}") from None
        spec = build_spec(request, persistent_dir)
        _sudo("install", "-d", "-o", "root", "-g", "root", "-m", "755", ETC_DIR)
        install_key(key_file)
        ensure_sync_user()
        ensure_rclone()
        with open(datasync.__file__) as f:
            _install_text(f.read(), RUNNER_PATH)
        _install_text(json.dumps(spec, indent=2) + "\n", SPEC_PATH)
        service, timer = unit_texts(spec["interval_minutes"])
        changed = [_install_text(service, service_path()), _install_text(timer, timer_path())]
        if any(changed):
            _sudo("systemctl", "daemon-reload")
    finally:
        # Transferred for this handoff only: never left in ubuntu's home, even on failure.
        _discard(key_file)
        _discard(request_file)
    log(f"Sync to {spec['remote']} is in place: {len(spec['paths'])} paths, "
        f"every {spec['interval_minutes']} minutes (not started).")


def start() -> None:
    """Enable and start the timer (the first run follows at once if the instance has been
    up for two minutes)."""
    if not os.path.exists(timer_path()):
        raise SetupError("Sync is not set up on this instance, so its timer cannot be started.")
    _sudo("systemctl", "enable", "--now", f"{UNIT_NAME}.timer")
    # A timer that was already running keeps its old interval until restarted.
    _sudo("systemctl", "try-restart", f"{UNIT_NAME}.timer")
    log("Sync timer started.")


def remove() -> None:
    """Stop the timer and remove units, key, spec and runner copy (silent when absent).

    The sync user, rclone and the last-success file are left: none is a secret, and all
    die with the instance.
    """
    files = [KEY_PATH, SPEC_PATH, RUNNER_PATH, service_path(), timer_path()]
    if not any(os.path.lexists(p) for p in files):
        return
    _sudo_quiet("systemctl", "disable", "--now", f"{UNIT_NAME}.timer")
    _sudo_quiet("systemctl", "stop", f"{UNIT_NAME}.service")
    _sudo("rm", "-f", *files)
    _sudo("systemctl", "daemon-reload")
    log("Sync removed from this instance.")


def main(action: str, persistent_dir: str, key_file: str | None = None,
         request_file: str | None = None, start_timer: bool = False) -> int:
    """The remote tool's ``datasync`` command. 0, or 1 with one sentence on stderr."""
    try:
        if action == "setup":
            if not key_file or not request_file:
                # Still never leave what was transferred behind.
                _discard(key_file)
                _discard(request_file)
                raise SetupError("Sync setup needs --key-file and --spec-file.")
            setup(persistent_dir, key_file, request_file)
            if start_timer:
                start()
        elif action == "start":
            start()
        else:
            remove()
    except SetupError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as e:
        print(f"error: Sync {action} failed: '{' '.join(map(str, e.cmd))}' exited "
              f"{e.returncode}.", file=sys.stderr)
        return 1
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"error: Sync {action} failed: {e}", file=sys.stderr)
        return 1
    return 0
