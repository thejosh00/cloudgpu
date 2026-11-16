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
restore can run in between: ``restore`` copies the declared paths down from the bucket in a
one-off restricted run as the sync user, which can write the declared paths and nothing
else, then shares what arrived with the ``cloudgpu-app`` group and records the restore.
``final`` is the last upload before the instance is terminated: it stops the timer, lets
a run in progress finish, runs the service once more and says whether that run worked.
``remove`` stops the timer and takes the units, key, spec
and runner copy away, quietly when there was nothing.

Nothing is put on the persistent filesystem, where ``cloudgpu-app`` can write.
"""

from __future__ import annotations

import calendar
import grp
import hashlib
import json
import os
import platform
import pwd
import shutil
import subprocess
import sys
import tempfile
import time
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
RESTORE_UNIT_NAME = "cloudgpu-datasync-restore"
# Ours on top of the unit's own limit, so a systemd-run that never returns is still ended.
RESTORE_WAIT_SECONDS = 6 * 60 * 60 + 300
# The same for the final sync: RUN_TIMEOUT plus a margin, for each of its two waits.
FINAL_WAIT_SECONDS = 6 * 60 * 60 + 300
FINAL_POLL_SECONDS = 5

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


# --- restore -----------------------------------------------------------------------

def _app_gid() -> int:
    return grp.getgrnam(APP_USER).gr_gid


def _timer_active() -> bool:
    done = subprocess.run(["sudo", "systemctl", "is-active", "--quiet", f"{UNIT_NAME}.timer"],
                          check=False, timeout=120)
    return done.returncode == 0


def _make_restore_dir(persistent_dir: str, rel: str) -> None:
    """Create a declared path that isn't there yet, as this (provisioning) user, not root.

    Directories on the way are plain (755), as the installers' own ``makedirs`` leaves
    them. The declared directory itself gets the ``cloudgpu-app`` group, group write and
    setgid, so the sync user (in that group) can write into it and what it writes keeps
    the group. Nothing is opened through a link: each directory is opened with
    O_NOFOLLOW and changed through that handle.
    """
    parts = datasync._parts(rel)
    current = persistent_dir
    for i, part in enumerate(parts):
        current = os.path.join(current, part)
        last = i == len(parts) - 1
        try:
            os.mkdir(current, 0o755)
        except FileExistsError:
            continue
        if last:
            fd = os.open(current, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fchown(fd, -1, _app_gid())
                os.fchmod(fd, 0o2775)
            finally:
                os.close(fd)


def prepare_restore_paths(persistent_dir: str, paths: list[str]) -> list[str]:
    """The declared directories a restore may write, created where missing.

    A path that is a symlink, or under one, is left out and said (the runner says it
    too and restores nothing there); one that is something else is left for the runner
    to fail on by name.
    """
    writable: list[str] = []
    for path in paths:
        rel = "/".join(datasync._parts(path))
        kind, detail = datasync.classify(persistent_dir, rel)
        if kind == "missing":
            try:
                _make_restore_dir(persistent_dir, rel)
            except OSError as e:
                raise SetupError(f"Could not create {rel} to restore into: "
                                 f"{e.strerror or e}.") from None
            kind, detail = datasync.classify(persistent_dir, rel)
        if kind == "symlink":
            log(f"Not restoring into {rel}: {detail} is a symlink.")
            continue
        if kind != "dir":
            continue
        full = os.path.join(persistent_dir, rel)
        if any(c.isspace() for c in full):
            raise SetupError(f"The synced path {full!r} has a space in it, which the "
                             "restricted restore cannot be limited to.")
        writable.append(full)
    return writable


def restore_run_command(writable: list[str]) -> list[str]:
    """The one-off restricted restore (pure; testable without systemd).

    The upload service's restrictions, as a transient unit, plus ReadWritePaths for the
    declared directories only: the rest of the persistent dir, state.json included,
    stays read-only to it. The key reaches it as the upload's does (LoadCredential).
    UMask 0002 so what rclone creates is group-writable from the start.
    """
    properties = [
        f"User={SYNC_USER}",
        f"Group={SYNC_USER}",
        f"SupplementaryGroups={APP_USER}",
        f"LoadCredential={CREDENTIAL_NAME}:{KEY_PATH}",
        f"StateDirectory={STATE_DIR_NAME}",
        "StateDirectoryMode=0750",
        f"Environment=HOME={STATE_DIR}",
        "UMask=0002",
        f"RuntimeMaxSec={RUN_TIMEOUT}",
        "NoNewPrivileges=yes",
        "ProtectSystem=strict",
        "ProtectHome=yes",
        "PrivateTmp=yes",
        *(f"ReadWritePaths={path}" for path in writable),
    ]
    cmd = ["systemd-run", f"--unit={RESTORE_UNIT_NAME}", "--wait", "--collect", "--pipe",
           "--quiet"]
    for prop in properties:
        cmd += ["-p", prop]
    script = (f"exec /usr/bin/python3 {RUNNER_PATH} restore --no-record --spec {SPEC_PATH} "
              f'--rclone-config "${{CREDENTIALS_DIRECTORY}}/{CREDENTIAL_NAME}"')
    return [*cmd, "/bin/sh", "-c", script]


def share_restored(writable: list[str]) -> None:
    """Give what was restored the group and modes of files the apps wrote themselves
    (the commands of ``service.migrate_ownership``). ``chgrp -h`` and ``chmod -R`` change
    no link's target, and every path here was checked to be a real directory."""
    for path in writable:
        if os.path.islink(path) or not os.path.isdir(path):
            continue
        _sudo("chgrp", "-hR", APP_USER, path)
        _sudo("chmod", "-R", "g+rwX", path)
        _sudo("find", path, "-type", "d", "-exec", "chmod", "g+s", "{}", "+")


def restore(persistent_dir: str) -> None:
    """Restore this filesystem from the bucket if it hasn't been (between setup and start).

    The record in ``<persistent dir>/cloudgpu/state.json`` is read and written here, by
    the provisioning user that owns that file, never by the restricted run.
    """
    try:
        spec = datasync.load_spec(SPEC_PATH)
    except (OSError, ValueError) as e:
        raise SetupError(f"Sync is not set up on this instance, so nothing can be restored "
                         f"({e}).") from None
    remote = spec["remote"].rstrip("/")
    try:
        record = datasync.read_restore_record(persistent_dir)
    except (OSError, ValueError) as e:
        raise SetupError(f"Could not read {datasync.state_file(persistent_dir)}: {e}") from None
    if record is not None and record["remote"] == remote:
        log(f"This filesystem was already restored from {remote} at "
            f"{record.get('at', 'an unrecorded time')}; nothing downloaded.")
        return

    # No upload may run while files are arriving: stop the timer and any run in progress.
    timer_was_active = _timer_active()
    if timer_was_active:
        _sudo("systemctl", "stop", f"{UNIT_NAME}.timer")
    _sudo_quiet("systemctl", "stop", f"{UNIT_NAME}.service")
    writable: list[str] = []
    try:
        writable = prepare_restore_paths(persistent_dir, spec["paths"])
        log(f"Restoring {len(spec['paths'])} paths from {remote}...")
        try:
            code = subprocess.run(["sudo", *restore_run_command(writable)], check=False,
                                  timeout=RESTORE_WAIT_SECONDS).returncode
        finally:
            # Also after a failure: the paths that did restore stay, and must be usable.
            share_restored(writable)
        if code != 0:
            raise SetupError(f"The restore from {remote} failed (the paths are named above). "
                             "Nothing is recorded, so the next up restores again.")
        try:
            at = datasync.write_restore_record(persistent_dir, remote, time.time())
        except (OSError, ValueError) as e:
            raise SetupError(f"The restore copied everything but could not record it in "
                             f"{datasync.state_file(persistent_dir)}: {e}") from None
        log(f"Restored from {remote}; recorded {at}.")
    finally:
        if timer_was_active:
            _sudo("systemctl", "start", f"{UNIT_NAME}.timer")


# --- final sync --------------------------------------------------------------------

def _service_property(name: str) -> str:
    done = subprocess.run(["sudo", "systemctl", "show", f"--property={name}", "--value",
                           f"{UNIT_NAME}.service"],
                          check=False, timeout=120, capture_output=True, text=True)
    return (done.stdout or "").strip()


def _last_success() -> tuple[float, str] | None:
    """(seconds, text as written) of the last good run, or None. Read with sudo: the
    directory is the sync user's."""
    done = subprocess.run(["sudo", "cat", LAST_SUCCESS_FILE], check=False, timeout=120,
                          capture_output=True, text=True)
    text = (done.stdout or "").strip()
    if done.returncode != 0 or not text:
        return None
    try:
        return calendar.timegm(time.strptime(text, "%Y-%m-%dT%H:%M:%SZ")), text
    except ValueError:
        return None


def _final_failure(why: str) -> SetupError:
    last = _last_success()
    then = (f"The last successful sync finished at {last[1]}." if last
            else "There has been no successful sync from this instance.")
    return SetupError(f"{why} {then}")


def final() -> None:
    """One complete upload that started after this was asked for, before termination.

    The timer is stopped first so no later run starts. A run in progress is waited for,
    not joined: it may already have listed directories that changed since. Then the
    service runs once more. After any failure the timer is started again.
    """
    if not all(os.path.exists(p) for p in (SPEC_PATH, service_path(), timer_path())):
        raise SetupError("Sync is not set up on this instance, so there is no final sync "
                         "to do.")
    service = f"{UNIT_NAME}.service"
    _sudo("systemctl", "stop", f"{UNIT_NAME}.timer")
    done = False
    try:
        deadline = time.monotonic() + FINAL_WAIT_SECONDS
        waited = False
        while _service_property("ActiveState") in ("activating", "deactivating"):
            if not waited:
                log("A sync run is in progress; waiting for it to finish...")
                waited = True
            if time.monotonic() >= deadline:
                raise _final_failure("The sync run in progress did not finish in "
                                     f"{RUN_TIMEOUT}, so no final sync was run.")
            time.sleep(FINAL_POLL_SECONDS)
        log("Running the final sync...")
        # Whole seconds, as the last-success time is written.
        started = int(time.time())
        try:
            # Starting a oneshot returns when its run has ended, non-zero if it failed.
            code = subprocess.run(["sudo", "systemctl", "start", service], check=False,
                                  timeout=FINAL_WAIT_SECONDS).returncode
        except subprocess.TimeoutExpired:
            raise _final_failure(f"The final sync did not finish in {RUN_TIMEOUT}.") from None
        result = _service_property("Result")
        if code != 0 or result != "success":
            raise _final_failure(f"The final sync run failed (result: {result or 'unknown'}).")
        last = _last_success()
        if last is None or last[0] < started:
            raise _final_failure("The final sync run ended without recording a success.")
        done = True
        log(f"Final sync finished at {last[1]}. The sync timer stays stopped.")
    finally:
        if not done:
            again = subprocess.run(["sudo", "systemctl", "start", f"{UNIT_NAME}.timer"],
                                   check=False, timeout=120)
            log("The sync timer is running again." if again.returncode == 0
                else "The sync timer could not be started again.")


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
    _sudo_quiet("systemctl", "stop", f"{RESTORE_UNIT_NAME}.service")
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
        elif action == "restore":
            restore(persistent_dir)
        elif action == "final":
            final()
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
