"""Upload the synced paths to the person's bucket. Stdlib-only; runs on the instance.

One run copies each declared path with ``rclone copy <persistent dir>/<path>
<remote>/<path>``. It only ever uploads: ``copy``, never ``sync``, and no flag that
deletes, so removing a file on the instance removes nothing from the bucket.

Everything it needs arrives as data, in a JSON spec file::

    {"remote": "b2:bkt/small", "persistent_dir": "/lambda/nfs/fs", "paths": ["apps/x/output"],
     "rclone": "/path/to/rclone", "rclone_config": "/path/to/rclone.conf",
     "last_success_file": "/path/to/last-success"}

The last three may be given as arguments instead (an argument wins). The paths are
worked out on the person's machine and are not recomputed here. Nothing here knows which
storage provider the remote is. The bucket key is only ever inside the rclone config
file: this module passes that file's path to rclone and never opens it.

Symlinks are never followed: app code can write inside the synced directories and could
point a link at any file the sync user can read. rclone ignores links inside a path by
default (no ``--copy-links``, no ``--links``); the declared path itself and the
directories above it inside the persistent dir are checked here, and a path with a link
in it is skipped and named.

This file must stay standalone (stdlib imports only, no package-relative imports): it
runs from a root-owned copy outside the package, as a user that is not root, via
``python3 <copy> run --spec <file>``.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import tempfile
import time

# --fast-list: one recursive listing of the bucket side of a path (one call per 1,000
# files) instead of one listing per directory. It changes how the destination is listed,
# not what is copied.
RCLONE_FLAGS = ("--fast-list",)


def log(msg: str) -> None:
    print(f">>> {msg}", file=sys.stderr, flush=True)


def path_problem(path: object) -> str | None:
    """Why a declared path is refused, or None when it is acceptable."""
    if not isinstance(path, str) or not path.strip("/"):
        return "not a path"
    if path.startswith("/"):
        return "absolute"
    parts = path.split("/")
    if ".." in parts:
        return "contains '..'"
    if path.startswith("-"):
        return "starts with '-'"
    return None


def _parts(path: str) -> list[str]:
    return [p for p in path.split("/") if p not in ("", ".")]


def classify(persistent_dir: str, path: str) -> tuple[str, str]:
    """What is at ``path`` inside the persistent dir, without following any link.

    Returns (kind, detail): ``("dir", "")`` when every component is a real directory;
    ``("symlink", <the relative path of the link>)``; ``("missing", "")``; or
    ``("other", <what>)`` when a component exists but is not a directory.
    """
    current = persistent_dir
    walked: list[str] = []
    for part in _parts(path):
        current = os.path.join(current, part)
        walked.append(part)
        try:
            st = os.lstat(current)
        except FileNotFoundError:
            return "missing", ""
        except OSError as e:
            return "other", f"{'/'.join(walked)}: {e.strerror or e}"
        if stat.S_ISLNK(st.st_mode):
            return "symlink", "/".join(walked)
        if not stat.S_ISDIR(st.st_mode):
            return "other", f"{'/'.join(walked)} is not a directory"
    return "dir", ""


def has_files(directory: str) -> bool:
    """True if there is at least one regular file under ``directory`` (links not followed)."""
    for root, _dirs, files in os.walk(directory, followlinks=False):
        for name in files:
            try:
                if stat.S_ISREG(os.lstat(os.path.join(root, name)).st_mode):
                    return True
            except OSError:
                continue
    return False


def rclone_command(rclone: str, rclone_config: str, source: str, dest: str) -> list[str]:
    """The one rclone call for one path (pure; testable without rclone)."""
    return [rclone, "copy", "--config", rclone_config, *RCLONE_FLAGS, source, dest]


def _child_env() -> dict[str, str]:
    """The environment without RCLONE_* variables: each one is an rclone flag in disguise
    (RCLONE_COPY_LINKS=true would follow links), so none is passed on."""
    return {k: v for k, v in os.environ.items() if not k.upper().startswith("RCLONE_")}


def write_last_success(path: str, when: float) -> str:
    """Write the finish time (UTC, ISO-8601) atomically. Returns the text written."""
    text = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(when))
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".last-success.", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return text


def load_spec(spec_file: str) -> dict:
    with open(spec_file) as f:
        spec = json.load(f)
    if not isinstance(spec, dict):
        raise ValueError("the spec is not a JSON object")
    remote = spec.get("remote")
    if not isinstance(remote, str) or ":" not in remote or remote.startswith("-"):
        raise ValueError("'remote' must look like name:bucket/path")
    persistent_dir = spec.get("persistent_dir")
    if not isinstance(persistent_dir, str) or not persistent_dir.startswith("/"):
        raise ValueError("'persistent_dir' must be an absolute path")
    if not isinstance(spec.get("paths"), list):
        raise ValueError("'paths' must be a list")
    return spec


def run(
    spec_file: str,
    rclone: str | None = None,
    rclone_config: str | None = None,
    last_success_file: str | None = None,
) -> int:
    """One sync run (the timer's entry point). 0 when nothing failed, 1 otherwise."""
    try:
        spec = load_spec(spec_file)
    except (OSError, ValueError) as e:
        log(f"no valid sync spec at {spec_file}: {e}")
        return 1
    settings = {
        "rclone": rclone or spec.get("rclone"),
        "rclone_config": rclone_config or spec.get("rclone_config"),
        "last_success_file": last_success_file or spec.get("last_success_file"),
    }
    for name, value in settings.items():
        if not isinstance(value, str) or not value.startswith("/"):
            log(f"sync spec {spec_file}: '{name}' must be an absolute path "
                f"(in the spec or as --{name.replace('_', '-')})")
            return 1

    remote = spec["remote"].rstrip("/")
    persistent_dir = spec["persistent_dir"]
    copied: list[str] = []
    failed: list[str] = []
    linked: list[str] = []
    env = _child_env()

    for path in spec["paths"]:
        problem = path_problem(path)
        if problem:
            log(f"refused {path!r}: {problem}; a synced path must be relative and inside "
                "the persistent dir")
            failed.append(str(path))
            continue
        rel = "/".join(_parts(path))
        kind, detail = classify(persistent_dir, rel)
        if kind == "missing":
            continue
        if kind == "symlink":
            log(f"skipped {rel}: {detail} is a symlink, and links are not followed")
            linked.append(rel)
            continue
        if kind == "other":
            log(f"failed {rel}: {detail}")
            failed.append(rel)
            continue
        source = os.path.join(persistent_dir, rel)
        if not has_files(source):
            continue  # nothing to upload, so don't spend a listing call on it
        cmd = rclone_command(settings["rclone"], settings["rclone_config"], source, f"{remote}/{rel}")
        try:
            code = subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL, check=False).returncode
        except OSError as e:
            log(f"failed {rel}: could not run {settings['rclone']}: {e.strerror or e}")
            failed.append(rel)
            continue
        if code != 0:
            log(f"failed {rel}: rclone exited {code}")
            failed.append(rel)
        else:
            copied.append(rel)

    if failed:
        log(f"sync failed for: {', '.join(failed)} ({len(copied)} copied). "
            "The last-success time is unchanged.")
        return 1
    if linked:
        log(f"sync left out symlinked paths: {', '.join(linked)} ({len(copied)} copied). "
            "The last-success time is unchanged until they are real directories again.")
        return 0
    try:
        finished = write_last_success(settings["last_success_file"], time.time())
    except OSError as e:
        log(f"sync copied everything but could not record it in "
            f"{settings['last_success_file']}: {e.strerror or e}")
        return 1
    log(f"sync ok: {len(copied)} copied to {remote}, finished {finished}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="datasync.py")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="copy the declared paths to the bucket once")
    p.add_argument("--spec", required=True, help="JSON spec file")
    p.add_argument("--rclone", help="rclone binary (overrides the spec)")
    p.add_argument("--rclone-config", help="rclone config file (overrides the spec)")
    p.add_argument("--last-success-file", help="where to record the finish time (overrides the spec)")
    args = parser.parse_args(argv)
    return run(args.spec, args.rclone, args.rclone_config, args.last_success_file)


if __name__ == "__main__":
    sys.exit(main())
