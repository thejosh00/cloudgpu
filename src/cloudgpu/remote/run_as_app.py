#!/usr/bin/env python3
"""Run one provisioning step as cloudgpu-app, with only the environment variables it names.

    python3 "$CLOUDGPU_RUN_AS_APP" --pass HF_TOKEN --pass HF_HOME -- <venv>/bin/python3 -c "..."

cloudgpu-app can write the app venvs and app_dir, and ubuntu has passwordless sudo and holds every
secret during provisioning. So anything from those directories must run through this, never
directly. Provision scripts reach it through CLOUDGPU_RUN_AS_APP, which `cloudgpu up` sets; it's a
script rather than a function because comfylib.py is vendored and not every profile has a copy.
Stdlib-only.

Exits with the step's own exit code, or COULD_NOT_RUN if the step never started. There's no
fallback to running as the caller.

Named variables travel on stdin to a small shell that sets them and then execs the step, for
three reasons: sudo strips the environment, argv would show the values in the process list, and
--preserve-env depends on the instance's sudoers. If they don't arrive, the end marker doesn't
either, and the step doesn't run. The same shell sets the umask, because sudo applies its own
(022 by default), and without group write `pip_install` as ubuntu breaks on the next `up`. It also
does the `cd`, so the caller doesn't need to be able to enter a directory cloudgpu-app owns.

The installers call `app_step` directly, for what their steps need beyond that: a working
directory, plain settings like PATH (sudo may reset it, and `npm` starts with `#!/usr/bin/env
node`), a timeout, and captured output.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping

APP_USER = "cloudgpu-app"
COULD_NOT_RUN = 125
_STOP_GRACE = 10  # seconds a step that timed out gets between SIGTERM and SIGKILL

_END = "cloudgpu-env-end"
_STARTED = "cloudgpu-step-started"
_BOOTSTRAP = f"""umask 002
cd "$1" || exit {COULD_NOT_RUN}
started=$2
shift 2
while IFS= read -r line; do
  if [ "$line" = {_END} ]; then
    [ -z "$started" ] || echo "$started"
    exec "$@"
  fi
  export "$line" || exit {COULD_NOT_RUN}
done
echo "run_as_app: the named variables didn't arrive through sudo; the step did not run" >&2
exit {COULD_NOT_RUN}
"""


class CouldNotRun(RuntimeError):
    """The step never started. The message must never contain a variable's value."""


def _step_groups(sudo_pid: int) -> set[int]:
    """The process groups of sudo and of everything under it, apart from the caller's own.

    A step's children stay in its group even once their parent has gone, so a group still finds
    them. sudo's own group is always there: sudo runs in one of its own (see run_stoppable), and run
    without a pty it leaves the step in it. With a pty the step is in a group further down.
    """
    groups = {sudo_pid}
    try:
        table = subprocess.run(["ps", "-A", "-o", "pid=", "-o", "ppid=", "-o", "pgid="],
                               capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        table = ""
    children: dict[int, list[int]] = {}
    group_of: dict[int, int] = {}
    for line in table.splitlines():
        fields = line.split()
        if len(fields) == 3 and all(f.isdigit() for f in fields):
            pid, ppid, pgid = map(int, fields)
            children.setdefault(ppid, []).append(pid)
            group_of[pid] = pgid
    todo, seen = [sudo_pid], set()
    while todo:
        pid = todo.pop()
        if pid not in seen:
            seen.add(pid)
            groups.add(group_of.get(pid, sudo_pid))
            todo.extend(children.get(pid, ()))
    return {g for g in groups if g > 1 and g != os.getpgrp()}


def _groups_running(groups: set[int]) -> bool:
    """Whether any process in ``groups`` is still running (a zombie has stopped)."""
    try:
        table = subprocess.run(["ps", "-A", "-o", "pgid=", "-o", "stat="],
                               capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    for line in table.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) in groups \
                and not fields[1].startswith("Z"):
            return True
    return False


def _stop_step(proc: subprocess.Popen) -> None:
    """Stop everything the step started, its children included, before app_step gives up on it.

    Signalling sudo isn't enough. Python started it in Python's own process group, and sudo
    1.9.9 (Ubuntu 22.04) drops any signal from a process in its own group, so the SIGTERM never
    reached the step; SIGKILL can't be relayed at all and stopped only sudo. Even relayed, a
    signal reaches only the step's first process, never pip's children, and a step may ignore it.
    So the step's process groups are signalled as APP_USER (the caller can't signal its
    processes), SIGTERM first and SIGKILL after _STOP_GRACE.
    """
    groups = _step_groups(proc.pid)
    kill = ["sudo", "-n", "-u", APP_USER, "--", "kill"]
    sudo_env = {"PATH": os.environ.get("PATH", os.defpath)}

    def signal_groups(name: str) -> None:
        if not groups:
            return
        try:
            subprocess.run([*kill, f"-{name}", "--", *(f"-{g}" for g in sorted(groups))],
                           cwd="/", env=sudo_env, stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        except (OSError, subprocess.SubprocessError):
            pass  # sudo's own relaying and the SIGKILL to sudo below are all that's left

    def wait_until_stopped(seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while True:
            proc.poll()  # reap sudo, so it doesn't count as running
            if not _groups_running(groups):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)

    proc.terminate()  # sudo relays it now that the signal comes from outside its group
    signal_groups("TERM")
    if not wait_until_stopped(_STOP_GRACE):
        signal_groups("KILL")
        proc.kill()
        wait_until_stopped(5)
    proc.wait()


def run_stoppable(argv: list[str], *, input: str | None = None, timeout: float | None = None,
                  env: Mapping[str, str] | None = None,
                  capture: bool = False) -> subprocess.CompletedProcess:
    """Run a ``sudo -u APP_USER`` command line from /, as subprocess.run (without ``check``),
    except that a timeout or ^C first stops everything it started (see _stop_step)."""
    pipes = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE} if capture else {}
    with subprocess.Popen(
        argv, cwd="/",  # the caller's cwd may be unreadable to APP_USER
        env=env, stdin=subprocess.PIPE if input is not None else None, text=True, **pipes,
        # Its own process group, so sudo relays our signals and the step's group isn't ours.
        preexec_fn=os.setpgrp,
    ) as proc:
        try:
            out, err = proc.communicate(input, timeout=timeout)
        except BaseException:
            # Timed out, or interrupted: a terminal's ^C no longer reaches the step's group.
            _stop_step(proc)
            raise
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


def _env_lines(names: Iterable[str], env: Mapping[str, str]) -> str:
    values = {name: os.environ.get(name) for name in names}
    values.update(env)
    lines = []
    for name, value in values.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise CouldNotRun(f"'{name}' isn't a variable name")
        if not value:
            raise CouldNotRun(f"{name} was named for this step but isn't set")
        if "\n" in value:
            raise CouldNotRun(f"{name} contains a newline, so it can't be passed")
        lines.append(f"{name}={value}\n")
    return "".join(lines) + _END + "\n"


def app_step(
    cmd: list[str],
    *,
    cwd: str = "/",
    names: Iterable[str] = (),
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run ``cmd`` as APP_USER in ``cwd``. Nothing of the caller's environment comes along.

    ``names`` are variables to copy from the caller, ``env`` is settings given here. Raises
    CouldNotRun if the step never started, so with ``capture`` a non-zero exit is always the
    step's own verdict and never a refused sudo. Otherwise as subprocess.run: CalledProcessError
    under ``check``, and TimeoutExpired, by which time the step and all it started have stopped.
    """
    stdin = _env_lines(names, env or {})
    sudo = ["sudo", "-n", "-H", "-u", APP_USER, "--"]
    # sudo gets only PATH, so the secrets don't depend on its env_reset to be stripped.
    sudo_env = {"PATH": os.environ.get("PATH", os.defpath)}
    started = _STARTED if capture else ""
    try:
        # A separate probe, so a refused sudo can't be read as the step's own failure.
        probe = subprocess.run(
            [*sudo, "/bin/sh", "-c", "exit 0"], cwd="/", env=sudo_env,
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
        )
        if probe.returncode != 0:
            raise CouldNotRun(
                f"couldn't run as '{APP_USER}': {probe.stderr.strip() or 'sudo failed'}. "
                "Installing or recovering an app creates that user, so check that this "
                "profile's cloudgpu.toml lists the app in `apps`."
            )
        proc = run_stoppable(
            [*sudo, "/bin/sh", "-c", _BOOTSTRAP, "cloudgpu-run-as-app", cwd, started, *cmd],
            input=stdin, timeout=timeout, env=sudo_env, capture=capture)
        out, err = proc.stdout, proc.stderr
    except OSError as e:
        raise CouldNotRun(f"couldn't run as '{APP_USER}': {e}") from e
    if capture:
        if not out.startswith(_STARTED + "\n"):
            raise CouldNotRun(f"couldn't run as '{APP_USER}': {err.strip() or 'no output'}")
        out = out[len(_STARTED) + 1:]
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, out, err)
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def run_as_app(names: list[str], cmd: list[str]) -> int:
    """Run ``cmd`` as APP_USER and return its exit code. Raises CouldNotRun if it never started."""
    return app_step(cmd, names=names, check=False).returncode


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run a step as cloudgpu-app.")
    parser.add_argument("--pass", dest="names", action="append", default=[], metavar="NAME",
                        help="pass this environment variable to the step (repeatable)")
    if "--" not in argv or argv.index("--") == len(argv) - 1:
        parser.error("give the step after --")
    split = argv.index("--")
    args = parser.parse_args(argv[:split])
    try:
        return run_as_app(args.names, argv[split + 1:])
    except CouldNotRun as e:
        print(f"run_as_app: {e}", file=sys.stderr, flush=True)
        return COULD_NOT_RUN


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
