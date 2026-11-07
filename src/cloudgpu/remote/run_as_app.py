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
from collections.abc import Iterable, Mapping

APP_USER = "cloudgpu-app"
COULD_NOT_RUN = 125

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
    under ``check``, and TimeoutExpired, by which time the step has been told to stop.
    """
    stdin = _env_lines(names, env or {})
    sudo = ["sudo", "-n", "-H", "-u", APP_USER, "--"]
    # sudo gets only PATH, so the secrets don't depend on its env_reset to be stripped.
    sudo_env = {"PATH": os.environ.get("PATH", os.defpath)}
    started = _STARTED if capture else ""
    pipes = {"stdout": subprocess.PIPE, "stderr": subprocess.PIPE} if capture else {}
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
        with subprocess.Popen(
            [*sudo, "/bin/sh", "-c", _BOOTSTRAP, "cloudgpu-run-as-app", cwd, started, *cmd],
            cwd="/",  # the caller's cwd may be unreadable to APP_USER
            env=sudo_env, stdin=subprocess.PIPE, text=True, **pipes,
        ) as proc:
            try:
                out, err = proc.communicate(stdin, timeout=timeout)
            except subprocess.TimeoutExpired:
                # sudo passes SIGTERM on to the step. SIGKILL would stop only sudo.
                proc.terminate()
                try:
                    proc.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise
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
