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
(022 by default), and without group write `pip_install` as ubuntu breaks on the next `up`.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

APP_USER = "cloudgpu-app"
COULD_NOT_RUN = 125

_END = "cloudgpu-env-end"
_BOOTSTRAP = f"""umask 002
while IFS= read -r line; do
  if [ "$line" = {_END} ]; then exec "$@"; fi
  export "$line" || exit {COULD_NOT_RUN}
done
echo "run_as_app: the named variables didn't arrive through sudo; the step did not run" >&2
exit {COULD_NOT_RUN}
"""


class CouldNotRun(RuntimeError):
    """The step never started. The message must never contain a variable's value."""


def _env_lines(names: list[str]) -> str:
    lines = []
    for name in names:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise CouldNotRun(f"'{name}' isn't a variable name")
        value = os.environ.get(name)
        if not value:
            raise CouldNotRun(f"{name} was named for this step but isn't set")
        if "\n" in value:
            raise CouldNotRun(f"{name} contains a newline, so it can't be passed")
        lines.append(f"{name}={value}\n")
    return "".join(lines) + _END + "\n"


def run_as_app(names: list[str], cmd: list[str]) -> int:
    """Run ``cmd`` as APP_USER and return its exit code. Raises CouldNotRun if it never started."""
    stdin = _env_lines(names)
    sudo = ["sudo", "-n", "-H", "-u", APP_USER, "--"]
    # sudo gets only PATH, so the secrets don't depend on its env_reset to be stripped.
    env = {"PATH": os.environ.get("PATH", os.defpath)}
    try:
        # A separate probe, so a refused sudo can't be read as the step's own failure.
        probe = subprocess.run(
            [*sudo, "/bin/sh", "-c", "exit 0"], cwd="/", env=env,
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
        )
        if probe.returncode != 0:
            raise CouldNotRun(
                f"couldn't run as '{APP_USER}': {probe.stderr.strip() or 'sudo failed'}. "
                "Installing or recovering an app creates that user, so check that this "
                "profile's cloudgpu.toml lists the app in `apps`."
            )
        return subprocess.run(
            [*sudo, "/bin/sh", "-c", _BOOTSTRAP, "cloudgpu-run-as-app", *cmd],
            cwd="/",  # the caller's cwd may be unreadable to APP_USER
            env=env, input=stdin, text=True,
        ).returncode
    except OSError as e:
        raise CouldNotRun(f"couldn't run as '{APP_USER}': {e}") from e


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
