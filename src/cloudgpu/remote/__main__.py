"""Remote entry point - runs on the Lambda instance via SSH. Stdlib-only.

Usage: python3 <path-to-remote> --persistent-dir <path> <command> [args]
"""

from __future__ import annotations

import argparse
import json
import sys

from .state import State
from .detect import detect_all
from .recovery import setup_path
from .apps.comfyui import ComfyUIInstaller
from .apps.aitoolkit import AIToolkitInstaller

APP_INSTALLERS = {
    "comfyui": ComfyUIInstaller,
    "ai-toolkit": AIToolkitInstaller,
}


def cmd_detect(state: State, args: argparse.Namespace) -> None:
    """Detect instance environment."""
    result = detect_all()
    print(json.dumps(result, indent=2))


def cmd_install(state: State, args: argparse.Namespace) -> None:
    """Install an app."""
    app_name = args.app
    if app_name not in APP_INSTALLERS:
        print(f"Unknown app: {app_name}. Available: {', '.join(APP_INSTALLERS)}", file=sys.stderr)
        sys.exit(1)

    installer = APP_INSTALLERS[app_name](state)
    installer.install()

    # Ensure cloudgpu/bin is on PATH
    setup_path(state)


def cmd_recover(state: State, args: argparse.Namespace) -> None:
    """Recover all installed apps."""
    from .recovery import recover_all
    recover_all(state, APP_INSTALLERS)


def cmd_autoterminate(state: State, args: argparse.Namespace) -> None:
    """Arm (hours > 0) or disarm (hours <= 0) the instance lifetime cap."""
    from . import autoterminate
    if args.hours > 0:
        if not args.instance_id or not args.key_file:
            print("--instance-id and --key-file are required to arm", file=sys.stderr)
            sys.exit(2)
        autoterminate.arm(args.hours, args.instance_id, args.key_file)
    else:
        autoterminate.disarm()


def cmd_status(state: State, args: argparse.Namespace) -> None:
    """Show status of all installed apps."""
    detection = detect_all()
    apps = {}
    for app_name, app_state in state.list_apps().items():
        if app_name in APP_INSTALLERS:
            installer = APP_INSTALLERS[app_name](state)
            apps[app_name] = installer.get_status()
        else:
            apps[app_name] = {**app_state, "status": "unknown_installer"}

    print(json.dumps({"detection": detection, "apps": apps}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="CloudGPU remote tool")
    parser.add_argument("--persistent-dir", required=True, help="Path to persistent directory")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("detect", help="Detect instance environment")

    install_parser = subparsers.add_parser("install", help="Install an app")
    install_parser.add_argument("--app", required=True, choices=list(APP_INSTALLERS.keys()))

    subparsers.add_parser("recover", help="Recover all installed apps")
    subparsers.add_parser("status", help="Show status")

    at_parser = subparsers.add_parser("autoterminate", help="Arm/disarm the instance lifetime cap")
    at_parser.add_argument("--hours", type=float, required=True, help="Cap in hours; <= 0 disarms")
    at_parser.add_argument("--instance-id", default=None, help="This instance's Lambda id")
    at_parser.add_argument("--key-file", default=None,
                           help="File with a LAMBDA_API_KEY=... line (deleted after reading)")

    args = parser.parse_args()
    state = State(args.persistent_dir)

    commands = {
        "detect": cmd_detect,
        "install": cmd_install,
        "recover": cmd_recover,
        "status": cmd_status,
        "autoterminate": cmd_autoterminate,
    }

    commands[args.command](state, args)


if __name__ == "__main__":
    main()
