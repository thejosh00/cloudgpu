"""Remote entry point - runs on the Lambda instance via SSH. Stdlib-only.

Usage: python3 <path-to-remote> --persistent-dir <path> <command> [args]
"""

from __future__ import annotations

import argparse
import json
import sys

from .instance import NotOnInstance
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
    if recover_all(state, APP_INSTALLERS):
        sys.exit(1)


def cmd_clear_reprovision(state: State, args: argparse.Namespace) -> None:
    """Drop an app's re-provision mark, once its profiles were provisioned again."""
    state.clear_reprovision(args.app)


def cmd_verify(state: State, args: argparse.Namespace) -> None:
    """Report whether each named service is running (read-only; the caller decides)."""
    from . import service
    settle = service.SETTLE_SECONDS if args.settle is None else args.settle
    print(json.dumps({"services": service.verify_services(args.service, settle=settle)}, indent=2))


def cmd_autoterminate(state: State, args: argparse.Namespace) -> None:
    """Arm (hours > 0) or disarm (hours <= 0) the instance lifetime cap."""
    from . import autoterminate
    if args.hours > 0:
        if not args.instance_id or not args.key_file:
            print("--instance-id and --key-file are required to arm", file=sys.stderr)
            sys.exit(2)
        autoterminate.arm(args.hours, args.instance_id, args.key_file, state.persistent_dir)
    else:
        autoterminate.disarm()


def cmd_datasync(state: State, args: argparse.Namespace) -> None:
    """Put the bucket sync in place (setup), start its timer (start), restore this filesystem
    from the bucket (restore), do the last upload before termination (final), or take the sync
    away (remove)."""
    from . import datasync_setup
    code = datasync_setup.main(args.action, state.persistent_dir, key_file=args.key_file,
                               request_file=args.spec_file, start_timer=args.start)
    if code:
        sys.exit(code)


def cmd_status(state: State, args: argparse.Namespace) -> None:
    """Show status of all installed apps, and of the bucket sync when it is installed."""
    detection = detect_all()
    apps = {}
    for app_name, app_state in state.list_apps().items():
        if app_name in APP_INSTALLERS:
            installer = APP_INSTALLERS[app_name](state)
            apps[app_name] = installer.get_status()
        else:
            apps[app_name] = {**app_state, "status": "unknown_installer"}

    data = {"detection": detection, "apps": apps}
    # Only when sync is installed: otherwise the answer is exactly what it was before 3.4.
    from . import datasync_setup
    sync_info = datasync_setup.sync_status()
    if sync_info is not None:
        data["sync"] = sync_info
    print(json.dumps(data, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="CloudGPU remote tool")
    parser.add_argument("--persistent-dir", required=True, help="Path to persistent directory")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("detect", help="Detect instance environment")

    install_parser = subparsers.add_parser("install", help="Install an app")
    install_parser.add_argument("--app", required=True, choices=list(APP_INSTALLERS.keys()))

    subparsers.add_parser("recover", help="Recover all installed apps")
    subparsers.add_parser("status", help="Show status")

    clear_parser = subparsers.add_parser("clear-reprovision",
                                         help="Drop an app's re-provision mark (after provisioning)")
    clear_parser.add_argument("--app", required=True)

    verify_parser = subparsers.add_parser("verify", help="Report whether services are running")
    verify_parser.add_argument("--service", action="append", required=True)
    verify_parser.add_argument("--settle", type=float, default=None,
                               help="Seconds each must stay up (default: the install settle); 0 looks once")

    at_parser = subparsers.add_parser("autoterminate", help="Arm/disarm the instance lifetime cap")
    at_parser.add_argument("--hours", type=float, required=True, help="Cap in hours; <= 0 disarms")
    at_parser.add_argument("--instance-id", default=None, help="This instance's Lambda id")
    at_parser.add_argument("--key-file", default=None,
                           help="File with a LAMBDA_API_KEY=... line (deleted after reading)")

    ds_parser = subparsers.add_parser("datasync", help="Set up, start, restore from or remove the bucket sync")
    ds_parser.add_argument("action", choices=["setup", "start", "restore", "final", "remove"])
    ds_parser.add_argument("--key-file", default=None,
                           help="Transferred rclone config holding the bucket key (deleted after)")
    ds_parser.add_argument("--spec-file", default=None,
                           help="Transferred JSON: remote, paths, interval_minutes (deleted after)")
    ds_parser.add_argument("--start", action="store_true",
                           help="With setup: also start the timer")

    args = parser.parse_args()
    state = State(args.persistent_dir)

    commands = {
        "detect": cmd_detect,
        "install": cmd_install,
        "recover": cmd_recover,
        "status": cmd_status,
        "clear-reprovision": cmd_clear_reprovision,
        "verify": cmd_verify,
        "autoterminate": cmd_autoterminate,
        "datasync": cmd_datasync,
    }

    try:
        commands[args.command](state, args)
    except NotOnInstance as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
