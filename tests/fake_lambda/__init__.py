"""A fake of Lambda Cloud's API, so `cloudgpu up` can run with no GPU and no network.

Standard library only, on an ephemeral loopback port. A test says what each launch does
and then asserts on what the fake saw: every launch request, and which instances are alive.

Shapes come from Lambda's published spec (https://docs.lambda.com/api/cloud/spec.json).
Assumptions it doesn't settle are marked ASSUMPTION below.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

API_BASE_ENV = "CLOUDGPU_TEST_LAMBDA_API_BASE"
SHIMS = Path(__file__).parent / "shims"

# What a launch can do. `start_*` modes leave a live instance behind the failed reply.
LAUNCH_MODES = {
    "ok",
    "start_then_502",  # ASSUMPTION: a gateway error can follow a launch that took effect
    "start_then_drop",  # ASSUMPTION: as above, with the connection closed and no reply
    "insufficient_capacity",  # published: HTTP 400, instance-operations/launch/insufficient-capacity
    "rate_limited",  # published: HTTP 429, global/rate-limited
    "bad_request",  # published: HTTP 400, global/invalid-parameters
}


class FakeLambda:
    """The endpoints `up` calls. Use as a context manager; `base_url` is what to point at."""

    def __init__(self, *, launches=(), then="ok", boot_polls=0, filesystem="box-fs", region="us-fake-1"):
        """
        Args:
            launches: what each launch does, in order (see LAUNCH_MODES).
            then: what every launch after those does.
            boot_polls: how many reads a launched instance answers as booting, with no ip.
            filesystem: a filesystem that already exists, or None for none.
            region: the one region, which always has capacity for gpu_1x_a10.
        """
        unknown = {*launches, then} - LAUNCH_MODES
        if unknown:
            raise ValueError(f"unknown launch modes: {sorted(unknown)}")
        self.launch_script = list(launches)
        self.then = then
        self.boot_polls = boot_polls
        self.region = region
        self.filesystems = [filesystem] if filesystem else []
        self.launches: list[dict] = []  # every launch body received, whatever it then did
        self.requests: list[tuple[str, str]] = []  # every (method, path) received
        self.instances: list[dict] = []
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(self))
        self._server.daemon_threads = True

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/api/v1"

    def __enter__(self) -> "FakeLambda":
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()

    def alive(self) -> list[str]:
        """Ids of every instance that would be billing. The fake never terminates one."""
        with self._lock:
            return [inst["id"] for inst in self.instances]

    def add_instance(self, *, name="box", boot_polls=0) -> str:
        """An instance already on the filesystem, as if an earlier `up` had launched it."""
        with self._lock:
            return self._start(name, list(self.filesystems), boot_polls)

    def _start(self, name, file_systems, boot_polls) -> str:
        n = len(self.instances) + 1
        self.instances.append(
            {
                "id": f"fake-instance-{n}",
                "name": name,
                "file_system_names": file_systems,
                "ssh_key_names": ["fake-key"],
                "region": {"name": self.region, "description": "Fake region"},
                "instance_type": {"name": "gpu_1x_a10", "price_cents_per_hour": 75},
                "_ip": f"192.0.2.{n}",  # TEST-NET-1: routes nowhere, and only the shims see it
                "_booting_for": boot_polls,
            }
        )
        return self.instances[-1]["id"]

    def _read(self, inst: dict) -> dict:
        """The instance as a read sees it. Each read while booting uses up one boot poll."""
        public = {k: v for k, v in inst.items() if not k.startswith("_")}
        if inst["_booting_for"] > 0:
            inst["_booting_for"] -= 1
            return {**public, "status": "booting"}
        return {**public, "status": "active", "ip": inst["_ip"]}

    def _launch(self, body: dict) -> tuple[str, list[str]]:
        """Record a launch. Returns its mode, and the instances it started."""
        self.launches.append(body)
        mode = self.launch_script.pop(0) if self.launch_script else self.then
        if mode in ("ok", "start_then_502", "start_then_drop"):
            # ASSUMPTION: a launch on a filesystem already in use is accepted. The spec doesn't
            # say, and only lists filesystem-in-use for deleting one.
            return mode, [self._start(body.get("name"), body.get("file_system_names") or [], self.boot_polls)]
        return mode, []


def _handler_for(fake: FakeLambda) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def _send(self, status: int, payload: dict | None = None, *, raw: bytes = b"", headers=()) -> None:
            body = json.dumps(payload).encode() if payload is not None else raw
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            for key, value in headers:
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, code: str, message: str, headers=()) -> None:
            self._send(status, {"error": {"code": code, "message": message}}, headers=headers)

        def do_GET(self) -> None:
            with fake._lock:
                fake.requests.append(("GET", self.path))
                if self.path == "/api/v1/instance-types":
                    self._send(200, {"data": {"gpu_1x_a10": {
                        "instance_type": {"name": "gpu_1x_a10", "price_cents_per_hour": 75},
                        "regions_with_capacity_available": [{"name": fake.region, "description": "Fake region"}],
                    }}})
                elif self.path == "/api/v1/instances":
                    self._send(200, {"data": [fake._read(i) for i in fake.instances]})
                elif self.path.startswith("/api/v1/instances/"):
                    wanted = self.path.rsplit("/", 1)[1]
                    found = [i for i in fake.instances if i["id"] == wanted]
                    if found:
                        self._send(200, {"data": fake._read(found[0])})
                    else:
                        self._error(404, "global/object-does-not-exist", "Instance not found.")
                elif self.path == "/api/v1/file-systems":
                    self._send(200, {"data": [
                        {"id": f"fs-{name}", "name": name, "region": {"name": fake.region}} for name in fake.filesystems
                    ]})
                elif self.path == "/api/v1/ssh-keys":
                    self._send(200, {"data": [{"id": "k1", "name": "fake-key", "public_key": "ssh-ed25519 FAKE"}]})
                else:
                    self._error(404, "global/object-does-not-exist", f"Fake has no GET {self.path}")

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            with fake._lock:
                fake.requests.append(("POST", self.path))
                if self.path == "/api/v1/filesystems":
                    fake.filesystems.append(body["name"])
                    self._send(200, {"data": {"id": f"fs-{body['name']}", "name": body["name"]}})
                elif self.path == "/api/v1/instance-operations/launch":
                    mode, started = fake._launch(body)
                    if mode == "ok":
                        self._send(200, {"data": {"instance_ids": started}})
                    elif mode == "start_then_502":
                        self._send(502, raw=b"<html>502 Bad Gateway</html>")
                    elif mode == "start_then_drop":
                        self.close_connection = True
                        self.connection.shutdown(socket.SHUT_RDWR)
                    elif mode == "insufficient_capacity":
                        self._error(400, "instance-operations/launch/insufficient-capacity",
                                    "Not enough capacity to fulfill launch request.")
                    elif mode == "rate_limited":
                        # ASSUMPTION: the spec lists the 429 but no Retry-After header.
                        self._error(429, "global/rate-limited", "Too many requests.", headers=[("Retry-After", "0")])
                    else:
                        self._error(400, "global/invalid-parameters", "Invalid request.")
                else:
                    self._error(404, "global/object-does-not-exist", f"Fake has no POST {self.path}")

    return Handler


@dataclass
class UpResult:
    returncode: int
    output: str
    state: dict  # the config's local state after the run; {} if none
    shim_calls: list[dict]  # every ssh and rsync call, in order


def run_up(fake: FakeLambda, workdir: Path, *, poll_seconds=0, max_hours=1.0, timeout=60, base_url=None, state=None) -> UpResult:
    """Run the real `cloudgpu up` in a subprocess against the fake, with ssh and rsync shimmed.

    The environment is built from nothing, with a throwaway HOME, so no real key, ~/.ssh or
    secrets file is in reach. The proxy variables point at a closed port: if the override
    were ever ignored, the request would fail here rather than reach the real API.
    """
    (workdir / "home").mkdir(parents=True, exist_ok=True)
    configs = workdir / "configs"
    configs.mkdir(exist_ok=True)
    (configs / "box.toml").write_text(
        f'filesystem = "{fake.filesystems[0] if fake.filesystems else "box-fs"}"\n'
        'gpu = ["a10"]\nssh_key = "fake-key"\n'
        f"poll_seconds = {poll_seconds}\nmax_hours = {max_hours}\n"
    )
    state_file = configs / "box.state.json"
    if state is not None:
        state_file.write_text(json.dumps(state))
    shim_log = workdir / "shim-calls.jsonl"
    env = {
        "PATH": f"{SHIMS}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(workdir / "home"),
        "COLUMNS": "200",
        "LAMBDA_API_KEY": "dummy-key-not-real",
        API_BASE_ENV: base_url or fake.base_url,
        "http_proxy": "http://127.0.0.1:9",
        "https_proxy": "http://127.0.0.1:9",
        "FAKE_SHIM_LOG": str(shim_log),
        "FAKE_SHIM_FILESYSTEM": fake.filesystems[0] if fake.filesystems else "box-fs",
    }
    proc = subprocess.run(
        [sys.executable, "-c", "from cloudgpu.cli import cli; cli()", "up", "--config", "box"],
        cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout,
    )
    return UpResult(
        returncode=proc.returncode,
        output=proc.stdout + proc.stderr,
        state=json.loads(state_file.read_text()) if state_file.exists() else {},
        shim_calls=[json.loads(line) for line in shim_log.read_text().splitlines()] if shim_log.exists() else [],
    )
