"""A fake of Lambda Cloud's API, so `cloudgpu up` can run with no GPU and no network.

Standard library only, on an ephemeral loopback port. A test says what each launch does
and then asserts on what the fake saw: every launch request, and which instances are alive.

Shapes come from Lambda's published spec (https://docs.lambda.com/api/cloud/spec.json).
Assumptions it doesn't settle are marked ASSUMPTION below.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
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
    "start_then_200_html",  # ASSUMPTION: as above, with a proxy's page and a 200
    "start_then_stall",  # ASSUMPTION: as above, with headers and part of the body, then nothing
    "insufficient_capacity",  # published: HTTP 400, instance-operations/launch/insufficient-capacity
    "rate_limited",  # published: HTTP 429, global/rate-limited
    "bad_request",  # published: HTTP 400, global/invalid-parameters
}

# What a read of an instance or of the list can do instead of answering normally.
READ_MODES = {
    "ok",
    "http_500",  # ASSUMPTION: the spec lists no 5xx for reads, but gateways answer them anyway
    "http_502",  # ASSUMPTION: as above, with an HTML body
    "http_503",  # ASSUMPTION: as above
    "http_500_not_found_code",  # ASSUMPTION: the not-found code on a status that isn't 404
    "http_429",  # published: global/rate-limited
    "http_401",  # published: global/invalid-api-key
    "http_403",  # published: global/forbidden
    "not_found",  # published: HTTP 404, global/object-does-not-exist, for GET /instances/{id}
    "404_other_code",  # published: HTTP 404, global/not-found, a resource or route not found
    "404_no_code",  # ASSUMPTION: a proxy's HTML 404
    "200_html",  # ASSUMPTION: a proxy's page with a 200
    "200_no_status",  # ASSUMPTION: an instance with no status field, or a list entry with none
    "200_unlisted_status",  # ASSUMPTION: a status InstanceStatus doesn't list
    "200_not_a_list",  # ASSUMPTION: a list read that answers with an object
    "drop",  # ASSUMPTION: the connection closes with no reply
    "stall",  # ASSUMPTION: headers and part of the body, then nothing
}


class FakeLambda:
    """The endpoints `up` calls. Use as a context manager; `base_url` is what to point at."""

    def __init__(
        self, *, launches=(), then="ok", boot_polls=0, filesystem="box-fs", region="us-fake-1",
        instance_reads=(), instance_reads_then="ok", list_reads=(), list_reads_then="ok",
        boot_ends="active",
    ):
        """
        Args:
            launches: what each launch does, in order (see LAUNCH_MODES).
            then: what every launch after those does.
            boot_polls: how many reads a launched instance answers as booting, with no ip.
            filesystem: a filesystem that already exists, or None for none.
            region: the one region, which always has capacity for gpu_1x_a10.
            instance_reads: what each GET /instances/{id} does, in order (see READ_MODES).
            instance_reads_then: what every read after those does.
            list_reads: what each GET /instances does, in order.
            list_reads_then: what every list after those does.
            boot_ends: the status a launched instance has once its boot polls are used up:
                "active" (with an ip), or "terminated" / "error" (no ip) for one that died booting.
        """
        unknown = {*launches, then} - LAUNCH_MODES
        if unknown:
            raise ValueError(f"unknown launch modes: {sorted(unknown)}")
        unknown = {*instance_reads, instance_reads_then, *list_reads, list_reads_then} - READ_MODES
        if unknown:
            raise ValueError(f"unknown read modes: {sorted(unknown)}")
        self.instance_read_script = list(instance_reads)
        self.instance_reads_then = instance_reads_then
        self.list_read_script = list(list_reads)
        self.list_reads_then = list_reads_then
        self._stop = threading.Event()
        self.launch_script = list(launches)
        self.then = then
        self.boot_polls = boot_polls
        if boot_ends not in ("active", "terminated", "error"):
            raise ValueError(f"unknown boot_ends: {boot_ends!r}")
        self.boot_ends = boot_ends
        # Called with each GET's path before it's answered, outside the lock (a test can look
        # at the state file at that moment).
        self.on_get = None
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
        self._stop.set()
        self._server.shutdown()
        self._server.server_close()

    def alive(self) -> list[str]:
        """Ids of every instance that would be billing: any not terminated."""
        with self._lock:
            return [inst["id"] for inst in self.instances if inst["_status"] != "terminated"]

    def count(self, method: str, path_end: str) -> int:
        """How many requests of this method had a path ending like this."""
        with self._lock:
            return sum(1 for m, p in self.requests if m == method and p.endswith(path_end))

    def add_instance(self, *, name="box", boot_polls=0, status=None, ip=True) -> str:
        """An instance already on the filesystem, as if an earlier `up` had launched it.

        Args:
            status: if set, every read answers this InstanceStatus and boot_polls doesn't apply.
            ip: whether a read with `status` set shows an ip.
        """
        with self._lock:
            inst_id = self._start(name, list(self.filesystems), boot_polls)
            self.instances[-1]["_status"] = status
            self.instances[-1]["_shows_ip"] = ip
            return inst_id

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
                "_status": None,
                "_shows_ip": True,
            }
        )
        return self.instances[-1]["id"]

    def _read(self, inst: dict) -> dict:
        """The instance as a read sees it. Each read while booting uses up one boot poll."""
        public = {k: v for k, v in inst.items() if not k.startswith("_")}
        if inst["_status"]:
            return {**public, "status": inst["_status"], **({"ip": inst["_ip"]} if inst["_shows_ip"] else {})}
        if inst["_booting_for"] > 0:
            inst["_booting_for"] -= 1
            return {**public, "status": "booting"}
        if self.boot_ends != "active":
            inst["_status"], inst["_shows_ip"] = self.boot_ends, False
            return {**public, "status": self.boot_ends}
        return {**public, "status": "active", "ip": inst["_ip"]}

    def _launch(self, body: dict) -> tuple[str, list[str]]:
        """Record a launch. Returns its mode, and the instances it started."""
        self.launches.append(body)
        mode = self.launch_script.pop(0) if self.launch_script else self.then
        if mode in ("ok", "start_then_502", "start_then_drop", "start_then_200_html", "start_then_stall"):
            # ASSUMPTION: a launch on a filesystem already in use is accepted. The spec doesn't
            # say, and only lists filesystem-in-use for deleting one.
            return mode, [self._start(body.get("name"), body.get("file_system_names") or [], self.boot_polls)]
        return mode, []

    def _read_mode(self, path: str) -> str | None:
        """What this GET should do, using up a step of its script. None if it has no script."""
        if path == "/api/v1/instances":
            return self.list_read_script.pop(0) if self.list_read_script else self.list_reads_then
        if path.startswith("/api/v1/instances/"):
            return self.instance_read_script.pop(0) if self.instance_read_script else self.instance_reads_then
        return None


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

        def _answer_read_mode(self, mode: str) -> None:
            """Answer as this read mode says. Anything but ok finishes the request."""
            if mode == "http_500":
                self._send(500, raw=b"Internal Server Error")
            elif mode == "http_500_not_found_code":
                self._error(500, "global/object-does-not-exist", "Specified instance does not exist.")
            elif mode == "http_502":
                self._send(502, raw=b"<html>502 Bad Gateway</html>")
            elif mode == "http_503":
                self._error(503, "global/internal-error", "Unavailable.")
            elif mode == "http_429":
                self._error(429, "global/rate-limited", "Too many requests.", headers=[("Retry-After", "0")])
            elif mode == "http_401":
                self._error(401, "global/invalid-api-key", "API key was invalid, expired, or deleted.")
            elif mode == "http_403":
                self._error(403, "global/forbidden", "Forbidden.")
            elif mode == "not_found":
                self._error(404, "global/object-does-not-exist", "Specified instance does not exist.")
            elif mode == "404_other_code":
                self._error(404, "global/not-found", "Not found.")
            elif mode == "404_no_code":
                self._send(404, raw=b"<html>404 Not Found</html>")
            elif mode == "200_html":
                self._send(200, raw=b"<html>Please wait</html>")
            elif mode == "200_no_status":
                entry = {"id": "fake-instance-1", "file_system_names": list(fake.filesystems)}
                self._send(200, {"data": [entry] if self.path.endswith("/instances") else entry})
            elif mode == "200_unlisted_status":
                entry = {"id": "fake-instance-1", "status": "hibernating", "file_system_names": list(fake.filesystems)}
                self._send(200, {"data": [entry] if self.path.endswith("/instances") else entry})
            elif mode == "200_not_a_list":
                self._send(200, {"data": {}})  # empty, so iterating it finds nothing rather than failing
            elif mode == "drop":
                self.close_connection = True
                self.connection.shutdown(socket.SHUT_RDWR)

        def _stall(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self.wfile.write(b'{"data"')
            self.wfile.flush()
            fake._stop.wait(30)
            self.close_connection = True

        def do_GET(self) -> None:
            if fake.on_get is not None:
                fake.on_get(self.path)
            with fake._lock:
                fake.requests.append(("GET", self.path))
                mode = fake._read_mode(self.path)
                if mode not in (None, "ok", "stall"):
                    self._answer_read_mode(mode)
                    return
            if mode == "stall":  # outside the lock, so other requests still get answers
                self._stall()
                return
            with fake._lock:
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
            stall = False
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
                    elif mode == "start_then_200_html":
                        self._send(200, raw=b"<html>Please wait</html>")
                    elif mode == "start_then_stall":
                        stall = True  # answered below, outside the lock
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
                elif self.path == "/api/v1/instance-operations/terminate":
                    for inst in fake.instances:
                        if inst["id"] in body.get("instance_ids", []):
                            inst["_status"] = "terminated"
                    self._send(200, {"data": {"terminated_instances": []}})
                else:
                    self._error(404, "global/object-does-not-exist", f"Fake has no POST {self.path}")
            if stall:
                self._stall()

    return Handler


@dataclass
class UpResult:
    returncode: int
    output: str
    state: dict  # the config's local state after the run; {} if none
    shim_calls: list[dict]  # every ssh and rsync call, in order
    state_bytes: bytes | None = None  # the state file exactly as it was left; None if none


def run_up(fake: FakeLambda, workdir: Path, **kwargs) -> UpResult:
    """Run the real `cloudgpu up` in a subprocess against the fake. See run_cloudgpu."""
    return run_cloudgpu(fake, workdir, ["up", "--config", "box"], **kwargs)


def run_cloudgpu(
    fake: FakeLambda, workdir: Path, args: list[str], *,
    poll_seconds=0, max_hours=1.0, timeout=60, base_url=None, state=None, api_key="dummy-key-not-real", pace=None,
    config_extra="", shim_env=None, interrupt_when=None,
) -> UpResult:
    """Run a real `cloudgpu` command in a subprocess against the fake, with ssh and rsync shimmed.

    `api_key=None` leaves the key out. `pace` divides its timeouts and retry delays.
    `config_extra` is appended to the config's TOML; `shim_env` adds FAKE_SHIM_* settings
    for the ssh shim (FAKE_SHIM_APPS, FAKE_SHIM_RECOVER_EXIT). `interrupt_when`, if given, is
    checked every few milliseconds; once it's true the command gets a SIGINT, as from a Ctrl-C.

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
        f"poll_seconds = {poll_seconds}\nmax_hours = {max_hours}\n{config_extra}"
    )
    state_file = configs / "box.state.json"
    if state is not None:
        state_file.write_text(json.dumps(state))
    shim_log = workdir / "shim-calls.jsonl"
    env = {
        "PATH": f"{SHIMS}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(workdir / "home"),
        "COLUMNS": "200",
        API_BASE_ENV: base_url or fake.base_url,
        "http_proxy": "http://127.0.0.1:9",
        "https_proxy": "http://127.0.0.1:9",
        "FAKE_SHIM_LOG": str(shim_log),
        "FAKE_SHIM_FILESYSTEM": fake.filesystems[0] if fake.filesystems else "box-fs",
    }
    env.update(shim_env or {})
    if api_key is not None:
        env["LAMBDA_API_KEY"] = api_key
    if pace is not None:
        env["CLOUDGPU_TEST_LAMBDA_PACE"] = str(pace)
    proc = subprocess.Popen(
        [sys.executable, "-c", "from cloudgpu.cli import cli; cli()", *args],
        cwd=workdir, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        if interrupt_when is not None:
            deadline = time.monotonic() + timeout
            while proc.poll() is None and not interrupt_when():
                if time.monotonic() > deadline:
                    raise subprocess.TimeoutExpired(args, timeout)
                time.sleep(0.01)
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise
    return UpResult(
        returncode=proc.returncode,
        output=stdout + stderr,
        state=json.loads(state_file.read_text()) if state_file.exists() else {},
        shim_calls=[json.loads(line) for line in shim_log.read_text().splitlines()] if shim_log.exists() else [],
        state_bytes=state_file.read_bytes() if state_file.exists() else None,
    )
