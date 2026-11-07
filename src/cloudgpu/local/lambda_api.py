"""Lambda Cloud API client (https://docs-api.lambda.ai/api/cloud).

The API key is read from the LAMBDA_API_KEY env var.
"""

from __future__ import annotations

import ipaddress
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

API_BASE = "https://cloud.lambda.ai/api/v1"

API_KEY_ENV = "LAMBDA_API_KEY"

# Test-only: points the client at tests/fake_lambda. The API key is sent to whatever this
# names, so only a loopback address is accepted, and no proxy is ever used for it.
API_BASE_ENV = "CLOUDGPU_TEST_LAMBDA_API_BASE"
_DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# cloud.lambda.ai is fronted by Cloudflare, which 403s the default
# "Python-urllib/x.y" user agent. Send an explicit UA so requests get through.
USER_AGENT = "cloudgpu/0.1.0"


class LambdaAPIError(Exception):
    """Raised when a Lambda Cloud API request fails."""


def _api_key() -> str:
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise LambdaAPIError(
            f"{API_KEY_ENV} environment variable not set. "
            "Create an API key at https://cloud.lambda.ai/api-keys and export it, e.g.\n"
            f"  export {API_KEY_ENV}=secret_..."
        )
    return key


def _is_loopback_url(url: str) -> bool:
    """True only for http(s) to a literal loopback IP. A hostname could resolve anywhere."""
    try:
        parts = urllib.parse.urlsplit(url)
        address = ipaddress.ip_address(parts.hostname or "")
        parts.port  # raises ValueError on a bad port
    except ValueError:
        return False
    return (
        parts.scheme in ("http", "https")
        and address.is_loopback
        and "@" not in parts.netloc
        and not parts.query
        and not parts.fragment
    )


def api_base_override() -> str | None:
    """The test-only API base if one is set, else None.

    Raises:
        LambdaAPIError: If it is set to anything but a loopback address.
    """
    override = os.environ.get(API_BASE_ENV)
    if not override:
        return None
    if not _is_loopback_url(override):
        raise LambdaAPIError(
            f"Refusing {API_BASE_ENV}={override!r}: the Lambda API key is sent to whatever it "
            "names, so it must be a loopback address such as http://127.0.0.1:8000/api/v1. "
            "Unset it to use the real API."
        )
    return override.rstrip("/")


def _request(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    """Make an authenticated request and return the unwrapped ``data`` payload.

    Args:
        method: HTTP method (GET, POST, DELETE).
        path: Path under the API base, starting with "/".
        body: JSON-serializable request body, or None for no body.

    Raises:
        LambdaAPIError: On missing key, network failure, or non-2xx response.
    """
    override = api_base_override()
    url = f"{override or API_BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }

    if data is not None:
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        open_url = _DIRECT_OPENER.open if override else urllib.request.urlopen
        with open_url(req, timeout=60) as resp:
            payload = resp.read().decode()
    except urllib.error.HTTPError as e:
        raise LambdaAPIError(_format_http_error(e)) from e
    except urllib.error.URLError as e:
        raise LambdaAPIError(f"Could not reach Lambda Cloud API: {e.reason}") from e

    if not payload:
        return None
    parsed = json.loads(payload)
    # Successful responses are wrapped in {"data": ...}; fall back to the raw body.
    return parsed.get("data", parsed) if isinstance(parsed, dict) else parsed


def _format_http_error(e: urllib.error.HTTPError) -> str:
    """Extract Lambda's structured error ({code, message, suggestion}) if present."""
    detail = ""
    try:
        parsed = json.loads(e.read().decode())
        err = parsed.get("error", {}) if isinstance(parsed, dict) else {}
        parts = [err.get("code"), err.get("message"), err.get("suggestion")]
        detail = " - ".join(p for p in parts if p)
    except (ValueError, OSError):
        pass
    base = f"Lambda API request failed (HTTP {e.code})"
    return f"{base}: {detail}" if detail else base


# --- Filesystems ----------------------------------------------------------


def list_filesystems() -> list[dict]:
    """List all filesystems."""
    return _request("GET", "/file-systems")


def create_filesystem(name: str, region: str) -> dict:
    """Create a filesystem in the given region."""
    return _request("POST", "/filesystems", {"name": name, "region": region})


def delete_filesystem(filesystem_id: str) -> Any:
    """Delete a filesystem by id."""
    return _request("DELETE", f"/filesystems/{filesystem_id}")


# --- Instances ------------------------------------------------------------


def list_instances() -> list[dict]:
    """List running instances."""
    return _request("GET", "/instances")


def get_instance(instance_id: str) -> dict:
    """Get a single instance by id (status, ip, instance_type, region, ...)."""
    return _request("GET", f"/instances/{instance_id}")


def list_ssh_keys() -> list[dict]:
    """List the account's SSH keys (name + public_key)."""
    return _request("GET", "/ssh-keys")


def list_instance_types() -> dict:
    """List available instance types, keyed by instance type name."""
    return _request("GET", "/instance-types")


def launch_instance(
    *,
    region_name: str,
    instance_type_name: str,
    ssh_key_names: list[str],
    file_system_names: list[str] | None = None,
    name: str | None = None,
) -> dict:
    """Launch an instance.

    Args:
        region_name: Region to launch in, e.g. "us-tx-1".
        instance_type_name: Instance type, e.g. "gpu_1x_a10".
        ssh_key_names: SSH key names to install (the API expects exactly one).
        file_system_names: Optional filesystems to mount.
        name: Optional instance name.
    """
    body: dict[str, Any] = {
        "region_name": region_name,
        "instance_type_name": instance_type_name,
        "ssh_key_names": ssh_key_names,
    }
    if file_system_names:
        body["file_system_names"] = file_system_names
    if name:
        body["name"] = name
    return _request("POST", "/instance-operations/launch", body)


def restart_instances(instance_ids: list[str]) -> dict:
    """Restart one or more instances by id."""
    return _request("POST", "/instance-operations/restart", {"instance_ids": instance_ids})


def terminate_instances(instance_ids: list[str]) -> dict:
    """Terminate one or more instances by id."""
    return _request("POST", "/instance-operations/terminate", {"instance_ids": instance_ids})
