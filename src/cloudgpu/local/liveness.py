"""Is an instance gone, alive, or can't we tell?

`up` and `down` act on the answer by launching, clearing state or deleting a filesystem, so only
a clear answer counts. Everything else is UNKNOWN, retried here, and left for the caller to stop on.

Statuses and the not-found code are Lambda's published ones (https://docs.lambda.com/api/cloud/spec.json).
"""

from __future__ import annotations

import enum
import time
from collections.abc import Callable
from dataclasses import dataclass

from rich.markup import escape

from cloudgpu.local import display, lambda_api

# How long to wait before the second and third attempts.
RETRY_DELAYS_SECONDS = (2, 5)

NOT_FOUND_CODE = "global/object-does-not-exist"

_ALIVE_STATUSES = {"booting", "active", "unhealthy", "terminating", "preempted"}
_GONE_STATUS = "terminated"


class Kind(enum.Enum):
    GONE = "gone"  # for a listing: nothing there
    ALIVE = "alive"  # exists, and not necessarily usable: see usable()
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Answer:
    kind: Kind
    instance: dict | None = None
    detail: str = ""  # for UNKNOWN: what went wrong and what was tried


class _CantTell(Exception):
    def __init__(self, reason: str, *, retryable: bool = True):
        super().__init__(reason)
        self.retryable = retryable


def usable(inst: dict) -> bool:
    """True if `up` can use it now: active, with an ip."""
    return inst.get("status") == "active" and bool(inst.get("ip"))


def describe(inst: dict) -> str:
    """Its status in words, for a message."""
    status = inst.get("status")
    return "active but has no ip yet" if status == "active" else f"'{status}'"


def _status_of(inst: dict) -> str | None:
    status = inst.get("status")
    if isinstance(status, str) and (status == _GONE_STATUS or status in _ALIVE_STATUSES):
        return status
    return None


def _read_once(instance_id: str) -> Answer:
    try:
        inst = lambda_api.get_instance(instance_id)
    except lambda_api.LambdaAPIError as e:
        if e.status == 404 and e.code == NOT_FOUND_CODE:
            return Answer(Kind.GONE)
        raise _CantTell(str(e), retryable=e.retryable) from e
    except lambda_api.LambdaReplyLost as e:
        raise _CantTell(str(e)) from e
    if not isinstance(inst, dict):
        raise _CantTell("Lambda's reply wasn't an instance")
    status = _status_of(inst)
    if status is None:
        raise _CantTell(f"Lambda's reply had no status it publishes (got {inst.get('status')!r})")
    return Answer(Kind.GONE) if status == _GONE_STATUS else Answer(Kind.ALIVE, instance=inst)


def _mounting_once(filesystem: str) -> Answer:
    try:
        instances = lambda_api.list_instances()
    except lambda_api.LambdaAPIError as e:
        raise _CantTell(str(e), retryable=e.retryable) from e
    except lambda_api.LambdaReplyLost as e:
        raise _CantTell(str(e)) from e
    if not isinstance(instances, list):
        raise _CantTell("Lambda's instance list wasn't a list")

    found = []
    for inst in instances:
        if not isinstance(inst, dict):
            raise _CantTell("Lambda's instance list had an entry that wasn't an instance")
        status = _status_of(inst)
        if status is None:
            raise _CantTell(f"Lambda listed an instance with no status it publishes (got {inst.get('status')!r})")
        names = inst.get("file_system_names") or []
        if not isinstance(names, list):
            raise _CantTell("Lambda listed an instance whose filesystems weren't a list")
        if filesystem in names and status != _GONE_STATUS:
            found.append(inst)
    if not found:
        return Answer(Kind.GONE)
    return Answer(Kind.ALIVE, instance=next((i for i in found if usable(i)), found[0]))


def _with_retries(what: str, once: Callable[[], Answer]) -> Answer:
    """Ask, with a few retries on an unclear answer. A retry that can't help isn't made."""
    started = time.monotonic()
    attempts = 0
    for delay in (*RETRY_DELAYS_SECONDS, None):
        attempts += 1
        try:
            return once()
        except _CantTell as e:
            failure = e
        if delay is None or not failure.retryable:
            break
        display.warn(escape(f"Couldn't tell {what} ({failure}); retrying in {delay}s..."))
        time.sleep(delay / lambda_api.pace())
    tries = "Tried once" if attempts == 1 else f"Tried {attempts} times over {time.monotonic() - started:.0f}s"
    return Answer(Kind.UNKNOWN, detail=f"Couldn't tell {what}: {failure}\n{tries}.")


def read_instance(instance_id: str) -> Answer:
    """GONE only on the published not-found or status terminated. ALIVE carries the instance."""
    return _with_retries(f"whether instance {instance_id} is still there", lambda: _read_once(instance_id))


def find_mounting(filesystem: str) -> Answer:
    """GONE if nothing not terminated mounts it. Else ALIVE, preferring a usable instance."""
    return _with_retries(f"what is mounting filesystem '{filesystem}'", lambda: _mounting_once(filesystem))
