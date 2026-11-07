"""Capacity samples: the data behind the go/no-go on region failover (plans/v2.md, phase 1).

Each sample is one JSON line: ``{"timestamp": "<UTC>", "types": {"<type>": ["<region>", ...]}}``.
An empty region list means the type was listed with no capacity anywhere, so anything else
that goes wrong has to stop before a line is written, or it would read as "no capacity".
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def log_path() -> Path:
    return Path.home() / ".local" / "state" / "cloudgpu" / "capacity.jsonl"


def regions_by_type(types: object) -> dict[str, list[str]]:
    """Reduce a ``list_instance_types`` response to type -> available region names.

    Raises:
        ValueError: If the response isn't one, rather than guessing there's no capacity.
    """
    if not isinstance(types, dict) or not types:
        raise ValueError("the response lists no instance types")
    result = {}
    for name, entry in types.items():
        regions = entry.get("regions_with_capacity_available") if isinstance(entry, dict) else None
        if not isinstance(regions, list):
            raise ValueError(f"no region list for {name!r}")
        if not all(isinstance(r, dict) and r.get("name") for r in regions):
            raise ValueError(f"a region without a name for {name!r}")
        result[name] = [r["name"] for r in regions]
    return result


def append_sample(types: object) -> Path:
    """Append one sample as a single write, creating the directory if needed. Returns the path."""
    sample = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "types": regions_by_type(types),
    }
    line = (json.dumps(sample) + "\n").encode()
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        written = os.write(fd, line)
    finally:
        os.close(fd)
    if written != len(line):
        raise OSError(f"wrote {written} of {len(line)} bytes to {path}")
    return path
