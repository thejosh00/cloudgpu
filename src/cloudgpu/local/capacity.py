"""Capacity samples: the data behind the go/no-go on region failover (plans/v2.md, phase 1).

Each sample is one JSON line: ``{"timestamp": "<UTC>", "types": {"<type>": ["<region>", ...]}}``.
An empty region list means the type was listed with no capacity anywhere, so anything else
that goes wrong has to stop before a line is written, or it would read as "no capacity".
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

HOME = "home"
ELSEWHERE = "elsewhere"
NONE = "none"
NOT_LISTED = "not_listed"


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
        "timestamp": datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT),
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


def _is_sample(value: object) -> bool:
    """Whether a parsed line has the shape ``append_sample`` writes."""
    if not isinstance(value, dict):
        return False
    timestamp, types = value.get("timestamp"), value.get("types")
    if not isinstance(timestamp, str) or not isinstance(types, dict) or not types:
        return False
    try:
        if datetime.strptime(timestamp, TIMESTAMP_FORMAT).strftime(TIMESTAMP_FORMAT) != timestamp:
            return False
    except ValueError:
        return False
    return all(
        isinstance(regions, list) and all(isinstance(r, str) and r for r in regions)
        for regions in types.values()
    )


def read_samples(path: Path) -> tuple[list[dict], int]:
    """Return the log's good samples and how many lines were skipped. A missing log has neither.

    A write cut short can leave a bad line and the next sample can join onto it, so a bad line
    is counted rather than fatal.

    Raises:
        OSError: If the log exists but can't be read.
    """
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return [], 0
    lines = data.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    samples, skipped = [], 0
    for line in lines:
        try:
            value = json.loads(line)
        except ValueError:
            skipped += 1
            continue
        if _is_sample(value):
            samples.append(value)
        else:
            skipped += 1
    return samples, skipped


def bucket(types: dict[str, list[str]], targets: list[str], home: str) -> str:
    """Put one sample in exactly one bucket for a config's instance types and home region."""
    listed = [t for t in targets if t in types]
    if not listed:
        return NOT_LISTED
    regions = {r for t in listed for r in types[t]}
    if home in regions:
        return HOME
    return ELSEWHERE if regions else NONE


def summarize(samples: list[dict], targets: list[str], home: str) -> dict:
    """Count samples per bucket, and rank regions by how many samples had capacity there."""
    counts = Counter({HOME: 0, ELSEWHERE: 0, NONE: 0, NOT_LISTED: 0})
    regions: Counter[str] = Counter()
    for sample in samples:
        types = sample["types"]
        counts[bucket(types, targets, home)] += 1
        regions.update({r for t in targets for r in types.get(t, [])})
    ranked = sorted(regions.items(), key=lambda item: (-item[1], item[0]))
    return {"total": len(samples), "counts": dict(counts), "regions": ranked}


def percent(count: int, total: int) -> str:
    """A percentage to one decimal, rounded down: 24.96% must not read as reaching 25%."""
    tenths = count * 1000 // total
    return f"{tenths // 10}.{tenths % 10}%"
