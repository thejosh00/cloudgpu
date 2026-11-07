"""Tests for `cloudgpu capacity log` (no network, no key; HOME in a tmp dir)."""

from __future__ import annotations

import io
import json
import os
from datetime import datetime, timedelta, timezone

import pytest
from click.testing import CliRunner

from cloudgpu import cli
from cloudgpu.local import lambda_api

TYPES = {
    "gpu_1x_gh200": {
        "instance_type": {"name": "gpu_1x_gh200", "price_cents_per_hour": 149},
        "regions_with_capacity_available": [
            {"name": "us-east-3", "description": "Washington DC"},
            {"name": "us-west-2", "description": "California"},
        ],
    },
    "gpu_8x_h100": {
        "instance_type": {"name": "gpu_8x_h100", "price_cents_per_hour": 2392},
        "regions_with_capacity_available": [],
    },
}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(lambda_api.API_KEY_ENV, raising=False)
    return tmp_path


def _log_file(home):
    return home / ".local" / "state" / "cloudgpu" / "capacity.jsonl"


def _sample(monkeypatch, types=TYPES):
    monkeypatch.setattr(lambda_api, "list_instance_types", lambda: types)
    return CliRunner().invoke(cli.cli, ["capacity", "log"])


def _lines(home):
    return _log_file(home).read_text().splitlines()


def test_sample_appends_one_parseable_line(home, monkeypatch):
    result = _sample(monkeypatch)
    assert result.exit_code == 0, result.output
    lines = _lines(home)
    assert len(lines) == 1
    assert json.loads(lines[0])["types"]["gpu_1x_gh200"] == ["us-east-3", "us-west-2"]


def test_second_sample_adds_a_line_and_leaves_the_first_alone(home, monkeypatch):
    assert _sample(monkeypatch).exit_code == 0
    first = _log_file(home).read_bytes()
    assert _sample(monkeypatch).exit_code == 0
    after = _log_file(home).read_bytes()
    assert after.startswith(first)
    assert len(after.splitlines()) == 2


def test_creates_the_state_directory(home, monkeypatch):
    assert not (home / ".local").exists()
    assert _sample(monkeypatch).exit_code == 0
    assert _log_file(home).is_file()


def test_type_with_no_capacity_stays_with_empty_regions(home, monkeypatch):
    assert _sample(monkeypatch).exit_code == 0
    assert json.loads(_lines(home)[0])["types"]["gpu_8x_h100"] == []


def test_timestamp_is_current_utc(home, monkeypatch):
    assert _sample(monkeypatch).exit_code == 0
    stamp = json.loads(_lines(home)[0])["timestamp"]
    assert stamp.endswith("Z")
    parsed = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    assert abs(datetime.now(timezone.utc) - parsed) < timedelta(minutes=1)


def test_records_only_timestamp_types_and_region_names(home, monkeypatch):
    monkeypatch.setenv(lambda_api.API_KEY_ENV, "secret_do_not_record")
    assert _sample(monkeypatch).exit_code == 0
    text = _log_file(home).read_text()
    assert json.loads(text).keys() == {"timestamp", "types"}
    for leaked in ("secret_do_not_record", "price_cents_per_hour", "Washington", "description"):
        assert leaked not in text


def test_the_whole_line_goes_in_one_write(home, monkeypatch):
    writes = []
    real_write = os.write

    def counting_write(fd, data):
        writes.append(bytes(data))
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", counting_write)
    assert _sample(monkeypatch).exit_code == 0
    assert len(writes) == 1
    assert writes[0] == _log_file(home).read_bytes()
    assert writes[0].endswith(b"\n")
    assert writes[0].count(b"\n") == 1


def test_a_write_cut_short_is_a_failure(home, monkeypatch):
    real_write = os.write
    monkeypatch.setattr(os, "write", lambda fd, data: real_write(fd, data[:5]))
    result = _sample(monkeypatch)
    assert result.exit_code == 1
    assert "capacity" in result.output


def test_api_error_exits_nonzero_and_leaves_the_file_unchanged(home, monkeypatch):
    assert _sample(monkeypatch).exit_code == 0
    before = _log_file(home).read_bytes()

    def boom():
        raise lambda_api.LambdaAPIError("Lambda API request failed (HTTP 503)")

    monkeypatch.setattr(lambda_api, "list_instance_types", boom)
    result = CliRunner().invoke(cli.cli, ["capacity", "log"])
    assert result.exit_code == 1
    assert "HTTP 503" in result.output
    assert _log_file(home).read_bytes() == before


def test_missing_key_exits_nonzero_and_appends_nothing(home, monkeypatch):
    def no_network(*a, **k):
        pytest.fail("must not reach the network without a key")

    monkeypatch.setattr(lambda_api.urllib.request, "urlopen", no_network)
    result = CliRunner().invoke(cli.cli, ["capacity", "log"])
    assert result.exit_code == 1
    assert lambda_api.API_KEY_ENV in result.output
    assert not _log_file(home).exists()


@pytest.mark.parametrize(
    "types",
    [
        pytest.param(None, id="empty-response"),
        pytest.param({}, id="no-types-listed"),
        pytest.param([], id="not-a-mapping"),
        pytest.param({"error": {"code": "x", "message": "y"}}, id="error-body-on-a-200"),
        pytest.param({"gpu_a": {"instance_type": {}}}, id="region-list-missing"),
        pytest.param({"gpu_a": {"regions_with_capacity_available": None}}, id="region-list-null"),
        pytest.param({"gpu_a": {"regions_with_capacity_available": [{"id": 1}]}}, id="region-without-name"),
    ],
)
def test_a_response_that_cant_be_read_is_never_recorded_as_no_capacity(home, monkeypatch, types):
    result = _sample(monkeypatch, types)
    assert result.exit_code == 1
    assert "capacity" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert not _log_file(home).exists()


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b"<html>Just a moment...</html>", id="html-on-a-200"),
        pytest.param(b"", id="empty-body"),
        pytest.param(b'{"error": {"code": "x"}}', id="error-body-on-a-200"),
    ],
)
def test_a_bad_200_through_the_real_client_appends_nothing(home, monkeypatch, body):
    monkeypatch.setenv(lambda_api.API_KEY_ENV, "test_key")

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(lambda_api.urllib.request, "urlopen", lambda *a, **k: Response(body))
    result = CliRunner().invoke(cli.cli, ["capacity", "log"])
    assert result.exit_code == 1
    assert not _log_file(home).exists()
    assert isinstance(result.exception, SystemExit)  # a message, not a traceback


def test_unwritable_state_directory_is_a_clean_failure(home, monkeypatch):
    (home / ".local").write_text("a file where the directory should be")
    result = _sample(monkeypatch)
    assert result.exit_code == 1
    assert "capacity" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
