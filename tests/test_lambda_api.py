"""Tests for the Lambda Cloud API client."""

from __future__ import annotations

import io
import json
import urllib.error
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from cloudgpu.local import lambda_api
from cloudgpu.local.lambda_api import LambdaAPIError


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("LAMBDA_API_KEY", "secret_test")


@contextmanager
def _fake_response(payload: str):
    """Mimic the context manager returned by urllib.request.urlopen."""
    resp = io.BytesIO(payload.encode())
    yield resp


def _patch_urlopen(payload: str):
    """Patch urlopen to return ``payload`` and capture the Request passed in."""
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["req"] = req
        captured["timeout"] = timeout
        return _fake_response(payload)

    return patch.object(lambda_api.urllib.request, "urlopen", side_effect=fake_urlopen), captured


class TestApiKey:
    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("LAMBDA_API_KEY", raising=False)
        with pytest.raises(LambdaAPIError, match="LAMBDA_API_KEY"):
            lambda_api.list_instances()


class TestRequest:
    def test_get_builds_authenticated_request(self, api_key):
        patcher, captured = _patch_urlopen(json.dumps({"data": [{"id": "i-1"}]}))
        with patcher:
            result = lambda_api.list_instances()

        req = captured["req"]
        assert result == [{"id": "i-1"}]
        assert req.full_url == "https://cloud.lambda.ai/api/v1/instances"
        assert req.get_method() == "GET"
        assert req.headers["Authorization"] == "Bearer secret_test"
        # Cloudflare 403s the default urllib UA, so a custom one must be sent.
        assert req.headers["User-agent"] == lambda_api.USER_AGENT
        assert req.data is None

    def test_post_sends_json_body(self, api_key):
        patcher, captured = _patch_urlopen(json.dumps({"data": {"id": "fs-1"}}))
        with patcher:
            result = lambda_api.create_filesystem("myfs", "us-tx-1")

        req = captured["req"]
        assert result == {"id": "fs-1"}
        assert req.get_method() == "POST"
        assert req.full_url == "https://cloud.lambda.ai/api/v1/filesystems"
        assert json.loads(req.data.decode()) == {"name": "myfs", "region": "us-tx-1"}
        assert req.headers["Content-type"] == "application/json"

    def test_unwraps_data_key(self, api_key):
        patcher, _ = _patch_urlopen(json.dumps({"data": {"hello": "world"}}))
        with patcher:
            assert lambda_api.list_instance_types() == {"hello": "world"}

    def test_empty_body_returns_none(self, api_key):
        patcher, _ = _patch_urlopen("")
        with patcher:
            assert lambda_api.delete_filesystem("fs-1") is None

    def test_http_error_is_parsed(self, api_key):
        body = json.dumps(
            {"error": {"code": "global/invalid-api-key", "message": "Invalid key"}}
        )
        err = urllib.error.HTTPError(
            url="https://cloud.lambda.ai/api/v1/instances",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(body.encode()),
        )
        with patch.object(lambda_api.urllib.request, "urlopen", side_effect=err):
            with pytest.raises(LambdaAPIError, match="invalid-api-key"):
                lambda_api.list_instances()

    def test_network_error_is_wrapped(self, api_key):
        err = urllib.error.URLError("connection refused")
        with patch.object(lambda_api.urllib.request, "urlopen", side_effect=err):
            with pytest.raises(LambdaAPIError, match="Could not reach"):
                lambda_api.list_instances()


class TestOperations:
    def _run(self, api_key, fn, *args, payload="{}", **kwargs):
        patcher, captured = _patch_urlopen(payload)
        with patcher:
            fn(*args, **kwargs)
        return captured["req"]

    def test_list_filesystems_path(self, api_key):
        req = self._run(api_key, lambda_api.list_filesystems, payload='{"data": []}')
        assert req.full_url.endswith("/file-systems")
        assert req.get_method() == "GET"

    def test_delete_filesystem_path(self, api_key):
        req = self._run(api_key, lambda_api.delete_filesystem, "fs-9")
        assert req.full_url.endswith("/filesystems/fs-9")
        assert req.get_method() == "DELETE"

    def test_launch_includes_optional_fields(self, api_key):
        req = self._run(
            api_key,
            lambda_api.launch_instance,
            payload='{"data": {"instance_ids": ["i-1"]}}',
            region_name="us-tx-1",
            instance_type_name="gpu_1x_a10",
            ssh_key_names=["my-key"],
            file_system_names=["myfs"],
            name="box",
        )
        assert req.full_url.endswith("/instance-operations/launch")
        body = json.loads(req.data.decode())
        assert body == {
            "region_name": "us-tx-1",
            "instance_type_name": "gpu_1x_a10",
            "ssh_key_names": ["my-key"],
            "file_system_names": ["myfs"],
            "name": "box",
        }

    def test_launch_omits_empty_optionals(self, api_key):
        req = self._run(
            api_key,
            lambda_api.launch_instance,
            payload='{"data": {}}',
            region_name="us-tx-1",
            instance_type_name="gpu_1x_a10",
            ssh_key_names=["my-key"],
            file_system_names=None,
            name=None,
        )
        body = json.loads(req.data.decode())
        assert "file_system_names" not in body
        assert "name" not in body

    def test_restart_body(self, api_key):
        req = self._run(api_key, lambda_api.restart_instances, ["i-1", "i-2"])
        assert req.full_url.endswith("/instance-operations/restart")
        assert json.loads(req.data.decode()) == {"instance_ids": ["i-1", "i-2"]}

    def test_terminate_body(self, api_key):
        req = self._run(api_key, lambda_api.terminate_instances, ["i-3"])
        assert req.full_url.endswith("/instance-operations/terminate")
        assert json.loads(req.data.decode()) == {"instance_ids": ["i-3"]}
