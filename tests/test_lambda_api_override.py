"""CLOUDGPU_TEST_LAMBDA_API_BASE decides where the Lambda API key is sent."""

from __future__ import annotations

import io
import json

import pytest

from cloudgpu.local import lambda_api
from tests.fake_lambda import API_BASE_ENV, FakeLambda, run_up

NOT_LOOPBACK = [
    "https://cloud.lambda.ai/api/v1",
    "http://192.0.2.1:8000",
    "http://0.0.0.0:8000",
    "http://localhost:8000",  # a hostname can resolve anywhere
    "http://127.0.0.1.example.com",
    "http://2130706433",
    "http://127.0.0.1@example.com",
    "http://example.com#@127.0.0.1",
    "http://example.com\\@127.0.0.1",
    "http://127.0.0.1?@example.com",
    "http://127.0.0.1:notaport",
    "ftp://127.0.0.1",
    "127.0.0.1:8000",
]


@pytest.fixture
def sent(monkeypatch):
    """Requests that went out, with both ways out replaced so that nothing can really leave."""
    requests = []

    def record(req, timeout=None):
        requests.append(req)
        return io.BytesIO(json.dumps({"data": []}).encode())

    monkeypatch.setenv(lambda_api.API_KEY_ENV, "dummy-key-not-real")
    monkeypatch.setattr("urllib.request.urlopen", record)
    if hasattr(lambda_api, "_DIRECT_OPENER"):
        monkeypatch.setattr(lambda_api._DIRECT_OPENER, "open", record)
    return requests


@pytest.mark.parametrize("url", NOT_LOOPBACK)
def test_non_loopback_override_is_refused_before_any_request(monkeypatch, sent, url):
    monkeypatch.setenv(API_BASE_ENV, url)

    with pytest.raises(lambda_api.LambdaAPIError, match="Refusing"):
        lambda_api.list_instances()

    assert sent == []


@pytest.mark.parametrize("url", ["http://127.0.0.1:8000/api/v1", "http://127.0.0.2:1/api/v1/", "http://[::1]:8000/api/v1"])
def test_loopback_override_is_used(monkeypatch, sent, url):
    monkeypatch.setenv(API_BASE_ENV, url)

    lambda_api.list_instances()

    assert [r.full_url for r in sent] == [url.rstrip("/") + "/instances"]


@pytest.mark.parametrize("value", [None, ""])
def test_api_base_is_the_real_one_when_unset(monkeypatch, sent, value):
    if value is None:
        monkeypatch.delenv(API_BASE_ENV, raising=False)
    else:
        monkeypatch.setenv(API_BASE_ENV, value)

    lambda_api.list_instances()

    assert lambda_api.API_BASE == "https://cloud.lambda.ai/api/v1"
    assert [r.full_url for r in sent] == ["https://cloud.lambda.ai/api/v1/instances"]


def test_cli_refuses_before_any_request_and_keeps_local_state(tmp_path):
    tracked = {"instance_id": "real-instance", "host": "ubuntu@203.0.113.1"}
    with FakeLambda() as fake:
        result = run_up(fake, tmp_path, base_url="http://192.0.2.1:8000", state=tracked)

    assert result.returncode == 2, result.output
    assert "Refusing" in result.output and API_BASE_ENV in result.output
    assert result.state == tracked  # a refusal isn't "the tracked instance is gone"
    assert fake.requests == [] and result.shim_calls == []


def test_override_goes_direct_even_with_a_proxy_set(tmp_path):
    """run_up sets a closed proxy port, so this reaches the fake only if the proxy is skipped."""
    with FakeLambda() as fake:
        result = run_up(fake, tmp_path)

    assert result.returncode == 0, result.output
    assert fake.requests
