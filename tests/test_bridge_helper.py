"""Bridge helper tests: CORS + Private-Network-Access headers and proxying.

The helper is what makes the World-UI bridge work in Chrome (PNA) — these
tests verify the exact headers a browser requires before it will let an
HTTPS page call http://localhost.
"""
import json
import threading
import urllib.request
from http.server import HTTPServer

import pytest

from examples.bridge_helper import make_handler
from tests.test_live_path import MockHandler


@pytest.fixture(scope="module")
def stack():
    llm = HTTPServer(("127.0.0.1", 0), MockHandler)
    helper = HTTPServer(("127.0.0.1", 0),
                        make_handler(f"http://127.0.0.1:{llm.server_address[1]}"))
    for srv in (llm, helper):
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{helper.server_address[1]}"
    llm.shutdown()
    helper.shutdown()


def _req(url, method="GET", headers=None, body=None):
    req = urllib.request.Request(url, method=method, headers=headers or {},
                                 data=json.dumps(body).encode() if body else None)
    with urllib.request.urlopen(req, timeout=15) as r:
        return r, r.read().decode()


def test_preflight_allows_private_network(stack):
    """Chrome sends exactly this preflight before HTTPS->localhost fetches."""
    req = urllib.request.Request(stack + "/v1/models", method="OPTIONS", headers={
        "Origin": "https://8000-example.e2b.app",
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Private-Network": "true"})
    with urllib.request.urlopen(req, timeout=10) as r:
        assert r.status == 204
        assert r.headers.get("Access-Control-Allow-Private-Network") == "true"
        assert r.headers.get("Access-Control-Allow-Origin") == "*"
        assert r.headers.get("Access-Control-Allow-Headers") == "*"


def test_models_proxied_with_cors(stack):
    r, body = _req(stack + "/v1/models", headers={"Origin": "https://x.example"})
    assert r.headers.get("Access-Control-Allow-Origin") == "*"
    ids = [m["id"] for m in json.loads(body)["data"]]
    assert "qwen2.5-coder:7b" in ids


def test_chat_proxied(stack):
    _, body = _req(stack + "/v1/chat/completions", method="POST", body={
        "model": "demo-mini-3b",
        "messages": [{"role": "user", "content": "FAMA:STEP:implement\nping moving_average"}]})
    data = json.loads(body)
    assert data["choices"][0]["message"]["content"]
    assert data["usage"]["total_tokens"] > 0


def test_unreachable_target_reports_502():
    dead = HTTPServer(("127.0.0.1", 0), make_handler("http://127.0.0.1:9"))
    threading.Thread(target=dead.serve_forever, daemon=True).start()
    try:
        with pytest.raises(Exception) as ei:
            _req(f"http://127.0.0.1:{dead.server_address[1]}/v1/models")
        assert "502" in str(ei.value) or "HTTPError" in type(ei.value).__name__
    finally:
        dead.shutdown()
