"""Live HTTP path test.

The sandbox network policy blocks real LLM endpoints, so this test runs a
local OpenAI-compatible mock server and drives the FULL non-scripted code
path through it: provider detection -> catalog -> routing -> real HTTP call
-> response parsing -> usage/cost accounting -> end-to-end task.
With a reachable endpoint (e.g. OpenRouter) the identical code path is used.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

TASK_TEXT = ("Napisz prostą funkcję Python moving_average(data, window) w pliku "
             "moving_average.py, liczącą średnią kroczącą. Bez zewnętrznych bibliotek.")

UNDERSTANDING = {
    "goal": "Implement moving_average(data, window) computing the rolling average",
    "deliverable": "moving_average.py with the function plus passing tests",
    "constraints": ["pure Python standard library only"],
    "risks": ["wrong edge-case behaviour"],
    "risk_level": "low",
    "complexity": "simple",
    "uncertainties": [],
    "ambiguities": [],
    "success_criteria": ["function returns correct rolling averages", "tests pass"],
    "domain": "software",
    "task_type": "code_generation",
    "required_capabilities": [
        {"capability": "backend", "min_quality": 0.6, "importance": 0.9, "why": "impl"},
        {"capability": "unit_testing", "min_quality": 0.5, "importance": 0.7, "why": "proof"}],
    "autonomy": "minimal",
    "verification_requirements": ["deterministic_test"],
    "interpretation": "rolling mean; window > len(data) returns []",
    "clarifying_questions": [],
    "confidence": 0.9,
}
IMPLEMENT = {
    "files": {"moving_average.py":
              'def moving_average(data, window):\n'
              '    if window <= 0:\n'
              '        raise ValueError("window must be positive")\n'
              '    if window > len(data):\n'
              '        return []\n'
              '    return [sum(data[i:i + window]) / window\n'
              '            for i in range(len(data) - window + 1)]\n'},
    "actions": [],
    "output": "Implemented moving_average with edge cases.",
    "artifact": "moving_average.py",
    "confidence": 0.9,
    "assumptions_used": [],
}
TEST_STEP = {
    "files": {"test_moving_average.py":
              'from moving_average import moving_average\n\n'
              'def test_basic():\n'
              '    assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]\n\n'
              'def test_window_too_large():\n'
              '    assert moving_average([1], 5) == []\n'},
    "actions": [{"tool": "test_run", "kwargs": {"target": "."}}],
    "output": "2 deterministic tests written and passing.",
    "artifact": "test_moving_average.py",
    "confidence": 0.95,
    "assumptions_used": [],
}


class MockHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        prompt = "\n".join(m.get("content", "") for m in body.get("messages", []))
        if "FAMA:PHASE:UNDERSTANDING" in prompt:
            payload = UNDERSTANDING
        elif "FAMA:STEP:implement" in prompt:
            payload = IMPLEMENT
        elif "FAMA:STEP:test" in prompt:
            payload = TEST_STEP
        else:
            payload = {"note": "mock-model-generic"}
        content = json.dumps(payload)
        resp = {
            "id": "chatcmpl-mock", "object": "chat.completion",
            "model": body.get("model", "mock"),
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 340, "total_tokens": 460},
        }
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture()
def mock_provider(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), MockHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    port = server.server_address[1]
    monkeypatch.setenv("OPENAI_API_KEY", "mock-key")
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{port}/v1")
    for k in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    yield port
    server.shutdown()


@pytest.mark.asyncio
async def test_live_http_path_end_to_end(mock_provider, tmp_path):
    from fama.llm import LLMGateway
    from fama.orchestrator import FAMA
    from fama.store import Store

    gw = LLMGateway()
    status = gw.provider_status()
    assert status["any_real"] is True
    assert status["openai"] is True and not status["scripted"]

    f = FAMA(gw, Store(":memory:"), base_dir=str(tmp_path))
    task, st = f.create_task(TASK_TEXT)
    await f.run(st, allow_assumptions=True)

    assert st.task.result_status.value == "verified", st.task.result_summary
    # usage accounting must come from real HTTP responses (non-scripted path)
    assert gw.call_log and all(not c["scripted"] for c in gw.call_log)
    assert gw.total_tokens["input"] > 0 and gw.total_tokens["output"] > 0
    assert gw.total_cost > 0
    # verification oracle actually ran
    assert any(r.kind.value == "deterministic_test" and r.verdict == "pass"
               for r in st.oracle_runs)
    await gw.close()
