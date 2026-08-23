"""Browser-bridge integration test.

Simulates what the World UI does: registers local models on the bridge,
then polls pending requests and answers them with fixture responses —
exactly the loop the browser runs against the user's local Ollama/LM Studio.
"""
import time

import pytest
from fastapi.testclient import TestClient

from fama.server import app
from tests.test_live_path import IMPLEMENT, TASK_TEXT, TEST_STEP, UNDERSTANDING


def _fixture_for(messages):
    prompt = "\n".join(m.get("content", "") for m in messages)
    import json
    if "FAMA:PHASE:UNDERSTANDING" in prompt:
        return json.dumps(UNDERSTANDING)
    if "FAMA:STEP:implement" in prompt:
        return json.dumps(IMPLEMENT)
    if "FAMA:STEP:test" in prompt:
        return json.dumps(TEST_STEP)
    return json.dumps({"note": "bridge-test-generic"})


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_bridge_end_to_end(client):
    # 1) the "browser" registers the user's local models
    r = client.post("/api/bridge/models", json={"models": ["test-coder:7b"],
                                                "base_url": "http://localhost:11434/v1"})
    assert r.json()["ok"]
    assert any(m["model_id"] == "test-coder:7b" for m in r.json()["models"])
    # bridge is now a live provider
    health = client.get("/api/health").json()
    assert health["providers"]["bridge"] is True
    assert health["providers"]["any_real"] is True

    # 2) submit a task — it will wait on the bridge
    tid = client.post("/api/tasks", json={"input": TASK_TEXT,
                                          "allow_assumptions": True}).json()["task_id"]

    # 3) serve the bridge like the browser does
    answered = 0
    deadline = time.time() + 90
    final = None
    while time.time() < deadline:
        pend = client.get("/api/bridge/pending").json()
        for p in pend["pending"]:
            ok = client.post("/api/bridge/complete", json={
                "id": p["id"], "content": _fixture_for(p["messages"]),
                "tokens_in": 120, "tokens_out": 340}).json()["ok"]
            if ok:
                answered += 1
        state = client.get(f"/api/tasks/{tid}").json()
        if state and state["task"]["status"] in ("completed", "failed", "blocked", "uncertain"):
            final = state
            break
        time.sleep(0.2)

    assert final is not None, "task did not finish"
    assert final["task"]["result_status"] == "verified", final["task"]["result_summary"]
    assert answered >= 3, f"bridge should have served >=3 requests, served {answered}"
    # oracle actually ran and the models used are bridge/*
    assert any(r["kind"] == "deterministic_test" and r["verdict"] == "pass"
               for r in final["oracle_runs"])
    status = client.get("/api/bridge/status").json()
    assert status["enabled"] and status["served"] >= 3

    # 4) disconnect — pending requests (if any) must fail fast
    client.post("/api/bridge/disable")
    assert client.get("/api/bridge/status").json()["enabled"] is False


def test_bridge_fail_reports_error(client):
    client.post("/api/bridge/models", json={"models": ["test-mini:3b"]})
    tid = client.post("/api/tasks", json={"input": TASK_TEXT,
                                          "allow_assumptions": True}).json()["task_id"]
    # fail the understanding request like a browser that cannot reach localhost
    deadline = time.time() + 30
    failed = False
    while time.time() < deadline:
        pend = client.get("/api/bridge/pending").json()
        for p in pend["pending"]:
            client.post("/api/bridge/fail", json={"id": p["id"],
                                                  "error": "test: browser cannot reach localhost"})
            failed = True
        state = client.get(f"/api/tasks/{tid}").json()
        if state and state["task"]["status"] in ("completed", "failed", "blocked", "uncertain"):
            break
        time.sleep(0.2)
    assert failed
    assert state["task"]["result_status"] == "blocked"  # honest failure, no fake success
    client.post("/api/bridge/disable")
