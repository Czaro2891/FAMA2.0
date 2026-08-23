"""API server tests (TestClient, scripted scenario runs only)."""
import asyncio

import pytest
from fastapi.testclient import TestClient

from fama.server import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "providers" in body


def test_scenarios_list(client):
    r = client.get("/api/scenarios")
    names = [s["name"] for s in r.json()["scenarios"]]
    assert "payments-bug" in names and "weak-tests" in names


def test_agents_and_models(client):
    agents = client.get("/api/agents").json()["agents"]
    assert any(a["name"] == "coder" for a in agents)
    models = client.get("/api/models").json()["models"]
    assert len(models) >= 5


def test_submit_without_input_fails(client):
    r = client.post("/api/tasks", json={"input": ""})
    assert r.status_code == 400


def test_run_scenario_and_fetch_state(client):
    r = client.post("/api/scenarios/simple-function/run")
    assert r.status_code == 200
    task_id = r.json()["task_id"]
    assert r.json()["scripted"] is True
    # wait for completion
    for _ in range(120):
        state = client.get(f"/api/tasks/{task_id}").json()
        if state and state["task"]["status"] in ("completed", "failed", "blocked", "uncertain"):
            break
        import time
        time.sleep(0.25)
    assert state["task"]["result_status"] == "verified"
    assert state["evidence"]["nodes"], "evidence graph must be populated"
    assert state["decisions"], "decision trace must be recorded"
    events = client.get(f"/api/tasks/{task_id}/events").json()
    types = [e["type"] for e in events]
    assert "strategy_selected" in types and "oracle_run" in types


def test_task_list_contains_scenario(client):
    r = client.post("/api/scenarios/simple-function/run")
    tid = r.json()["task_id"]
    lst = client.get("/api/tasks").json()
    assert any(t["id"] == tid and t["scripted"] for t in lst)


def test_unknown_task_404(client):
    assert client.get("/api/tasks/nope").status_code == 404


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "FAMA 2.0" in r.text
