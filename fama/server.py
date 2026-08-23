"""FAMA API server + World interface (sec. 38).

Everything the World UI shows comes from real system state — never from
simulation.  Includes SSE live stream, task submission, clarification and
approval endpoints, offline scenario demos (clearly labelled SCRIPTED) and
recorded replays.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .core import AutonomyLevel, Event, TaskStatus
from .llm import LLMGateway, ScriptedProvider
from .orchestrator import FAMA
from .scenarios import SCENARIOS, record_replay, scenario_fixtures, scripted_gateway
from .store import Store

WORLD_DIR = Path(__file__).parent / "world"
REPLAY_DIR = Path(os.environ.get("FAMA_REPLAYS", ".fama/replays"))


class FAMAApp:
    def __init__(self, base_dir: str = ".fama"):
        self.base_dir = base_dir
        allow_scripted = os.environ.get("FAMA_ALLOW_SCRIPTED") == "1"
        self.live = FAMA(LLMGateway(allow_scripted=False), Store(), base_dir=base_dir)
        self.scripted_mode = allow_scripted
        self.demo_famas: dict[str, FAMA] = {}      # task_id -> scripted FAMA
        self.background: dict[str, asyncio.Task] = {}
        self.clients: list[asyncio.Queue] = []
        for bus in (self.live.bus,):
            bus.subscribe(self._fan_in)

    # forward events from any bus into the SSE stream
    def _fan_in(self, ev: Event):
        for q in list(self.clients):
            try:
                q.put_nowait(ev.to_dict())
            except Exception:
                pass

    def fama_for(self, task_id: str) -> Optional[FAMA]:
        if task_id in self.live.states:
            return self.live
        return self.demo_famas.get(task_id)

    def all_states(self):
        out = list(self.live.states.items())
        for f in self.demo_famas.values():
            out.extend(f.states.items())
        return out


app = FastAPI(title="FAMA 2.0 — Adaptive Agent Operating System", version=__version__)
fama_app = FAMAApp()


@app.get("/api/health")
async def health():
    prov = fama_app.live.gateway.provider_status()
    return {"status": "ok", "version": __version__,
            "providers": prov,
            "scripted_mode": fama_app.scripted_mode,
            "warning": None if prov["any_real"] else
                       "No real LLM provider configured — live tasks will block honestly. "
                       "Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY (optionally OPENAI_BASE_URL). "
                       "Scenario demos (SCRIPTED) and replays remain available."}


@app.get("/api/doctor")
async def doctor():
    g = fama_app.live.gateway
    prov = g.provider_status()
    models = [m.to_dict() for m in g.catalog()]
    hints = []
    if not prov["any_real"]:
        hints.append("set OPENAI_API_KEY or ANTHROPIC_API_KEY to enable live tasks")
    if prov["openai_compatible"]:
        hints.append(f"custom endpoint: {os.environ.get('OPENAI_BASE_URL')}")
    if not (os.environ.get("TAVILY_API_KEY") or os.environ.get("BRAVE_API_KEY")):
        hints.append("web_search needs TAVILY_API_KEY or BRAVE_API_KEY (research tasks)")
    return {"providers": prov, "models": models, "hints": hints,
            "sandbox_note": "soft sandbox: rlimits + scrubbed env + workspace scope; "
                            "no kernel network isolation in this environment"}


@app.post("/api/tasks")
async def create_task(req: Request):
    body = await req.json()
    text = str(body.get("input", "")).strip()
    if not text:
        return JSONResponse({"error": "input required"}, status_code=400)
    autonomy = body.get("autonomy")
    try:
        autonomy = AutonomyLevel(autonomy) if autonomy else None
    except ValueError:
        autonomy = None
    files = body.get("workspace_files") or {}
    max_cost = float(body.get("max_cost_usd", 2.0))
    f = fama_app.live
    task, st = f.create_task(text, autonomy=autonomy, workspace_files=files,
                             max_cost_usd=max_cost)
    allow_assumptions = bool(body.get("allow_assumptions", False))

    async def _run():
        try:
            await f.run(st, allow_assumptions=allow_assumptions)
        except Exception as e:
            f.emit(st, "kernel_error", f"task crashed: {e}", level="error")

    fama_app.background[task.id] = asyncio.create_task(_run())
    return {"task_id": task.id, "status": task.status.value}


@app.get("/api/tasks")
async def list_tasks():
    out = []
    for tid, st in fama_app.all_states():
        t = st.task
        out.append({"id": t.id, "input": t.input[:140], "status": t.status.value,
                    "result": t.result_status.value if t.result_status else None,
                    "created_at": t.created_at, "cost_usd": t.cost_usd,
                    "scripted": tid in fama_app.demo_famas})
    out.sort(key=lambda x: x["created_at"], reverse=True)
    return out


@app.get("/api/tasks/{task_id}")
async def task_state(task_id: str):
    f = fama_app.fama_for(task_id)
    if not f:
        return JSONResponse({"error": "unknown task"}, status_code=404)
    d = f.state_dict(task_id)
    if d:
        d["scripted"] = task_id in fama_app.demo_famas
    return d


@app.get("/api/tasks/{task_id}/events")
async def task_events(task_id: str, limit: int = 2000):
    f = fama_app.fama_for(task_id)
    if not f:
        return JSONResponse({"error": "unknown task"}, status_code=404)
    return [e.to_dict() for e in f.bus.history(task_id, limit=limit)]


@app.post("/api/tasks/{task_id}/clarify")
async def clarify(task_id: str, req: Request):
    body = await req.json()
    f = fama_app.fama_for(task_id)
    if not f:
        return JSONResponse({"error": "unknown task"}, status_code=404)
    ok = f.answer_clarification(task_id, [str(a) for a in body.get("answers", [])])
    return {"ok": ok}


@app.post("/api/tasks/{task_id}/approve")
async def approve(task_id: str, req: Request):
    body = await req.json()
    f = fama_app.fama_for(task_id)
    if not f:
        return JSONResponse({"error": "unknown task"}, status_code=404)
    ok = f.decide_gate(task_id, str(body.get("gate_id", "")), bool(body.get("approve", True)))
    return {"ok": ok}


@app.get("/api/agents")
async def agents():
    f = fama_app.live
    return {"agents": [a.to_dict() for a in f.registry.list()],
            "performance": f.registry.performance.to_dict()}


@app.get("/api/models")
async def models():
    return {"models": [m.to_dict() for m in fama_app.live.gateway.catalog()],
            "note": "prices are approximate planning estimates"}


@app.get("/api/metrics")
async def metrics():
    f = fama_app.live
    return f.metrics.snapshot(f.gateway, f.registry)


@app.get("/api/scenarios")
async def scenarios():
    return {"scenarios": [{"name": s.name, "title": s.title, "description": s.description,
                           "task": s.task} for s in SCENARIOS.values()]}


@app.post("/api/scenarios/{name}/run")
async def run_scenario(name: str):
    """Deterministic offline demo — clearly labelled SCRIPTED everywhere."""
    sc = SCENARIOS.get(name)
    if not sc:
        return JSONResponse({"error": "unknown scenario"}, status_code=404)
    demo = FAMA(scripted_gateway(sc), Store(":memory:"), base_dir=f"{fama_app.base_dir}/demo")
    task, st = demo.create_task(sc.task, workspace_files=sc.files)
    fama_app.demo_famas[task.id] = demo
    demo.bus.subscribe(fama_app._fan_in)

    async def _run():
        answers = list(sc.clarify_answers) if sc.clarify_answers else None
        while not runner.done():
            if answers and st.task.status == TaskStatus.AWAITING_CLARIFICATION:
                demo.answer_clarification(task.id, answers)
                answers = None
            await asyncio.sleep(0.05)
    runner = asyncio.create_task(demo.run(st))

    async def _watch():
        await runner
    fama_app.background[task.id] = asyncio.create_task(_run())
    fama_app.background[task.id + ":run"] = runner
    return {"task_id": task.id, "scripted": True,
            "note": "deterministic scripted demo — not a real model run"}


@app.get("/api/replays")
async def replays():
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(REPLAY_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            out.append({"name": p.stem, "title": data.get("title", p.stem),
                        "description": data.get("description", ""),
                        "recorded_at": data.get("recorded_at")})
        except Exception:
            continue
    return {"replays": out}


@app.get("/api/replays/{name}")
async def replay(name: str):
    p = REPLAY_DIR / f"{name}.json"
    if not p.exists():
        return JSONResponse({"error": "unknown replay"}, status_code=404)
    return json.loads(p.read_text())


@app.post("/api/replays/{name}/record")
async def record(name: str):
    sc = SCENARIOS.get(name)
    if not sc:
        return JSONResponse({"error": "unknown scenario"}, status_code=404)
    rep = await asyncio.get_event_loop().run_in_executor(None, record_replay, sc, REPLAY_DIR)
    return {"ok": True, "replay": name, "events": len(rep["events"])}


@app.get("/api/stream")
async def stream(task_id: Optional[str] = None, replay_from: int = 0):
    q: asyncio.Queue = asyncio.Queue()
    fama_app.clients.append(q)
    # replay existing history first (missed events)
    if task_id:
        for ev in fama_app.live.bus.history(task_id)[replay_from:]:
            q.put_nowait(ev.to_dict())

    async def gen():
        try:
            while True:
                try:
                    item = await asyncio.wait_for(q.get(), timeout=20)
                    yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            fama_app.clients.remove(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


app.mount("/static", StaticFiles(directory=str(WORLD_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (WORLD_DIR / "index.html").read_text()
