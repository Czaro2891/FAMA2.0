"""FAMA 2.0 command line interface."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from . import __version__
from .core import AutonomyLevel, TaskStatus


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fama",
                                description="FAMA 2.0 — Adaptive Agent Operating System")
    p.add_argument("--version", action="version", version=f"FAMA {__version__}")
    sub = p.add_subparsers(dest="cmd")

    serve = sub.add_parser("serve", help="run the API server + World UI")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    serve.add_argument("--allow-scripted", action="store_true",
                       help="also enable scripted demos in live mode (labelled)")

    run = sub.add_parser("run", help="run a task end-to-end in the terminal")
    run.add_argument("input", help="task description")
    run.add_argument("--autonomy", choices=[a.value for a in AutonomyLevel], default=None)
    run.add_argument("--max-cost", type=float, default=2.0)
    run.add_argument("--yes", action="store_true",
                     help="accept explicit assumptions instead of asking")
    run.add_argument("--workspace", default=None, help="copy files from dir into task workspace")

    demo = sub.add_parser("demo", help="run a scripted demo scenario (deterministic, offline)")
    demo.add_argument("scenario", help="scenario name (see `fama demo --list`)")
    demo.add_argument("--list", action="store_true")

    record = sub.add_parser("record", help="record scenario replays for the World UI")
    record.add_argument("scenario", nargs="?", default=None)
    record.add_argument("--all", action="store_true")

    sub.add_parser("doctor", help="check providers, models and environment")
    sub.add_parser("agents", help="list registered agents")
    sub.add_parser("models", help="list models in the catalog")
    sub.add_parser("memory", help="show strategy memory")
    return p


# ---------------------------------------------------------------- commands

def cmd_doctor():
    from .llm import LLMGateway
    g = LLMGateway()
    prov = g.provider_status()
    print("FAMA doctor")
    print("  providers:")
    for k, v in prov.items():
        if k == "any_real":
            continue
        print(f"    {k:18s} {'OK' if v else '—'}")
    if not prov["any_real"]:
        print("  ! no real provider: set OPENAI_API_KEY and/or ANTHROPIC_API_KEY")
        print("    (OPENAI_BASE_URL adds any OpenAI-compatible endpoint; FAMA_MODELS registers models)")
    if not (os.environ.get("TAVILY_API_KEY") or os.environ.get("BRAVE_API_KEY")):
        print("  ! web_search disabled: set TAVILY_API_KEY or BRAVE_API_KEY")
    print("  models:")
    for m in g.catalog():
        print(f"    {'*' if m.available else ' '} {m.id:38s} "
              f"{'+'.join(c.value for c in m.classes):24s} "
              f"${m.price_in}/${m.price_out} per Mtok")
    print("  sandbox: soft (rlimits + env scrub + workspace scope); "
          "no kernel network isolation in this environment")


def cmd_agents():
    from .agents import ARCHETYPES
    for a in ARCHETYPES:
        print(f"{a.name:14s} caps: {', '.join(f'{c.id}({c.quality})' for c in a.capabilities)}")
        print(f"{'':14s} models: {','.join(a.preferred_models)} · tools: {','.join(a.tools)}")


def cmd_models():
    from .llm import LLMGateway
    g = LLMGateway()
    for m in g.catalog():
        print(f"{'*' if m.available else ' '} {m.id:38s} "
              f"{'+'.join(c.value for c in m.classes):24s}")


def cmd_memory():
    from .memory import StrategyMemory
    mem = StrategyMemory()
    if not mem.entries:
        print("strategy memory is empty — run tasks first")
        return
    for e in mem.entries[-20:]:
        print(f"{e.ts[:19]} {e.fingerprint:52s} {e.strategy_pattern:24s} "
              f"-> {e.result:12s} ver={e.verification_strength} replans={e.replans}")


def cmd_run(args):
    from .llm import LLMGateway
    from .orchestrator import FAMA
    from .store import Store

    files = {}
    if args.workspace:
        from pathlib import Path
        for p in Path(args.workspace).rglob("*"):
            if p.is_file() and not any(x.startswith(".") for x in p.parts):
                files[str(p.relative_to(args.workspace))] = p.read_text(errors="replace")
    gateway = LLMGateway()
    fama = FAMA(gateway, Store(), base_dir=".fama")
    task, st = fama.create_task(args.input,
                                autonomy=AutonomyLevel(args.autonomy) if args.autonomy else None,
                                workspace_files=files, max_cost_usd=args.max_cost)
    _print_stream(fama, task.id)
    try:
        asyncio.run(fama.run(st, allow_assumptions=args.yes))
    finally:
        asyncio.run(gateway.close())
    _print_report(fama, task.id)


def _print_stream(fama, task_id):
    from .core import Event

    def show(ev: Event):
        icon = {"success": "✓", "warning": "!", "error": "✗", "info": "·"}[ev.level]
        print(f"  {icon} [{ev.phase or ev.type}] {ev.title}")
    fama.bus.subscribe(show)


def _print_report(fama, task_id):
    st = fama.states[task_id]
    t = st.task
    print("\n" + "=" * 70)
    print(f"RESULT: {t.result_status.value.upper() if t.result_status else '?'}")
    print(f"  {t.result_summary}")
    print(f"  cost ${t.cost_usd:.4f} · {t.duration_s:.1f}s · plans V{t.plan_versions} "
          f"· failures {t.failure_count} · replans {t.replan_count}")
    if st.oracle_runs:
        print("  oracles:")
        for r in st.oracle_runs:
            print(f"    - {r.kind.value:26s} {r.verdict:8s} {r.detail[:80]}")
    if t.final_artifact:
        print(f"  artifact: {t.final_artifact}")


def cmd_demo(args):
    from .scenarios import SCENARIOS, run_scenario
    if args.list or args.scenario not in SCENARIOS:
        print("scenarios:")
        for s in SCENARIOS.values():
            print(f"  {s.name:20s} {s.title}")
        if not args.list:
            print("\n(deterministic scripted provider — clearly not a real model run)")
        return
    sc = SCENARIOS[args.scenario]
    print(f"SCENARIO: {sc.title}  [SCRIPTED — deterministic demo, not a model run]\n")
    got = asyncio.run(_run_demo(sc))
    fama, tid = got
    _print_report(fama, tid)


async def _run_demo(sc):
    from .scenarios import run_scenario
    return await run_scenario(sc, base_dir=".fama/demo")


def cmd_record(args):
    from .scenarios import SCENARIOS, record_replay
    names = list(SCENARIOS) if args.all or not args.scenario else [args.scenario]
    for name in names:
        rep = record_replay(SCENARIOS[name])
        print(f"recorded {name}: {len(rep['events'])} events -> .fama/replays/{name}.json")


def cmd_serve(args):
    import uvicorn
    if args.allow_scripted:
        os.environ["FAMA_ALLOW_SCRIPTED"] = "1"
    from .server import app
    print(f"FAMA 2.0 World UI → http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "doctor":
        cmd_doctor()
    elif args.cmd == "agents":
        cmd_agents()
    elif args.cmd == "models":
        cmd_models()
    elif args.cmd == "memory":
        cmd_memory()
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "demo":
        cmd_demo(args)
    elif args.cmd == "record":
        cmd_record(args)
    elif args.cmd == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
