"""End-to-end scenario tests: different problems -> different system decisions
(spec sec. 45, 49)."""

import pytest

from fama.scenarios import SCENARIOS, run_scenario


async def _run(name, tmp):
    f, tid = await run_scenario(SCENARIOS[name], base_dir=str(tmp / f"s-{name}"))
    return f, f.states[tid]


@pytest.mark.asyncio
async def test_simple_function_cheap_path(tmp_path):
    f, st = await _run("simple-function", tmp_path)
    assert st.task.result_status.value == "verified"
    assert st.chosen_strategy.pattern == "specialist"
    # low risk => single deterministic-test oracle only (verification budget)
    assert len(st.oracle_runs) == 1
    assert st.oracle_runs[0].kind.value == "deterministic_test"
    assert st.task.replan_count == 0


@pytest.mark.asyncio
async def test_payments_bug_high_risk_full_verification(tmp_path):
    f, st = await _run("payments-bug", tmp_path)
    assert st.task.result_status.value == "verified"
    assert st.chosen_strategy.pattern == "diagnose_fix_verify"
    kinds = [r.kind.value for r in st.oracle_runs]
    assert "mutation" in kinds and "independent_implementation" in kinds
    mut = next(r for r in st.oracle_runs if r.kind.value == "mutation")
    assert mut.verdict == "pass" and mut.measurements["score"] >= 0.7
    # evidence graph must contain the fix action and oracle nodes
    assert len(st.evidence.nodes) >= 8
    assert len(st.decisions.records) >= 1


@pytest.mark.asyncio
async def test_tech_compare_research_strategy(tmp_path):
    f, st = await _run("tech-compare", tmp_path)
    assert st.task.result_status.value == "verified"
    assert st.chosen_strategy.pattern == "research_synthesis"
    kinds = [r.kind.value for r in st.oracle_runs]
    assert "external_source" in kinds


@pytest.mark.asyncio
async def test_optimize_benchmark_differential(tmp_path):
    f, st = await _run("optimize-algorithm", tmp_path)
    assert st.task.result_status.value == "verified"
    assert st.chosen_strategy.pattern == "optimize_measure"
    kinds = [r.kind.value for r in st.oracle_runs]
    assert "benchmark" in kinds and "differential" in kinds


@pytest.mark.asyncio
async def test_vague_app_asks_before_assuming(tmp_path):
    f, st = await _run("vague-app", tmp_path)
    # clarification must have been requested and answered
    types = [e.type for e in f.bus.history(st.task.id)]
    assert "clarification_requested" in types
    assert "clarification_received" in types
    assert st.task.result_status.value == "verified"


@pytest.mark.asyncio
async def test_weak_tests_triggers_contradiction_and_replan(tmp_path):
    f, st = await _run("weak-tests", tmp_path)
    types = [e.type for e in f.bus.history(st.task.id)]
    # the adaptive star: weak verification -> escalation -> refutation -> new plan
    assert "verification_escalated" in types
    assert "verification_failure" in types
    assert "plan_changed" in types
    assert st.task.replan_count == 1
    assert st.task.plan_versions == 2
    assert st.task.result_status.value == "verified"
    # round 1 mutation must be weak; a property-based oracle must refute the claim
    mut1 = [r for r in st.oracle_runs if r.kind.value == "mutation"][0]
    assert mut1.verdict == "weak"
    prop = [r for r in st.oracle_runs if r.kind.value == "property_based"][0]
    assert prop.verdict == "refuted"


@pytest.mark.asyncio
async def test_different_problems_choose_different_strategies(tmp_path):
    """Sec. 49: różne problemy prowadzą do różnych decyzji systemu."""
    patterns = {}
    for name in ("simple-function", "payments-bug", "tech-compare",
                 "optimize-algorithm", "weak-tests"):
        f, st = await _run(name, tmp_path)
        patterns[name] = st.chosen_strategy.pattern
    assert len(set(patterns.values())) >= 4, patterns
    assert patterns["simple-function"] != patterns["payments-bug"]
    assert patterns["tech-compare"] == "research_synthesis"
    assert patterns["optimize-algorithm"] == "optimize_measure"


@pytest.mark.asyncio
async def test_memory_learning_after_runs(tmp_path):
    """Second run of the same problem recalls history (sec. 23-24)."""
    from fama.scenarios import scripted_gateway
    from fama.orchestrator import FAMA
    from fama.store import Store

    sc = SCENARIOS["simple-function"]
    f = FAMA(scripted_gateway(sc), Store(":memory:"), base_dir=str(tmp_path / "mem"))
    task1, st1 = f.create_task(sc.task, workspace_files=sc.files)
    await f.run(st1)
    assert f.memory.entries, "strategy memory should record the run"
    task2, st2 = f.create_task(sc.task, workspace_files=sc.files)
    await f.run(st2)
    types = [e.type for e in f.bus.history(task2.id)]
    assert "memory_recalled" in types
    # the recalled strategy enters as a candidate and is re-evaluated
    assert any("memory prior" in (s["strategy"].get("rationale") or "")
               for s in f.states[task2.id].evaluated_strategies)
