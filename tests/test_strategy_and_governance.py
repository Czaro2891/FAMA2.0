import pytest

from fama.agents import AgentRegistry, PerformanceTracker, SelectionEngine
from fama.core import (AutonomyLevel, Complexity, OracleKind, RequiredCapability,
                       RiskLevel, TaskType, TaskUnderstanding)
from fama.governance import Governance
from fama.llm import LLMGateway
from fama.memory import StrategyMemory
from fama.strategy import StrategyEngine, weight_profile
from fama.twin import DigitalTwin


def und(**kw):
    d = dict(goal="g", deliverable="d", domain="software",
             task_type=TaskType.DEBUGGING, risk_level=RiskLevel.HIGH,
             complexity=Complexity.MODERATE)
    d.update(kw)
    return TaskUnderstanding(**d)


def engine():
    reg = AgentRegistry(PerformanceTracker())
    twin = DigitalTwin(reg, LLMGateway())
    return StrategyEngine(twin, StrategyMemory()), reg


def test_weight_profiles_depend_on_risk():
    simple = TaskUnderstanding(risk_level=RiskLevel.LOW, complexity=Complexity.TRIVIAL)
    critical = TaskUnderstanding(risk_level=RiskLevel.CRITICAL, complexity=Complexity.COMPLEX)
    name_s, w_s = weight_profile(simple)
    name_c, w_c = weight_profile(critical)
    assert name_s == "cheap_fast"
    assert name_c == "safety_critical"
    assert w_s["cost"] > w_c["cost"]
    assert w_c["verification"] > w_s["verification"]


def test_debugging_high_risk_prefers_diagnose_strategy():
    se, _ = engine()
    ev = se.search(und(), AutonomyLevel.HIGH)
    assert ev[0].strategy.pattern == "diagnose_fix_verify"


def test_trivial_low_risk_prefers_cheapest():
    se, _ = engine()
    u = und(task_type=TaskType.CODE_GENERATION, risk_level=RiskLevel.LOW,
            complexity=Complexity.TRIVIAL)
    ev = se.search(u, AutonomyLevel.MINIMAL)
    assert ev[0].strategy.pattern == "specialist"
    # specialist should dominate on cost factor
    sp = next(e for e in ev if e.strategy.pattern == "specialist")
    assert sp.factors["cost"] >= max(e.factors["cost"] for e in ev) - 0.01


def test_research_uses_research_strategy_with_sources():
    se, _ = engine()
    u = und(task_type=TaskType.RESEARCH, risk_level=RiskLevel.MEDIUM)
    ev = se.search(u, AutonomyLevel.STANDARD)
    assert ev[0].strategy.pattern == "research_synthesis"
    assert OracleKind.EXTERNAL_SOURCE in ev[0].strategy.verification_bundle


def test_strategy_estimates_are_labelled_predictions():
    se, _ = engine()
    ev = se.search(und(), AutonomyLevel.STANDARD)
    for e in ev:
        assert e.strategy.twin_prediction is True
        assert e.twin.twin_prediction is True


def test_autonomy_level_widens_strategy_search():
    se, _ = engine()
    u = und(task_type=TaskType.CODE_GENERATION)
    minimal = se.generate(u, AutonomyLevel.MINIMAL, None)
    high = se.generate(u, AutonomyLevel.HIGH, None)
    assert len(high) > len(minimal)


def test_selection_prefers_real_capability_quality():
    reg = AgentRegistry(PerformanceTracker())
    sel = SelectionEngine(reg)
    u = und(task_type=TaskType.CODE_GENERATION)
    reqs = [RequiredCapability("backend", 0.6, 0.9)]
    members, trace = sel.select_team(reqs, u)
    assert members, "should find candidates for backend"
    best = max(reg.agents.values(), key=lambda a: a.capability_quality("backend"))
    assert members[0].agent.capability_quality("backend") == \
        best.capability_quality("backend")


def test_memory_prior_influences_second_run():
    se, reg = engine()
    u = und()
    # record a failed dual run and a successful diagnose run in memory
    from fama.strategy import TEMPLATES
    dual = TEMPLATES["dual_implementation"](u)
    diag = TEMPLATES["diagnose_fix_verify"](u)
    se.memory.record(u, dual, result="failed", verification_strength=0.2,
                     cost_usd=0.5, seconds=60)
    se.memory.record(u, diag, result="verified", verification_strength=0.9,
                     cost_usd=0.4, seconds=50)
    prior_dual, n_dual = se.memory.prior_success("dual_implementation", u)
    assert prior_dual == 0.0 and n_dual == 1
    recall = se.memory.recall(u)
    assert recall is not None and recall.n_total == 2
    ev = se.search(u, AutonomyLevel.HIGH, recall)
    assert ev[0].strategy.pattern == "diagnose_fix_verify"
    dual_ev = next(e for e in ev if e.strategy.pattern == "dual_implementation")
    assert dual_ev.factors["memory_prior"] == 0.0


# ---------------------------------------------------------------- governance

def test_governance_flags_production_and_requires_approval():
    from fama.core import Task
    g = Governance()
    und_prod = und(goal="deploy to production")
    task = Task(id="t1", input="wdroż zmianę na produkcję")
    flags = g.assess(task, und_prod)
    assert flags["production"] is True
    gates = g.approval_required(task, und_prod, flags)
    assert any(gt.category == "production" for gt in gates)


def test_governance_network_disabled_by_default():
    g = Governance()
    with pytest.raises(PermissionError):
        g.check_tool("web_fetch")


def test_governance_custom_agent_limits():
    g = Governance()
    assert g.custom_agent_allowed(0) is True
    assert g.custom_agent_allowed(1) is False
    limits = g.custom_agent_limits()
    assert limits["probation"] is True
    assert limits["trust"] < 0.5
    assert "web_search" not in limits["tools_whitelist"]


def test_agent_factory_creates_probation_agent_for_gap():
    from fama.agents import AgentFactory
    g = Governance()
    reg = AgentRegistry(PerformanceTracker())
    factory = AgentFactory(reg, g)
    gap = RequiredCapability("some_exotic_capability", 0.6, 0.9)
    agent, how = factory.try_fill_gap(gap)
    assert agent is not None
    assert agent.probation is True
    assert agent.custom is True
    assert "probation" in how or "created" in how
