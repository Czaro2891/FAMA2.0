"""Capability System + Agent Model + Registry + Selection (sec. 6-9, 21-22).

Agents are executors of capability sets.  An agent is selected because its
real capabilities match task requirements — never because of its name.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .core import (AgentDNA, AgentPerformanceRecord, AgentSpec, Capability,
                   CapabilitySource, ModelClass, RequiredCapability, RiskLevel,
                   TaskUnderstanding, new_id)

# ---------------------------------------------------------------- capability catalog (sec. 6)

def _cap(cid: str, name: str, source: CapabilitySource, quality: float,
         verification: str = "", cost: float = 1.0) -> Capability:
    return Capability(cid, name, source, quality, verification, cost)


SYSTEM_CAPABILITIES: dict[str, Capability] = {
    c.id: c for c in [
        _cap("python_execution", "Python execution", CapabilitySource.TOOL_BASED, 0.95, "sandbox runs exit code", 0.2),
        _cap("debugging", "Debugging", CapabilitySource.HYBRID, 0.8, "fixed code passes independent tests", 1.5),
        _cap("unit_testing", "Unit / integration testing", CapabilitySource.HYBRID, 0.85, "test run + mutation score", 1.0),
        _cap("mutation_testing", "Mutation testing", CapabilitySource.TOOL_BASED, 0.9, "mutant kill-rate determinism", 1.2),
        _cap("static_analysis", "Static analysis", CapabilitySource.TOOL_BASED, 0.7, "AST checks", 0.4),
        _cap("frontend", "Frontend development", CapabilitySource.MODEL_BASED, 0.7, "review + build", 1.2),
        _cap("backend", "Backend development", CapabilitySource.MODEL_BASED, 0.75, "tests + review", 1.2),
        _cap("database", "Database operations", CapabilitySource.MODEL_BASED, 0.65, "domain rules", 1.2),
        _cap("sql", "SQL", CapabilitySource.MODEL_BASED, 0.7, "execution on schema", 1.0),
        _cap("security_analysis", "Security analysis", CapabilitySource.MODEL_BASED, 0.65, "checklist + scanner", 1.5),
        _cap("research", "Research", CapabilitySource.HYBRID, 0.75, "source validation", 1.5),
        _cap("source_validation", "Source validation", CapabilitySource.HYBRID, 0.7, "cross-source agreement", 1.0),
        _cap("statistical_analysis", "Statistical analysis", CapabilitySource.MODEL_BASED, 0.7, "recomputation", 1.3),
        _cap("data_processing", "Data processing", CapabilitySource.HYBRID, 0.75, "schema + invariant checks", 1.0),
        _cap("experiment_execution", "Experiment execution", CapabilitySource.TOOL_BASED, 0.85, "deterministic reruns", 1.0),
        _cap("code_review", "Code review", CapabilitySource.MODEL_BASED, 0.75, "independent reviewer agreement", 1.0),
        _cap("architecture", "Software architecture", CapabilitySource.MODEL_BASED, 0.75, "critique + constraints", 1.5),
        _cap("optimization", "Optimization", CapabilitySource.HYBRID, 0.7, "benchmark deltas", 1.5),
        _cap("benchmarking", "Benchmarking", CapabilitySource.TOOL_BASED, 0.9, "repeat variance", 0.5),
        _cap("writing", "Writing / synthesis", CapabilitySource.MODEL_BASED, 0.7, "critique pass", 1.0),
        _cap("critique", "Adversarial critique", CapabilitySource.MODEL_BASED, 0.7, "counterexample survival", 1.2),
        _cap("planning", "Planning / decomposition", CapabilitySource.MODEL_BASED, 0.75, "plan survives execution", 1.0),
    ]
}


def cap(cid: str) -> Capability:
    return SYSTEM_CAPABILITIES.get(cid, Capability(cid, cid, CapabilitySource.MODEL_BASED, 0.5))


# ---------------------------------------------------------------- builtin archetypes (sec. 7)

def _agent(name: str, description: str, caps: list[tuple[str, float]], domains: list[str],
           tools: list[str], models: list[str], dna: AgentDNA | None = None,
           reliability: float = 0.8, trust: float = 0.8, cost: float = 1.0,
           risk: RiskLevel = RiskLevel.LOW, failure_modes: list[str] | None = None) -> AgentSpec:
    capabilities = [Capability(cid, cap(cid).name, cap(cid).source, q, cap(cid).verification)
                    for cid, q in caps]
    return AgentSpec(
        id=new_id("agent-" + name), name=name, description=description,
        capabilities=capabilities, limitations=[],
        permissions=["workspace"], tools=tools, preferred_models=models,
        domains=domains, reliability=reliability, trust=trust,
        cost_factor=cost, availability=1.0, risk_profile=risk,
        failure_modes=failure_modes or [], dna=dna or AgentDNA(preferred_model_classes=models,
                                                               preferred_tools=tools))


ARCHETYPES: list[AgentSpec] = [
    _agent("analyst", "Problem analyst & decomposer",
           [("planning", 0.8), ("architecture", 0.7), ("static_analysis", 0.6)],
           ["software", "general"], ["fs_read", "fs_list"], ["reasoning"],
           AgentDNA(reasoning_style="analytical", verification_bias=0.8, cost_sensitivity=0.3),
           reliability=0.85, cost=1.2),
    _agent("coder", "Implementation specialist",
           [("backend", 0.85), ("frontend", 0.7), ("python_execution", 0.8), ("debugging", 0.7)],
           ["software"], ["fs_read", "fs_write", "fs_list", "python_run", "test_run", "git"],
           ["coding"],
           AgentDNA(reasoning_style="pragmatic", coding_style="clean", verification_bias=0.6),
           reliability=0.82, failure_modes=["overconfident about untested edge cases"]),
    _agent("coder-2", "Independent second implementer (diversity source)",
           [("backend", 0.8), ("python_execution", 0.8), ("debugging", 0.65)],
           ["software"], ["fs_read", "fs_write", "fs_list", "python_run", "test_run"],
           ["coding", "reasoning"],
           AgentDNA(reasoning_style="analytical", coding_style="defensive",
                    verification_bias=0.8, cost_sensitivity=0.3),
           reliability=0.8, cost=1.1),
    _agent("reviewer", "Code reviewer",
           [("code_review", 0.85), ("static_analysis", 0.7), ("backend", 0.6)],
           ["software"], ["fs_read", "fs_list", "git"], ["reasoning"],
           AgentDNA(reasoning_style="analytical", verification_bias=0.9, creativity=0.2),
           reliability=0.88, cost=0.9),
    _agent("tester", "Test engineer",
           [("unit_testing", 0.88), ("python_execution", 0.8), ("data_processing", 0.6)],
           ["software"], ["fs_read", "fs_write", "fs_list", "python_run", "test_run", "mutation"],
           ["coding"],
           AgentDNA(reasoning_style="adversarial", verification_bias=0.95, creativity=0.5),
           reliability=0.9, cost=1.0),
    _agent("security", "Security analyst",
           [("security_analysis", 0.8), ("code_review", 0.65), ("static_analysis", 0.6)],
           ["security", "software"], ["fs_read", "fs_list"], ["adversarial", "reasoning"],
           AgentDNA(reasoning_style="adversarial", risk_tolerance=0.15,
                    verification_bias=0.9, creativity=0.3),
           reliability=0.85, cost=1.3, risk=RiskLevel.MEDIUM),
    _agent("researcher", "Research & source validation",
           [("research", 0.85), ("source_validation", 0.8), ("writing", 0.7)],
           ["research", "general"], ["web_search", "web_fetch", "fs_read", "fs_write"],
           ["reasoning"],
           AgentDNA(reasoning_style="analytical", exploration=0.8, verification_bias=0.7),
           reliability=0.8, cost=1.2),
    _agent("skeptic", "Adversarial critic / contradiction seeker",
           [("critique", 0.85), ("unit_testing", 0.7), ("security_analysis", 0.55)],
           ["general", "software"], ["fs_read", "fs_write", "fs_list", "python_run"],
           ["adversarial"],
           AgentDNA(reasoning_style="adversarial", verification_bias=0.95,
                    creativity=0.7, risk_tolerance=0.3),
           reliability=0.83, cost=1.1),
    _agent("data-analyst", "Data analysis & statistics",
           [("statistical_analysis", 0.8), ("data_processing", 0.8), ("python_execution", 0.75)],
           ["data"], ["fs_read", "fs_write", "fs_list", "python_run"], ["reasoning"],
           reliability=0.8, cost=1.1),
    _agent("optimizer", "Performance optimization specialist",
           [("optimization", 0.8), ("benchmarking", 0.85), ("python_execution", 0.75), ("debugging", 0.6)],
           ["software"], ["fs_read", "fs_write", "fs_list", "python_run", "benchmark"],
           ["coding", "reasoning"],
           AgentDNA(reasoning_style="analytical", verification_bias=0.8),
           reliability=0.78, cost=1.2),
    _agent("verifier", "Independent verification orchestrator",
           [("unit_testing", 0.8), ("mutation_testing", 0.85), ("benchmarking", 0.7),
            ("experiment_execution", 0.8)],
           ["software", "general"], ["fs_read", "fs_write", "fs_list", "python_run", "test_run", "mutation", "benchmark"],
           ["reasoning", "coding"],
           AgentDNA(reasoning_style="adversarial", verification_bias=1.0, cost_sensitivity=0.2),
           reliability=0.9, cost=1.0),
]

ARCHETYPE_BY_NAME = {a.name: a for a in ARCHETYPES}


# ---------------------------------------------------------------- registry (sec. 8)

@dataclass
class TeamMember:
    agent: AgentSpec
    role: str = ""                # e.g. implementer | verifier | critic
    model: str | None = None      # provider-qualified model id chosen by routing
    model_class: str | None = None
    steps: list[str] = field(default_factory=list)
    score: float = 0.0
    score_breakdown: dict = field(default_factory=dict)

    def to_dict(self):
        from .core import dc_to_dict
        return dc_to_dict(self)


class AgentRegistry:
    """Dynamic registry answering: who exists, what can they do, how well."""

    def __init__(self, performance: "PerformanceTracker | None" = None):
        self.agents: dict[str, AgentSpec] = {a.id: a for a in ARCHETYPES}
        self.utilization: dict[str, int] = {}
        self.performance = performance or PerformanceTracker()

    def list(self) -> list[AgentSpec]:
        return sorted(self.agents.values(), key=lambda a: a.name)

    def by_name(self, name: str) -> Optional[AgentSpec]:
        for a in self.agents.values():
            if a.name == name:
                return a
        return None

    def register(self, spec: AgentSpec) -> AgentSpec:
        self.agents[spec.id] = spec
        return spec

    def mark_busy(self, agent_id: str, delta: int = 1):
        self.utilization[agent_id] = max(0, self.utilization.get(agent_id, 0) + delta)

    def candidates(self, required: RequiredCapability) -> list[AgentSpec]:
        out = []
        for a in self.agents.values():
            if a.capability_quality(required.capability) >= required.min_quality:
                out.append(a)
        return out

    def capability_gap(self, requirements: list[RequiredCapability]) -> list[RequiredCapability]:
        gaps = []
        for r in requirements:
            if not self.candidates(r):
                gaps.append(r)
        return gaps


# ---------------------------------------------------------------- selection (sec. 9)

class SelectionEngine:
    """Capability matching: TASK -> required caps -> available caps -> team.

    Prefers the smallest team that guarantees the required quality level.
    """

    def __init__(self, registry: AgentRegistry, models_available: list[str] | None = None):
        self.registry = registry
        self.models_available = models_available or []

    def score(self, agent: AgentSpec, req: RequiredCapability, und: TaskUnderstanding,
              already_picked: list[AgentSpec], strict_min: bool = True) -> Optional[dict]:
        q = agent.capability_quality(req.capability)
        if strict_min and q < req.min_quality:
            return None
        perf = self.registry.performance.record_for(agent.id, req.capability)
        cap_match = q                                  # absolute capability quality
        domain_fit = 1.0 if und.domain in agent.domains or "general" in agent.domains else 0.5
        history = perf.success_rate if (perf.successes + perf.failures) >= 3 else 0.5
        availability = agent.availability
        cost_score = 1.0 / (1.0 + agent.cost_factor)
        latency_score = 1.0 / (1.0 + agent.latency_factor)
        # diversity: penalise same-model picks already in team (common-mode risk, sec. 31/40)
        diversity = 1.0
        for picked in already_picked:
            overlap = set(agent.preferred_models) & set(picked.preferred_models)
            if overlap:
                diversity -= 0.35 / max(1, len(already_picked))
        diversity = max(0.0, diversity)
        trust = agent.trust
        composite = (0.30 * cap_match + 0.15 * reliability_norm(agent) + 0.10 * domain_fit +
                     0.10 * history + 0.05 * availability + 0.10 * cost_score +
                     0.05 * latency_score + 0.10 * diversity + 0.05 * trust)
        return {"agent": agent.name, "agent_id": agent.id,
                "capability_match": round(cap_match, 3), "reliability": round(reliability_norm(agent), 3),
                "domain_fit": round(domain_fit, 3), "history": round(history, 3),
                "availability": availability, "cost_score": round(cost_score, 3),
                "latency_score": round(latency_score, 3), "diversity": round(diversity, 3),
                "trust": round(trust, 3), "composite": round(composite, 4),
                "perf_records": perf.successes + perf.failures}

    def select_team(self, requirements: list[RequiredCapability],
                    und: TaskUnderstanding, max_agents: int = 6,
                    avoid: set[str] | None = None) -> tuple[list[TeamMember], list[dict]]:
        """Greedy weighted set-cover over capability requirements."""
        picked: list[AgentSpec] = []
        members: list[TeamMember] = []
        trace: list[dict] = []
        remaining = sorted(requirements, key=lambda r: -r.importance)
        for req in remaining:
            if any(p.capability_quality(req.capability) >= req.min_quality for p in picked):
                continue
            scored = []
            for a in self.registry.candidates(req):
                if avoid and a.name in avoid:
                    continue
                s = self.score(a, req, und, picked)
                if s:
                    scored.append((s["composite"], s, a))
            if not scored:
                trace.append({"capability": req.capability, "candidates": [],
                              "decision": "capability_gap"})
                continue
            scored.sort(key=lambda x: -x[0])
            best_score, best, best_agent = scored[0]
            picked.append(best_agent)
            trace.append({"capability": req.capability,
                          "candidates": [{k: v for k, v in s.items() if k != "agent_id"}
                                         for _, s, _ in scored[:4]],
                          "selected": best_agent.name, "score": best_score,
                          "decision": "best capability fit"})
        # hard cap: smallest sufficient team
        if len(picked) > max_agents:
            trace.append({"decision": f"team capped at {max_agents} (smallest sufficient team rule)"})
            picked = picked[:max_agents]
        for a in picked:
            members.append(TeamMember(agent=a, role="executor", score=0.0))
        return members, trace


def reliability_norm(a: AgentSpec) -> float:
    return a.reliability


# ---------------------------------------------------------------- performance (sec. 22)

class PerformanceTracker:
    """Per (agent, capability) performance profiles — not a single score."""

    def __init__(self):
        self.records: dict[tuple[str, str], AgentPerformanceRecord] = {}

    def record_for(self, agent_id: str, capability: str) -> AgentPerformanceRecord:
        key = (agent_id, capability)
        if key not in self.records:
            self.records[key] = AgentPerformanceRecord(agent_id=agent_id, capability=capability)
        return self.records[key]

    def record_outcome(self, agent_id: str, capability: str, success: bool,
                       quality: float, cost_usd: float = 0.0, seconds: float = 0.0):
        from .core import now_utc
        r = self.record_for(agent_id, capability)
        total = r.successes + r.failures
        r.avg_quality = (r.avg_quality * total + quality) / (total + 1)
        r.avg_cost_usd = (r.avg_cost_usd * total + cost_usd) / (total + 1)
        r.avg_seconds = (r.avg_seconds * total + seconds) / (total + 1)
        if success:
            r.successes += 1
        else:
            r.failures += 1
        r.last_used = now_utc()

    def profile(self, agent_id: str) -> list[AgentPerformanceRecord]:
        return [r for (aid, _), r in self.records.items() if aid == agent_id]

    def to_dict(self):
        return [r.to_dict() for r in self.records.values()]


# ---------------------------------------------------------------- custom agents (sec. 21)

class AgentFactory:
    """Create agents only when existing capabilities are insufficient."""

    def __init__(self, registry: AgentRegistry, governance):
        self.registry = registry
        self.governance = governance

    def try_fill_gap(self, gap: RequiredCapability) -> tuple[Optional[AgentSpec], str]:
        """Order of attempts: extend existing -> retool -> new agent (probation)."""
        # 1. extend an existing close agent
        best = None
        for a in self.registry.list():
            q = a.capability_quality(gap.capability)
            if best is None or q > best.capability_quality(gap.capability):
                best = a
        if best is not None and best.capability_quality(gap.capability) >= gap.min_quality * 0.9:
            base = cap(gap.capability)
            best.capabilities.append(Capability(base.id, base.name, base.source,
                                                max(gap.min_quality, 0.6), base.verification))
            return best, f"extended existing agent '{best.name}' with capability '{gap.capability}'"
        # 2. new agent — governance-limited, probation
        custom_count = sum(1 for a in self.registry.agents.values() if a.custom)
        if not self.governance.custom_agent_allowed(custom_count):
            return None, "custom agent creation blocked by governance"
        limits = self.governance.custom_agent_limits()
        spec = AgentSpec(
            id=new_id("agent-custom"), name=f"custom-{gap.capability.replace('_', '-')}",
            description=f"Dynamically created for missing capability: {gap.capability}",
            capabilities=[Capability(gap.capability, cap(gap.capability).name,
                                     cap(gap.capability).source, 0.6, cap(gap.capability).verification)],
            permissions=limits["permissions"], tools=limits["tools_whitelist"],
            preferred_models=["reasoning"], reliability=0.5, trust=limits["trust"],
            custom=True, probation=True,
            system_prompt=f"You are a specialist executor for: {gap.capability}. "
                          f"You are in probation: be conservative, verify assumptions, "
                          f"report uncertainty explicitly.",
            dna=AgentDNA(reasoning_style="pragmatic", verification_bias=0.9,
                         risk_tolerance=0.2, cost_sensitivity=0.4))
        self.registry.register(spec)
        return spec, (f"created new agent '{spec.name}' (probation, trust={spec.trust}, "
                      f"tools={spec.tools})")
