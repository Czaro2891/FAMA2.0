"""Strategy Engine (sec. 11-13) + Assumption Engine (sec. 14).

FAMA first searches for a WAY to solve the problem, then picks people/models
for it.  Base strategies are only starting points — candidates are composed,
twin-simulated and utility-scored with task-dependent weights.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .core import (Assumption, AssumptionStatus, AutonomyLevel, Complexity,
                   OracleKind, RiskLevel, Strategy, StrategyStep, TaskUnderstanding,
                   clamp, new_id)
from .memory import Recall, StrategyMemory
from .twin import DigitalTwin, TwinEstimate

# ---------------------------------------------------------------- step templates

def S(name: str, goal: str, capability: str, *, deps: list[str] | None = None,
      verif: list[OracleKind] | None = None, parallelizable: bool = True,
      tokens: int = 2600) -> StrategyStep:
    return StrategyStep(id=new_id("ss"), name=name, goal=goal, capability=capability,
                        inputs=deps or [], verification=verif or [],
                        parallelizable=parallelizable, estimated_tokens=tokens)


TEST = [OracleKind.DETERMINISTIC_TEST]
TEST_MUT = [OracleKind.DETERMINISTIC_TEST, OracleKind.MUTATION]
DUAL = [OracleKind.INDEPENDENT_IMPLEMENTATION, OracleKind.DIFFERENTIAL]
RESEARCH_V = [OracleKind.EXTERNAL_SOURCE]


def template_single_specialist(und: TaskUnderstanding) -> Strategy:
    ver = TEST if und.risk_level.value in ("negligible", "low") else TEST_MUT
    return Strategy(
        id=new_id("strat"), name="A · Single specialist + test", pattern="specialist",
        description="One implementer covers the requirement; a deterministic test oracle checks the result. "
                    "Cheapest sensible path for low-risk work.",
        steps=[S("implement", "Produce the required artifact", _main_cap(und), verif=ver, tokens=4200),
               S("test", "Run/extend tests proving success criteria", "unit_testing", deps=["implement"],
                 verif=ver, tokens=2000)],
        verification_bundle=ver, team_size=1)


def template_pipeline(und: TaskUnderstanding) -> Strategy:
    ver = TEST_MUT if und.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) else TEST
    return Strategy(
        id=new_id("strat"), name="B · Analysis → implementation → review → tests → security",
        pattern="pipeline",
        description="Specialist pipeline: analysis, implementation, independent review, tests, "
                    "security pass and verification. Balanced quality for medium/high risk.",
        steps=[S("analyze", "Formalize approach and constraints", "planning", tokens=1800),
               S("implement", "Implement the change", _main_cap(und), deps=["analyze"], tokens=4200),
               S("review", "Independent review of the artifact", "code_review", deps=["implement"], tokens=2200),
               S("test", "Independent tests of success criteria", "unit_testing", deps=["implement"], verif=ver, tokens=2600),
               S("security", "Security pass", "security_analysis", deps=["implement"], tokens=1800),
               S("verify", "Verification bundle execution", "mutation_testing", deps=["test", "review"],
                 verif=ver, parallelizable=False, tokens=1600)],
        verification_bundle=ver + [OracleKind.MUTATION] if und.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) else ver,
        team_size=4)


def template_dual(und: TaskUnderstanding) -> Strategy:
    return Strategy(
        id=new_id("strat"), name="C · Two independent implementations → differential",
        pattern="dual_implementation",
        description="Two implementers work independently (different models/styles); results are compared "
                    "on shared inputs — disagreement localises the error. Strongest correctness signal.",
        steps=[S("analyze", "Shared specification & comparison harness", "planning", tokens=2000),
               S("implement", "Implementation #1", _main_cap(und), deps=["analyze"], tokens=4200),
               S("implement_2", "Implementation #2 (independent, different model)", _main_cap(und),
                 deps=["analyze"], tokens=4200),
               S("differential", "Differential comparison on shared inputs", "unit_testing",
                 deps=["implement", "implement_2"], verif=DUAL, tokens=1600),
               S("test", "Deterministic tests of agreed solution", "unit_testing", deps=["differential"],
                 verif=TEST, tokens=2000)],
        verification_bundle=DUAL + TEST, team_size=3, redundancy=2)


def template_research(und: TaskUnderstanding) -> Strategy:
    return Strategy(
        id=new_id("strat"), name="D · Research → sources → analysis → critique → synthesis",
        pattern="research_synthesis",
        description="Search, validate sources, analyse, adversarially critique the draft, then synthesise. "
                    "Claims are bound to sources; contradictions are surfaced, not hidden.",
        steps=[S("research", "Gather candidate sources", "research", verif=RESEARCH_V, tokens=5000),
               S("validate_sources", "Validate & cross-check sources", "source_validation",
                 deps=["research"], verif=RESEARCH_V, tokens=2600),
               S("analyze", "Analysis of validated material", "data_processing", deps=["validate_sources"], tokens=3200),
               S("critique", "Adversarial critique: counterarguments, edge cases", "critique",
                 deps=["analyze"], tokens=2200),
               S("synthesis", "Synthesis with sources and counterarguments", "writing",
                 deps=["critique"], verif=[OracleKind.EXTERNAL_SOURCE], tokens=3200)],
        verification_bundle=[OracleKind.EXTERNAL_SOURCE, OracleKind.DOMAIN_RULE], team_size=3)


def template_diagnose_fix(und: TaskUnderstanding) -> Strategy:
    return Strategy(
        id=new_id("strat"), name="E · Reproduce → diagnose → fix → independent tests → mutation",
        pattern="diagnose_fix_verify",
        description="Reproduce the failure, diagnose the root cause, fix, then verify with tests "
                    "written independently of the fix, plus mutation testing of the test suite.",
        steps=[S("analyze", "Reproduce & diagnose the failure", "debugging", tokens=2600),
               S("fix", "Implement the fix", "backend", deps=["analyze"], tokens=3600),
               S("test", "Independent tests (written from the spec, not the fix)", "unit_testing",
                 deps=["fix"], verif=TEST_MUT, tokens=2800),
               S("review", "Independent review of the fix", "code_review", deps=["fix"], tokens=2000),
               S("verify", "Mutation testing of the new suite", "mutation_testing",
                 deps=["test"], verif=[OracleKind.MUTATION], parallelizable=False, tokens=1400)],
        verification_bundle=TEST_MUT + [OracleKind.INDEPENDENT_IMPLEMENTATION]
        if und.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) else TEST_MUT,
        team_size=3)


def template_optimize(und: TaskUnderstanding) -> Strategy:
    return Strategy(
        id=new_id("strat"), name="F · Profile → optimize → benchmark → differential",
        pattern="optimize_measure",
        description="Measure first, optimize second: profiling, then optimization, then benchmark "
                    "comparison against the baseline implementation (kept as reference for differential).",
        steps=[S("profile", "Profile / measure the baseline", "benchmarking", verif=[OracleKind.BENCHMARK], tokens=1400),
               S("optimize", "Optimize based on measurements", "optimization", deps=["profile"], tokens=4200),
               S("benchmark", "Benchmark optimized vs baseline", "benchmarking", deps=["optimize"],
                 verif=[OracleKind.BENCHMARK], tokens=1200),
               S("differential", "Differential correctness check vs baseline", "unit_testing",
                 deps=["optimize"], verif=DUAL, tokens=1600)],
        verification_bundle=[OracleKind.BENCHMARK, OracleKind.DIFFERENTIAL], team_size=2)


def template_minimal(und: TaskUnderstanding) -> Strategy:
    return Strategy(
        id=new_id("strat"), name="M · Minimal single pass", pattern="minimal",
        description="Single executor, no verification beyond the artifact itself. Only acceptable "
                    "for negligible-risk work.",
        steps=[S("execute", "Produce the artifact", _main_cap(und), tokens=3000)],
        verification_bundle=[], team_size=1)


def _main_cap(und: TaskUnderstanding) -> str:
    by_type = {"code_generation": "backend", "debugging": "debugging",
               "code_review": "code_review", "software_architecture": "architecture",
               "research": "research", "data_analysis": "data_processing",
               "optimization": "optimization", "security": "security_analysis",
               "writing": "writing", "experiment": "experiment_execution"}
    if by_type.get(und.task_type.value):
        return by_type[und.task_type.value]
    for rc in sorted(und.required_capabilities, key=lambda r: -r.importance):
        if rc.capability not in ("unit_testing", "critique"):
            return rc.capability
    return "planning"


TEMPLATES = {
    "specialist": template_single_specialist,
    "pipeline": template_pipeline,
    "dual_implementation": template_dual,
    "research_synthesis": template_research,
    "diagnose_fix_verify": template_diagnose_fix,
    "optimize_measure": template_optimize,
    "minimal": template_minimal,
}

# which patterns fit which task types (starting points, not rules)
FIT = {
    "code_generation": {"specialist": 0.9, "pipeline": 0.8, "dual_implementation": 0.7, "minimal": 0.4},
    "debugging": {"diagnose_fix_verify": 1.0, "pipeline": 0.8, "dual_implementation": 0.6, "specialist": 0.4},
    "optimization": {"optimize_measure": 1.0, "pipeline": 0.7, "dual_implementation": 0.6},
    "research": {"research_synthesis": 1.0, "pipeline": 0.4},
    "data_analysis": {"research_synthesis": 0.6, "pipeline": 0.8, "optimize_measure": 0.3},
    "security": {"pipeline": 0.9, "diagnose_fix_verify": 0.8},
    "code_review": {"specialist": 0.9, "pipeline": 0.6},
    "software_architecture": {"pipeline": 0.9, "research_synthesis": 0.6},
    "writing": {"research_synthesis": 0.8, "specialist": 0.6},
    "composite": {"pipeline": 0.8, "dual_implementation": 0.6},
    "unknown": {"pipeline": 0.6, "specialist": 0.5},
}


# ---------------------------------------------------------------- utility (sec. 13)

WEIGHTS = {
    "cheap_fast": {"quality": 0.16, "success": 0.22, "verification": 0.09,
                   "cost": 0.23, "latency": 0.17, "risk": 0.05, "fit": 0.08},
    "balanced": {"quality": 0.18, "success": 0.18, "verification": 0.18,
                 "cost": 0.13, "latency": 0.11, "risk": 0.10, "fit": 0.12},
    "quality_first": {"quality": 0.20, "success": 0.18, "verification": 0.26,
                      "cost": 0.05, "latency": 0.04, "risk": 0.15, "fit": 0.12},
    "safety_critical": {"quality": 0.18, "success": 0.16, "verification": 0.30,
                        "cost": 0.03, "latency": 0.02, "risk": 0.16, "fit": 0.15},
}


def weight_profile(und: TaskUnderstanding) -> tuple[str, dict]:
    if und.risk_level == RiskLevel.CRITICAL:
        return "safety_critical", WEIGHTS["safety_critical"]
    if und.risk_level == RiskLevel.HIGH:
        return "quality_first", WEIGHTS["quality_first"]
    if und.complexity in (Complexity.TRIVIAL, Complexity.SIMPLE) and \
       und.risk_level in (RiskLevel.NEGLIGIBLE, RiskLevel.LOW):
        return "cheap_fast", WEIGHTS["cheap_fast"]
    return "balanced", WEIGHTS["balanced"]


@dataclass
class EvaluatedStrategy:
    strategy: Strategy
    twin: TwinEstimate
    factors: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)
    weight_profile: str = ""

    def to_dict(self):
        from .core import dc_to_dict
        return {"strategy": dc_to_dict(self.strategy), "twin": dc_to_dict(self.twin),
                "factors": self.factors, "weights": self.weights,
                "weight_profile": self.weight_profile}


class StrategyEngine:
    """Strategy search + utility evaluation (sec. 12-13)."""

    def __init__(self, twin: DigitalTwin, memory: StrategyMemory):
        self.twin = twin
        self.memory = memory

    # ---------------------------------------------------------- search

    def generate(self, und: TaskUnderstanding, autonomy: AutonomyLevel,
                 recall: Recall | None) -> list[Strategy]:
        fit = FIT.get(und.task_type.value, FIT["unknown"])
        wanted = {"minimal": 2, "standard": 3, "high": 4, "critical": 4}[autonomy.value]
        ordered = sorted(fit.items(), key=lambda kv: -kv[1])
        patterns = [p for p, _ in ordered[:wanted]]
        # memory-informed candidate: best empirical pattern for this fingerprint
        if recall is not None and recall.pattern_bias:
            best_pattern = max(recall.pattern_bias.items(), key=lambda kv: kv[1])[0]
            if best_pattern not in patterns:
                patterns.append(best_pattern)
        strategies: list[Strategy] = []
        for p in patterns:
            if p in TEMPLATES:
                s = TEMPLATES[p](und)
                prior, n = self.memory.prior_success(p, und)
                if n:
                    s.memory_ref = f"memory: {n} past run(s), success {prior:.0%}"
                    s.rationale = (s.rationale or "") + f" [memory prior: {prior:.0%} over {n} run(s)]"
                strategies.append(s)
        return strategies

    # ---------------------------------------------------------- evaluation

    def evaluate(self, strat: Strategy, und: TaskUnderstanding,
                 recall: Recall | None = None) -> EvaluatedStrategy:
        est = self.twin.simulate(strat, und)
        profile, weights = weight_profile(und)

        fit = FIT.get(und.task_type.value, {}).get(strat.pattern, 0.4)
        prior, n = self.memory.prior_success(strat.pattern, und)
        if recall and strat.pattern in recall.pattern_bias:
            prior_mix = recall.pattern_bias[strat.pattern]
        else:
            prior_mix = prior

        quality = clamp(0.35 + 0.6 * fit + (0.12 * prior_mix if n else 0.0))
        success = clamp(est.est_success_prob * (1 + 0.1 * (prior_mix - 0.5) if n else 1.0))
        verification = est.verification_strength
        cost = clamp(1.0 / (1.0 + est.est_cost_usd))          # cheaper -> higher score
        latency = clamp(1.0 / (1.0 + est.est_seconds / 60.0)) # faster -> higher score
        risk = clamp(1.0 - {"negligible": 0.05, "low": 0.1, "medium": 0.25,
                            "high": 0.45, "critical": 0.6}[und.risk_level.value] *
                     (1.0 - est.verification_strength))        # strong verification reduces risk factor

        factors = {"quality": round(quality, 3), "success": round(success, 3),
                   "verification": round(verification, 3), "cost": round(cost, 3),
                   "latency": round(latency, 3), "risk": round(risk, 3),
                   "fit": round(fit, 2), "memory_prior": round(prior_mix, 3) if n else None}
        utility = sum(weights[k] * factors[k] for k in weights)
        strat.est_cost_usd = est.est_cost_usd
        strat.est_seconds = est.est_seconds
        strat.est_success_prob = est.est_success_prob
        strat.verification_strength = est.verification_strength
        strat.twin_prediction = True
        strat.utility = round(utility, 4)
        strat.scores = factors
        strat.weights = weights
        if not strat.rationale:
            strat.rationale = f"pattern fit {fit:.0%} for {und.task_type.value}"
        return EvaluatedStrategy(strat, est, factors, weights, profile)

    def search(self, und: TaskUnderstanding, autonomy: AutonomyLevel,
               recall: Recall | None = None) -> list[EvaluatedStrategy]:
        cands = self.generate(und, autonomy, recall)
        evaluated = [self.evaluate(s, und, recall) for s in cands]
        evaluated.sort(key=lambda e: -e.strategy.utility)
        return evaluated


# ---------------------------------------------------------------- assumptions (sec. 14)

PROBE_HINTS = [
    (re.compile(r"test|pytest|suite", re.I), "probe:test_run", 'probe:test_run:{"target": "."}'),
    (re.compile(r"workspace|katalog|plik|file|repo|kod", re.I), "probe:fs_list", 'probe:fs_list:{"sub": "."}'),
    (re.compile(r"python", re.I), "probe:python", 'probe:python:{"file": "__fama_probe__.py"}'),
]


class AssumptionEngine:
    """Identify assumptions, grade them, and check them before critical phases."""

    def __init__(self):
        self.assumptions: list[Assumption] = []

    def build(self, und: TaskUnderstanding, interpretation: str = "") -> list[Assumption]:
        self.assumptions = []
        for amb in und.ambiguities:
            self.assumptions.append(Assumption(
                id=new_id("asm"), statement=f"Interpretation resolves ambiguity: {amb}",
                confidence=max(0.35, und.confidence - 0.1), importance=0.8,
                risk=und.risk_level, verification_method="llm_check",
                note=interpretation or und.interpretation))
        for unc in und.uncertainties:
            method = "deferred"
            for rx, kind, call in PROBE_HINTS:
                if rx.search(unc):
                    method = f"probe_check:{kind}"
                    break
            self.assumptions.append(Assumption(
                id=new_id("asm"), statement=unc, confidence=0.45, importance=0.6,
                risk=RiskLevel.LOW, verification_method=method))
        return self.assumptions

    def add(self, statement: str, confidence: float = 0.5, importance: float = 0.6,
            method: str = "deferred", risk: RiskLevel = RiskLevel.LOW) -> Assumption:
        a = Assumption(id=new_id("asm"), statement=statement, confidence=confidence,
                       importance=importance, risk=risk, verification_method=method)
        self.assumptions.append(a)
        return a

    def critical_unverified(self) -> list[Assumption]:
        return [a for a in self.assumptions
                if a.importance >= 0.7 and a.status in (AssumptionStatus.PROPOSED,
                                                        AssumptionStatus.CHECKING)]

    def to_dict(self):
        from .core import dc_to_dict
        return dc_to_dict(self.assumptions)
