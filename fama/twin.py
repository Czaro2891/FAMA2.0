"""Digital Twin — what-if strategy simulation (sec. 34).

Simulations are PREDICTIONS.  A prediction is never written as a real
result; every twin output is labelled twin_prediction.
"""
from __future__ import annotations

from dataclasses import dataclass

from .agents import AgentRegistry
from .core import Complexity, RiskLevel, Strategy, TaskUnderstanding, clamp
from .llm import LLMGateway

# rough token demand per step-kind (planning estimates)
STEP_TOKENS = {
    "analyze": 1800, "implement": 4200, "implement_2": 4200, "review": 2200,
    "test": 2600, "security": 1800, "verify": 2200, "differential": 1200,
    "research": 5000, "validate_sources": 2600, "synthesis": 3200, "critique": 2200,
    "diagnose": 2600, "fix": 3600, "profile": 1400, "benchmark": 1200,
    "optimize": 4200, "clarify": 400, "execute": 2600,
}
CLASS_PRICE = {"cheap": (0.3, 1.2), "fast": (0.5, 2.0), "reasoning": (2.0, 9.0),
               "coding": (2.5, 11.0), "adversarial": (4.0, 16.0), "local": (0.0, 0.0),
               "vision": (2.0, 8.0)}
CLASS_LATENCY = {"cheap": 4, "fast": 3, "reasoning": 14, "coding": 10,
                 "adversarial": 16, "local": 6, "vision": 8}

COMPLEXITY_PENALTY = {Complexity.TRIVIAL: 0.03, Complexity.SIMPLE: 0.06,
                      Complexity.MODERATE: 0.12, Complexity.COMPLEX: 0.22,
                      Complexity.WICKED: 0.35}

ORACLE_STRENGTH = {
    "deterministic_test": 0.60, "independent_implementation": 0.92,
    "differential": 0.85, "property_based": 0.72, "metamorphic": 0.66,
    "mutation": 0.80, "benchmark": 0.60, "external_source": 0.70,
    "domain_rule": 0.50, "human": 0.95,
}


@dataclass
class TwinEstimate:
    strategy_id: str
    est_cost_usd: float
    est_seconds: float
    est_success_prob: float
    verification_strength: float
    twin_prediction: bool = True   # always True — predictions are never results
    detail: dict = None

    def __post_init__(self):
        if self.detail is None:
            self.detail = {}


class DigitalTwin:
    def __init__(self, registry: AgentRegistry, gateway: LLMGateway):
        self.registry = registry
        self.gateway = gateway

    def simulate(self, strat: Strategy, und: TaskUnderstanding) -> TwinEstimate:
        models = {m.id: m for m in self.gateway.available_models()} or {}
        step_costs, step_times = [], []
        p_success = 1.0
        for st in strat.steps:
            toks = STEP_TOKENS.get(st.name, st.estimated_tokens)
            # pick the cheapest preferred class actually available
            pin, pout, lat = 2.0, 9.0, 10.0
            for cls_name in ("cheap", "fast", "coding", "reasoning"):
                m = next((m for m in models.values()
                          if any(c.value == cls_name for c in m.classes)), None)
                if m is not None:
                    pin, pout, lat = m.price_in, m.price_out, CLASS_LATENCY[cls_name]
                    break
            step_costs.append((toks / 1e6) * (pin + pout * 0.45))
            step_times.append(lat + 1.0)
            # per-step success: capability quality + retries exist, so base is forgiving
            quality = 0.6
            for a in self.registry.list():
                if a.capability_quality(st.capability) > quality:
                    quality = a.capability_quality(st.capability)
            p_step = clamp(0.75 + 0.22 * quality -
                           0.25 * COMPLEXITY_PENALTY.get(und.complexity, 0.12))
            p_success *= p_step
        # redundancy (dual implementations) rescues the final result if either works
        if strat.redundancy > 1:
            p_fail_single = clamp(1.0 - p_success ** (1.0 / max(1, len(strat.steps))))
            p_success = clamp(1.0 - (p_fail_single ** strat.redundancy))
        ver_strength = self.bundle_strength(strat.verification_bundle)
        # parallelism: rough critical path = half of serial time when parallelizable
        par = sum(1 for s in strat.steps if s.parallelizable) / max(1, len(strat.steps))
        est_seconds = sum(step_times) * (1.0 - 0.4 * par)
        return TwinEstimate(
            strategy_id=strat.id,
            est_cost_usd=round(sum(step_costs) * (1.15 if strat.redundancy > 1 else 1.0), 4),
            est_seconds=round(est_seconds, 1),
            est_success_prob=round(clamp(p_success), 3),
            verification_strength=ver_strength,
            detail={"step_costs": [round(c, 5) for c in step_costs],
                    "parallel_fraction": round(par, 2)})

    @staticmethod
    def bundle_strength(bundle: list) -> float:
        if not bundle:
            return 0.0
        vals = [ORACLE_STRENGTH.get(getattr(k, "value", str(k)), 0.5) for k in bundle]
        # diminishing returns: strongest oracle dominates, extras add a bit
        vals.sort(reverse=True)
        out = vals[0]
        for v in vals[1:]:
            out += v * (1.0 - out) * 0.55
        return round(min(1.0, out), 3)
