"""Strategy Memory & True Learning (sec. 23, 24).

Storing an outcome is STATE STORAGE.  Learning happens only when the stored
outcome changes a future decision and measurably improves it — this module
feeds retrieval priors back into strategy search, and records whether the
memory-informed choice actually won (adaptive learning loop).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .core import (Complexity, RiskLevel, Strategy, TaskType, TaskUnderstanding,
                   clamp, now_utc)


def fingerprint(und: TaskUnderstanding) -> str:
    risk_bucket = {"negligible": "low", "low": "low", "medium": "medium",
                   "high": "high", "critical": "high"}[und.risk_level.value]
    return f"{und.task_type.value}|{und.domain}|{risk_bucket}|{und.complexity.value}"


@dataclass
class MemoryEntry:
    id: str
    fingerprint: str
    strategy_pattern: str
    strategy_name: str
    verification_bundle: list[str] = field(default_factory=list)
    team: list[str] = field(default_factory=list)
    assumptions_status: dict = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    replans: int = 0
    verification_strength: float = 0.0
    mutation_score: float | None = None
    cost_usd: float = 0.0
    seconds: float = 0.0
    result: str = ""
    ts: str = field(default_factory=now_utc)

    @property
    def success(self) -> bool:
        return self.result == "verified"


@dataclass
class Recall:
    entries: list[MemoryEntry]
    n_success: int
    n_total: int
    pattern_bias: dict           # pattern -> empirical success rate
    message: str = ""


class StrategyMemory:
    def __init__(self, store=None):
        self.store = store  # optional Store for persistence
        self.entries: list[MemoryEntry] = []
        self._seq = 0
        if store is not None:
            self._load()

    def _load(self):
        for key, val in self.store.list("stratmem"):
            try:
                self.entries.append(MemoryEntry(**val))
            except Exception:
                continue

    def _persist(self, e: MemoryEntry):
        if self.store is not None:
            self.store.put("stratmem", e.id, e.__dict__)

    def record(self, und: TaskUnderstanding, strat: Strategy, *, result: str,
               verification_strength: float, cost_usd: float, seconds: float,
               failures: list[str] | None = None, replans: int = 0,
               mutation_score: float | None = None, team: list[str] | None = None,
               assumptions_status: dict | None = None) -> MemoryEntry:
        self._seq += 1
        e = MemoryEntry(
            id=f"mem-{fingerprint(und).replace('|', '_')}-{self._seq:04d}",
            fingerprint=fingerprint(und),
            strategy_pattern=strat.pattern,
            strategy_name=strat.name,
            verification_bundle=[getattr(v, "value", str(v)) for v in strat.verification_bundle],
            team=team or [], assumptions_status=assumptions_status or {},
            failures=failures or [], replans=replans,
            verification_strength=round(verification_strength, 3),
            mutation_score=round(mutation_score, 3) if mutation_score is not None else None,
            cost_usd=round(cost_usd, 5), seconds=round(seconds, 1), result=result)
        self.entries.append(e)
        self._persist(e)
        return e

    def recall(self, und: TaskUnderstanding) -> Optional[Recall]:
        exact = [e for e in self.entries if e.fingerprint == fingerprint(und)]
        if not exact:
            # partial match: same task type
            partial = [e for e in self.entries
                       if e.fingerprint.split("|")[0] == und.task_type.value]
            if not partial:
                return None
            return self._recall_from(partial, und, partial=True)
        return self._recall_from(exact, und, partial=False)

    def _recall_from(self, entries: list[MemoryEntry], und: TaskUnderstanding,
                     partial: bool) -> Recall:
        n_succ = sum(1 for e in entries if e.success)
        bias: dict[str, float] = {}
        for e in entries:
            lst = bias.setdefault(e.strategy_pattern, [])
            bias[e.strategy_pattern] = lst + [1 if e.success else 0]
        bias = {p: (sum(v) / len(v), len(v)) for p, v in bias.items()}
        msg = (f"{len(entries)} similar past task(s) [{fingerprint(und)}] — "
               f"historical strategies are a reference point, not truth; they enter "
               f"search as candidates and are re-evaluated")
        if partial:
            msg += " (partial match: same task type only)"
        return Recall(entries=entries[-8:], n_success=n_succ, n_total=len(entries),
                      pattern_bias={p: round(s, 3) for p, (s, _) in bias.items()},
                      message=msg)

    def prior_success(self, pattern: str, und: TaskUnderstanding) -> tuple[float, int]:
        """Empirical prior for a pattern under this fingerprint (n=0 -> (0, 0))."""
        rows = [e for e in self.entries
                if e.fingerprint == fingerprint(und) and e.strategy_pattern == pattern]
        if not rows:
            return 0.0, 0
        return sum(1 for e in rows if e.success) / len(rows), len(rows)
