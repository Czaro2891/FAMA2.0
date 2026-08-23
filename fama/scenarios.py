"""Demo scenarios (spec sec. 45) with deterministic scripted fixtures.

Each scenario proves a different system behaviour:
  simple-function  -> low risk => cheap single-specialist strategy, 1 test oracle
  payments-bug     -> high risk debugging => diagnose/fix/verify, mutation +
                      independent implementation oracles
  tech-compare     -> research => strategy D, sources, critique, synthesis
  optimize-algorithm -> profile/benchmark/differential oracles
  vague-app        -> ambiguity detected => asks instead of assuming
  weak-tests       -> VERIFICATION WEAK => contradiction refutes result =>
                      strategy change (PLAN V2) => verified

The scripted provider is a deterministic TEST DOUBLE used for tests and
recorded replays only.  With real API keys set, the same scenarios run
against real models.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .core import TaskStatus
from .llm import LLMGateway, ScriptedProvider
from .orchestrator import FAMA
from .store import Store


@dataclass
class Scenario:
    name: str
    title: str
    description: str
    task: str
    files: dict[str, str] = field(default_factory=dict)
    fixtures: list[dict] = field(default_factory=list)
    clarify_answers: list[str] | None = None
    allow_assumptions: bool = False


def _fx(pairs: list[tuple[str, dict]]) -> list[dict]:
    return [{"match": m, "text": json.dumps(t, ensure_ascii=False)} for m, t in pairs]


# ================================================================ 1. simple function

SIMPLE = Scenario(
    name="simple-function",
    title="Prosta funkcja Python",
    description="Low risk, simple problem → cheapest sensible strategy (single specialist), "
                "single deterministic-test oracle.",
    task="Napisz prostą funkcję Python moving_average(data, window) w pliku moving_average.py, "
         "liczącą średnią kroczączą. Bez zewnętrznych bibliotek.",
    fixtures=_fx([
        (r"(?s)FAMA:PHASE:UNDERSTANDING.*moving_average", {
            "goal": "Implement moving_average(data, window) computing the rolling average",
            "deliverable": "moving_average.py with the function plus passing tests",
            "constraints": ["pure Python standard library only"],
            "risks": ["wrong edge-case behaviour (window larger than data, window<=0)"],
            "risk_level": "low",
            "complexity": "simple",
            "uncertainties": ["behaviour when window > len(data)"],
            "ambiguities": [],
            "success_criteria": ["function returns correct rolling averages",
                                  "test suite passes deterministically"],
            "domain": "software",
            "task_type": "code_generation",
            "required_capabilities": [
                {"capability": "backend", "min_quality": 0.6, "importance": 0.9, "why": "implementation"},
                {"capability": "unit_testing", "min_quality": 0.5, "importance": 0.7, "why": "prove correctness"}],
            "autonomy": "minimal",
            "verification_requirements": ["deterministic_test"],
            "interpretation": "standard rolling mean; window > len(data) returns empty list",
            "clarifying_questions": [],
            "confidence": 0.9}),
        (r"FAMA:STEP:implement", {
            "files": {"moving_average.py":
                      'def moving_average(data, window):\n'
                      '    """Rolling average over `window` consecutive elements."""\n'
                      '    if window <= 0:\n'
                      '        raise ValueError("window must be positive")\n'
                      '    if window > len(data):\n'
                      '        return []\n'
                      '    return [sum(data[i:i + window]) / window\n'
                      '            for i in range(len(data) - window + 1)]\n'},
            "actions": [],
            "output": "Implemented moving_average in moving_average.py. Edge cases: window<=0 raises "
                      "ValueError, window>len(data) returns [].",
            "artifact": "moving_average.py",
            "confidence": 0.9,
            "assumptions_used": ["rolling mean semantics"]}),
        (r"FAMA:STEP:test", {
            "files": {"test_moving_average.py":
                      'from moving_average import moving_average\n\n'
                      'def test_basic():\n'
                      '    assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]\n\n'
                      'def test_window_equals_len():\n'
                      '    assert moving_average([2, 4], 2) == [3.0]\n\n'
                      'def test_window_too_large():\n'
                      '    assert moving_average([1], 5) == []\n\n'
                      'def test_invalid_window():\n'
                      '    import pytest\n'
                      '    with pytest.raises(ValueError):\n'
                      '        moving_average([1, 2], 0)\n'},
            "actions": [{"tool": "test_run", "kwargs": {"target": "."}}],
            "output": "Wrote 4 deterministic tests covering basic rolling, boundary window, "
                      "window>len and invalid window. Suite passes.",
            "artifact": "test_moving_average.py",
            "confidence": 0.95,
            "assumptions_used": []}),
    ]))

# ================================================================ 2. payments bug

PAYMENTS_BUGGY = '''"""Payments module (contains a settlement bug — order #1042)."""

PRICES = {"WIDGET": 49.99, "CABLE": 19.99, "HUB": 79.93}


def order_total(items, discount=0.0):
    """Total order value in PLN.

    items: list of (sku, qty); discount: fraction 0..1.
    """
    total = 0.0
    for sku, qty in items:
        line = PRICES[sku] * qty
        line = round(line * (1 - discount), 2)   # per-line rounding with floats
        total += line
    return round(total, 2)
'''

PAYMENTS_WEAK_TESTS = '''from payments import order_total


def test_single_item_no_discount():
    assert order_total([("WIDGET", 1)]) == 49.99


def test_order_1042():
    # NOTE: this assertion mirrors the current implementation (per-line rounding)
    assert order_total([("HUB", 2), ("WIDGET", 1), ("CABLE", 1)], discount=0.1) == 206.85
'''

PAYMENTS_FIXED = '''"""Payments module — fixed: money handled in integer cents, discount applied
to the exact total, single final rounding (correct for order #1042)."""

PRICES_CENTS = {"WIDGET": 4999, "CABLE": 1999, "HUB": 7993}


def order_total(items, discount=0.0):
    """Total order value in PLN. items: list of (sku, qty); discount: 0..1."""
    total_cents = sum(PRICES_CENTS[sku] * qty for sku, qty in items)
    total_cents = int(round(total_cents * (1 - discount)))
    return total_cents / 100
'''

PAYMENTS_REFERENCE = '''"""Independent reference implementation (written from the spec, not the fix).

Money semantics: sum all lines in cents, apply discount once, round half up.
"""

_PRICES = {"WIDGET": 4999, "CABLE": 1999, "HUB": 7993}


def order_total(items, discount=0.0):
    from decimal import Decimal, ROUND_HALF_UP
    cents = sum(Decimal(_PRICES[s]) * q for s, q in items)
    cents = (cents * (Decimal(1) - Decimal(str(discount)))).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP)
    return float(cents) / 100
'''

PAYMENTS = Scenario(
    name="payments-bug",
    title="Błąd w systemie płatności",
    description="High-risk debugging: existing tests pass but mirror the bug. FAMA must use "
                "independent tests written from the spec, mutation testing and an independent "
                "reference implementation.",
    task="W module payments.py jest błąd rozliczeniowy: zamówienie #1042 (2×HUB, 1×WIDGET, "
         "1×CABLE, kod rabatowy 10%) ma total 206.85 zł, a powinno wynosić 206.86 zł. Znajdź "
         "przyczynę i popraw. Istniejące testy przechodzą. To system płatności — błąd dotyka "
         "klientów, więc wynik musi być zweryfikowany niezależnie.",
    files={"payments.py": PAYMENTS_BUGGY, "test_payments.py": PAYMENTS_WEAK_TESTS},
    fixtures=_fx([
        (r"(?s)FAMA:PHASE:UNDERSTANDING.*payments", {
            "goal": "Find and fix the settlement bug in payments.py: order #1042 totals 206.85 "
                    "instead of 206.86",
            "deliverable": "fixed payments.py + independent test proving 206.86",
            "constraints": ["existing public API must not change",
                            "existing tests pass today but may mirror the bug"],
            "risks": ["financial error affecting customers",
                      "fixing only the symptom without root cause",
                      "accepting the fix based on the same weak tests"],
            "risk_level": "high",
            "complexity": "moderate",
            "uncertainties": ["whether existing tests mirror the buggy behaviour"],
            "ambiguities": [],
            "success_criteria": ["order #1042 totals exactly 206.86",
                                 "new tests are independent of the implementation",
                                 "test suite passes after the fix"],
            "domain": "software",
            "task_type": "debugging",
            "required_capabilities": [
                {"capability": "debugging", "min_quality": 0.6, "importance": 0.9, "why": "root cause"},
                {"capability": "backend", "min_quality": 0.6, "importance": 0.8, "why": "fix"},
                {"capability": "unit_testing", "min_quality": 0.6, "importance": 0.9,
                 "why": "independent proof"},
                {"capability": "code_review", "min_quality": 0.5, "importance": 0.6, "why": "review fix"}],
            "autonomy": "high",
            "verification_requirements": ["deterministic_test", "mutation", "independent_implementation"],
            "interpretation": "money must be computed in cents; discount applied once to the total",
            "clarifying_questions": [],
            "confidence": 0.85}),
        (r"FAMA:STEP:analyze", {
            "files": {"__probe_1042__.py":
                      'from payments import order_total\n'
                      'got = order_total([("HUB", 2), ("WIDGET", 1), ("CABLE", 1)], discount=0.1)\n'
                      'print("order #1042 total:", got)\n'
                      'assert got == 206.86, f"BUG CONFIRMED: got {got}, expected 206.86"\n'},
            "actions": [{"tool": "python_run", "kwargs": {"file": "__probe_1042__.py"}}],
            "output": "Root cause: prices are floats and the discount is rounded PER LINE "
                      "(round(line*(1-d),2)), so cents drift per line (143.87+44.99+17.99=206.85). "
                      "Correct semantics: sum all lines in integer cents, apply discount once, "
                      "round once → 206.86. Existing test mirrors the bug (asserts 206.85), so it "
                      "CANNOT serve as the oracle — independent tests are required.",
            "artifact": "__probe_1042__.py",
            "confidence": 0.92,
            "assumptions_used": ["discount applies to the summed total"]}),
        (r"FAMA:STEP:fix", {
            "files": {"payments.py": PAYMENTS_FIXED,
                      "payments_reference.py": PAYMENTS_REFERENCE},
            "actions": [{"tool": "test_run", "kwargs": {"target": "."}}],
            "output": "Fixed payments.py: money in integer cents, discount applied once to the "
                      "total, single final rounding. Also wrote payments_reference.py — an "
                      "independent Decimal-based implementation from the spec for differential "
                      "verification. NOTE: old test_order_1042 now fails as expected (it asserted "
                      "the buggy 206.85) — the tester step replaces it with spec-based tests.",
            "artifact": "payments.py",
            "confidence": 0.9,
            "assumptions_used": []}),
        (r"FAMA:STEP:test", {
            "files": {"test_payments.py":
                      'from payments import order_total\n'
                      'from payments_reference import order_total as ref_total\n\n'
                      'CENTS = {"WIDGET": 4999, "CABLE": 1999, "HUB": 7993}\n\n'
                      'def _expected(items, discount):\n'
                      '    total = sum(CENTS[s] * q for s, q in items)\n'
                      '    return round(total * (1 - discount)) / 100\n\n'
                      'def test_order_1042_regression():\n'
                      '    items = [("HUB", 2), ("WIDGET", 1), ("CABLE", 1)]\n'
                      '    assert order_total(items, discount=0.1) == 206.86\n'
                      '    assert order_total(items, discount=0.1) == _expected(items, 0.1)\n\n'
                      'def test_no_discount():\n'
                      '    items = [("WIDGET", 1), ("CABLE", 2)]\n'
                      '    assert order_total(items) == _expected(items, 0.0)\n\n'
                      'def test_half_discount_single_line():\n'
                      '    assert order_total([("HUB", 1)], discount=0.5) == _expected([("HUB", 1)], 0.5)\n\n'
                      'def test_matches_reference_implementation():\n'
                      '    items = [("HUB", 2), ("WIDGET", 1), ("CABLE", 1)]\n'
                      '    assert order_total(items, discount=0.1) == ref_total(items, discount=0.1)\n'},
            "actions": [{"tool": "test_run", "kwargs": {"target": "."}}],
            "output": "Replaced the mirroring test with spec-based tests: expected values computed "
                      "independently in cents, plus a direct comparison against the reference "
                      "implementation. All tests pass.",
            "artifact": "test_payments.py",
            "confidence": 0.95,
            "assumptions_used": []}),
        (r"FAMA:STEP:review", {
            "files": {},
            "actions": [],
            "output": "Review: API unchanged; cents math correct; single rounding point; no float "
                      "accumulation; reference implementation uses Decimal independently.",
            "artifact": "",
            "confidence": 0.9,
            "assumptions_used": []}),
        (r"FAMA:STEP:security", {
            "files": {},
            "actions": [],
            "output": "Security: no secrets, no eval, no injection surface; discount bounded by caller.",
            "artifact": "",
            "confidence": 0.85,
            "assumptions_used": []}),
    ]))

# ================================================================ 3. tech compare

RESEARCH = Scenario(
    name="tech-compare",
    title="Porównanie technologii",
    description="Research problem → strategy D: research → source validation → analysis → "
                "adversarial critique → synthesis. Claims bound to sources.",
    task="Porównaj PostgreSQL, MongoDB i Redis pod kątem przechowywania historii transakcji "
         "finansowych (append-heavy, wymagany audyt, zapytania analityczne) i wybierz najlepszy. "
         "Uzasadnij źródłami.",
    fixtures=_fx([
        (r"(?s)FAMA:PHASE:UNDERSTANDING.*PostgreSQL", {
            "goal": "Compare PostgreSQL vs MongoDB vs Redis for financial transaction history "
                    "(append-heavy, audit, analytics) and recommend one",
            "deliverable": "comparison with sources, counterarguments and a justified recommendation",
            "constraints": ["claims must cite sources"],
            "risks": ["recommendation based on marketing material instead of documentation"],
            "risk_level": "medium",
            "complexity": "moderate",
            "uncertainties": ["write throughput numbers under specific hardware"],
            "ambiguities": [],
            "success_criteria": ["at least 2 independent sources per key claim",
                                 "counterarguments present",
                                 "clear recommendation with rationale"],
            "domain": "research",
            "task_type": "research",
            "required_capabilities": [
                {"capability": "research", "min_quality": 0.6, "importance": 0.9, "why": "gather material"},
                {"capability": "source_validation", "min_quality": 0.6, "importance": 0.9,
                 "why": "avoid single-source bias"},
                {"capability": "critique", "min_quality": 0.5, "importance": 0.7, "why": "counterarguments"},
                {"capability": "writing", "min_quality": 0.5, "importance": 0.6, "why": "synthesis"}],
            "autonomy": "standard",
            "verification_requirements": ["external_source"],
            "interpretation": "",
            "clarifying_questions": [],
            "confidence": 0.85}),
        (r"FAMA:STEP:research", {
            "files": {},
            "actions": [{"tool": "web_search", "kwargs": {"query":
                          "PostgreSQL append-only audit table write throughput vs MongoDB"}}],
            "output": "Primary sources located:\n"
                      "1. PostgreSQL WAL & durability: https://www.postgresql.org/docs/current/wal-intro.html\n"
                      "2. MongoDB WiredTiger journaling: https://www.mongodb.com/docs/manual/core/journaling/\n"
                      "3. Redis persistence model: https://redis.io/docs/latest/operate/oss_and_stack/management/persistence/\n"
                      "4. PostgreSQL OLTP/OLAP positioning: https://www.postgresql.org/docs/current/intro-whatis.html\n"
                      "Marketing pages discarded; only vendor docs and benchmarks kept.",
            "artifact": "",
            "confidence": 0.8,
            "assumptions_used": []}),
        (r"FAMA:STEP:validate_sources", {
            "files": {},
            "actions": [],
            "output": "Source validation: 4 sources span 3 vendors (no single-vendor bias); each key "
                      "property (durability model, query capability) is covered by at least 2 "
                      "sources. URLs reachable at fetch time: "
                      "https://www.postgresql.org/docs/current/wal-intro.html ; "
                      "https://www.mongodb.com/docs/manual/core/journaling/ ; "
                      "https://redis.io/docs/latest/operate/oss_and_stack/management/persistence/",
            "artifact": "",
            "confidence": 0.8,
            "assumptions_used": []}),
        (r"FAMA:STEP:analyze", {
            "files": {},
            "actions": [],
            "output": "Analysis (per sources): PostgreSQL — ACID, WAL append-friendly, rich SQL "
                      "analytics, row versioning useful for audit (source: postgresql.org/docs). "
                      "MongoDB — flexible schema, WiredTiger journal (source: mongodb.com/docs), "
                      "aggregation pipeline covers moderate analytics; multi-document ACID needs "
                      "replica set. Redis — in-memory with RDB/AOF persistence (source: "
                      "redis.io/docs); fastest but RAM-bound: transaction history outgrows memory.",
            "artifact": "",
            "confidence": 0.8,
            "assumptions_used": []}),
        (r"FAMA:STEP:critique", {
            "files": {},
            "actions": [],
            "output": "Counterarguments: (1) MongoDB could win if schema volatility dominates — "
                      "but transaction history is schema-stable, weakening that case. (2) Redis "
                      "with AOF everysec is durable enough for some audit regimes — but RAM cost "
                      "at multi-year history scale is prohibitive. (3) PostgreSQL analytics is "
                      "slower than column stores — acknowledged, out of compared set.",
            "artifact": "",
            "confidence": 0.75,
            "assumptions_used": []}),
        (r"FAMA:STEP:synthesis", {
            "files": {"recommendation.md":
                      "# Recommendation: PostgreSQL\n\n"
                      "For append-heavy financial transaction history with audit and analytics:\n\n"
                      "- **PostgreSQL**: ACID + WAL (https://www.postgresql.org/docs/current/wal-intro.html), "
                      "SQL analytics, audit-friendly. Recommended.\n"
                      "- **MongoDB**: viable (https://www.mongodb.com/docs/manual/core/journaling/) "
                      "but adds operational complexity without an advantage for stable schemas.\n"
                      "- **Redis**: excellent speed (https://redis.io/docs/latest/operate/oss_and_stack/"
                      "management/persistence/) but RAM-bound — unsuitable as system of record.\n\n"
                      "Counterarguments considered and rejected in the critique step.\n"},
            "actions": [],
            "output": "Synthesized recommendation.md: PostgreSQL, with per-source justification and "
                      "rejected counterarguments.",
            "artifact": "recommendation.md",
            "confidence": 0.85,
            "assumptions_used": []}),
    ]))

# ================================================================ 4. optimize algorithm

DEDUPE_SLOW = '''def dedupe(items):
    """Remove duplicates keeping the first occurrence (order-preserving)."""
    seen = []
    for x in items:
        if x not in seen:
            seen.append(x)
    return seen


def bench_args():
    import random
    random.seed(7)
    return [random.randint(0, 500) for _ in range(4000)],
'''

DEDUPE_FAST = '''def dedupe(items):
    """Remove duplicates keeping the first occurrence (order-preserving). O(n)."""
    return list(dict.fromkeys(items))


def bench_args():
    import random
    random.seed(7)
    return [random.randint(0, 500) for _ in range(4000)],
'''

OPTIMIZE = Scenario(
    name="optimize-algorithm",
    title="Optymalizacja algorytmu",
    description="Optimization → measure first: profiling, optimization, benchmark comparison vs "
                "baseline kept as reference, differential correctness check.",
    task="Funkcja dedupe w dedupe.py ma złożoność O(n²) (lista + 'in') i jest zbyt wolna na "
         "dużych listach. Zoptymalizuj ją, zachowując dokładnie tę samą semantykę: kolejność i "
         "pierwsze wystąpienia.",
    files={"dedupe.py": DEDUPE_SLOW},
    fixtures=_fx([
        (r"(?s)FAMA:PHASE:UNDERSTANDING.*dedupe", {
            "goal": "Optimize dedupe() from O(n^2) to O(n) preserving exact semantics",
            "deliverable": "optimized dedupe.py + benchmark evidence of speedup + correctness proof",
            "constraints": ["semantics identical: order-preserving, first occurrence"],
            "risks": ["optimization changes semantics on edge cases (unhashable items)"],
            "risk_level": "medium",
            "complexity": "moderate",
            "uncertainties": ["whether inputs may contain unhashable items"],
            "ambiguities": [],
            "success_criteria": ["benchmark shows speedup",
                                 "differential vs baseline shows identical outputs"],
            "domain": "software",
            "task_type": "optimization",
            "required_capabilities": [
                {"capability": "optimization", "min_quality": 0.6, "importance": 0.9, "why": "core work"},
                {"capability": "benchmarking", "min_quality": 0.7, "importance": 0.9,
                 "why": "measure, don't guess"},
                {"capability": "unit_testing", "min_quality": 0.5, "importance": 0.7,
                 "why": "semantics guard"}],
            "autonomy": "standard",
            "verification_requirements": ["benchmark", "differential"],
            "interpretation": "inputs are hashable scalars (ints/strs) per current usage",
            "clarifying_questions": [],
            "confidence": 0.85}),
        (r"FAMA:STEP:profile", {
            "files": {},
            "actions": [{"tool": "benchmark", "kwargs": {"file": "dedupe.py", "function": "dedupe"}}],
            "output": "Baseline measured on 4000 random ints: see benchmark JSON. Complexity is "
                      "O(n²) because 'x not in seen' scans a list. Optimization target: O(n) via "
                      "dict/set membership.",
            "artifact": "dedupe.py",
            "confidence": 0.9,
            "assumptions_used": ["items are hashable"]}),
        (r"FAMA:STEP:optimize", {
            "files": {"dedupe.py": DEDUPE_FAST,
                      "dedupe_reference.py": DEDUPE_SLOW},
            "actions": [],
            "output": "Optimized dedupe with dict.fromkeys (insertion-ordered, O(n)). Baseline "
                      "kept as dedupe_reference.py for differential verification. bench_args "
                      "unchanged for fair comparison.",
            "artifact": "dedupe.py",
            "confidence": 0.9,
            "assumptions_used": []}),
        (r"FAMA:STEP:benchmark", {
            "files": {},
            "actions": [{"tool": "benchmark", "kwargs": {"file": "dedupe.py", "function": "dedupe"}}],
            "output": "Benchmark of optimized version executed (see measurement). Compare with the "
                      "baseline profile run from the earlier step — speedup is one order of "
                      "magnitude on the 4000-element workload.",
            "artifact": "",
            "confidence": 0.85,
            "assumptions_used": []}),
        (r"FAMA:STEP:differential", {
            "files": {"__agent_diff__.py":
                      "import importlib.util\n"
                      "def load(p, n):\n"
                      "    spec = importlib.util.spec_from_file_location(n, p)\n"
                      "    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
                      "    return m\n"
                      "a = load('dedupe.py', 'a').dedupe\n"
                      "b = load('dedupe_reference.py', 'b').dedupe\n"
                      "import random\n"
                      "random.seed(11)\n"
                      "bad = 0\n"
                      "for n in (0, 1, 10, 500):\n"
                      "    data = [random.randint(-9, 9) for _ in range(n)]\n"
                      "    if a(list(data)) != b(list(data)):\n"
                      "        bad += 1\n"
                      "print('SEMANTIC_MISMATCHES', bad)\n"},
            "actions": [{"tool": "python_run", "kwargs": {"file": "__agent_diff__.py"}}],
            "output": "Differential semantics check: optimized vs baseline identical on random "
                      "workloads incl. empty and duplicate-heavy inputs.",
            "artifact": "",
            "confidence": 0.9,
            "assumptions_used": []}),
    ]))

# ================================================================ 5. vague app

UTILS = '''def slugify(text):
    return "-".join(text.lower().split())


def parse_kv(s, sep="="):
    pairs = {}
    for line in s.splitlines():
        k, v = line.split(sep, 1)
        pairs[k.strip()] = v.strip()
    return pairs
'''

VAGUE = Scenario(
    name="vague-app",
    title="„Zrób coś z tą aplikacją, żeby była lepsza”",
    description="Ambiguous request → FAMA detects ambiguity, refuses to fake certainty, asks "
                "the user; only after the answer does it plan.",
    task="Zrób coś z tą aplikacją, żeby była lepsza.",
    files={"utils.py": UTILS},
    clarify_answers=[
        "Chodzi o jakość i niezawodność: dodaj testy jednostkowe do utils.py i popraw obsługę "
        "błędnych danych wejściowych (parse_kv nie może się wywalać na złych liniach).",
        "Tylko utils.py, reszta aplikacji mnie nie interesuje.",
    ],
    fixtures=_fx([
        (r"(?s)FAMA:PHASE:UNDERSTANDING.*User clarifications", {
            "goal": "Add unit tests for utils.py and make parse_kv robust to malformed lines",
            "deliverable": "hardened utils.py + passing test suite",
            "constraints": ["only utils.py and its tests may change"],
            "risks": ["changing slugify behaviour accidentally"],
            "risk_level": "low",
            "complexity": "simple",
            "uncertainties": [],
            "ambiguities": [],
            "success_criteria": ["parse_kv skips malformed lines instead of raising",
                                 "tests cover slugify and parse_kv"],
            "domain": "software",
            "task_type": "code_generation",
            "required_capabilities": [
                {"capability": "backend", "min_quality": 0.6, "importance": 0.8, "why": "hardening"},
                {"capability": "unit_testing", "min_quality": 0.5, "importance": 0.8, "why": "tests"}],
            "autonomy": "minimal",
            "verification_requirements": ["deterministic_test"],
            "interpretation": "robustness = skip malformed lines; slugify unchanged",
            "clarifying_questions": [],
            "confidence": 0.88}),
        (r"(?s)FAMA:PHASE:UNDERSTANDING.*Zrób coś", {
            "goal": "make the application 'better' — UNDEFINED",
            "deliverable": "unknown",
            "constraints": [],
            "risks": ["doing the wrong kind of improvement"],
            "risk_level": "medium",
            "complexity": "simple",
            "uncertainties": ["what 'better' means: performance, quality, tests, UX?",
                              "which part of the application matters"],
            "ambiguities": ["'lepsza' has more than one interpretation",
                            "scope: whole application or a specific module?"],
            "success_criteria": [],
            "domain": "software",
            "task_type": "unknown",
            "required_capabilities": [],
            "autonomy": "standard",
            "verification_requirements": [],
            "interpretation": "",
            "clarifying_questions": [
                "Co dokładnie znaczy „lepsza” — wydajność, jakość kodu, testy, UX?",
                "Który fragment aplikacji najbardziej Cię boli?"],
            "confidence": 0.2}),
        (r"FAMA:STEP:implement", {
            "files": {"utils.py":
                      'def slugify(text):\n'
                      '    return "-".join(text.lower().split())\n\n\n'
                      'def parse_kv(s, sep="="):\n'
                      '    """Parse key=value lines; malformed lines are skipped."""\n'
                      '    pairs = {}\n'
                      '    for line in s.splitlines():\n'
                      '        if sep not in line:\n'
                      '            continue\n'
                      '        k, v = line.split(sep, 1)\n'
                      '        pairs[k.strip()] = v.strip()\n'
                      '    return pairs\n',
                      "test_utils.py":
                      'from utils import slugify, parse_kv\n\n'
                      'def test_slugify():\n'
                      '    assert slugify("Ala Ma Kota") == "ala-ma-kota"\n\n'
                      'def test_parse_kv_basic():\n'
                      '    assert parse_kv("a=1\\nb = 2") == {"a": "1", "b": "2"}\n\n'
                      'def test_parse_kv_skips_malformed():\n'
                      '    assert parse_kv("good=1\\nBROKEN\\nalso=2") == {"good": "1", "also": "2"}\n\n'
                      'def test_parse_kv_empty():\n'
                      '    assert parse_kv("") == {}\n'},
            "actions": [{"tool": "test_run", "kwargs": {"target": "."}}],
            "output": "Hardened parse_kv (skips malformed lines, no crash) + 4 unit tests. "
                      "slugify unchanged.",
            "artifact": "utils.py",
            "confidence": 0.9,
            "assumptions_used": ["slugify must stay unchanged"]}),
        (r"FAMA:STEP:test", {
            "files": {},
            "actions": [{"tool": "test_run", "kwargs": {"target": "."}}],
            "output": "Suite executed: all tests pass, including malformed-input regression tests.",
            "artifact": "test_utils.py",
            "confidence": 0.9,
            "assumptions_used": []}),
    ]))

# ================================================================ 6. weak tests (adaptive star)

PALINDROME_FLAWED = '''def is_palindrome(s):
    """Palindrome check ignoring case and spaces (per spec: also punctuation)."""
    if s is None:
        return False
    t = s.lower()
    t = t.replace(" ", "")
    if len(t) == 0:
        return True
    if len(t) > 1000:
        return False
    return t == t[::-1]
'''

PALINDROME_WEAK_TESTS = '''from is_palindrome import is_palindrome


def test_simple_true():
    assert is_palindrome("kajak")

def test_simple_false():
    assert not is_palindrome("kot")
'''

PALINDROME_FIXED = '''import re


def is_palindrome(s):
    """Palindrome check ignoring case, spaces and punctuation."""
    if s is None:
        return False
    t = re.sub(r"[^a-z0-9]", "", s.lower())
    return t == t[::-1]
'''

PALINDROME_STRONG_TESTS = '''from is_palindrome import is_palindrome


def test_simple_true():
    assert is_palindrome("kajak")

def test_simple_false():
    assert not is_palindrome("kot")

def test_none():
    assert is_palindrome(None) is False

def test_empty():
    assert is_palindrome("") is True

def test_ignores_case():
    assert is_palindrome("KaJak")

def test_ignores_spaces():
    assert is_palindrome("never odd or even")

def test_ignores_punctuation():
    assert is_palindrome("A man, a plan, a canal: Panama")
    assert not is_palindrome("abc, def!")
'''

WEAK = Scenario(
    name="weak-tests",
    title="Słaba weryfikacja → kontradykcja → zmiana strategii",
    description="A sloppy specialist passes its own weak tests. Mutation testing reports "
                "VERIFICATION WEAK, the Contradiction Engine refutes the claim, FAMA changes "
                "strategy (PLAN V2: pipeline) and the corrected result is verified strongly.",
    task="Napisz funkcję is_palindrome(s) w pliku is_palindrome.py: sprawdza palindromy "
         "ignorując wielkość liter, spacje oraz znaki interpunkcyjne.",
    fixtures=_fx([
        (r"(?s)FAMA:PHASE:UNDERSTANDING.*is_palindrome", {
            "goal": "Implement is_palindrome(s): palindrome check ignoring case, spaces and punctuation",
            "deliverable": "is_palindrome.py + tests",
            "constraints": ["ignore case, spaces, punctuation"],
            "risks": ["punctuation handling forgotten — most common mistake for this task"],
            "risk_level": "medium",
            "complexity": "simple",
            "uncertainties": [],
            "ambiguities": [],
            "success_criteria": ["handles case, spaces and punctuation",
                                 "test suite proves all three"],
            "domain": "software",
            "task_type": "code_generation",
            "required_capabilities": [
                {"capability": "backend", "min_quality": 0.6, "importance": 0.9, "why": "implementation"},
                {"capability": "unit_testing", "min_quality": 0.6, "importance": 0.9,
                 "why": "prove all three behaviours"}],
            "autonomy": "standard",
            "verification_requirements": ["deterministic_test", "mutation"],
            "interpretation": "",
            "clarifying_questions": [],
            "confidence": 0.9}),
        # ---------------- PLAN V2 (pipeline) fixtures — must be matched BEFORE V1
        (r"(?s)FAMA:STEP:implement_2.*PLAN V2", {
            "files": {"is_palindrome.py": PALINDROME_FIXED,
                      "is_palindrome_reference.py": PALINDROME_FIXED.replace(
                          "def is_palindrome", "def is_palindrome"),
                      "test_is_palindrome.py": PALINDROME_STRONG_TESTS},
            "actions": [{"tool": "test_run", "kwargs": {"target": "."}}],
            "output": "Independent implementation #2 (same spec, defensive style) written to "
                      "is_palindrome_reference.py for differential verification.",
            "artifact": "is_palindrome_reference.py",
            "confidence": 0.9,
            "assumptions_used": []}),
        (r"(?s)FAMA:STEP:differential", {
            "files": {"__agent_diff__.py":
                      "import importlib.util\n"
                      "def load(p, n):\n"
                      "    spec = importlib.util.spec_from_file_location(n, p)\n"
                      "    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
                      "    return m\n"
                      "a = load('is_palindrome.py', 'a').is_palindrome\n"
                      "b = load('is_palindrome_reference.py', 'b').is_palindrome\n"
                      "cases = ['kajak', 'A man, a plan, a canal: Panama', 'abc, def!', '', 'KaJak']\n"
                      "bad = [c for c in cases if a(c) != b(c)]\n"
                      "print('MISMATCHES', len(bad))\n"},
            "actions": [{"tool": "python_run", "kwargs": {"file": "__agent_diff__.py"}}],
            "output": "Differential check: both implementations agree on all shared inputs.",
            "artifact": "",
            "confidence": 0.9,
            "assumptions_used": []}),
        (r"(?s)FAMA:STEP:analyze.*PLAN V2", {
            "files": {},
            "actions": [],
            "output": "Post-mortem of V1: implementation handled case+spaces but NOT punctuation; "
                      "tests covered only trivial cases so mutation score was weak and adversarial "
                      "counter-tests refuted the claim. Requirements for V2: strip all "
                      "non-alphanumerics, test None/empty/case/spaces/punctuation explicitly.",
            "artifact": "",
            "confidence": 0.9,
            "assumptions_used": []}),
        (r"(?s)FAMA:STEP:implement.*PLAN V2", {
            "files": {"is_palindrome.py": PALINDROME_FIXED,
                      "test_is_palindrome.py": PALINDROME_STRONG_TESTS},
            "actions": [{"tool": "test_run", "kwargs": {"target": "."}}],
            "output": "Corrected implementation: regex strips everything except [a-z0-9] after "
                      "lowercasing — case, spaces AND punctuation handled. Test suite covers all "
                      "spec behaviours incl. the previously fatal punctuation cases.",
            "artifact": "is_palindrome.py",
            "confidence": 0.92,
            "assumptions_used": ["spec's 'ignoring punctuation' means strip before comparing"]}),
        (r"(?s)FAMA:STEP:review.*PLAN V2", {
            "files": {},
            "actions": [],
            "output": "Review: implementation matches spec; no behavioural gaps; tests assert both "
                      "positive and negative punctuation cases.",
            "artifact": "",
            "confidence": 0.88,
            "assumptions_used": []}),
        (r"(?s)FAMA:STEP:security.*PLAN V2", {
            "files": {},
            "actions": [],
            "output": "Security: regex safe, no ReDoS risk for this pattern, no inputs unsafe.",
            "artifact": "",
            "confidence": 0.8,
            "assumptions_used": []}),
        # ---------------- PLAN V1 (specialist) fixtures
        (r"(?s)FAMA:STEP:implement.*PLAN V1", {
            "files": {"is_palindrome.py": PALINDROME_FLAWED,
                      "test_is_palindrome.py": PALINDROME_WEAK_TESTS},
            "actions": [],
            "output": "Implemented is_palindrome (case-insensitive, spaces removed). Wrote tests.",
            "artifact": "is_palindrome.py",
            "confidence": 0.75,
            "assumptions_used": ["punctuation probably not critical"]}),
        (r"FAMA:STEP:test", {
            "files": {},
            "actions": [{"tool": "test_run", "kwargs": {"target": "."}}],
            "output": "Tests pass: 'kajak' True, 'kot' False.",
            "artifact": "test_is_palindrome.py",
            "confidence": 0.7,
            "assumptions_used": []}),
    ]))

# contradiction fixtures used when verification escalates (weak-tests scenario)
CONTRADICTION_PALINDROME = {
    "counter_tests": [
        {"name": "punctuation is ignored per spec",
         "code": "assert fn('A man, a plan, a canal: Panama') is True"},
        {"name": "spaces inside words",
         "code": "assert fn('never odd or even') is True"},
        {"name": "non-palindrome with punctuation",
         "code": "assert fn('abc, def!') is False"},
    ],
    "expected_refutation": "implementation likely keeps punctuation, violating the spec",
    "confidence_claim_is_wrong": 0.8,
}

# ================================================================ registry

SCENARIOS: dict[str, Scenario] = {
    s.name: s for s in [SIMPLE, PAYMENTS, RESEARCH, OPTIMIZE, VAGUE, WEAK]
}


def scenario_fixtures(sc: Scenario) -> list[dict]:
    """Fixtures incl. phase-generic ones (autopsy, contradiction)."""
    fx = list(sc.fixtures)
    fx.append({"match": r"(?s)FAMA:PHASE:CONTRADICTION.*is_palindrome",
               "text": json.dumps(CONTRADICTION_PALINDROME)})
    fx.append({"match": r"FAMA:PHASE:CONTRADICTION",
               "text": json.dumps({
                   "counter_tests": [
                       {"name": "empty input edge case",
                        "code": "assert fn([]) == []"},
                       {"name": "single element",
                        "code": "assert fn([7]) == [7]"}],
                   "expected_refutation": "edge-case handling of empty collections",
                   "confidence_claim_is_wrong": 0.3})})
    fx.append({"match": r"failure autopsy",
               "text": json.dumps({
                   "root_cause": "step context was insufficient for a confident artifact",
                   "wrong_agent": False, "wrong_model": False, "wrong_tool": False,
                   "wrong_plan": True, "bad_assumption": True,
                   "verification_insufficient": False,
                   "lesson": "add an analysis step before execution when inputs are unclear",
                   "confidence": 0.6})})
    return fx


def scripted_gateway(sc: Scenario) -> LLMGateway:
    return LLMGateway(scripted=ScriptedProvider(scenario_fixtures(sc)),
                      allow_scripted=True)


async def run_scenario(sc: Scenario, fama: FAMA | None = None,
                       base_dir: str = ".fama") -> tuple[FAMA, str]:
    """Run a scenario to completion (deterministic, offline). Returns (fama, task_id)."""
    f = fama or FAMA(scripted_gateway(sc), Store(":memory:"), base_dir=base_dir)
    task, st = f.create_task(sc.task, workspace_files=sc.files)
    runner = asyncio.create_task(f.run(st, allow_assumptions=sc.allow_assumptions))
    answers = list(sc.clarify_answers) if sc.clarify_answers else None
    while not runner.done():
        if answers and st.task.status == TaskStatus.AWAITING_CLARIFICATION:
            f.answer_clarification(task.id, answers)
            answers = None
        await asyncio.sleep(0.05)
    await runner
    return f, task.id


def record_replay(sc: Scenario, out_dir: str | Path = ".fama/replays") -> dict:
    """Run offline and dump the full event stream + final state as a replay file."""
    import asyncio as _a

    async def _go():
        f = FAMA(scripted_gateway(sc), Store(":memory:"), base_dir=".fama/record")
        task, st = f.create_task(sc.task, workspace_files=sc.files)
        runner = _a.create_task(f.run(st, allow_assumptions=sc.allow_assumptions))
        answers = list(sc.clarify_answers) if sc.clarify_answers else None
        while not runner.done():
            if answers and st.task.status == TaskStatus.AWAITING_CLARIFICATION:
                f.answer_clarification(task.id, answers)
                answers = None
            await _a.sleep(0.05)
        await runner
        return f, task.id

    f, task_id = _a.run(_go())
    replay = {
        "scenario": sc.name,
        "title": sc.title,
        "description": sc.description,
        "recorded_at": f"{__import__('datetime').datetime.now().isoformat()}",
        "task_id": task_id,
        "events": [e.to_dict() for e in f.bus.history(task_id)],
        "final_state": f.state_dict(task_id),
        "metrics": f.metrics.snapshot(f.gateway, f.registry),
        "scripted": True,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{sc.name}.json").write_text(json.dumps(replay, ensure_ascii=False, indent=1))
    return replay
