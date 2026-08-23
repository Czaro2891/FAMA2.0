import ast

import pytest

from fama.core import OracleKind
from fama.governance import Governance
from fama.tools import Sandbox, ToolRouter
from fama.verification import (CommonModeDetector, DifferentialRunner,
                               MetamorphicVerifier, MutationTester,
                               VerificationBudget, generate_mutants)


SOURCE = '''
def apply_discount(cents, pct):
    """pct in 0..100; returns discounted cents (int)."""
    factor = (100 - pct) / 100
    return int(round(cents * factor))
'''


@pytest.fixture()
def tools(tmp_path):
    sbx = Sandbox(str(tmp_path))
    ws = sbx.workspace("ws")
    r = ToolRouter(sbx, ws, Governance())
    r.grant("verify", ["fs_read", "fs_write", "fs_list", "python_run", "test_run",
                       "mutation", "benchmark"])
    yield r
    sbx.cleanup()


def test_mutants_are_single_fault_and_distinct():
    mutants = generate_mutants(SOURCE, max_mutants=20)
    assert mutants, "should generate mutants"
    codes = [c for _, c in mutants]
    assert len(set(codes)) == len(codes)
    # each mutant must still parse
    for c in codes:
        ast.parse(c)


def test_mutation_tester_strong_suite_passes(tools):
    (tools.workspace / "discount.py").write_text(SOURCE)
    (tools.workspace / "test_discount.py").write_text(
        "from discount import apply_discount\n"
        "def test_twenty_percent():\n    assert apply_discount(1000, 20) == 800\n"
        "def test_zero_percent():\n    assert apply_discount(1000, 0) == 1000\n"
        "def test_half():\n    assert apply_discount(123, 50) == 62\n")
    run = MutationTester(tools).run("discount.py", max_mutants=6)
    assert run.verdict == "pass", run.detail
    assert run.measurements["score"] >= 0.7


def test_mutation_tester_weak_suite_is_flagged(tools):
    (tools.workspace / "discount.py").write_text(SOURCE)
    # weak: only pct=0 — arithmetic mutants around the discount survive
    (tools.workspace / "test_discount.py").write_text(
        "from discount import apply_discount\n"
        "def test_zero():\n    assert apply_discount(1000, 0) == 1000\n")
    run = MutationTester(tools).run("discount.py", max_mutants=6)
    assert run.verdict == "weak", f"expected weak, got {run.verdict}: {run.detail}"


def test_metamorphic_involution(tools):
    (tools.workspace / "rev.py").write_text("def rev(xs):\n    return xs[::-1]\n")
    run = MetamorphicVerifier(tools).run("reverse_involution", "rev.py", "rev")
    assert run.verdict == "pass"


def test_metamorphic_fails_on_broken_relation(tools):
    (tools.workspace / "broken.py").write_text(
        "def broken(xs):\n    return sorted(xs)\n")  # sorting is NOT an involution
    run = MetamorphicVerifier(tools).run("reverse_involution", "broken.py", "broken")
    assert run.verdict == "fail"


def test_differential_agreement_and_mismatch(tools):
    (tools.workspace / "a.py").write_text("def f(x):\n    return abs(x)\n")
    (tools.workspace / "b_ok.py").write_text("def f(x):\n    if x < 0:\n        return -x\n    return x\n")
    (tools.workspace / "b_bad.py").write_text("def f(x):\n    return x * x\n")
    ok = DifferentialRunner(tools).run("a.py", "b_ok.py", "f")
    assert ok.verdict == "pass"
    bad = DifferentialRunner(tools).run("a.py", "b_bad.py", "f")
    assert bad.verdict == "fail" and bad.measurements["mismatches"] > 0


def test_verification_budget_escalates_by_risk():
    from fama.core import Complexity, RiskLevel, TaskUnderstanding
    low = VerificationBudget(TaskUnderstanding(risk_level=RiskLevel.LOW))
    high = VerificationBudget(TaskUnderstanding(risk_level=RiskLevel.HIGH))
    critical = VerificationBudget(TaskUnderstanding(risk_level=RiskLevel.CRITICAL))
    assert low.max_oracles < high.max_oracles < critical.max_oracles
    assert high.require_independent and not low.require_independent
    assert critical.require_human
    assert high.escalate("weak") is True
    assert high.max_oracles == 4


def test_common_mode_detects_shared_model_and_duplicate_oracle():
    det = CommonModeDetector()
    team = [
        {"name": "coder", "model": "openai/gpt-4o", "role": "producer"},
        {"name": "reviewer", "model": "openai/gpt-4o", "role": "verifier"},
    ]
    risk = det.analyze(team, ["deterministic_test", "deterministic_test"])
    assert risk.score > 0.3
    assert any("same model" in f or "not independent" in f for f in risk.findings)
    assert any("duplicate" in f or "same oracle" in f for f in risk.findings)


def test_common_mode_clean_team_is_low():
    det = CommonModeDetector()
    team = [
        {"name": "coder", "model": "openai/gpt-4o", "role": "producer"},
        {"name": "skeptic", "model": "anthropic/claude-opus-4.1", "role": "verifier"},
    ]
    risk = det.analyze(team, ["deterministic_test", "mutation"])
    assert risk.score < 0.1
