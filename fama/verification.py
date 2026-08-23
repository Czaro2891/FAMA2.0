"""Verification System (sec. 25-31).

"Agent said it works" is not evidence.  Each result type gets an
appropriate source of truth: deterministic tests, independent
implementations, property/metamorphic checks, mutation testing,
differential comparison, benchmarks, external sources, domain rules,
human approval.

Re-running the same test is NOT the same as independently establishing
truth — the Oracle Engine tracks the strength and independence of every
oracle it runs.
"""
from __future__ import annotations

import ast
import copy
import json
import random
import re
from dataclasses import dataclass, field

from .core import OracleKind, OracleRun, RiskLevel, TaskUnderstanding, clamp, new_id
from .tools import Sandbox, ToolRouter

ORACLE_STRENGTH = {
    OracleKind.DETERMINISTIC_TEST: 0.60,
    OracleKind.INDEPENDENT_IMPLEMENTATION: 0.92,
    OracleKind.DIFFERENTIAL: 0.85,
    OracleKind.PROPERTY_BASED: 0.72,
    OracleKind.METAMORPHIC: 0.66,
    OracleKind.MUTATION: 0.80,
    OracleKind.BENCHMARK: 0.60,
    OracleKind.EXTERNAL_SOURCE: 0.70,
    OracleKind.DOMAIN_RULE: 0.50,
    OracleKind.HUMAN: 0.95,
}


class VerificationBudget:
    """Not every task needs the same level of control (sec. 26)."""

    def __init__(self, und: TaskUnderstanding):
        self.risk = und.risk_level
        base = {RiskLevel.NEGLIGIBLE: 1, RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2,
                RiskLevel.HIGH: 3, RiskLevel.CRITICAL: 4}[self.risk]
        self.max_oracles = base
        self.require_independent = self.risk in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        self.require_human = self.risk == RiskLevel.CRITICAL
        self.escalation_level = 0

    def escalate(self, reason: str) -> bool:
        """Raise the control level (e.g. after VERIFICATION WEAK)."""
        if self.escalation_level >= 2:
            return False
        self.escalation_level += 1
        self.max_oracles += 1
        self.require_independent = True
        return True

    def to_dict(self):
        return {"risk": self.risk.value, "max_oracles": self.max_oracles,
                "require_independent": self.require_independent,
                "require_human": self.require_human,
                "escalation_level": self.escalation_level}


# ---------------------------------------------------------------- mutation testing (sec. 29)

class _Mutator(ast.NodeTransformer):
    """Mutate only the target_index-th mutation point (one controlled fault)."""

    def __init__(self, target_index: int):
        self.target = target_index
        self.count = -1

    def _hit(self) -> bool:
        self.count += 1
        return self.count == self.target

    def visit_Compare(self, node):
        self.generic_visit(node)
        swaps = {ast.Lt: ast.LtE, ast.Gt: ast.GtE, ast.LtE: ast.Lt,
                 ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
                 ast.Is: ast.IsNot, ast.IsNot: ast.Is}
        if len(node.ops) == 1 and self._hit():
            new_op = swaps.get(type(node.ops[0]))
            if new_op:
                node.ops = [new_op()]
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        swaps = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div}
        if self._hit() and swaps.get(type(node.op)):
            node.op = swaps[type(node.op)]()
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if self._hit() and isinstance(node.op, (ast.And, ast.Or)):
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if self._hit() and isinstance(node.op, ast.Not):
            return node.operand
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, int) and not isinstance(node.value, bool) and self._hit():
            node.value = node.value + 1
        return node


class _MutationPointCounter(ast.NodeVisitor):
    def __init__(self):
        self.n = 0

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) == 1:
            self.n += 1

    def visit_BinOp(self, node):
        self.generic_visit(node)
        self.n += 1

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        self.n += 1

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            self.n += 1

    def visit_Constant(self, node):
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            self.n += 1


def generate_mutants(source: str, max_mutants: int = 10) -> list[tuple[int, str]]:
    """Controlled faults injected into the code (sec. 29): one fault per mutant."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    counter = _MutationPointCounter()
    counter.visit(tree)
    if counter.n == 0:
        return []
    mutants: list[tuple[int, str]] = []
    seen: set[str] = set()
    try:
        original = ast.unparse(tree)
    except Exception:
        return []
    for target in range(min(max_mutants, counter.n)):
        try:
            mutated = _Mutator(target).visit(copy.deepcopy(tree))
            ast.fix_missing_locations(mutated)
            code = ast.unparse(mutated)
            if code != original and code not in seen:
                seen.add(code)
                mutants.append((target, code))
        except Exception:
            continue
    return mutants


class MutationTester:
    def __init__(self, tools: ToolRouter, step_id: str = "verify"):
        self.tools = tools
        self.step_id = step_id

    def run(self, target_file: str, max_mutants: int = 10,
            weak_threshold: float = 0.7) -> OracleRun:
        ws = self.tools.workspace
        src = (ws / target_file)
        if not src.exists():
            return OracleRun(new_id("orun"), OracleKind.MUTATION, target_file,
                             "inconclusive", 0.0, f"target {target_file} not found")
        original = src.read_text()
        mutants = generate_mutants(original, max_mutants)
        if not mutants:
            return OracleRun(new_id("orun"), OracleKind.MUTATION, target_file,
                             "inconclusive", 0.2, "no mutants could be generated")
        killed = 0
        survivors: list[int] = []
        for mid, code in mutants:
            src.write_text(code)
            res = self.tools.call(self.step_id, "test_run", target=".")
            if not res.ok:
                killed += 1
            else:
                survivors.append(mid)
        src.write_text(original)   # restore
        # sanity: original must still pass, else suite is broken
        sanity = self.tools.call(self.step_id, "test_run", target=".")
        score = killed / len(mutants) if mutants else 0.0
        verdict = "pass"
        detail = (f"mutation score {score:.0%} ({killed}/{len(mutants)} killed); "
                  f"survivors: {survivors[:8]}")
        if not sanity.ok:
            verdict, detail = "fail", "restored original failed tests — suite/env broken"
        elif score < weak_threshold:
            verdict = "weak"   # not a failure of the code, but of the VERIFICATION
            detail += " → VERIFICATION WEAK"
        return OracleRun(new_id("orun"), OracleKind.MUTATION, target_file, verdict,
                         ORACLE_STRENGTH[OracleKind.MUTATION] * (0.5 + 0.5 * score),
                         detail, {"score": round(score, 3), "killed": killed,
                                  "total": len(mutants), "survivors": survivors[:8]})


# ---------------------------------------------------------------- metamorphic (sec. 30)

METAMORPHIC_RELATIONS = {
    "sort_idempotent": {
        "description": "sorting already-sorted input returns the same result",
        "build": """
def relation(fn):
    import random
    for n in (0, 1, 7, 40):
        data = [random.randint(-99, 99) for _ in range(n)]
        once = fn(list(data)); twice = fn(list(once))
        assert once == twice, f"not idempotent for n={n}: {once} vs {twice}"
    return True
"""},
    "reverse_involution": {
        "description": "reversing twice restores the input",
        "build": """
def relation(fn):
    import random
    for n in (0, 1, 7, 40):
        data = [random.randint(-99, 99) for _ in range(n)]
        assert fn(fn(list(data))) == data, f"not an involution for n={n}"
    return True
"""},
    "serialize_roundtrip": {
        "description": "parse(serialize(x)) preserves information",
        "build": """
def relation(fn):
    import random, string
    for n in (0, 3, 20):
        x = ''.join(random.choice(string.printable[:62]) for _ in range(n))
        assert fn(fn(x)) == x, f"roundtrip failed for n={n}"
    return True
"""},
}


class MetamorphicVerifier:
    def __init__(self, tools: ToolRouter, step_id: str = "verify"):
        self.tools = tools
        self.step_id = step_id

    def run(self, relation: str, module_file: str, function: str) -> OracleRun:
        rel = METAMORPHIC_RELATIONS.get(relation)
        if not rel:
            return OracleRun(new_id("orun"), OracleKind.METAMORPHIC, f"{module_file}:{function}",
                             "inconclusive", 0.0, f"unknown relation {relation}")
        script = (
            "import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('mm', {str(module_file)!r})\n"
            "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
            f"fn = getattr(mod, {function!r})\n"
            + rel["build"] +
            "\nrelation(fn)\n"
            "print('METAMORPHIC_OK')\n")
        (self.tools.workspace / "__fama_mm__.py").write_text(script)
        res = self.tools.call(self.step_id, "python_run", file="__fama_mm__.py")
        ok = res.ok and "METAMORPHIC_OK" in (res.stdout or "")
        return OracleRun(new_id("orun"), OracleKind.METAMORPHIC,
                         f"{module_file}:{function} [{relation}]",
                         "pass" if ok else "fail",
                         ORACLE_STRENGTH[OracleKind.METAMORPHIC],
                         f"relation: {rel['description']}",
                         {"exit": res.exit_code, "stderr_tail": (res.stderr or "")[-300:]})


# ---------------------------------------------------------------- differential (sec. 12C / 25)

class DifferentialRunner:
    """Compare two independent implementations on shared inputs."""

    def __init__(self, tools: ToolRouter, step_id: str = "verify"):
        self.tools = tools
        self.step_id = step_id

    def run(self, file_a: str, file_b: str, function: str,
            inputs_builder: str = "list(range(50)) + [-1, 0, 99]") -> OracleRun:
        script = (
            "import importlib.util\n"
            f"def load(p, n):\n"
            f"    spec = importlib.util.spec_from_file_location(n, p)\n"
            "    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m\n"
            f"a = load({str(file_a)!r}, 'impl_a').{function}\n"
            f"b = load({str(file_b)!r}, 'impl_b').{function}\n"
            f"cases = {inputs_builder}\n"
            "mism = []\n"
            "for c in cases:\n"
            "    try: ra = repr(a(c))\n"
            "    except Exception as e: ra = f'ERR:{type(e).__name__}'\n"
            "    try: rb = repr(b(c))\n"
            "    except Exception as e: rb = f'ERR:{type(e).__name__}'\n"
            "    if ra != rb: mism.append((c, ra[:80], rb[:80]))\n"
            "print('DIFF_DONE', len(mism))\n"
            "for m in mism[:10]: print('MISMATCH', m)\n")
        (self.tools.workspace / "__fama_diff__.py").write_text(script)
        res = self.tools.call(self.step_id, "python_run", file="__fama_diff__.py")
        out = res.stdout or ""
        m = re.search(r"DIFF_DONE (\d+)", out)
        if not m:
            return OracleRun(new_id("orun"), OracleKind.DIFFERENTIAL,
                             f"{file_a} vs {file_b}", "inconclusive", 0.1,
                             "differential harness failed", {"stderr": (res.stderr or "")[-300:]})
        n_mism = int(m.group(1))
        n_cases = len(eval(inputs_builder))  # planning constant, not model output
        if n_mism == 0:
            verdict, detail = "pass", f"both implementations agree on all {n_cases} inputs"
        else:
            verdict = "fail"
            detail = (f"{n_mism}/{n_cases} mismatches: " +
                      "; ".join(out.strip().splitlines()[1:4]))
        return OracleRun(new_id("orun"), OracleKind.DIFFERENTIAL,
                         f"{file_a} vs {file_b}", verdict,
                         ORACLE_STRENGTH[OracleKind.DIFFERENTIAL], detail,
                         {"mismatches": n_mism, "cases": n_cases})


# ---------------------------------------------------------------- contradiction (sec. 28)

class ContradictionEngine:
    """Actively try to refute our own conclusions.

    Deterministic part: edge-case generators.  Model part: an adversarial
    prompt executed by a DIFFERENT model class than the producer (else it
    would risk common-mode confirmation).
    """

    EDGE_CASES = [
        "empty input", "single element", "all-equal elements", "negative numbers",
        "zeros", "very large values", "duplicates", "reverse-sorted input",
        "None where not expected", "unicode strings",
    ]

    def __init__(self, tools: ToolRouter, gateway, step_id: str = "verify"):
        self.tools = tools
        self.gateway = gateway
        self.step_id = step_id

    async def generate_counter_tests(self, claim: str, artifact_summary: str,
                                     exclude_models: list[str]) -> str:
        from .llm import LLMMessage
        from .llm import LLMRequest, ModelClass
        prompt = (
            "FAMA:PHASE:CONTRADICTION\n"
            f"Claim under test: {claim}\n\n"
            f"Artifact summary:\n{artifact_summary}\n\n"
            "Try to REFUTE the claim. Produce a single JSON object:\n"
            '{"counter_tests": [{"name": "...", "code": "python test code asserting the claim fails"}], '
            '"expected_refutation": "which claim aspect is most likely wrong", '
            '"confidence_claim_is_wrong": 0.0-1.0}\n'
            "Target the weakest edges: boundaries, invariants, edge cases. No prose outside JSON.")
        resp = await self.gateway.complete(LLMRequest(
            messages=[LLMMessage("system",
                                 "You are the Contradiction Engine of FAMA 2.0. Your only job is to "
                                 "break the claim. Be concrete and adversarial. Output JSON only."),
                      LLMMessage("user", prompt)],
            model_class=ModelClass.ADVERSARIAL, exclude_models=exclude_models,
            max_tokens=1400, temperature=0.3, json_mode=True, purpose="contradiction"))
        return resp.text

    async def run(self, claim: str, artifact_summary: str, runner_file: str,
                  function_name: str, exclude_models: list[str]) -> OracleRun:
        from .llm import ModelError, extract_json
        try:
            raw = await self.generate_counter_tests(claim, artifact_summary, exclude_models)
            data = extract_json(raw)
        except (ModelError, ValueError) as e:
            return OracleRun(new_id("orun"), OracleKind.PROPERTY_BASED, claim,
                             "inconclusive", 0.1, f"counter-test generation failed: {e}")
        tests = data.get("counter_tests", [])[:6]
        if not tests:
            return OracleRun(new_id("orun"), OracleKind.PROPERTY_BASED, claim,
                             "inconclusive", 0.2, "no counter-tests produced")
        harness = [
            "import importlib.util, json",
            f"spec = importlib.util.spec_from_file_location('t', {str(runner_file)!r})",
            "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)",
            f"fn = getattr(mod, {function_name!r}, None)",
            "outcomes = {}",
        ]
        for i, t in enumerate(tests):
            body = "\n".join("    " + ln for ln in str(t.get("code", "")).splitlines())
            harness += [
                f"def _t{i}(fn):", body or "    pass",
                "try:",
                f"    _t{i}(fn); outcomes[{i!r}] = 'pass'",
                "except AssertionError:",
                f"    outcomes[{i!r}] = 'refuted'",
                "except Exception as e:",
                f"    outcomes[{i!r}] = 'error:' + type(e).__name__",
            ]
        harness += ["print(json.dumps(outcomes))"]
        (self.tools.workspace / "__fama_contra__.py").write_text("\n".join(harness))
        res = self.tools.call(self.step_id, "python_run", file="__fama_contra__.py")
        outcomes = {}
        try:
            m = re.search(r"\{.*\}", res.stdout or "", re.DOTALL)
            if m:
                outcomes = json.loads(m.group(0))
        except Exception:
            pass
        refuted = [k for k, v in outcomes.items() if v == "refuted"]
        errors = [k for k, v in outcomes.items() if str(v).startswith("error")]
        if refuted:
            verdict, strength = "refuted", ORACLE_STRENGTH[OracleKind.PROPERTY_BASED]
            detail = (f"{len(refuted)}/{len(outcomes)} counter-tests REFUTED the claim "
                      f"(expected refutation: {data.get('expected_refutation', '?')})")
        elif outcomes and not errors:
            verdict, strength = "pass", ORACLE_STRENGTH[OracleKind.PROPERTY_BASED] * 0.9
            detail = f"claim survived {len(outcomes)} adversarial counter-tests"
        else:
            verdict, strength = "inconclusive", 0.2
            detail = f"counter-tests errored ({len(errors)}); cannot treat as confirmation"
        return OracleRun(new_id("orun"), OracleKind.PROPERTY_BASED, claim, verdict,
                         round(strength, 3), detail,
                         {"outcomes": outcomes, "expected_refutation":
                          data.get("expected_refutation", "")})


# ---------------------------------------------------------------- common-mode (sec. 31, 40)

@dataclass
class CommonModeRisk:
    score: float          # 0..1
    findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self):
        from .core import dc_to_dict
        return dc_to_dict(self)


class CommonModeDetector:
    def analyze(self, team: list[dict], oracles: list[str],
                assumptions: list[dict] | None = None) -> CommonModeRisk:
        findings: list[str] = []
        recs: list[str] = []
        score = 0.0
        models = [m.get("model") or "" for m in team]
        nonempty = [m for m in models if m]
        if nonempty and len(set(nonempty)) < len(nonempty):
            n_dup = len(nonempty) - len(set(nonempty))
            score += 0.25 * n_dup
            findings.append(f"{n_dup} team member pair(s) share the same model")
            recs.append("route verifier/critic to a different model class than the producer")
        producers = [m for m in team if m.get("role") == "producer"]
        verifiers = [m for m in team if m.get("role") in ("verifier", "critic")]
        for p in producers:
            for v in verifiers:
                if p.get("model") and p.get("model") == v.get("model"):
                    score += 0.35
                    findings.append(f"verifier '{v.get('name')}' uses the same model as "
                                    f"producer '{p.get('name')}' — verification is not independent")
        if len(set(oracles)) < len(oracles):
            score += 0.15
            findings.append("the same oracle kind is counted twice — independent truth not increased")
            recs.append("add an oracle of a different kind instead of repeating one")
        if assumptions:
            unchecked = [a for a in assumptions if a.get("status") in ("proposed", "deferred")]
            if len(unchecked) >= 2:
                score += 0.1 * len(unchecked)
                findings.append(f"{len(unchecked)} assumptions remain unchecked — shared blind spot")
                recs.append("check critical assumptions before trusting the result")
        return CommonModeRisk(round(clamp(score), 3), findings, recs)
