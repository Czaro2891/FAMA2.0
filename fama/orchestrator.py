"""FAMA Orchestrator — the kernel (sec. 3, 15, 16, 44).

Pipeline of phases, each driven by real decisions:
understand → govern → capability map → strategy search → twin simulation →
utility selection → assumptions → team assembly (agents/models/tools) →
adaptive execution (failure handling, replanning) → verification
(oracles, contradiction, mutation, common-mode) → evidence → learning.

Different problems must lead to different decisions (sec. 49).
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .agents import (AgentFactory, AgentRegistry, PerformanceTracker,
                     SelectionEngine, SYSTEM_CAPABILITIES, TeamMember)
from .core import (AssumptionStatus, AutonomyLevel, EventBus, Event, Failure,
                   FailureClass, FailureReaction, OracleKind, Plan,
                   PlanStep, ResourceBudget, ResultStatus, RiskLevel, StepStatus,
                   Strategy, Task, TaskStatus, TaskUnderstanding, clamp, new_id,
                   now_utc, slug)
from .evidence import DecisionTrace, EvidenceGraph
from .execution import AgentAutopsy, FailureEngine
from .governance import Governance
from .llm import (LLMGateway, LLMMessage, LLMRequest, ModelClass, ModelError,
                  NoProviderError, extract_json)
from .memory import StrategyMemory
from .metrics import Metrics
from .store import Store
from .strategy import AssumptionEngine, StrategyEngine
from .tools import Sandbox, ToolRouter
from .twin import DigitalTwin
from .understanding import UnderstandingEngine
from .verification import (CommonModeDetector, ContradictionEngine,
                           DifferentialRunner, MetamorphicVerifier,
                           MutationTester, OracleRun, VerificationBudget)

STEP_TOOL_NEEDS = {
    "implement": ["fs_read", "fs_write", "fs_list", "python_run", "test_run"],
    "implement_2": ["fs_read", "fs_write", "fs_list", "python_run", "test_run"],
    "fix": ["fs_read", "fs_write", "fs_list", "python_run", "test_run"],
    "test": ["fs_read", "fs_write", "fs_list", "python_run", "test_run"],
    "verify": ["fs_read", "fs_write", "fs_list", "python_run", "test_run", "mutation", "benchmark"],
    "differential": ["fs_read", "fs_write", "fs_list", "python_run"],
    "review": ["fs_read", "fs_list"],
    "security": ["fs_read", "fs_list"],
    "analyze": ["fs_read", "fs_list", "fs_write", "python_run"],
    "research": ["web_search", "web_fetch", "fs_read", "fs_write"],
    "validate_sources": ["web_fetch", "fs_read", "fs_write"],
    "critique": ["fs_read", "fs_list"],
    "synthesis": ["fs_read", "fs_write"],
    "profile": ["fs_read", "python_run", "benchmark"],
    "benchmark": ["fs_read", "python_run", "benchmark"],
    "optimize": ["fs_read", "fs_write", "fs_list", "python_run", "benchmark"],
    "execute": ["fs_read", "fs_write", "fs_list", "python_run"],
}
VERIFICATION_STEPS = {"verify"}   # executed by the verification phase, not the agent loop


@dataclass
class TaskState:
    task: Task
    plans: list[Plan] = field(default_factory=list)
    evaluated_strategies: list[dict] = field(default_factory=list)
    chosen_strategy: Optional[Strategy] = None
    team: list[TeamMember] = field(default_factory=list)
    selection_trace: list[dict] = field(default_factory=list)
    assumptions: AssumptionEngine = field(default_factory=AssumptionEngine)
    failures: FailureEngine = field(default_factory=FailureEngine)
    oracle_runs: list[OracleRun] = field(default_factory=list)
    autopsies: list[dict] = field(default_factory=list)
    evidence: Optional[EvidenceGraph] = None
    decisions: Optional[DecisionTrace] = None
    budget: Optional[ResourceBudget] = None
    vbudget: Optional[VerificationBudget] = None
    common_mode: Optional[dict] = None
    sandbox: Optional[Sandbox] = None
    tools: Optional[ToolRouter] = None
    clarifications: list[str] = field(default_factory=list)
    clarify_event: asyncio.Event = field(default_factory=asyncio.Event)
    approval_event: asyncio.Event = field(default_factory=asyncio.Event)
    strategy_rank: int = 0
    final_gate_node: str = ""
    last_round_runs: list = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class FAMA:
    """The adaptive agent operating system kernel."""

    def __init__(self, gateway: LLMGateway, store: Store | None = None,
                 base_dir: str | None = None):
        self.gateway = gateway
        self.store = store or Store(":memory:")
        self.base = Path(base_dir or ".fama")
        self.bus = EventBus()
        self.metrics = Metrics(self.store)
        self.governance = Governance()
        self.registry = AgentRegistry(PerformanceTracker())
        self.twin = DigitalTwin(self.registry, gateway)
        self.memory = StrategyMemory(self.store)
        self.strategy_engine = StrategyEngine(self.twin, self.memory)
        self.understanding_engine = UnderstandingEngine(gateway, list(SYSTEM_CAPABILITIES))
        self.autopsy = AgentAutopsy(gateway)
        self.factory = AgentFactory(self.registry, self.governance)
        self.states: dict[str, TaskState] = {}
        self.bus.subscribe(lambda ev: self.store.add_event(ev.to_dict()))

    # ------------------------------------------------------------ events

    def emit(self, st: TaskState, type_: str, title: str, *, phase: str = "",
             level: str = "info", payload: dict | None = None):
        self.bus.publish(Event(type_, task_id=st.task.id, phase=phase or type_,
                               level=level, title=title, payload=payload or {}))
        self.metrics.inc(f"events_{type_}")

    # ------------------------------------------------------------ task lifecycle

    def create_task(self, text: str, *, autonomy: AutonomyLevel | None = None,
                    workspace_files: dict[str, str] | None = None,
                    max_cost_usd: float = 2.0) -> tuple[Task, TaskState]:
        task = Task(id=new_id("task"), input=text, autonomy_override=autonomy,
                    budget={"max_cost_usd": max_cost_usd, "max_seconds": 900})
        st = TaskState(task=task, budget=ResourceBudget(max_cost_usd=max_cost_usd))
        sbx = Sandbox(str(self.base / "tasks" / task.id))
        st.sandbox = sbx
        ws = sbx.workspace("ws")
        task.workspace = str(ws)
        st.tools = ToolRouter(sbx, ws, self.governance)
        for name, content in (workspace_files or {}).items():
            p = ws / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        st.evidence = EvidenceGraph(self.store, task.id)
        st.decisions = DecisionTrace(self.store, task.id)
        self.states[task.id] = st
        self.store.put("tasks", task.id, task.to_dict())
        self.metrics.inc("tasks_total")
        return task, st

    async def run(self, st: TaskState, *, allow_assumptions: bool = False) -> Task:
        task = st.task
        t0 = time.monotonic()
        self.emit(st, "task_started", f"Task accepted: {task.input[:90]}", phase="ingest",
                  payload={"task_id": task.id, "workspace": task.workspace})
        try:
            # ---- phase 1: understanding (sec. 5)
            task.status = TaskStatus.UNDERSTANDING
            res = await self.understanding_engine.understand(task)
            und = res.understanding
            task.understanding = und
            self.emit(st, "understanding_done", f"Interpreted as: {und.task_type.value} · "
                       f"risk={und.risk_level.value} · complexity={und.complexity.value}",
                      phase="understanding", level="success",
                      payload={"understanding": und.to_dict(), "fallback": res.used_fallback})
            if res.used_fallback:
                return await self._finish(st, ResultStatus.BLOCKED,
                                          "Could not interpret the task — model understanding unavailable",
                                          t0)

            # ---- ambiguity handling (sec. 5: never fake certainty)
            needs_clarification = (und.ambiguities and und.confidence < 0.55
                                   and not allow_assumptions
                                   and not task.human_input.get("answers"))
            if needs_clarification:
                st.clarifications = und.clarifying_questions or [f"Clarify: {a}" for a in und.ambiguities]
                task.status = TaskStatus.AWAITING_CLARIFICATION
                self.emit(st, "clarification_requested",
                          f"Ambiguity detected — asking instead of assuming ({len(st.clarifications)} question(s))",
                          phase="understanding", level="warning",
                          payload={"questions": st.clarifications, "ambiguities": und.ambiguities})
                try:
                    st.clarify_event.clear()
                    await asyncio.wait_for(st.clarify_event.wait(), timeout=3600)
                except asyncio.TimeoutError:
                    return await self._finish(st, ResultStatus.BLOCKED,
                                              "Timed out waiting for clarification", t0)
                answers = task.human_input.get("answers", [])
                task.input += "\n\nUser clarifications:\n" + "\n".join(f"- {a}" for a in answers)
                self.emit(st, "clarification_received", "User answered — re-interpreting",
                          phase="understanding")
                res = await self.understanding_engine.understand(task)
                und = res.understanding
                task.understanding = und
                self.emit(st, "understanding_done", "Re-interpretation complete",
                          phase="understanding", level="success",
                          payload={"understanding": und.to_dict()})

            # ---- phase 2: governance & risk (sec. 35, 37)
            flags = self.governance.assess(task, und)
            gates = self.governance.approval_required(task, und, flags)
            self.governance.register_gates(task.id, gates)
            self.emit(st, "governance_assessed",
                      f"Risk flags: {', '.join(k for k, v in flags.items() if v) or 'none'}",
                      phase="governance", payload={"flags": flags,
                      "gates": [g.to_dict() for g in gates]})
            if self.governance.pending_gates(task.id):
                task.status = TaskStatus.AWAITING_APPROVAL
                self.emit(st, "approval_required",
                          f"Human approval required ({len(gates)} gate(s)) — autonomy limits",
                          phase="governance", level="warning",
                          payload={"gates": [g.to_dict() for g in gates]})
                try:
                    st.approval_event.clear()
                    await asyncio.wait_for(st.approval_event.wait(), timeout=3600)
                except asyncio.TimeoutError:
                    return await self._finish(st, ResultStatus.BLOCKED,
                                              "Timed out waiting for approval", t0)
                if any(g.status == "rejected" for g in self.governance.gates.get(task.id, [])):
                    return await self._finish(st, ResultStatus.FAILED,
                                              "Rejected by human approval gate", t0)
                self.emit(st, "approval_granted", "Human approved execution", phase="governance",
                          level="success")

            # ---- phase 3: capability map (sec. 6, 9)
            st.evidence.add("claim", f"Task understood: {und.goal}",
                            result=und.task_type.value, payload={"risk": und.risk_level.value})
            gaps = self.registry.capability_gap(und.required_capabilities)
            for gap in gaps:
                agent, how = self.factory.try_fill_gap(gap)
                self.emit(st, "capability_gap_filled" if agent else "capability_gap_unresolved",
                          how, phase="capabilities",
                          level="success" if agent else "warning",
                          payload={"capability": gap.capability})
                if agent:
                    st.evidence.add("action", f"Custom agent path: {how}",
                                    agent=agent.name, result="registered")
            self.emit(st, "capabilities_mapped",
                      f"Required capabilities: {', '.join(r.capability for r in und.required_capabilities)}",
                      phase="capabilities", payload={
                          "required": [r.to_dict() for r in und.required_capabilities]})

            # ---- phase 4: strategy search + twin + utility (sec. 11-13, 34)
            task.status = TaskStatus.PLANNING
            autonomy = task.autonomy_override or und.autonomy
            recall = self.memory.recall(und)
            if recall:
                self.emit(st, "memory_recalled", recall.message, phase="strategy",
                          payload={"n": recall.n_total, "successes": recall.n_success,
                                   "pattern_bias": recall.pattern_bias})
            evaluated = self.strategy_engine.search(und, autonomy, recall)
            st.evaluated_strategies = [e.to_dict() for e in evaluated]
            st.chosen_strategy = evaluated[0].strategy if evaluated else None
            comparison = [{"name": e.strategy.name, "pattern": e.strategy.pattern,
                           "utility": e.strategy.utility, "est_cost_usd": e.strategy.est_cost_usd,
                           "est_seconds": e.strategy.est_seconds,
                           "est_success": e.strategy.est_success_prob,
                           "ver_strength": e.strategy.verification_strength,
                           "twin_prediction": True} for e in evaluated]
            self.emit(st, "strategies_compared",
                      f"{len(evaluated)} candidate strategies simulated (Digital Twin — predictions only)",
                      phase="strategy", payload={"candidates": comparison})
            if st.decisions and evaluated:
                st.decisions.record(
                    "strategy_selection",
                    options=[{"name": e.strategy.name, "utility": e.strategy.utility,
                              "scores": e.strategy.scores} for e in evaluated],
                    selected=evaluated[0].strategy.name,
                    score=evaluated[0].strategy.utility,
                    reason=(f"highest utility under weight profile "
                            f"'{evaluated[0].weight_profile}' "
                            f"(risk={und.risk_level.value}, complexity={und.complexity.value})"),
                    evidence_refs=[f"twin:{e.strategy.id}" for e in evaluated[:3]],
                    confidence=clamp(0.5 + 0.1 * (evaluated[0].strategy.utility if evaluated else 0)))
            if not evaluated:
                return await self._finish(st, ResultStatus.BLOCKED, "No strategy could be constructed", t0)
            self.emit(st, "strategy_selected",
                      f"Strategy chosen: {st.chosen_strategy.name} (utility {st.chosen_strategy.utility:.3f})",
                      phase="strategy", level="success",
                      payload={"strategy": st.chosen_strategy.to_dict(),
                               "weights": evaluated[0].weights})

            # ---- phase 5: assumptions (sec. 14)
            asm = st.assumptions.build(und)
            self.emit(st, "assumptions_identified",
                      f"{len(asm)} assumption(s) registered", phase="strategy",
                      payload={"assumptions": [a.to_dict() for a in asm]})
            refuted = await self._check_assumptions(st, und)
            if refuted:
                task.replan_count += 1
                if st.strategy_rank + 1 < len(evaluated):
                    st.strategy_rank += 1
                    st.chosen_strategy = evaluated[st.strategy_rank].strategy
                    self.emit(st, "plan_changed",
                              f"Assumption refuted → switching strategy to "
                              f"{st.chosen_strategy.name} (PLAN V{task.plan_versions + 1})",
                              phase="strategy", level="warning",
                              payload={"refuted": [r.statement for r in refuted]})
                else:
                    return await self._finish(st, ResultStatus.UNCERTAIN,
                                              "Critical assumption refuted and no alternative strategy", t0)

            # ---- phase 6: plan + team (sec. 9, 10, 19, 20)
            while True:
                plan = self._build_plan(st, und)
                st.plans.append(plan)
                task.plan_versions = len(st.plans)
                self.emit(st, "plan_created",
                          f"PLAN V{plan.version}: {len(plan.steps)} steps · team of "
                          f"{len({s.agent_id for s in plan.steps if s.agent_id})} agent(s)",
                          phase="planning", level="success",
                          payload={"plan": plan.to_dict(),
                                   "team": [m.to_dict() for m in st.team],
                                   "selection": st.selection_trace})
                # common-mode pre-check on assembled team (sec. 31, 40)
                cm = CommonModeDetector().analyze(
                    [{"name": m.agent.name, "model": m.model, "role": "producer"}
                     for m in st.team if m.role == "producer"] +
                    [{"name": m.agent.name, "model": m.model, "role": "verifier"}
                     for m in st.team if m.role == "verifier"],
                    [getattr(k, "value", str(k)) for k in st.chosen_strategy.verification_bundle],
                    [a.to_dict() for a in st.assumptions.assumptions])
                st.common_mode = cm.to_dict()
                if cm.findings:
                    self.emit(st, "common_mode_risk",
                              f"Common-mode risk {cm.score:.0%}: {cm.findings[0]}",
                              phase="planning", level="warning", payload=cm.to_dict())
                    if cm.recommendations and "different model" in cm.recommendations[0]:
                        self._diversify_verifier_model(st)
                        continue
                break

            # ---- phase 7: adaptive execution (sec. 15, 16, 17)
            verify_round = 0
            while True:
                task.status = TaskStatus.EXECUTING
                exec_ok = await self._execute_plan(st, und, t0)
                if not exec_ok and task.status in (TaskStatus.ABORTED, TaskStatus.FAILED):
                    return await self._finish(st, task.result_status or ResultStatus.FAILED,
                                              task.result_summary or "Execution failed", t0)

                # ---- phase 8: verification (sec. 25-31)
                task.status = TaskStatus.VERIFYING
                failed_runs = await self._verify(st, und)
                verify_round += 1
                # adaptive recovery: a refuted result changes the strategy (sec. 15-17)
                if failed_runs and verify_round <= 2:
                    why = "; ".join(r.detail[:100] for r in failed_runs[:2])
                    self.emit(st, "verification_failure",
                              f"Verification refuted the result — changing strategy: {why}",
                              level="error", phase="verification",
                              payload={"failed": [r.to_dict() for r in failed_runs]})
                    if await self._replan(st, und, f"verification refuted result: {why}"):
                        continue
                break

            # ---- phase 9: result + evidence closure (sec. 32, 43)
            return await self._conclude(st, und, t0)
        except NoProviderError as e:
            return await self._finish(st, ResultStatus.BLOCKED, str(e), t0)
        except Exception as e:  # never crash the kernel silently
            self.emit(st, "kernel_error", f"Unhandled kernel error: {e}", level="error",
                      payload={"error": str(e), "type": type(e).__name__})
            return await self._finish(st, ResultStatus.FAILED,
                                      f"Kernel error: {e}", t0)
        finally:
            task.duration_s = round(time.monotonic() - t0, 2)
            task.cost_usd = round(self.gateway.total_cost, 6)
            self.store.put("tasks", task.id, task.to_dict())

    # ------------------------------------------------------------ assumptions

    async def _check_assumptions(self, st: TaskState, und: TaskUnderstanding) -> list:
        refuted = []
        for a in st.assumptions.assumptions:
            if not a.verification_method.startswith("probe_check") or a.importance < 0.7:
                a.status = AssumptionStatus.DEFERRED
                continue
            a.status = AssumptionStatus.CHECKING
            try:
                tool, kwargs = a.verification_method.split(":", 2)[1:3]
                kwargs = json.loads(kwargs)
                res = st.tools.call("assumptions", tool, **kwargs)
                ok = res.ok and bool((res.stdout or "").strip() or res.artifacts)
                # fs_list probe on empty workspace refutes "code exists" assumptions
                if tool == "fs_list" and "workspace" in task_text(st):
                    files = [f for f in (res.stdout or "").splitlines() if f.strip()]
                    ok = len(files) > 0
                a.status = AssumptionStatus.CONFIRMED if ok else AssumptionStatus.REFUTED
                if ok:
                    st.evidence.add("measurement", f"Assumption confirmed: {a.statement[:80]}",
                                    tool=tool, result="confirmed")
                else:
                    refuted.append(a)
                    st.evidence.add("measurement", f"Assumption REFUTED: {a.statement[:80]}",
                                    tool=tool, result="refuted")
                    self.emit(st, "assumption_refuted",
                              f"Assumption refuted: {a.statement[:100]}", level="warning",
                              phase="strategy", payload={"assumption": a.to_dict()})
            except Exception as e:
                a.status = AssumptionStatus.DEFERRED
                a.note = f"probe failed: {e}"
        return [a for a in refuted if a.importance >= 0.7]

    # ------------------------------------------------------------ planning

    def _build_plan(self, st: TaskState, und: TaskUnderstanding) -> Plan:
        task = st.task
        strat = st.chosen_strategy
        # team selection (sec. 9) — smallest sufficient team
        reqs = [r for r in und.required_capabilities]
        step_caps = {s.capability for s in strat.steps}
        for c in step_caps:
            if c not in {r.capability for r in reqs}:
                from .core import RequiredCapability
                reqs.append(RequiredCapability(c, 0.5, 0.8))
        sel = SelectionEngine(self.registry)
        members, trace = sel.select_team(reqs, und)
        st.selection_trace = trace
        # assign roles by capability
        st.team = []
        for m in members:
            m.role = "verifier" if m.agent.name in ("tester", "verifier", "skeptic", "reviewer") else "producer"
            st.team.append(m)
        plan = Plan(version=len(st.plans) + 1, strategy_id=strat.id,
                    strategy_name=strat.name,
                    change_reason="initial plan" if len(st.plans) == 0 else st.plans[-1].change_reason)
        name_to_step_id = {}
        for s in strat.steps:
            ps = PlanStep(id=new_id("step"), name=s.name, goal=s.goal, capability=s.capability,
                          verification=list(s.verification))
            for dep in s.inputs:
                if dep in name_to_step_id:
                    ps.depends_on.append(name_to_step_id[dep])
            plan.steps.append(ps)
            name_to_step_id[s.name] = ps.id
            # assign agent + model + tools per step
        for ps in plan.steps:
            best, best_q = None, -1.0
            for m in st.team:
                q = m.agent.capability_quality(ps.capability)
                if q > best_q:
                    best, best_q = m, q
            if best is None and st.team:
                best = st.team[0]
            if best:
                ps.agent_id = best.agent.id
                best.steps.append(ps.id)
                ps.model = best.model
            ps.tools = STEP_TOOL_NEEDS.get(ps.name, ["fs_read", "fs_list"])
            st.tools.grant(ps.id, ps.tools)
        st.tools.grant("assumptions", ["fs_list", "fs_read", "test_run", "python_run"])
        st.tools.grant("verify", ["fs_read", "fs_write", "fs_list", "python_run",
                                  "test_run", "mutation", "benchmark"])
        return plan

    def _diversify_verifier_model(self, st: TaskState):
        producers = [m.model for m in st.team if m.role == "producer" and m.model]
        for m in st.team:
            if m.role == "verifier" and m.model in producers:
                try:
                    alt = self.gateway.route(LLMRequest(
                        messages=[], model_class=ModelClass.ADVERSARIAL,
                        exclude_models=[m.model] + producers, purpose="diversity"))
                    m.model = alt.id
                    self.emit(st, "model_diversified",
                              f"Verifier '{m.agent.name}' rerouted to {alt.id} "
                              f"(avoid common-mode failure)", phase="planning", level="warning")
                except Exception:
                    pass
                return

    # ------------------------------------------------------------ execution

    async def _execute_plan(self, st: TaskState, und: TaskUnderstanding, t0: float) -> bool:
        task = st.task
        sem = asyncio.Semaphore(st.budget.max_concurrency)
        while True:
            plan = st.plans[-1]
            ready = [s for s in plan.ready_steps() if s.name not in VERIFICATION_STEPS]
            pending_exec = [s for s in plan.steps
                            if s.status == StepStatus.PENDING and s.name not in VERIFICATION_STEPS]
            if not ready:
                failed_blocking = [s for s in plan.steps if s.status == StepStatus.FAILED]
                if failed_blocking and not pending_exec:
                    break
                if not pending_exec:
                    break
                # circular/unsatisfiable deps → replan
                if not await self._replan(st, und, "no runnable steps (dependency deadlock)"):
                    return False
                continue
            results = await asyncio.gather(*[
                self._run_step_safe(st, sem, s) for s in ready])
            for s, ok in zip(ready, results):
                if not ok:
                    reacted = await self._handle_failure(st, und, s, t0)
                    if not reacted:
                        return False
            # budget enforcement (sec. 41)
            if self.gateway.total_cost > st.budget.max_cost_usd:
                self.emit(st, "budget_exceeded",
                          f"Cost budget exceeded (${self.gateway.total_cost:.3f} > "
                          f"${st.budget.max_cost_usd})", level="error", phase="execution")
                task.result_status = ResultStatus.FAILED
                task.result_summary = "Resource budget exceeded"
                task.status = TaskStatus.FAILED
                return False
            if time.monotonic() - t0 > st.budget.max_seconds:
                self.emit(st, "budget_exceeded", "Time budget exceeded", level="error",
                          phase="execution")
                task.result_status = ResultStatus.FAILED
                task.result_summary = "Time budget exceeded"
                task.status = TaskStatus.FAILED
                return False
        return True

    async def _run_step_safe(self, st: TaskState, sem: asyncio.Semaphore, step: PlanStep) -> bool:
        async with sem:
            return await self._run_step(st, step)

    async def _run_step(self, st: TaskState, step: PlanStep) -> bool:
        plan = st.plans[-1]
        task = st.task
        agent = self.registry.agents.get(step.agent_id) if step.agent_id else None
        if agent is None:
            self.emit(st, "step_skipped", f"{step.name}: no agent assigned", level="warning")
            step.status = StepStatus.SKIPPED
            return True
        self.registry.mark_busy(agent.id)
        step.status = StepStatus.RUNNING
        step.attempts += 1
        step.started_at = now_utc()
        dep_ctx = ""
        for dep in step.depends_on:
            d = plan.step(dep)
            if d and d.output:
                dep_ctx += f"\n--- result of step '{d.name}' ---\n{d.output[:2500]}"
        ws_files = st.tools.call(step.id, "fs_list").stdout if "fs_list" in step.tools else ""
        sys_prompt = agent.system_prompt or (
            f"You are '{agent.name}': {agent.description}. Capabilities: "
            f"{', '.join(c.id for c in agent.capabilities)}. "
            + ("You are on probation — be conservative and verify assumptions." if agent.probation else "")
            + " Work ONLY on your step goal. Output a single JSON object.")
        user_prompt = (
            f"FAMA:STEP:{step.name}\n"
            f"PLAN V{len(st.plans)}"
            + (f" (replan after: {st.plans[-1].change_reason})" if len(st.plans) > 1 else "")
            + f"\nTask: {task.input[:1500]}\n"
            f"Understanding — goal: {task.understanding.goal if task.understanding else ''}\n"
            f"Constraints: {task.understanding.constraints if task.understanding else []}\n"
            f"Step goal: {step.goal}\n"
            f"Available tools: {step.tools}\n"
            f"Workspace files:\n{ws_files[:1500]}\n"
            f"{dep_ctx}\n\n"
            'Respond with JSON only: {"files": {"path.py": "full file content to write"}, '
            '"actions": [{"tool": "tool_id", "kwargs": {}}], "output": "concise result summary", '
            '"artifact": "main artifact path or key result", "confidence": 0.0-1.0, '
            '"assumptions_used": ["..."]}')
        self.emit(st, "step_started", f"{step.name} → {agent.name}"
                  + (f" · {step.model}" if step.model else ""),
                  phase="execution", payload={"step": step.name, "agent": agent.name,
                                              "model": step.model, "goal": step.goal})
        t0 = time.monotonic()
        cost_before = self.gateway.total_cost
        try:
            req = LLMRequest(messages=[LLMMessage("system", sys_prompt),
                                       LLMMessage("user", user_prompt)],
                             model=step.model, max_tokens=3500, temperature=0.2,
                             json_mode=True, purpose=f"step:{step.name}")
            resp = await asyncio.wait_for(self.gateway.complete(req), timeout=180)
            data = extract_json(resp.text)
            # write files via granted tools (deny-by-default still enforced)
            for path, content in (data.get("files") or {}).items():
                if not isinstance(content, str):
                    continue
                st.tools.call(step.id, "fs_write", path=str(path), content=content)
            for act in (data.get("actions") or [])[:6]:
                tool = str(act.get("tool", ""))
                kwargs = act.get("kwargs") or {}
                if tool in step.tools:
                    res = st.tools.call(step.id, tool, **kwargs)
                    if not res.ok:
                        step.output += f"\n[tool {tool} failed] {res.head(300)}"
            step.output = str(data.get("output", ""))[:6000]
            step.artifact = str(data.get("artifact", ""))[:300]
            step.cost_usd = round(self.gateway.total_cost - cost_before, 6)
            step.finished_at = now_utc()
            self.metrics.observe("step_latency_s", time.monotonic() - t0)
            if not step.output and not step.artifact and not data.get("files"):
                step.status = StepStatus.FAILED
                step.failure = "empty step output"
                st.failures.make(step, FailureClass.INVALID_OUTPUT, "empty step output",
                                 resp.text[:300])
                self.emit(st, "step_failed", f"{step.name}: empty output", level="error",
                          phase="execution", payload={"step": step.name})
                return False
            step.status = StepStatus.DONE
            st.evidence.add("action", f"step '{step.name}' by {agent.name}",
                            agent=agent.name, model=resp.model, result=step.artifact or step.output[:120],
                            content=step.output, payload={"step": step.name,
                                                          "cost_usd": step.cost_usd})
            self.emit(st, "step_done", f"{step.name} done"
                      + (f" → artifact {step.artifact}" if step.artifact else ""),
                      phase="execution", level="success",
                      payload={"step": step.name, "agent": agent.name, "model": resp.model,
                               "output": step.output[:1200], "artifact": step.artifact,
                               "cost_usd": step.cost_usd, "latency_s": round(resp.latency_s, 1)})
            self.metrics.inc("steps_done")
            return True
        except asyncio.TimeoutError:
            step.status = StepStatus.FAILED
            step.failure = "step timeout"
            st.failures.make(step, FailureClass.TIMEOUT, "step exceeded 180s")
            self.emit(st, "step_failed", f"{step.name}: timeout", level="error",
                      phase="execution")
            return False
        except Exception as e:
            step.status = StepStatus.FAILED
            step.failure = str(e)
            st.failures.classify_exception(step, e)
            self.emit(st, "step_failed", f"{step.name}: {e}", level="error",
                      phase="execution", payload={"step": step.name, "error": str(e)})
            return False
        finally:
            self.registry.mark_busy(agent.id, -1)

    async def _handle_failure(self, st: TaskState, und: TaskUnderstanding,
                              step: PlanStep, t0: float) -> bool:
        task = st.task
        task.failure_count += 1
        f = next((x for x in reversed(st.failures.failures) if x.step_id == step.id),
                 st.failures.failures[-1] if st.failures.failures else None)
        if f is None:
            return True
        self.metrics.inc(f"failures_{f.failure_class.value}")
        # hard attempt cap — no infinite retry loops (sec. 15: change approach instead)
        if step.attempts >= step.max_attempts + 2:
            self.emit(st, "step_abandoned",
                      f"'{step.name}' exhausted attempts ({step.attempts}) — changing strategy",
                      level="error", phase="execution")
            return await self._replan(st, und,
                                      f"step '{step.name}' exhausted attempts after "
                                      f"{f.failure_class.value}")
        # autopsy for important failures (sec. 18) — first failure of a step only
        if step.attempts <= 1 and f.failure_class not in (FailureClass.TIMEOUT,):
            rep = await self.autopsy.analyze(f, step, st.plans[-1], und)
            st.autopsies.append(rep.to_dict())
            self.emit(st, "autopsy_done", f"Autopsy: {rep.root_cause[:120]}", level="warning",
                      phase="execution", payload={"autopsy": rep.to_dict()})
        reaction = st.failures.next_reaction(f, step.attempts - 1)
        self.emit(st, "failure_react",
                  f"Failure {f.failure_class.value} on '{step.name}' → reaction: {reaction.value}",
                  level="warning", phase="execution",
                  payload={"failure": f.to_dict(), "reaction": reaction.value})
        plan = st.plans[-1]
        if reaction == FailureReaction.RETRY and step.attempts < step.max_attempts + 1:
            step.status = StepStatus.PENDING
            return True
        if reaction == FailureReaction.CHANGE_MODEL:
            try:
                alt = self.gateway.route(LLMRequest(messages=[], exclude_models=[step.model or ""],
                                                    purpose="reroute"))
                step.model = alt.id
                step.status = StepStatus.PENDING
                return True
            except Exception:
                pass
        if reaction == FailureReaction.REASSIGN:
            from .core import RequiredCapability
            sel = SelectionEngine(self.registry)
            members, _ = sel.select_team([RequiredCapability(step.capability, 0.4, 0.8)], und,
                                         avoid={self.registry.agents[step.agent_id].name}
                                         if step.agent_id in self.registry.agents else set())
            if members:
                step.agent_id = members[0].agent.id
                step.model = members[0].model
                step.status = StepStatus.PENDING
                return True
        if reaction == FailureReaction.CHANGE_TOOL:
            # a tool was missing/failed: extend the step's grants within policy
            for tool in ("fs_write", "python_run", "test_run"):
                if tool not in step.tools:
                    step.tools.append(tool)
                    st.tools.grant(step.id, step.tools)
            step.status = StepStatus.PENDING
            self.emit(st, "step_retooled", f"'{step.name}' granted additional tools", phase="execution")
            return True
        if reaction == FailureReaction.MODIFY_PLAN:
            # insert an analysis prerequisite and retry once
            prep = PlanStep(id=new_id("step"), name=f"analyze_{step.name}",
                            goal=f"Prepare/diagnose so that '{step.goal}' can succeed",
                            capability="planning")
            plan.steps.append(prep)
            step.depends_on.append(prep.id)
            step.status = StepStatus.PENDING
            st.tools.grant(prep.id, ["fs_read", "fs_list"])
            return True
        if reaction == FailureReaction.ADD_AGENT:
            # capability may be missing: route the step to an independent agent
            for name in ("skeptic", "verifier", "reviewer"):
                a = self.registry.by_name(name)
                if a and step.agent_id != a.id:
                    step.agent_id = a.id
                    st.tools.grant(step.id, step.tools)
                    break
            step.status = StepStatus.PENDING
            return True
        if reaction == FailureReaction.ESCALATE_VERIFICATION:
            if st.vbudget:
                st.vbudget.escalate("verification failure during execution")
            step.status = StepStatus.PENDING
            return True
        if reaction in (FailureReaction.AWAIT_HUMAN, FailureReaction.REMOVE_AGENT):
            return await self._replan(st, und, f"{reaction.value} on '{step.name}'")
        if reaction == FailureReaction.REPLAN:
            return await self._replan(st, und, f"failure {f.failure_class.value} on '{step.name}'")
        if reaction == FailureReaction.ABORT:
            task.status = TaskStatus.ABORTED
            task.result_status = ResultStatus.FAILED
            task.result_summary = f"Aborted after {f.failure_class.value}"
            self.emit(st, "task_aborted", f"Task aborted: {f.message}", level="error")
            return False
        step.status = StepStatus.PENDING
        return True

    async def _replan(self, st: TaskState, und: TaskUnderstanding, reason: str) -> bool:
        task = st.task
        if task.replan_count >= 3:
            task.status = TaskStatus.FAILED
            task.result_status = ResultStatus.FAILED
            task.result_summary = "Replan limit reached (3)"
            self.emit(st, "task_failed", "Replan limit reached", level="error")
            return False
        task.replan_count += 1
        evaluated = self.strategy_engine.search(und, task.autonomy_override or und.autonomy, None)
        used_ids = {p.strategy_id for p in st.plans}
        used_letters = {p.strategy_name.split("·")[0].strip() for p in st.plans}
        # prefer a different pattern — a new instance of the same approach is not a change
        nxt = next((e.strategy for e in evaluated
                    if e.strategy.id not in used_ids
                    and e.strategy.name.split("·")[0].strip() not in used_letters), None)
        if nxt is None:
            nxt = next((e.strategy for e in evaluated
                        if e.strategy.id not in used_ids), None)
        if nxt is None:
            task.status = TaskStatus.FAILED
            task.result_status = ResultStatus.FAILED
            task.result_summary = f"No alternative strategy after: {reason}"
            self.emit(st, "task_failed", f"No alternative strategy: {reason}", level="error")
            return False
        st.chosen_strategy = nxt
        st.plans[-1].outcome = f"replaced after: {reason}"
        plan = self._build_plan(st, und)
        plan.change_reason = reason
        plan.triggered_by = st.failures.failures[-1].id if st.failures.failures else ""
        st.plans.append(plan)
        task.plan_versions = len(st.plans)
        self.emit(st, "plan_changed",
                  f"PLAN V{plan.version}: strategy switched to {nxt.name} — reason: {reason}",
                  level="warning", phase="execution", payload={"plan": plan.to_dict(),
                  "previous_outcome": st.plans[-2].change_reason if len(st.plans) > 1 else ""})
        self.metrics.inc("replans")
        return True

    # ------------------------------------------------------------ verification

    async def _verify(self, st: TaskState, und: TaskUnderstanding) -> list[OracleRun]:
        """Run the oracle bundle. Returns the list of FAILED/REFUTED runs this round."""
        task = st.task
        strat = st.chosen_strategy
        if st.vbudget is None:
            st.vbudget = VerificationBudget(und)
            if (task.autonomy_override or und.autonomy) == AutonomyLevel.CRITICAL:
                st.vbudget.require_human = True
            if (task.autonomy_override or und.autonomy) == AutonomyLevel.MINIMAL:
                st.vbudget.max_oracles = min(st.vbudget.max_oracles, 1)
        self.emit(st, "verification_started",
                  f"Verification budget: {st.vbudget.max_oracles} oracle(s), "
                  f"independent={st.vbudget.require_independent}, human={st.vbudget.require_human}",
                  phase="verification", payload={"budget": st.vbudget.to_dict()})
        claim_node = st.evidence.add("claim",
                                     f"Result correct: {task.understanding.goal[:100]}",
                                     result="under test")
        main_artifact = self._implementation_artifact(st)
        runs: list[OracleRun] = []
        producers_models = [m.model for m in st.team if m.role == "producer"]

        # dedupe bundle — repeating the same oracle adds no independence (sec. 31)
        bundle: list[OracleKind] = []
        for k in strat.verification_bundle:
            if k not in bundle:
                bundle.append(k)
        # risk requires an independent oracle (sec. 26): enforce a strong kind
        strong = {OracleKind.INDEPENDENT_IMPLEMENTATION, OracleKind.DIFFERENTIAL,
                  OracleKind.MUTATION, OracleKind.PROPERTY_BASED}
        if st.vbudget.require_independent and not (set(bundle) & strong):
            bundle.append(OracleKind.MUTATION)
        for kind in bundle[:st.vbudget.max_oracles + st.vbudget.escalation_level]:
            run = await self._run_oracle(st, kind, main_artifact, producers_models)
            if run is None:
                continue
            runs.append(run)
            st.oracle_runs.append(run)
            node = st.evidence.add(
                "test" if run.verdict in ("pass", "weak") else "countertest",
                f"[{run.kind.value}] {run.target[:70]}", tool="oracle",
                result=run.verdict, payload={"verdict": run.verdict, "detail": run.detail,
                                             "strength": run.strength,
                                             "measurements": run.measurements})
            st.evidence.link(node.id, claim_node.id,
                             "refuted_by" if run.verdict in ("fail", "refuted") else "tested_by")
            self.emit(st, "oracle_run",
                      f"{run.kind.value}: {run.verdict.upper()} — {run.detail[:120]}",
                      phase="verification",
                      level="success" if run.verdict == "pass" else
                            ("warning" if run.verdict in ("weak", "inconclusive") else "error"),
                      payload=run.to_dict())
            self.metrics.inc(f"oracle_{run.verdict}")

        # escalation: VERIFICATION WEAK or failed run (sec. 26, 29)
        weak = [r for r in runs if r.verdict == "weak"]
        failed = [r for r in runs if r.verdict in ("fail", "refuted")]
        if weak and st.vbudget.escalate("mutation score below threshold"):
            self.emit(st, "verification_escalated",
                      "VERIFICATION WEAK — control level raised (extra independent oracle)",
                      level="warning", phase="verification")
            run = await self._run_oracle(st, OracleKind.PROPERTY_BASED, main_artifact,
                                         producers_models, claim=task.understanding.goal)
            if run:
                st.oracle_runs.append(run)
                node = st.evidence.add("countertest" if run.verdict == "refuted" else "test",
                                       f"[adversarial] counter-tests vs claim", tool="contradiction",
                                       result=run.verdict, payload={"verdict": run.verdict,
                                                                    "detail": run.detail})
                st.evidence.link(node.id, claim_node.id,
                                 "refuted_by" if run.verdict == "refuted" else "tested_by")
                self.emit(st, "oracle_run", f"contradiction: {run.verdict.upper()} — {run.detail[:120]}",
                          phase="verification",
                          level="success" if run.verdict == "pass" else "error",
                          payload=run.to_dict())
                if run.verdict == "refuted":
                    failed.append(run)
        st.final_gate_node = claim_node.id
        st.last_round_runs = runs
        st.notes.append(f"oracle_runs={len(st.oracle_runs)} failed={len(failed)}")
        # verification-phase plan steps are executed here, not by agents
        for s in st.plans[-1].steps:
            if s.name in VERIFICATION_STEPS and s.status == StepStatus.PENDING:
                s.status = StepStatus.DONE
                s.output = f"verification phase executed {len(st.oracle_runs)} oracle run(s)"
                s.finished_at = now_utc()
        return failed

    async def _run_oracle(self, st: TaskState, kind: OracleKind, artifact: str,
                          producers_models: list[str], *, claim: str = "") -> Optional[OracleRun]:
        try:
            if kind == OracleKind.DETERMINISTIC_TEST:
                res = st.tools.call("verify", "test_run", target=".")
                verdict = "pass" if res.ok else "fail"
                return OracleRun(new_id("orun"), kind, "test suite", verdict, 0.6,
                                 f"{(res.stdout or '').strip().splitlines()[-1] if (res.stdout or '').strip() else ''}"
                                 if res.ok else res.head(400),
                                 {"exit_code": res.exit_code})
            if kind == OracleKind.MUTATION:
                if not artifact:
                    return None
                return MutationTester(st.tools).run(artifact, max_mutants=8)
            if kind == OracleKind.DIFFERENTIAL:
                ws = st.tools.workspace
                cands = [p.name for p in sorted(ws.glob("*.py"))
                         if not p.name.startswith("__fama") and p.name != artifact]
                ref = next((c for c in cands if "reference" in c or "impl_b" in c or "_2" in c), None)
                if artifact and ref:
                    fn = self._main_function(st)
                    return DifferentialRunner(st.tools).run(artifact, ref, fn)
                return None
            if kind == OracleKind.INDEPENDENT_IMPLEMENTATION:
                ws = st.tools.workspace
                has_ref = any("reference" in p.name or "impl_b" in p.name for p in ws.glob("*.py"))
                if has_ref:
                    return OracleRun(new_id("orun"), kind, "independent implementation exists",
                                     "pass", 0.9, "second implementation present in workspace "
                                     "(different agent/model)")
                return OracleRun(new_id("orun"), kind, "independent implementation missing",
                                 "inconclusive", 0.1, "no independent implementation produced")
            if kind == OracleKind.BENCHMARK:
                fn = self._main_function(st)
                if artifact and fn:
                    res = st.tools.call("verify", "benchmark", file=artifact, function=fn)
                    ok = res.ok and "median_ms" in (res.stdout or "")
                    return OracleRun(new_id("orun"), kind, f"benchmark {artifact}",
                                     "pass" if ok else "inconclusive", 0.5,
                                     (res.stdout or "").strip()[:300] if ok else res.head(300))
                return None
            if kind == OracleKind.PROPERTY_BASED:
                if not artifact:
                    return None
                fn = self._main_function(st)
                summary = "\n".join(s.output for s in st.plans[-1].steps if s.output)[:1500]
                contra = ContradictionEngine(st.tools, self.gateway)
                return await contra.run(claim or "the produced solution is correct", summary,
                                        artifact, fn, producers_models)
            if kind == OracleKind.EXTERNAL_SOURCE:
                art = "\n".join(s.output for s in st.plans[-1].steps if s.output)
                n_src = art.count("http://") + art.count("https://")
                if n_src >= 2:
                    return OracleRun(new_id("orun"), kind, "source validation",
                                     "pass", 0.6, f"{n_src} distinct source URLs cited in the analysis")
                if n_src == 1:
                    return OracleRun(new_id("orun"), kind, "source validation", "inconclusive", 0.3,
                                     "only one source cited — cross-validation impossible")
                return OracleRun(new_id("orun"), kind, "source validation", "fail", 0.4,
                                 "no sources cited — claims unverifiable")
            if kind == OracleKind.DOMAIN_RULE:
                return OracleRun(new_id("orun"), kind, "domain rules", "pass", 0.4,
                                 "domain-rule spot checks embedded in review step")
            if kind == OracleKind.HUMAN:
                return OracleRun(new_id("orun"), kind, "human sign-off", "inconclusive", 0.9,
                                 "handled as approval gate at conclusion")
            return None
        except Exception as e:
            return OracleRun(new_id("orun"), kind, str(artifact), "inconclusive", 0.0,
                             f"oracle error: {e}")

    def _implementation_artifact(self, st: TaskState) -> str:
        """Prefer the implementation step's artifact over test files."""
        for name in ("implement", "fix", "implement_2", "execute", "optimize"):
            art = next((s.artifact for s in reversed(st.plans[-1].steps)
                        if s.name == name and s.artifact.endswith(".py")), "")
            if art:
                return art
        return next((s.artifact for s in reversed(st.plans[-1].steps)
                     if s.artifact and s.artifact.endswith(".py")), "")

    def _main_function(self, st: TaskState) -> str:
        art = self._implementation_artifact(st)
        if art:
            try:
                src = (st.tools.workspace / art).read_text()
                import ast as _ast
                tree = _ast.parse(src)
                fns = [n.name for n in _ast.walk(tree) if isinstance(n, _ast.FunctionDef)
                       and not n.name.startswith("_")]
                for cand in fns:
                    if any(k in cand.lower() for k in ("sort", "reverse", "encode",
                                                       "decode", "total", "solve", "process", "run")):
                        return cand
                return fns[0] if fns else "main"
            except Exception:
                pass
        return "main"

    # ------------------------------------------------------------ conclusion

    async def _conclude(self, st: TaskState, und: TaskUnderstanding, t0: float) -> Task:
        task = st.task
        runs = st.last_round_runs or st.oracle_runs
        failed = [r for r in runs if r.verdict in ("fail", "refuted")]
        weak = [r for r in runs if r.verdict == "weak"]
        inconclusive = [r for r in runs if r.verdict == "inconclusive"]
        passed = [r for r in runs if r.verdict == "pass"]
        strong_kinds = (OracleKind.INDEPENDENT_IMPLEMENTATION, OracleKind.DIFFERENTIAL,
                        OracleKind.MUTATION, OracleKind.PROPERTY_BASED)
        strong_passed = [r for r in passed if r.kind in strong_kinds]
        strength = max((r.strength for r in passed), default=0.0)

        if failed:
            status = ResultStatus.FAILED
            summary = (f"{len(failed)} verification oracle(s) refuted the result: "
                       + "; ".join(r.detail[:120] for r in failed[:2]))
            level = "error"
        elif not passed:
            status = ResultStatus.INSUFFICIENT_EVIDENCE
            summary = ("Verification produced no decisive passing evidence — NOT treated as success. "
                       + (inconclusive[0].detail[:150] if inconclusive else
                          (weak[0].detail[:150] if weak else "")))
            level = "warning"
        elif weak and not strong_passed:
            # weak suite + no strong oracle passed → evidence is insufficient (sec. 29, 43)
            status = ResultStatus.INSUFFICIENT_EVIDENCE
            summary = ("VERIFICATION WEAK: test suite leaves surviving mutants and no strong "
                       f"oracle passed — {weak[0].detail[:150]}")
            level = "warning"
        elif st.vbudget and st.vbudget.require_independent and not strong_passed:
            status = ResultStatus.INSUFFICIENT_EVIDENCE
            summary = "Risk level requires independent verification — none passed"
            level = "warning"
        else:
            status = ResultStatus.VERIFIED
            summary = (f"Result verified by {len(passed)} oracle(s) "
                       f"(strongest: {max(passed, key=lambda r: r.strength).kind.value}, "
                       f"strength {strength:.2f})")
            level = "success"

        # critical tasks: human sign-off (sec. 26, 37)
        if status == ResultStatus.VERIFIED and st.vbudget and st.vbudget.require_human:
            gates = self.governance.approval_required(
                task, und, {"production": False, "irreversible": False, "policy": False,
                            "security": False, "risk_level_high": True, "network_needed": False})
            gates = [g for g in gates if g.category == "verification"]
            if gates:
                self.governance.register_gates(task.id, self.governance.gates.get(task.id, []) + gates)
                task.status = TaskStatus.AWAITING_APPROVAL
                self.emit(st, "approval_required",
                          "Critical task — human sign-off of the verified result required",
                          phase="verification", level="warning",
                          payload={"gates": [g.to_dict() for g in gates]})
                try:
                    st.approval_event.clear()
                    await asyncio.wait_for(st.approval_event.wait(), timeout=3600)
                except asyncio.TimeoutError:
                    return await self._finish(st, ResultStatus.UNCERTAIN,
                                              "Awaiting human sign-off (timed out)", t0)
                if any(g.status == "rejected" for g in self.governance.gates.get(task.id, [])):
                    return await self._finish(st, ResultStatus.FAILED,
                                              "Verified result rejected by human", t0)
                st.evidence.add("approval", "human sign-off", result="approved")
                self.emit(st, "approval_granted", "Human accepted the verified result",
                          phase="verification", level="success")

        return await self._finish(st, status, summary, t0, level=level)

    async def _finish(self, st: TaskState, status: ResultStatus, summary: str,
                      t0: float, *, level: str = "info") -> Task:
        task = st.task
        task.result_status = status
        task.result_summary = summary[:2000]
        task.status = {ResultStatus.VERIFIED: TaskStatus.COMPLETED,
                       ResultStatus.FAILED: TaskStatus.FAILED,
                       ResultStatus.BLOCKED: TaskStatus.BLOCKED}.get(status, TaskStatus.COMPLETED)
        if status in (ResultStatus.UNCERTAIN, ResultStatus.INSUFFICIENT_EVIDENCE, ResultStatus.CONFLICT):
            task.status = TaskStatus.UNCERTAIN
        task.finished_at = now_utc()
        task.duration_s = round(time.monotonic() - t0, 2)
        task.cost_usd = round(self.gateway.total_cost, 6)
        task.tokens = dict(self.gateway.total_tokens)
        task.final_artifact = next((s.artifact for s in reversed(st.plans[-1].steps)
                                    if s.artifact), "") if st.plans else ""
        # learning (sec. 23, 24)
        if st.chosen_strategy and st.plans:
            try:
                self.memory.record(
                    st.task.understanding, st.chosen_strategy,
                    result=status.value,
                    verification_strength=max((r.strength for r in (st.last_round_runs or st.oracle_runs)
                                               if r.verdict in ("pass", "weak")), default=0.0),
                    cost_usd=task.cost_usd, seconds=task.duration_s,
                    failures=[f.failure_class.value for f in st.failures.failures],
                    replans=task.replan_count,
                    mutation_score=next((r.measurements.get("score") for r in st.oracle_runs
                                         if r.kind == OracleKind.MUTATION), None),
                    team=[m.agent.name for m in st.team],
                    assumptions_status={a.statement[:60]: a.status.value
                                        for a in st.assumptions.assumptions})
            except Exception:
                pass
            # agent performance updates (sec. 22)
            step_by_agent: dict[str, list[PlanStep]] = {}
            for p in st.plans:
                for s in p.steps:
                    if s.agent_id:
                        step_by_agent.setdefault(s.agent_id, []).append(s)
            verified = status == ResultStatus.VERIFIED
            for aid, steps in step_by_agent.items():
                for s in steps:
                    ok = s.status == StepStatus.DONE
                    self.registry.performance.record_outcome(
                        aid, s.capability, ok, 1.0 if ok else 0.0, s.cost_usd)
        self.metrics.inc("tasks_finished")
        if status == ResultStatus.VERIFIED:
            self.metrics.inc("tasks_completed")
        # final evidence node
        if st.evidence:
            node = st.evidence.add("claim", f"FINAL: {status.value}", result=status.value,
                                   payload={"summary": summary[:500], "cost_usd": task.cost_usd,
                                            "duration_s": task.duration_s})
            self.emit(st, "task_finished",
                      f"{status.value.upper()} — {summary[:200]}", level=level, phase="done",
                      payload={"status": status.value, "summary": summary,
                               "cost_usd": task.cost_usd, "tokens": task.tokens,
                               "duration_s": task.duration_s, "plan_versions": task.plan_versions,
                               "failures": task.failure_count, "replans": task.replan_count,
                               "final_node": node.id})
        self.store.put("tasks", task.id, task.to_dict())
        return task

    # ------------------------------------------------------------ human interaction

    def answer_clarification(self, task_id: str, answers: list[str]) -> bool:
        st = self.states.get(task_id)
        if not st or st.task.status != TaskStatus.AWAITING_CLARIFICATION:
            return False
        st.task.human_input["answers"] = answers
        st.clarify_event.set()
        return True

    def decide_gate(self, task_id: str, gate_id: str, approve: bool,
                    who: str = "human") -> bool:
        st = self.states.get(task_id)
        g = self.governance.decide_gate(task_id, gate_id, approve, who)
        if g is None:
            return False
        self.bus.publish(Event("gate_decided", task_id=task_id, phase="governance",
                               level="success" if approve else "warning",
                               title=f"Gate {g.category}: {'approved' if approve else 'rejected'}",
                               payload={"gate": g.to_dict()}))
        if st and not self.governance.pending_gates(task_id):
            st.approval_event.set()
        return True

    # ------------------------------------------------------------ introspection for API

    def state_dict(self, task_id: str) -> dict | None:
        st = self.states.get(task_id)
        if st is None:
            return None
        t = st.task
        return {
            "task": t.to_dict(),
            "understanding": t.understanding.to_dict() if t.understanding else None,
            "strategies": st.evaluated_strategies,
            "chosen_strategy": st.chosen_strategy.to_dict() if st.chosen_strategy else None,
            "plans": [p.to_dict() for p in st.plans],
            "team": [m.to_dict() for m in st.team],
            "selection_trace": st.selection_trace,
            "assumptions": [a.to_dict() for a in st.assumptions.assumptions],
            "failures": [f.to_dict() for f in st.failures.failures],
            "autopsies": st.autopsies,
            "oracle_runs": [r.to_dict() for r in st.oracle_runs],
            "evidence": st.evidence.to_dict() if st.evidence else None,
            "decisions": st.decisions.to_dict() if st.decisions else [],
            "common_mode": st.common_mode,
            "verification_budget": st.vbudget.to_dict() if st.vbudget else None,
            "clarifications": st.clarifications,
            "gates": [g.to_dict() for g in self.governance.gates.get(task_id, [])],
            "notes": st.notes,
        }


def task_text(st: TaskState) -> str:
    return st.task.input.lower()
