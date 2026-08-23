"""Failure Engine + Agent Autopsy (sec. 17, 18).

A failure is classified, not just logged.  The class determines the
reaction: retry, reassign, change model, change tool, add agent, modify
plan, replan, escalate verification, await human, abort.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .core import (Failure, FailureClass, FailureReaction, Plan, PlanStep,
                   StepStatus, TaskUnderstanding)
from .llm import LLMGateway, LLMMessage, ModelError

# reaction policy per failure class (first choice, then fallbacks)
REACTION_POLICY: dict[FailureClass, list[FailureReaction]] = {
    FailureClass.MODEL_FAILURE: [FailureReaction.CHANGE_MODEL, FailureReaction.RETRY,
                                 FailureReaction.REASSIGN],
    FailureClass.AGENT_FAILURE: [FailureReaction.REASSIGN, FailureReaction.CHANGE_MODEL,
                                 FailureReaction.REPLAN],
    FailureClass.TOOL_FAILURE: [FailureReaction.CHANGE_TOOL, FailureReaction.MODIFY_PLAN,
                                FailureReaction.REPLAN],
    FailureClass.PLAN_FAILURE: [FailureReaction.MODIFY_PLAN, FailureReaction.REPLAN],
    FailureClass.CAPABILITY_GAP: [FailureReaction.ADD_AGENT, FailureReaction.REPLAN,
                                  FailureReaction.ABORT],
    FailureClass.INVALID_OUTPUT: [FailureReaction.RETRY, FailureReaction.REASSIGN,
                                  FailureReaction.ADD_AGENT],
    FailureClass.VERIFICATION_FAILURE: [FailureReaction.ESCALATE_VERIFICATION,
                                        FailureReaction.MODIFY_PLAN, FailureReaction.REPLAN],
    FailureClass.ASSUMPTION_FAILURE: [FailureReaction.REPLAN],
    FailureClass.RESOURCE_FAILURE: [FailureReaction.MODIFY_PLAN, FailureReaction.ABORT],
    FailureClass.TIMEOUT: [FailureReaction.CHANGE_MODEL, FailureReaction.RETRY,
                           FailureReaction.MODIFY_PLAN],
    FailureClass.ENVIRONMENT_FAILURE: [FailureReaction.RETRY, FailureReaction.ABORT],
    FailureClass.GOVERNANCE_BLOCK: [FailureReaction.AWAIT_HUMAN, FailureReaction.ABORT],
}


class FailureEngine:
    def __init__(self):
        self.failures: list[Failure] = []

    def classify_exception(self, step: PlanStep, exc: Exception) -> Failure:
        if isinstance(exc, PermissionError):
            cls = FailureClass.GOVERNANCE_BLOCK
        elif "NoProviderError" in type(exc).__name__ or "provider" in str(exc).lower():
            cls = FailureClass.MODEL_FAILURE
        elif isinstance(exc, TimeoutError) or "timeout" in str(exc).lower():
            cls = FailureClass.TIMEOUT
        elif isinstance(exc, ModelError):
            cls = FailureClass.MODEL_FAILURE
        elif "tool" in str(exc).lower():
            cls = FailureClass.TOOL_FAILURE
        else:
            cls = FailureClass.AGENT_FAILURE
        return self.make(step, cls, str(exc), type(exc).__name__)

    def classify_result(self, step: PlanStep, output: str, *, invalid: bool = False,
                        empty_artifact: bool = False, tool_failed: bool = False) -> Optional[Failure]:
        if invalid:
            return self.make(step, FailureClass.INVALID_OUTPUT,
                             "step output failed structural validation", output[:400])
        if empty_artifact:
            return self.make(step, FailureClass.INVALID_OUTPUT,
                             "step produced no artifact", output[:400])
        if tool_failed:
            return self.make(step, FailureClass.TOOL_FAILURE, "tool execution failed", output[:400])
        return None

    def make(self, step: PlanStep, cls: FailureClass, message: str, detail: str = "") -> Failure:
        reactions = REACTION_POLICY[cls]
        f = Failure(id=f"fail-{len(self.failures) + 1:03d}", step_id=step.id,
                    agent_id=step.agent_id or "", model=step.model or "",
                    failure_class=cls, message=message, detail=detail,
                    reaction=reactions[0])
        self.failures.append(f)
        return f

    def next_reaction(self, f: Failure, attempt: int) -> FailureReaction:
        options = REACTION_POLICY[f.failure_class]
        idx = min(attempt, len(options) - 1)
        return options[idx]

    def by_class(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.failures:
            out[f.failure_class.value] = out.get(f.failure_class.value, 0) + 1
        return out


@dataclass
class AutopsyReport:
    failure_id: str
    root_cause: str = ""
    wrong_agent: bool = False
    wrong_model: bool = False
    wrong_tool: bool = False
    wrong_plan: bool = False
    bad_assumption: bool = False
    verification_insufficient: bool = False
    lesson: str = ""
    confidence: float = 0.5

    def to_dict(self):
        from .core import dc_to_dict
        return dc_to_dict(self)


AUTOPSY_PROMPT = """You perform a failure autopsy for FAMA 2.0.
Failure context:
- step: {step_name} ({step_goal})
- failure class: {fclass}
- message: {message}
- agent: {agent}
- model: {model}
- attempt #: {attempt}
- plan step status before failure: {prev_status}
Respond with JSON only:
{{"root_cause": "...", "wrong_agent": true/false, "wrong_model": true/false,
"wrong_tool": true/false, "wrong_plan": true/false, "bad_assumption": true/false,
"verification_insufficient": true/false, "lesson": "one actionable sentence",
"confidence": 0.0-1.0}}"""


class AgentAutopsy:
    """Post-failure root-cause analysis feeding future decisions (sec. 18)."""

    def __init__(self, gateway: LLMGateway):
        self.gateway = gateway
        self.reports: list[AutopsyReport] = []

    async def analyze(self, f: Failure, step: PlanStep, plan: Plan,
                      und: TaskUnderstanding) -> AutopsyReport:
        from .llm import LLMRequest, ModelClass, extract_json
        rep = AutopsyReport(failure_id=f.id)
        try:
            resp = await self.gateway.complete(LLMRequest(
                messages=[LLMMessage("system",
                                     "You are the Autopsy module of FAMA 2.0. Diagnose the root "
                                     "cause of this failure honestly. JSON only."),
                          LLMMessage("user", AUTOPSY_PROMPT.format(
                              step_name=step.name, step_goal=step.goal,
                              fclass=f.failure_class.value, message=f.message[:500],
                              agent=f.agent_id, model=f.model, attempt=step.attempts,
                              prev_status=step.status.value))],
                model_class=ModelClass.REASONING, max_tokens=700, temperature=0.1,
                json_mode=True, purpose="autopsy"))
            data = extract_json(resp.text)
            rep.root_cause = str(data.get("root_cause", ""))[:400]
            rep.wrong_agent = bool(data.get("wrong_agent", False))
            rep.wrong_model = bool(data.get("wrong_model", False))
            rep.wrong_tool = bool(data.get("wrong_tool", False))
            rep.wrong_plan = bool(data.get("wrong_plan", False))
            rep.bad_assumption = bool(data.get("bad_assumption", False))
            rep.verification_insufficient = bool(data.get("verification_insufficient", False))
            rep.lesson = str(data.get("lesson", ""))[:300]
            rep.confidence = float(data.get("confidence", 0.5))
        except Exception as e:
            rep.root_cause = f"autopsy model call failed: {e}"
            rep.confidence = 0.2
        self.reports.append(rep)
        return rep
