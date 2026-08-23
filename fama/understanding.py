"""Task Understanding (sec. 5).

Transforms an unstructured user request into a formal executable problem.
Ambiguity is never auto-assumed-away: FAMA asks, or states explicit
assumptions (which the Assumption Engine tracks).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .core import (AutonomyLevel, Complexity, OracleKind, RiskLevel,
                   RequiredCapability, Task, TaskType, TaskUnderstanding, clamp)
from .llm import LLMGateway, LLMMessage, ModelClass, ModelError, extract_json

SYSTEM_PROMPT = """You are the Task Understanding layer of FAMA 2.0, an adaptive agent operating system.
Your job: convert the user's raw request into a formal executable problem model.
Respond with a single JSON object, no prose, using exactly this schema:
{
 "goal": "what must be achieved (one sentence)",
 "deliverable": "what artifact/output must exist when done",
 "constraints": ["..."],
 "risks": ["consequences of an error here"],
 "risk_level": "negligible|low|medium|high|critical",
 "complexity": "trivial|simple|moderate|complex|wicked",
 "uncertainties": ["things not yet known"],
 "ambiguities": ["aspects with more than one interpretation"],
 "success_criteria": ["objectively checkable completion criteria"],
 "domain": "e.g. software|security|research|data|general",
 "task_type": "code_generation|debugging|code_review|software_architecture|research|data_analysis|experiment|optimization|security|planning|automation|design|writing|composite|unknown",
 "required_capabilities": [{"capability": "<id from allowed list>", "min_quality": 0.0-1.0, "importance": 0.0-1.0, "why": "..."}],
 "autonomy": "minimal|standard|high|critical",
 "verification_requirements": ["deterministic_test|independent_implementation|differential|property_based|metamorphic|mutation|benchmark|external_source|domain_rule|human"],
 "interpretation": "the safest reasonable interpretation you choose (may be empty)",
 "clarifying_questions": ["questions worth asking if ambiguity is significant"],
 "confidence": 0.0-1.0
}
Be honest: if the request is vague, raise ambiguities and lower confidence. Never invent certainty."""

MARKER = "FAMA:PHASE:UNDERSTANDING"


def _safe_enum(val, enum_cls, default):
    try:
        return enum_cls(str(val).strip().lower())
    except Exception:
        return default


@dataclass
class UnderstandingResult:
    understanding: TaskUnderstanding
    used_fallback: bool = False
    notes: list[str] = field(default_factory=list)


class UnderstandingEngine:
    def __init__(self, gateway: LLMGateway, allowed_capabilities: list[str]):
        self.gateway = gateway
        self.allowed = set(allowed_capabilities)

    async def understand(self, task: Task) -> UnderstandingResult:
        caps_list = ", ".join(sorted(self.allowed))
        user = (f"[{MARKER}]\nUser request:\n\"\"\"\n{task.input}\n\"\"\"\n"
                f"Allowed capability ids: {caps_list}\n"
                + (f"Workspace files exist: {task.workspace}\n" if task.workspace else "")
                + (f"User autonomy preference: {task.autonomy_override.value}\n"
                   if task.autonomy_override else ""))
        notes: list[str] = []
        for attempt in range(2):
            try:
                resp = await self.gateway.complete(LLMRequest_wrap(
                    messages=[LLMMessage("system", SYSTEM_PROMPT), LLMMessage("user", user)],
                    model_class=ModelClass.REASONING, max_tokens=1800, temperature=0.1,
                    json_mode=True, purpose="task_understanding"))
                data = extract_json(resp.text)
                und = self._from_json(data)
                return UnderstandingResult(und, notes=notes)
            except (ModelError, ValueError, KeyError) as e:
                notes.append(f"understanding attempt {attempt + 1} failed: {e}")
        return UnderstandingResult(self._fallback(task), used_fallback=True, notes=notes)

    def _from_json(self, d: dict) -> TaskUnderstanding:
        und = TaskUnderstanding()
        und.goal = str(d.get("goal", "")).strip()
        und.deliverable = str(d.get("deliverable", "")).strip()
        und.constraints = [str(x) for x in d.get("constraints", [])][:12]
        und.risks = [str(x) for x in d.get("risks", [])][:12]
        und.risk_level = _safe_enum(d.get("risk_level"), RiskLevel, RiskLevel.MEDIUM)
        und.complexity = _safe_enum(d.get("complexity"), Complexity, Complexity.MODERATE)
        und.uncertainties = [str(x) for x in d.get("uncertainties", [])][:12]
        und.ambiguities = [str(x) for x in d.get("ambiguities", [])][:8]
        und.success_criteria = [str(x) for x in d.get("success_criteria", [])][:10]
        und.domain = str(d.get("domain", "general")).strip().lower() or "general"
        und.task_type = _safe_enum(d.get("task_type"), TaskType, TaskType.UNKNOWN)
        for rc in d.get("required_capabilities", [])[:14]:
            if isinstance(rc, dict) and rc.get("capability") in self.allowed:
                und.required_capabilities.append(RequiredCapability(
                    capability=rc["capability"],
                    min_quality=clamp(float(rc.get("min_quality", 0.5))),
                    importance=clamp(float(rc.get("importance", 0.7))),
                    why=str(rc.get("why", ""))[:200]))
        if not und.required_capabilities:
            und.required_capabilities = self._default_caps(und)
        und.autonomy = _safe_enum(d.get("autonomy"), AutonomyLevel, AutonomyLevel.STANDARD)
        for v in d.get("verification_requirements", [])[:8]:
            try:
                und.verification_requirements.append(OracleKind(str(v).strip().lower()))
            except ValueError:
                pass
        und.interpretation = str(d.get("interpretation", "")).strip()
        und.clarifying_questions = [str(x) for x in d.get("clarifying_questions", [])][:5]
        und.confidence = clamp(float(d.get("confidence", 0.5)))
        return und

    def _default_caps(self, und: TaskUnderstanding) -> list[RequiredCapability]:
        mapping = {
            TaskType.CODE_GENERATION: ["backend", "unit_testing"],
            TaskType.DEBUGGING: ["debugging", "unit_testing"],
            TaskType.CODE_REVIEW: ["code_review"],
            TaskType.ARCHITECTURE: ["architecture"],
            TaskType.RESEARCH: ["research", "source_validation"],
            TaskType.DATA_ANALYSIS: ["data_processing", "statistical_analysis"],
            TaskType.OPTIMIZATION: ["optimization", "benchmarking"],
            TaskType.SECURITY: ["security_analysis", "code_review"],
            TaskType.WRITING: ["writing", "critique"],
            TaskType.EXPERIMENT: ["experiment_execution"],
            TaskType.PLANNING: ["planning"],
        }
        ids = mapping.get(und.task_type, ["planning"])
        return [RequiredCapability(i, 0.5, 0.8) for i in ids]

    def _fallback(self, task: Task) -> TaskUnderstanding:
        und = TaskUnderstanding()
        und.goal = task.input[:200]
        und.task_type = TaskType.UNKNOWN
        und.ambiguities = ["model understanding unavailable — request could not be interpreted"]
        und.uncertainties = ["no reliable interpretation without a model"]
        und.confidence = 0.05
        und.risk_level = RiskLevel.MEDIUM
        und.required_capabilities = []
        return und


def LLMRequest_wrap(**kw):
    from .llm import LLMRequest
    return LLMRequest(**kw)
