"""FAMA core: fundamental types, event bus, identifiers.

Everything in FAMA revolves around the TASK (sec. 4).  Agents, models,
tools and strategies are resources selected per problem.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import itertools
import json
import threading
import time
import uuid
from typing import Any, Callable, Optional


# ---------------------------------------------------------------- utilities

def now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def monotonic() -> float:
    return time.monotonic()


def slug(text: str, maxlen: int = 48) -> str:
    keep = [c if c.isalnum() else "-" for c in text.strip().lower()]
    out = "".join(keep).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    return (out or "x")[:maxlen]


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=_json_default, indent=1)


def _json_default(o: Any):
    if isinstance(o, (enum.Enum,)):
        return o.value
    if dataclasses.is_dataclass(o):
        return dc_to_dict(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def dc_to_dict(o: Any) -> dict:
    if isinstance(o, enum.Enum):
        return o.value
    if dataclasses.is_dataclass(o):
        out = {}
        for f in dataclasses.fields(o):
            v = getattr(o, f.name)
            out[f.name] = dc_to_dict(v)
        return out
    if isinstance(o, dict):
        return {k: dc_to_dict(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [dc_to_dict(v) for v in o]
    if isinstance(o, set):
        return sorted(o)
    return o


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------- enums

class TaskType(str, enum.Enum):
    CODE_GENERATION = "code_generation"
    DEBUGGING = "debugging"
    CODE_REVIEW = "code_review"
    ARCHITECTURE = "software_architecture"
    RESEARCH = "research"
    DATA_ANALYSIS = "data_analysis"
    EXPERIMENT = "experiment"
    OPTIMIZATION = "optimization"
    SECURITY = "security"
    PLANNING = "planning"
    AUTOMATION = "automation"
    DESIGN = "design"
    WRITING = "writing"
    COMPOSITE = "composite"
    UNKNOWN = "unknown"


class RiskLevel(str, enum.Enum):
    NEGLIGIBLE = "negligible"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Complexity(str, enum.Enum):
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    WICKED = "wicked"


class AutonomyLevel(str, enum.Enum):
    MINIMAL = "minimal"
    STANDARD = "standard"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, enum.Enum):
    CREATED = "created"
    UNDERSTANDING = "understanding"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    UNCERTAIN = "uncertain"
    BLOCKED = "blocked"


class StepStatus(str, enum.Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    REPLANNED = "replanned"


class FailureClass(str, enum.Enum):
    AGENT_FAILURE = "agent_failure"
    MODEL_FAILURE = "model_failure"
    TOOL_FAILURE = "tool_failure"
    PLAN_FAILURE = "plan_failure"
    CAPABILITY_GAP = "capability_gap"
    INVALID_OUTPUT = "invalid_output"
    VERIFICATION_FAILURE = "verification_failure"
    ASSUMPTION_FAILURE = "assumption_failure"
    RESOURCE_FAILURE = "resource_failure"
    TIMEOUT = "timeout"
    ENVIRONMENT_FAILURE = "environment_failure"
    GOVERNANCE_BLOCK = "governance_block"


class FailureReaction(str, enum.Enum):
    RETRY = "retry"
    REASSIGN = "reassign"
    CHANGE_MODEL = "change_model"
    CHANGE_TOOL = "change_tool"
    ADD_AGENT = "add_agent"
    REMOVE_AGENT = "remove_agent"
    MODIFY_PLAN = "modify_plan"
    REPLAN = "replan"
    ESCALATE_VERIFICATION = "escalate_verification"
    AWAIT_HUMAN = "await_human"
    ABORT = "abort"


class OracleKind(str, enum.Enum):
    DETERMINISTIC_TEST = "deterministic_test"
    INDEPENDENT_IMPLEMENTATION = "independent_implementation"
    DIFFERENTIAL = "differential"
    PROPERTY_BASED = "property_based"
    METAMORPHIC = "metamorphic"
    MUTATION = "mutation"
    BENCHMARK = "benchmark"
    EXTERNAL_SOURCE = "external_source"
    DOMAIN_RULE = "domain_rule"
    HUMAN = "human"


class ModelClass(str, enum.Enum):
    CHEAP = "cheap"
    FAST = "fast"
    REASONING = "reasoning"
    CODING = "coding"
    ADVERSARIAL = "adversarial"
    LOCAL = "local"
    VISION = "vision"


class ProviderKind(str, enum.Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENAI_COMPATIBLE = "openai_compatible"
    SCRIPTED = "scripted"  # deterministic test double; never used unless explicitly enabled


class CapabilitySource(str, enum.Enum):
    NATIVE = "native"
    TOOL_BASED = "tool_based"
    MODEL_BASED = "model_based"
    HYBRID = "hybrid"


class AssumptionStatus(str, enum.Enum):
    PROPOSED = "proposed"
    CHECKING = "checking"
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    DEFERRED = "deferred"


class EvidenceKind(str, enum.Enum):
    CLAIM = "claim"
    ARTIFACT = "artifact"
    ACTION = "action"
    MEASUREMENT = "measurement"
    TEST = "test"
    COUNTERTEST = "countertest"
    SOURCE = "source"
    DECISION = "decision"
    FAILURE = "failure"
    APPROVAL = "approval"


class ResultStatus(str, enum.Enum):
    VERIFIED = "verified"
    FAILED = "failed"
    UNCERTAIN = "uncertain"
    BLOCKED = "blocked"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICT = "conflict"


# ---------------------------------------------------------------- event bus

class Event:
    """Immutable system event. The World UI renders real state from these."""

    __slots__ = ("id", "ts", "task_id", "type", "phase", "level", "title", "payload")

    def __init__(self, type_: str, *, task_id: str | None = None,
                 phase: str = "", level: str = "info", title: str = "",
                 payload: dict | None = None):
        self.id = new_id("ev")
        self.ts = now_utc()
        self.task_id = task_id
        self.type = type_
        self.phase = phase
        self.level = level  # info | success | warning | error
        self.title = title
        self.payload = payload or {}

    def to_dict(self) -> dict:
        return {"id": self.id, "ts": self.ts, "task_id": self.task_id,
                "type": self.type, "phase": self.phase, "level": self.level,
                "title": self.title, "payload": dc_to_dict(self.payload)}


class EventBus:
    """In-process pub/sub. Persisted + broadcast (SSE) by subscribers."""

    def __init__(self):
        self._lock = threading.Lock()
        self._subs: list[Callable[[Event], None]] = []
        self._counter = itertools.count()
        self._seq: list[Event] = []
        self._dropped = 0

    def subscribe(self, fn: Callable[[Event], None]) -> Callable[[], None]:
        with self._lock:
            self._subs.append(fn)

        def unsub():
            with self._lock:
                if fn in self._subs:
                    self._subs.remove(fn)

        return unsub

    def publish(self, ev: Event) -> Event:
        with self._lock:
            subs = list(self._subs)
            self._seq.append(ev)
            if len(self._seq) > 20000:
                self._dropped += len(self._seq) - 10000
                self._seq = self._seq[-10000:]
        for fn in subs:
            try:
                fn(ev)
            except Exception:  # a broken subscriber never stops the bus
                pass
        return ev

    def history(self, task_id: str | None = None, limit: int = 500) -> list[Event]:
        with self._lock:
            seq = list(self._seq)
        if task_id is not None:
            seq = [e for e in seq if e.task_id == task_id]
        return seq[-limit:]


# ---------------------------------------------------------------- task model (sec. 4/5)

@dataclasses.dataclass
class Capability:
    """An actual ability to perform an operation class (sec. 6)."""
    id: str
    name: str
    source: CapabilitySource = CapabilitySource.MODEL_BASED
    quality: float = 0.6          # 0..1 base quality of this system capability
    verification: str = ""        # how capability quality is verified
    cost_weight: float = 1.0      # relative cost of exercising it

    def to_dict(self):
        return dc_to_dict(self)


@dataclasses.dataclass
class RequiredCapability:
    capability: str
    min_quality: float = 0.5
    importance: float = 1.0       # 0..1 — how critical for this task
    why: str = ""

    def to_dict(self):
        return dc_to_dict(self)


@dataclasses.dataclass
class TaskUnderstanding:
    """Structured model of an unstructured request (sec. 5)."""
    goal: str = ""
    deliverable: str = ""
    constraints: list[str] = dataclasses.field(default_factory=list)
    risks: list[str] = dataclasses.field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.MEDIUM
    complexity: Complexity = Complexity.MODERATE
    uncertainties: list[str] = dataclasses.field(default_factory=list)
    ambiguities: list[str] = dataclasses.field(default_factory=list)
    success_criteria: list[str] = dataclasses.field(default_factory=list)
    domain: str = "general"
    task_type: TaskType = TaskType.UNKNOWN
    required_capabilities: list[RequiredCapability] = dataclasses.field(default_factory=list)
    autonomy: AutonomyLevel = AutonomyLevel.STANDARD
    verification_requirements: list[OracleKind] = dataclasses.field(default_factory=list)
    interpretation: str = ""       # chosen safe interpretation, if any
    clarifying_questions: list[str] = dataclasses.field(default_factory=list)
    confidence: float = 0.5

    def to_dict(self):
        return dc_to_dict(self)


@dataclasses.dataclass
class Task:
    """The primary object of FAMA (sec. 4)."""
    id: str
    input: str
    status: TaskStatus = TaskStatus.CREATED
    understanding: Optional[TaskUnderstanding] = None
    workspace: str = ""            # directory the task operates on
    autonomy_override: Optional[AutonomyLevel] = None
    budget: dict = dataclasses.field(default_factory=dict)  # {max_cost_usd, max_seconds, max_tokens}
    created_at: str = dataclasses.field(default_factory=now_utc)
    finished_at: str | None = None
    result_status: Optional[ResultStatus] = None
    result_summary: str = ""
    final_artifact: str = ""
    cost_usd: float = 0.0
    tokens: dict = dataclasses.field(default_factory=lambda: {"input": 0, "output": 0})
    duration_s: float = 0.0
    plan_versions: int = 0
    failure_count: int = 0
    replan_count: int = 0
    tags: list[str] = dataclasses.field(default_factory=list)
    human_input: dict = dataclasses.field(default_factory=dict)  # clarifications / approvals

    def to_dict(self):
        return dc_to_dict(self)


# ---------------------------------------------------------------- agent model (sec. 7)

@dataclasses.dataclass
class AgentDNA:
    """Traits that really influence agent selection and strategy (sec. 7)."""
    reasoning_style: str = "balanced"      # analytical | pragmatic | creative | adversarial
    coding_style: str = "clean"            # minimal | clean | defensive | exploratory
    risk_tolerance: float = 0.4
    creativity: float = 0.4
    verification_bias: float = 0.6         # tendency to double-check own output
    exploration: float = 0.4
    cost_sensitivity: float = 0.5
    error_tolerance: float = 0.4
    preferred_model_classes: list[str] = dataclasses.field(default_factory=list)
    preferred_tools: list[str] = dataclasses.field(default_factory=list)

    def to_dict(self):
        return dc_to_dict(self)


@dataclasses.dataclass
class AgentSpec:
    """Executor of a set of capabilities (sec. 7/8)."""
    id: str
    name: str
    description: str = ""
    capabilities: list[Capability] = dataclasses.field(default_factory=list)
    limitations: list[str] = dataclasses.field(default_factory=list)
    permissions: list[str] = dataclasses.field(default_factory=list)
    tools: list[str] = dataclasses.field(default_factory=list)
    preferred_models: list[str] = dataclasses.field(default_factory=list)  # model classes
    domains: list[str] = dataclasses.field(default_factory=list)
    reliability: float = 0.8
    trust: float = 0.8
    cost_factor: float = 1.0       # multiplier over base model price
    latency_factor: float = 1.0
    availability: float = 1.0
    risk_profile: RiskLevel = RiskLevel.LOW
    failure_modes: list[str] = dataclasses.field(default_factory=list)
    dna: AgentDNA = dataclasses.field(default_factory=AgentDNA)
    custom: bool = False           # dynamically created (sec. 21)
    probation: bool = False        # new agents start with low trust
    system_prompt: str = ""

    def capability_ids(self) -> set[str]:
        return {c.id for c in self.capabilities}

    def capability_quality(self, cap_id: str) -> float:
        for c in self.capabilities:
            if c.id == cap_id:
                return c.quality
        return 0.0

    def to_dict(self):
        return dc_to_dict(self)


@dataclasses.dataclass
class AgentPerformanceRecord:
    """Performance is a profile per (agent, capability) — not one number (sec. 22)."""
    agent_id: str
    capability: str
    successes: int = 0
    failures: int = 0
    avg_quality: float = 0.0       # 0..1 judged by verification outcomes
    avg_cost_usd: float = 0.0
    avg_seconds: float = 0.0
    last_used: str = ""

    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total else 0.0

    def to_dict(self):
        return dc_to_dict(self) | {"success_rate": self.success_rate}


# ---------------------------------------------------------------- assumptions (sec. 14)

@dataclasses.dataclass
class Assumption:
    id: str
    statement: str
    confidence: float = 0.5
    importance: float = 0.5
    risk: RiskLevel = RiskLevel.LOW
    verification_method: str = "deferred"   # probe:<tool>:<call> | llm_check | deferred | human
    status: AssumptionStatus = AssumptionStatus.PROPOSED
    evidence_ref: str = ""
    note: str = ""

    def to_dict(self):
        return dc_to_dict(self)


# ---------------------------------------------------------------- strategy (sec. 11-13)

@dataclasses.dataclass
class StrategyStep:
    """A slot in a strategy: what must happen, not who does it."""
    id: str
    name: str
    goal: str
    capability: str
    inputs: list[str] = dataclasses.field(default_factory=list)   # step ids
    verification: list[OracleKind] = dataclasses.field(default_factory=list)
    parallelizable: bool = True
    estimated_tokens: int = 2500

    def to_dict(self):
        return dc_to_dict(self)


@dataclasses.dataclass
class Strategy:
    id: str
    name: str
    pattern: str                    # specialist | pipeline | dual_implementation | research_synthesis | ...
    description: str = ""
    steps: list[StrategyStep] = dataclasses.field(default_factory=list)
    verification_bundle: list[OracleKind] = dataclasses.field(default_factory=list)
    team_size: int = 1
    redundancy: int = 1             # independent implementations
    est_cost_usd: float = 0.0
    est_seconds: float = 0.0
    est_success_prob: float = 0.7
    verification_strength: float = 0.5
    utility: float = 0.0
    scores: dict = dataclasses.field(default_factory=dict)   # factor -> score
    weights: dict = dataclasses.field(default_factory=dict)  # factor -> weight (task-dependent)
    rationale: str = ""
    twin_prediction: bool = False   # estimates are predictions, never results
    memory_ref: str = ""            # strategy-memory provenance, if any

    def to_dict(self):
        return dc_to_dict(self)


# ---------------------------------------------------------------- plan (sec. 10/16)

@dataclasses.dataclass
class PlanStep:
    id: str
    name: str
    goal: str
    capability: str
    depends_on: list[str] = dataclasses.field(default_factory=list)
    agent_id: str | None = None
    model: str | None = None
    tools: list[str] = dataclasses.field(default_factory=list)
    verification: list[OracleKind] = dataclasses.field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    max_attempts: int = 2
    artifact: str = ""              # produced artifact path/content ref
    output: str = ""                # textual result
    failure: str = ""
    cost_usd: float = 0.0
    tokens: dict = dataclasses.field(default_factory=lambda: {"input": 0, "output": 0})
    started_at: str | None = None
    finished_at: str | None = None

    def to_dict(self):
        return dc_to_dict(self)


@dataclasses.dataclass
class Plan:
    version: int
    strategy_id: str
    strategy_name: str = ""
    steps: list[PlanStep] = dataclasses.field(default_factory=list)
    change_reason: str = ""
    triggered_by: str = ""          # event/failure id that caused replan
    created_at: str = dataclasses.field(default_factory=now_utc)
    outcome: str = ""               # measured outcome of this plan version

    def step(self, sid: str) -> PlanStep | None:
        for s in self.steps:
            if s.id == sid:
                return s
        return None

    def ready_steps(self) -> list[PlanStep]:
        done = {s.id for s in self.steps if s.status in (StepStatus.DONE, StepStatus.SKIPPED)}
        return [s for s in self.steps
                if s.status == StepStatus.PENDING and all(d in done for d in s.depends_on)]

    def to_dict(self):
        return dc_to_dict(self)


# ---------------------------------------------------------------- failures (sec. 17)

@dataclasses.dataclass
class Failure:
    id: str
    step_id: str = ""
    agent_id: str = ""
    model: str = ""
    tool: str = ""
    failure_class: FailureClass = FailureClass.AGENT_FAILURE
    message: str = ""
    detail: str = ""
    reaction: FailureReaction = FailureReaction.RETRY
    ts: str = dataclasses.field(default_factory=now_utc)

    def to_dict(self):
        return dc_to_dict(self)


# ---------------------------------------------------------------- verification (sec. 25-31)

@dataclasses.dataclass
class OracleRun:
    id: str
    kind: OracleKind
    target: str                     # artifact under test
    verdict: str = ""               # pass | fail | inconclusive | refuted
    strength: float = 0.5           # evidential strength of this oracle
    detail: str = ""
    measurements: dict = dataclasses.field(default_factory=dict)
    ts: str = dataclasses.field(default_factory=now_utc)

    def to_dict(self):
        return dc_to_dict(self)


# ---------------------------------------------------------------- evidence (sec. 32/33)

@dataclasses.dataclass
class EvidenceNode:
    id: str
    kind: EvidenceKind
    label: str
    agent: str = ""
    model: str = ""
    tool: str = ""
    result: str = ""
    content_hash: str = ""
    payload: dict = dataclasses.field(default_factory=dict)
    ts: str = dataclasses.field(default_factory=now_utc)


@dataclasses.dataclass
class EvidenceEdge:
    src: str
    dst: str
    relation: str   # produced | tested_by | refuted_by | supported_by | measured_by | decided_by | caused


@dataclasses.dataclass
class DecisionRecord:
    """Outcome of a decision process — not private model chain-of-thought (sec. 33)."""
    id: str
    decision: str
    options: list[dict] = dataclasses.field(default_factory=list)
    selected: str = ""
    score: float = 0.0
    reason: str = ""
    evidence_refs: list[str] = dataclasses.field(default_factory=list)
    confidence: float = 0.5
    risk: RiskLevel = RiskLevel.LOW
    ts: str = dataclasses.field(default_factory=now_utc)


# ---------------------------------------------------------------- resource budget (sec. 41)

@dataclasses.dataclass
class ResourceBudget:
    max_cost_usd: float = 2.0
    max_seconds: float = 900.0
    max_tokens: int = 400_000
    max_concurrency: int = 3

    def to_dict(self):
        return dc_to_dict(self)
