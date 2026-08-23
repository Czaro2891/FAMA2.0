"""Governance (sec. 35, 37).

Central security & permission rules.  Every agent acts within permissions;
dynamic agent creation can never bypass governance.  Human approval is
required for production / irreversible / high-risk / policy-changing
actions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .core import (AutonomyLevel, Event, RiskLevel, Task, TaskUnderstanding,
                   new_id, now_utc)

# Keywords that flag production / irreversible / policy contexts
PRODUCTION_PATTERNS = [
    r"\bprodukcyjn", r"\bproduction\b", r"\bdeploy", r"\bwdroż", r"\bmigrate\b",
    r"\bdrop\s+table\b", r"\bdelete\s+from\b", r"\bprod\b",
]
IRREVERSIBLE_PATTERNS = [
    r"\busuń\s+(bazę|tabel|dane)", r"\bdrop\b", r"\btruncate\b", r"\bformat\b",
    r"\birreversible\b", r"\bnieodwracaln",
]
POLICY_PATTERNS = [
    r"\bzmi(e|ę)n(a|y)\s+polityk", r"\bpolicy\s+change", r"\bzmień\s+uprawnien",
    r"\bpermissions\s+change", r"\bodbezpiecz", r"\bwyłącz\s+bezpieczeństw",
]
SECURITY_PATTERNS = [
    r"\bsecurity\b", r"\bpodatnośc", r"\bvulnerab", r"\bexploit", r"\bhashowan",
    r"\bauth", r"\bsekret", r"\bsecret", r"\bklucz\s+api", r"\bapi\s+key",
]


@dataclass
class ApprovalGate:
    id: str
    reason: str
    category: str            # production | irreversible | policy | security | risk | verification
    detail: str = ""
    status: str = "pending"  # pending | approved | rejected
    decided_by: str = ""
    decided_at: str = ""

    def to_dict(self):
        from .core import dc_to_dict
        return dc_to_dict(self)


@dataclass
class GovernanceState:
    allow_network: bool = False
    allow_production: bool = False
    allow_custom_agents: bool = True
    max_custom_agents_per_task: int = 1
    require_approval_above_risk: RiskLevel = RiskLevel.HIGH
    human_verification_for: tuple = (AutonomyLevel.CRITICAL,)

    def to_dict(self):
        from .core import dc_to_dict
        return dc_to_dict(self)


class Governance:
    """Central policy object shared by orchestrator, tools and agent factory."""

    def __init__(self, bus: Event | None = None):
        from .core import EventBus
        self.bus = bus if isinstance(bus, EventBus) else None
        self.state = GovernanceState()
        self.gates: dict[str, list[ApprovalGate]] = {}   # task_id -> gates
        self.tool_denies: set[str] = set()

    # ------------------------------------------------------------ risk assessment

    def assess(self, task: Task, und: TaskUnderstanding) -> dict:
        text = f"{task.input}\n{und.goal}".lower()
        flags = {
            "production": any(re.search(p, text) for p in PRODUCTION_PATTERNS),
            "irreversible": any(re.search(p, text) for p in IRREVERSIBLE_PATTERNS),
            "policy": any(re.search(p, text) for p in POLICY_PATTERNS),
            "security": any(re.search(p, text) for p in SECURITY_PATTERNS),
            "risk_level_high": und.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL),
            "network_needed": und.task_type.value in ("research", "data_analysis") or
                              any("research" in c.capability or "source" in c.capability
                                  for c in und.required_capabilities),
        }
        if flags["network_needed"]:
            self.state.allow_network = True   # scoped: only fetch/search tools, sandbox still scrubbed
        return flags

    def approval_required(self, task: Task, und: TaskUnderstanding, flags: dict) -> list[ApprovalGate]:
        gates: list[ApprovalGate] = []
        autonomy = task.autonomy_override or und.autonomy
        if flags["production"] and not self.state.allow_production:
            gates.append(ApprovalGate(new_id("gate"), "Production impact detected",
                                      "production",
                                      "Task text indicates production systems. Human approval required before execution."))
        if flags["irreversible"]:
            gates.append(ApprovalGate(new_id("gate"), "Potentially irreversible action",
                                      "irreversible",
                                      "Task may perform irreversible operations. Human approval required."))
        if flags["policy"]:
            gates.append(ApprovalGate(new_id("gate"), "Policy / permission change",
                                      "policy",
                                      "Task attempts to change security policy. Human approval required."))
        if und.risk_level == RiskLevel.CRITICAL or autonomy == AutonomyLevel.CRITICAL:
            gates.append(ApprovalGate(new_id("gate"), "Critical risk verification sign-off",
                                      "verification",
                                      "Final result of a critical task requires human acceptance."))
        return gates

    def register_gates(self, task_id: str, gates: list[ApprovalGate]):
        self.gates[task_id] = gates

    def pending_gates(self, task_id: str) -> list[ApprovalGate]:
        return [g for g in self.gates.get(task_id, []) if g.status == "pending"]

    def decide_gate(self, task_id: str, gate_id: str, approve: bool, who: str = "human") -> Optional[ApprovalGate]:
        for g in self.gates.get(task_id, []):
            if g.id == gate_id:
                g.status = "approved" if approve else "rejected"
                g.decided_by = who
                g.decided_at = now_utc()
                return g
        return None

    # ------------------------------------------------------------ enforcement

    def check_tool(self, tool: str):
        if tool in self.tool_denies:
            raise PermissionError(f"tool '{tool}' denied by governance policy")
        if tool in ("web_fetch", "web_search") and not self.state.allow_network:
            raise PermissionError("network tools require governance to allow network egress for this task")

    def custom_agent_allowed(self, current_custom: int) -> bool:
        return (self.state.allow_custom_agents and
                current_custom < self.state.max_custom_agents_per_task)

    def custom_agent_limits(self) -> dict:
        """New agents start with low trust and restricted permissions (sec. 21)."""
        return {
            "trust": 0.35,
            "permissions": ["workspace"],
            "tools_whitelist": ["fs_read", "fs_write", "fs_list", "python_run", "test_run"],
            "probation": True,
            "sandbox_required": True,
        }
