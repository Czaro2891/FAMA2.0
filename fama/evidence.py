"""Evidence Graph (sec. 32) + Decision Trace (sec. 33).

Every significant result carries a chain of evidence that can answer:
"Why did FAMA accept this result?"  Decision traces record the OUTCOME of a
decision process (options, scores, reasons) — never private model
chain-of-thought.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

from .core import (DecisionRecord, EvidenceEdge, EvidenceKind, EvidenceNode,
                   RiskLevel, new_id, now_utc)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


class EvidenceGraph:
    def __init__(self, store=None, task_id: str = ""):
        self.store = store
        self.task_id = task_id
        self.nodes: dict[str, EvidenceNode] = {}
        self.edges: list[EvidenceEdge] = []

    def add(self, kind, label: str, *, agent: str = "", model: str = "",
            tool: str = "", result: str = "", content: str = "",
            payload: dict | None = None, nid: str | None = None) -> EvidenceNode:
        if isinstance(kind, str):
            kind = EvidenceKind(kind)
        n = EvidenceNode(id=nid or new_id("ev-" + kind.value[:4]), kind=kind, label=label,
                         agent=agent, model=model, tool=tool, result=result,
                         content_hash=content_hash(content) if content else "",
                         payload=payload or {})
        self.nodes[n.id] = n
        self._persist(n)
        return n

    def link(self, src: str, dst: str, relation: str):
        self.edges.append(EvidenceEdge(src, dst, relation))
        if self.store is not None:
            self.store.put("evidence", f"{self.task_id}:edge:{len(self.edges)}",
                           {"src": src, "dst": dst, "relation": relation})

    def _persist(self, n: EvidenceNode):
        if self.store is not None:
            self.store.put("evidence", f"{self.task_id}:{n.id}",
                           {"id": n.id, "kind": n.kind.value, "label": n.label,
                            "agent": n.agent, "model": n.model, "tool": n.tool,
                            "result": n.result, "hash": n.content_hash,
                            "payload": n.payload, "ts": n.ts})

    def refuted(self) -> list[EvidenceNode]:
        return [n for n in self.nodes.values()
                if n.payload.get("verdict") in ("refuted", "fail")]

    def why(self, node_id: str) -> dict:
        """Reconstruct the evidence chain supporting a node."""
        chain: dict = {"node": self._node_dict(self.nodes.get(node_id)), "supports": [], "refutes": []}
        for e in self.edges:
            if e.dst == node_id:
                sub = {"relation": e.relation, "node": self._node_dict(self.nodes.get(e.src))}
                if e.relation in ("refuted_by", "countertest"):
                    chain["refutes"].append(sub)
                else:
                    chain["supports"].append(sub)
        return chain

    def _node_dict(self, n: Optional[EvidenceNode]) -> dict:
        if n is None:
            return {}
        return {"id": n.id, "kind": n.kind.value, "label": n.label, "agent": n.agent,
                "model": n.model, "tool": n.tool, "result": n.result,
                "hash": n.content_hash, "ts": n.ts}

    def to_dict(self) -> dict:
        return {"nodes": [self._node_dict(n) for n in self.nodes.values()],
                "edges": [{"src": e.src, "dst": e.dst, "relation": e.relation}
                          for e in self.edges]}


class DecisionTrace:
    def __init__(self, store=None, task_id: str = ""):
        self.store = store
        self.task_id = task_id
        self.records: list[DecisionRecord] = []

    def record(self, decision: str, options: list[dict], selected: str, score: float,
               reason: str, evidence_refs: list[str] | None = None,
               confidence: float = 0.6, risk: RiskLevel = RiskLevel.LOW) -> DecisionRecord:
        r = DecisionRecord(id=new_id("dec"), decision=decision, options=options,
                           selected=selected, score=round(score, 4), reason=reason,
                           evidence_refs=evidence_refs or [], confidence=confidence,
                           risk=risk)
        self.records.append(r)
        if self.store is not None:
            self.store.put("decisions", f"{self.task_id}:{r.id}",
                           {"id": r.id, "ts": r.ts, "decision": decision,
                            "options": options, "selected": selected, "score": r.score,
                            "reason": reason, "evidence_refs": r.evidence_refs,
                            "confidence": confidence, "risk": risk.value})
        return r

    def to_dict(self):
        from .core import dc_to_dict
        return dc_to_dict(self.records)
