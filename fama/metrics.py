"""Observability (sec. 39).

Metrics derive from real events only — no decorative numbers.
"""
from __future__ import annotations

import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Histogram:
    values: list[float] = field(default_factory=list)

    def observe(self, v: float):
        self.values.append(v)
        if len(self.values) > 5000:
            self.values = self.values[-2500:]

    def summary(self) -> dict:
        if not self.values:
            return {"count": 0}
        vs = sorted(self.values)
        return {"count": len(vs), "mean": round(statistics.fmean(vs), 3),
                "p50": round(vs[len(vs) // 2], 3),
                "p95": round(vs[int(len(vs) * 0.95)], 3),
                "max": round(vs[-1], 3)}


class Metrics:
    def __init__(self, store=None):
        self.store = store
        self.counters: dict[str, int] = defaultdict(int)
        self.gauges: dict[str, float] = {}
        self.hists: dict[str, Histogram] = defaultdict(Histogram)
        self.t0 = time.time()

    def inc(self, name: str, n: int = 1):
        self.counters[name] += n

    def gauge(self, name: str, value: float):
        self.gauges[name] = value

    def observe(self, name: str, value: float):
        self.hists[name].observe(value)

    def snapshot(self, gateway=None, registry=None) -> dict:
        out: dict = {
            "uptime_s": round(time.time() - self.t0, 1),
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {k: h.summary() for k, h in self.hists.items()},
        }
        if gateway is not None:
            out["model_usage"] = gateway.usage
            out["total_cost_usd"] = round(gateway.total_cost, 6)
            out["total_tokens"] = gateway.total_tokens
        if registry is not None:
            out["agent_utilization"] = dict(registry.utilization)
            out["agents_registered"] = len(registry.agents)
            out["agent_performance"] = registry.performance.to_dict()
        tasks = self.counters.get("tasks_total", 0)
        done = self.counters.get("tasks_completed", 0)
        out["success_rate"] = round(done / tasks, 3) if tasks else None
        if self.store is not None:
            self.store.put("metrics", "latest", out)
        return out
