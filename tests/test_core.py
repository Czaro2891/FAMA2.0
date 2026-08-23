import pytest

from fama.core import (Event, EventBus, Plan, PlanStep, StepStatus, Task,
                       TaskType, RiskLevel, dc_to_dict)


def test_event_bus_pubsub():
    bus = EventBus()
    got = []
    unsub = bus.subscribe(got.append)
    ev = bus.publish(Event("test", title="hello"))
    assert len(got) == 1 and got[0].title == "hello"
    unsub()
    bus.publish(Event("test2"))
    assert len(got) == 1
    assert len(bus.history()) == 2


def test_event_bus_broken_subscriber_does_not_break():
    bus = EventBus()
    got = []
    bus.subscribe(lambda e: 1 / 0)
    bus.subscribe(got.append)
    bus.publish(Event("x"))
    assert len(got) == 1


def test_plan_ready_steps_respects_deps():
    p = Plan(version=1, strategy_id="s")
    a = PlanStep(id="a", name="a", goal="", capability="x")
    b = PlanStep(id="b", name="b", goal="", capability="y", depends_on=["a"])
    c = PlanStep(id="c", name="c", goal="", capability="z")
    p.steps = [a, b, c]
    ready = p.ready_steps()
    assert {s.id for s in ready} == {"a", "c"}
    a.status = StepStatus.DONE
    c.status = StepStatus.DONE
    ready = p.ready_steps()
    assert {s.id for s in ready} == {"b"}


def test_task_serializes():
    t = Task(id="task-x", input="hello")
    d = dc_to_dict(t)
    assert d["id"] == "task-x"
    assert d["status"] == "created"


def test_enums_values():
    assert TaskType.CODE_GENERATION.value == "code_generation"
    assert RiskLevel.CRITICAL.value == "critical"
