import pytest

from fama.core import ModelClass
from fama.llm import (LLMGateway, LLMRequest, ModelInfo, NoProviderError,
                      ProviderKind, ScriptedProvider, extract_json)


def _gw(monkeypatch=None, fixtures=None, env=None):
    if env and monkeypatch:
        for k, v in env.items():
            monkeypatch.setenv(k, v)
    return LLMGateway(scripted=ScriptedProvider(fixtures or []), allow_scripted=True)


@pytest.mark.asyncio
async def test_scripted_complete():
    gw = _gw(fixtures=[{"match": "ping", "text": '{"pong": true}'}])
    from fama.llm import LLMMessage
    resp = await gw.complete(LLMRequest(
        messages=[LLMMessage("user", "ping world")], purpose="test"))
    assert resp.scripted is True
    assert extract_json(resp.text)["pong"] is True


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('noise before {"a": [1,2,{"b":3}]} noise after') == {"a": [1, 2, {"b": 3}]}
    with pytest.raises(Exception):
        extract_json("no json here")


def test_route_no_provider(monkeypatch):
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL",
              "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    gw = LLMGateway()
    with pytest.raises(NoProviderError):
        gw.route(LLMRequest(messages=[]))


def test_route_respects_exclusions(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    gw = LLMGateway()
    models = gw.available_models()
    assert models, "openai key should enable openai models"
    first = models[0]
    m = gw.route(LLMRequest(messages=[], exclude_models=[first.id]))
    assert m.id != first.id


def test_openrouter_provider_enables_models(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    gw = LLMGateway()
    s = gw.provider_status()
    assert s["openrouter"] is True and s["any_real"] is True
    ids = [m.id for m in gw.available_models()]
    assert "openrouter/openai/gpt-4o-mini" in ids
    assert "openrouter/anthropic/claude-sonnet-4.5" in ids
    # direct OpenAI models must NOT be enabled by an OpenRouter key
    assert "openai/gpt-4o-mini" not in ids
    m = gw.route(LLMRequest(messages=[], model_class=ModelClass.CHEAP))
    assert m.id.startswith("openrouter/")


def test_env_registered_models(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("FAMA_MODELS",
                       "openai_compatible:my-model:fast+cheap:0.1:0.2;openai:broken-entry")
    gw = LLMGateway()
    ids = [m.id for m in gw.catalog()]
    assert "openai_compatible/my-model" in ids
    m = next(m for m in gw.catalog() if m.id == "openai_compatible/my-model")
    assert m.price_in == 0.1 and ModelClass.FAST in m.classes


def test_cost_math():
    m = ModelInfo("x/y", "y", ProviderKind.OPENAI, [], 2.0, 10.0, 5, 0.9)
    assert abs(m.cost(1_000_000, 1_000_000) - 12.0) < 1e-9
