"""LLM Gateway — models are resources (sec. 19).

Real providers only in production:
  * OpenAI            (OPENAI_API_KEY)
  * Anthropic         (ANTHROPIC_API_KEY)
  * OpenAI-compatible (OPENAI_API_KEY + OPENAI_BASE_URL, e.g. OpenRouter/Ollama/vLLM)

Additional models can be registered via env:
  FAMA_MODELS="provider:model_id:classes:price_in:price_out;..."
  classes is '+'-joined from: cheap fast reasoning coding adversarial local vision

A ScriptedProvider exists ONLY as a deterministic test double for the test
suite and recorded replays.  It is clearly labelled and must never be
presented as intelligence (sec. 48: evidence over claims).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

from .core import ModelClass, ProviderKind, clamp, new_id

PRICE_NOTE = "prices are approximate planning estimates, not billing"

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines, '#' comments, no overrides)."""
    try:
        p = Path(path)
        if not p.exists():
            return
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k.strip(), v)
    except Exception:
        pass


_load_dotenv()


@dataclass
class ModelInfo:
    id: str                      # provider-qualified: openai/gpt-4o-mini
    model_id: str                # provider-native id
    provider: ProviderKind
    classes: list[ModelClass]
    price_in: float              # USD / Mtok input
    price_out: float             # USD / Mtok output
    latency_s: float             # typical seconds per medium call
    quality: float               # 0..1 general capability estimate
    context: int = 128_000
    available: bool = True
    note: str = ""

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in / 1e6) * self.price_in + (tokens_out / 1e6) * self.price_out

    def to_dict(self):
        from .core import dc_to_dict
        return dc_to_dict(self) | {"classes": [c.value for c in self.classes]}


def _mc(*names: str) -> list[ModelClass]:
    return [ModelClass(n) for n in names]


CATALOG: list[ModelInfo] = [
    ModelInfo("openai/gpt-4o-mini", "gpt-4o-mini", ProviderKind.OPENAI,
              _mc("cheap", "fast"), 0.15, 0.60, 4.0, 0.72),
    ModelInfo("openai/gpt-4o", "gpt-4o", ProviderKind.OPENAI,
              _mc("coding"), 2.50, 10.00, 9.0, 0.90),
    ModelInfo("openai/o4-mini", "o4-mini", ProviderKind.OPENAI,
              _mc("reasoning", "coding"), 1.10, 4.40, 18.0, 0.88),
    ModelInfo("anthropic/claude-haiku-4.5", "claude-haiku-4-5", ProviderKind.ANTHROPIC,
              _mc("cheap", "fast"), 1.00, 5.00, 5.0, 0.80),
    ModelInfo("anthropic/claude-sonnet-4.5", "claude-sonnet-4-5", ProviderKind.ANTHROPIC,
              _mc("coding", "reasoning"), 3.00, 15.00, 12.0, 0.92),
    ModelInfo("anthropic/claude-opus-4.1", "claude-opus-4-1", ProviderKind.ANTHROPIC,
              _mc("reasoning", "adversarial"), 15.00, 75.00, 25.0, 0.95),
]

DEFAULT_COMPAT_MODELS = [
    "fast:qwen2.5:7b",  # placeholder replaced at runtime by FAMA_MODELS or sensible default
]

# models served through OpenRouter (openai-compatible API)
OPENROUTER_MODELS = [
    ("openai/gpt-4o-mini", _mc("cheap", "fast"), 0.15, 0.60, 5.0, 0.72),
    ("anthropic/claude-3.5-haiku", _mc("cheap", "fast"), 0.80, 4.00, 6.0, 0.78),
    ("openai/gpt-4o", _mc("coding"), 2.50, 10.00, 10.0, 0.90),
    ("anthropic/claude-sonnet-4.5", _mc("coding", "reasoning"), 3.00, 15.00, 13.0, 0.92),
    ("openai/o4-mini", _mc("reasoning", "coding"), 1.10, 4.40, 20.0, 0.88),
    ("anthropic/claude-opus-4.1", _mc("adversarial", "reasoning"), 15.00, 75.00, 28.0, 0.95),
]


class ModelError(Exception):
    pass


class NoProviderError(ModelError):
    """No real LLM provider configured — system must honestly block."""


@dataclass
class LLMMessage:
    role: str
    content: str


@dataclass
class LLMRequest:
    messages: list[LLMMessage]
    model: str | None = None              # explicit provider-qualified id
    model_class: ModelClass | None = None # routing hint
    max_tokens: int = 2048
    temperature: float = 0.2
    json_mode: bool = False
    purpose: str = ""                     # accounting label (understanding/step/verify...)
    exclude_models: list[str] = field(default_factory=list)  # diversity / common-mode avoidance


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: ProviderKind
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_s: float
    scripted: bool = False
    raw: dict = field(default_factory=dict)


class ScriptedProvider:
    """Deterministic fixture provider for tests/replays. Not AI — by design."""

    def __init__(self, fixtures: list[dict]):
        # fixtures: [{"match": substr-or-regex, "model_class": optional, "text": ...}]
        self.fixtures = fixtures
        self.calls: list[str] = []

    def respond(self, req: LLMRequest) -> str:
        prompt = "\n".join(m.content for m in req.messages)
        self.calls.append(prompt)
        want_class = req.model_class.value if req.model_class else None
        for fx in self.fixtures:
            pat = fx.get("match", "")
            cls = fx.get("model_class")
            if pat and re.search(pat, prompt, re.IGNORECASE | re.DOTALL) and \
               (cls is None or cls == want_class or req.model_class is None):
                return fx["text"]
        # generic fallback so orchestrations never crash in tests
        return json.dumps({"note": "scripted-fallback", "prompt_head": prompt[:200]})


class LLMGateway:
    """Single entry point for every model call. Tracks usage & cost."""

    def __init__(self, *, scripted: ScriptedProvider | None = None,
                 allow_scripted: bool = False, http_timeout: float = 120.0):
        self.scripted = scripted
        self.allow_scripted = allow_scripted
        self.http_timeout = http_timeout
        self.usage: dict[str, dict] = {}     # model -> {in,out,cost,calls}
        self.total_cost = 0.0
        self.total_tokens = {"input": 0, "output": 0}
        self.call_log: list[dict] = []
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    # ---------------------------------------------------------- providers

    def provider_status(self) -> dict:
        s = {
            "openai": bool(os.environ.get("OPENAI_API_KEY")),
            "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "openai_compatible": bool(os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_BASE_URL")),
            "openrouter": bool(os.environ.get("OPENROUTER_API_KEY")),
            "scripted": self.allow_scripted,
        }
        s["any_real"] = s["openai"] or s["anthropic"] or s["openai_compatible"] or s["openrouter"]
        return s

    def _env_models(self) -> list[ModelInfo]:
        out: list[ModelInfo] = []
        raw = os.environ.get("FAMA_MODELS", "")
        for entry in [e.strip() for e in raw.split(";") if e.strip()]:
            parts = entry.split(":")
            if len(parts) < 3:
                continue
            prov, mid, classes = parts[0], parts[1], parts[2]
            pin = float(parts[3]) if len(parts) > 3 and parts[3] else 0.5
            pout = float(parts[4]) if len(parts) > 4 and parts[4] else 1.5
            try:
                pk = ProviderKind(prov)
            except ValueError:
                continue
            mcs = [_mc(c.strip())[0] for c in classes.split("+") if c.strip()]
            out.append(ModelInfo(f"{prov}/{mid}", mid, pk, mcs, pin, pout, 8.0, 0.8,
                                 note="registered via FAMA_MODELS"))
        return out

    def catalog(self) -> list[ModelInfo]:
        models = [m for m in CATALOG]
        status = self.provider_status()
        for m in models:
            if m.provider == ProviderKind.OPENAI:
                m.available = status["openai"]
            elif m.provider == ProviderKind.ANTHROPIC:
                m.available = status["anthropic"]
        env = self._env_models()
        if env:
            have = {m.id for m in models}
            for m in env:
                if m.provider == ProviderKind.OPENAI and status["openai_compatible"]:
                    m.available = True
                    if m.id not in have:
                        models.append(m)
                elif m.provider in (ProviderKind.OPENAI_COMPATIBLE,) and status["openai_compatible"]:
                    m.available = True
                    if m.id not in have:
                        models.append(m)
        if status["openai_compatible"]:
            base = os.environ.get("FAMA_COMPAT_DEFAULT_MODEL", "").strip()
            if base and f"openai_compatible/{base}" not in {m.id for m in models}:
                models.append(ModelInfo(f"openai_compatible/{base}", base,
                                        ProviderKind.OPENAI_COMPATIBLE,
                                        _mc("fast", "coding"), 0.4, 1.2, 8.0, 0.8,
                                        note="default model of OPENAI_BASE_URL endpoint"))
        if status["openrouter"]:
            have = {m.id for m in models}
            for mid, classes, pin, pout, lat, q in OPENROUTER_MODELS:
                mid_qualified = f"openrouter/{mid}"
                if mid_qualified not in have:
                    models.append(ModelInfo(mid_qualified, mid,
                                            ProviderKind.OPENAI_COMPATIBLE, classes,
                                            pin, pout, lat, q, note="via OpenRouter"))
        return models

    def available_models(self) -> list[ModelInfo]:
        return [m for m in self.catalog() if m.available]

    def route(self, req: LLMRequest) -> ModelInfo:
        """Model routing (sec. 19): problem kind, cost, quality, diversity."""
        models = [m for m in self.available_models()
                  if m.id not in req.exclude_models]
        if not models:
            raise NoProviderError(
                "No LLM provider configured. Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY "
                "(optionally OPENAI_BASE_URL for any OpenAI-compatible endpoint), "
                "or FAMA_MODELS to register models.")
        if req.model:
            for m in models:
                if m.id == req.model:
                    return m
            raise ModelError(f"Requested model {req.model} is not available")
        if req.model_class:
            pref = [m for m in models if req.model_class in m.classes]
            if pref:
                pref.sort(key=lambda m: (m.price_in + m.price_out) / (m.quality + 0.01))
                return pref[0]
        models.sort(key=lambda m: (m.price_in + m.price_out) / (m.quality + 0.01))
        return models[0]

    # ---------------------------------------------------------- api calls

    def _client_(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.http_timeout)
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def complete(self, req: LLMRequest) -> LLMResponse:
        t0 = time.monotonic()
        if self.allow_scripted and self.scripted is not None:
            model = self.route(req) if self._any_or_scripted() else self._scripted_model(req)
            text = self.scripted.respond(req)
            tin, tout = self._estimate_tokens(req, text)
            resp = LLMResponse(text, model.id if model else "scripted/fixture",
                               ProviderKind.SCRIPTED, tin, tout, 0.0,
                               time.monotonic() - t0, scripted=True)
            self._account(resp, req)
            return resp
        model = self.route(req)
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                text, tin, tout = await self._call_provider(model, req)
                resp = LLMResponse(text, model.id, model.provider, tin, tout,
                                   model.cost(tin, tout), time.monotonic() - t0)
                self._account(resp, req)
                return resp
            except NoProviderError:
                raise
            except Exception as e:  # 429/5xx/network — retry with backoff
                last_err = e
                await asyncio.sleep(1.5 * (attempt + 1))
        raise ModelError(f"model call failed after retries: {last_err}")

    def _any_or_scripted(self) -> bool:
        return False  # scripted mode never routes to real providers

    def _scripted_model(self, req: LLMRequest) -> Optional[ModelInfo]:
        return ModelInfo("scripted/fixture", "fixture", ProviderKind.SCRIPTED,
                         _mc("cheap", "fast"), 0, 0, 0.0, 0.5, note="deterministic test double")

    async def _call_provider(self, m: ModelInfo, req: LLMRequest) -> tuple[str, int, int]:
        client = self._client_()
        if m.provider in (ProviderKind.OPENAI, ProviderKind.OPENAI_COMPATIBLE):
            orkey = os.environ.get("OPENROUTER_API_KEY", "")
            if m.provider == ProviderKind.OPENAI_COMPATIBLE and orkey:
                base, key = OPENROUTER_BASE, orkey
            else:
                base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
                key = os.environ.get("OPENAI_API_KEY", "")
            body: dict = {
                "model": m.model_id,
                "messages": [{"role": x.role, "content": x.content} for x in req.messages],
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
            }
            if req.json_mode:
                body["response_format"] = {"type": "json_object"}
            r = await client.post(f"{base}/chat/completions",
                                  headers={"Authorization": f"Bearer {key}"},
                                  json=body)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"] or ""
            u = data.get("usage", {})
            return text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        if m.provider == ProviderKind.ANTHROPIC:
            key = os.environ.get("ANTHROPIC_API_KEY", "")
            sys_msgs = [x.content for x in req.messages if x.role == "system"]
            others = [{"role": x.role, "content": x.content}
                      for x in req.messages if x.role != "system"]
            body = {
                "model": m.model_id,
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
                "messages": others,
                "system": "\n".join(sys_msgs) or None,
            }
            r = await client.post("https://api.anthropic.com/v1/messages",
                                  headers={"x-api-key": key,
                                           "anthropic-version": "2023-06-01"},
                                  json=body)
            r.raise_for_status()
            data = r.json()
            text = "".join(b.get("text", "") for b in data.get("content", []))
            u = data.get("usage", {})
            return text, u.get("input_tokens", 0), u.get("output_tokens", 0)
        raise NoProviderError(f"provider {m.provider} not callable")

    def _estimate_tokens(self, req: LLMRequest, text: str) -> tuple[int, int]:
        tin = sum(len(m.content) for m in req.messages) // 4
        return tin, len(text) // 4

    def _account(self, resp: LLMResponse, req: LLMRequest):
        u = self.usage.setdefault(resp.model, {"in": 0, "out": 0, "cost": 0.0, "calls": 0})
        u["in"] += resp.tokens_in
        u["out"] += resp.tokens_out
        u["cost"] += resp.cost_usd
        u["calls"] += 1
        self.total_cost += resp.cost_usd
        self.total_tokens["input"] += resp.tokens_in
        self.total_tokens["output"] += resp.tokens_out
        self.call_log.append({
            "id": new_id("llm"), "ts": time.time(), "model": resp.model,
            "purpose": req.purpose, "tokens_in": resp.tokens_in,
            "tokens_out": resp.tokens_out, "cost_usd": round(resp.cost_usd, 6),
            "latency_s": round(resp.latency_s, 3), "scripted": resp.scripted,
        })


def extract_json(text: str) -> dict:
    """Robustly pull the first JSON object out of a model response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    start = -1
    raise ModelError("no valid JSON object in model response")
