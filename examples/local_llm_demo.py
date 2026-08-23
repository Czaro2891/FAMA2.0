"""Lokalny serwer DEMO zgodny z OpenAI API — do testowania FAMA 2.0 bez modelu.

⚠️ UCZCIWOŚĆ (spec §48): to NIE jest model AI. Zwraca deterministyczne
odpowiedzi fixture dla jednego zadania (funkcja moving_average), aby:
  • przetestować integrację FAMA z lokalnym endpointem (Ollama/LM Studio/…),
  • sprawdzić instalację offline.

Użycie:
    python examples/local_llm_demo.py                # port 8788
    python examples/local_llm_demo.py --port 9001

Następnie w drugim terminalu:
    OPENAI_BASE_URL=http://127.0.0.1:8788/v1 python -m fama run \
        "Napisz prostą funkcję Python moving_average(data, window) w pliku
         moving_average.py, liczącą średnią kroczącą. Bez zewnętrznych bibliotek." --yes
"""
from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fama.llm import LLMMessage, LLMRequest, ScriptedProvider
from fama.scenarios import SCENARIOS, scenario_fixtures

TASK_SUBSTR = "moving_average"
NOTE = ("This is a deterministic DEMO server, not an AI model. "
        "It only answers the scripted moving_average task.")


class DemoHandler(BaseHTTPRequestHandler):
    provider = ScriptedProvider(scenario_fixtures(SCENARIOS["simple-function"]))

    def _send(self, code: int, payload: dict):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [
                {"id": "demo-mini-3b", "object": "model", "owned_by": "fama-demo"},
                {"id": "demo-coder-7b", "object": "model", "owned_by": "fama-demo"},
            ]})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": {"message": "not found"}})
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        prompt = "\n".join(m.get("content", "") for m in body.get("messages", []))
        req = LLMRequest(messages=[LLMMessage("user", prompt)],
                         model=body.get("model"))
        content = self.provider.respond(req)
        if TASK_SUBSTR not in prompt:
            content = json.dumps({"note": NOTE})
        self._send(200, {
            "id": "chatcmpl-fama-demo", "object": "chat.completion",
            "model": body.get("model", "demo-mini-3b"),
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(prompt) // 4,
                      "completion_tokens": len(content) // 4,
                      "total_tokens": (len(prompt) + len(content)) // 4},
        })

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8788)
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"FAMA demo server (NOT an AI model) → http://{args.host}:{args.port}/v1")
    print(f"  models: demo-mini-3b, demo-coder-7b · scripted task: moving_average")
    srv.serve_forever()


if __name__ == "__main__":
    main()
