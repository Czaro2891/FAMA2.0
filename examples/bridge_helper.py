#!/usr/bin/env python3
"""FAMA Bridge Helper — lokalny przekaźnik CORS/PNA dla Twojego LLM.

Problem: World UI (HTTPS, sandbox) nie może odpytać http://localhost —
przeglądarki (Chrome: Private Network Access) wymagają od serwera lokalnego
nagłówka `Access-Control-Allow-Private-Network: true` oraz zgody CORS.
Ollama/LM Studio tego nie wysyłają. Ten helper tak.

Co robi:
  • słucha na 127.0.0.1:8790,
  • odpowiada na preflight OPTIONS (CORS * + Private-Network: true),
  • przekazuje /v1/* do Twojego lokalnego modelu (domyślnie Ollama :11434).

Użycie (na Twoim komputerze, Python 3.9+, zero zależności):
    # terminal 1 — model:
    ollama serve                      # lub inny serwer zgodny z OpenAI API
    # terminal 2 — helper:
    python examples/bridge_helper.py                        # cel: Ollama :11434
    python examples/bridge_helper.py --target http://127.0.0.1:1234   # LM Studio
Potem w World UI: panel Bridge → URL: http://localhost:8790/v1 → „Połącz".
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Allow-Private-Network": "true",   # kluczowe dla Chrome PNA
    "Access-Control-Max-Age": "86400",
}


def make_handler(target: str):
    class Handler(BaseHTTPRequestHandler):
        def _cors(self):
            for k, v in CORS_HEADERS.items():
                self.send_header(k, v)

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _proxy(self, method: str):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else None
            url = target + self.path
            req = urllib.request.Request(url, data=body, method=method)
            for h in ("Content-Type", "Authorization"):
                if self.headers.get(h):
                    req.add_header(h, self.headers[h])
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    data, code = r.read(), r.status
                    ctype = r.headers.get("Content-Type", "application/json")
            except urllib.error.HTTPError as e:
                data, code = e.read(), e.code
                ctype = e.headers.get("Content-Type", "application/json")
            except Exception as e:
                data = json.dumps({"error": {
                    "message": f"bridge_helper nie może osiągnąć {target}: {e}"
                }}).encode()
                code, ctype = 502, "application/json"
            self.send_response(code)
            self._cors()
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._proxy("GET")

        def do_POST(self):
            self._proxy("POST")

        def log_message(self, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description="FAMA Bridge Helper (CORS/PNA relay)")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--target", default="http://127.0.0.1:11434",
                    help="bazowy adres lokalnego modelu (bez /v1)")
    args = ap.parse_args()
    target = args.target.rstrip("/")
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(target))
    print(f"FAMA Bridge Helper → http://{args.host}:{args.port}  (cel: {target})")
    print(f"W World UI (panel Bridge) wpisz URL: http://localhost:{args.port}/v1")
    print("Nagłówki CORS + Private-Network aktywne. Ctrl+C aby zatrzymać.")
    srv.serve_forever()


if __name__ == "__main__":
    main()
