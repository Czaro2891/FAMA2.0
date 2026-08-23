"""Tools & Sandbox (sec. 20, 36).

Policy: DENY BY DEFAULT.  An agent receives exactly the tools its step
requires; everything else is inaccessible.

The sandbox is a *best-effort local* isolation layer: rlimits, scrubbed
environment, workspace-scoped filesystem, wall-clock timeout.  It does NOT
provide kernel-level network isolation in this environment (unshare -n is
unavailable), which is reported honestly in the sandbox report and by
governance risk flags.  Production deployments should wrap steps in a
container/VM sandbox.
"""
from __future__ import annotations

import os
import resource
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .core import new_id, now_utc



# some environments (corporate/sandbox egress proxies) MITM TLS with a CA
# that certifi does not know; the system store does. Prefer the system store.
import ssl as _ssl
try:
    _SSL_CTX = _ssl.create_default_context()
except Exception:  # pragma: no cover
    _SSL_CTX = True

class ToolError(Exception):
    pass


@dataclass
class ToolResult:
    ok: bool
    tool: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_s: float = 0.0
    artifacts: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def head(self, n: int = 4000) -> str:
        out = (self.stdout or "").strip()
        err = (self.stderr or "").strip()
        txt = out + ("\n[stderr]\n" + err if err else "")
        return txt[-n:]

    def to_dict(self):
        return {"ok": self.ok, "tool": self.tool, "exit_code": self.exit_code,
                "duration_s": round(self.duration_s, 3), "artifacts": self.artifacts,
                "meta": self.meta, "stdout_tail": (self.stdout or "")[-1500:],
                "stderr_tail": (self.stderr or "")[-800:]}


@dataclass
class SandboxReport:
    run_id: str
    fs_scope: str = "workspace temp dir"
    network_isolated: bool = False
    cpu_seconds: float = 15.0
    memory_mb: int = 512
    wall_timeout_s: float = 90.0
    env_scrubbed: bool = True

    def to_dict(self):
        return {"run_id": self.run_id, "fs_scope": self.fs_scope,
                "network_isolated": self.network_isolated,
                "cpu_seconds": self.cpu_seconds, "memory_mb": self.memory_mb,
                "wall_timeout_s": self.wall_timeout_s,
                "env_scrubbed": self.env_scrubbed,
                "warning": None if self.network_isolated else
                           "kernel network isolation unavailable here; subprocess could reach the network"}


def _limits(report: SandboxReport):
    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (int(report.cpu_seconds),) * 2)
        resource.setrlimit(resource.RLIMIT_AS, (report.memory_mb * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_NOFILE, (128,) * 2)
    return apply


def _clean_env(home: str) -> dict:
    keep = {}
    for k in ("PATH", "LANG", "LC_ALL", "TMPDIR", "SYSTEM_ROOT"):
        if k in os.environ:
            keep[k] = os.environ[k]
    keep["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")
    keep["HOME"] = home
    keep["PYTHONDONTWRITEBYTECODE"] = "1"
    keep["PYTHONHASHSEED"] = "0"
    for k in list(os.environ):
        if "KEY" in k or "TOKEN" in k or "SECRET" in k or "PASSWORD" in k:
            keep.pop(k, None)
    return keep


class Sandbox:
    """Controlled execution environment for untrusted/dynamic artifacts."""

    def __init__(self, base_dir: str | None = None):
        self.base = Path(base_dir or tempfile.mkdtemp(prefix="fama-sbx-"))
        self.base.mkdir(parents=True, exist_ok=True)
        self.reports: list[SandboxReport] = []

    def workspace(self, name: str = "work") -> Path:
        p = self.base / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def run(self, argv: list[str], *, cwd: Path | None = None,
            timeout: float | None = None, report: SandboxReport | None = None) -> ToolResult:
        rep = report or SandboxReport(run_id=new_id("sbx"))
        self.reports.append(rep)
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                argv, cwd=str(cwd or self.base), env=_clean_env(str(self.base)),
                capture_output=True, text=True,
                timeout=timeout or rep.wall_timeout_s,
                preexec_fn=_limits(rep))
            ok = proc.returncode == 0
            return ToolResult(ok, argv[0] if argv else "?", proc.stdout, proc.stderr,
                              proc.returncode, time.monotonic() - t0, meta={"sandbox": rep.to_dict()})
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
            return ToolResult(False, argv[0] if argv else "?", out, err or "sandbox wall-clock timeout",
                              None, time.monotonic() - t0, meta={"sandbox": rep.to_dict(), "timeout": True})
        except Exception as e:
            return ToolResult(False, argv[0] if argv else "?", "", str(e), None,
                              time.monotonic() - t0, meta={"sandbox": rep.to_dict()})

    def cleanup(self):
        shutil.rmtree(self.base, ignore_errors=True)


# ---------------------------------------------------------------- tool registry

TOOL_CATALOG: dict[str, dict] = {
    "fs_read":       {"desc": "read files inside the task workspace", "risk": "low"},
    "fs_write":      {"desc": "write files inside the task workspace", "risk": "low"},
    "fs_list":       {"desc": "list task workspace", "risk": "low"},
    "python_run":    {"desc": "execute Python in the sandbox", "risk": "medium"},
    "test_run":      {"desc": "run the test suite in the sandbox", "risk": "low"},
    "git":           {"desc": "git status/diff inside workspace", "risk": "low"},
    "benchmark":     {"desc": "measure runtime of a function/program", "risk": "low"},
    "mutation":      {"desc": "generate mutants and measure test kill rate", "risk": "medium"},
    "web_fetch":     {"desc": "fetch a public URL (research)", "risk": "medium"},
    "web_search":    {"desc": "search the web (research)", "risk": "medium"},
    "web_reader":    {"desc": "read any page as clean text (Jina Reader; Agent Reach channel)", "risk": "medium"},
    "gh_api":        {"desc": "GitHub API: repos, issues, code search (Agent Reach channel)", "risk": "low"},
    "rss_read":      {"desc": "read an RSS/Atom feed (Agent Reach channel)", "risk": "medium"},
    "youtube_transcript": {"desc": "fetch video info/subtitles via yt-dlp (Agent Reach channel)", "risk": "medium"},
}


class ToolRouter:
    """Dynamic tool assignment with deny-by-default enforcement."""

    def __init__(self, sandbox: Sandbox, workspace: Path, governance):
        self.sandbox = sandbox
        self.workspace = Path(workspace).resolve()
        self.governance = governance
        self.grants: dict[str, list[str]] = {}   # step_id -> allowed tools

    def grant(self, step_id: str, tools: list[str]):
        allowed = [t for t in tools if t in TOOL_CATALOG]
        self.grants[step_id] = sorted(set(allowed))

    def check(self, step_id: str, tool: str):
        if tool not in TOOL_CATALOG:
            raise ToolError(f"unknown tool: {tool}")
        if tool not in self.grants.get(step_id, []):
            raise ToolError(f"tool '{tool}' not granted to step {step_id} (deny-by-default)")
        self.governance.check_tool(tool)

    # ---------------------------------------------------------- tools

    def _safe_path(self, rel: str) -> Path:
        p = (self.workspace / rel).resolve()
        if not str(p).startswith(str(self.workspace.resolve())):
            raise ToolError(f"path escapes workspace: {rel}")
        return p

    def call(self, step_id: str, tool: str, **kwargs) -> ToolResult:
        self.check(step_id, tool)
        fn = getattr(self, f"_{tool}")
        return fn(**kwargs)

    def _fs_list(self, sub: str = ".") -> ToolResult:
        p = self._safe_path(sub)
        rows = []
        for f in sorted(p.rglob("*")):
            if any(part.startswith(".") and part != "." for part in f.parts[len(self.workspace.parts):]):
                continue
            if f.is_file():
                rows.append(str(f.relative_to(self.workspace)))
        return ToolResult(True, "fs_list", "\n".join(rows) or "(empty)", meta={"files": len(rows)})

    def _fs_read(self, path: str) -> ToolResult:
        p = self._safe_path(path)
        if not p.exists():
            return ToolResult(False, "fs_read", "", f"no such file: {path}")
        return ToolResult(True, "fs_read", p.read_text(errors="replace")[:200_000])

    def _fs_write(self, path: str, content: str) -> ToolResult:
        p = self._safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return ToolResult(True, "fs_write", f"wrote {len(content)} bytes to {path}",
                          artifacts=[path])

    def _python_run(self, file: str, args: list[str] | None = None) -> ToolResult:
        return self.sandbox.run(["python3", file] + (args or []), cwd=self.workspace)

    def _test_run(self, target: str = ".", max_files: int = 50) -> ToolResult:
        argv = ["python3", "-m", "pytest", "-x", "-q", target]
        res = self.sandbox.run(argv, cwd=self.workspace)
        if res.exit_code is not None and "No module named pytest" in (res.stderr or ""):
            argv = ["python3", "-m", "unittest", "discover", "-s", target, "-v"]
            res = self.sandbox.run(argv, cwd=self.workspace)
        return res

    def _git(self, args: list[str]) -> ToolResult:
        allowed_first = {"status", "diff", "log"}
        if args and args[0] not in allowed_first:
            return ToolResult(False, "git", "", f"git {args[0]} not permitted (read-only)")
        return self.sandbox.run(["git"] + args, cwd=self.workspace)

    def _benchmark(self, file: str, function: str, repeats: int = 5) -> ToolResult:
        script = (
            "import importlib.util, json, time, statistics\n"
            f"spec = importlib.util.spec_from_file_location('bm_target', {file!r})\n"
            "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
            f"fn = getattr(mod, {function!r})\n"
            "data=[]\n"
            "import random\n"
            "random.seed(42)\n"
            "for r in range(%d):\n"
            "    args = getattr(mod, 'bench_args', lambda: (list(range(2000)),))( )\n"
            "    t0=time.perf_counter(); fn(*args); data.append((time.perf_counter()-t0)*1000)\n"
            "print(json.dumps({'runs_ms': data, 'median_ms': statistics.median(data), 'mean_ms': statistics.mean(data)}))\n"
        ) % max(1, min(repeats, 20))
        (self.workspace / "__fama_bench__.py").write_text(script)
        res = self.sandbox.run(["python3", "__fama_bench__.py"], cwd=self.workspace)
        return res

    def _web_fetch(self, url: str) -> ToolResult:
        if not self.governance.state.allow_network:
            return ToolResult(False, "web_fetch", "", "network egress not permitted by governance")
        try:
            import httpx
            r = httpx.get(url, timeout=20.0, verify=_SSL_CTX, follow_redirects=True,
                          headers={"User-Agent": "FAMA2.0-research/2.0"})
            ok = r.status_code < 400
            return ToolResult(ok, "web_fetch", r.text[:120_000] if ok else "",
                              "" if ok else f"HTTP {r.status_code}",
                              r.status_code, meta={"url": url, "status": r.status_code})
        except Exception as e:
            return ToolResult(False, "web_fetch", "", str(e))

    def _web_search(self, query: str) -> ToolResult:
        if not self.governance.state.allow_network:
            return ToolResult(False, "web_search", "", "network egress not permitted by governance")
        api_key = os.environ.get("BRAVE_API_KEY") or os.environ.get("TAVILY_API_KEY")
        try:
            import httpx
            if os.environ.get("TAVILY_API_KEY"):
                r = httpx.post("https://api.tavily.com/search", timeout=20.0, json={
                    "api_key": os.environ["TAVILY_API_KEY"], "query": query, "max_results": 8})
                r.raise_for_status()
                data = r.json()
                lines = [f"{i.get('title','')}\n{i.get('url','')}\n{i.get('content','')}"
                         for i in data.get("results", [])]
                return ToolResult(True, "web_search", "\n\n".join(lines), meta={"engine": "tavily"})
            if api_key:
                r = httpx.get("https://api.search.brave.com/res/v1/web/search", timeout=20.0,
                              headers={"X-Subscription-Token": api_key,
                                       "Accept": "application/json"},
                              params={"q": query, "count": 8})
                r.raise_for_status()
                items = r.json().get("web", {}).get("results", [])
                lines = [f"{i.get('title','')}\n{i.get('url','')}\n{i.get('description','')}"
                         for i in items]
                return ToolResult(True, "web_search", "\n\n".join(lines), meta={"engine": "brave"})
            return ToolResult(False, "web_search", "",
                              "no search engine configured (set TAVILY_API_KEY or BRAVE_API_KEY)")
        except Exception as e:
            return ToolResult(False, "web_search", "", str(e))

    # ------------------------------------------------- Agent Reach channels

    def _net(self):
        if not self.governance.state.allow_network:
            return ToolResult(False, "net", "",
                              "network egress not permitted by governance")
        return None

    def _web_reader(self, url: str) -> ToolResult:
        blocked = self._net()
        if blocked:
            return ToolResult(False, "web_reader", "", blocked.stderr)
        try:
            import httpx
            # direct fetch first; fall back to Jina Reader for JS-heavy pages
            for target in (url, f"https://r.jina.ai/{url}"):
                try:
                    r = httpx.get(target, timeout=25.0, follow_redirects=True, verify=_SSL_CTX,
                                  headers={"User-Agent": "FAMA2.0-research/2.0"})
                    if r.status_code < 400 and r.text.strip():
                        return ToolResult(True, "web_reader", r.text[:120_000],
                                          meta={"via": "direct" if target == url else "jina"})
                except Exception:
                    continue
            return ToolResult(False, "web_reader", "", f"could not read {url}")
        except Exception as e:
            return ToolResult(False, "web_reader", "", str(e))

    def _gh_api(self, path: str) -> ToolResult:
        blocked = self._net()
        if blocked:
            return ToolResult(False, "gh_api", "", blocked.stderr)
        try:
            import httpx, json as _json
            if not path.startswith("/"):
                path = "/" + path
            r = httpx.get(f"https://api.github.com{path}", timeout=20.0, verify=_SSL_CTX,
                          headers={"Accept": "application/vnd.github+json",
                                   "User-Agent": "FAMA2.0-research/2.0"})
            ok = r.status_code < 400
            return ToolResult(ok, "gh_api", r.text[:150_000] if ok else "",
                              "" if ok else f"HTTP {r.status_code}",
                              r.status_code, meta={"path": path})
        except Exception as e:
            return ToolResult(False, "gh_api", "", str(e))

    def _rss_read(self, url: str, limit: int = 15) -> ToolResult:
        blocked = self._net()
        if blocked:
            return ToolResult(False, "rss_read", "", blocked.stderr)
        try:
            import httpx
            import xml.etree.ElementTree as ET
            r = httpx.get(url, timeout=20.0, verify=_SSL_CTX, follow_redirects=True,
                          headers={"User-Agent": "FAMA2.0-research/2.0"})
            r.raise_for_status()
            items = parse_feed(r.text, limit)
            text = "\n\n".join(f"{i['title']}\n{i['link']}" for i in items)
            return ToolResult(True, "rss_read", text or "(empty feed)",
                              meta={"items": len(items)})
        except Exception as e:
            return ToolResult(False, "rss_read", "", str(e))

    def _youtube_transcript(self, url: str) -> ToolResult:
        blocked = self._net()
        if blocked:
            return ToolResult(False, "youtube_transcript", "", blocked.stderr)
        import shutil
        ytdlp = shutil.which("yt-dlp") or str(Path.home() / ".agent-reach-venv/bin/yt-dlp")
        if not Path(ytdlp).exists():
            return ToolResult(False, "youtube_transcript", "",
                              "yt-dlp not installed (Agent Reach provides it)")
        res = self.sandbox.run([ytdlp, "--skip-download", "--write-auto-sub",
                                "--sub-langs", "en.*,pl", "--sub-format", "vtt/srt",
                                "-o", "__fama_yt__", url], cwd=self.workspace)
        subs = sorted(self.workspace.glob("__fama_yt__*.vtt")) + \
            sorted(self.workspace.glob("__fama_yt__*.srt"))
        text = ""
        if subs:
            raw = subs[0].read_text(errors="replace")
            lines = [l.strip() for l in raw.splitlines()
                     if l.strip() and not l.strip().isdigit()
                     and "-->" not in l and not l.startswith(("WEBVTT", "Kind:", "Language:"))]
            text = "\n".join(dict.fromkeys(lines))[:100_000]
        return ToolResult(res.ok or bool(text), "youtube_transcript", text,
                          "" if text else res.head(300),
                          meta={"video_info": res.ok and res.stdout or ""})


def parse_feed(xml_text: str, limit: int = 15) -> list[dict]:
    """Parse RSS 2.0 / Atom into items (title, link) — stdlib only."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_text.encode() if isinstance(xml_text, str) else xml_text)
    out = []
    for item in root.iter():
        tag = item.tag.split("}")[-1]
        if tag in ("item", "entry"):
            title = link = ""
            for ch in item:
                ctag = ch.tag.split("}")[-1]
                if ctag == "title" and ch.text:
                    title = ch.text.strip()
                if ctag == "link" and not link:
                    link = (ch.get("href") or ch.text or "").strip()
            if title or link:
                out.append({"title": title, "link": link})
            if len(out) >= limit:
                break
    return out
