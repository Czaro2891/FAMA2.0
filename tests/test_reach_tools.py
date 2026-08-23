"""Agent Reach channel tools integration tests.

Agent Reach (https://github.com/Panniantong/agent-reach) installs upstream
tools; FAMA routes them as capabilities. These tests cover catalog entries,
governance gating, feed parsing and (when the network allows) a live GitHub
API call.
"""
import pytest

from fama.governance import Governance
from fama.tools import Sandbox, ToolRouter, parse_feed

RSS_SAMPLE = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>t</title>
<item><title>Alpha</title><link>https://a.example/1</link></item>
<item><title>Beta</title><link>https://a.example/2</link></item>
</channel></rss>"""

ATOM_SAMPLE = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>Gamma</title><link href="https://b.example/1"/></entry>
</feed>"""


@pytest.fixture()
def router(tmp_path):
    sbx = Sandbox(str(tmp_path))
    r = ToolRouter(sbx, sbx.workspace("ws"), Governance())
    yield r
    sbx.cleanup()


def test_reach_tools_in_catalog():
    from fama.tools import TOOL_CATALOG
    for t in ("web_reader", "gh_api", "rss_read", "youtube_transcript"):
        assert t in TOOL_CATALOG


def test_reach_tools_deny_by_default(router):
    router.governance.state.allow_network = False
    cases = {
        "web_reader": {"url": "https://example.com"},
        "gh_api": {"path": "/repos/x/y"},
        "rss_read": {"url": "https://example.com/feed"},
        "youtube_transcript": {"url": "https://youtube.com/watch?v=x"},
    }
    for t, kwargs in cases.items():
        router.grant("s1", [t])
        res = router.call("s1", t, **kwargs)
        assert res.ok is False
        assert "not permitted" in res.stderr, f"{t}: {res.stderr}"


def test_parse_feed_rss_and_atom():
    items = parse_feed(RSS_SAMPLE)
    assert items[0]["title"] == "Alpha" and items[1]["link"].endswith("/2")
    atom = parse_feed(ATOM_SAMPLE)
    assert atom[0]["title"] == "Gamma" and atom[0]["link"] == "https://b.example/1"


def test_research_steps_grant_reach_tools():
    from fama.orchestrator import STEP_TOOL_NEEDS
    for t in ("web_reader", "gh_api", "rss_read", "youtube_transcript"):
        assert t in STEP_TOOL_NEEDS["research"], f"{t} missing in research"
    for t in ("web_reader", "gh_api"):
        assert t in STEP_TOOL_NEEDS["validate_sources"], f"{t} missing in validate_sources"


def test_gh_api_live_when_network_allows(router):
    """Live check — skipped automatically when GitHub API is unreachable."""
    import ssl
    import httpx
    try:
        probe = httpx.get("https://api.github.com", timeout=10.0,
                          verify=ssl.create_default_context())
    except Exception:
        pytest.skip("network egress to api.github.com unavailable")
    router.governance.state.allow_network = True
    router.grant("s1", ["gh_api"])
    res = router.call("s1", "gh_api", path="/repos/Panniantong/agent-reach")
    assert res.ok is True
    assert "agent-reach" in res.stdout.lower()
