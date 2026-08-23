import pytest

from fama.governance import Governance
from fama.tools import Sandbox, ToolError, ToolRouter


@pytest.fixture()
def router(tmp_path):
    sbx = Sandbox(str(tmp_path / "sbx"))
    ws = sbx.workspace("ws")
    gov = Governance()
    r = ToolRouter(sbx, ws, gov)
    yield r
    sbx.cleanup()


def test_deny_by_default(router):
    with pytest.raises(ToolError):
        router.call("step-1", "fs_write", path="x.txt", content="hi")


def test_grant_enables_only_granted(router):
    router.grant("step-1", ["fs_read"])
    res = router.call("step-1", "fs_read", path="anything.txt")
    assert res.ok is False  # file doesn't exist, but the CALL was permitted
    with pytest.raises(ToolError):
        router.call("step-1", "fs_write", path="x.txt", content="hi")


def test_unknown_tool_rejected(router):
    with pytest.raises(ToolError):
        router.call("step-1", "rm_rf_everything")


def test_path_escape_blocked(router):
    router.grant("step-1", ["fs_read", "fs_write"])
    with pytest.raises(ToolError):
        router.call("step-1", "fs_read", path="../../etc/passwd")


def test_fs_write_read_roundtrip(router):
    router.grant("step-1", ["fs_write", "fs_read", "fs_list"])
    router.call("step-1", "fs_write", path="sub/a.py", content="print('hi')")
    res = router.call("step-1", "fs_read", path="sub/a.py")
    assert "print" in res.stdout


def test_python_run_in_sandbox(router, tmp_path):
    router.grant("step-1", ["fs_write", "python_run"])
    router.call("step-1", "fs_write", path="t.py", content="print('sandbox-ok')")
    res = router.call("step-1", "python_run", file="t.py")
    assert res.ok and "sandbox-ok" in res.stdout


def test_sandbox_env_scrubbed_of_secrets(router):
    import os
    os.environ["FAMA_TEST_SECRET"] = "supersecret"
    try:
        router.grant("step-1", ["fs_write", "python_run"])
        router.call("step-1", "fs_write", path="env.py",
                    content="import os; print(os.environ.get('FAMA_TEST_SECRET', 'SCRUBBED'))")
        res = router.call("step-1", "python_run", file="env.py")
        assert "SCRUBBED" in res.stdout and "supersecret" not in res.stdout
    finally:
        del os.environ["FAMA_TEST_SECRET"]


def test_sandbox_wall_timeout(tmp_path):
    from fama.tools import Sandbox
    sbx = Sandbox(str(tmp_path / "sbx"))
    (sbx.workspace("ws") / "loop.py").write_text("while True: pass")
    res = sbx.run(["python3", "loop.py"], cwd=sbx.workspace("ws"), timeout=3)
    assert res.ok is False
    assert res.meta.get("timeout") is True or "timeout" in (res.stderr or "").lower()
    sbx.cleanup()


def test_sandbox_cpu_limit(tmp_path):
    import time
    from fama.tools import Sandbox, SandboxReport
    sbx = Sandbox(str(tmp_path / "sbx"))
    rep = SandboxReport(run_id="t", cpu_seconds=2, wall_timeout_s=30)
    (sbx.workspace("ws") / "loop.py").write_text("while True: pass")
    t0 = time.time()
    res = sbx.run(["python3", "loop.py"], cwd=sbx.workspace("ws"), report=rep)
    assert res.ok is False
    assert time.time() - t0 < 20
    sbx.cleanup()


def test_git_readonly(router):
    router.grant("step-1", ["git"])
    res = router.call("step-1", "git", args=["status"])
    res2 = router.call("step-1", "git", args=["push", "origin", "main"])
    assert res2.ok is False  # write operations not permitted


def test_network_denied_by_default(router):
    router.grant("step-1", ["web_fetch"])
    # deny-by-default is enforced by governance before the tool even runs
    with pytest.raises(PermissionError):
        router.call("step-1", "web_fetch", url="https://example.com")
    # and the tool itself double-checks when called directly
    res = router._web_fetch(url="https://example.com")
    assert res.ok is False and "not permitted" in res.stderr
