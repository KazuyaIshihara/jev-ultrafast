"""Offline checks for the MCP result shaping and session bookkeeping."""

from jev_ultrafast import mcp_server


class FakeBrowser:
    def observe(self, screenshot=False):
        return {"url": "https://example.test/after", "title": "After", "text": "hello", "actions": []}

    def close(self):
        pass


class FakeAgent:
    def __init__(self):
        self.browser = FakeBrowser()
        self.screenshots = False
        self.state = {"page": self.browser.observe()}
        self.closed = False

    def snapshot(self):
        return {**self.state, "elements": [{"index": "1", "label": "Go", "role": "button", "operations": ["CLICK"]}]}

    def close(self):
        self.closed = True


def _snapshot(status):
    return {
        "status": status,
        "elapsed_ms": 1234,
        "page": {"url": "https://example.test/x", "title": "X", "text": "t" * 5000},
        "history": [{"step": 1, "operation": "CLICK", "action": "Go", "text": None}],
        "decisions": [{"usage": {"input_tokens": 100}}, {"usage": {"input_tokens": 50}}],
        "elements": [{"index": str(i), "label": f"e{i}"} for i in range(60)],
    }


def test_done_result_is_compact_and_has_no_elements():
    out = mcp_server._result("abc", None, _snapshot("done"))
    assert out["status"] == "done" and out["jev_calls"] == 2 and out["jev_input_tokens"] == 150
    assert len(out["page_text"]) == mcp_server.TEXT_EXCERPT
    assert "elements" not in out and out["steps"][0]["operation"] == "CLICK"


def test_blocked_result_offers_capped_elements_and_hint():
    out = mcp_server._result("abc", None, _snapshot("blocked"))
    assert len(out["elements"]) == mcp_server.MAX_ELEMENTS and "continue_goal" in out["hint"]


def test_close_session_forgets_agent():
    import anyio

    agent = FakeAgent()
    mcp_server.SESSIONS["s1"] = agent
    out = anyio.run(mcp_server.close_session, "s1")
    assert out["closed"] and agent.closed and "s1" not in mcp_server.SESSIONS
    assert anyio.run(mcp_server.close_session, "s1")["closed"] is False


def test_inspect_unknown_session():
    import anyio

    assert "error" in anyio.run(mcp_server.inspect_session, "nope")
