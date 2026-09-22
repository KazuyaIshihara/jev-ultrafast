"""MCP server (mcp 2.x MCPServer): a reasoning agent plans, Jev executes bounded browser goals on BrowserOS neo.

Tools keep one browser tab per session so a planner can chain goals (log in, then search, then fill).
Model output never becomes selectors or scripts; every action resolves to an observed element.
"""

import os
import uuid

import anyio
from mcp.server.mcpserver import MCPServer

from .agent import Agent

SERVER = MCPServer(
    "jev-neo",
    instructions=(
        "Fast, cheap browser execution on BrowserOS neo. Give act_toward_goal ONE bounded goal that "
        "ends on a visible state (a page open, a result list shown, a form filled). Plan multi-stage "
        "work yourself and chain goals with continue_goal on the same session. Verify outcomes from "
        "the returned url/title/page_text; a 'done' status is the executor's claim, not proof."
    ),
)
SESSIONS: dict[str, Agent] = {}
MAX_ELEMENTS = 40
TEXT_EXCERPT = int(os.environ.get("JEV_MCP_TEXT_EXCERPT", "1500"))


def _result(session_id, agent, snapshot):
    """Compact, planner-facing summary of a run. Elements are included only when the executor stopped short."""
    page = snapshot["page"]
    history = snapshot["history"]
    usage_in = sum(d.get("usage", {}).get("input_tokens", 0) for d in snapshot["decisions"])
    out = {
        "session_id": session_id,
        "status": snapshot["status"],
        "url": page["url"],
        "title": page["title"],
        "elapsed_ms": snapshot["elapsed_ms"],
        "steps": [
            {"step": h["step"], "operation": h["operation"], "element": h["action"], "text": h["text"]}
            for h in history
        ],
        "jev_calls": len(snapshot["decisions"]),
        "jev_input_tokens": usage_in,
        "page_text": page["text"][:TEXT_EXCERPT],
    }
    if snapshot["status"] != "done":
        out["elements"] = snapshot["elements"][:MAX_ELEMENTS]
        out["hint"] = (
            "Executor stopped without DONE. Inspect elements/page_text, then continue_goal with a narrower "
            "goal, or handle the page yourself with BrowserOS neo tools."
        )
    return out


def _run(agent):
    snapshot = None
    for snapshot in agent.run():
        pass
    return snapshot if snapshot is not None else agent.snapshot()


def _start(url, goal):
    session_id = uuid.uuid4().hex[:8]
    agent = Agent(url, goal)
    SESSIONS[session_id] = agent
    return session_id, agent


@SERVER.tool()
async def act_toward_goal(url: str, goal: str) -> dict:
    """Open url in a new BrowserOS neo tab and let Jev pursue ONE bounded goal (clicks, typing, selects, scrolls).

    Returns status done|blocked, the final url/title, executed steps, and a page_text excerpt for verification.
    The tab stays open; reuse its session_id with continue_goal or close it with close_session.
    """
    session_id, agent = await anyio.to_thread.run_sync(_start, url, goal)
    snapshot = await anyio.to_thread.run_sync(_run, agent)
    return _result(session_id, agent, snapshot)


@SERVER.tool()
async def continue_goal(session_id: str, goal: str) -> dict:
    """Pursue a new bounded goal on an existing session's tab, keeping its logged-in state and current page."""
    agent = SESSIONS.get(session_id)
    if agent is None:
        return {"error": f"Unknown session_id {session_id!r}. Start with act_toward_goal."}
    await anyio.to_thread.run_sync(agent.next_goal, goal)
    snapshot = await anyio.to_thread.run_sync(_run, agent)
    return _result(session_id, agent, snapshot)


@SERVER.tool()
async def inspect_session(session_id: str) -> dict:
    """Re-observe a session's current page without acting: url, title, page_text excerpt, and actionable elements."""
    agent = SESSIONS.get(session_id)
    if agent is None:
        return {"error": f"Unknown session_id {session_id!r}."}

    def observe():
        agent.state["page"] = agent.browser.observe(screenshot=False)
        return agent.snapshot()

    snapshot = await anyio.to_thread.run_sync(observe)
    page = snapshot["page"]
    return {
        "session_id": session_id,
        "url": page["url"],
        "title": page["title"],
        "page_text": page["text"][:TEXT_EXCERPT],
        "elements": snapshot["elements"][:MAX_ELEMENTS],
    }


@SERVER.tool()
async def close_session(session_id: str) -> dict:
    """Close a session's tab and forget it."""
    agent = SESSIONS.pop(session_id, None)
    if agent is None:
        return {"closed": False, "reason": "unknown session_id"}
    await anyio.to_thread.run_sync(agent.close)
    return {"closed": True, "open_sessions": len(SESSIONS)}


def main():
    SERVER.run(transport="stdio")


if __name__ == "__main__":
    main()
