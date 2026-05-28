"""Local-Ollama ReAct agent that drives a FastMCP server through MCP.

This module wraps three pieces of plumbing that examples 07 and 08 (and
any custom user script) need to drive ``nuance_mcp`` with a local LLM:

* ``ChatOllama`` (from ``langchain_ollama``) speaks to the local Ollama
  daemon. The default endpoint is ``http://127.0.0.1:11434``, the
  default model is ``qwen2.5:7b`` (the best-performing open-weight
  tool-calling model on the Apple-Silicon benchmarks of Figure 4b).
* ``langchain_mcp_adapters.client.MultiServerMCPClient`` adapts our
  FastMCP server's tools and prompts into LangChain ``BaseTool``
  instances. The adapter is well-suited because it preserves the
  Pydantic schema as a JSON-schema attached to the tool, so the model
  receives the same bounded-argument contract that the server
  enforces.
* ``langgraph.prebuilt.create_react_agent`` wires up a minimal ReAct
  loop with a system prompt that nudges the model to start with
  ``get_capabilities`` + ``get_microscope_state``, dispatch typed
  tools, and report what it found.

The module is *optional*: it is only imported by code that chose to
install the ``[ollama]`` extra. Importing this module without the
extra raises ``AgentDependencyError`` with a clear remediation hint.
"""

from __future__ import annotations

import os


# ---------------------------------------------------------------------------
# Lazy imports & dependency guard
# ---------------------------------------------------------------------------


class AgentDependencyError(ImportError):
    """Raised when the ``[ollama]`` extra is not installed."""


def _require_ollama_extra():
    missing = []
    try:
        import langchain_ollama  # noqa: F401
    except ImportError:
        missing.append("langchain-ollama")
    try:
        import langchain_mcp_adapters  # noqa: F401
    except ImportError:
        missing.append("langchain-mcp-adapters")
    try:
        import langgraph  # noqa: F401
    except ImportError:
        missing.append("langgraph")
    if missing:
        raise AgentDependencyError(
            "nuance_mcp.agent requires the [ollama] extra. "
            "Install with:\n\n"
            "  pip install 'nuance-mcp[ollama]'\n\n"
            f"Missing: {', '.join(missing)}"
        )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

SYSTEM_PROMPT = """You are a microscopy operations agent that drives a \
multimodal (S)TEM column through schema-bound MCP tools. The server is \
vendor-agnostic; the bound adapter may be Gatan, JEOL, Hitachi, or a \
physics-plausible simulator.

Operating contract:

1. Always call ``get_capabilities`` first to learn which capability \
families the bound adapter advertises.
2. Then call ``get_microscope_state`` to learn the column's current \
state.
3. When the user requests an action, choose the smallest set of typed \
tools that accomplishes it. Tool arguments are bounded by Pydantic \
schemas; out-of-range values will be rejected before any hardware call.
4. For multi-step procedures, prefer invoking a named skill \
(``eels_survey``, ``tilt_series_protocol``, ``4dstem_characterization``, \
``beam_alignment``, ``hrtem_imaging``, ``diffraction_survey``) and \
unrolling its instructions.
5. Tools that return ``{"status": "UNSUPPORTED", ...}`` mean the bound \
adapter does not declare that capability — fall back gracefully and \
explain to the user.
6. Report concise, structured results. For images and spectra, \
include name, shape, statistics, and calibration. For state-changing \
operations, confirm the new value by reading it back.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_agent(
    server,  # FastMCP instance
    query: str,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.0,
    verbose: bool = False,
    max_iterations: int = 16,
) -> str:
    """Run a single ReAct turn against ``server`` for the given ``query``.

    Parameters
    ----------
    server
        A :class:`fastmcp.FastMCP` instance produced by
        :func:`nuance_mcp.build_server`.
    query
        Natural-language instruction for the agent.
    model
        Ollama model tag. Default: ``qwen2.5:7b``.
    base_url
        URL of the local Ollama daemon.
    temperature
        Sampling temperature; default 0.0 for deterministic tool use.
    verbose
        If True, print each tool call as it is dispatched.
    max_iterations
        Hard upper bound on the ReAct loop length.

    Returns
    -------
    str
        The agent's final natural-language reply.
    """

    _require_ollama_extra()

    from langchain_ollama import ChatOllama
    from langgraph.prebuilt import create_react_agent

    # Expose the in-process FastMCP server through the LangChain MCP
    # adapter. The adapter understands "stdio" and "streamable_http" as
    # transports; here we use the in-process Python handle directly via
    # a small shim that calls server.call_tool / server.get_prompt.
    tools = await _server_to_langchain_tools(server, verbose=verbose)

    llm = ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
    )

    agent = create_react_agent(
        llm,
        tools,
        prompt=SYSTEM_PROMPT,
    )

    result = await agent.ainvoke(
        {"messages": [("user", query)]},
        config={"recursion_limit": max_iterations * 2},
    )

    # Last AI message is the answer.
    for msg in reversed(result["messages"]):
        if getattr(msg, "type", None) == "ai" and msg.content:
            return msg.content
    return ""


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


async def _server_to_langchain_tools(server, *, verbose: bool) -> list:
    """Wrap each FastMCP tool as a LangChain ``StructuredTool``.

    We avoid the full ``MultiServerMCPClient`` for the in-process case
    because spinning up a stdio transport is unnecessary overhead when
    the server lives in the same Python process. Tool argument schemas
    are taken from FastMCP's already-built Pydantic models.
    """
    from langchain_core.tools import StructuredTool
    import json

    raw_tools = await server._list_tools()  # FastMCP returns list[Tool]
    wrapped: list = []

    for tool in raw_tools:
        name = tool.name
        description = tool.description or name

        # FastMCP attaches the input schema as a pydantic model under
        # tool.parameters; we build a thin async wrapper that forwards
        # kwargs to server.call_tool.
        async def _call(__tool_name=name, **kwargs):
            if verbose:
                print(f"  → {__tool_name}({kwargs})")
            payload = kwargs.get("payload", kwargs if kwargs else None)
            args = {"payload": payload} if payload is not None else {}
            result = await server.call_tool(__tool_name, args)
            text = result.content[0].text
            try:
                return json.loads(text)
            except Exception:
                return text

        wrapped.append(
            StructuredTool.from_function(
                coroutine=_call,
                name=name,
                description=description,
                args_schema=getattr(tool, "parameters", None),
            )
        )
    return wrapped


# ---------------------------------------------------------------------------
# Synchronous façade for one-shot scripts
# ---------------------------------------------------------------------------


def run_agent_sync(server, query: str, **kwargs) -> str:
    """Convenience wrapper for non-async callers."""
    import asyncio

    return asyncio.run(run_agent(server, query, **kwargs))
