"""FastMCP server with vendor-agnostic adapter selection.

Usage::

    nuance-mcp serve --adapter gatan
    nuance-mcp serve --adapter jeol --mode offline
    nuance-mcp serve --adapter simulator         # default
    nuance-mcp serve --adapter gatan --transport http --port 8000

The server is a thin glue layer: it instantiates the chosen adapter,
registers the generic typed tools (:mod:`nuance_mcp.tools`) against it,
registers the vendor-portable skills (:mod:`nuance_mcp.core.skills`),
and starts the chosen MCP transport.
"""

from __future__ import annotations

import os
from typing import Optional

from fastmcp import FastMCP

from .core import register_skills, MicroscopeAdapter
from .tools import register_tools
from .adapters import load_adapter, available_adapters


INSTRUCTIONS_TEMPLATE = """\
NUANCE-MCP — vendor-agnostic LLM control of multimodal electron microscopy.

This server is bound to a {vendor} ({model}) adapter. The tool surface is
identical regardless of vendor; capability availability is exposed through
``get_capabilities``. Always call ``get_microscope_state`` first.

Capabilities declared by this adapter:
{caps}
"""


def build_server(
    adapter_name: str = "simulator",
    adapter_kwargs: Optional[dict] = None,
) -> FastMCP:
    """Instantiate the adapter, register tools + skills, return the server."""

    adapter_cls = load_adapter(adapter_name)
    adapter: MicroscopeAdapter = adapter_cls(**(adapter_kwargs or {}))
    adapter.open()

    caps = ", ".join(sorted(c.value for c in adapter.capabilities)) or "(none)"
    mcp = FastMCP(
        name="nuance_mcp",
        instructions=INSTRUCTIONS_TEMPLATE.format(
            vendor=adapter.vendor,
            model=adapter.model,
            caps=caps,
        ),
    )

    register_tools(mcp, adapter)
    register_skills(mcp)
    return mcp


def run_server(
    adapter_name: str = "simulator",
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    adapter_kwargs: Optional[dict] = None,
) -> None:
    mcp = build_server(adapter_name, adapter_kwargs)
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="http", host=host, port=port)
