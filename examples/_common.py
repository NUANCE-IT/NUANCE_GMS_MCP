"""Helpers shared by every example.

* ``--adapter NAME`` selects the backend (``gatan``, ``jeol``, ``simulator``).
  Falls back to the simulator if the requested adapter cannot connect.
* ``call(server, tool, **fields)`` is a small wrapper that calls a typed
  tool with the ``payload=`` argument FastMCP expects and returns the
  parsed JSON dict from the first content frame.

Run any example with::

    python examples/01_basic_query.py                          # simulator
    python examples/01_basic_query.py --adapter gatan          # GMS
    python examples/01_basic_query.py --adapter jeol           # PyJEM
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from nuance_mcp import build_server


def parse_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--adapter", default="simulator",
                   choices=["simulator", "gatan", "jeol", "hitachi"])
    p.add_argument("--mode", default=None,
                   help="Adapter-specific mode (e.g. JEOL online/offline).")
    return p.parse_args()


def make_server(args: argparse.Namespace):
    kwargs = {}
    if args.mode is not None:
        kwargs["mode"] = args.mode
    try:
        return build_server(args.adapter, adapter_kwargs=kwargs)
    except Exception as exc:
        print(f"[examples] {args.adapter!r} adapter failed: {exc}",
              file=sys.stderr)
        print(f"[examples] falling back to simulator.", file=sys.stderr)
        return build_server("simulator")


async def call(server, tool: str, **fields) -> dict[str, Any]:
    """Call ``tool`` and return the JSON-decoded payload."""
    payload = fields.pop("payload", None)
    if payload is None and fields:
        payload = fields
    args = {"payload": payload} if payload is not None else {}
    result = await server.call_tool(tool, args)
    text = result.content[0].text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def banner(title: str) -> None:
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


def kv(label: str, value: Any) -> None:
    print(f"  {label:<28}: {value}")
