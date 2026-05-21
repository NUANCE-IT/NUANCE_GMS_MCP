"""Command-line entry point for nuance-mcp."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .adapters import available_adapters
from .server import run_server


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="nuance-mcp",
        description="Vendor-agnostic MCP server for multimodal electron "
        "microscopy. Built-in adapters: " + ", ".join(available_adapters()) + ".",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Run the MCP server.")
    serve.add_argument(
        "--adapter", default="simulator", help="Adapter name (default: simulator)."
    )
    serve.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--mode", default=None, help="Adapter-specific mode (e.g. JEOL online/offline)."
    )

    sub.add_parser("list-adapters", help="List available adapter names.")

    p.add_argument("--version", action="version", version=f"nuance-mcp {__version__}")
    args = p.parse_args(argv)

    if args.cmd == "list-adapters":
        for name in available_adapters():
            print(name)
        return

    if args.cmd == "serve":
        kw = {}
        if args.mode is not None:
            kw["mode"] = args.mode
        banner = (
            f"┌────────────────────────────────────────────\n"
            f"│ nuance-mcp v{__version__}\n"
            f"│ adapter   : {args.adapter}\n"
            f"│ transport : {args.transport}\n"
            f"└────────────────────────────────────────────"
        )
        print(banner, file=sys.stderr, flush=True)
        run_server(
            adapter_name=args.adapter,
            transport=args.transport,
            host=args.host,
            port=args.port,
            adapter_kwargs=kw,
        )
