"""Example 01 — Basic microscope-state query.

Connects to the chosen adapter, prints capability list and column state.
This is the first call an agent (or operator script) should make on every
new session, so it doubles as a connectivity test.

Run:
    python examples/01_basic_query.py                       # simulator
    python examples/01_basic_query.py --adapter gatan       # GMS via bridge
    python examples/01_basic_query.py --adapter jeol        # PyJEM in-process
"""

from __future__ import annotations
import asyncio
from _common import parse_args, make_server, call, banner, kv


async def main():
    args = parse_args(__doc__)
    server = make_server(args)

    caps = await call(server, "get_capabilities")
    banner("Adapter capabilities")
    kv("Vendor",  caps["vendor"])
    kv("Model",   caps["model"])
    kv("Bridge?", caps["bridge_required"])
    kv("Caps (n)", len(caps["capabilities"]))
    for c in caps["capabilities"]:
        kv("",       c)

    state = await call(server, "get_microscope_state")
    banner("Microscope state")
    kv("High tension (kV)", state.get("high_tension_kV"))
    kv("Mode",              state.get("mode"))
    kv("Magnification",     state.get("magnification"))
    kv("Stage α (deg)",     state.get("stage_alpha_deg"))
    kv("Stage β (deg)",     state.get("stage_beta_deg"))


if __name__ == "__main__":
    asyncio.run(main())
