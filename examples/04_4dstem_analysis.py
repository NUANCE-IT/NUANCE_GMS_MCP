"""Example 04 — 4D-STEM acquisition + derived analyses.

Acquires a small 4D-STEM scan and runs the bundled derived-analysis tools:
virtual bright-field, virtual HAADF, CoM, and DPC magnitude.

Skipped automatically when the adapter does not advertise ``4dstem``.

Run:
    python examples/04_4dstem_analysis.py
    python examples/04_4dstem_analysis.py --adapter gatan
"""

from __future__ import annotations
import asyncio
from _common import parse_args, make_server, call, banner, kv


async def main():
    args = parse_args(__doc__)
    server = make_server(args)

    caps = await call(server, "get_capabilities")
    if "4dstem" not in caps["capabilities"]:
        print(f"This adapter does not advertise 4D-STEM.")
        return

    banner("4D-STEM acquisition")
    ds = await call(server, "acquire_4d_stem",
                    payload={"scan_x": 32, "scan_y": 32,
                             "dwell_us": 1000.0,
                             "camera_length_mm": 300.0,
                             "convergence_mrad": 10.0})
    kv("Name",  ds.get("name"))
    kv("Shape", ds.get("shape"))
    kv("Exposure (s)", ds.get("exposure_s"))

    if "analysis.com_dpc" in caps["capabilities"]:
        banner("Run CoM/DPC analysis")
        out = await call(server, "run_4dstem_analysis")
        kv("response", str(out)[:120])


if __name__ == "__main__":
    asyncio.run(main())
