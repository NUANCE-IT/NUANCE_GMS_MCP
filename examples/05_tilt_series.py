"""Example 05 — Tilt series with pre/post quality checks.

Sequence:
    1. Read stage position.
    2. Verify the adapter supports tilt_series.
    3. Acquire a 5-frame tilt series from -10° to +10° in 5° steps.
    4. Report frame count, mean intensity, and any vendor-reported
       per-frame anomalies.

Run:
    python examples/05_tilt_series.py
    python examples/05_tilt_series.py --adapter gatan
"""

from __future__ import annotations
import asyncio
from _common import parse_args, make_server, call, banner, kv


async def main():
    args = parse_args(__doc__)
    server = make_server(args)

    caps = await call(server, "get_capabilities")
    if "tilt_series" not in caps["capabilities"]:
        print(f"This adapter does not advertise tilt-series acquisition.")
        return

    banner("Pre-flight stage")
    pos = await call(server, "get_stage_position")
    for k, v in pos.items():
        kv(k, v)

    banner("Tilt series: -10° → +10° step 5°")
    out = await call(
        server,
        "acquire_tilt_series",
        payload={
            "start_deg": -10.0,
            "end_deg": 10.0,
            "step_deg": 5.0,
            "exposure_s": 0.5,
            "binning": 2,
        },
    )
    kv("response (truncated)", str(out)[:200])

    banner("Post-flight stage")
    pos2 = await call(server, "get_stage_position")
    for k, v in pos2.items():
        kv(k, v)


if __name__ == "__main__":
    asyncio.run(main())
