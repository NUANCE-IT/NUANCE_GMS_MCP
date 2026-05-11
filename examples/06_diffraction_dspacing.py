"""Example 06 — Diffraction acquisition and radial profile.

Sequence:
    1. Switch to diffraction mode (if supported).
    2. Acquire a diffraction pattern.
    3. Run ``compute_radial_profile`` to extract d-spacings.

Run:
    python examples/06_diffraction_dspacing.py
    python examples/06_diffraction_dspacing.py --adapter gatan
"""

from __future__ import annotations
import asyncio
from _common import parse_args, make_server, call, banner, kv


async def main():
    args = parse_args(__doc__)
    server = make_server(args)

    caps = await call(server, "get_capabilities")
    if "diffraction" not in caps["capabilities"]:
        print(f"This adapter does not advertise diffraction.")
        return

    banner("Diffraction pattern")
    img = await call(server, "acquire_diffraction",
                     payload={"exposure_s": 0.5,
                              "camera_length_mm": 300.0, "binning": 1})
    kv("Name",  img.get("name"))
    kv("Shape", img.get("shape"))
    stats = img.get("statistics", {})
    kv("Max counts", stats.get("max"))

    if "analysis.radial_profile" in caps["capabilities"]:
        banner("Radial profile")
        out = await call(server, "compute_radial_profile",
                         payload={"mode": "diffraction",
                                  "smooth_sigma": 1.5})
        kv("response (truncated)", str(out)[:200])


if __name__ == "__main__":
    asyncio.run(main())
