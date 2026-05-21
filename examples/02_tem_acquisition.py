"""Example 02 — TEM image acquisition with bounded validation.

Sequence:
    1. Check capabilities (skip example if ``tem`` not advertised).
    2. Acquire a TEM image with 0.5 s exposure, 2× binning.
    3. Print the image shape, calibration, and basic statistics.
    4. Demonstrate the schema-bound rejection path with a deliberately
       out-of-range exposure.

Run:
    python examples/02_tem_acquisition.py
    python examples/02_tem_acquisition.py --adapter gatan
"""

from __future__ import annotations
import asyncio
from _common import parse_args, make_server, call, banner, kv


async def main():
    args = parse_args(__doc__)
    server = make_server(args)

    caps = await call(server, "get_capabilities")
    if "tem" not in caps["capabilities"]:
        print(f"This adapter ({caps['vendor']}) does not advertise TEM. Aborting.")
        return

    img = await call(
        server,
        "acquire_tem_image",
        payload={"exposure_s": 0.5, "binning": 2, "processing": 3},
    )
    banner("Acquired TEM image")
    kv("Name", img.get("name"))
    kv("Shape", img.get("shape"))
    kv("dtype", img.get("dtype"))
    stats = img.get("statistics", {})
    for k in ("min", "max", "mean", "std"):
        if k in stats:
            kv(f"{k}", f"{stats[k]:.4g}")
    cal = img.get("calibration", {})
    kv("Pixel size", f"{cal.get('scale')} {cal.get('unit', '')}")

    banner("Negative control — schema rejection")
    try:
        await call(
            server,
            "acquire_tem_image",
            payload={"exposure_s": 1000.0, "binning": 1, "processing": 3},
        )
        print("  unexpected: 1000 s exposure was NOT rejected!")
    except Exception as exc:
        print(f"  rejected before any vendor call: {type(exc).__name__}")
        print(f"  message: {str(exc)[:200]}")


if __name__ == "__main__":
    asyncio.run(main())
