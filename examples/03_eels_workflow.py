"""Example 03 — EELS survey (ZLP reference + core-loss).

Mirrors the ``eels_survey`` skill, but executed directly from Python so
the example is reproducible without an LLM agent. Useful for unit-testing
adapter EELS wiring before exposing it to a model.

Run:
    python examples/03_eels_workflow.py
    python examples/03_eels_workflow.py --adapter gatan
"""

from __future__ import annotations
import asyncio
from _common import parse_args, make_server, call, banner, kv


async def main():
    args = parse_args(__doc__)
    server = make_server(args)

    caps = await call(server, "get_capabilities")
    if "eels" not in caps["capabilities"]:
        print(f"This adapter does not advertise EELS.")
        return

    banner("ZLP reference")
    zlp = await call(server, "acquire_eels",
                     payload={"exposure_s": 0.1,
                              "energy_offset_eV": 0.0,
                              "slit_width_eV": 0.0,
                              "dispersion_idx": 0,
                              "full_vertical_binning": True})
    kv("n_channels",    zlp.get("n_channels"))
    kv("dispersion",    f"{zlp.get('dispersion_eV_per_ch')} eV/ch")
    kv("max counts",    f"{zlp.get('statistics', {}).get('max'):.0f}")

    banner("Core-loss centred at 500 eV")
    cl = await call(server, "acquire_eels",
                    payload={"exposure_s": 1.0,
                             "energy_offset_eV": 500.0,
                             "slit_width_eV": 0.0,
                             "dispersion_idx": 1,
                             "full_vertical_binning": True})
    kv("n_channels",    cl.get("n_channels"))
    kv("dispersion",    f"{cl.get('dispersion_eV_per_ch')} eV/ch")
    kv("energy range",  cl.get("energy_range_eV"))
    kv("max counts",    f"{cl.get('statistics', {}).get('max'):.0f}")


if __name__ == "__main__":
    asyncio.run(main())
