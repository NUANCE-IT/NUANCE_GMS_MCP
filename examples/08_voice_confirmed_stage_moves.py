"""Example 08 — Operator-confirmed stage moves.

Schema-bound validation is *necessary* but not *sufficient* for full
instrument safety. For high-risk operations (large stage tilts, beam-on
moves) facility policy typically requires a human-in-the-loop
confirmation. This example shows the pattern: the agent proposes a
``set_stage_position`` payload, the human is asked to approve, and the
call is only dispatched after explicit consent.

The same pattern applies to ``set_beam_parameters``, beam blanker
toggles, and detector insertions.

Run:
    python examples/08_voice_confirmed_stage_moves.py \\
        --transcript "Tilt the stage to plus 45 degrees alpha."
    python examples/08_voice_confirmed_stage_moves.py --adapter gatan \\
        --transcript "Move stage to x=200 microns y=-150 microns."
"""

from __future__ import annotations
import argparse, asyncio
from _common import make_server, call, banner, kv


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", default="simulator")
    p.add_argument("--mode", default=None)
    p.add_argument("--transcript", required=True,
                   help="Spoken or typed natural-language stage command.")
    return p.parse_args()


def _propose(transcript: str) -> dict:
    """Toy natural-language parser. Real deployments would use the
    Ollama agent; this stub keeps the example dependency-free."""
    t = transcript.lower()
    out: dict[str, float] = {}
    import re
    for axis, key in [("alpha", "alpha_deg"), ("beta", "beta_deg"),
                      ("x", "x_um"), ("y", "y_um"), ("z", "z_um")]:
        m = re.search(rf"{axis}[^-\d]*(-?[\d.]+)", t)
        if m:
            out[key] = float(m.group(1))
    if "plus" in t and "alpha_deg" in out and out["alpha_deg"] < 0:
        out["alpha_deg"] *= -1
    if "minus" in t and "alpha_deg" in out and out["alpha_deg"] > 0:
        out["alpha_deg"] *= -1
    return out


async def main():
    args = _parse()
    server = make_server(args)

    banner("Proposed move")
    proposed = _propose(args.transcript)
    if not proposed:
        print("  could not parse a stage target from transcript; aborting.")
        return
    for k, v in proposed.items(): kv(k, v)

    # Operator confirmation gate
    try:
        ok = input("\nApprove this move? [y/N] ").strip().lower() == "y"
    except EOFError:
        ok = False
    if not ok:
        print("  declined; no hardware call issued.")
        return

    banner("Dispatching set_stage_position")
    try:
        out = await call(server, "set_stage_position", payload=proposed)
        kv("response", out)
    except Exception as exc:
        # Schema rejection arrives here (e.g. alpha=95 → ValidationError)
        print(f"  call rejected: {type(exc).__name__}")
        print(f"  message: {str(exc)[:200]}")


if __name__ == "__main__":
    asyncio.run(main())
