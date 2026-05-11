# JEOL adapter (`JEOLAdapter`)

The JEOL adapter wraps JEOL's PyJEM/TEM3 Python interface. Unlike the
Gatan adapter it does **not** require a separate bridge: PyJEM loads
inside the JEOL TEMcenter Python interpreter and provides a
behaviourally identical `PyJEM.offline` simulator that is safe to
import anywhere.

## Why no bridge?

PyJEM is importable in the same interpreter that runs the FastMCP
server. The adapter therefore talks to TEM3 in-process, which keeps
the path short and avoids a second JSON contract. The cost is that
the server must run on the JEOL PC (or in a container that has PyJEM
mounted) — there is no cross-machine option without TEMcenter remote
desktop or an equivalent vendor-supplied tunnel.

```
   nuance-mcp serve --adapter jeol
                ↓ stdio/HTTP MCP
   FastMCP server  (Python — JEOL PC)
                ↓ import PyJEM.TEM3
   TEM3 → Stage3, EOS3, HT3, Lens3, GUN3, FEG3, Filter3, …
                ↓
   JEOL TEMcenter / live microscope
```

## Declared capabilities

`tem`, `stem`, `stem.haadf`, `stem.bf`, `stem.abf`, `eels`, `eds`,
`diffraction`, `tilt_series`, `stage`, `stage.tilt`, `optics`,
`detectors`, `ht`, `feg`, `apertures`, `beam_blanker`,
`analysis.radial_profile`, `analysis.image_filter`, `live_jobs`.

(Note: `4dstem` is not yet advertised — PyJEM exposes scan + camera
primitives but no integrated 4D-STEM acquisition mode. Will be
added once a vendor-side helper is wired up.)

## Online vs offline

| Mode      | Behaviour                                                              |
|-----------|------------------------------------------------------------------------|
| `auto`    | (default) try `PyJEM.TEM3` first; fall back to `PyJEM.offline.TEM3`.   |
| `online`  | Force the real driver. Used on the JEOL PC during sessions.            |
| `offline` | Force the in-process simulator. Used for demos and CI.                 |

Set the mode programmatically:

```python
from nuance_mcp import build_server
server = build_server("jeol", adapter_kwargs={"mode": "offline"})
```

Or by environment variable:

```bash
NUANCE_MCP_JEOL_MODE=offline nuance-mcp serve --adapter jeol
```

## TEM3 sub-system memoisation

`Stage3`, `EOS3`, `HT3`, `Lens3`, `GUN3`, `FEG3`, `Filter3`, `Apt3`,
`Def3`, `MDS3`, `Nitrogen3`, `Vacuum3`, `Camera3`, `Scan3`,
`Detector3` are stateful drivers; the adapter memoises a single
instance per subsystem so repeated tool calls do not reset the
internal move queues. This mirrors the pattern in the standalone
`jeol_mcp` package that this adapter supersedes.

## Safety gates

The standalone `jeol_mcp` package shipped a `safety.py` module that
required `confirm=true` for hardware-modifying tools and offered a
dry-run mode. That policy is now expressed at two layers:

1. **Schema-bound bounded execution** — the same Pydantic bounds used
   by every adapter (e.g. `alpha_deg ∈ [-80°, +80°]`,
   `exposure_s ∈ [10⁻³, 60]`) reject malformed requests before the
   adapter is touched.
2. **Operator confirmation** — facility-policy gates remain the
   responsibility of the deploying institution. See
   [`docs/safety.md`](../safety.md) for the layered safety model.

`Example 08` shows the operator-confirmation pattern explicitly.

## Smoke test

```bash
python examples/01_basic_query.py --adapter jeol --mode offline
```

Expected output: `vendor: JEOL`, capability list of 17 entries,
microscope state populated from the offline driver. If you have a
live TEMcenter session running, drop `--mode offline` to talk to the
real column.

## What this adapter supersedes

Replaces the standalone `jeol-mcp` PyPI package. The per-subsystem
helpers from `jeol_mcp.tools.*` (Stage, EOS, HT, Lens, GUN, FEG,
Filter, Apt, Def, MDS, Nitrogen, Vacuum, Camera, Scan, Detector3,
EDS) are folded into private modules under
`nuance_mcp/adapters/jeol/_tem3/` during migration; their
behaviour is unchanged.
