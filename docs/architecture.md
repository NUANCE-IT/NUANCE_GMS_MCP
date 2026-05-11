# Architecture

`nuance-mcp` is organised in four layers. Each layer depends only
on the one below; vendor code lives entirely inside the bottom layer.

```
                ┌──────────────────────────────────────────────┐
   Agent  ─►    │  MCP Prompts (skills) — vendor-portable      │   ← core/skills.py
                │  Multi-step protocols composed from tools    │
                ├──────────────────────────────────────────────┤
                │  Typed MCP tools — schema-validated entry    │   ← tools/__init__.py
                │  ▸ Pydantic v2 input check at the boundary   │      core/schemas.py
                │  ▸ Live-job lifecycle (start/status/result)  │      core/lifecycle.py
                ├──────────────────────────────────────────────┤
                │  MicroscopeAdapter ABC — the contract        │   ← core/adapter.py
                │  ▸ Capabilities, methods, lifecycle          │      core/capabilities.py
                ├──────────────────────────────────────────────┤
                │  Vendor adapters                             │   ← adapters/<vendor>/
                │  ▸ Gatan (DM bridge)                         │      adapters/gatan/
                │  ▸ JEOL (PyJEM in-process)                   │      adapters/jeol/
                │  ▸ Hitachi (skeleton)                        │      adapters/hitachi/
                │  ▸ Simulator (physics-plausible default)     │      core/simulator.py
                └──────────────────────────────────────────────┘
                                 ↕  (only Gatan/Hitachi)
                                 │
                ┌────────────────┴──────────────────────────────┐
   Microscope  │  Vendor host process (DM, Hitachi SDK, …)     │
        PC     │  ZeroMQ REP socket  ──  bridge plugin         │
                └───────────────────────────────────────────────┘
```

## Layer 1 — `MicroscopeAdapter`

A Python ABC. It enumerates every operation as a method with a
vendor-neutral signature (e.g. `acquire_tem(exposure_s, binning,
processing, roi) -> ImageReturn`), and exposes adapter metadata
(`vendor`, `model`, `capabilities`, `bridge_required`,
`is_thread_safe`). Concrete adapters subclass it and implement only
the methods their declared capabilities cover; everything else inherits
a default that raises `CapabilityUnavailable`.

This is the single chokepoint that makes the framework
vendor-independent. The tool, skill, and lifecycle layers see nothing
but this interface.

## Layer 2 — Typed tools + lifecycle

`tools/__init__.py` binds a FastMCP instance to a given adapter and
registers 30 typed tools. Each tool:

1. Accepts a Pydantic v2 `payload` model.
2. Validates the payload against physical and operational bounds.
3. Dispatches to the adapter method of the same name.
4. Returns a JSON-serialisable payload.
5. Translates `CapabilityUnavailable` into a structured
   `{"status": "UNSUPPORTED", "reason": "..."}` response.

The live-job lifecycle (`start_live_processing_job`,
`get_live_processing_job_status`, `…_result`, `stop_live_processing_job`)
is owned by the server when the adapter does not declare `LIVE_JOBS`,
and forwarded to the adapter otherwise. From the agent's perspective
the contract is identical.

## Layer 3 — Skills (MCP prompts)

`core/skills.py` registers six declarative protocols:

| Skill                       | Protocol                                                           |
|-----------------------------|--------------------------------------------------------------------|
| `eels_survey`               | ZLP reference → core-loss → edge ID                                |
| `tilt_series_protocol`      | Tomographic tilt series with pre/post quality checks               |
| `4dstem_characterization`   | vBF/HAADF → CoM → DPC → (optional) orientation map                 |
| `beam_alignment`            | Beam centring, stigmation, focus verification                      |
| `hrtem_imaging`             | Survey → HRTEM → FFT → d-spacing match                             |
| `diffraction_survey`        | Diffraction pattern → radial profile → phase identification        |

Skill bodies use the vendor-neutral tool names exclusively and start
with a `get_capabilities` check. The same skill therefore unrolls
into identical tool sequences regardless of which adapter is mounted.

## Layer 4 — Adapters

Three reference adapters and one stub ship with the framework:

| Adapter      | Capabilities | Transport to vendor              | Bridge required? |
|--------------|--------------|----------------------------------|------------------|
| `simulator`  | 20 families  | in-process NumPy / SciPy         | no               |
| `gatan`      | 17 families  | DigitalMicrograph via ZeroMQ     | yes              |
| `jeol`       | 17 families  | PyJEM / TEM3 (online or offline) | no               |
| `hitachi`    | 8 families   | Hitachi SDK (not yet wired)      | yes (planned)    |

Third-party adapters self-register through the
`nuance_mcp.adapters` entry-point group.

## Why the bridge is small

For vendor APIs that load only inside the host process
(DigitalMicrograph, Hitachi SDK), the adapter must cross a
process boundary. We do this with a **minimal versioned JSON contract**
over ZeroMQ:

* Default binding `127.0.0.1:5555` (loopback). Facility-LAN exposure
  is opt-in via env var, with the operator responsible for firewall,
  allow-list, and reverse-proxy guards.
* `hello` handshake negotiates protocol version and capability list.
* Method names match `MicroscopeAdapter` exactly, so the bridge can be
  treated as a transparent transport.

Full specification: [`docs/spec/nuance-mcp-bridge-1.0.md`](spec/nuance-mcp-bridge-1.0.md).

## Why the simulator shares the schema layer

Hardware-independent tests run against the simulator using **the same
schema layer** that guards the live path. A bug introduced in a
schema definition therefore breaks CI immediately, regardless of
adapter. This is why we treat the simulator as a peer adapter rather
than as a special-case fallback.
