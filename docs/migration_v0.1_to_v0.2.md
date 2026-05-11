# Migration from v0.1 to v0.2

`nuance-mcp` v0.2 consolidates the v0.1 packages
`nuance-gms-mcp` (Gatan) and `jeol-mcp` (JEOL) into a single
vendor-agnostic framework. This document covers what changes, what
keeps working, and how to update existing scripts.

## TL;DR

```bash
# v0.1
pip install nuance-gms-mcp
python -m gms_mcp.server

# v0.2
pip install "nuance-mcp[gatan]"
nuance-mcp serve --adapter gatan
```

```python
# v0.1
from gms_mcp.server import mcp                # GMS implicit
await mcp.call_tool("gms_acquire_tem_image", {"payload": {...}})

# v0.2
from nuance_mcp import build_server
server = build_server("gatan")                 # explicit adapter
await server.call_tool("acquire_tem_image", {"payload": {...}})
```

## What changed

| v0.1                                      | v0.2                                                |
|-------------------------------------------|-----------------------------------------------------|
| Package name `nuance-gms-mcp`             | Package name `nuance-mcp`                       |
| Package name `jeol-mcp` (separate)        | Folded in as `nuance-mcp[jeol]`                 |
| Import `gms_mcp.*`                        | Import `nuance_mcp.*`                           |
| Tool names `gms_acquire_tem_image`        | Tool names `acquire_tem_image` (legacy aliases kept)|
| `DMSimulator`                             | `nuance_mcp.core.SimulatorAdapter`              |
| `gms_mcp.dm_plugin`                       | `nuance_mcp.adapters.gatan.bridge`              |
| Bridge bind default `tcp://0.0.0.0:5555`  | Bridge bind default `tcp://127.0.0.1:5555`          |
| Ad-hoc JSON contract                      | Versioned `nuance-mcp-bridge/1.0`               |
| Vendor implicit (one repo per vendor)     | Vendor explicit via `--adapter` flag                |

## Tool-name renames

Drop the `gms_` prefix:

| v0.1                                      | v0.2                                  |
|-------------------------------------------|---------------------------------------|
| `gms_get_microscope_state`                | `get_microscope_state`                |
| `gms_get_front_image`                     | `get_front_image`                     |
| `gms_acquire_tem_image`                   | `acquire_tem_image`                   |
| `gms_acquire_stem`                        | `acquire_stem`                        |
| `gms_acquire_4d_stem`                     | `acquire_4d_stem`                     |
| `gms_acquire_eels`                        | `acquire_eels`                        |
| `gms_acquire_diffraction`                 | `acquire_diffraction`                 |
| `gms_get_stage_position`                  | `get_stage_position`                  |
| `gms_set_stage_position`                  | `set_stage_position`                  |
| `gms_set_beam_parameters`                 | `set_beam_parameters`                 |
| `gms_configure_detectors`                 | `configure_detectors`                 |
| `gms_apply_image_filter`                  | `apply_image_filter`                  |
| `gms_compute_radial_profile`              | `compute_radial_profile`              |
| `gms_compute_max_fft`                     | `compute_max_fft`                     |
| `gms_run_4dstem_analysis`                 | `run_4dstem_analysis`                 |
| `gms_run_4dstem_maximum_spot_mapping`     | `run_4dstem_maximum_spot_mapping`     |
| `gms_acquire_tilt_series`                 | `acquire_tilt_series`                 |
| `gms_start_live_processing_job`           | `start_live_processing_job`           |
| `gms_get_live_processing_job_status`      | `get_live_processing_job_status`      |
| `gms_get_live_processing_job_result`      | `get_live_processing_job_result`      |
| `gms_stop_live_processing_job`            | `stop_live_processing_job`            |

The legacy `gms_*` names are accepted as aliases for one
deprecation cycle (v0.2.x). They will be removed in v0.3.

## Bridge migration

On the GMS PC, replace:

```python
# v0.1
exec(open("gms_mcp/dm_plugin.py").read())
```

with:

```python
# v0.2
from nuance_mcp.adapters.gatan.bridge import start_bridge
start_bridge()
```

The wire protocol is mostly compatible, but the bridge now requires
a `v: "nuance-mcp-bridge/1.0"` envelope field. v0.1 clients will
receive a clear `error` response telling them to upgrade.

## Environment variables

| v0.1                                | v0.2                                            |
|-------------------------------------|-------------------------------------------------|
| `GMS_SIMULATE=1`                    | `nuance-mcp serve --adapter simulator`      |
| `GMS_MCP_ZMQ=tcp://host:5555`       | unchanged for backwards compatibility           |
| `GMS_MCP_ZMQ_BIND=…`                | unchanged for backwards compatibility           |
| —                                   | `NUANCE_MCP_JEOL_MODE={auto,online,offline}`|
| `OLLAMA_BASE_URL=http://…:11434`    | unchanged                                       |
| `OLLAMA_MODEL=qwen2.5:7b`           | unchanged                                       |

## Skills

Skill names are unchanged. Skill bodies now use the vendor-neutral
tool names (no `gms_` prefix) and start with a `get_capabilities`
check, so they unroll correctly under any adapter.

## Tests

v0.1 had `tests/test_gms_mcp.py` with 135 tests. v0.2 splits these
into:

- `tests/test_adapter_contract.py` — structural invariants every
  adapter must satisfy.
- `tests/test_schemas.py` — Pydantic bounds.
- `tests/test_lifecycle.py` — live-job state machine.
- `tests/test_simulator_io.py` — simulator-specific I/O.
- `tests/test_tool_dispatch.py` — generic FastMCP wiring.
- `tests/test_bridge_protocol.py` — JSON envelope and capability negotiation.
- `tests/integration_ollama/` — gated behind `OLLAMA_MODEL=...`.

Existing v0.1 test functions can be lifted into the v0.2 layout with
a one-line import update; the assertions themselves are unchanged.

## Citation

If you have cited `nuance-gms-mcp` or `jeol-mcp` in a manuscript,
please add or transition to:

```bibtex
@software{dosReis2026MicroscopyMCP,
  author    = {dos Reis, Roberto and Dravid, Vinayak P.},
  title     = {NUANCE-MCP: A Vendor-Agnostic Schema-Bound Tool
               Protocol for Local LLM-Orchestrated Multimodal Electron
               Microscopy},
  version   = {0.2.0},
  year      = {2026},
  url       = {https://github.com/NUANCE-IT/nuance-mcp},
}
```

The underlying universal-MCP-for-instrumentation framework is
disclosed under Northwestern Invention Disclosure
Disc-ID-25-05-22-002 (Technology ID 2025-136), accepted 3 June 2025.

## Getting help

- Open an issue on the repo
  ([github.com/NUANCE-IT/nuance-mcp](https://github.com/NUANCE-IT/nuance-mcp)).
- Tag `migration` on the issue title.
- Include the v0.1 line that broke and the v0.2 invocation you tried.
