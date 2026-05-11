# NUANCE-MCP Bridge Protocol — version 1.0

**Status:** *Draft, May 2026.* Implemented by `nuance_mcp.adapters.gatan.bridge` (reference). Intended to be implemented by any vendor whose Python interface loads only inside the acquisition host process (Gatan/DM, Hitachi SDK, etc.). Vendors whose API loads in-process (e.g. JEOL PyJEM) do not need this protocol.

## 1. Purpose

This document specifies the on-the-wire contract between a `nuance_mcp` FastMCP server (running on the local workstation) and a vendor *bridge plugin* (running as a daemon thread inside the vendor's host-process Python interpreter). The protocol is intentionally minimal: it carries only typed, vendor-neutral method calls plus their structured results. Schema validation has already been performed by the server before any message crosses this wire.

## 2. Transport

- **Default:** ZeroMQ `REQ`/`REP` over TCP, bound to `tcp://127.0.0.1:5555` (loopback).
- **Facility LAN:** opt-in via the environment variable `NUANCE_MCP_BRIDGE_BIND` (or vendor-specific equivalent such as `GMS_MCP_ZMQ_BIND`). When bound to a non-loopback interface, the operator is responsible for host firewall, IP allow-list, and reverse-proxy guards.
- **Wire format:** a single JSON object per request, a single JSON object per response. UTF-8. No framing beyond ZeroMQ.

## 3. Message envelope

Every request:

```json
{
  "v":      "nuance-mcp-bridge/1.0",
  "id":     "uuid-or-counter (optional)",
  "method": "<method name>",
  "params": { ... method-specific arguments ... }
}
```

Every response:

```json
{
  "v":      "nuance-mcp-bridge/1.0",
  "id":     "<echoed if present>",
  "status": "ok" | "error" | "unsupported",
  "result": { ... method-specific payload ... },     // when status=ok
  "error":  "<human-readable string>"                // when status=error|unsupported
}
```

Bridges MUST reject messages with a different `v` prefix with `status=error` and a clear `error` message. This is the version negotiation mechanism.

## 4. Capability negotiation: the `hello` handshake

The first call after connection MUST be `hello`:

```json
{ "v": "nuance-mcp-bridge/1.0", "method": "hello",
  "params": { "client": "nuance-mcp", "client_version": "0.2.0a1" } }
```

The bridge replies with its identity and supported capability list:

```json
{
  "v":      "nuance-mcp-bridge/1.0",
  "status": "ok",
  "result": {
    "server":  "nuance-mcp-bridge",
    "version": "1.0",
    "vendor":  "Gatan",
    "model":   "GMS 3.60",
    "capabilities": ["tem", "stem", "stem.haadf", "4dstem", "eels",
                     "diffraction", "stage", "stage.tilt", "optics",
                     "detectors", "live_jobs", "workspace",
                     "analysis.radial_profile", "analysis.max_fft",
                     "analysis.com_dpc", "analysis.max_spot_map"]
  }
}
```

`capabilities` MUST be a JSON array of strings drawn from `nuance_mcp.core.capabilities.Capability`. Unknown capability strings are silently ignored by clients to preserve forward compatibility.

## 5. Method dispatch

Method names match `MicroscopeAdapter` exactly:

| Family            | Method names                                                                                               |
|-------------------|------------------------------------------------------------------------------------------------------------|
| Diagnostics       | `get_microscope_state`, `get_front_image`, `get_image_shift`, `workspace_list_images`                      |
| Acquisition       | `acquire_tem`, `acquire_stem`, `acquire_4d_stem`, `acquire_eels`, `acquire_diffraction`                    |
| Stage/Optics      | `get_stage_position`, `set_stage_position`, `set_beam_parameters`, `set_magnification`, `set_image_shift`, `set_brightness`, `change_focus_relative`, `stop_stage`, `set_condenser_stigmation` |
| Detectors         | `configure_detectors`                                                                                       |
| Workflow          | `acquire_tilt_series`, `start_live_processing_job`, `get_live_processing_job_status`, `get_live_processing_job_result`, `stop_live_processing_job` |
| Derived analyses  | `apply_image_filter`, `compute_radial_profile`, `compute_max_fft`, `run_4dstem_analysis`, `run_4dstem_maximum_spot_mapping`, `run_script_template` |

A bridge that does not implement a method MUST reply with `status=unsupported`. The server translates that into a structured tool response (`{"status": "UNSUPPORTED", "reason": ...}`) for the agent.

## 6. Payload encodings

### 6.1 Images

Image-returning methods reply with:

```json
{
  "name":  "Front_TEM",
  "shape": [2048, 2048],
  "data_dtype": "float32",
  "data_b64":   "<base64 of raw little-endian bytes>",
  "calibration": { "scale": 0.043068, "unit": "nm" },
  "metadata":    { "exposure_s": 1.0, "high_tension_kV": 200.0,
                    "magnification": 800000.0 },
  "tags":        { "<vendor tag path>": <value>, ... }
}
```

Clients are responsible for `numpy.frombuffer(b64decode(data_b64), dtype=data_dtype).reshape(shape)`.

### 6.2 Spectra

```json
{
  "name": "EELS_core_loss",
  "n_channels": 2048,
  "counts_dtype": "float32",
  "counts_b64":   "<base64>",
  "energy_eV":    [<2048 floats>],
  "dispersion_eV_per_ch": 0.25,
  "exposure_s": 1.0,
  "tags": { ... }
}
```

### 6.3 Live jobs

`start_live_processing_job` returns `{job_id, job_type, state, iterations, started_at, last_update, error}`.
`get_live_processing_job_status` returns the same.
`get_live_processing_job_result` returns `{summary, derived: <image or spectrum payload as above>}`.
`stop_live_processing_job` returns the final summary.

## 7. Error semantics

| Server-side condition                          | Wire response                          |
|------------------------------------------------|----------------------------------------|
| Argument schema rejection                      | Never reaches the bridge — server returns `ValidationError` to the agent before sending. |
| Method unknown to this bridge                  | `{ "status": "unsupported", "error": "..." }`              |
| Method known but vendor refused                | `{ "status": "error",       "error": "<vendor message>" }` |
| Transport timeout                              | Server raises `BridgeTimeout` to the agent; the bridge is not informed. |

## 8. Liveness

Bridges SHOULD call the vendor's GUI-pump entry (e.g. `DM.DoEvents()` for Gatan) at least every 100 ms while waiting for messages, so the host application remains responsive during long polls.

## 9. Forward compatibility

- New methods MUST NOT change the meaning of existing method names.
- New optional fields in request/response objects are permitted at any time; receivers MUST ignore unknown fields.
- Breaking changes increment the version prefix (`nuance-mcp-bridge/2.0`) and trigger an explicit handshake failure with the older partner.

## 10. Reference implementation

`nuance_mcp/adapters/gatan/bridge.py` is the reference plugin and runs inside GMS 3.60. A future Hitachi bridge will use the same protocol; the JEOL adapter does not need it because PyJEM is in-process.
