# Gatan adapter (`GatanGMSAdapter`)

The Gatan adapter exposes [Gatan Microscopy Suite] 3.60 to a
`nuance-mcp` server through a versioned ZeroMQ bridge that runs as a
daemon thread inside the GMS Python environment.

[Gatan Microscopy Suite]: https://www.gatan.com/products/tem-analysis/gatan-microscopy-suite-software

## Why a bridge?

`import DigitalMicrograph as DM` succeeds **only** inside the running
DM process. The bridge plugin runs there; the FastMCP server runs as
an ordinary Python process on the workstation. They speak the
versioned JSON contract documented in
[`docs/spec/nuance-mcp-bridge-1.0.md`](../spec/nuance-mcp-bridge-1.0.md).

```
   nuance-mcp serve --adapter gatan
                ↓ stdio/HTTP MCP
   FastMCP server (Python — your workstation)
                ↓ ZeroMQ REQ
   Bridge plugin (GMS Python — microscope PC)
                ↓ DigitalMicrograph
   GMS 3.60 → camera, DigiScan, EELS, stage, optics
```

## Declared capabilities

The adapter advertises 17 capability families:

`tem`, `stem`, `stem.haadf`, `stem.bf`, `stem.abf`, `4dstem`, `eels`,
`diffraction`, `tilt_series`, `stage`, `stage.tilt`, `optics`,
`detectors`, `analysis.image_filter`, `analysis.radial_profile`,
`analysis.max_fft`, `analysis.com_dpc`, `analysis.max_spot_map`,
`analysis.script_template`, `live_jobs`, `workspace`.

## Installation

```bash
# Workstation (where the FastMCP server runs)
pip install "nuance-mcp[gatan,ollama]"

# Microscope PC (inside the GMS Python environment)
cd C:\ProgramData\Miniconda3\envs\GMS_VENV_PYTHON
pip install pyzmq --break-system-packages
```

## Running the bridge

In the GMS Python console (or via `File → Open Script…` then Execute):

```python
from nuance_mcp.adapters.gatan.bridge import start_bridge
start_bridge()
```

The plugin binds `tcp://127.0.0.1:5555` by default. A dialog box
confirms the bind address.

### Facility-LAN deployment (opt-in)

To expose the bridge to other machines on the facility VLAN:

```python
import os
os.environ["GMS_MCP_ZMQ_BIND"] = "tcp://10.0.5.42:5555"
from nuance_mcp.adapters.gatan.bridge import start_bridge
start_bridge()
```

Then add the matching host firewall rule and an IP allow-list. For
HTTPS exposure, terminate TLS at a reverse proxy (nginx/Caddy) in
front of the FastMCP server; do **not** expose the raw ZeroMQ socket.

## Connecting from the agent side

```bash
nuance-mcp serve --adapter gatan
# or for a different bridge endpoint:
GMS_MCP_ZMQ=tcp://192.168.1.10:5555 nuance-mcp serve --adapter gatan
```

The adapter's constructor accepts `endpoint=` and `timeout_ms=`
kwargs if you prefer programmatic configuration:

```python
from nuance_mcp import build_server

server = build_server(
    "gatan",
    adapter_kwargs={"endpoint": "tcp://microscope-pc:5555",
                    "timeout_ms": 8000},
)
```

## Live-DM caveats

A handful of GMS controls are not exposed through DM's PythonReference
API and are reachable only through DM-script. The bridge wraps these
behind named templates and the tool layer returns a structured
`{"status": "UNSUPPORTED", "reason": ...}` if the bridge declines:

- Some EELS GIF live-control paths (acquisition itself works; live GIF
  shutter / drift tube tweaks need a DM-script wrapper).
- Programmatic camera-length setting (use the GUI or a script template).
- Detector global DS-signal toggles.

The `gms_run_script_template` tool exists exactly to cover these
cases without hard-coding vendor scripts in the adapter.

## Workspace coherence

When the bridge is running, derived images (filter outputs, FFT maps,
CoM/DPC maps, max-spot maps) appear in the GMS workspace alongside the
original acquisitions. This is what makes the live-job lifecycle
useful in practice: an operator can interact with the agent's derived
images using the rest of the GMS UI.

## Smoke test

```python
import asyncio, json
from nuance_mcp import build_server

async def main():
    server = build_server("gatan",
                          adapter_kwargs={"mode": "bridge"})
    res = await server.call_tool("get_microscope_state", {})
    print(json.loads(res.content[0].text))

asyncio.run(main())
```

If the bridge is not running, the call raises a clean timeout; the
adapter never silently falls back. To develop without the bridge, use
`--adapter simulator` instead.
