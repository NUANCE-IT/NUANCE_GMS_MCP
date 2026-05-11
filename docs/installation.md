# Installation

`nuance-mcp` ships as a single PyPI package with vendor-specific
extras. The core (schema layer, simulator, lifecycle, skills) installs
everywhere; the vendor SDK dependencies install only when you opt in.

## Core

```bash
pip install nuance-mcp
nuance-mcp serve --adapter simulator
```

This is enough to run the entire validation suite, drive an Ollama agent
against the simulator, and develop new skills without any hardware.

## Vendor extras

| Extra              | What you get                                                                                              |
|--------------------|-----------------------------------------------------------------------------------------------------------|
| `[gatan]`          | The Gatan adapter and `pyzmq` for the host-process bridge.                                                |
| `[jeol]`           | The JEOL adapter. PyJEM itself is JEOL-proprietary and must be installed manually inside TEMcenter.       |
| `[hitachi]`        | Placeholder; capability set declared but no implementation yet. Contributions welcome.                    |
| `[ollama]`         | LangChain/LangGraph adapters for local Ollama agent orchestration.                                        |
| `[voice]`          | `faster-whisper` + `sounddevice` for push-to-talk voice control.                                          |
| `[all]`            | Convenience alias for `[ollama,voice,gatan,jeol]`.                                                        |

```bash
pip install "nuance-mcp[gatan,ollama]"            # GMS + local LLM
pip install "nuance-mcp[jeol]"                    # JEOL only
pip install "nuance-mcp[gatan,jeol,voice,ollama]" # everything
```

## Python versions

CPython 3.10, 3.11, and 3.12 are supported in CI. Earlier versions are
not tested.

## Gatan: installing the host-side bridge

On the GMS PC, inside the GMS Python environment:

```bash
# 1. Install pyzmq inside the GMS interpreter
cd C:\ProgramData\Miniconda3\envs\GMS_VENV_PYTHON
pip install pyzmq --break-system-packages

# 2. Run the bridge plugin from GMS's Python console
exec(open("nuance_mcp/adapters/gatan/bridge.py").read())
```

The plugin binds `tcp://127.0.0.1:5555` by default. To expose it on the
facility LAN (firewall and allow-list required), set
`GMS_MCP_ZMQ_BIND=tcp://<facility-vlan-ip>:5555` before running the
plugin. See [`docs/safety.md`](safety.md) for deployment guidance.

## JEOL: PyJEM

PyJEM ships only on JEOL TEM PCs. On a developer laptop the adapter
falls back to `PyJEM.offline` automatically (set
`NUANCE_MCP_JEOL_MODE=offline` to force this on a TEM PC for
demos). No bridge is required because PyJEM loads in-process.

## Verifying the install

```bash
nuance-mcp list-adapters
# simulator
# gatan
# jeol
# hitachi

nuance-mcp serve --adapter simulator
# starts the FastMCP server on stdio
```

## Source install (development)

```bash
git clone https://github.com/NUANCE-IT/nuance-mcp
cd nuance-mcp
pip install -e ".[all]"
pytest tests/ -q
```
