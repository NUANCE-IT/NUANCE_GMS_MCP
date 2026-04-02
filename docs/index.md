# GMS-MCP Documentation

**GMS-MCP** is an open-source Model Context Protocol (MCP) server that
connects Gatan Microscopy Suite 3.60 to local large language models,
enabling natural-language electron microscopy control without cloud
dependencies.

---

## Documentation Contents

| Page | Description |
|---|---|
| [Installation](installation.md) | Full setup guide for all deployment scenarios |
| [Architecture](architecture.md) | Two-process bridge, ZeroMQ design, DMSimulator |
| [Tools Reference](tools_reference.md) | Complete API reference for all 21 MCP tools |
| [DM API Reference](dm_api_reference.md) | DigitalMicrograph Python API quick reference |
| [Live GMS Setup](gms_live_setup.md) | Step-by-step guide for connecting a real microscope |

---

## Quick navigation

- **First time?** Start with [Installation](installation.md)
- **Want to test without a microscope?** See [DMSimulator](architecture.md#dmsimulator)
- **Connecting a real GMS instance?** See [Live GMS Setup](gms_live_setup.md)
- **Looking for a specific tool?** See [Tools Reference](tools_reference.md)

---

## Repository structure

```text
NUANCE_GMS_MCP/
├── .github/workflows/ci.yml     # lint + typecheck + test matrix + build
├── .gitignore
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE                       # MIT, Roberto dos Reis & Vinayak P. Dravid
├── README.md                     # badges, highlights, quick-start
├── pyproject.toml                # packaging, ruff, mypy, pytest config
├── src/gms_mcp/
│   ├── __init__.py               # version
│   ├── server.py                 # FastMCP server — 21 tools
│   ├── simulator.py              # DMSimulator physics twin
│   ├── client.py                 # Ollama ReAct agent
│   └── dm_plugin.py              # ZeroMQ bridge with persistent live-job backend
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # session fixtures, GMS_SIMULATE=1
│   └── test_gms_mcp.py           # 107 tests (92 hardware-independent)
├── examples/
│   ├── 01_basic_query.py
│   ├── 02_tem_acquisition.py
│   ├── 03_eels_workflow.py
│   ├── 04_4dstem_analysis.py
│   ├── 05_tilt_series.py
│   ├── 06_diffraction_dspacing.py
│   ├── 07_voice_acquisition.py
│   └── 08_voice_confirmed_stage_moves.py
└── docs/
    ├── index.md
    ├── installation.md
    ├── architecture.md           # ASCII diagram, data-flow walkthrough
    ├── tools_reference.md        # full API for all 21 tools
    ├── dm_api_reference.md       # DM Python quick reference
    └── gms_live_setup.md         # microscope PC wiring guide
```

---

## Reference

**Paper:** dos Reis, R. & Dravid, V.P. (2025).
*GMS-MCP: A Vendor-Agnostic, Privacy-Preserving Model Context Protocol
Server for Multimodal Electron Microscopy Control via Local Large
Language Models.* arXiv:2025.XXXXX.
