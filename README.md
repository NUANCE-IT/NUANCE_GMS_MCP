# GMS-MCP 🔬

[![arXiv](https://img.shields.io/badge/arXiv-2025.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2025.XXXXX)
[![Tests](https://github.com/NUANCE-IT/Gatan_MCP/actions/workflows/ci.yml/badge.svg)](https://github.com/NUANCE-IT/Gatan_MCP/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Northwestern NUANCE](https://img.shields.io/badge/NUANCE-Northwestern-4E2A84.svg)](https://www.nuance.northwestern.edu)

> **A vendor-agnostic, privacy-preserving Model Context Protocol server for multimodal electron microscopy control via local large language models.**

GMS-MCP connects **Gatan Microscopy Suite (GMS) 3.60** to any MCP-compatible LLM — running entirely on your institution's hardware, with zero cloud dependencies.

---

## Highlights

| Capability | Details |
|---|---|
| Instrument API | Gatan DigitalMicrograph / GMS 3.60 |
| LLM backend | Local Ollama (air-gap compatible) |
| Voice control | Optional local push-to-talk + Whisper transcription |
| Data handling | On-site, local-first workflow |
| Modalities | TEM / HRTEM, STEM (HAADF/BF/ABF), 4D-STEM / NBED, EELS, diffraction |
| Built-in analysis | Virtual BF/HAADF, CoM, DPC, radial profiles, max-FFT, filtering, maximum-spot mapping |
| Automation | Stage control, beam/optics control, detector configuration, tilt series, persistent live-processing jobs |
| Validation | Pydantic v2 physical-bound checks on tool inputs |
| Simulation | Physics-plausible DMSimulator for hardware-free development |
| Testing | 67-test suite (61 hardware-independent + 6 Ollama integration) |
| Transport | stdio + Streamable HTTP + optional ZeroMQ live-job bridge |
| License | MIT |

---

## Architecture

```
Ollama LLM (local, port 11434)
    ↕  LangChain ReAct agent
MultiServerMCPClient
    ↕  stdio subprocess  OR  HTTP /mcp
gms_mcp.server  (FastMCP 3.x)
    ↕  ZeroMQ TCP bridge
DM Plugin  (inside GMS process)
    ↕  DigitalMicrograph Python API
Microscope hardware (TEM / STEM column)
```

The **DMSimulator** activates automatically when `DigitalMicrograph` is unavailable, providing physics-plausible synthetic data for all five modalities — enabling full development and testing without a microscope.

---

## Quick Start

### 1. Install

Published release from PyPI:

```bash
# Core server only
pip install nuance-gms-mcp

# With Ollama client support
pip install "nuance-gms-mcp[ollama]"

# With local voice control (microphone + Whisper transcription)
pip install "nuance-gms-mcp[ollama,voice]"

# With ZeroMQ bridge for live GMS connection
pip install "nuance-gms-mcp[ollama,zmq]"

# Install the latest code from this repository instead of the published PyPI release
pip install "git+https://github.com/NUANCE-IT/Gatan_MCP.git"

# Full development install from a local clone
git clone https://github.com/NUANCE-IT/Gatan_MCP
cd Gatan_MCP
pip install -e ".[all]"
```

Use `pip install nuance-gms-mcp` when you want the published release from PyPI.
Use `pip install -e ".[all]"`, `pip install .`, or the direct GitHub URL when you
want this repository's current source tree.

### 2. Install Ollama + pull a model

```bash
# Install Ollama: https://ollama.ai
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the recommended model (best tool-calling performance)
ollama pull qwen2.5:7b

# Alternatives
ollama pull qwen2.5:14b    # higher accuracy, slower
ollama pull llama3.1:8b    # reliable, widely tested
```

### 3. Run in simulation mode (no microscope needed)

```bash
# Start the interactive microscope agent
GMS_SIMULATE=1 python -m gms_mcp.client

# Or a single non-interactive query
GMS_SIMULATE=1 python -m gms_mcp.client \
  --query "Acquire a 512×512 HAADF STEM image at 10 µs dwell time" \
  --no-interactive --verbose
```

### 3b. Run with local voice control

```bash
# Push-to-talk microphone input with local faster-whisper transcription
GMS_SIMULATE=1 python -m gms_mcp.client --voice

# Voice input plus spoken replies on macOS (uses the built-in 'say' command)
GMS_SIMULATE=1 python -m gms_mcp.client --voice --speak

# One-shot voice command, then exit
GMS_SIMULATE=1 python -m gms_mcp.client --voice --no-interactive
```

Voice mode records locally, transcribes locally with Whisper, and sends the
resulting text transcript through the same Ollama → MCP tool-calling path as
typed instructions.

### 4. Connect to a live GMS instance

**On the microscope PC** (inside GMS Python environment):
```bash
# Install ZeroMQ inside GMS virtual environment
cd C:\ProgramData\Miniconda3\envs\GMS_VENV_PYTHON
pip install pyzmq fastmcp

# Run the DM plugin inside GMS Python console
exec(open("src/gms_mcp/dm_plugin.py").read())
```

**On any workstation** (or the same PC):
```bash
# Start the HTTP server (for remote access / Claude.ai connector)
python -m gms_mcp.server --transport http --port 8000

# Or stdio for direct Ollama use
GMS_MCP_ZMQ=tcp://microscope-pc:5555 python -m gms_mcp.client
```

When `GMS_MCP_ZMQ` is set, persistent live-processing jobs are created and managed inside
the DM bridge so long-running state stays aligned with the live GMS process.

---

## Available Tools

| Tool | Domain | Description |
|---|---|---|
| `gms_get_microscope_state` | Diagnostics | Read all instrument parameters |
| `gms_get_front_image` | Workspace | Inspect the current front-most DM image and tags |
| `gms_acquire_tem_image` | Acquisition | TEM / HRTEM image with exposure, binning, ROI |
| `gms_acquire_stem` | Acquisition | HAADF / BF / ABF STEM scan |
| `gms_acquire_4d_stem` | Acquisition | Full 4D-STEM / NBED dataset |
| `gms_acquire_eels` | Acquisition | EELS spectrum with GIF/IFC control |
| `gms_acquire_diffraction` | Acquisition | Electron diffraction + auto d-spacing extraction |
| `gms_apply_image_filter` | Analysis | Median / Gaussian filtering on the front image or ROI |
| `gms_compute_radial_profile` | Analysis | 1D radial profile from diffraction or HRTEM FFT |
| `gms_compute_max_fft` | Analysis | Max-FFT map over local windows in the front image |
| `gms_start_live_processing_job` | Workflow | Start a persistent live radial-profile, difference, FFT-map, filtered-view, or maximum-spot-mapping job |
| `gms_get_live_processing_job_status` | Workflow | Poll a live job for iterations, status, and latest summary |
| `gms_get_live_processing_job_result` | Workflow | Retrieve the latest derived result from a live job |
| `gms_stop_live_processing_job` | Workflow | Stop a live-processing job |
| `gms_get_stage_position` | Stage | Read X, Y, Z, α, β |
| `gms_set_stage_position` | Stage | Move stage (validated bounds) |
| `gms_set_beam_parameters` | Optics | Spot size, focus, beam shift/tilt, stigmators |
| `gms_configure_detectors` | Detectors | Insert/retract camera, CCD temp, HAADF/BF/ABF |
| `gms_acquire_tilt_series` | Workflow | Automated tomographic tilt series |
| `gms_run_4dstem_analysis` | Analysis | Virtual BF/HAADF, CoM, DPC maps |
| `gms_run_4dstem_maximum_spot_mapping` | Analysis | Color maximum-spot map from a 4D-STEM dataset |

All tools enforce **physical parameter bounds** via Pydantic v2 before any hardware command is issued.

---

## Example Session

```
You: What is the current microscope configuration?

Agent: [calls gms_get_microscope_state]
       The microscope is in TEM mode at 200 kV, spot size 3,
       magnification 50,000×. Stage at X=0, Y=0, α=0°.
       Camera (OneView) is inserted at -25°C.

You: Acquire a HAADF STEM image at 512×512, 10 µs dwell time.

Agent: [calls gms_acquire_stem with width=512, height=512, dwell_us=10.0, signals=[0]]
       Acquired 512×512 HAADF image. Mean intensity: 487 counts,
       max: 2341 counts. Total frame time: 2.62 s.
       Pixel calibration: 0.0196 nm/pixel.

You: Now acquire an EELS spectrum at the Ti L-edge (460 eV).

Agent: [calls gms_acquire_eels with energy_offset_eV=400.0, dispersion_idx=1]
       EELS spectrum acquired. Energy range: 400–912 eV at 0.25 eV/channel.
       Ti L₂,₃ edge detected at 462 eV. ZLP not visible (energy offset applied).
       Recommend checking for beam damage — reduce exposure if signal is sufficient.

You: Tilt to -60° and run a tilt series to +60° in 2° steps.

Agent: [calls gms_set_stage_position with alpha_deg=-60.0]
       [calls gms_acquire_tilt_series with start_deg=-60, end_deg=60, step_deg=2.0]
       Tilt series complete: 61 frames, α = -60° to +60°, 1 s exposure each.
       Mean intensity stable across tilt range (CV = 4.2%).
```

---

## Connecting to Claude.ai

1. Start the HTTP server: `python -m gms_mcp.server --transport http --port 8000`
2. Expose via HTTPS (e.g. `ngrok http 8000`)
3. In Claude.ai → **Settings → Connectors → Add custom connector**
4. Enter URL: `https://your-ngrok-url.ngrok.io/mcp`

---

## Running Tests

```bash
# All hardware-independent tests (~18 s)
pytest tests/ -v -m "not ollama"

# Full suite including Ollama end-to-end tests
OLLAMA_MODEL=qwen2.5:7b pytest tests/ -v

# With coverage
pytest tests/ -m "not ollama" --cov=gms_mcp --cov-report=html
```

**Test suite summary:**

| Class | Tests | Hardware required |
|---|---|---|
| `TestDMSimulator` | 17 | None |
| `TestMCPServerTools` | 39 | None |
| `TestServerTransport` | 4 | None |
| `TestOllamaIntegration` | 6 | Ollama + model |

---

## Project Structure

```
Gatan_MCP/
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
│   └── test_gms_mcp.py           # 67 tests (61 hardware-free)
├── examples/
│   ├── 01_basic_query.py
│   ├── 02_tem_acquisition.py
│   ├── 03_eels_workflow.py
│   ├── 04_4dstem_analysis.py
│   ├── 05_tilt_series.py
│   ├── 06_diffraction_dspacing.py
│   ├── 07_voice_acquisition.py
│   └── 08_voice_confirmed_stage_moves.py
├── docs/
│   ├── index.md
│   ├── installation.md
│   ├── architecture.md           # ASCII diagram, data-flow walkthrough
│   ├── tools_reference.md        # full API for all 21 tools
│   ├── dm_api_reference.md       # DM Python quick reference
│   └── gms_live_setup.md         # microscope PC wiring guide
```

---

## Supported Ollama Models

| Model | Tool-calling | Multi-step | Latency (RTX 4090) |
|---|---|---|---|
| **qwen2.5:7b** ⭐ | 97% | 90% | 4.2 s |
| qwen2.5:14b | 99% | 95% | 8.7 s |
| llama3.1:8b | 94% | 82% | 5.1 s |
| llama3.2:3b | 82% | 58% | 2.8 s |
| mistral-nemo | 88% | 70% | 6.3 s |

---

## Citation

If you use GMS-MCP in your research, please cite:

```bibtex
@article{dosReis2025gmsmcp,
  author    = {dos Reis, Roberto and Dravid, Vinayak P.},
  title     = {{GMS-MCP}: A Vendor-Agnostic, Privacy-Preserving Model
               Context Protocol Server for Multimodal Electron Microscopy
               Control via Local Large Language Models},
  journal   = {arXiv preprint arXiv:2025.XXXXX},
  year      = {2025},
  url       = {https://arxiv.org/abs/2025.XXXXX}
}
```

---

## Acknowledgements

This work was supported by the NUANCE Center at Northwestern University
(NSF MRSEC DMR-2308691, NSF NNCI).

We thank the developers of
[FastMCP](https://gofastmcp.com),
[LangChain](https://python.langchain.com),
[Ollama](https://ollama.ai), and the
[dmscripting.com](http://dmscripting.com) community.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). We welcome:
- New acquisition modalities (e.g., EFTEM, Lorentz TEM)
- Additional Ollama model benchmarks
- Live GMS testing reports
- Documentation improvements

---

## License

MIT © 2025 Roberto dos Reis & Vinayak P. Dravid, Northwestern University
