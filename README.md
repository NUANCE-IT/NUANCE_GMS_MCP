# nuance-mcp

A library for interfacing with transmission electron microscopes (TEM) via MCP (Model Context Protocol).

## Installation

```bash
pip install nuance-mcp
```

## Supported Hardware

The package provides adapters for multiple microscope vendors:

- **JEOL**: Full support for JEOL JEM-2100F, 2100, 2200FS, 3200FB, 3200, 3400FB, 3400FS, etc.
- **Gatan**: Bridge adapter for GMS-EELS and DM (Detector Manager) operations
- **Hitachi**: Adapter for Hitachi 2100, 2200, 2700, 2800, 2100II, 2200II, 2100UP, etc.

## Usage Examples

### JEOL Adapter

```python
from nuance_mcp.adapters.jeol import adapter

mcp = FastMCP("JEOL Adapter")
mcp.add_resource("microscope_state", adapter.get_microscope_state())
```

### Gatan Bridge Adapter

```python
from nuance_mcp.adapters.gatan import adapter

mcp = FastMCP("Gatan Adapter")
mcp.add_resource("eels_data", adapter.get_eels_data())
```

### Hitachi Adapter

```python
from nuance_mcp.adapters.hitachi import adapter

mcp = FastMCP("Hitachi Adapter")
mcp.add_resource("stem_image", adapter.acquire_stem())
```

## Features

- **Multi-vendor support**: Works with JEOL, Gatan, and Hitachi systems
- **Image acquisition**: TEM, STEM, 4D-STEM imaging
- **Spectrum acquisition**: EELS data collection
- **Diffraction**: SAD/HRTEM patterns
- **Detector control**: CCD, direct detection cameras
- **Image processing**: Filters, FFT analysis, spot mapping, DPC
- **Stage control**: X/Y positioning, tilting series
- **Live processing**: Job submission and monitoring

## Supported Capabilities

The following capabilities are supported by JEOL adapters:

- **TEM/STEM**: Conventional and annular dark field imaging
- **EELS/EDS**: Spectroscopy data acquisition
- **FFT/DPC**: Fourier transform and differential phase contrast processing
- **Image filters**: Radial profiles, spot mapping
- **Live jobs**: Asynchronous data processing

## Project status & the `/goals` command

`scripts/project_status.py` is a live status probe. It imports the FastMCP
server (simulator adapter), measures the protocol surface — typed tools,
skills, live-job types, capabilities — runs the test suite, reads git
state, and prints an 11-milestone roadmap with a completion percentage:

```bash
python scripts/project_status.py            # human-readable report
python scripts/project_status.py --json     # machine-readable
python scripts/project_status.py --no-tests # skip the pytest run (fast)
```

It exits non-zero when the suite is red, so it doubles as a CI gate and
runs as the `Project status probe` step in `.github/workflows/ci.yml`.

`.claude/commands/goals.md` is a [Claude Code](https://docs.claude.com/en/docs/claude-code)
custom slash command. Inside Claude Code, type `/goals` for the full
roadmap + live status, `/goals next` for prioritised next steps,
`/goals status` for a terse summary, or `/goals M1`…`M11` to drill into a
single milestone. See [`docs/project_status.md`](docs/project_status.md)
for details.

## License

MIT
