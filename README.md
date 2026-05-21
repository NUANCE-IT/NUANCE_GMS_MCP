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

## Academic Citation

If you use this work or its concepts in your research, please cite our associated preprint:

**Title:** Schema-Bound LLM Control of Scientific Instrumentation through Model Context Protocol Skills  
**Authors:** Roberto dos Reis, Vinayak P. Dravid  
*(arXiv ID and URL pending)*

```bibtex
@misc{dosReis2026SchemaBound,
  title={Schema-Bound LLM Control of Scientific Instrumentation through Model Context Protocol Skills},
  author={Roberto dos Reis and Vinayak P. Dravid},
  year={2026},
  eprint={PENDING},
  archivePrefix={arXiv}
}
```

## Intellectual Property Disclosure

This software, schemas, and the associated methodologies for agentic control of scientific instrumentation using the Model Context Protocol (MCP) were developed in whole or in part at Northwestern University. 

This work is the subject of the following Northwestern University Invention Disclosure:
- **Disclosure Title:** Universal Control Protocol for Scientific Instrumentation Using Extended Model Context Protocol
- **Invention ID:** Disc-ID-25-05-22-002
- **Tech ID:** 2025-136
- **Inventors:** Roberto Moreno Souza dos Reis, Vinayak P. Dravid

Commercialization, licensing inquiries, and related intellectual property matters should be directed to the Innovation and New Ventures Office (INVO) at Northwestern University.

## License

MIT
