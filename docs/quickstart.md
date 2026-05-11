# Quickstart

Five minutes from install to a working microscope agent.

## 1. Install

```bash
pip install "nuance-mcp[gatan,ollama]"   # or [jeol], or core only
```

## 2. Start the server

```bash
nuance-mcp serve --adapter simulator         # no hardware needed
# or
nuance-mcp serve --adapter gatan             # GMS via the bridge
nuance-mcp serve --adapter jeol              # PyJEM in-process
```

The server speaks MCP over stdio by default. Streamable HTTP for remote
MCP clients is optional:

```bash
nuance-mcp serve --adapter gatan --transport http --port 8000
```

## 3. Drive it from Python

```python
import asyncio, json
from nuance_mcp import build_server

async def main():
    server = build_server("simulator")
    res = await server.call_tool("get_microscope_state", {})
    print(json.loads(res.content[0].text))

asyncio.run(main())
```

## 4. Drive it from an Ollama agent

```python
import asyncio
from nuance_mcp.agent import run_agent
from nuance_mcp import build_server

async def main():
    server = build_server("gatan")
    answer = await run_agent(
        server,
        "Acquire a 256×256 HAADF STEM image at 5 µs dwell time and report mean intensity.",
        model="qwen2.5:7b",
    )
    print(answer)

asyncio.run(main())
```

## 5. Use a skill

Skills are vendor-portable multi-step protocols exposed as MCP prompts.
Any MCP-compatible client (Claude.ai, Ollama through LangChain) can
discover and invoke them by name:

```python
prompt = await server.get_prompt("eels_survey",
                                  {"material": "TiO2", "core_loss_eV": "456"})
```

Or, from Python directly:

```python
from nuance_mcp.core.skills import register_skills
# already done by build_server(); just shown for reference
```

The agent will unroll the prompt into a sequence of validated tool
calls and report results at the end.

## 6. Run the example scripts

The repo ships eight end-to-end examples. Every example runs against
the simulator by default and accepts `--adapter gatan` (or `jeol`) to
hit live hardware:

```bash
python examples/01_basic_query.py
python examples/02_tem_acquisition.py --adapter gatan
python examples/03_eels_workflow.py
python examples/04_4dstem_analysis.py --adapter gatan
python examples/05_tilt_series.py --adapter gatan
python examples/06_diffraction_dspacing.py --adapter gatan
python examples/07_voice_acquisition.py --adapter gatan
python examples/08_voice_confirmed_stage_moves.py --adapter gatan \
    --transcript "Tilt the stage to plus 45 degrees alpha."
```

## Next

- [Architecture](architecture.md) — the four-layer design.
- [Adapters](adapters/) — adapter-specific docs (Gatan, JEOL, Hitachi).
- [Tool reference](tools_reference.md) — all 30 typed tools.
- [Safety and deployment](safety.md) — bounded execution vs facility safety.
- [Contributing a new vendor adapter](contributing_a_vendor.md).
