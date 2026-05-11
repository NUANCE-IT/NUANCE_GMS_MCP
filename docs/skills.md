# Skills (MCP prompts)

Skills are declarative, parameterised multi-step protocols exposed as
MCP **prompts**. They are first-class MCP primitives alongside tools,
so any MCP-compatible client (Claude.ai, Ollama via LangChain) can
discover and invoke them by name.

Skills are *vendor-portable*. Each skill body uses the
vendor-neutral tool names exclusively and begins with a
`get_capabilities` check, so the same skill unrolls into identical
tool sequences regardless of which adapter is mounted.

## Catalogue

| Skill                        | Arguments                                  | Protocol                                                                                  |
|------------------------------|--------------------------------------------|-------------------------------------------------------------------------------------------|
| `eels_survey`                | `material`, `core_loss_eV`                 | Verify caps → ZLP reference → core-loss spectrum centred at `core_loss_eV` → edge ID      |
| `tilt_series_protocol`       | `start_deg`, `end_deg`, `step_deg`, `save_dir` | Pre-flight stage check → `acquire_tilt_series` → post-flight stability report          |
| `4dstem_characterization`    | `scan_size`, `material`, `convergence_mrad`| `acquire_4d_stem` → vBF/HAADF/CoM/DPC → optional `run_4dstem_maximum_spot_mapping`        |
| `beam_alignment`             | *(none)*                                   | State check → centring → HRTEM + FFT inspection → stigmation adjustments → final report   |
| `hrtem_imaging`              | `material`, `zone_axis`                    | Survey → HRTEM → FFT → `compute_radial_profile` → phase match                             |
| `diffraction_survey`         | `material`, `camera_length_mm`             | `acquire_diffraction` → radial profile → phase match against `material` reference         |

## Invoking a skill

From an MCP client:

```python
prompt = await client.get_prompt(
    "eels_survey",
    {"material": "TiO2", "core_loss_eV": "456"},
)
# `prompt` is the unrolled instruction string; the agent executes it
# step-by-step using the typed tool calls referenced inside.
```

From the FastMCP server side:

```python
from nuance_mcp import build_server
server = build_server("gatan")
text = (await server.get_prompt("eels_survey",
                                  {"material": "TiO2",
                                   "core_loss_eV": "456"})).messages[0].content.text
```

In Claude.ai, skills appear in the prompt picker alongside other MCP
prompts; just click the one you want.

## Skill anatomy

Every skill body has the same shape:

```text
Step 1 — Verify capabilities
  Call: get_capabilities
  Check: required capability families.

Step 2 — Verify instrument state
  Call: get_microscope_state

Step 3..N — Tool sequence with explicit arguments

Step (last) — Report
  Summarise results, recommend next step.
```

This shape is what makes skills auditable: the agent's trace becomes
`(skill_name, arguments, ordered tool calls, results)`, which is
richer provenance than tool calls alone and amenable to FAIR-style
recording.

## Adding a new skill

Skills are plain Python functions decorated with `@mcp.prompt`.
To add one, edit `nuance_mcp/core/skills.py`:

```python
@mcp.prompt(name="my_protocol",
            description="One-sentence summary.")
def my_protocol(material: str = "unknown",
                target_kV: str = "200") -> str:
    return f"""You are running my custom protocol on {material}.

Step 1 — Verify capabilities
  Call: get_capabilities
  Required: 'tem', 'eels'.

Step 2 — ...
"""
```

MCP prompt arguments are strings by spec; numeric arguments are coerced
inside the skill function. Skills should not call tools directly; they
return text that the agent unrolls into tool calls.
