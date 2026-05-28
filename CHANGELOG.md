# Changelog

All notable changes to `nuance-mcp` are documented in this file. The
format is loosely based on [Keep a Changelog]; releases follow
[Semantic Versioning].

[Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html

## 0.2.0 — May 2026

### Highlights

This release **renames the package from `nuance-gms-mcp` / `jeol-mcp` to
`nuance-mcp`** and reorganizes the codebase as a vendor-agnostic
framework with pluggable adapters. The schema-bound tool layer, persistent
live-job lifecycle, declarative skill catalog, and physics-plausible
simulator now sit above an explicit `MicroscopeAdapter` contract. The two
v0.1 codebases — Gatan/GMS and JEOL/PyJEM — become reference adapters under
this contract; Hitachi is included as a skeleton.

### Added

- **Project-status tooling.** `scripts/project_status.py` is a live probe
  that measures the protocol surface (typed tools, skills, live-job types,
  capabilities) from the running server, runs the test suite, reads git
  state, and prints an 11-milestone roadmap. It backs a new `/goals`
  Claude Code slash command (`.claude/commands/goals.md`) and runs as a
  `Project status probe` step in CI; it exits non-zero on a red suite so
  the slash command, the command line, and CI share one source of truth.
  Documented in `docs/project_status.md`; smoke-tested in
  `tests/test_status.py`.
- **`nuance_mcp.core.adapter.MicroscopeAdapter`** — abstract base class
  defining every typed operation as a vendor-neutral method.
- **`nuance_mcp.core.capabilities.Capability`** — 24-entry vocabulary
  for adapter capability declarations; the tool layer translates absent
  capabilities into structured `{"status": "UNSUPPORTED"}` responses.
- **`nuance_mcp.core.simulator.SimulatorAdapter`** — default in-process
  physics-plausible backend. CI runs against this; teaching deployments
  can ship with it as the only adapter.
- **`nuance_mcp.adapters.gatan.GatanGMSAdapter`** — reference adapter
  for Gatan Microscopy Suite 3.60 over a versioned ZeroMQ bridge.
- **`nuance_mcp.adapters.gatan.bridge`** — host-side plugin (renamed
  from `gms_mcp.dm_plugin`); now speaks `nuance-mcp-bridge/1.0`.
- **`nuance_mcp.adapters.jeol.JEOLAdapter`** — reference adapter for
  JEOL TEM3 over PyJEM (online) or PyJEM.offline (developer laptops). No
  bridge required because PyJEM loads in-process.
- **`nuance_mcp.adapters.hitachi.HitachiAdapter`** — skeleton with
  declared capability set; raises `CapabilityUnavailable` on every method
  until a maintainer wires it up.
- **CLI** — `nuance-mcp serve --adapter {gatan|jeol|simulator|...}`,
  `nuance-mcp list-adapters`.
- **Entry points** — third-party adapters self-register through the
  `nuance_mcp.adapters` entry-point group.
- **Bridge spec** — `docs/spec/nuance-mcp-bridge-1.0.md` documents the
  JSON contract end-to-end (envelope, handshake, capability negotiation,
  image/spectrum/live-job encodings, error semantics).
- **Contract tests** — `tests/test_adapter_contract.py` runs against every
  adapter and asserts the structural invariants.

### Changed

- **Tool names lose the `gms_` prefix.** `gms_acquire_tem_image` is now
  `acquire_tem_image`; `gms_get_microscope_state` → `get_microscope_state`;
  and so on for all 30 tools. Legacy names are accepted as aliases for one
  deprecation cycle.
- **Bridge default binding is now loopback only**
  (`tcp://127.0.0.1:5555`). Facility-LAN exposure is opt-in via
  `NUANCE_MCP_BRIDGE_BIND` (or the vendor-specific equivalent
  `GMS_MCP_ZMQ_BIND`).
- **`DMSimulator` is renamed `MicroscopeSimulator`** and moved to
  `nuance_mcp.core.simulator`. The Gatan-specific quirks (return-shape
  oddities mirroring `DM.*` calls) live under
  `nuance_mcp.adapters.gatan.simulator` instead.
- **Skill bodies now use vendor-neutral tool names.** Every skill starts
  with a `get_capabilities` check so absent vendor support produces a
  clear failure mode rather than an opaque tool error.
- **Manuscript framing** — the paper now treats the system as
  NUANCE-MCP with Gatan, JEOL, Hitachi, and simulator as the four
  reference adapters; GMS is the first reference target, not the only
  one.

### Migration

See `docs/migration_v0.1_to_v0.2.md`. The short version:

```python
# v0.1 (gms_mcp)
from gms_mcp.server import build_server
server = build_server()                          # GMS implicit

# v0.2 (nuance_mcp)
from nuance_mcp import build_server
server = build_server("gatan")                   # explicit adapter
```

Tool names are renamed but legacy aliases keep older clients working.
The host-side bridge plugin moves to
`nuance_mcp/adapters/gatan/bridge.py`; the JSON contract is now
versioned and the bridge advertises capabilities on `hello`.

## 0.1.2 — April 2026

Final v0.1 release before the rename. 30 typed tools, 5 live-job classes,
6 skills, 132/135 automated tests passing. See the v0.1
[CHANGELOG](https://github.com/NUANCE-IT/NUANCE_GMS_MCP/blob/v0.1.2/CHANGELOG.md)
for the full history.
