# Contributing a vendor adapter

Adding a new vendor — Thermo Fisher, Oxford, or anyone else — is
implemented as a subclass of `MicroscopeAdapter`. The framework does
not need to be modified.

## Checklist

1. **Pick a name.** Lower-case, short, unique among installed
   adapters. Examples: `thermo`, `oxford`, `delong`.
2. **Subclass `MicroscopeAdapter`.** Set `vendor`, `model`,
   `capabilities`, `bridge_required`, `is_thread_safe`.
3. **Implement `open()`, `close()`, and `get_state()`.** These are
   abstract; everything else has a default that raises
   `CapabilityUnavailable`. Implement only the methods whose
   capability you advertised.
4. **(Optional) Write a bridge plugin** if your vendor API loads only
   inside the acquisition host process. Speak the JSON contract
   documented in
   [`docs/spec/nuance-mcp-bridge-1.0.md`](spec/nuance-mcp-bridge-1.0.md).
5. **Register the adapter** through your package's entry points:

   ```toml
   [project.entry-points."nuance_mcp.adapters"]
   thermo = "thermo_nuance_mcp:ThermoAdapter"
   ```

   After `pip install thermo-nuance-mcp`, the adapter is
   discoverable as `nuance-mcp --adapter thermo`.
6. **Add structural tests.** Reuse
   [`tests/test_adapter_contract.py`](https://github.com/NUANCE-IT/nuance-mcp/blob/main/tests/test_adapter_contract.py)
   as a template; the structural assertions there apply to every
   adapter. Live-hardware tests can be gated behind an environment
   variable.
7. **Add a docs page** under `docs/adapters/<name>.md` covering the
   declared capabilities, installation, online/offline behaviour,
   and any vendor-specific caveats.

## Minimal example

```python
# thermo_nuance_mcp.py
from nuance_mcp.core.adapter import MicroscopeAdapter, MicroscopeState
from nuance_mcp.core.capabilities import Capability


class ThermoAdapter(MicroscopeAdapter):
    vendor = "Thermo Fisher"
    model = "Velox / iSciter"
    bridge_required = False
    is_thread_safe = False
    capabilities = frozenset({
        Capability.TEM, Capability.STEM, Capability.EELS,
        Capability.EDS, Capability.DIFFRACTION,
        Capability.STAGE, Capability.OPTICS,
    })

    def open(self) -> None:
        import velox    # vendor SDK
        self._svc = velox.connect()

    def close(self) -> None:
        self._svc = None

    def get_state(self) -> MicroscopeState:
        s = self._svc.column_state()
        return MicroscopeState(
            vendor=self.vendor, model=self.model,
            high_tension_kV=s.kv, mode=s.mode_name,
            magnification=s.mag, stage_x_um=s.x, stage_y_um=s.y,
            stage_alpha_deg=s.alpha,
        )

    def get_front_image(self, include_data, include_tags) -> dict:
        img = self._svc.last_image()
        return {"name": img.name, "shape": list(img.shape)}

    def acquire_tem(self, exposure_s, binning, processing, roi):
        ...
```

Drop this into a separate PyPI package, declare the entry point, and
your users can install both NUANCE-MCP and your adapter with a
single `pip install`.

## When to put the adapter in this repo vs your own

| In-tree                                           | Out-of-tree                                                |
|---------------------------------------------------|------------------------------------------------------------|
| Vendor SDK is open-source / Python-installable    | Vendor SDK requires a licence or vendor-specific installer |
| Maintainer wants NUANCE-MCP CI to cover it    | Maintainer wants independent release cadence               |
| Adapter is < ~500 LOC                             | Adapter ships its own bridge plugin, examples, docs        |

The Gatan and JEOL adapters are in-tree because their dependencies
are widely available and the maintainers (NUANCE) ship the framework
itself. A Thermo Fisher adapter would more naturally live in a
separate `thermo-nuance-mcp` package because Velox installation
is licensed.

## Bridge implementations

If your vendor API needs a bridge, mirror
`nuance_mcp/adapters/gatan/bridge.py`:

- Bind ZeroMQ REP on `127.0.0.1:5555` by default (or the next free
  port; document it in your adapter's docs page).
- Implement the `hello` handshake; advertise your capability list.
- Dispatch methods by name; method names match `MicroscopeAdapter`.
- Pump the vendor's GUI event loop at least every 100 ms while
  waiting.
- Return either `{"status": "ok", "result": …}` or `{"status":
  "error|unsupported", "error": …}`.

The contract is fully specified in
[`docs/spec/nuance-mcp-bridge-1.0.md`](spec/nuance-mcp-bridge-1.0.md);
follow it verbatim and your bridge is interoperable with the
reference clients.
