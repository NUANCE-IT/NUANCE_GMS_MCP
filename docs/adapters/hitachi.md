# Hitachi adapter (`HitachiAdapter`) — skeleton

A placeholder is shipped so that:

1. Skill authors can declare a capability dependency on Hitachi
   features and write protocols that target them.
2. CI can exercise the structural-test path against a Hitachi-typed
   adapter even when no SDK is installed.
3. A clear hook exists for a maintainer (or vendor) to fill in.

## Declared capabilities

The skeleton advertises the families we expect a Hitachi HT/HF-series
adapter to support once implemented:

`tem`, `stem`, `diffraction`, `stage`, `stage.tilt`, `optics`,
`detectors`, `tilt_series`.

Every method currently raises `CapabilityUnavailable`, which the
tool layer surfaces as
`{"status": "UNSUPPORTED", "reason": "Hitachi adapter is a placeholder. ..."}`.

## Where to add the implementation

```
nuance_mcp/adapters/hitachi/
├── __init__.py        ← re-exports HitachiAdapter
└── adapter.py         ← the placeholder; replace open()/close() and
                          implement the method stubs.
```

Hitachi's `HRTEM Live` / `Stem Remote` SDKs are host-process-bound
(similar to Gatan/DM). The expected pattern is therefore:

1. Set `bridge_required = True`.
2. Write a small bridge plugin under
   `nuance_mcp/adapters/hitachi/bridge.py` that runs inside the
   Hitachi control software and speaks the same
   `nuance-mcp-bridge/1.0` JSON contract used by Gatan. The
   Hitachi SDK calls live there.
3. The adapter (this file) becomes a thin ZeroMQ client analogous to
   `adapters/gatan/adapter.py`.

If your Hitachi SDK loads in-process (some HT-series builds do), the
JEOL adapter is the closer model and no bridge is required.

## Contributing

PRs welcome. See [`contributing_a_vendor.md`](../contributing_a_vendor.md)
for the full checklist (capability declaration, schema bounds, tests,
docs).
