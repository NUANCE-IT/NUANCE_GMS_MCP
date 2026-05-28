# Safety and deployment

`nuance-mcp` provides **bounded execution at the tool boundary** —
the lowest layer of a layered safety model. This document is explicit
about what that does and does not cover, and gives concrete deployment
guidance.

## What bounded execution means

Every typed tool validates its argument object against a Pydantic v2
schema before the adapter is touched. The schema encodes physical
and operational bounds:

| Example                 | Bound                              |
|-------------------------|------------------------------------|
| Stage α tilt            | `[-80°, +80°]`                     |
| Stage β tilt            | `[-30°, +30°]`                     |
| Stage XY range          | `[-5000, +5000] µm`                |
| TEM exposure            | `[10⁻³, 60] s`                     |
| STEM dwell time         | `[0.5, 10⁴] µs`                    |
| EELS energy offset      | `[-200, 3000] eV`                  |
| ROI                     | 4-element, monotone, in-image      |

Out-of-bound requests fail at the boundary with a structured
`ValidationError` and **never reach the instrument**. The behavior
is independent of how the calling agent phrased the request.

This is a useful property — but it is *not* facility safety.

## What bounded execution does **not** cover

Schema validation **does not** replace:

- **Hardware interlocks.** Beam-blanker activation, column-vacuum
  interlocks, stage limit switches, and detector-insertion interlocks
  remain the property of the instrument and its vendor, and are
  unchanged by the agent stack.
- **Operator approval.** For high-risk or unusual operations,
  facility policy may require human-in-the-loop confirmation. See
  example 08 for the pattern.
- **Permission tiers and authentication.** The default loopback
  binding combined with optional facility-VLAN deployment does not
  by itself implement user-authentication or role-based access.
  Production deployments should add authentication at the transport
  layer (HTTPS reverse proxy with API keys or OAuth) and tier tools
  by risk class.
- **Beam-damage and sample-policy models.** Bounded exposures and
  dwell times are not the same as a calibrated beam-damage model for
  a specific specimen. Damage-aware agents will require additional,
  specimen-specific policy gates layered above the schema.
- **Facility-specific policies.** Acquisition limits, instrument-time
  windows, and data-retention rules are facility policy, not
  protocol. They are best enforced by wrapping the FastMCP server
  with a facility-specific policy decorator.

## Recommended deployment modes

| Mode                          | Default binding                                  | Required guards                                |
|-------------------------------|--------------------------------------------------|------------------------------------------------|
| Single workstation (default)  | `127.0.0.1:5555` (loopback)                      | none (single-user)                             |
| Facility VLAN                 | `10.x.y.z:5555` via `GMS_MCP_ZMQ_BIND`           | host firewall + IP allow-list                  |
| Remote MCP client             | HTTPS reverse proxy in front of `/mcp`           | TLS, API keys / OAuth, audit log               |

The protocol layer itself does not implement authentication or
authorisation; those layers are the deployer's responsibility.

## The human-confirmation pattern

For tools that change instrument state, the canonical pattern is:

```python
proposed = parse_intent(user_message)   # turn natural language into args
assert isinstance(proposed, dict)

# Human confirmation gate
if not operator_approves(proposed):
    return "Move declined; no hardware call issued."

# Dispatch through the validated tool
result = await server.call_tool("set_stage_position",
                                  {"payload": proposed})
```

This pattern is shown explicitly in
[`examples/08_voice_confirmed_stage_moves.py`](https://github.com/NUANCE-IT/nuance-mcp/blob/main/examples/08_voice_confirmed_stage_moves.py).
The confirmation step is **not** part of the protocol; the protocol
ensures that *if* a move is dispatched, its arguments are in-range.
Whether to require confirmation, and for which operations, is a
facility-policy decision.

## Audit trail

Every tool call (including those rejected by schema validation) emits
a structured record:

```json
{
  "tool":   "set_stage_position",
  "args":   {"alpha_deg": 95.0},
  "status": "ValidationError",
  "reason": "alpha_deg must be ≤ 80",
  "ts":     "2026-05-11T03:23:38Z"
}
```

These records are the basis for FAIR-style provenance. We
recommend persisting them to a facility logging system; the FastMCP
server itself does not retain them.
