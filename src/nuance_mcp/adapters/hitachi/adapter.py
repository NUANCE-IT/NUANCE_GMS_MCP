"""Hitachi adapter (skeleton).

Hitachi's HT-series and HF-series TEMs expose a remote control protocol
(HRTEM Live / Stem Remote) and a Python wrapper that historically required
a per-instrument SDK licence. The adapter below registers the *intended*
capabilities so skills can fail fast with a structured error message until
the implementation lands.
"""

from __future__ import annotations

from ...core.adapter import MicroscopeAdapter, MicroscopeState, CapabilityUnavailable
from ...core.capabilities import Capability


class HitachiAdapter(MicroscopeAdapter):
    """Skeleton Hitachi adapter — raises :class:`CapabilityUnavailable` for
    every advertised capability until a maintainer wires it up."""

    vendor = "Hitachi"
    model = "HT/HF-series"
    bridge_required = True       # Hitachi SDK is typically host-bound
    is_thread_safe = False
    capabilities = frozenset({
        Capability.TEM, Capability.STEM, Capability.DIFFRACTION,
        Capability.STAGE, Capability.STAGE_TILT, Capability.OPTICS,
        Capability.DETECTORS, Capability.TILT_SERIES,
    })

    def open(self) -> None:
        raise CapabilityUnavailable(
            "Hitachi adapter is a placeholder. Contributions welcome — "
            "implement Hitachi SDK calls in adapters/hitachi/adapter.py."
        )

    def close(self) -> None:
        pass

    def get_state(self) -> MicroscopeState:
        raise CapabilityUnavailable("Hitachi adapter not implemented")

    def get_front_image(self, include_data, include_tags) -> dict:
        raise CapabilityUnavailable("Hitachi adapter not implemented")
