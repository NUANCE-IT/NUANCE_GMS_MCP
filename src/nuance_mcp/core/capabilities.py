"""Vendor-agnostic capability vocabulary.

Each :class:`Capability` represents a *family* of operations that an adapter
may declare. The schema layer never inspects vendor-specific behaviour; it
only checks ``capability in adapter.capabilities`` before dispatching a tool
that requires that family.

Capabilities are intentionally coarse-grained. A finer-grained probe (e.g.
"is iDPC enabled on this CEOS corrector?") is the responsibility of the
adapter itself, surfaced as part of its diagnostics payload.
"""

from __future__ import annotations
from enum import Enum
from typing import Iterable


class Capability(str, Enum):
    """Coarse capability families that an adapter may declare."""

    # Imaging modalities
    TEM = "tem"
    STEM = "stem"
    STEM_HAADF = "stem.haadf"
    STEM_BF = "stem.bf"
    STEM_ABF = "stem.abf"
    FOURD_STEM = "4dstem"
    EELS = "eels"
    EDS = "eds"
    DIFFRACTION = "diffraction"
    TILT_SERIES = "tilt_series"

    # Column control
    STAGE = "stage"
    STAGE_TILT = "stage.tilt"
    OPTICS = "optics"
    DETECTORS = "detectors"
    BEAM_BLANKER = "beam_blanker"
    APERTURES = "apertures"
    HT = "ht"
    FEG = "feg"

    # Derived analyses
    RADIAL_PROFILE = "analysis.radial_profile"
    MAX_FFT = "analysis.max_fft"
    IMAGE_FILTER = "analysis.image_filter"
    COM_DPC = "analysis.com_dpc"
    MAX_SPOT_MAP = "analysis.max_spot_map"
    SCRIPT_TEMPLATE = "analysis.script_template"

    # Server-side primitives
    LIVE_JOBS = "live_jobs"
    WORKSPACE = "workspace"

    @classmethod
    def from_strings(cls, names: Iterable[str]) -> "frozenset[Capability]":
        out = set()
        for n in names:
            try:
                out.add(cls(n))
            except ValueError:
                # Unknown capability — silently ignore so older clients
                # can talk to newer adapters that advertise new families.
                pass
        return frozenset(out)
