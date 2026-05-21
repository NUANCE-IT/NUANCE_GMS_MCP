"""Vendor-neutral core for nuance-mcp.

Public API:
    MicroscopeAdapter, MicroscopeState, ImageReturn, SpectrumReturn,
    CapabilityUnavailable, Capability,
    JobRegistry, JobState,
    SimulatorAdapter,
    register_skills,
"""

from .adapter import (
    MicroscopeAdapter,
    MicroscopeState,
    ImageReturn,
    SpectrumReturn,
    CapabilityUnavailable,
)
from .capabilities import Capability
from .lifecycle import JobRegistry, JobState, JobRecord
from .simulator import SimulatorAdapter
from .skills import register_skills

__all__ = [
    "MicroscopeAdapter",
    "MicroscopeState",
    "ImageReturn",
    "SpectrumReturn",
    "CapabilityUnavailable",
    "Capability",
    "JobRegistry",
    "JobState",
    "JobRecord",
    "SimulatorAdapter",
    "register_skills",
]
