"""Contract tests — run against every adapter, no vendor hardware required.

These tests assert structural properties that every MicroscopeAdapter must
satisfy. The simulator adapter is run live; the Gatan, JEOL, and Hitachi
adapters are introspected without instantiation (since their vendor
backends are not importable on CI).
"""

from __future__ import annotations

import pytest

from nuance_mcp.core import (
    MicroscopeAdapter, SimulatorAdapter, Capability, CapabilityUnavailable,
)
from nuance_mcp.adapters import (
    available_adapters, load_adapter,
)


# ---------------------------------------------------------------------------
# Properties of the ABC
# ---------------------------------------------------------------------------

def test_abc_has_required_abstracts():
    abstracts = MicroscopeAdapter.__abstractmethods__
    for must in ("open", "close", "get_state", "get_front_image"):
        assert must in abstracts


def test_all_builtin_adapters_resolve():
    names = available_adapters()
    assert {"simulator", "gatan", "jeol", "hitachi"}.issubset(set(names))
    for name in names:
        cls = load_adapter(name)
        assert issubclass(cls, MicroscopeAdapter), name


def test_every_adapter_declares_vendor_and_capabilities():
    for name in available_adapters():
        cls = load_adapter(name)
        assert cls.vendor != "unknown", f"{name}: vendor not set"
        assert isinstance(cls.capabilities, frozenset), name
        for cap in cls.capabilities:
            assert isinstance(cap, Capability), f"{name}: bad cap {cap!r}"


# ---------------------------------------------------------------------------
# Live tests against the simulator
# ---------------------------------------------------------------------------

@pytest.fixture
def sim():
    a = SimulatorAdapter(seed=42)
    a.open()
    yield a
    a.close()


def test_simulator_state(sim):
    s = sim.get_state()
    assert s.vendor == "simulator"
    assert s.mode == "TEM"
    assert s.stage_alpha_deg == 0.0


def test_simulator_acquire_tem(sim):
    img = sim.acquire_tem(exposure_s=0.5, binning=1,
                          processing=3, roi=None)
    assert img.data.ndim == 2
    assert img.name == "SimTEM"


def test_simulator_stage_setter_roundtrip(sim):
    sim.set_stage_position(alpha_deg=12.5)
    assert sim.get_stage_position()["alpha_deg"] == 12.5


def test_simulator_eels_spectrum_shape(sim):
    spec = sim.acquire_eels(exposure_s=0.5, energy_offset_eV=400.0,
                            slit_width_eV=10.0, dispersion_idx=1,
                            full_vertical_binning=True)
    assert spec.counts.shape == spec.energy_eV.shape
    assert spec.dispersion_eV_per_ch == 0.25


def test_simulator_live_job_lifecycle(sim):
    rec = sim.start_live_processing_job(job_type="radial_profile",
                                         poll_interval_s=0.1)
    jid = rec["job_id"]
    st = sim.get_live_processing_job_status(jid)
    assert st["state"] in ("running", "result_ready", "pending")
    sim.stop_live_processing_job(jid)


def test_unsupported_capability_is_structured():
    a = SimulatorAdapter()
    a.open()
    # The simulator advertises COM/DPC, so this should *not* raise.
    # Pick a capability the simulator does NOT advertise:
    assert Capability.EDS not in a.capabilities
    with pytest.raises(CapabilityUnavailable):
        a._require(Capability.EDS)
