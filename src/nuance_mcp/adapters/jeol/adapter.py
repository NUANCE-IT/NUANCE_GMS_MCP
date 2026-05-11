"""JEOL adapter implementation.

PyJEM is importable directly inside the JEOL TEMcenter Python interpreter
and ships an in-process ``PyJEM.offline`` simulator that is safe to import
anywhere. This adapter therefore does *not* require an external bridge —
the abstraction layer talks to PyJEM in-process, with online/offline
selected at ``open()`` time using the existing NUACE_JEOL_MCP pattern.

The implementation is a thin façade over the per-subsystem helpers used by
the standalone ``jeol_mcp`` package; in the unified repo those helpers live
next door under ``nuance_mcp.adapters.jeol._tem3`` (one file per JEOL
TEM3 subsystem: ``Stage3``, ``EOS3``, ``HT3``, ``Lens3``, …). The skeleton
below shows the structure and a few concrete methods; the migration brings
the rest of the v0.1 ``jeol_mcp.tools.*`` modules in unchanged.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Optional

from ...core.adapter import (
    MicroscopeAdapter, MicroscopeState, ImageReturn, SpectrumReturn,
    CapabilityUnavailable,
)
from ...core.capabilities import Capability


class JEOLAdapter(MicroscopeAdapter):
    """Adapter for JEOL TEM3 / PyJEM.

    Mode resolution (auto by default) mirrors the existing ``jeol_mcp``
    ``adapters.py`` logic: try ``PyJEM`` online first, fall back to
    ``PyJEM.offline`` if the COM driver is unavailable.
    """

    vendor = "JEOL"
    model = "TEM3 / PyJEM"
    bridge_required = False   # PyJEM runs in-process
    is_thread_safe = False
    capabilities = frozenset({
        Capability.TEM, Capability.STEM, Capability.STEM_HAADF,
        Capability.STEM_BF, Capability.STEM_ABF,
        Capability.EELS, Capability.EDS, Capability.DIFFRACTION,
        Capability.TILT_SERIES,
        Capability.STAGE, Capability.STAGE_TILT,
        Capability.OPTICS, Capability.DETECTORS,
        Capability.HT, Capability.FEG, Capability.APERTURES,
        Capability.BEAM_BLANKER,
        Capability.RADIAL_PROFILE, Capability.IMAGE_FILTER,
        Capability.LIVE_JOBS,
    })
    # Note: 4D-STEM not currently advertised — PyJEM exposes scan + camera
    # primitives but no integrated 4D-STEM acquisition mode. Will be added
    # once a vendor-side helper is available.

    def __init__(self, *, mode: str = "auto") -> None:
        super().__init__()
        self._mode = mode
        self._tem3 = None      # PyJEM[.offline].TEM3 module
        self._stage = None
        self._eos = None
        self._ht = None
        self._lens = None
        self._gun = None
        self._feg = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def open(self) -> None:
        backend = self._resolve_backend()
        if backend == "offline":
            self._tem3 = importlib.import_module("PyJEM.offline.TEM3")
            self._cam_mod = importlib.import_module("PyJEM.offline.detector")
        else:
            self._tem3 = importlib.import_module("PyJEM.TEM3")
            self._cam_mod = importlib.import_module("PyJEM.detector")

        # Memoised vendor singletons (mirrors jeol_mcp.adapters)
        self._stage = self._tem3.Stage3()
        self._eos   = self._tem3.EOS3()
        self._ht    = self._tem3.HT3()
        self._lens  = self._tem3.Lens3()
        self._gun   = self._tem3.GUN3()
        self._feg   = self._tem3.FEG3()

    def close(self) -> None:
        # PyJEM has no explicit shutdown; release references for GC.
        self._tem3 = None

    def _resolve_backend(self) -> str:
        requested = os.environ.get("NUANCE_MCP_JEOL_MODE", self._mode)
        if requested == "offline":
            return "offline"
        try:
            importlib.import_module("PyJEM.TEM3")
            return "online"
        except Exception:
            return "offline"

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_state(self) -> MicroscopeState:
        ht_kV = float(self._ht.GetHtValue()) / 1000.0
        mode_idx = self._eos.GetFunctionMode()[0]
        mode_name = {0: "TEM", 1: "STEM", 2: "DIFFRACTION"}.get(mode_idx, str(mode_idx))
        sx, sy, sz, sa, sb = self._stage.GetPos()
        return MicroscopeState(
            vendor=self.vendor, model=self.model,
            high_tension_kV=ht_kV, mode=mode_name,
            magnification=float(self._eos.GetMagValue()[0]),
            spot_size=int(self._eos.GetSpotSize()),
            stage_x_um=sx, stage_y_um=sy, stage_z_um=sz,
            stage_alpha_deg=sa, stage_beta_deg=sb,
            illumination_mode="Convergent" if mode_idx == 1 else "Parallel",
        )

    def get_front_image(self, include_data, include_tags) -> dict:
        # PyJEM front-image equivalent is the most recent detector acquisition.
        # Skeleton: return a placeholder so the test harness exercises the path.
        return {"status": "no front image cached"}

    # ------------------------------------------------------------------
    # Stage / Optics
    # ------------------------------------------------------------------
    def get_stage_position(self):
        sx, sy, sz, sa, sb = self._stage.GetPos()
        return {"x_um": sx, "y_um": sy, "z_um": sz,
                "alpha_deg": sa, "beta_deg": sb}

    def set_stage_position(self, **kw):
        if "x_um" in kw: self._stage.SetX(float(kw["x_um"]))
        if "y_um" in kw: self._stage.SetY(float(kw["y_um"]))
        if "z_um" in kw: self._stage.SetZ(float(kw["z_um"]))
        if "alpha_deg" in kw: self._stage.SetTiltXAngle(float(kw["alpha_deg"]))
        if "beta_deg" in kw:  self._stage.SetTiltYAngle(float(kw["beta_deg"]))
        return self.get_stage_position()

    def stop_stage(self):
        self._stage.Stop()
        return {"stopped": True}

    def set_magnification(self, magnification):
        # PyJEM uses MagValue index; we look up the closest match.
        # (skeleton — the helper that maps to nearest index lives in v0.1)
        self._eos.SetSelector(int(magnification))
        return {"magnification": float(self._eos.GetMagValue()[0])}

    # Other set/get methods follow the same pattern; they are stubbed here
    # and re-use the v0.1 jeol_mcp.tools.* implementations during migration.

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------
    def acquire_tem(self, exposure_s, binning, processing, roi):
        cam = self._cam_mod.Camera(0)
        cam.SetExposureTime(int(exposure_s * 1000))
        cam.SetBinning(binning)
        arr = cam.Acquire()
        return ImageReturn(data=arr, name="JEOL_TEM",
                           pixel_size_nm=None, exposure_s=exposure_s,
                           tags={"processing": processing, "binning": binning})

    def acquire_eels(self, exposure_s, energy_offset_eV, slit_width_eV,
                     dispersion_idx, full_vertical_binning):
        # Skeleton — PyJEM EELS access is via the Filter sub-module.
        raise CapabilityUnavailable(
            "JEOL EELS acquisition wiring is under development"
        )
