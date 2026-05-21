"""The vendor-agnostic :class:`MicroscopeAdapter` contract.

This module defines the *only* surface that the schema, lifecycle, skill, and
simulator layers depend on. A vendor backend is a Python class that:

  1. inherits from :class:`MicroscopeAdapter`
  2. implements the abstract methods it declares as supported in
     :attr:`MicroscopeAdapter.capabilities`
  3. is registered through an entry point under the
     ``nuance_mcp.adapters`` group, so::

         pip install nuance-mcp[gatan]
         nuance-mcp --adapter gatan

     discovers and instantiates it without import-time vendor dependencies
     in the core package.

The contract intentionally accepts and returns *pre-validated* values:
Pydantic validation is performed by the FastMCP tool layer in
``nuance_mcp.tools.*`` *before* the adapter is touched. Adapters may
still reject calls (e.g.\ if the underlying instrument refuses), but they
do not duplicate schema validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .capabilities import Capability


# -----------------------------------------------------------------------------
# Lightweight value objects (not Pydantic — those live in schemas.py)
# -----------------------------------------------------------------------------


@dataclass
class MicroscopeState:
    """Snapshot of the column. Vendor adapters fill what they know."""

    vendor: str
    model: str
    high_tension_kV: Optional[float] = None
    mode: Optional[str] = None  # "TEM" | "STEM" | "DIFFRACTION" | ...
    magnification: Optional[float] = None
    spot_size: Optional[int] = None
    brightness: Optional[float] = None
    focus_um: Optional[float] = None
    stage_x_um: Optional[float] = None
    stage_y_um: Optional[float] = None
    stage_z_um: Optional[float] = None
    stage_alpha_deg: Optional[float] = None
    stage_beta_deg: Optional[float] = None
    beam_shift_x: Optional[float] = None
    beam_shift_y: Optional[float] = None
    beam_tilt_x: Optional[float] = None
    beam_tilt_y: Optional[float] = None
    illumination_mode: Optional[str] = None
    detector_state: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageReturn:
    """Generic acquisition payload."""

    data: np.ndarray
    name: str
    pixel_size_nm: Optional[float] = None
    pixel_unit: str = "nm"
    exposure_s: Optional[float] = None
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpectrumReturn:
    """1-D spectrum (EELS / EDS)."""

    counts: np.ndarray  # shape (N,)
    energy_eV: np.ndarray  # shape (N,) — calibrated x-axis
    name: str
    exposure_s: Optional[float] = None
    dispersion_eV_per_ch: Optional[float] = None
    tags: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# The abstract adapter
# -----------------------------------------------------------------------------


class CapabilityUnavailable(NotImplementedError):
    """Raised when an adapter is asked to perform an unsupported operation."""


class MicroscopeAdapter(ABC):
    """Vendor-agnostic interface to a microscopy column.

    A concrete adapter declares its *capabilities* (which operation families
    it supports) and implements the corresponding methods. Methods whose
    capability is not declared may simply ``raise CapabilityUnavailable``.

    The adapter does not validate arguments — it receives values already
    constrained by the schema layer. It is responsible for:

      * translating those values into the vendor's native API,
      * returning structured results in the shapes documented below,
      * raising :class:`CapabilityUnavailable` for unsupported families,
      * being safe to call from multiple threads if it advertises
        ``LIVE_JOBS`` (see :class:`MicroscopeAdapter.is_thread_safe`).
    """

    # ------------------------------------------------------------------
    # Class-level metadata
    # ------------------------------------------------------------------
    vendor: str = "unknown"
    model: str = "unknown"
    bridge_required: bool = False
    is_thread_safe: bool = False
    capabilities: frozenset[Capability] = frozenset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __init__(self, **kwargs) -> None:
        """Subclasses can take vendor-specific kwargs (endpoint, mode, ...)."""

    @abstractmethod
    def open(self) -> None:
        """Establish whatever connection the adapter needs (idempotent)."""

    @abstractmethod
    def close(self) -> None:
        """Tear down resources. Idempotent."""

    def supports(self, cap: Capability) -> bool:
        return cap in self.capabilities

    def _require(self, cap: Capability) -> None:
        if cap not in self.capabilities:
            raise CapabilityUnavailable(
                f"adapter {self.vendor}/{self.model} does not support {cap.value}"
            )

    # ==================================================================
    # Diagnostics  (always required)
    # ==================================================================
    @abstractmethod
    def get_state(self) -> MicroscopeState: ...

    @abstractmethod
    def get_front_image(self, include_data: bool, include_tags: bool) -> dict: ...

    def get_image_shift(self) -> dict:
        self._require(Capability.OPTICS)
        raise CapabilityUnavailable("image-shift readback not implemented")

    def workspace_list_images(self) -> list[dict]:
        self._require(Capability.WORKSPACE)
        return []

    # ==================================================================
    # Acquisition
    # ==================================================================
    def acquire_tem(
        self, exposure_s: float, binning: int, processing: int, roi: Optional[list[int]]
    ) -> ImageReturn:
        self._require(Capability.TEM)
        raise CapabilityUnavailable("acquire_tem not implemented")

    def acquire_stem(
        self,
        width: int,
        height: int,
        dwell_us: float,
        rotation_deg: float,
        signals: list[int],
    ) -> ImageReturn:
        self._require(Capability.STEM)
        raise CapabilityUnavailable("acquire_stem not implemented")

    def acquire_4d_stem(
        self,
        scan_x: int,
        scan_y: int,
        dwell_us: float,
        camera_length_mm: Optional[float],
        convergence_mrad: Optional[float],
    ) -> ImageReturn:
        self._require(Capability.FOURD_STEM)
        raise CapabilityUnavailable("acquire_4d_stem not implemented")

    def acquire_eels(
        self,
        exposure_s: float,
        energy_offset_eV: float,
        slit_width_eV: float,
        dispersion_idx: int,
        full_vertical_binning: bool,
    ) -> SpectrumReturn:
        self._require(Capability.EELS)
        raise CapabilityUnavailable("acquire_eels not implemented")

    def acquire_diffraction(
        self, exposure_s: float, camera_length_mm: Optional[float], binning: int
    ) -> ImageReturn:
        self._require(Capability.DIFFRACTION)
        raise CapabilityUnavailable("acquire_diffraction not implemented")

    # ==================================================================
    # Stage / Optics / Detectors
    # ==================================================================
    def get_stage_position(self) -> dict:
        self._require(Capability.STAGE)
        raise CapabilityUnavailable("get_stage_position not implemented")

    def set_stage_position(self, **kwargs) -> dict:
        self._require(Capability.STAGE)
        raise CapabilityUnavailable("set_stage_position not implemented")

    def set_beam_parameters(self, **kwargs) -> dict:
        self._require(Capability.OPTICS)
        raise CapabilityUnavailable("set_beam_parameters not implemented")

    def configure_detectors(self, **kwargs) -> dict:
        self._require(Capability.DETECTORS)
        raise CapabilityUnavailable("configure_detectors not implemented")

    def set_magnification(self, magnification: float) -> dict:
        self._require(Capability.OPTICS)
        raise CapabilityUnavailable("set_magnification not implemented")

    def set_image_shift(self, x: float, y: float) -> dict:
        self._require(Capability.OPTICS)
        raise CapabilityUnavailable("set_image_shift not implemented")

    def set_brightness(self, value: float) -> dict:
        self._require(Capability.OPTICS)
        raise CapabilityUnavailable("set_brightness not implemented")

    def change_focus_relative(self, delta_um: float) -> dict:
        self._require(Capability.OPTICS)
        raise CapabilityUnavailable("change_focus_relative not implemented")

    def stop_stage(self) -> dict:
        self._require(Capability.STAGE)
        raise CapabilityUnavailable("stop_stage not implemented")

    def set_condenser_stigmation(self, x: float, y: float) -> dict:
        self._require(Capability.OPTICS)
        raise CapabilityUnavailable("set_condenser_stigmation not implemented")

    # ==================================================================
    # Derived analyses
    # ==================================================================
    def apply_image_filter(self, **kwargs) -> ImageReturn:
        self._require(Capability.IMAGE_FILTER)
        raise CapabilityUnavailable("apply_image_filter not implemented")

    def compute_radial_profile(self, **kwargs) -> dict:
        self._require(Capability.RADIAL_PROFILE)
        raise CapabilityUnavailable("compute_radial_profile not implemented")

    def compute_max_fft(self, **kwargs) -> ImageReturn:
        self._require(Capability.MAX_FFT)
        raise CapabilityUnavailable("compute_max_fft not implemented")

    def run_4dstem_analysis(self, **kwargs) -> dict:
        self._require(Capability.COM_DPC)
        raise CapabilityUnavailable("run_4dstem_analysis not implemented")

    def run_4dstem_maximum_spot_mapping(self, **kwargs) -> ImageReturn:
        self._require(Capability.MAX_SPOT_MAP)
        raise CapabilityUnavailable("run_4dstem_maximum_spot_mapping not implemented")

    def run_script_template(self, template: str, params: dict) -> dict:
        self._require(Capability.SCRIPT_TEMPLATE)
        raise CapabilityUnavailable("run_script_template not implemented")

    # ==================================================================
    # Workflow / Lifecycle (live processing)
    # ==================================================================
    def acquire_tilt_series(self, **kwargs) -> dict:
        self._require(Capability.TILT_SERIES)
        raise CapabilityUnavailable("acquire_tilt_series not implemented")

    def start_live_processing_job(self, **kwargs) -> dict:
        self._require(Capability.LIVE_JOBS)
        raise CapabilityUnavailable("start_live_processing_job not implemented")

    def get_live_processing_job_status(self, job_id: str) -> dict:
        self._require(Capability.LIVE_JOBS)
        raise CapabilityUnavailable("get_live_processing_job_status not implemented")

    def get_live_processing_job_result(
        self, job_id: str, include_data: bool = False
    ) -> dict:
        self._require(Capability.LIVE_JOBS)
        raise CapabilityUnavailable("get_live_processing_job_result not implemented")

    def stop_live_processing_job(self, job_id: str) -> dict:
        self._require(Capability.LIVE_JOBS)
        raise CapabilityUnavailable("stop_live_processing_job not implemented")
