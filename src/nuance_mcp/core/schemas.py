"""Pydantic v2 input schemas — vendor-neutral.

Every typed tool registered by :mod:`nuance_mcp.tools` validates its
argument object against one of the schemas defined here *before* dispatching
to the adapter. Bounds are physical, not vendor-specific: e.g. stage tilt is
±80°, exposure ∈ [10⁻³, 60] s. Adapters may further refine these limits at
runtime (e.g. an older stage that only tilts to ±60° will reject 70° at the
hardware level), but the schema layer is the *first* line of bounded
execution and is independent of the underlying vendor.
"""

from __future__ import annotations
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StrictModel(BaseModel):
    """Forbid unknown fields; strip whitespace from strings."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


def _validate_roi(v: Optional[list[int]]) -> Optional[list[int]]:
    if v is None:
        return None
    if len(v) != 4:
        raise ValueError("roi must have exactly 4 elements: [top, left, bottom, right]")
    t, l, b, r = v
    if t < 0 or l < 0 or b <= t or r <= l:
        raise ValueError("roi must define a positive [top, left, bottom, right] region")
    return v


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class FrontImageInput(_StrictModel):
    include_data: bool = Field(default=False)
    include_tags: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------


class AcquireTEMInput(_StrictModel):
    exposure_s: float = Field(default=1.0, ge=0.001, le=60.0)
    binning: int = Field(default=1, ge=1, le=8)
    processing: int = Field(default=3, ge=1, le=3)
    roi: Optional[list[int]] = Field(default=None)

    @field_validator("roi")
    @classmethod
    def _v_roi(cls, v):
        return _validate_roi(v)


class AcquireSTEMInput(_StrictModel):
    width: int = Field(default=512, ge=64, le=4096)
    height: int = Field(default=512, ge=64, le=4096)
    dwell_us: float = Field(default=10.0, ge=0.5, le=10_000.0)
    rotation_deg: float = Field(default=0.0, ge=-180.0, le=180.0)
    signals: list[int] = Field(default=[0, 1])


class Acquire4DSTEMInput(_StrictModel):
    scan_x: int = Field(default=64, ge=8, le=512)
    scan_y: int = Field(default=64, ge=8, le=512)
    dwell_us: float = Field(default=1000.0, ge=100.0, le=100_000.0)
    camera_length_mm: Optional[float] = Field(default=None, ge=20.0, le=2000.0)
    convergence_mrad: Optional[float] = Field(default=None, ge=0.1, le=50.0)


class AcquireEELSInput(_StrictModel):
    exposure_s: float = Field(default=1.0, ge=0.001, le=60.0)
    energy_offset_eV: float = Field(default=0.0, ge=-200.0, le=3000.0)
    slit_width_eV: float = Field(default=10.0, ge=0.0, le=100.0)
    dispersion_idx: int = Field(default=0, ge=0, le=3)
    full_vertical_binning: bool = Field(default=True)


class AcquireDiffractionInput(_StrictModel):
    exposure_s: float = Field(default=0.5, ge=0.001, le=60.0)
    camera_length_mm: Optional[float] = Field(default=None, ge=20.0, le=2000.0)
    binning: int = Field(default=1, ge=1, le=8)


# ---------------------------------------------------------------------------
# Stage / Optics / Detectors
# ---------------------------------------------------------------------------


class SetStageInput(_StrictModel):
    x_um: Optional[float] = Field(default=None, ge=-5000.0, le=5000.0)
    y_um: Optional[float] = Field(default=None, ge=-5000.0, le=5000.0)
    z_um: Optional[float] = Field(default=None, ge=-500.0, le=500.0)
    alpha_deg: Optional[float] = Field(default=None, ge=-80.0, le=80.0)
    beta_deg: Optional[float] = Field(default=None, ge=-30.0, le=30.0)


class SetBeamInput(_StrictModel):
    spot_size: Optional[int] = Field(default=None, ge=1, le=11)
    focus_um: Optional[float] = Field(default=None)
    shift_x: Optional[float] = Field(default=None)
    shift_y: Optional[float] = Field(default=None)
    tilt_x: Optional[float] = Field(default=None)
    tilt_y: Optional[float] = Field(default=None)
    obj_stig_x: Optional[float] = Field(default=None)
    obj_stig_y: Optional[float] = Field(default=None)


class SetDetectorInput(_StrictModel):
    insert_camera: Optional[bool] = Field(default=None)
    target_temp_c: Optional[float] = Field(default=None, ge=-60.0, le=30.0)
    haadf_enabled: Optional[bool] = Field(default=None)
    bf_enabled: Optional[bool] = Field(default=None)
    abf_enabled: Optional[bool] = Field(default=None)


class SetMagnificationInput(_StrictModel):
    magnification: float = Field(ge=10.0, le=2_000_000.0)


class SetImageShiftInput(_StrictModel):
    x: float
    y: float


class SetBrightnessInput(_StrictModel):
    value: float = Field(ge=0.0, le=1.0)


class ChangeFocusRelativeInput(_StrictModel):
    delta_um: float = Field(ge=-100.0, le=100.0)


class SetCondenserStigmationInput(_StrictModel):
    x: float = Field(ge=-1.0, le=1.0)
    y: float = Field(ge=-1.0, le=1.0)


# ---------------------------------------------------------------------------
# Tilt series
# ---------------------------------------------------------------------------


class TiltSeriesInput(_StrictModel):
    start_deg: float = Field(default=-60.0, ge=-80.0, le=0.0)
    end_deg: float = Field(default=60.0, ge=0.0, le=80.0)
    step_deg: float = Field(default=2.0, ge=0.5, le=10.0)
    exposure_s: float = Field(default=1.0, ge=0.001, le=60.0)
    binning: int = Field(default=2, ge=1, le=8)
    save_dir: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# Derived analyses
# ---------------------------------------------------------------------------


class ImageFilterInput(_StrictModel):
    roi: Optional[list[int]] = Field(default=None)
    median_size: int = Field(default=0, ge=0, le=21)
    gaussian_sigma: float = Field(default=0.0, ge=0.0, le=20.0)
    output_name: str = Field(default="Filtered_Image")
    show_result: bool = Field(default=True)

    @field_validator("roi")
    @classmethod
    def _v_roi(cls, v):
        return _validate_roi(v)


class RadialProfileInput(_StrictModel):
    mode: str = Field(default="fft")
    roi: Optional[list[int]] = Field(default=None)
    binning: int = Field(default=1, ge=1, le=16)
    mask_center_lines: bool = Field(default=True)
    mask_percent: float = Field(default=5.0, ge=0.0, le=50.0)
    profile_metric: str = Field(default="radial_max_minus_mean")
    smooth_sigma: float = Field(default=1.0, ge=0.0, le=10.0)

    @field_validator("roi")
    @classmethod
    def _v_roi(cls, v):
        return _validate_roi(v)


class MaxFFTInput(_StrictModel):
    roi: Optional[list[int]] = Field(default=None)
    fft_size: int = Field(default=256, ge=32, le=1024)
    spacing: int = Field(default=256, ge=1, le=1024)
    log_scale: bool = Field(default=True)
    output_name: str = Field(default="FFT_Max")
    show_result: bool = Field(default=True)

    @field_validator("roi")
    @classmethod
    def _v_roi(cls, v):
        return _validate_roi(v)


class MaxSpotMapInput(_StrictModel):
    mask_center_radius_px: float = Field(default=5.0, ge=0.0, le=512.0)
    map_var: str = Field(default="theta")
    subtract_mean_background: bool = Field(default=False)
    gaussian_sigma: float = Field(default=0.0, ge=0.0, le=10.0)
    output_name: str = Field(default="4DSTEM_Maximum_Spot_Map")
    show_result: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Live-processing jobs
# ---------------------------------------------------------------------------

LIVE_JOB_TYPES = (
    "radial_profile",
    "difference",
    "fft_map",
    "filtered_view",
    "maximum_spot_mapping",
)


class StartLiveProcessingJobInput(_StrictModel):
    job_type: str = Field(description="One of: " + " | ".join(LIVE_JOB_TYPES))
    poll_interval_s: float = Field(default=0.5, ge=0.05, le=60.0)
    roi: Optional[list[int]] = Field(default=None)
    show_result: bool = Field(default=False)
    output_name: Optional[str] = Field(default=None)
    history_length: int = Field(default=200, ge=8, le=2000)
    # job-type-specific knobs are accepted but enforced inside the adapter
    profile_mode: str = Field(default="fft")
    binning: int = Field(default=1, ge=1, le=16)
    fft_size: int = Field(default=256, ge=32, le=1024)
    spacing: int = Field(default=256, ge=1, le=1024)
    avg_period_1: int = Field(default=5, ge=1, le=1000)
    avg_period_2: int = Field(default=10, ge=1, le=1000)
    median_size: int = Field(default=0, ge=0, le=21)
    gaussian_sigma: float = Field(default=0.0, ge=0.0, le=20.0)
    mask_center_radius_px: float = Field(default=5.0, ge=0.0, le=512.0)
    map_var: str = Field(default="theta")

    @field_validator("job_type")
    @classmethod
    def _v_job(cls, v):
        if v not in LIVE_JOB_TYPES:
            raise ValueError(f"job_type must be one of {LIVE_JOB_TYPES}, got {v!r}")
        return v

    @field_validator("roi")
    @classmethod
    def _v_roi(cls, v):
        return _validate_roi(v)


class LiveProcessingJobQuery(_StrictModel):
    job_id: str
    include_data: bool = Field(default=False)
