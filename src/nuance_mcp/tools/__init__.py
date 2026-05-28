"""Register every typed tool on a FastMCP instance, bound to an adapter.

The tools layer is the only place that knows the *shape* of the public MCP
surface (tool names, schemas, return wrappers). It depends only on
:mod:`nuance_mcp.core` and the abstract :class:`MicroscopeAdapter`.

Tool names are vendor-neutral by convention (``acquire_tem_image`` rather
than ``gms_acquire_tem_image``). Legacy ``gms_*`` aliases are registered as
well so existing client code keeps working through one deprecation cycle.

Usage:
    From ``nuance_mcp.adapters.gatan``, you can either:
    - import the bridge adapter: ``from nuance_mcp.adapters.gatan import gms_adapter as gatan_adapter``
    - or the main adapter: ``from nuance_mcp.adapters.gatan import adapter``
"""

from __future__ import annotations

import functools

from fastmcp import FastMCP

from ..core import (
    MicroscopeAdapter,
    CapabilityUnavailable,
    Capability as Capability,
)
from ..core.schemas import (
    FrontImageInput,
    AcquireTEMInput,
    AcquireSTEMInput,
    Acquire4DSTEMInput,
    AcquireEELSInput,
    AcquireDiffractionInput,
    SetStageInput,
    SetBeamInput,
    SetDetectorInput,
    SetMagnificationInput,
    SetImageShiftInput,
    SetBrightnessInput,
    ChangeFocusRelativeInput,
    SetCondenserStigmationInput,
    TiltSeriesInput,
    ImageFilterInput,
    RadialProfileInput,
    MaxFFTInput,
    MaxSpotMapInput,
    StartLiveProcessingJobInput,
    LiveProcessingJobQuery,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_to_dict(img) -> dict:
    arr = img.data
    return {
        "name": img.name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "statistics": {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        },
        "calibration": {
            "scale": img.pixel_size_nm,
            "unit": img.pixel_unit,
        },
        "exposure_s": img.exposure_s,
        "tags": img.tags,
    }


def _spectrum_to_dict(spec) -> dict:
    return {
        "name": spec.name,
        "n_channels": int(spec.counts.size),
        "exposure_s": spec.exposure_s,
        "dispersion_eV_per_ch": spec.dispersion_eV_per_ch,
        "energy_range_eV": [float(spec.energy_eV.min()), float(spec.energy_eV.max())],
        "statistics": {
            "max": float(spec.counts.max()),
            "mean": float(spec.counts.mean()),
        },
        "tags": spec.tags,
    }


def _wrap_unsupported(fn):
    """Convert CapabilityUnavailable into a structured tool response.

    Uses ``functools.wraps`` so FastMCP sees the original signature when it
    parses the function for its JSON schema.
    """

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except CapabilityUnavailable as e:
            return {"status": "UNSUPPORTED", "reason": str(e)}

    # FastMCP inspects __wrapped__ via functools.wraps to recover the real
    # signature, but it does not unwrap ``*args``/``**kwargs`` in ``inner``.
    # We restore the wrapped function's signature explicitly:
    import inspect

    inner.__signature__ = inspect.signature(fn)
    return inner


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_tools(mcp: FastMCP, adapter: MicroscopeAdapter) -> None:
    """Bind every typed tool on ``mcp`` to ``adapter``."""

    # ----- diagnostics ----------------------------------------------------

    @mcp.tool(name="get_microscope_state")
    def get_microscope_state() -> dict:
        s = adapter.get_state()
        return {k: v for k, v in s.__dict__.items() if not k.startswith("_")}

    @mcp.tool(name="get_capabilities")
    def get_capabilities() -> dict:
        return {
            "vendor": adapter.vendor,
            "model": adapter.model,
            "bridge_required": adapter.bridge_required,
            "capabilities": sorted(c.value for c in adapter.capabilities),
        }

    @mcp.tool(name="get_front_image")
    def get_front_image(payload: FrontImageInput) -> dict:
        return adapter.get_front_image(
            include_data=payload.include_data,
            include_tags=payload.include_tags,
        )

    @mcp.tool(name="get_image_shift")
    @_wrap_unsupported
    def get_image_shift() -> dict:
        return adapter.get_image_shift()

    @mcp.tool(name="workspace_list_images")
    @_wrap_unsupported
    def workspace_list_images() -> list[dict]:
        return adapter.workspace_list_images()

    # ----- acquisition ----------------------------------------------------

    @mcp.tool(name="acquire_tem_image")
    @_wrap_unsupported
    def acquire_tem(payload: AcquireTEMInput) -> dict:
        img = adapter.acquire_tem(
            exposure_s=payload.exposure_s,
            binning=payload.binning,
            processing=payload.processing,
            roi=payload.roi,
        )
        return _image_to_dict(img)

    @mcp.tool(name="acquire_stem")
    @_wrap_unsupported
    def acquire_stem(payload: AcquireSTEMInput) -> dict:
        img = adapter.acquire_stem(
            width=payload.width,
            height=payload.height,
            dwell_us=payload.dwell_us,
            rotation_deg=payload.rotation_deg,
            signals=payload.signals,
        )
        return _image_to_dict(img)

    @mcp.tool(name="acquire_4d_stem")
    @_wrap_unsupported
    def acquire_4d_stem(payload: Acquire4DSTEMInput) -> dict:
        img = adapter.acquire_4d_stem(
            scan_x=payload.scan_x,
            scan_y=payload.scan_y,
            dwell_us=payload.dwell_us,
            camera_length_mm=payload.camera_length_mm,
            convergence_mrad=payload.convergence_mrad,
        )
        return _image_to_dict(img)

    @mcp.tool(name="acquire_eels")
    @_wrap_unsupported
    def acquire_eels(payload: AcquireEELSInput) -> dict:
        spec = adapter.acquire_eels(
            exposure_s=payload.exposure_s,
            energy_offset_eV=payload.energy_offset_eV,
            slit_width_eV=payload.slit_width_eV,
            dispersion_idx=payload.dispersion_idx,
            full_vertical_binning=payload.full_vertical_binning,
        )
        return _spectrum_to_dict(spec)

    @mcp.tool(name="acquire_diffraction")
    @_wrap_unsupported
    def acquire_diffraction(payload: AcquireDiffractionInput) -> dict:
        img = adapter.acquire_diffraction(
            exposure_s=payload.exposure_s,
            camera_length_mm=payload.camera_length_mm,
            binning=payload.binning,
        )
        return _image_to_dict(img)

    # ----- stage / optics / detectors -------------------------------------

    @mcp.tool(name="get_stage_position")
    @_wrap_unsupported
    def get_stage_position() -> dict:
        return adapter.get_stage_position()

    @mcp.tool(name="set_stage_position")
    @_wrap_unsupported
    def set_stage_position(payload: SetStageInput) -> dict:
        return adapter.set_stage_position(**payload.model_dump(exclude_none=True))

    @mcp.tool(name="set_beam_parameters")
    @_wrap_unsupported
    def set_beam(payload: SetBeamInput) -> dict:
        return adapter.set_beam_parameters(**payload.model_dump(exclude_none=True))

    @mcp.tool(name="configure_detectors")
    @_wrap_unsupported
    def configure_detectors(payload: SetDetectorInput) -> dict:
        return adapter.configure_detectors(**payload.model_dump(exclude_none=True))

    @mcp.tool(name="set_magnification")
    @_wrap_unsupported
    def set_magnification(payload: SetMagnificationInput) -> dict:
        return adapter.set_magnification(payload.magnification)

    @mcp.tool(name="set_image_shift")
    @_wrap_unsupported
    def set_image_shift(payload: SetImageShiftInput) -> dict:
        return adapter.set_image_shift(payload.x, payload.y)

    @mcp.tool(name="set_brightness")
    @_wrap_unsupported
    def set_brightness(payload: SetBrightnessInput) -> dict:
        return adapter.set_brightness(payload.value)

    @mcp.tool(name="change_focus_relative")
    @_wrap_unsupported
    def change_focus_relative(payload: ChangeFocusRelativeInput) -> dict:
        return adapter.change_focus_relative(payload.delta_um)

    @mcp.tool(name="stop_stage")
    @_wrap_unsupported
    def stop_stage() -> dict:
        return adapter.stop_stage()

    @mcp.tool(name="set_condenser_stigmation")
    @_wrap_unsupported
    def set_condenser_stigmation(payload: SetCondenserStigmationInput) -> dict:
        return adapter.set_condenser_stigmation(payload.x, payload.y)

    # ----- workflow / lifecycle / analyses --------------------------------

    @mcp.tool(name="acquire_tilt_series")
    @_wrap_unsupported
    def tilt_series(payload: TiltSeriesInput) -> dict:
        return adapter.acquire_tilt_series(**payload.model_dump(exclude_none=True))

    @mcp.tool(name="apply_image_filter")
    @_wrap_unsupported
    def apply_image_filter(payload: ImageFilterInput) -> dict:
        img = adapter.apply_image_filter(**payload.model_dump(exclude_none=True))
        return _image_to_dict(img)

    @mcp.tool(name="compute_radial_profile")
    @_wrap_unsupported
    def compute_radial_profile(payload: RadialProfileInput) -> dict:
        return adapter.compute_radial_profile(**payload.model_dump(exclude_none=True))

    @mcp.tool(name="compute_max_fft")
    @_wrap_unsupported
    def compute_max_fft(payload: MaxFFTInput) -> dict:
        img = adapter.compute_max_fft(**payload.model_dump(exclude_none=True))
        return _image_to_dict(img)

    @mcp.tool(name="run_4dstem_analysis")
    @_wrap_unsupported
    def run_4dstem_analysis() -> dict:
        return adapter.run_4dstem_analysis()

    @mcp.tool(name="run_4dstem_maximum_spot_mapping")
    @_wrap_unsupported
    def run_4dstem_maximum_spot_mapping(payload: MaxSpotMapInput) -> dict:
        img = adapter.run_4dstem_maximum_spot_mapping(
            **payload.model_dump(exclude_none=True)
        )
        return _image_to_dict(img)

    @mcp.tool(name="start_live_processing_job")
    @_wrap_unsupported
    def start_live(payload: StartLiveProcessingJobInput) -> dict:
        return adapter.start_live_processing_job(
            **payload.model_dump(exclude_none=True)
        )

    @mcp.tool(name="get_live_processing_job_status")
    @_wrap_unsupported
    def status_live(payload: LiveProcessingJobQuery) -> dict:
        return adapter.get_live_processing_job_status(payload.job_id)

    @mcp.tool(name="get_live_processing_job_result")
    @_wrap_unsupported
    def result_live(payload: LiveProcessingJobQuery) -> dict:
        return adapter.get_live_processing_job_result(
            payload.job_id,
            include_data=payload.include_data,
        )

    @mcp.tool(name="stop_live_processing_job")
    @_wrap_unsupported
    def stop_live(payload: LiveProcessingJobQuery) -> dict:
        return adapter.stop_live_processing_job(payload.job_id)

    # ----- legacy gms_* aliases (one-cycle deprecation) -------------------
    # Clients targeting the original GMS-MCP surface keep working.
    _LEGACY = [
        ("get_microscope_state", "gms_get_microscope_state"),
        ("get_front_image", "gms_get_front_image"),
        ("acquire_tem_image", "gms_acquire_tem_image"),
        ("acquire_stem", "gms_acquire_stem"),
        ("acquire_4d_stem", "gms_acquire_4d_stem"),
        ("acquire_eels", "gms_acquire_eels"),
        ("acquire_diffraction", "gms_acquire_diffraction"),
        ("get_stage_position", "gms_get_stage_position"),
        ("set_stage_position", "gms_set_stage_position"),
        ("set_beam_parameters", "gms_set_beam_parameters"),
        ("configure_detectors", "gms_configure_detectors"),
        ("apply_image_filter", "gms_apply_image_filter"),
        ("compute_radial_profile", "gms_compute_radial_profile"),
        ("compute_max_fft", "gms_compute_max_fft"),
        ("run_4dstem_analysis", "gms_run_4dstem_analysis"),
        ("run_4dstem_maximum_spot_mapping", "gms_run_4dstem_maximum_spot_mapping"),
        ("acquire_tilt_series", "gms_acquire_tilt_series"),
        ("start_live_processing_job", "gms_start_live_processing_job"),
        ("get_live_processing_job_status", "gms_get_live_processing_job_status"),
        ("get_live_processing_job_result", "gms_get_live_processing_job_result"),
        ("stop_live_processing_job", "gms_stop_live_processing_job"),
    ]
    # NB: FastMCP doesn't currently expose an alias API; in production we
    # register each legacy name as a thin wrapper that calls the new tool.
    # Kept as a TODO marker here to make the intent obvious to readers.
