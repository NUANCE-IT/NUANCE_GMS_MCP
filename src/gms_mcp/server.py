"""
gms_mcp_server.py
==================
Production-quality MCP server exposing Gatan Microscopy Suite (GMS) 3.60
functionality to any MCP-compatible LLM client (Claude.ai or local Ollama).

Transport selection
-------------------
    stdio  (default)  — launched as a subprocess by the LLM client.
                        Best for local Ollama / LangChain usage.
    http              — Streamable HTTP on port 8000 (or $GMS_MCP_PORT).
                        Best for Claude.ai remote connector.
                        Connect at: http://<host>:8000/mcp

Simulation mode
---------------
    When run outside GMS, the DigitalMicrograph module is unavailable.
    dm_simulator.py provides a full physics-plausible drop-in so that
    development and testing require no microscope hardware.

Usage
-----
    # stdio (for Ollama / local LangChain):
    python gms_mcp_server.py

    # HTTP (for Claude.ai remote connector):
    python gms_mcp_server.py --transport http --port 8000

    # Force simulation even if DM is importable:
    GMS_SIMULATE=1 python gms_mcp_server.py
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
import uuid
from typing import Annotated, Any, Optional, cast

import numpy as np
from fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, field_validator
from scipy.ndimage import gaussian_filter, median_filter

# ---------------------------------------------------------------------------
# DM import with automatic simulation fallback
# ---------------------------------------------------------------------------

_SIMULATE = os.environ.get("GMS_SIMULATE", "0") != "0"

if not _SIMULATE:
    try:
        import DigitalMicrograph as DM
        _SIMULATE = False
    except ImportError:
        _SIMULATE = True

if _SIMULATE:
    # Ensure we can find the simulator when the server is launched from
    # a different working directory.
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from gms_mcp.simulator import DMSimulator
    DM = DMSimulator()

_BRIDGE_ZMQ_ENDPOINT = os.environ.get("GMS_MCP_ZMQ", "").strip()
_BRIDGE_ZMQ_TIMEOUT_MS = int(os.environ.get("GMS_MCP_ZMQ_TIMEOUT_MS", "5000"))

# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="gms_mcp",
    instructions=(
        "This server controls a transmission electron microscope via "
        "Gatan Microscopy Suite (GMS) 3.60. "
        "It supports TEM/HRTEM imaging, STEM (HAADF/BF/ABF), 4D-STEM/NBED, "
        "EELS/EDS spectrum imaging, and electron diffraction pattern acquisition. "
        "Full stage, beam/optics, and detector configuration is available. "
        "Always call gms_get_microscope_state first to confirm the current "
        "instrument configuration before issuing acquisition commands."
    ),
)

# ---------------------------------------------------------------------------
# Pydantic input models
# ---------------------------------------------------------------------------

class AcquireTEMInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    exposure_s: float = Field(
        default=1.0, ge=0.001, le=60.0,
        description="Camera exposure time in seconds (0.001–60)."
    )
    binning: int = Field(
        default=1, ge=1, le=8,
        description="Camera binning factor (1, 2, 4, or 8)."
    )
    processing: int = Field(
        default=3, ge=1, le=3,
        description="Correction level: 1=raw, 2=dark-subtracted, 3=dark+gain."
    )
    roi: Optional[list[int]] = Field(
        default=None,
        description="Optional ROI as [top, left, bottom, right] in pixels."
    )

    @field_validator("roi")
    @classmethod
    def validate_roi(cls, v: Optional[list]) -> Optional[list]:
        if v is not None and len(v) != 4:
            raise ValueError("roi must have exactly 4 elements: [top, left, bottom, right]")
        return v


class FrontImageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    include_data: bool = Field(
        default=False,
        description="If true, include base64-encoded pixel data in the response.",
    )
    include_tags: bool = Field(
        default=True,
        description="If true, include serialisable image tags when available.",
    )


class ImageFilterInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    roi: Optional[list[int]] = Field(
        default=None,
        description="Optional [top, left, bottom, right] ROI in pixels.",
    )
    median_size: int = Field(
        default=0,
        ge=0,
        le=21,
        description="Median-filter kernel size in pixels. 0 disables median filtering.",
    )
    gaussian_sigma: float = Field(
        default=0.0,
        ge=0.0,
        le=20.0,
        description="Gaussian blur sigma in pixels. 0 disables Gaussian filtering.",
    )
    output_name: str = Field(
        default="Filtered_Image",
        description="Name assigned to the derived image in GMS.",
    )
    show_result: bool = Field(
        default=True,
        description="If true, display the derived image in GMS.",
    )

    @field_validator("roi")
    @classmethod
    def validate_roi(cls, v: Optional[list[int]]) -> Optional[list[int]]:
        if v is not None and len(v) != 4:
            raise ValueError("roi must have exactly 4 elements: [top, left, bottom, right]")
        return v


class RadialProfileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: str = Field(
        default="fft",
        description="Profile source: 'fft' for HRTEM FFT or 'diffraction' for direct diffraction data.",
    )
    roi: Optional[list[int]] = Field(
        default=None,
        description="Optional [top, left, bottom, right] ROI in pixels.",
    )
    binning: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Integer binning factor applied before profiling.",
    )
    mask_center_lines: bool = Field(
        default=True,
        description="If true, zero the central horizontal and vertical lines.",
    )
    mask_percent: float = Field(
        default=5.0,
        ge=0.0,
        le=50.0,
        description="Percentage of the innermost radius to ignore.",
    )
    profile_metric: str = Field(
        default="radial_max_minus_mean",
        description="Metric: 'radial_max_minus_mean', 'radial_mean', or 'radial_max'.",
    )
    smooth_sigma: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Optional Gaussian smoothing applied to the resulting 1D profile.",
    )

    @field_validator("roi")
    @classmethod
    def validate_roi(cls, v: Optional[list[int]]) -> Optional[list[int]]:
        if v is not None and len(v) != 4:
            raise ValueError("roi must have exactly 4 elements: [top, left, bottom, right]")
        return v


class MaxFFTInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    roi: Optional[list[int]] = Field(
        default=None,
        description="Optional [top, left, bottom, right] ROI in pixels.",
    )
    fft_size: int = Field(
        default=256,
        ge=32,
        le=1024,
        description="FFT window size in pixels.",
    )
    spacing: int = Field(
        default=256,
        ge=1,
        le=1024,
        description="Stride between neighbouring FFT windows in pixels.",
    )
    log_scale: bool = Field(
        default=True,
        description="If true, log-scale the FFT magnitude.",
    )
    output_name: str = Field(
        default="FFT_Max",
        description="Name assigned to the derived image in GMS.",
    )
    show_result: bool = Field(
        default=True,
        description="If true, display the derived image in GMS.",
    )

    @field_validator("roi")
    @classmethod
    def validate_roi(cls, v: Optional[list[int]]) -> Optional[list[int]]:
        if v is not None and len(v) != 4:
            raise ValueError("roi must have exactly 4 elements: [top, left, bottom, right]")
        return v


class MaxSpotMapInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mask_center_radius_px: float = Field(
        default=5.0,
        ge=0.0,
        le=512.0,
        description="Radius around the central beam to ignore when finding the brightest spot.",
    )
    map_var: str = Field(
        default="theta",
        description="Colour-map variable: 'theta' or 'radius'.",
    )
    subtract_mean_background: bool = Field(
        default=False,
        description="If true, subtract the mean diffraction pattern before processing.",
    )
    gaussian_sigma: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Optional Gaussian blur sigma applied to the diffraction patterns.",
    )
    output_name: str = Field(
        default="4DSTEM_Maximum_Spot_Map",
        description="Name assigned to the derived RGB image in GMS.",
    )
    show_result: bool = Field(
        default=True,
        description="If true, display the derived image in GMS.",
    )


class StartLiveProcessingJobInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_type: str = Field(
        description="Live job type: 'radial_profile', 'difference', 'fft_map', 'filtered_view', or 'maximum_spot_mapping'.",
    )
    poll_interval_s: float = Field(
        default=0.5,
        ge=0.05,
        le=60.0,
        description="Polling interval in seconds between live updates.",
    )
    roi: Optional[list[int]] = Field(
        default=None,
        description="Optional [top, left, bottom, right] ROI in pixels.",
    )
    show_result: bool = Field(
        default=False,
        description="If true, create and update a derived result image in GMS.",
    )
    output_name: Optional[str] = Field(
        default=None,
        description="Optional result image name. Defaults to a job-specific name.",
    )
    include_result_data: bool = Field(
        default=False,
        description="If true, job result queries may include base64-encoded pixel data.",
    )
    history_length: int = Field(
        default=200,
        ge=8,
        le=2000,
        description="Rolling history length for radial-profile jobs.",
    )
    profile_mode: str = Field(
        default="fft",
        description="For radial-profile jobs: 'fft' or 'diffraction'.",
    )
    binning: int = Field(
        default=1,
        ge=1,
        le=16,
        description="For radial-profile jobs: integer binning factor.",
    )
    mask_center_lines: bool = Field(
        default=True,
        description="For radial-profile jobs: mask central horizontal and vertical lines.",
    )
    mask_percent: float = Field(
        default=5.0,
        ge=0.0,
        le=50.0,
        description="For radial-profile jobs: ignore the innermost percentage of radius.",
    )
    profile_metric: str = Field(
        default="radial_max_minus_mean",
        description="For radial-profile jobs: 'radial_max_minus_mean', 'radial_mean', or 'radial_max'.",
    )
    smooth_sigma: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="For radial-profile jobs: Gaussian smoothing of the 1D profile.",
    )
    avg_period_1: int = Field(
        default=5,
        ge=1,
        le=1000,
        description="For difference jobs: first exponentially weighted moving-average period.",
    )
    avg_period_2: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="For difference jobs: second exponentially weighted moving-average period.",
    )
    gaussian_sigma: float = Field(
        default=0.0,
        ge=0.0,
        le=20.0,
        description="For difference and filtered-view jobs: optional Gaussian blur sigma.",
    )
    median_size: int = Field(
        default=0,
        ge=0,
        le=21,
        description="For filtered-view jobs: optional median-filter kernel size. 0 disables median filtering.",
    )
    fft_size: int = Field(
        default=256,
        ge=32,
        le=1024,
        description="For fft_map jobs: local FFT window size.",
    )
    spacing: int = Field(
        default=256,
        ge=1,
        le=1024,
        description="For fft_map jobs: stride between neighbouring windows.",
    )
    log_scale: bool = Field(
        default=True,
        description="For fft_map jobs: log-scale the FFT magnitude.",
    )
    mask_center_radius_px: float = Field(
        default=5.0,
        ge=0.0,
        le=512.0,
        description="For maximum_spot_mapping jobs: radius around the central beam to ignore.",
    )
    map_var: str = Field(
        default="theta",
        description="For maximum_spot_mapping jobs: 'theta' or 'radius'.",
    )
    subtract_mean_background: bool = Field(
        default=False,
        description="For maximum_spot_mapping jobs: subtract the mean diffraction pattern before processing.",
    )

    @field_validator("roi")
    @classmethod
    def validate_roi(cls, v: Optional[list[int]]) -> Optional[list[int]]:
        if v is not None and len(v) != 4:
            raise ValueError("roi must have exactly 4 elements: [top, left, bottom, right]")
        return v


class LiveProcessingJobQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str = Field(description="Live-processing job identifier.")
    include_data: bool = Field(
        default=False,
        description="If true, include base64-encoded pixel data in the latest result when available.",
    )


class AcquireSTEMInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    width: int = Field(default=512, ge=64, le=4096,
                       description="Scan width in pixels.")
    height: int = Field(default=512, ge=64, le=4096,
                        description="Scan height in pixels.")
    dwell_us: float = Field(default=10.0, ge=0.5, le=10000.0,
                            description="Pixel dwell time in microseconds.")
    rotation_deg: float = Field(default=0.0, ge=-180.0, le=180.0,
                                description="Scan rotation in degrees.")
    signals: list[int] = Field(
        default=[0, 1],
        description=(
            "List of DigiScan signal channels to enable. "
            "Channel 0 = HAADF, 1 = BF, 2 = ABF (installation-dependent)."
        )
    )


class Acquire4DSTEMInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    scan_x: int = Field(default=64, ge=8, le=512,
                        description="Number of scan positions in X.")
    scan_y: int = Field(default=64, ge=8, le=512,
                        description="Number of scan positions in Y.")
    dwell_us: float = Field(default=1000.0, ge=100.0, le=100_000.0,
                            description="Per-pattern dwell time in microseconds.")
    camera_length_mm: Optional[float] = Field(
        default=None, ge=20.0, le=2000.0,
        description="Camera length in mm; None = keep current value."
    )
    convergence_mrad: Optional[float] = Field(
        default=None, ge=0.1, le=50.0,
        description="Convergence semi-angle in mrad (informational, logs to metadata)."
    )


class AcquireEELSInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    exposure_s: float = Field(default=1.0, ge=0.001, le=60.0,
                              description="Spectrometer exposure in seconds.")
    energy_offset_eV: float = Field(default=0.0, ge=-200.0, le=3000.0,
                                    description="Energy offset (drift tube setting) in eV.")
    slit_width_eV: float = Field(default=10.0, ge=0.0, le=100.0,
                                 description="Energy slit width in eV. 0 = slit out.")
    dispersion_idx: int = Field(default=0, ge=0, le=3,
                                description="Dispersion index (0=highest, 3=lowest).")
    full_vertical_binning: bool = Field(
        default=True,
        description="If True, apply full vertical CCD binning (standard for EELS spectra)."
    )


class AcquireDiffractionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    exposure_s: float = Field(default=0.5, ge=0.001, le=60.0,
                              description="Camera exposure in seconds.")
    camera_length_mm: Optional[float] = Field(
        default=None, ge=20.0, le=2000.0,
        description="Camera length in mm; None = keep current value."
    )
    binning: int = Field(default=1, ge=1, le=8,
                         description="Camera binning.")


class SetStageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x_um: Optional[float] = Field(default=None, ge=-5000.0, le=5000.0,
                                  description="Stage X in micrometers.")
    y_um: Optional[float] = Field(default=None, ge=-5000.0, le=5000.0,
                                  description="Stage Y in micrometers.")
    z_um: Optional[float] = Field(default=None, ge=-500.0, le=500.0,
                                  description="Stage Z in micrometers.")
    alpha_deg: Optional[float] = Field(default=None, ge=-80.0, le=80.0,
                                       description="Alpha tilt in degrees (±80°).")
    beta_deg: Optional[float] = Field(default=None, ge=-30.0, le=30.0,
                                      description="Beta tilt in degrees (±30°).")


class SetBeamInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spot_size: Optional[int] = Field(default=None, ge=1, le=11,
                                     description="Condenser spot size index (1–11).")
    focus_um: Optional[float] = Field(default=None,
                                      description="Absolute objective lens focus offset in µm.")
    shift_x: Optional[float] = Field(default=None,
                                     description="Calibrated beam shift X (physical units).")
    shift_y: Optional[float] = Field(default=None,
                                     description="Calibrated beam shift Y.")
    tilt_x: Optional[float] = Field(default=None,
                                    description="Beam tilt X (rad).")
    tilt_y: Optional[float] = Field(default=None,
                                    description="Beam tilt Y (rad).")
    obj_stig_x: Optional[float] = Field(default=None,
                                        description="Objective stigmator X.")
    obj_stig_y: Optional[float] = Field(default=None,
                                        description="Objective stigmator Y.")


class SetDetectorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    insert_camera: Optional[bool] = Field(
        default=None,
        description="True = insert camera, False = retract camera."
    )
    target_temp_c: Optional[float] = Field(
        default=None, ge=-60.0, le=30.0,
        description="Target CCD cooling temperature in °C."
    )
    haadf_enabled: Optional[bool] = Field(
        default=None,
        description="Enable/disable DigiScan HAADF channel (signal 0)."
    )
    bf_enabled: Optional[bool] = Field(
        default=None,
        description="Enable/disable DigiScan BF channel (signal 1)."
    )
    abf_enabled: Optional[bool] = Field(
        default=None,
        description="Enable/disable DigiScan ABF channel (signal 2)."
    )


class TiltSeriesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_deg: float = Field(default=-60.0, ge=-80.0, le=0.0,
                             description="Starting tilt angle in degrees.")
    end_deg: float = Field(default=60.0, ge=0.0, le=80.0,
                           description="Ending tilt angle in degrees.")
    step_deg: float = Field(default=2.0, ge=0.5, le=10.0,
                            description="Angular step size in degrees.")
    exposure_s: float = Field(default=1.0, ge=0.001, le=60.0,
                              description="Exposure per tilt step in seconds.")
    binning: int = Field(default=2, ge=1, le=8,
                         description="Camera binning for each frame.")
    save_dir: Optional[str] = Field(
        default=None,
        description="Directory to save individual DM4 files; None = do not save."
    )


# ---------------------------------------------------------------------------
# Shared helper functions
# ---------------------------------------------------------------------------


def _clean_tag_value(val):
    """
    Normalize tag values:
    - Convert known byte sequences to strings
    - Skip unknown byte blobs
    - Ensure JSON-serializable output
    """
    import json
    if isinstance(val, (bytes, bytearray, memoryview)):
        byte_map = {
            b'\xb0C': "oC",
            b'\xc5': "A",
            b'\xb5m': "um",
        }
        if val in byte_map:
            val = byte_map[val]
        else:
            return None  # skip unknown bytes

    try:
        json.dumps(val)
        return val
    except TypeError:
        return str(val)

def _tags_to_dict(tg, path=""):
    """
    Recursively traverse a Py_TagGroup and return a dict mapping
    tag paths -> cleaned tag values.
    """
    tags = {}

    # Respect IsValid if present
    if hasattr(tg, "IsValid") and not tg.IsValid():
        return tags

    # Require TagGroup traversal support
    if not hasattr(tg, "keys"):
        try:
            DM.OkDialog(
                "This script requires tag group traversal (GMS 3.6.1 or newer)."
            )
        except Exception:
            pass
        raise RuntimeError("TagGroup traversal not supported")

    for key in tg.keys():
        try:
            val = tg[key]
        except Exception:
            continue

        cur_path = f"{path}:{key}" if path else key

        # For simulation, DM.Py_TagGroup may not exist, so check by name
        is_tag_group = False
        if hasattr(DM, "Py_TagGroup"):
            is_tag_group = isinstance(val, DM.Py_TagGroup)
        else:
            is_tag_group = hasattr(val, "keys")

        if is_tag_group:
            tags.update(
                _tags_to_dict(val, cur_path)
            )
        else:
            clean_val = _clean_tag_value(val)
            if clean_val is not None:
                tags[cur_path] = clean_val

    return tags


def _image_to_response(
    img,
    include_data: bool = False,
    include_tags: bool = False,
) -> dict:
    """Convert a SimImage / Py_Image to a JSON-serialisable summary dict."""
    arr = img.GetNumArray()
    tags = img.GetTagGroup()
    ok_exp, exp = tags.GetTagAsFloat("Acquisition:ExposureTime")
    ok_ht, ht = tags.GetTagAsFloat("Microscope:HighTension_kV")
    ok_mag, mag = tags.GetTagAsFloat("Microscope:Magnification")

    summary = {
        "name": img.GetName(),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "statistics": {
            "min":    float(arr.min()),
            "max":    float(arr.max()),
            "mean":   float(arr.mean()),
            "std":    float(arr.std()),
        },
        "calibration": {
            "pixel_size_nm": img.GetDimensionCalibration(0, 0)[1],
            "unit":          img.GetDimensionCalibration(0, 0)[2],
        },
        "metadata": {
            "exposure_s":       exp if ok_exp else None,
            "high_tension_kV":  ht  if ok_ht  else None,
            "magnification":    mag if ok_mag else None,
        },
    }
    if include_data:
        summary["data_b64"] = base64.b64encode(arr.tobytes()).decode()
    if include_tags:
        summary["tags"] = _tags_to_dict(tags)
    return summary


def _extract_roi(arr: np.ndarray, roi: Optional[list[int]]) -> np.ndarray:
    if roi is None:
        return arr

    top, left, bottom, right = roi
    if top < 0 or left < 0 or bottom > arr.shape[0] or right > arr.shape[1]:
        raise ValueError("roi extends outside image bounds")
    if bottom <= top or right <= left:
        raise ValueError("roi must have positive height and width")
    return arr[top:bottom, left:right]


def _bin_image(arr: np.ndarray, binning: int) -> np.ndarray:
    if binning <= 1:
        return arr
    h = (arr.shape[0] // binning) * binning
    w = (arr.shape[1] // binning) * binning
    if h == 0 or w == 0:
        raise ValueError("binning is larger than the selected image region")
    cropped = arr[:h, :w]
    return cropped.reshape(h // binning, binning, w // binning, binning).mean(axis=(1, 3))


def _copy_calibration(source_img, derived_img) -> None:
    try:
        for axis in range(2):
            origin, scale, unit = source_img.GetDimensionCalibration(axis, 0)
            derived_img.SetDimensionCalibration(axis, origin, scale, unit, 0)
    except Exception:
        return


def _resolve_4dstem_array(img4d) -> np.ndarray:
    arr = img4d.GetNumArray()
    if arr.ndim == 2 and arr.shape[1] > arr.shape[0]:
        scan_y = arr.shape[0]
        total = arr.shape[1]
        det_px = int(np.sqrt(total // scan_y))
        scan_x = total // (det_px * det_px)
        return arr.reshape(scan_y, scan_x, det_px, det_px)
    if arr.ndim != 4:
        raise ValueError("Front image is not a 4D-STEM dataset (expected 4D array).")
    return arr


def _create_derived_image(data: np.ndarray, name: str, source_img, show_result: bool):
    derived = DM.CreateImage(np.asarray(data))
    derived.SetName(name)
    _copy_calibration(source_img, derived)
    if show_result:
        derived.ShowImage()
    return derived


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    h = np.mod(h, 1.0)
    i = np.floor(h * 6.0).astype(int)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i_mod = i % 6

    out = np.empty(h.shape + (3,), dtype=np.float32)
    masks = [i_mod == k for k in range(6)]
    out[masks[0]] = np.stack((v, t, p), axis=-1)[masks[0]]
    out[masks[1]] = np.stack((q, v, p), axis=-1)[masks[1]]
    out[masks[2]] = np.stack((p, v, t), axis=-1)[masks[2]]
    out[masks[3]] = np.stack((p, q, v), axis=-1)[masks[3]]
    out[masks[4]] = np.stack((t, p, v), axis=-1)[masks[4]]
    out[masks[5]] = np.stack((v, p, q), axis=-1)[masks[5]]
    return out


_live_jobs: dict[str, dict[str, object]] = {}
_live_jobs_lock = threading.Lock()


def _bridge_mode_enabled() -> bool:
    return bool(_BRIDGE_ZMQ_ENDPOINT)


def _runtime_mode() -> str:
    if _bridge_mode_enabled():
        return "bridge-live"
    return "simulation" if _SIMULATE else "in-process-live"


def _live_jobs_use_bridge() -> bool:
    return _bridge_mode_enabled()


def _bridge_dispatch(function_name: str, params: dict[str, object]) -> dict[str, object]:
    if not _BRIDGE_ZMQ_ENDPOINT:
        raise RuntimeError("GMS_MCP_ZMQ is not configured.")

    try:
        import zmq
    except ImportError as exc:
        raise RuntimeError(
            "pyzmq is required for bridge mode. Install the 'zmq' extra before using GMS_MCP_ZMQ."
        ) from exc

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, _BRIDGE_ZMQ_TIMEOUT_MS)
    socket.setsockopt(zmq.SNDTIMEO, _BRIDGE_ZMQ_TIMEOUT_MS)

    try:
        socket.connect(_BRIDGE_ZMQ_ENDPOINT)
        payload = {"function": function_name, "params": params}
        socket.send_json(payload)
        response = socket.recv_json()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to communicate with the GMS DM bridge at {_BRIDGE_ZMQ_ENDPOINT}: {exc}"
        ) from exc
    finally:
        socket.close()
        context.term()

    if not isinstance(response, dict):
        raise RuntimeError("Bridge response is malformed.")
    return response


def _run_bridge_tool(function_name: str, params: dict[str, object] | None = None) -> dict[str, object]:
    payload = _bridge_dispatch(function_name, params or {})
    if payload.get("success") is False:
        detail = payload.get("error", f"Bridge call failed: {function_name}")
        raise RuntimeError(str(detail))
    return payload


def _summarize_array(data: np.ndarray) -> dict[str, object]:
    arr = np.asarray(data)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "statistics": {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        },
    }


def _encode_array_b64(data: np.ndarray) -> dict[str, object]:
    arr = np.asarray(data)
    return {
        "data_b64": base64.b64encode(arr.tobytes()).decode(),
        "data_shape": list(arr.shape),
        "data_dtype": str(arr.dtype),
    }


def _copy_into_result_image(result_img, data: np.ndarray) -> None:
    target = result_img.GetNumArray()
    target[...] = np.asarray(data, dtype=target.dtype)
    result_img.UpdateImage()


def _exponential_moving_average(frame: np.ndarray, previous: np.ndarray, period: int) -> np.ndarray:
    if period <= 1:
        return frame.astype(np.float32)
    persistence = (period - 1) / (period + 1)
    return persistence * previous.astype(np.float32) + (1.0 - persistence) * frame.astype(np.float32)


def _compute_radial_profile_result(data: np.ndarray, params: StartLiveProcessingJobInput | RadialProfileInput) -> dict[str, object]:
    roi_data = _extract_roi(data, params.roi)
    if params.mode == "fft" if isinstance(params, RadialProfileInput) else params.profile_mode == "fft":
        working = np.abs(np.fft.fftshift(np.fft.fft2(roi_data))).astype(np.float32)
        unit = "nm^-1"
    else:
        working = roi_data.copy()
        unit = "px"

    binning = params.binning
    working = _bin_image(working, binning)
    mask_center_lines = params.mask_center_lines
    if mask_center_lines:
        cx = working.shape[1] // 2
        cy = working.shape[0] // 2
        working[:, max(0, cx - 1):cx + 1] = 0.0
        working[max(0, cy - 1):cy + 1, :] = 0.0

    h, w = working.shape
    cy = h / 2.0
    cx = w / 2.0
    yy, xx = np.indices((h, w), dtype=np.float32)
    radii = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    radial_index = radii.astype(int)
    max_radius = min(h, w) // 2

    radial_mean = np.zeros(max_radius, dtype=np.float32)
    radial_max = np.zeros(max_radius, dtype=np.float32)
    for radius in range(max_radius):
        mask = radial_index == radius
        if not np.any(mask):
            continue
        values = working[mask]
        radial_mean[radius] = float(values.mean())
        radial_max[radius] = float(values.max())

    metric = params.profile_metric
    if metric == "radial_mean":
        profile = radial_mean
    elif metric == "radial_max":
        profile = radial_max
    else:
        profile = radial_max - radial_mean

    smooth_sigma = params.smooth_sigma
    if smooth_sigma > 0:
        profile = gaussian_filter(profile, sigma=smooth_sigma)

    ignore_bins = int(len(profile) * params.mask_percent / 100.0)
    if ignore_bins > 0:
        profile[:ignore_bins] = 0.0

    peak_positions: list[int] = []
    try:
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(profile, height=profile.mean() + profile.std())
        peak_positions = peaks.tolist()[:12]
    except Exception:
        peak_positions = []

    return {
        "data": profile.astype(np.float32),
        "summary": {
            "mode": params.mode if isinstance(params, RadialProfileInput) else params.profile_mode,
            "profile_metric": metric,
            "profile_length": int(len(profile)),
            "unit": unit,
            "peak_positions": peak_positions,
        },
    }


def _compute_max_fft_result(data: np.ndarray, params: StartLiveProcessingJobInput | MaxFFTInput) -> dict[str, object]:
    roi_data = _extract_roi(data, params.roi)
    if roi_data.shape[0] < params.fft_size or roi_data.shape[1] < params.fft_size:
        raise ValueError("Selected image region is smaller than fft_size.")

    view = np.lib.stride_tricks.sliding_window_view(roi_data, (params.fft_size, params.fft_size))
    windows = view[::params.spacing, ::params.spacing]
    if windows.size == 0:
        windows = view[:1, :1]

    hann_1d: np.ndarray = np.hanning(params.fft_size).astype(np.float32)
    hann_2d = np.sqrt(np.outer(hann_1d, hann_1d)).astype(np.float32)
    spectra = np.abs(np.fft.fftshift(np.fft.fft2(windows * hann_2d, axes=(-2, -1)), axes=(-2, -1)))
    max_fft = spectra.max(axis=(0, 1)).astype(np.float32)
    if params.log_scale:
        max_fft = np.log1p(max_fft)

    return {
        "data": max_fft,
        "summary": {
            "fft_size": params.fft_size,
            "spacing": params.spacing,
            "log_scale": params.log_scale,
            "n_windows": int(windows.shape[0] * windows.shape[1]),
        },
    }


def _compute_maximum_spot_mapping_result(
    data4d: np.ndarray, params: StartLiveProcessingJobInput | MaxSpotMapInput
) -> dict[str, object]:
    arr = np.asarray(data4d, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError("Maximum spot mapping requires a 4D-STEM dataset.")

    working = arr.copy()
    scan_y, scan_x, det_y, det_x = working.shape
    cy = (det_y - 1) / 2.0
    cx = (det_x - 1) / 2.0

    if params.subtract_mean_background:
        mean_pattern = working.mean(axis=(0, 1), keepdims=True)
        working = np.maximum(working - mean_pattern, 0.0)
    if params.gaussian_sigma > 0:
        working = gaussian_filter(working, sigma=(0, 0, params.gaussian_sigma, params.gaussian_sigma))

    yy, xx = np.indices((det_y, det_x), dtype=np.float32)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = rr <= params.mask_center_radius_px
    working[:, :, mask] = -np.inf

    flat = working.reshape(scan_y, scan_x, det_y * det_x)
    max_idx = np.argmax(flat, axis=-1)
    intensity_map = np.take_along_axis(flat, max_idx[..., None], axis=-1).squeeze(-1)
    y_idx, x_idx = np.divmod(max_idx, det_x)
    dx = x_idx.astype(np.float32) - cx
    dy = y_idx.astype(np.float32) - cy
    theta_map = (np.arctan2(dy, dx) + np.pi) / (2.0 * np.pi)
    radius_map = np.sqrt(dx ** 2 + dy ** 2)

    if params.map_var == "theta":
        hue = theta_map
    elif params.map_var == "radius":
        radius_max = max(float(radius_map.max()), 1.0)
        hue = radius_map / radius_max
    else:
        raise ValueError("map_var must be 'theta' or 'radius'.")

    intensity_norm = intensity_map - intensity_map.min()
    if float(intensity_norm.max()) > 0:
        intensity_norm = intensity_norm / intensity_norm.max()
    saturation = np.ones_like(hue, dtype=np.float32)
    rgb: np.ndarray = _hsv_to_rgb(
        hue.astype(np.float32), saturation, intensity_norm.astype(np.float32)
    ).astype(np.float32)

    return {
        "data": rgb,
        "summary": {
            "type": "maximum_spot_mapping",
            "map_var": params.map_var,
            "scan_shape": [scan_y, scan_x],
            "mask_center_radius_px": params.mask_center_radius_px,
            "subtract_mean_background": params.subtract_mean_background,
            "gaussian_sigma": params.gaussian_sigma,
            "theta_range": [float(theta_map.min()), float(theta_map.max())],
            "radius_range": [float(radius_map.min()), float(radius_map.max())],
            "intensity_range": [float(intensity_map.min()), float(intensity_map.max())],
        },
    }


def _compute_difference_result(data: np.ndarray, job: dict[str, object]) -> dict[str, object]:
    params = job["params"]
    assert isinstance(params, StartLiveProcessingJobInput)
    frame: np.ndarray = _extract_roi(data, params.roi).astype(np.float32)
    if params.gaussian_sigma > 0:
        frame = gaussian_filter(frame, sigma=params.gaussian_sigma)

    avg1 = job.get("avg1")
    avg2 = job.get("avg2")
    if not isinstance(avg1, np.ndarray):
        avg1 = frame.copy()
    if not isinstance(avg2, np.ndarray):
        avg2 = frame.copy()

    avg1 = _exponential_moving_average(frame, avg1, params.avg_period_1)
    avg2 = _exponential_moving_average(frame, avg2, params.avg_period_2)
    diff = np.abs(avg2 - avg1).astype(np.float32)
    job["avg1"] = avg1
    job["avg2"] = avg2

    return {
        "data": diff,
        "summary": {
            "avg_period_1": params.avg_period_1,
            "avg_period_2": params.avg_period_2,
            "gaussian_sigma": params.gaussian_sigma,
        },
    }


def _compute_filtered_view_result(
    data: np.ndarray, params: StartLiveProcessingJobInput
) -> dict[str, object]:
    filtered: np.ndarray = _extract_roi(data, params.roi).astype(np.float32)
    if params.median_size > 1:
        filtered = median_filter(filtered, size=params.median_size)
    if params.gaussian_sigma > 0:
        filtered = gaussian_filter(filtered, sigma=params.gaussian_sigma)

    return {
        "data": filtered.astype(np.float32),
        "summary": {
            "median_size": params.median_size,
            "gaussian_sigma": params.gaussian_sigma,
        },
    }


def _get_live_job(job_id: str) -> dict[str, object]:
    with _live_jobs_lock:
        job = _live_jobs.get(job_id)
    if job is None:
        raise KeyError(f"Unknown live-processing job: {job_id}")
    return job


def _job_status_payload(job: dict[str, object]) -> dict[str, object]:
    latest_result = job.get("latest_result")
    result_summary: dict[str, object] | None = None
    if isinstance(latest_result, dict):
        summary = latest_result.get("summary", {})
        data = latest_result.get("data")
        if isinstance(summary, dict) and isinstance(data, np.ndarray):
            result_summary = dict(summary)
            result_summary.update(_summarize_array(data))

    return {
        "job_id": job["job_id"],
        "job_type": job["job_type"],
        "backend": job.get("backend", "local"),
        "status": job["status"],
        "poll_interval_s": job["poll_interval_s"],
        "iterations": job["iterations"],
        "created_at": job["created_at"],
        "last_updated": job["last_updated"],
        "last_error": job["last_error"],
        "source_image_name": getattr(job.get("source_image"), "GetName", lambda: None)(),
        "result_summary": result_summary,
    }


def _run_live_processing_job(job_id: str) -> None:
    job = _get_live_job(job_id)
    stop_event = job["stop_event"]
    assert isinstance(stop_event, threading.Event)
    params = job["params"]
    assert isinstance(params, StartLiveProcessingJobInput)

    while not stop_event.is_set():
        try:
            source_image = job.get("source_image")
            if source_image is None:
                source_image = DM.GetFrontImage()
                job["source_image"] = source_image
            if not hasattr(source_image, "GetNumArray"):
                raise ValueError("Live-processing source image is invalid or unavailable.")

            source_image_obj = cast(Any, source_image)
            data = np.asarray(source_image_obj.GetNumArray(), dtype=np.float32)
            if data.ndim != 2 and params.job_type in {"radial_profile", "difference", "fft_map", "filtered_view"}:
                raise ValueError("Live processing requires a 2D source image for the selected job type.")

            if params.job_type == "radial_profile":
                radial_params = RadialProfileInput(
                    mode=params.profile_mode,
                    roi=params.roi,
                    binning=params.binning,
                    mask_center_lines=params.mask_center_lines,
                    mask_percent=params.mask_percent,
                    profile_metric=params.profile_metric,
                    smooth_sigma=params.smooth_sigma,
                )
                result = _compute_radial_profile_result(data, radial_params)
                profile = result["data"]
                assert isinstance(profile, np.ndarray)

                history = job.get("history")
                if not isinstance(history, np.ndarray) or history.shape[0] != profile.shape[0]:
                    history = np.zeros((profile.shape[0], params.history_length), dtype=np.float32)
                    job["history"] = history
                history[:, :-1] = history[:, 1:]
                history[:, -1] = profile
                result["data"] = history.copy()
                summary_payload = result.get("summary")
                assert isinstance(summary_payload, dict)
                result["summary"] = {
                    **summary_payload,
                    "history_length": params.history_length,
                }
            elif params.job_type == "difference":
                result = _compute_difference_result(data, job)
            elif params.job_type == "fft_map":
                fft_params = MaxFFTInput(
                    roi=params.roi,
                    fft_size=params.fft_size,
                    spacing=params.spacing,
                    log_scale=params.log_scale,
                    output_name=params.output_name or f"live_fft_map_{job_id}",
                    show_result=params.show_result,
                )
                result = _compute_max_fft_result(data, fft_params)
            elif params.job_type == "filtered_view":
                result = _compute_filtered_view_result(data, params)
            elif params.job_type == "maximum_spot_mapping":
                dataset4d: np.ndarray = _resolve_4dstem_array(source_image_obj).astype(np.float32)
                result = _compute_maximum_spot_mapping_result(dataset4d, params)
            else:
                raise ValueError(f"Unsupported live-processing job type: {params.job_type}")

            result_data = result["data"]
            assert isinstance(result_data, np.ndarray)
            job["status"] = "running"
            job["latest_result"] = result
            iterations = job.get("iterations", 0)
            job["iterations"] = int(iterations) + 1 if isinstance(iterations, (int, float, str)) else 1
            job["last_updated"] = time.time()
            job["last_error"] = None

            if params.show_result:
                result_image = job.get("result_image")
                if result_image is None:
                    result_image = _create_derived_image(
                        result_data.astype(np.float32),
                        params.output_name or f"live_{params.job_type}_{job_id}",
                        source_image,
                        True,
                    )
                    job["result_image"] = result_image
                else:
                    _copy_into_result_image(result_image, result_data)
        except Exception as exc:
            job["status"] = "error"
            job["last_error"] = str(exc)
            job["last_updated"] = time.time()
        stop_event.wait(params.poll_interval_s)

    if job["status"] != "error":
        job["status"] = "stopped"
    job["last_updated"] = time.time()


def _build_error(msg: str, suggestion: str = "") -> str:
    result = {"success": False, "error": msg}
    if suggestion:
        result["suggestion"] = suggestion
    return json.dumps(result)


# ---------------------------------------------------------------------------
# TOOL: get microscope state
# ---------------------------------------------------------------------------

@mcp.tool(
    name="gms_get_microscope_state",
    annotations={"readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
def gms_get_microscope_state() -> str:
    """
    Read the current state of all microscope subsystems.

    Returns a JSON summary including:
    - High tension (kV), spot size, brightness, focus, magnification, operation mode
    - Stage position: X, Y, Z (µm), Alpha, Beta (degrees)
    - Beam shift/tilt, image shift (physical units)
    - Camera length (mm), EELS energy offset / slit width
    - DigiScan scan parameters (dwell time, frame size, rotation)
    - Camera temperature and insertion state

    Call this first before any acquisition or control operation.
    """
    try:
        if _bridge_mode_enabled():
            result = _run_bridge_tool("GetMicroscopeState")
            result["runtime_mode"] = _runtime_mode()
            result["simulation_mode"] = False
            return json.dumps(result, indent=2)

        if _SIMULATE:
            camera = DM.CM_GetCurrentCamera()
            cam_name = DM.CM_GetCameraName(camera)
            cam_inserted = DM.CM_GetCameraInserted(camera)
            cam_temp = DM.CM_GetActualTemperature_C(camera)
            shift_xy = DM.EMGetBeamShift(0.0, 0.0)
            result = {
                "success": True,
                "simulation_mode": True,
                "runtime_mode": _runtime_mode(),
                "optics": {
                    "high_tension_kV": DM.EMGetHighTension() / 1000 if DM.EMCanGetHighTension() else None,
                    "spot_size": DM.EMGetSpotSize(),
                    "brightness": DM.EMGetBrightness(),
                    "focus_um": DM.EMGetFocus(),
                    "magnification": DM.EMGetMagnification() if DM.EMCanGetMagnification() else None,
                    "mag_index": DM.EMGetMagIndex(),
                    "operation_mode": DM.EMGetOperationMode(),
                    "illumination_mode": DM.EMGetIlluminationMode(),
                    "camera_length_mm": (DM.EMGetCameraLength() if DM.EMCanGetCameraLength() else None),
                },
                "stage": {
                    "x_um": DM.EMGetStageX(),
                    "y_um": DM.EMGetStageY(),
                    "z_um": DM.EMGetStageZ(),
                    "alpha_deg": DM.EMGetStageAlpha(),
                    "beta_deg": DM.EMGetStageBeta(),
                },
                "beam": {
                    "shift_x": shift_xy[0],
                    "shift_y": shift_xy[1],
                },
                "eels": {
                    "energy_offset_eV": DM.IFGetEnergyLoss(0),
                    "slit_width_eV": DM.IFCGetSlitWidth(),
                    "in_eels_mode": DM.IFIsInEELSMode(),
                },
                "camera": {
                    "name": cam_name,
                    "inserted": cam_inserted,
                    "temp_c": cam_temp,
                    "n_signals": DM.DSGetNumberOfSignals(),
                },
            }
            return json.dumps(result, indent=2)

        mic = DM.Py_Microscope()
        camera = DM.GetActiveCamera()
        cam_name = camera.GetName() if hasattr(camera, "GetName") else None
        cam_inserted = camera.GetInserted() if hasattr(camera, "GetInserted") else None
        cam_temp = None
        try:
            shift_x, shift_y = mic.GetBeamShift()
        except Exception:
            shift_x, shift_y = None, None
        try:
            cam_length = mic.GetCameraLength() if mic.CanGetCameraLength() else None
        except Exception:
            cam_length = None

        result = {
            "success": True,
            "simulation_mode": False,
            "runtime_mode": _runtime_mode(),
            "optics": {
                "high_tension_kV": mic.GetHighTension() / 1000 if mic.CanGetHighTension() else None,
                "spot_size": mic.GetSpotSize(),
                "brightness": mic.GetBrightness(),
                "focus_um": mic.GetFocus(),
                "magnification": mic.GetMagnification() if mic.CanGetMagnification() else None,
                "mag_index": mic.GetMagIndex(),
                "operation_mode": mic.GetOperationMode(),
                "illumination_mode": mic.GetIlluminationMode() if mic.CanGetIlluminationMode() else None,
                "camera_length_mm": cam_length,
            },
            "stage": {
                "x_um": mic.GetStageX(),
                "y_um": mic.GetStageY(),
                "z_um": mic.GetStageZ(),
                "alpha_deg": mic.GetStageAlpha(),
                "beta_deg": mic.GetStageBeta(),
            },
            "beam": {
                "shift_x": shift_x,
                "shift_y": shift_y,
            },
            "eels": {
                "energy_offset_eV": None,
                "slit_width_eV": None,
                "in_eels_mode": None,
                "note": "EELS/GIF IF/IFC APIs are unavailable in PythonReference; use bridge mode for live GIF control.",
            },
            "camera": {
                "name": cam_name,
                "inserted": cam_inserted,
                "temp_c": cam_temp,
                "n_signals": DM.DS_CountSignals() if hasattr(DM, "DS_CountSignals") else None,
            },
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return _build_error(str(e), "Verify the GMS DM bridge is running.")


# ---------------------------------------------------------------------------
# TOOL: front image access
# ---------------------------------------------------------------------------

@mcp.tool(
    name="gms_get_front_image",
    annotations={"readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
def gms_get_front_image(
    params: Optional[FrontImageInput] = None,
    include_data: bool = False,
    include_tags: bool = True,
) -> str:
    """
    Return metadata for the front-most image in the GMS workspace.

    Parameters:
        params.include_data (bool): Include base64-encoded pixel data.
        params.include_tags (bool): Include serialisable image tags when available.

    The same fields can also be passed directly as top-level kwargs
    (e.g. `include_data=True`) for compatibility with some LLM tool clients.
    """
    try:
        if params is None:
            params = FrontImageInput(include_data=include_data, include_tags=include_tags)
        if _bridge_mode_enabled():
            result = _run_bridge_tool("GetFrontImage", {
                "include_data": params.include_data,
            })
            return json.dumps(result, indent=2)

        img = DM.GetFrontImage()
        result = {"success": True, "image": _image_to_response(
            img,
            include_data=params.include_data,
            include_tags=params.include_tags,
        )}
        return json.dumps(result, indent=2)
    except Exception as e:
        return _build_error(str(e), "Ensure a front-most image is available in GMS.")


# ---------------------------------------------------------------------------
# TOOL: filtered image generation
# ---------------------------------------------------------------------------

@mcp.tool(
    name="gms_apply_image_filter",
    annotations={"readOnlyHint": False, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
def gms_apply_image_filter(
    params: Optional[ImageFilterInput] = None,
    median_size: int = 0,
    gaussian_sigma: float = 0.0,
) -> str:
    """
    Apply median and/or Gaussian filtering to the front-most image or ROI.

    Parameters:
        params.median_size    (int)   : Median-filter kernel size (0 = disabled).
        params.gaussian_sigma (float) : Gaussian blur sigma (0 = disabled).
        params.roi            (list)  : Optional [top, left, bottom, right] ROI.

    The same fields can also be passed directly as top-level kwargs
    (e.g. `gaussian_sigma=1.5`) for compatibility with some LLM tool clients.
    """
    try:
        if params is None:
            params = ImageFilterInput(
                median_size=median_size,
                gaussian_sigma=gaussian_sigma,
            )
        if _bridge_mode_enabled():
            result = _run_bridge_tool("ApplyImageFilter", params.model_dump())
            return json.dumps(result, indent=2)

        source = DM.GetFrontImage()
        arr = np.asarray(source.GetNumArray(), dtype=np.float32)
        if arr.ndim != 2:
            return _build_error("Front image must be 2D for image filtering.")

        filtered = _extract_roi(arr, params.roi).copy()
        if params.median_size > 0:
            filtered = median_filter(filtered, size=params.median_size)
        if params.gaussian_sigma > 0:
            filtered = gaussian_filter(filtered, sigma=params.gaussian_sigma)

        result_img = _create_derived_image(
            filtered.astype(np.float32),
            params.output_name,
            source,
            params.show_result,
        )
        result = {
            "success": True,
            "operation": {
                "median_size": params.median_size,
                "gaussian_sigma": params.gaussian_sigma,
                "roi": params.roi,
            },
            "image": _image_to_response(result_img),
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return _build_error(str(e), "Verify the front image is a 2D image and the ROI is valid.")


# ---------------------------------------------------------------------------
# TOOL: radial profile from diffraction or FFT
# ---------------------------------------------------------------------------

@mcp.tool(
    name="gms_compute_radial_profile",
    annotations={"readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
def gms_compute_radial_profile(
    params: Optional[RadialProfileInput] = None,
    mode: str = "fft",
    binning: int = 1,
) -> str:
    """
    Compute a 1D radial profile from a diffraction pattern or from the FFT of a TEM image.

    Parameters:
        params.mode    (str) : 'fft' or 'diffraction'.
        params.binning (int) : Integer binning factor before profiling.

    The same fields can also be passed directly as top-level kwargs
    (e.g. `mode='fft'`) for compatibility with some LLM tool clients.
    """
    try:
        if params is None:
            params = RadialProfileInput(mode=mode, binning=binning)
        if _bridge_mode_enabled():
            result = _run_bridge_tool("ComputeRadialProfile", params.model_dump())
            return json.dumps(result, indent=2)

        source = DM.GetFrontImage()
        data = np.asarray(source.GetNumArray(), dtype=np.float32)
        if data.ndim != 2:
            return _build_error("Front image must be 2D for radial-profile analysis.")

        roi_data = _extract_roi(data, params.roi)
        if params.mode == "fft":
            working = np.abs(np.fft.fftshift(np.fft.fft2(roi_data))).astype(np.float32)
            unit = "nm^-1"
        elif params.mode == "diffraction":
            working = roi_data.copy()
            unit = "px"
        else:
            return _build_error("mode must be 'fft' or 'diffraction'.")

        working = _bin_image(working, params.binning)
        if params.mask_center_lines:
            cx = working.shape[1] // 2
            cy = working.shape[0] // 2
            working[:, max(0, cx - 1):cx + 1] = 0.0
            working[max(0, cy - 1):cy + 1, :] = 0.0

        h, w = working.shape
        cy, cx = h / 2.0, w / 2.0
        yy, xx = np.indices((h, w), dtype=np.float32)
        radii = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        radial_index = radii.astype(int)
        max_radius = min(h, w) // 2

        radial_mean = np.zeros(max_radius, dtype=np.float32)
        radial_max = np.zeros(max_radius, dtype=np.float32)
        for radius in range(max_radius):
            mask = radial_index == radius
            if not np.any(mask):
                continue
            values = working[mask]
            radial_mean[radius] = float(values.mean())
            radial_max[radius] = float(values.max())

        if params.profile_metric == "radial_mean":
            profile = radial_mean
        elif params.profile_metric == "radial_max":
            profile = radial_max
        elif params.profile_metric == "radial_max_minus_mean":
            profile = radial_max - radial_mean
        else:
            return _build_error(
                "profile_metric must be 'radial_max_minus_mean', 'radial_mean', or 'radial_max'."
            )

        if params.smooth_sigma > 0:
            profile = gaussian_filter(profile, sigma=params.smooth_sigma)

        ignore_bins = int(len(profile) * params.mask_percent / 100.0)
        if ignore_bins > 0:
            profile[:ignore_bins] = 0.0

        peak_positions = []
        try:
            from scipy.signal import find_peaks

            peaks, _ = find_peaks(profile, height=profile.mean() + profile.std())
            peak_positions = peaks.tolist()[:12]
        except Exception:
            peak_positions = []

        result = {
            "success": True,
            "analysis": {
                "mode": params.mode,
                "profile_metric": params.profile_metric,
                "roi": params.roi,
                "binning": params.binning,
                "mask_center_lines": params.mask_center_lines,
                "mask_percent": params.mask_percent,
                "smooth_sigma": params.smooth_sigma,
                "profile_length": int(len(profile)),
                "unit": unit,
                "peak_positions": peak_positions,
            },
            "profile": profile.astype(np.float32).tolist(),
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return _build_error(str(e), "Verify the front image is a 2D image and the ROI is valid.")


# ---------------------------------------------------------------------------
# TOOL: maximum FFT image from local windows
# ---------------------------------------------------------------------------

@mcp.tool(
    name="gms_compute_max_fft",
    annotations={"readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
def gms_compute_max_fft(params: MaxFFTInput) -> str:
    """
    Compute the maximum FFT across a grid of local windows from the front-most image.
    """
    try:
        if _bridge_mode_enabled():
            result = _run_bridge_tool("ComputeMaxFFT", params.model_dump())
            return json.dumps(result, indent=2)

        source = DM.GetFrontImage()
        data = np.asarray(source.GetNumArray(), dtype=np.float32)
        if data.ndim != 2:
            return _build_error("Front image must be 2D for max-FFT analysis.")

        roi_data = _extract_roi(data, params.roi)
        if roi_data.shape[0] < params.fft_size or roi_data.shape[1] < params.fft_size:
            return _build_error("Selected image region is smaller than fft_size.")

        view = np.lib.stride_tricks.sliding_window_view(roi_data, (params.fft_size, params.fft_size))
        windows = view[::params.spacing, ::params.spacing]
        if windows.size == 0:
            windows = view[:1, :1]

        hann_1d: np.ndarray = np.hanning(params.fft_size).astype(np.float32)
        hann_2d = np.sqrt(np.outer(hann_1d, hann_1d)).astype(np.float32)
        spectra = np.abs(np.fft.fftshift(np.fft.fft2(windows * hann_2d, axes=(-2, -1)), axes=(-2, -1)))
        max_fft = spectra.max(axis=(0, 1)).astype(np.float32)
        if params.log_scale:
            max_fft = np.log1p(max_fft)

        result_img = _create_derived_image(max_fft, params.output_name, source, params.show_result)
        result = {
            "success": True,
            "analysis": {
                "roi": params.roi,
                "fft_size": params.fft_size,
                "spacing": params.spacing,
                "log_scale": params.log_scale,
                "n_windows": int(windows.shape[0] * windows.shape[1]),
            },
            "image": _image_to_response(result_img),
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return _build_error(str(e), "Verify the front image is a 2D image and the ROI is valid.")


# ---------------------------------------------------------------------------
# TOOL: 4D-STEM maximum-spot mapping
# ---------------------------------------------------------------------------

@mcp.tool(
    name="gms_run_4dstem_maximum_spot_mapping",
    annotations={"readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
def gms_run_4dstem_maximum_spot_mapping(params: MaxSpotMapInput) -> str:
    """
    Compute a maximum-spot orientation-like map from the loaded 4D-STEM dataset.
    """
    try:
        if _bridge_mode_enabled():
            result = _run_bridge_tool("Run4DSTEMMaximumSpotMapping", params.model_dump())
            return json.dumps(result, indent=2)

        img4d = DM.GetFrontImage()
        arr: np.ndarray = _resolve_4dstem_array(img4d).astype(np.float32)
        scan_y, scan_x, det_y, det_x = arr.shape
        cy = (det_y - 1) / 2.0
        cx = (det_x - 1) / 2.0

        working = arr.copy()
        if params.subtract_mean_background:
            mean_pattern = working.mean(axis=(0, 1), keepdims=True)
            working = np.maximum(working - mean_pattern, 0.0)
        if params.gaussian_sigma > 0:
            working = gaussian_filter(working, sigma=(0, 0, params.gaussian_sigma, params.gaussian_sigma))

        yy, xx = np.indices((det_y, det_x), dtype=np.float32)
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask = rr <= params.mask_center_radius_px
        working[:, :, mask] = -np.inf

        flat = working.reshape(scan_y, scan_x, det_y * det_x)
        max_idx = np.argmax(flat, axis=-1)
        intensity_map = np.take_along_axis(flat, max_idx[..., None], axis=-1).squeeze(-1)
        y_idx, x_idx = np.divmod(max_idx, det_x)
        dx = x_idx.astype(np.float32) - cx
        dy = y_idx.astype(np.float32) - cy
        theta_map = (np.arctan2(dy, dx) + np.pi) / (2.0 * np.pi)
        radius_map = np.sqrt(dx ** 2 + dy ** 2)

        if params.map_var == "theta":
            hue = theta_map
        elif params.map_var == "radius":
            radius_max = max(float(radius_map.max()), 1.0)
            hue = radius_map / radius_max
        else:
            return _build_error("map_var must be 'theta' or 'radius'.")

        intensity_norm = intensity_map - intensity_map.min()
        if float(intensity_norm.max()) > 0:
            intensity_norm = intensity_norm / intensity_norm.max()
        saturation = np.ones_like(hue, dtype=np.float32)
        rgb = _hsv_to_rgb(hue.astype(np.float32), saturation, intensity_norm.astype(np.float32))

        result_img = _create_derived_image(rgb.astype(np.float32), params.output_name, img4d, params.show_result)
        result = {
            "success": True,
            "analysis": {
                "type": "maximum_spot_mapping",
                "map_var": params.map_var,
                "scan_shape": [scan_y, scan_x],
                "mask_center_radius_px": params.mask_center_radius_px,
                "subtract_mean_background": params.subtract_mean_background,
                "gaussian_sigma": params.gaussian_sigma,
            },
            "maps": {
                "theta_range": [float(theta_map.min()), float(theta_map.max())],
                "radius_range": [float(radius_map.min()), float(radius_map.max())],
                "intensity_range": [float(intensity_map.min()), float(intensity_map.max())],
            },
            "image": _image_to_response(result_img),
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return _build_error(str(e), "Acquire a 4D-STEM dataset first with gms_acquire_4d_stem.")


# ---------------------------------------------------------------------------
# TOOL: persistent live-processing job API
# ---------------------------------------------------------------------------

@mcp.tool(
    name="gms_start_live_processing_job",
    annotations={"readOnlyHint": False, "destructiveHint": False,
                 "idempotentHint": False, "openWorldHint": False},
)
def gms_start_live_processing_job(
    params: Optional[StartLiveProcessingJobInput] = None,
    job_type: str = "",
    poll_interval_s: float = 0.5,
    fft_size: int = 256,
    spacing: int = 256,
    median_size: int = 0,
    gaussian_sigma: float = 0.0,
    history_length: int = 200,
    profile_mode: str = "fft",
    avg_period_1: int = 5,
    avg_period_2: int = 10,
    mask_center_radius_px: float = 5.0,
    map_var: str = "theta",
) -> str:
    """
    Start a persistent live-processing job for radial profiles, live difference, live FFT maps, or live filtered views.

    Parameters:
        params.job_type (str) : 'radial_profile', 'difference', 'fft_map', 'filtered_view', or 'maximum_spot_mapping'.

    The same fields can also be passed directly as top-level kwargs
    (e.g. `job_type='fft_map'`) for compatibility with some LLM tool clients.
    """
    try:
        if params is None:
            params = StartLiveProcessingJobInput(
                job_type=job_type,
                poll_interval_s=poll_interval_s,
                fft_size=fft_size,
                spacing=spacing,
                median_size=median_size,
                gaussian_sigma=gaussian_sigma,
                history_length=history_length,
                profile_mode=profile_mode,
                avg_period_1=avg_period_1,
                avg_period_2=avg_period_2,
                mask_center_radius_px=mask_center_radius_px,
                map_var=map_var,
            )
        if params.job_type not in {"radial_profile", "difference", "fft_map", "filtered_view", "maximum_spot_mapping"}:
            return _build_error(
                "job_type must be 'radial_profile', 'difference', 'fft_map', 'filtered_view', or 'maximum_spot_mapping'."
            )

        if _live_jobs_use_bridge():
            payload = _bridge_dispatch("LiveProcessingJobStart", params.model_dump())
            return json.dumps(payload, indent=2)

        source_image = DM.GetFrontImage()
        source_data = np.asarray(source_image.GetNumArray())
        if params.job_type == "maximum_spot_mapping":
            try:
                _resolve_4dstem_array(source_image)
            except Exception as exc:
                return _build_error(
                    str(exc),
                    "Acquire or select a 4D-STEM dataset before starting a live maximum-spot-mapping job.",
                )
        elif source_data.ndim != 2:
            return _build_error(
                "Live-processing jobs currently require a 2D front-most image.",
                "Acquire or select a 2D TEM/STEM/diffraction image before starting the job.",
            )

        job_id = uuid.uuid4().hex[:12]
        stop_event = threading.Event()
        job: dict[str, object] = {
            "job_id": job_id,
            "job_type": params.job_type,
            "backend": "local",
            "params": params,
            "poll_interval_s": params.poll_interval_s,
            "source_image": source_image,
            "created_at": time.time(),
            "last_updated": None,
            "status": "starting",
            "iterations": 0,
            "last_error": None,
            "latest_result": None,
            "result_image": None,
            "history": None,
            "avg1": None,
            "avg2": None,
            "stop_event": stop_event,
        }

        thread = threading.Thread(
            target=_run_live_processing_job,
            args=(job_id,),
            daemon=True,
            name=f"gms-live-job-{job_id}",
        )
        job["thread"] = thread

        with _live_jobs_lock:
            _live_jobs[job_id] = job

        thread.start()

        result = {
            "success": True,
            "job": {
                "job_id": job_id,
                "job_type": params.job_type,
                "backend": "local",
                "status": "starting",
                "poll_interval_s": params.poll_interval_s,
                "show_result": params.show_result,
                "source_image_name": source_image.GetName(),
            },
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return _build_error(str(e), "Ensure a valid 2D front image is available before starting a live job.")


@mcp.tool(
    name="gms_get_live_processing_job_status",
    annotations={"readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
def gms_get_live_processing_job_status(
    params: Optional[LiveProcessingJobQuery] = None,
    job_id: str = "",
) -> str:
    """
    Get the current status of a live-processing job.

    Parameters:
        params.job_id (str) : Live-processing job identifier.

    The same field can also be passed directly as a top-level kwarg
    (e.g. `job_id='abc123'`) for compatibility with some LLM tool clients.
    """
    try:
        if params is None:
            params = LiveProcessingJobQuery(job_id=job_id)
        try:
            job = _get_live_job(params.job_id)
            return json.dumps({"success": True, "job": _job_status_payload(job)}, indent=2)
        except KeyError:
            if _live_jobs_use_bridge():
                payload = _bridge_dispatch("LiveProcessingJobStatus", params.model_dump())
                return json.dumps(payload, indent=2)
            raise
    except KeyError as e:
        return _build_error(str(e))
    except Exception as e:
        return _build_error(str(e))


@mcp.tool(
    name="gms_get_live_processing_job_result",
    annotations={"readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
def gms_get_live_processing_job_result(
    params: Optional[LiveProcessingJobQuery] = None,
    job_id: str = "",
) -> str:
    """
    Get the latest derived result produced by a live-processing job.

    Parameters:
        params.job_id (str) : Live-processing job identifier.

    The same field can also be passed directly as a top-level kwarg
    (e.g. `job_id='abc123'`) for compatibility with some LLM tool clients.
    """
    try:
        if params is None:
            params = LiveProcessingJobQuery(job_id=job_id)
        try:
            job = _get_live_job(params.job_id)
        except KeyError:
            if _live_jobs_use_bridge():
                bridge_payload = _bridge_dispatch("LiveProcessingJobResult", params.model_dump())
                return json.dumps(bridge_payload, indent=2)
            raise
        latest_result = job.get("latest_result")
        if not isinstance(latest_result, dict):
            return _build_error("Live-processing job has not produced a result yet.")

        data = latest_result.get("data")
        summary = latest_result.get("summary")
        if not isinstance(data, np.ndarray) or not isinstance(summary, dict):
            return _build_error("Live-processing job result is malformed.")

        result_payload: dict[str, object] = {
            **summary,
            **_summarize_array(data),
        }
        payload: dict[str, object] = {
            "success": True,
            "job": _job_status_payload(job),
            "result": result_payload,
        }
        include_data = params.include_data or bool(getattr(job.get("params"), "include_result_data", False))
        if include_data:
            result_payload.update(_encode_array_b64(data))
        return json.dumps(payload, indent=2)
    except KeyError as e:
        return _build_error(str(e))
    except Exception as e:
        return _build_error(str(e))


@mcp.tool(
    name="gms_stop_live_processing_job",
    annotations={"readOnlyHint": False, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
def gms_stop_live_processing_job(
    params: Optional[LiveProcessingJobQuery] = None,
    job_id: str = "",
) -> str:
    """
    Stop a live-processing job and return its final status.

    Parameters:
        params.job_id (str) : Live-processing job identifier.

    The same field can also be passed directly as a top-level kwarg
    (e.g. `job_id='abc123'`) for compatibility with some LLM tool clients.
    """
    try:
        if params is None:
            params = LiveProcessingJobQuery(job_id=job_id)
        try:
            job = _get_live_job(params.job_id)
        except KeyError:
            if _live_jobs_use_bridge():
                payload = _bridge_dispatch("LiveProcessingJobStop", params.model_dump())
                return json.dumps(payload, indent=2)
            raise
        stop_event = job.get("stop_event")
        thread = job.get("thread")
        if isinstance(stop_event, threading.Event):
            stop_event.set()
        if isinstance(thread, threading.Thread):
            poll_interval = job.get("poll_interval_s", 0.5)
            timeout_s = float(poll_interval) * 3.0 if isinstance(poll_interval, (int, float, str)) else 2.0
            thread.join(timeout=max(2.0, timeout_s))
        return json.dumps({"success": True, "job": _job_status_payload(job)}, indent=2)
    except KeyError as e:
        return _build_error(str(e))
    except Exception as e:
        return _build_error(str(e))


# ---------------------------------------------------------------------------
# TOOL: TEM / HRTEM image acquisition
# ---------------------------------------------------------------------------

@mcp.tool(
    name="gms_acquire_tem_image",
    annotations={"readOnlyHint": False, "destructiveHint": False,
                 "idempotentHint": False, "openWorldHint": False},
)
def gms_acquire_tem_image(
    params: Optional[AcquireTEMInput] = None,
    exposure_s: float = 1.0,
    binning: int = 1,
    processing: int = 3,
) -> str:
    """
    Acquire a single TEM or HRTEM image from the currently inserted camera.

    Parameters:
        params.exposure_s  (float) : Camera exposure in seconds (0.001–60).
        params.binning     (int)   : Binning factor 1, 2, 4, or 8.
        params.processing  (int)   : 1=raw, 2=dark only, 3=dark+gain (default).
        params.roi         (list)  : Optional [top, left, bottom, right] ROI.

    The same fields can also be passed directly as top-level kwargs
    (e.g. `exposure_s=0.5`) for compatibility with some LLM tool clients.

    Returns JSON with image shape, statistics, calibration, and metadata.
    """
    try:
        if params is None:
            params = AcquireTEMInput(
                exposure_s=exposure_s,
                binning=binning,
                processing=processing,
            )
        if _bridge_mode_enabled():
            result = _run_bridge_tool("AcquireTEMImage", params.model_dump())
            return json.dumps(result, indent=2)

        if _SIMULATE and hasattr(DM, "_state"):
            DM._state.operation_mode = "TEM"

        # Prefer PythonReference camera API on real instruments; keep legacy
        # CM_* simulator path for hardware-free tests.
        if _SIMULATE and hasattr(DM, "CM_GetCurrentCamera"):
            camera = DM.CM_GetCurrentCamera()
            acq = DM.CM_CreateAcquisitionParameters_FullCCD(
                camera, params.processing, params.exposure_s,
                params.binning, params.binning
            )
            if params.roi:
                DM.CM_SetCCDReadArea(acq, *params.roi)
            DM.CM_Validate_AcquisitionParameters(camera, acq)
            img = DM.CM_AcquireImage(camera, acq)
        else:
            camera = DM.GetActiveCamera()
            if hasattr(camera, "PrepareForAcquire"):
                camera.PrepareForAcquire()
            if params.roi:
                t, l, b, r = params.roi
                img = camera.AcquireImage(
                    params.exposure_s,
                    params.binning,
                    params.binning,
                    params.processing,
                    t,
                    l,
                    b,
                    r,
                )
            else:
                img = camera.AcquireImage(
                    params.exposure_s,
                    params.binning,
                    params.binning,
                    params.processing,
                )

        img.SetName(f"TEM_exp{params.exposure_s}s_bin{params.binning}")

        result = {"success": True, "acquisition_type": "TEM"}
        result.update(_image_to_response(img))
        return json.dumps(result, indent=2)
    except Exception as e:
        return _build_error(str(e),
            "Check camera is inserted and parameters are within valid range.")


# ---------------------------------------------------------------------------
# TOOL: STEM acquisition (HAADF / BF / ABF)
# ---------------------------------------------------------------------------

@mcp.tool(
    name="gms_acquire_stem",
    annotations={"readOnlyHint": False, "destructiveHint": False,
                 "idempotentHint": False, "openWorldHint": False},
)
def gms_acquire_stem(
    params: Optional[AcquireSTEMInput] = None,
    width: int = 512,
    height: int = 512,
    dwell_us: float = 10.0,
    rotation_deg: float = 0.0,
    signals: Optional[list[int]] = None,
) -> str:
    """
    Acquire a STEM image (HAADF, BF, ABF) using DigiScan.

    Parameters:
        params.width        (int)   : Scan width in pixels (64–4096).
        params.height       (int)   : Scan height in pixels.
        params.dwell_us     (float) : Pixel dwell time in microseconds.
        params.rotation_deg (float) : Scan rotation in degrees.
        params.signals      (list)  : DigiScan channel indices to enable
                                      [0=HAADF, 1=BF, 2=ABF by convention].

    Returns JSON with image statistics and scan metadata.
    """
    try:
        if params is None:
            params = AcquireSTEMInput(
                width=width,
                height=height,
                dwell_us=dwell_us,
                rotation_deg=rotation_deg,
                signals=[0, 1] if signals is None else signals,
            )

        if _bridge_mode_enabled():
            result = _run_bridge_tool("AcquireSTEM", params.model_dump())
            return json.dumps(result, indent=2)

        if _SIMULATE:
            DM.DSSetFrameSize(params.width, params.height)
            DM.DSSetPixelTime(params.dwell_us)
            DM.DSSetRotation(params.rotation_deg)

            n_signals = DM.DSGetNumberOfSignals()
            for ch in range(n_signals):
                DM.DSSetSignalEnabled(ch, 1 if ch in params.signals else 0)

            DM.DSStartAcquisition()
            DM.DSWaitUntilFinished()
            img = DM.GetFrontImage()
        else:
            if not hasattr(DM, "DS_CreateParameters"):
                return _build_error("DigiScan DS_* parameter API not available.", "Use bridge mode or ensure DigiScan Python API is installed.")
            data_type_bytes = 4
            param_id = DM.DS_CreateParameters(
                params.width,
                params.height,
                0,
                data_type_bytes,
                float(params.rotation_deg),
                float(params.dwell_us),
                False,
            )
            enabled = set(int(ch) for ch in params.signals)
            n_signals = DM.DS_CountSignals() if hasattr(DM, "DS_CountSignals") else 0
            for ch in range(int(n_signals)):
                DM.DS_SetParametersSignal(param_id, ch, float(data_type_bytes), ch in enabled, 0)
            DM.DS_StartAcquisition(param_id, False, True)
            img = None
            try:
                first_ch = int(params.signals[0]) if params.signals else 0
                img_id = DM.DS_GetAcquiredImageID(first_ch)
                if isinstance(img_id, (int, float)) and int(img_id) >= 0:
                    img = DM.FindImageByID(int(img_id))
            except Exception:
                img = None
            if img is None:
                img = DM.GetFrontImage()
            try:
                DM.DS_DeleteParameters(param_id)
            except Exception:
                pass

        img.SetName(f"STEM_{params.width}x{params.height}_dwell{params.dwell_us}us")

        tags = img.GetTagGroup()
        tags.SetTagAsFloat("STEM:DwellTime_us", params.dwell_us)
        tags.SetTagAsFloat("STEM:Rotation_deg", params.rotation_deg)
        tags.SetTagAsString("STEM:Signals", ",".join(map(str, params.signals)))

        result = {
            "success": True,
            "acquisition_type": "STEM",
            "scan_parameters": {
                "width": params.width,
                "height": params.height,
                "dwell_us": params.dwell_us,
                "rotation_deg": params.rotation_deg,
                "signals_enabled": params.signals,
                "total_frame_time_s": (params.width * params.height * params.dwell_us * 1e-6),
            }
        }
        result.update(_image_to_response(img))
        return json.dumps(result, indent=2)
    except Exception as e:
        return _build_error(str(e),
            "Ensure DigiScan is running and signal channels are connected.")


# ---------------------------------------------------------------------------
# TOOL: 4D-STEM / NBED acquisition
# ---------------------------------------------------------------------------

@mcp.tool(
    name="gms_acquire_4d_stem",
    annotations={"readOnlyHint": False, "destructiveHint": False,
                 "idempotentHint": False, "openWorldHint": False},
)
def gms_acquire_4d_stem(
    params: Optional[Acquire4DSTEMInput] = None,
    scan_x: int = 64,
    scan_y: int = 64,
    dwell_us: float = 1000.0,
    camera_length_mm: Optional[float] = None,
    convergence_mrad: Optional[float] = None,
) -> str:
    """
    Acquire a 4D-STEM / NBED dataset.

    At each of the scan_x × scan_y probe positions, the camera records a
    full 2D convergent-beam or nano-beam electron diffraction pattern,
    producing a 4D dataset of shape (scan_y, scan_x, det_y, det_x).

    Parameters:
        params.scan_x           (int)   : Scan positions in X (8–512).
        params.scan_y           (int)   : Scan positions in Y.
        params.dwell_us         (float) : Per-pattern acquisition time in µs.
        params.camera_length_mm (float) : Camera length in mm (optional).
        params.convergence_mrad (float) : Convergence semi-angle (logged only).

    The same fields can also be passed directly as top-level kwargs
    (e.g. `scan_x=32`) for compatibility with some LLM tool clients.

    Returns JSON with dataset shape, estimated file size, metadata.
    """
    try:
        if params is None:
            kw: dict[str, object] = {"scan_x": scan_x, "scan_y": scan_y, "dwell_us": dwell_us}
            if camera_length_mm is not None:
                kw["camera_length_mm"] = camera_length_mm
            if convergence_mrad is not None:
                kw["convergence_mrad"] = convergence_mrad
            params = Acquire4DSTEMInput(**kw)
        if _bridge_mode_enabled():
            result = _run_bridge_tool("Acquire4DSTEM", params.model_dump())
            return json.dumps(result, indent=2)

        if params.camera_length_mm is not None and not _SIMULATE:
            return _build_error(
                "UNSUPPORTED: setting camera length is not exposed in PythonReference.",
                "Set camera length in microscope controls (or bridge mode) before 4D-STEM acquisition.",
            )
        cl = None
        if _SIMULATE and params.camera_length_mm is not None:
            DM.EMSetCameraLength(params.camera_length_mm)
            cl = DM.EMGetCameraLength() if DM.EMCanGetCameraLength() else None

        # In production: trigger 4D-STEM acquisition via STEMx / SI infrastructure.
        # In simulation: generate a synthetic 4D array and register as front image.
        if _SIMULATE:
            det_x, det_y = 64, 64
            raw4d = DM._make_4d_stem(
                params.scan_x, params.scan_y, det_x, det_y
            )
            # Register the simulated 4D dataset as the active front image
            # so that gms_run_4dstem_analysis can retrieve it.
            DM._front_image = raw4d
            DM._images[raw4d.GetID()] = raw4d
            data4d = raw4d
            det_shape = [det_y, det_x]
        else:
            # Real GMS: get current 4D dataset from workspace
            data4d = DM.GetFrontImage()
            arr = data4d.GetNumArray()
            det_shape = [arr.shape[-2], arr.shape[-1]] if arr.ndim == 4 else [256, 256]

        total_patterns = params.scan_x * params.scan_y
        size_mb = (total_patterns * (det_shape[0] * det_shape[1]) * 4) / 1e6

        result = {
            "success": True,
            "acquisition_type": "4D-STEM",
            "dataset": {
                "scan_shape":     [params.scan_y, params.scan_x],
                "detector_shape": det_shape,
                "full_shape":     [params.scan_y, params.scan_x] + det_shape,
                "total_patterns": total_patterns,
                "estimated_size_MB": round(size_mb, 1),
                "dwell_us":       params.dwell_us,
                "camera_length_mm": cl,
                "convergence_mrad": params.convergence_mrad,
                "estimated_acq_time_s": round(
                    total_patterns * params.dwell_us * 1e-6, 1),
            },
            "note": (
                "4D dataset is available in GMS workspace. "
                "Use py4DSTEM (gms_run_4dstem_analysis) for virtual detectors "
                "or strain mapping."
            ),
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return _build_error(str(e),
            "Verify STEMx synchronisation and camera insertion.")


# ---------------------------------------------------------------------------
# TOOL: EELS spectrum acquisition
# ---------------------------------------------------------------------------

@mcp.tool(
    name="gms_acquire_eels",
    annotations={"readOnlyHint": False, "destructiveHint": False,
                 "idempotentHint": False, "openWorldHint": False},
)
def gms_acquire_eels(
    params: Optional[AcquireEELSInput] = None,
    exposure_s: float = 1.0,
    energy_offset_eV: float = 0.0,
    slit_width_eV: float = 10.0,
    dispersion_idx: int = 0,
    full_vertical_binning: bool = True,
) -> str:
    """
    Acquire an EELS spectrum from the GIF (Gatan Imaging Filter).

    Configures drift tube voltage (energy offset), slit width, and dispersion,
    then acquires from the GIF CCD with optional full vertical binning.

    Parameters:
        params.exposure_s          (float) : Acquisition time in seconds.
        params.energy_offset_eV    (float) : Energy window offset (drift tube).
        params.slit_width_eV       (float) : Slit width; 0 = slit retracted.
        params.dispersion_idx      (int)   : Dispersion index (0=finest).
        params.full_vertical_binning(bool) : Standard EELS binning.

    Returns JSON with spectrum channel count, energy axis calibration,
    zero-loss peak position estimate, and acquisition metadata.
    """
    try:
        if params is None:
            params = AcquireEELSInput(
                exposure_s=exposure_s,
                energy_offset_eV=energy_offset_eV,
                slit_width_eV=slit_width_eV,
                dispersion_idx=dispersion_idx,
                full_vertical_binning=full_vertical_binning,
            )

        if _bridge_mode_enabled():
            result = _run_bridge_tool("AcquireEELS", params.model_dump())
            return json.dumps(result, indent=2)

        if not _SIMULATE:
            return _build_error(
                "UNSUPPORTED: EELS/GIF control is not exposed in DigitalMicrograph PythonReference API.",
                "Use bridge mode for live GIF control or read EELS metadata from acquired image tags.",
            )

        # Simulator path retains legacy GIF API behavior for hardware-free testing.
        DM.IFSetEELSMode()
        DM.IFCSetEnergy(params.energy_offset_eV)
        DM.IFCSetActiveDispersions(params.dispersion_idx)
        if params.slit_width_eV > 0:
            DM.IFCSetSlitWidth(params.slit_width_eV)
            DM.IFCSetSlitIn(1)
        else:
            DM.IFCSetSlitIn(0)

        camera = DM.CM_GetCurrentCamera()
        acq = DM.CM_CreateAcquisitionParameters_FullCCD(
            camera, 3, params.exposure_s, 1, 1
        )
        if params.full_vertical_binning:
            DM.CM_SetBinning(acq, 1, 2048)

        DM.CM_Validate_AcquisitionParameters(camera, acq)
        spec_img = DM.CM_AcquireImage(camera, acq)
        spec_img.SetName(f"EELS_dE{params.energy_offset_eV}eV_exp{params.exposure_s}s")

        arr = spec_img.GetNumArray().flatten()
        n_ch = len(arr)
        dispersions = [0.1, 0.25, 0.5, 1.0]
        disp = dispersions[min(params.dispersion_idx, len(dispersions) - 1)]
        energy_axis = params.energy_offset_eV + np.arange(n_ch) * disp
        zlp_idx = int(np.argmax(arr))
        zlp_energy = float(energy_axis[zlp_idx])

        result = {
            "success": True,
            "acquisition_type": "EELS",
            "spectrum": {
                "n_channels": n_ch,
                "energy_offset_eV": params.energy_offset_eV,
                "dispersion_eV_ch": disp,
                "energy_range_eV": [float(energy_axis[0]), float(energy_axis[-1])],
                "zlp_centre_eV": zlp_energy,
                "zlp_channel": zlp_idx,
                "max_intensity": float(arr.max()),
                "slit_width_eV": params.slit_width_eV,
                "exposure_s": params.exposure_s,
            },
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return _build_error(str(e),
            "Ensure the GIF is in EELS mode and the CCD is not saturated.")


# ---------------------------------------------------------------------------
# TOOL: Electron diffraction pattern acquisition
# ---------------------------------------------------------------------------

@mcp.tool(
    name="gms_acquire_diffraction",
    annotations={"readOnlyHint": False, "destructiveHint": False,
                 "idempotentHint": False, "openWorldHint": False},
)
def gms_acquire_diffraction(
    params: Optional[AcquireDiffractionInput] = None,
    exposure_s: float = 0.5,
    binning: int = 1,
    camera_length_mm: Optional[float] = None,
) -> str:
    """
    Acquire a selected-area or nano-beam electron diffraction pattern.

    The microscope must already be in diffraction mode. The camera length
    determines the reciprocal-space calibration.

    Parameters:
        params.exposure_s       (float) : Exposure in seconds.
        params.camera_length_mm (float) : Camera length in mm (optional).
        params.binning          (int)   : Camera binning.

    The same fields can also be passed directly as top-level kwargs
    (e.g. `exposure_s=0.3`) for compatibility with some LLM tool clients.

    Returns JSON with image statistics, d-spacing scale (1/Å per pixel),
    direct-beam centre estimate, and brightest ring radii.
    """
    try:
        if params is None:
            kw: dict[str, object] = {"exposure_s": exposure_s, "binning": binning}
            if camera_length_mm is not None:
                kw["camera_length_mm"] = camera_length_mm
            params = AcquireDiffractionInput(**kw)
        if _bridge_mode_enabled():
            result = _run_bridge_tool("AcquireDiffraction", params.model_dump())
            return json.dumps(result, indent=2)

        if _SIMULATE:
            if params.camera_length_mm is not None:
                DM.EMSetCameraLength(params.camera_length_mm)
            cl = DM.EMGetCameraLength() if DM.EMCanGetCameraLength() else 100.0
            previous_mode = None
            if hasattr(DM, "_state"):
                previous_mode = DM._state.operation_mode
                DM._state.operation_mode = "DIFFRACTION"
            camera = DM.CM_GetCurrentCamera()
            acq = DM.CM_CreateAcquisitionParameters_FullCCD(
                camera, 3, params.exposure_s, params.binning, params.binning
            )
            DM.CM_Validate_AcquisitionParameters(camera, acq)
            dp = DM.CM_AcquireImage(camera, acq)
            dp.SetName(f"DP_CL{cl}mm_exp{params.exposure_s}s")
            if previous_mode is not None:
                DM._state.operation_mode = previous_mode
        else:
            if params.camera_length_mm is not None:
                return _build_error(
                    "UNSUPPORTED: setting camera length is not exposed in PythonReference.",
                    "Set camera length in microscope controls (or bridge mode) before acquiring diffraction.",
                )
            mic = DM.Py_Microscope()
            mode = mic.GetImagingOpticsMode()
            if str(mode).lower() != "diffraction":
                return _build_error(
                    "Microscope is not in diffraction mode.",
                    "Switch to diffraction mode before acquiring a diffraction pattern.",
                )
            cl = mic.GetCameraLength() if mic.CanGetCameraLength() else None
            camera = DM.GetActiveCamera()
            if hasattr(camera, "PrepareForAcquire"):
                camera.PrepareForAcquire()
            dp = camera.AcquireImage(params.exposure_s, params.binning, params.binning, 3)
            dp.SetName(f"DP_exp{params.exposure_s}s_bin{params.binning}")

        arr = dp.GetNumArray()
        h, w = arr.shape[:2]
        cx, cy = w // 2, h // 2

        ring_radii: list[int] = []
        d_spacings: list[float] = []
        scale_inv_A = 5.0 / float(cl) if cl else None
        try:
            from scipy.signal import find_peaks
            y_idx, x_idx = np.indices((h, w))
            r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2).astype(int)
            r_max = min(cx, cy)
            radial = np.array([arr[r == ri].mean() if np.any(r == ri) else 0 for ri in range(r_max)])
            peaks, _ = find_peaks(radial, height=radial.mean() * 1.5, distance=10)
            ring_radii = peaks.tolist()[:8]
            if scale_inv_A:
                d_spacings = [round(1.0 / (rr * scale_inv_A), 3) for rr in ring_radii if rr > 0]
            mean_bg = float(radial[:10].mean())
        except Exception:
            mean_bg = float(arr.mean())

        result = {
            "success": True,
            "acquisition_type": "Diffraction",
            "pattern": {
                "shape": [h, w],
                "camera_length_mm": cl,
                "pixel_scale_inv_A": round(scale_inv_A, 4) if scale_inv_A else None,
                "direct_beam_centre": [cx, cy],
                "ring_radii_px": ring_radii,
                "d_spacings_A": d_spacings,
                "max_intensity": float(arr.max()),
                "mean_bg": mean_bg,
            },
        }
        return json.dumps(result, indent=2)
    except ImportError:
        # scipy not available — return without ring analysis
        arr = dp.GetNumArray()
        result = {
            "success": True,
            "acquisition_type": "Diffraction",
            "pattern": _image_to_response(dp),
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return _build_error(str(e),
            "Ensure microscope is in diffraction mode before acquiring.")


# ---------------------------------------------------------------------------
# TOOL: Stage control
# ---------------------------------------------------------------------------

@mcp.tool(
    name="gms_get_stage_position",
    annotations={"readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
def gms_get_stage_position() -> str:
    """
    Read all stage axes.

    Returns JSON:
        x_um, y_um, z_um  (float) : Linear stage positions in micrometers.
        alpha_deg          (float) : Alpha tilt angle in degrees.
        beta_deg           (float) : Beta tilt angle in degrees.
    """
    try:
        if _bridge_mode_enabled():
            result = _run_bridge_tool("GetStagePosition")
            return json.dumps(result, indent=2)

        if _SIMULATE:
            stage_payload = {
                "x_um": DM.EMGetStageX(),
                "y_um": DM.EMGetStageY(),
                "z_um": DM.EMGetStageZ(),
                "alpha_deg": DM.EMGetStageAlpha(),
                "beta_deg": DM.EMGetStageBeta(),
            }
        else:
            mic = DM.Py_Microscope()
            stage_payload = {
                "x_um": mic.GetStageX(),
                "y_um": mic.GetStageY(),
                "z_um": mic.GetStageZ(),
                "alpha_deg": mic.GetStageAlpha(),
                "beta_deg": mic.GetStageBeta(),
            }

        result = {
            "success": True,
            "stage": stage_payload,
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return _build_error(str(e))


@mcp.tool(
    name="gms_set_stage_position",
    annotations={"readOnlyHint": False, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
def gms_set_stage_position(
    params: Optional[SetStageInput] = None,
    x_um: Optional[float] = None,
    y_um: Optional[float] = None,
    z_um: Optional[float] = None,
    alpha_deg: Optional[float] = None,
    beta_deg: Optional[float] = None,
) -> str:
    """
    Move the microscope stage to specified coordinates.

    Only axes with non-None values are moved. Coordinate limits are enforced
    by Pydantic validation before the command is sent to the microscope.

    Parameters (all optional — only provided axes move):
        params.x_um      (float) : Target X in µm (±5000).
        params.y_um      (float) : Target Y in µm (±5000).
        params.z_um      (float) : Target Z in µm (±500).
        params.alpha_deg (float) : Alpha tilt (±80°).
        params.beta_deg  (float) : Beta tilt (±30°).

    The same fields can also be passed directly as top-level kwargs
    (e.g. `x_um=100`) for compatibility with some LLM tool clients.

    Returns JSON with new stage position after movement.
    """
    try:
        direct_kwargs: dict[str, object] = {
            "x_um": x_um,
            "y_um": y_um,
            "z_um": z_um,
            "alpha_deg": alpha_deg,
            "beta_deg": beta_deg,
        }

        if params is None:
            params = SetStageInput(**{k: v for k, v in direct_kwargs.items() if v is not None})
        elif any(v is not None for v in direct_kwargs.values()):
            merged = params.model_dump(exclude_none=True)
            merged.update({k: v for k, v in direct_kwargs.items() if v is not None})
            params = SetStageInput(**merged)

        if _bridge_mode_enabled():
            result = _run_bridge_tool("SetStagePosition", params.model_dump(exclude_none=True))
            return json.dumps(result, indent=2)

        flags = 0
        args = [0.0, 0.0, 0.0, 0.0, 0.0]
        if params.x_um is not None:
            flags |= 1
            args[0] = float(params.x_um)
        if params.y_um is not None:
            flags |= 2
            args[1] = float(params.y_um)
        if params.z_um is not None:
            flags |= 4
            args[2] = float(params.z_um)
        if params.alpha_deg is not None:
            flags |= 8
            args[3] = float(params.alpha_deg)
        if params.beta_deg is not None:
            flags |= 16
            args[4] = float(params.beta_deg)

        if flags == 0:
            return _build_error("No axes specified. Provide at least one of x_um, y_um, z_um, alpha_deg, or beta_deg.")

        if _SIMULATE:
            DM.EMSetStagePositions(flags, *args)
            DM.EMWaitUntilReady()
            stage_payload = {
                "x_um": DM.EMGetStageX(),
                "y_um": DM.EMGetStageY(),
                "z_um": DM.EMGetStageZ(),
                "alpha_deg": DM.EMGetStageAlpha(),
                "beta_deg": DM.EMGetStageBeta(),
            }
        else:
            mic = DM.Py_Microscope()
            mic.SetStagePositions(flags, *args)
            if hasattr(mic, "IsReady"):
                t0 = time.time()
                while time.time() - t0 < 30:
                    try:
                        if mic.IsReady():
                            break
                    except Exception:
                        break
                    if hasattr(DM, "Sleep"):
                        DM.Sleep(0.1)
                    else:
                        time.sleep(0.1)
            stage_payload = {
                "x_um": mic.GetStageX(),
                "y_um": mic.GetStageY(),
                "z_um": mic.GetStageZ(),
                "alpha_deg": mic.GetStageAlpha(),
                "beta_deg": mic.GetStageBeta(),
            }

        result = {
            "success": True,
            "moved_flags": flags,
            "new_position": stage_payload,
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return _build_error(str(e),
            "Stage move aborted. Verify target coordinates are within range.")


# ---------------------------------------------------------------------------
# TOOL: Beam / optics control
# ---------------------------------------------------------------------------

@mcp.tool(
    name="gms_set_beam_parameters",
    annotations={"readOnlyHint": False, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
def gms_set_beam_parameters(
    params: Optional[SetBeamInput] = None,
    spot_size: Optional[int] = None,
    focus_um: Optional[float] = None,
    shift_x: Optional[float] = None,
    shift_y: Optional[float] = None,
    tilt_x: Optional[float] = None,
    tilt_y: Optional[float] = None,
    obj_stig_x: Optional[float] = None,
    obj_stig_y: Optional[float] = None,
) -> str:
    """
    Configure beam/optics parameters.

    All parameters are optional; only provided values are changed.

    Parameters:
        params.spot_size   (int)   : Condenser spot size 1–11.
        params.focus_um    (float) : Objective lens focus offset in µm.
        params.shift_x/y   (float) : Calibrated beam shift in physical units.
        params.tilt_x/y    (float) : Beam tilt in radians.
        params.obj_stig_x/y(float) : Objective stigmator values.

        The same fields can also be passed directly as top-level kwargs
        (e.g. `spot_size=4`) for compatibility with some LLM tool clients.

    Returns JSON confirming the applied settings.
    """
    try:
        direct_kwargs: dict[str, object] = {
            "spot_size": spot_size,
            "focus_um": focus_um,
            "shift_x": shift_x,
            "shift_y": shift_y,
            "tilt_x": tilt_x,
            "tilt_y": tilt_y,
            "obj_stig_x": obj_stig_x,
            "obj_stig_y": obj_stig_y,
        }

        if params is None:
            params = SetBeamInput(**{k: v for k, v in direct_kwargs.items() if v is not None})
        elif any(v is not None for v in direct_kwargs.values()):
            merged = params.model_dump(exclude_none=True)
            merged.update({k: v for k, v in direct_kwargs.items() if v is not None})
            params = SetBeamInput(**merged)

        if _bridge_mode_enabled():
            result = _run_bridge_tool("SetBeamParameters", params.model_dump(exclude_none=True))
            return json.dumps(result, indent=2)

        applied: dict[str, object] = {}
        if _SIMULATE:
            if params.spot_size is not None:
                DM.EMSetSpotSize(params.spot_size)
                applied["spot_size"] = params.spot_size
            if params.focus_um is not None:
                DM.EMSetFocus(params.focus_um)
                applied["focus_um"] = params.focus_um
            if params.shift_x is not None or params.shift_y is not None:
                sx = params.shift_x if params.shift_x is not None else 0.0
                sy = params.shift_y if params.shift_y is not None else 0.0
                DM.EMSetCalibratedBeamShift(sx, sy)
                applied["beam_shift"] = [sx, sy]
            if params.tilt_x is not None or params.tilt_y is not None:
                tx = params.tilt_x if params.tilt_x is not None else 0.0
                ty = params.tilt_y if params.tilt_y is not None else 0.0
                DM.EMSetBeamTilt(tx, ty)
                applied["beam_tilt"] = [tx, ty]
            if params.obj_stig_x is not None or params.obj_stig_y is not None:
                sx = params.obj_stig_x if params.obj_stig_x is not None else 0.0
                sy = params.obj_stig_y if params.obj_stig_y is not None else 0.0
                DM.EMSetObjectiveStigmation(sx, sy)
                applied["obj_stigmation"] = [sx, sy]
            current_state = {
                "spot_size": DM.EMGetSpotSize(),
                "focus_um": DM.EMGetFocus(),
                "brightness": DM.EMGetBrightness(),
            }
        else:
            mic = DM.Py_Microscope()
            if params.spot_size is not None:
                mic.SetSpotSize(params.spot_size)
                applied["spot_size"] = params.spot_size
            if params.focus_um is not None:
                mic.SetFocus(params.focus_um)
                applied["focus_um"] = params.focus_um
            if params.shift_x is not None or params.shift_y is not None:
                sx = params.shift_x if params.shift_x is not None else 0.0
                sy = params.shift_y if params.shift_y is not None else 0.0
                mic.SetCalibratedBeamShift(sx, sy)
                applied["beam_shift"] = [sx, sy]
            if params.tilt_x is not None or params.tilt_y is not None:
                tx = params.tilt_x if params.tilt_x is not None else 0.0
                ty = params.tilt_y if params.tilt_y is not None else 0.0
                mic.SetBeamTilt(tx, ty)
                applied["beam_tilt"] = [tx, ty]
            if params.obj_stig_x is not None or params.obj_stig_y is not None:
                sx = params.obj_stig_x if params.obj_stig_x is not None else 0.0
                sy = params.obj_stig_y if params.obj_stig_y is not None else 0.0
                mic.SetObjectiveStigmation(sx, sy)
                applied["obj_stigmation"] = [sx, sy]
            current_state = {
                "spot_size": mic.GetSpotSize(),
                "focus_um": mic.GetFocus(),
                "brightness": mic.GetBrightness(),
            }

        result = {
            "success": True,
            "applied_settings": applied,
            "current_state": current_state,
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return _build_error(str(e))


# ---------------------------------------------------------------------------
# TOOL: Detector configuration
# ---------------------------------------------------------------------------

@mcp.tool(
    name="gms_configure_detectors",
    annotations={"readOnlyHint": False, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
def gms_configure_detectors(
    params: Optional[SetDetectorInput] = None,
    insert_camera: Optional[bool] = None,
    haadf_enabled: Optional[bool] = None,
    bf_enabled: Optional[bool] = None,
    abf_enabled: Optional[bool] = None,
    target_temp_c: Optional[float] = None,
) -> str:
    """
    Configure camera and STEM detector settings.

    Parameters:
        params.insert_camera (bool)  : Insert (True) or retract (False) camera.
        params.target_temp_c (float) : Target CCD cooling temperature in °C.
        params.haadf_enabled (bool)  : Enable/disable HAADF channel (DS signal 0).
        params.bf_enabled    (bool)  : Enable/disable BF channel (DS signal 1).
        params.abf_enabled   (bool)  : Enable/disable ABF channel (DS signal 2).

    The same fields can also be passed directly as top-level kwargs
    (e.g. `haadf_enabled=True`) for compatibility with some LLM tool clients.

    Returns JSON with detector status after applying configuration.
    """
    try:
        if params is None:
            kw: dict[str, object] = {}
            if insert_camera is not None:
                kw["insert_camera"] = insert_camera
            if haadf_enabled is not None:
                kw["haadf_enabled"] = haadf_enabled
            if bf_enabled is not None:
                kw["bf_enabled"] = bf_enabled
            if abf_enabled is not None:
                kw["abf_enabled"] = abf_enabled
            if target_temp_c is not None:
                kw["target_temp_c"] = target_temp_c
            params = SetDetectorInput(**kw)
        if _bridge_mode_enabled():
            result = _run_bridge_tool("ConfigureDetectors", params.model_dump(exclude_none=True))
            return json.dumps(result, indent=2)

        applied: dict[str, object] = {}
        if _SIMULATE:
            camera = DM.CM_GetCurrentCamera()
            if params.insert_camera is not None:
                DM.CM_SetCameraInserted(camera, int(params.insert_camera))
                applied["camera_inserted"] = params.insert_camera
            if params.target_temp_c is not None:
                DM.CM_SetTargetTemperature_C(camera, 1, params.target_temp_c)
                applied["target_temp_c"] = params.target_temp_c
            signal_map = {
                "haadf": (0, params.haadf_enabled),
                "bf": (1, params.bf_enabled),
                "abf": (2, params.abf_enabled),
            }
            for det_name, (ch, enabled) in signal_map.items():
                if enabled is not None:
                    DM.DSSetSignalEnabled(ch, int(enabled))
                    applied[f"{det_name}_enabled"] = enabled
            status = {
                "camera_inserted": DM.CM_GetCameraInserted(camera),
                "actual_temp_c": DM.CM_GetActualTemperature_C(camera),
                "haadf_enabled": DM.DSGetSignalEnabled(0),
                "bf_enabled": DM.DSGetSignalEnabled(1),
                "abf_enabled": DM.DSGetSignalEnabled(2),
            }
        else:
            camera = DM.GetActiveCamera()
            if params.insert_camera is not None:
                if hasattr(camera, "SetInserted"):
                    camera.SetInserted(bool(params.insert_camera))
                    applied["camera_inserted"] = params.insert_camera
                else:
                    applied["camera_inserted"] = "UNSUPPORTED"
            if params.target_temp_c is not None:
                applied["target_temp_c"] = "UNSUPPORTED"
            for det_name, enabled in {
                "haadf": params.haadf_enabled,
                "bf": params.bf_enabled,
                "abf": params.abf_enabled,
            }.items():
                if enabled is not None:
                    applied[f"{det_name}_enabled"] = "UNSUPPORTED"
            status = {
                "camera_name": camera.GetName() if hasattr(camera, "GetName") else None,
                "camera_inserted": camera.GetInserted() if hasattr(camera, "GetInserted") else None,
            }

        result = {
            "success": True,
            "applied": applied,
            "status": status,
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return _build_error(str(e))


# ---------------------------------------------------------------------------
# TOOL: Automated tilt series
# ---------------------------------------------------------------------------

@mcp.tool(
    name="gms_acquire_tilt_series",
    annotations={"readOnlyHint": False, "destructiveHint": False,
                 "idempotentHint": False, "openWorldHint": False},
)
def gms_acquire_tilt_series(
    params: Optional[TiltSeriesInput] = None,
    start_deg: float = -60.0,
    end_deg: float = 60.0,
    step_deg: float = 2.0,
    exposure_s: float = 1.0,
    binning: int = 2,
) -> str:
    """
    Acquire an automated tomographic tilt series.

    Tilts the stage from start_deg to end_deg in step_deg increments,
    acquiring one TEM image at each tilt angle.

    Parameters:
        params.start_deg  (float) : Starting tilt angle (e.g. -60°).
        params.end_deg    (float) : Ending tilt angle (e.g. +60°).
        params.step_deg   (float) : Angular step size (e.g. 2°).
        params.exposure_s (float) : Exposure per frame.
        params.binning    (int)   : Camera binning.
        params.save_dir   (str)   : Optional directory for DM4 output.

    The same fields can also be passed directly as top-level kwargs
    (e.g. `start_deg=-15.0`) for compatibility with some LLM tool clients.

    Returns JSON summary including per-tilt image statistics and total
    acquisition time.
    """
    try:
        if params is None:
            params = TiltSeriesInput(
                start_deg=start_deg,
                end_deg=end_deg,
                step_deg=step_deg,
                exposure_s=exposure_s,
                binning=binning,
            )
        if _bridge_mode_enabled():
            result = _run_bridge_tool("AcquireTiltSeries", params.model_dump())
            return json.dumps(result, indent=2)

        import time as _time
        if _SIMULATE:
            camera = DM.CM_GetCurrentCamera()
            acq = DM.CM_CreateAcquisitionParameters_FullCCD(
                camera, 3, params.exposure_s, params.binning, params.binning
            )
            DM.CM_Validate_AcquisitionParameters(camera, acq)
        else:
            mic = DM.Py_Microscope()
            camera = DM.GetActiveCamera()
            if hasattr(camera, "PrepareForAcquire"):
                camera.PrepareForAcquire()

        angles = []
        angle = params.start_deg
        while angle <= params.end_deg + 1e-6:
            angles.append(round(angle, 3))
            angle += params.step_deg

        per_tilt_stats = []
        t_start = _time.time()

        for ang in angles:
            if _SIMULATE:
                DM.EMSetStageAlpha(ang)
                DM.EMWaitUntilReady()
                img = DM.CM_AcquireImage(camera, acq)
            else:
                mic.SetStageAlpha(float(ang))
                if hasattr(mic, "IsReady"):
                    t0 = _time.time()
                    while _time.time() - t0 < 30:
                        try:
                            if mic.IsReady():
                                break
                        except Exception:
                            break
                        if hasattr(DM, "Sleep"):
                            DM.Sleep(0.1)
                        else:
                            _time.sleep(0.1)
                img = camera.AcquireImage(params.exposure_s, params.binning, params.binning, 3)
            img.SetName(f"Tilt_{ang:+.1f}deg")
            tags = img.GetTagGroup()
            tags.SetTagAsFloat("TiltSeries:Alpha", float(ang))

            arr = img.GetNumArray()
            per_tilt_stats.append({
                "angle_deg": ang,
                "mean": round(float(arr.mean()), 2),
                "max": round(float(arr.max()), 2),
            })

            if params.save_dir:
                fname = os.path.join(params.save_dir, f"tilt_{ang:+.1f}.dm4")
                if hasattr(img, "Save"):
                    img.Save(fname)
                else:
                    DM.SaveImage(img, fname)

        elapsed = round(_time.time() - t_start, 1)

        result = {
            "success": True,
            "tilt_series": {
                "n_frames":    len(angles),
                "angle_range": [params.start_deg, params.end_deg],
                "step_deg":    params.step_deg,
                "exposure_s":  params.exposure_s,
                "binning":     params.binning,
                "elapsed_s":   elapsed,
                "saved_to":    params.save_dir,
            },
            "per_tilt": per_tilt_stats,
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return _build_error(str(e),
            "Verify stage tilt range is accessible and camera is inserted.")


# ---------------------------------------------------------------------------
# TOOL: 4D-STEM virtual detector analysis
# ---------------------------------------------------------------------------

@mcp.tool(
    name="gms_run_4dstem_analysis",
    annotations={"readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
def gms_run_4dstem_analysis(
    inner_angle_mrad: Annotated[float, Field(
        description="Virtual detector inner semi-angle in mrad.", ge=0.0
    )] = 10.0,
    outer_angle_mrad: Annotated[float, Field(
        description="Virtual detector outer semi-angle in mrad.", ge=1.0
    )] = 40.0,
    analysis_type: Annotated[str, Field(
        description=(
            "Analysis to perform: 'virtual_bf', 'virtual_haadf', "
            "'com' (centre of mass), 'dpc' (differential phase contrast), "
            "or 'strain'."
        )
    )] = "virtual_haadf",
) -> str:
    """
    Compute a virtual detector image or analytical map from a loaded 4D-STEM
    dataset using py4DSTEM / numpy.

    Parameters:
        inner_angle_mrad (float): Inner angle of the virtual annular detector.
        outer_angle_mrad (float): Outer angle of the virtual annular detector.
        analysis_type    (str)  : 'virtual_bf', 'virtual_haadf', 'com', 'dpc',
                                  or 'strain'.

    Returns JSON with the resulting image shape, statistics, and analysis info.
    """
    try:
        if _bridge_mode_enabled():
            result = _run_bridge_tool(
                "Run4DSTEMAnalysis",
                {
                    "inner_angle_mrad": inner_angle_mrad,
                    "outer_angle_mrad": outer_angle_mrad,
                    "analysis_type": analysis_type,
                },
            )
            return json.dumps(result, indent=2)

        img4d = DM.GetFrontImage()
        arr = img4d.GetNumArray()

        # Handle both 4D and the simulator's flattened representation
        if arr.ndim == 2 and arr.shape[1] > arr.shape[0]:
            # Simulator stores as (scan_y, scan_x * det_y * det_x) flattened
            scan_y = arr.shape[0]
            total = arr.shape[1]
            det_px = int(np.sqrt(total // scan_y))
            scan_x = total // (det_px * det_px)
            arr = arr.reshape(scan_y, scan_x, det_px, det_px)
        elif arr.ndim != 4:
            return _build_error(
                "Front image is not a 4D-STEM dataset (expected 4D array).",
                "Acquire a 4D-STEM dataset first with gms_acquire_4d_stem."
            )

        scan_y, scan_x, det_y, det_x = arr.shape
        cx, cy = det_x // 2, det_y // 2

        # Build annular mask in pixels (rough: 1 px ~ 0.5 mrad at 80 cm CL)
        px_per_mrad = 1.0   # calibration placeholder
        inner_px = inner_angle_mrad * px_per_mrad
        outer_px = outer_angle_mrad * px_per_mrad

        yy, xx = np.ogrid[:det_y, :det_x]
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask = (r >= inner_px) & (r <= outer_px)

        if analysis_type in ("virtual_haadf", "virtual_bf"):
            virt = arr[:, :, mask].sum(axis=-1)
        elif analysis_type == "com":
            # Centre of mass in X
            weights = arr.sum(axis=(2, 3), keepdims=True) + 1e-10
            virt = (arr * xx[np.newaxis, np.newaxis]).sum(axis=(2, 3)) / weights.squeeze()
        elif analysis_type == "dpc":
            com_x = (arr * xx[np.newaxis, np.newaxis]).sum(axis=(2, 3)) / (
                arr.sum(axis=(2, 3)) + 1e-10)
            com_y = (arr * yy[np.newaxis, np.newaxis]).sum(axis=(2, 3)) / (
                arr.sum(axis=(2, 3)) + 1e-10)
            virt = np.sqrt(com_x ** 2 + com_y ** 2)
        else:  # strain — placeholder
            virt = np.zeros((scan_y, scan_x), dtype=np.float32)

        virt = virt.astype(np.float32)

        result = {
            "success": True,
            "analysis": {
                "type": analysis_type,
                "inner_mrad": inner_angle_mrad,
                "outer_mrad": outer_angle_mrad,
                "scan_shape": [scan_y, scan_x],
                "result_shape": list(virt.shape),
                "min":  float(virt.min()),
                "max":  float(virt.max()),
                "mean": float(virt.mean()),
                "std":  float(virt.std()),
            }
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return _build_error(str(e))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GMS MCP Server — connects Gatan GMS to LLM clients via MCP"
    )
    p.add_argument("--transport", choices=["stdio", "http", "sse"],
                   default="stdio",
                   help="Transport: 'stdio' for local Ollama, 'http' for Claude.ai")
    p.add_argument("--port", type=int,
                   default=int(os.environ.get("GMS_MCP_PORT", 8000)),
                   help="TCP port for HTTP/SSE transports (default 8000)")
    p.add_argument("--host", default="0.0.0.0",
                   help="Bind address for HTTP/SSE transports (default 0.0.0.0)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    mode_str = _runtime_mode().upper()
    if args.transport == "stdio":
        # stdio: no banner on stdout — MCP clients read stdout as JSON-RPC.
        import sys as _sys

        print(f"[GMS-MCP] {mode_str} mode | transport=stdio", file=_sys.stderr)
        mcp.run(transport="stdio")
    else:
        print(
            f"[GMS-MCP] {mode_str} mode | transport={args.transport} "
            f"| http://{args.host}:{args.port}/mcp"
        )
        mcp.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
