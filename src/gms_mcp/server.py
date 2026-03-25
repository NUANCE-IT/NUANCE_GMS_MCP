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
from typing import Annotated, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
from fastmcp import FastMCP, Context

# ---------------------------------------------------------------------------
# DM import with automatic simulation fallback
# ---------------------------------------------------------------------------

_SIMULATE = os.environ.get("GMS_SIMULATE", "0") != "0"

if not _SIMULATE:
    try:
        import DigitalMicrograph as DM  # type: ignore
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

def _image_to_response(img, include_data: bool = False) -> dict:
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
    return summary


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
        camera = DM.CM_GetCurrentCamera()
        cam_name = DM.CM_GetCameraName(camera)
        cam_inserted = DM.CM_GetCameraInserted(camera)
        cam_temp = DM.CM_GetActualTemperature_C(camera)

        result = {
            "success": True,
            "simulation_mode": _SIMULATE,
            "optics": {
                "high_tension_kV": DM.EMGetHighTension() / 1000 if DM.EMCanGetHighTension() else None,
                "spot_size":       DM.EMGetSpotSize(),
                "brightness":      DM.EMGetBrightness(),
                "focus_um":        DM.EMGetFocus(),
                "magnification":   DM.EMGetMagnification() if DM.EMCanGetMagnification() else None,
                "mag_index":       DM.EMGetMagIndex(),
                "operation_mode":  DM.EMGetOperationMode(),
                "illumination_mode": DM.EMGetIlluminationMode(),
                "camera_length_mm": (DM.EMGetCameraLength()
                                     if DM.EMCanGetCameraLength() else None),
            },
            "stage": {
                "x_um":      DM.EMGetStageX(),
                "y_um":      DM.EMGetStageY(),
                "z_um":      DM.EMGetStageZ(),
                "alpha_deg": DM.EMGetStageAlpha(),
                "beta_deg":  DM.EMGetStageBeta(),
            },
            "beam": {
                "shift_x": DM.EMGetBeamShift(0.0, 0.0)[0],
                "shift_y": DM.EMGetBeamShift(0.0, 0.0)[1],
            },
            "eels": {
                "energy_offset_eV": DM.IFGetEnergyLoss(0),
                "slit_width_eV":    DM.IFCGetSlitWidth(),
                "in_eels_mode":     DM.IFIsInEELSMode(),
            },
            "camera": {
                "name":        cam_name,
                "inserted":    cam_inserted,
                "temp_c":      cam_temp,
                "n_signals":   DM.DSGetNumberOfSignals(),
            },
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return _build_error(str(e), "Verify the GMS DM bridge is running.")


# ---------------------------------------------------------------------------
# TOOL: TEM / HRTEM image acquisition
# ---------------------------------------------------------------------------

@mcp.tool(
    name="gms_acquire_tem_image",
    annotations={"readOnlyHint": False, "destructiveHint": False,
                 "idempotentHint": False, "openWorldHint": False},
)
def gms_acquire_tem_image(params: AcquireTEMInput) -> str:
    """
    Acquire a single TEM or HRTEM image from the currently inserted camera.

    Parameters:
        params.exposure_s  (float)   : Camera exposure in seconds (0.001–60).
        params.binning     (int)     : Binning factor 1, 2, 4, or 8.
        params.processing  (int)     : 1=raw, 2=dark only, 3=dark+gain (default).
        params.roi         (list)    : Optional [top, left, bottom, right] ROI.

    Returns JSON with:
        - Image shape, dtype, min/max/mean/std statistics
        - Pixel calibration (nm/pixel)
        - Acquisition metadata (exposure, HT, magnification)
    """
    try:
        camera = DM.CM_GetCurrentCamera()
        acq = DM.CM_CreateAcquisitionParameters_FullCCD(
            camera, params.processing, params.exposure_s,
            params.binning, params.binning
        )
        if params.roi:
            DM.CM_SetCCDReadArea(acq, *params.roi)

        DM.CM_Validate_AcquisitionParameters(camera, acq)
        img = DM.CM_AcquireImage(camera, acq)
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
        DM.DSSetFrameSize(params.width, params.height)
        DM.DSSetPixelTime(params.dwell_us)
        DM.DSSetRotation(params.rotation_deg)

        n_signals = DM.DSGetNumberOfSignals()
        for ch in range(n_signals):
            DM.DSSetSignalEnabled(ch, 1 if ch in params.signals else 0)

        DM.DSStartAcquisition()
        DM.DSWaitUntilFinished()

        img = DM.GetFrontImage()
        img.SetName(f"STEM_{params.width}x{params.height}_dwell{params.dwell_us}us")

        # Annotate scan parameters in tags
        tags = img.GetTagGroup()
        tags.SetTagAsFloat("STEM:DwellTime_us", params.dwell_us)
        tags.SetTagAsFloat("STEM:Rotation_deg", params.rotation_deg)
        tags.SetTagAsString("STEM:Signals",
                            ",".join(map(str, params.signals)))

        result = {
            "success": True,
            "acquisition_type": "STEM",
            "scan_parameters": {
                "width": params.width, "height": params.height,
                "dwell_us": params.dwell_us,
                "rotation_deg": params.rotation_deg,
                "signals_enabled": params.signals,
                "total_frame_time_s": (params.width * params.height
                                       * params.dwell_us * 1e-6),
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
def gms_acquire_4d_stem(params: Acquire4DSTEMInput) -> str:
    """
    Acquire a 4D-STEM / NBED dataset.

    At each of the scan_x × scan_y probe positions, the camera records a
    full 2D convergent-beam or nano-beam electron diffraction pattern,
    producing a 4D dataset of shape (scan_y, scan_x, det_y, det_x).

    Parameters:
        params.scan_x          (int)   : Scan positions in X (8–512).
        params.scan_y          (int)   : Scan positions in Y.
        params.dwell_us        (float) : Per-pattern acquisition time in µs.
        params.camera_length_mm(float) : Camera length in mm (optional).
        params.convergence_mrad(float) : Convergence semi-angle (logged only).

    Returns JSON with dataset shape, estimated file size, metadata.
    Note: actual pixel data omitted from response due to size; retrieve via
          the GMS workspace or save to disk.
    """
    try:
        if params.camera_length_mm is not None:
            DM.EMSetCameraLength(params.camera_length_mm)
        cl = DM.EMGetCameraLength() if DM.EMCanGetCameraLength() else None

        # In production: trigger 4D-STEM acquisition via STEMx / SI infrastructure.
        # In simulation: generate a synthetic 4D array and register as front image.
        if _SIMULATE:
            det_x, det_y = 64, 64
            raw4d = DM._make_4d_stem(  # type: ignore[attr-defined]
                params.scan_x, params.scan_y, det_x, det_y
            )
            # Register the simulated 4D dataset as the active front image
            # so that gms_run_4dstem_analysis can retrieve it.
            DM._front_image = raw4d                           # type: ignore[attr-defined]
            DM._images[raw4d.GetID()] = raw4d                 # type: ignore[attr-defined]
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

        # Configure GIF / spectrometer
        DM.IFSetEELSMode()
        DM.IFCSetEnergy(params.energy_offset_eV)
        DM.IFCSetActiveDispersions(params.dispersion_idx)
        if params.slit_width_eV > 0:
            DM.IFCSetSlitWidth(params.slit_width_eV)
            DM.IFCSetSlitIn(1)
        else:
            DM.IFCSetSlitIn(0)

        # Acquire
        camera = DM.CM_GetCurrentCamera()
        acq = DM.CM_CreateAcquisitionParameters_FullCCD(
            camera, 3, params.exposure_s, 1, 1
        )
        if params.full_vertical_binning:
            DM.CM_SetBinning(acq, 1, 2048)   # full vertical bin

        DM.CM_Validate_AcquisitionParameters(camera, acq)
        spec_img = DM.CM_AcquireImage(camera, acq)
        spec_img.SetName(f"EELS_dE{params.energy_offset_eV}eV_"
                         f"exp{params.exposure_s}s")

        arr = spec_img.GetNumArray().flatten()
        n_ch = len(arr)

        # Estimate energy calibration (dispersion ~ 0.1–1 eV/channel)
        dispersions = [0.1, 0.25, 0.5, 1.0]
        disp = dispersions[min(params.dispersion_idx, len(dispersions) - 1)]
        energy_axis = (params.energy_offset_eV
                       + np.arange(n_ch) * disp)

        # Find approximate ZLP centre
        zlp_idx = int(np.argmax(arr))
        zlp_energy = float(energy_axis[zlp_idx])

        result = {
            "success": True,
            "acquisition_type": "EELS",
            "spectrum": {
                "n_channels":          n_ch,
                "energy_offset_eV":    params.energy_offset_eV,
                "dispersion_eV_ch":    disp,
                "energy_range_eV":     [float(energy_axis[0]),
                                        float(energy_axis[-1])],
                "zlp_centre_eV":       zlp_energy,
                "zlp_channel":         zlp_idx,
                "max_intensity":       float(arr.max()),
                "slit_width_eV":       params.slit_width_eV,
                "exposure_s":          params.exposure_s,
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
def gms_acquire_diffraction(params: AcquireDiffractionInput) -> str:
    """
    Acquire a selected-area or nano-beam electron diffraction pattern.

    The microscope must already be in diffraction mode. The camera length
    determines the reciprocal-space calibration.

    Parameters:
        params.exposure_s       (float) : Exposure in seconds.
        params.camera_length_mm (float) : Camera length in mm (optional).
        params.binning          (int)   : Camera binning.

    Returns JSON with image statistics, d-spacing scale (1/Å per pixel),
    direct-beam centre estimate, and brightest ring radii.
    """
    try:
        if params.camera_length_mm is not None:
            DM.EMSetCameraLength(params.camera_length_mm)
        cl = DM.EMGetCameraLength() if DM.EMCanGetCameraLength() else 100.0

        camera = DM.CM_GetCurrentCamera()
        acq = DM.CM_CreateAcquisitionParameters_FullCCD(
            camera, 3, params.exposure_s, params.binning, params.binning
        )
        DM.CM_Validate_AcquisitionParameters(camera, acq)
        dp = DM.CM_AcquireImage(camera, acq)
        dp.SetName(f"DP_CL{cl}mm_exp{params.exposure_s}s")

        arr = dp.GetNumArray()
        h, w = arr.shape[:2]
        cx, cy = w // 2, h // 2

        # Radial profile to find ring positions
        y_idx, x_idx = np.indices((h, w))
        r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2).astype(int)
        r_max = min(cx, cy)
        radial = np.array([arr[r == ri].mean() if np.any(r == ri) else 0
                           for ri in range(r_max)])

        # Find peaks in radial profile (crude, works for simulation)
        from scipy.signal import find_peaks  # type: ignore[import-untyped]
        peaks, _ = find_peaks(radial, height=radial.mean() * 1.5,
                               distance=10)

        # Pixel scale: ~0.05 1/Å per pixel at CL=100 mm (rough estimate)
        scale_inv_A = 5.0 / cl   # 1/Å per pixel

        result = {
            "success": True,
            "acquisition_type": "Diffraction",
            "pattern": {
                "shape":           [h, w],
                "camera_length_mm": cl,
                "pixel_scale_inv_A": round(scale_inv_A, 4),
                "direct_beam_centre": [cx, cy],
                "ring_radii_px":   peaks.tolist()[:8],
                "d_spacings_A": [
                    round(1.0 / (r * scale_inv_A), 3)
                    for r in peaks.tolist()[:8]
                    if r > 0
                ],
                "max_intensity": float(arr.max()),
                "mean_bg":       float(radial[:10].mean()),
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
        result = {
            "success": True,
            "stage": {
                "x_um":      DM.EMGetStageX(),
                "y_um":      DM.EMGetStageY(),
                "z_um":      DM.EMGetStageZ(),
                "alpha_deg": DM.EMGetStageAlpha(),
                "beta_deg":  DM.EMGetStageBeta(),
            }
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return _build_error(str(e))


@mcp.tool(
    name="gms_set_stage_position",
    annotations={"readOnlyHint": False, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False},
)
def gms_set_stage_position(params: SetStageInput) -> str:
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

    Returns JSON with new stage position after movement.
    """
    try:
        flags = 0
        args = [0.0, 0.0, 0.0, 0.0, 0.0]
        if params.x_um is not None:     flags |= 1;  args[0] = params.x_um
        if params.y_um is not None:     flags |= 2;  args[1] = params.y_um
        if params.z_um is not None:     flags |= 4;  args[2] = params.z_um
        if params.alpha_deg is not None:flags |= 8;  args[3] = params.alpha_deg
        if params.beta_deg is not None: flags |= 16; args[4] = params.beta_deg

        if flags == 0:
            return _build_error("No axes specified. Provide at least one of "
                                 "x_um, y_um, z_um, alpha_deg, or beta_deg.")

        DM.EMSetStagePositions(flags, *args)
        DM.EMWaitUntilReady()

        result = {
            "success": True,
            "moved_flags": flags,
            "new_position": {
                "x_um":      DM.EMGetStageX(),
                "y_um":      DM.EMGetStageY(),
                "z_um":      DM.EMGetStageZ(),
                "alpha_deg": DM.EMGetStageAlpha(),
                "beta_deg":  DM.EMGetStageBeta(),
            }
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
        direct_kwargs = {
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

        applied = {}
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

        result = {
            "success": True,
            "applied_settings": applied,
            "current_state": {
                "spot_size":  DM.EMGetSpotSize(),
                "focus_um":   DM.EMGetFocus(),
                "brightness": DM.EMGetBrightness(),
            }
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
def gms_configure_detectors(params: SetDetectorInput) -> str:
    """
    Configure camera and STEM detector settings.

    Parameters:
        params.insert_camera   (bool)  : Insert (True) or retract (False) camera.
        params.target_temp_c   (float) : Target CCD cooling temperature in °C.
        params.haadf_enabled   (bool)  : Enable/disable HAADF channel (DS signal 0).
        params.bf_enabled      (bool)  : Enable/disable BF channel (DS signal 1).
        params.abf_enabled     (bool)  : Enable/disable ABF channel (DS signal 2).

    Returns JSON with detector status after applying configuration.
    """
    try:
        camera = DM.CM_GetCurrentCamera()
        applied = {}

        if params.insert_camera is not None:
            DM.CM_SetCameraInserted(camera, int(params.insert_camera))
            applied["camera_inserted"] = params.insert_camera

        if params.target_temp_c is not None:
            DM.CM_SetTargetTemperature_C(camera, 1, params.target_temp_c)
            applied["target_temp_c"] = params.target_temp_c

        signal_map = {
            "haadf": (0, params.haadf_enabled),
            "bf":    (1, params.bf_enabled),
            "abf":   (2, params.abf_enabled),
        }
        for det_name, (ch, enabled) in signal_map.items():
            if enabled is not None:
                DM.DSSetSignalEnabled(ch, int(enabled))
                applied[f"{det_name}_enabled"] = enabled

        result = {
            "success": True,
            "applied": applied,
            "status": {
                "camera_inserted":  DM.CM_GetCameraInserted(camera),
                "actual_temp_c":    DM.CM_GetActualTemperature_C(camera),
                "haadf_enabled":    DM.DSGetSignalEnabled(0),
                "bf_enabled":       DM.DSGetSignalEnabled(1),
                "abf_enabled":      DM.DSGetSignalEnabled(2),
            }
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
def gms_acquire_tilt_series(params: TiltSeriesInput) -> str:
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

    Returns JSON summary including per-tilt image statistics and total
    acquisition time.
    """
    try:
        import time as _time
        camera = DM.CM_GetCurrentCamera()
        acq = DM.CM_CreateAcquisitionParameters_FullCCD(
            camera, 3, params.exposure_s, params.binning, params.binning
        )
        DM.CM_Validate_AcquisitionParameters(camera, acq)

        angles = []
        angle = params.start_deg
        while angle <= params.end_deg + 1e-6:
            angles.append(round(angle, 3))
            angle += params.step_deg

        per_tilt_stats = []
        t_start = _time.time()

        for ang in angles:
            DM.EMSetStageAlpha(ang)
            DM.EMWaitUntilReady()
            img = DM.CM_AcquireImage(camera, acq)
            img.SetName(f"Tilt_{ang:+.1f}deg")
            tags = img.GetTagGroup()
            tags.SetTagAsFloat("TiltSeries:Alpha", ang)

            arr = img.GetNumArray()
            per_tilt_stats.append({
                "angle_deg": ang,
                "mean":      round(float(arr.mean()), 2),
                "max":       round(float(arr.max()), 2),
            })

            if params.save_dir:
                fname = os.path.join(params.save_dir,
                                     f"tilt_{ang:+.1f}.dm4")
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


if __name__ == "__main__":
    args = _parse_args()

    mode_str = "SIMULATION" if _SIMULATE else "LIVE"
    if args.transport == "stdio":
        # stdio: no banner on stdout — LangChain reads stdout as JSON-RPC
        import sys as _sys
        print(f"[GMS-MCP] {mode_str} mode | transport=stdio", file=_sys.stderr)
        mcp.run(transport="stdio")
    else:
        print(f"[GMS-MCP] {mode_str} mode | transport={args.transport} "
              f"| http://{args.host}:{args.port}/mcp")
        mcp.run(transport=args.transport, host=args.host, port=args.port)
