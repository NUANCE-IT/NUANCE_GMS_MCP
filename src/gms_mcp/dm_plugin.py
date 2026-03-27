"""
dm_plugin.py
=============
ZeroMQ bridge plugin that runs **inside** the Gatan Microscopy Suite (GMS)
Python environment, exposing the DigitalMicrograph (DM) Python API to the
GMS-MCP FastMCP server running as a separate process.

Usage (inside the GMS Python console or as a background script)
---------------------------------------------------------------
    exec(open("dm_plugin.py").read())

    # Or import as a module if the path is on sys.path:
    from gms_mcp.dm_plugin import start_bridge, stop_bridge
    start_bridge()   # starts listening in a background thread
    # ... do microscopy work ...
    stop_bridge()    # clean shutdown

Network
-------
    Binds to tcp://0.0.0.0:5555 by default.
    Configurable via environment variable GMS_MCP_ZMQ_PORT.

Security note
-------------
    The ZeroMQ socket is bound to all interfaces (0.0.0.0).
    In a production facility, restrict to the instrument LAN:
        GMS_MCP_ZMQ_BIND=tcp://192.168.1.x:5555
    and configure the firewall to block external access.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from typing import Any

try:
    import DigitalMicrograph as DM
except ImportError:
    raise RuntimeError(
        "dm_plugin.py must run inside GMS. "
        "Import DigitalMicrograph failed."
    )

try:
    import zmq
except ImportError:
    raise RuntimeError(
        "pyzmq is required. Install it inside the GMS environment:\n"
        "  pip install pyzmq --break-system-packages"
    )

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_PORT = 5555
_DEFAULT_BIND = f"tcp://0.0.0.0:{_DEFAULT_PORT}"

ZMQ_BIND = os.environ.get("GMS_MCP_ZMQ_BIND", _DEFAULT_BIND)

# ---------------------------------------------------------------------------
# Command dispatcher
# ---------------------------------------------------------------------------

def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy types to Python-native JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    return obj


def _image_to_dict(img, include_data: bool = False) -> dict:
    """Serialize a DM Py_Image to a JSON-safe dict."""
    arr = img.GetNumArray()
    tags = img.GetTagGroup()
    ok_exp, exp = tags.GetTagAsFloat("Acquisition:ExposureTime")
    ok_ht, ht   = tags.GetTagAsFloat("Microscope:HighTension_kV")
    ok_mag, mag = tags.GetTagAsFloat("Microscope:Magnification")
    origin, scale, unit = img.GetDimensionCalibration(0, 0)

    result = {
        "name":  img.GetName(),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "statistics": {
            "min":  float(arr.min()),
            "max":  float(arr.max()),
            "mean": float(arr.mean()),
            "std":  float(arr.std()),
        },
        "calibration": {
            "origin":     float(origin),
            "scale":      float(scale),
            "unit":       unit if isinstance(unit, str) else unit.decode("utf-8", errors="replace"),
        },
        "metadata": {
            "exposure_s":      exp  if ok_exp  else None,
            "high_tension_kV": ht   if ok_ht   else None,
            "magnification":   mag  if ok_mag  else None,
        },
    }
    if include_data:
        result["data_b64"] = base64.b64encode(arr.tobytes()).decode()
        result["data_shape"] = list(arr.shape)
        result["data_dtype"] = str(arr.dtype)
    return result


_live_jobs: dict[str, dict[str, Any]] = {}
_live_jobs_lock = threading.Lock()


def _extract_roi(data: np.ndarray, roi: list[int] | None) -> np.ndarray:
    if roi is None:
        return np.asarray(data)
    if len(roi) != 4:
        raise ValueError("roi must have exactly 4 elements: [top, left, bottom, right]")
    top, left, bottom, right = [int(v) for v in roi]
    if top < 0 or left < 0 or bottom <= top or right <= left:
        raise ValueError("roi must define a positive [top, left, bottom, right] region")
    return np.asarray(data)[top:bottom, left:right]


def _bin_image(data: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return np.asarray(data)
    h, w = data.shape
    h_trim = h - (h % factor)
    w_trim = w - (w % factor)
    if h_trim <= 0 or w_trim <= 0:
        raise ValueError("Binning factor is too large for the selected image region.")
    trimmed = np.asarray(data[:h_trim, :w_trim], dtype=np.float32)
    return trimmed.reshape(h_trim // factor, factor, w_trim // factor, factor).mean(axis=(1, 3))


def _resolve_4dstem_array(img) -> np.ndarray:
    arr = img.GetNumArray()
    if arr.ndim == 2 and arr.shape[1] > arr.shape[0]:
        scan_y = arr.shape[0]
        total = arr.shape[1]
        det_px = int(np.sqrt(total // scan_y))
        scan_x = total // (det_px * det_px)
        return arr.reshape(scan_y, scan_x, det_px, det_px)
    if arr.ndim != 4:
        raise ValueError("Front image is not a 4D-STEM dataset (expected 4D array).")
    return arr


def _create_derived_image(data: np.ndarray, name: str, source_img):
    derived = DM.CreateImage(np.asarray(data, dtype=np.float32))
    derived.SetName(name)
    try:
        for dim in range(len(data.shape)):
            origin, scale, unit = source_img.GetDimensionCalibration(dim, 0)
            derived.SetDimensionCalibration(dim, origin, scale, unit)
    except Exception:
        pass
    derived.ShowImage()
    return derived


def _copy_into_result_image(result_img, data: np.ndarray) -> None:
    target = result_img.GetNumArray()
    target[...] = np.asarray(data, dtype=target.dtype)
    result_img.UpdateImage()


def _summarize_array(data: np.ndarray) -> dict[str, Any]:
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


def _encode_array_b64(data: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(data)
    return {
        "data_b64": base64.b64encode(arr.tobytes()).decode(),
        "data_shape": list(arr.shape),
        "data_dtype": str(arr.dtype),
    }


def _exponential_moving_average(frame: np.ndarray, previous: np.ndarray, period: int) -> np.ndarray:
    if period <= 1:
        return frame.astype(np.float32)
    persistence = (period - 1) / (period + 1)
    return persistence * previous.astype(np.float32) + (1.0 - persistence) * frame.astype(np.float32)


def _compute_radial_profile_result(data: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
    roi_data = _extract_roi(data, params.get("roi"))
    mode = params.get("profile_mode", "fft")
    if mode == "fft":
        working = np.abs(np.fft.fftshift(np.fft.fft2(roi_data))).astype(np.float32)
        unit = "nm^-1"
    else:
        working = roi_data.astype(np.float32)
        unit = "px"

    working = _bin_image(working, int(params.get("binning", 1)))
    if bool(params.get("mask_center_lines", True)):
        cx = working.shape[1] // 2
        cy = working.shape[0] // 2
        working[:, max(0, cx - 1):cx + 1] = 0.0
        working[max(0, cy - 1):cy + 1, :] = 0.0

    h, w = working.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    radii = np.sqrt((xx - (w / 2.0)) ** 2 + (yy - (h / 2.0)) ** 2)
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

    metric = str(params.get("profile_metric", "radial_max_minus_mean"))
    if metric == "radial_mean":
        profile = radial_mean
    elif metric == "radial_max":
        profile = radial_max
    else:
        profile = radial_max - radial_mean

    smooth_sigma = float(params.get("smooth_sigma", 1.0))
    if smooth_sigma > 0:
        profile = gaussian_filter(profile, sigma=smooth_sigma)

    ignore_bins = int(len(profile) * float(params.get("mask_percent", 5.0)) / 100.0)
    if ignore_bins > 0:
        profile[:ignore_bins] = 0.0

    return {
        "data": profile.astype(np.float32),
        "summary": {
            "mode": mode,
            "profile_metric": metric,
            "profile_length": int(len(profile)),
            "unit": unit,
        },
    }


def _compute_max_fft_result(data: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
    roi_data: np.ndarray = _extract_roi(data, params.get("roi")).astype(np.float32)
    fft_size = int(params.get("fft_size", 256))
    spacing = int(params.get("spacing", 256))
    if roi_data.shape[0] < fft_size or roi_data.shape[1] < fft_size:
        raise ValueError("Selected image region is smaller than fft_size.")

    view = np.lib.stride_tricks.sliding_window_view(roi_data, (fft_size, fft_size))
    windows = view[::spacing, ::spacing]
    if windows.size == 0:
        windows = view[:1, :1]

    hann_1d: np.ndarray = np.hanning(fft_size).astype(np.float32)
    hann_2d = np.sqrt(np.outer(hann_1d, hann_1d)).astype(np.float32)
    spectra = np.abs(np.fft.fftshift(np.fft.fft2(windows * hann_2d, axes=(-2, -1)), axes=(-2, -1)))
    max_fft = spectra.max(axis=(0, 1)).astype(np.float32)
    if bool(params.get("log_scale", True)):
        max_fft = np.log1p(max_fft)

    return {
        "data": max_fft,
        "summary": {
            "fft_size": fft_size,
            "spacing": spacing,
            "log_scale": bool(params.get("log_scale", True)),
            "n_windows": int(windows.shape[0] * windows.shape[1]),
        },
    }


def _compute_difference_result(data: np.ndarray, job: dict[str, Any]) -> dict[str, Any]:
    params = job["params"]
    frame: np.ndarray = _extract_roi(data, params.get("roi")).astype(np.float32)
    gaussian_sigma = float(params.get("gaussian_sigma", 0.0))
    if gaussian_sigma > 0:
        frame = gaussian_filter(frame, sigma=gaussian_sigma)

    avg1 = job.get("avg1")
    avg2 = job.get("avg2")
    if not isinstance(avg1, np.ndarray):
        avg1 = frame.copy()
    if not isinstance(avg2, np.ndarray):
        avg2 = frame.copy()

    avg1 = _exponential_moving_average(frame, avg1, int(params.get("avg_period_1", 5)))
    avg2 = _exponential_moving_average(frame, avg2, int(params.get("avg_period_2", 10)))
    job["avg1"] = avg1
    job["avg2"] = avg2

    return {
        "data": np.abs(avg2 - avg1).astype(np.float32),
        "summary": {
            "avg_period_1": int(params.get("avg_period_1", 5)),
            "avg_period_2": int(params.get("avg_period_2", 10)),
            "gaussian_sigma": gaussian_sigma,
        },
    }


def _compute_filtered_view_result(data: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
    filtered: np.ndarray = _extract_roi(data, params.get("roi")).astype(np.float32)
    median_size = int(params.get("median_size", 0))
    gaussian_sigma = float(params.get("gaussian_sigma", 0.0))
    if median_size > 1:
        filtered = median_filter(filtered, size=median_size)
    if gaussian_sigma > 0:
        filtered = gaussian_filter(filtered, sigma=gaussian_sigma)

    return {
        "data": filtered.astype(np.float32),
        "summary": {
            "median_size": median_size,
            "gaussian_sigma": gaussian_sigma,
        },
    }


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


def _compute_maximum_spot_mapping_result(data4d: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
    if data4d.ndim != 4:
        raise ValueError("Maximum spot mapping requires a 4D-STEM dataset.")

    working = data4d.astype(np.float32).copy()
    scan_y, scan_x, det_y, det_x = working.shape
    cy = (det_y - 1) / 2.0
    cx = (det_x - 1) / 2.0

    if bool(params.get("subtract_mean_background", False)):
        mean_pattern = working.mean(axis=(0, 1), keepdims=True)
        working = np.maximum(working - mean_pattern, 0.0)

    gaussian_sigma = float(params.get("gaussian_sigma", 0.0))
    if gaussian_sigma > 0:
        working = gaussian_filter(working, sigma=(0, 0, gaussian_sigma, gaussian_sigma))

    yy, xx = np.indices((det_y, det_x), dtype=np.float32)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = rr <= float(params.get("mask_center_radius_px", 5.0))
    working[:, :, mask] = -np.inf

    flat = working.reshape(scan_y, scan_x, det_y * det_x)
    max_idx = np.argmax(flat, axis=-1)
    intensity_map = np.take_along_axis(flat, max_idx[..., None], axis=-1).squeeze(-1)
    y_idx, x_idx = np.divmod(max_idx, det_x)
    dx = x_idx.astype(np.float32) - cx
    dy = y_idx.astype(np.float32) - cy
    theta_map = (np.arctan2(dy, dx) + np.pi) / (2.0 * np.pi)
    radius_map = np.sqrt(dx ** 2 + dy ** 2)

    map_var = str(params.get("map_var", "theta"))
    if map_var == "theta":
        hue = theta_map
    elif map_var == "radius":
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
            "map_var": map_var,
            "scan_shape": [scan_y, scan_x],
            "mask_center_radius_px": float(params.get("mask_center_radius_px", 5.0)),
            "subtract_mean_background": bool(params.get("subtract_mean_background", False)),
            "gaussian_sigma": gaussian_sigma,
            "theta_range": [float(theta_map.min()), float(theta_map.max())],
            "radius_range": [float(radius_map.min()), float(radius_map.max())],
            "intensity_range": [float(intensity_map.min()), float(intensity_map.max())],
        },
    }


def _get_live_job(job_id: str) -> dict[str, Any]:
    with _live_jobs_lock:
        job = _live_jobs.get(job_id)
    if job is None:
        raise KeyError(f"Unknown live-processing job: {job_id}")
    return job


def _job_status_payload(job: dict[str, Any]) -> dict[str, Any]:
    latest_result = job.get("latest_result")
    result_summary = None
    if isinstance(latest_result, dict):
        summary = latest_result.get("summary")
        data = latest_result.get("data")
        if isinstance(summary, dict) and isinstance(data, np.ndarray):
            result_summary = dict(summary)
            result_summary.update(_summarize_array(data))

    source_image = job.get("source_image")
    source_name = None
    if source_image is not None:
        try:
            source_name = source_image.GetName()
        except Exception:
            source_name = None

    return {
        "job_id": job["job_id"],
        "job_type": job["job_type"],
        "backend": "bridge",
        "status": job["status"],
        "poll_interval_s": job["poll_interval_s"],
        "iterations": job["iterations"],
        "created_at": job["created_at"],
        "last_updated": job["last_updated"],
        "last_error": job["last_error"],
        "source_image_name": source_name,
        "result_summary": result_summary,
    }


def _run_live_processing_job(job_id: str) -> None:
    job = _get_live_job(job_id)
    stop_event = job["stop_event"]
    params = job["params"]

    while not stop_event.is_set():
        try:
            source_image = job.get("source_image")
            if source_image is None:
                source_image = DM.GetFrontImage()
                job["source_image"] = source_image

            data = np.asarray(source_image.GetNumArray(), dtype=np.float32)
            job_type = str(params.get("job_type", ""))
            if job_type == "radial_profile":
                if data.ndim != 2:
                    raise ValueError("Live processing requires a 2D source image for the selected job type.")
                result = _compute_radial_profile_result(data, params)
                profile = result["data"]
                history = job.get("history")
                history_length = int(params.get("history_length", 200))
                if not isinstance(history, np.ndarray) or history.shape[0] != profile.shape[0]:
                    history = np.zeros((profile.shape[0], history_length), dtype=np.float32)
                    job["history"] = history
                history[:, :-1] = history[:, 1:]
                history[:, -1] = profile
                result["data"] = history.copy()
                result["summary"] = {
                    **result["summary"],
                    "history_length": history_length,
                }
            elif job_type == "difference":
                if data.ndim != 2:
                    raise ValueError("Live processing requires a 2D source image for the selected job type.")
                result = _compute_difference_result(data, job)
            elif job_type == "fft_map":
                if data.ndim != 2:
                    raise ValueError("Live processing requires a 2D source image for the selected job type.")
                result = _compute_max_fft_result(data, params)
            elif job_type == "filtered_view":
                if data.ndim != 2:
                    raise ValueError("Live processing requires a 2D source image for the selected job type.")
                result = _compute_filtered_view_result(data, params)
            elif job_type == "maximum_spot_mapping":
                dataset4d: np.ndarray = _resolve_4dstem_array(source_image).astype(np.float32)
                result = _compute_maximum_spot_mapping_result(dataset4d, params)
            else:
                raise ValueError(f"Unsupported live-processing job type: {job_type}")

            result_data = np.asarray(result["data"], dtype=np.float32)
            job["status"] = "running"
            job["latest_result"] = result
            job["iterations"] = int(job.get("iterations", 0)) + 1
            job["last_updated"] = time.time()
            job["last_error"] = None

            if bool(params.get("show_result", False)):
                result_image = job.get("result_image")
                if result_image is None:
                    result_image = _create_derived_image(
                        result_data,
                        str(params.get("output_name") or f"live_{job_type}_{job_id}"),
                        source_image,
                    )
                    job["result_image"] = result_image
                else:
                    _copy_into_result_image(result_image, result_data)
        except Exception as exc:
            job["status"] = "error"
            job["last_error"] = str(exc)
            job["last_updated"] = time.time()

        stop_event.wait(float(params.get("poll_interval_s", 0.5)))

    if job["status"] != "error":
        job["status"] = "stopped"
    job["last_updated"] = time.time()


def _dispatch(cmd: dict) -> dict:
    """
    Route a JSON command to the appropriate DM API call.

    Every handler must return a JSON-serializable dict with at minimum
    {"success": True/False}.
    """
    func   = cmd.get("function", "")
    params = cmd.get("params", {})

    # ── State queries ─────────────────────────────────────────────────────
    if func == "EM_GetState":
        return {
            "success": True,
            "high_tension_V":   DM.EMGetHighTension()   if DM.EMCanGetHighTension()   else None,
            "magnification":    DM.EMGetMagnification() if DM.EMCanGetMagnification() else None,
            "mag_index":        DM.EMGetMagIndex(),
            "spot_size":        DM.EMGetSpotSize(),
            "brightness":       DM.EMGetBrightness(),
            "focus":            DM.EMGetFocus(),
            "operation_mode":   DM.EMGetOperationMode(),
            "illumination_mode":DM.EMGetIlluminationMode(),
            "camera_length_mm": DM.EMGetCameraLength() if DM.EMCanGetCameraLength() else None,
        }

    # ── Stage ──────────────────────────────────────────────────────────────
    if func == "EMGetStagePositions":
        return {
            "success":   True,
            "x_um":      DM.EMGetStageX(),
            "y_um":      DM.EMGetStageY(),
            "z_um":      DM.EMGetStageZ(),
            "alpha_deg": DM.EMGetStageAlpha(),
            "beta_deg":  DM.EMGetStageBeta(),
        }

    if func == "EMSetStagePositions":
        flags = int(params.get("flags", 0))
        DM.EMSetStagePositions(
            flags,
            float(params.get("x",     0.0)),
            float(params.get("y",     0.0)),
            float(params.get("z",     0.0)),
            float(params.get("alpha", 0.0)),
            float(params.get("beta",  0.0)),
        )
        DM.EMWaitUntilReady()
        return {
            "success":   True,
            "x_um":      DM.EMGetStageX(),
            "y_um":      DM.EMGetStageY(),
            "z_um":      DM.EMGetStageZ(),
            "alpha_deg": DM.EMGetStageAlpha(),
            "beta_deg":  DM.EMGetStageBeta(),
        }

    if func == "EMStopStage":
        DM.EMStopStage()
        return {"success": True}

    # ── Optics ─────────────────────────────────────────────────────────────
    if func == "EMSetSpotSize":
        DM.EMSetSpotSize(int(params["spot_size"]))
        return {"success": True, "spot_size": DM.EMGetSpotSize()}

    if func == "EMSetFocus":
        DM.EMSetFocus(float(params["focus"]))
        return {"success": True, "focus": DM.EMGetFocus()}

    if func == "EMChangeFocus":
        DM.EMChangeFocus(float(params["delta"]))
        return {"success": True, "focus": DM.EMGetFocus()}

    if func == "EMSetCalibratedBeamShift":
        DM.EMSetCalibratedBeamShift(float(params["x"]), float(params["y"]))
        return {"success": True}

    if func == "EMSetBeamTilt":
        DM.EMSetBeamTilt(float(params["x"]), float(params["y"]))
        return {"success": True}

    if func == "EMSetObjectiveStigmation":
        DM.EMSetObjectiveStigmation(float(params["x"]), float(params["y"]))
        return {"success": True}

    if func == "EMSetCameraLength":
        DM.EMSetCameraLength(float(params["camera_length_mm"]))
        return {
            "success": True,
            "camera_length_mm": DM.EMGetCameraLength() if DM.EMCanGetCameraLength() else None,
        }

    # ── Camera / CCD ───────────────────────────────────────────────────────
    if func == "CM_GetCameraInfo":
        cam = DM.CM_GetCurrentCamera()
        return {
            "success":    True,
            "name":       DM.CM_GetCameraName(cam),
            "identifier": DM.CM_GetCameraIdentifier(cam),
            "inserted":   DM.CM_GetCameraInserted(cam),
            "temp_c":     DM.CM_GetActualTemperature_C(cam),
        }

    if func == "CM_SetCameraInserted":
        cam = DM.CM_GetCurrentCamera()
        DM.CM_SetCameraInserted(cam, int(params["inserted"]))
        return {"success": True, "inserted": DM.CM_GetCameraInserted(cam)}

    if func == "CM_SetTargetTemperature":
        cam = DM.CM_GetCurrentCamera()
        DM.CM_SetTargetTemperature_C(cam, 1, float(params["temp_c"]))
        return {"success": True}

    if func == "CM_AcquireImage":
        cam = DM.CM_GetCurrentCamera()
        acq = DM.CM_CreateAcquisitionParameters_FullCCD(
            cam,
            int(params.get("processing", 3)),
            float(params.get("exposure", 1.0)),
            int(params.get("binning", 1)),
            int(params.get("binning", 1)),
        )
        if "roi" in params:
            DM.CM_SetCCDReadArea(acq, *[int(v) for v in params["roi"]])
        DM.CM_Validate_AcquisitionParameters(cam, acq)
        img = DM.CM_AcquireImage(cam, acq)
        img.SetName(params.get("name", "MCP_Acquisition"))
        img.ShowImage()
        return {"success": True, **_image_to_dict(img, params.get("include_data", False))}

    # ── DigiScan / STEM ────────────────────────────────────────────────────
    if func == "DS_Configure":
        DM.DSSetFrameSize(int(params.get("width", 512)), int(params.get("height", 512)))
        DM.DSSetPixelTime(float(params.get("dwell_us", 10.0)))
        DM.DSSetRotation(float(params.get("rotation_deg", 0.0)))
        if "flyback_us" in params:
            DM.DSSetFlybackTime(float(params["flyback_us"]))
        n = DM.DSGetNumberOfSignals()
        enabled = params.get("signals", [0, 1])
        for ch in range(n):
            DM.DSSetSignalEnabled(ch, 1 if ch in enabled else 0)
        return {"success": True}

    if func == "DS_Acquire":
        DM.DSStartAcquisition()
        DM.DSWaitUntilFinished()
        img = DM.GetFrontImage()
        return {"success": True, **_image_to_dict(img, params.get("include_data", False))}

    # ── GIF / EELS ─────────────────────────────────────────────────────────
    if func == "EELS_Configure":
        DM.IFSetEELSMode()
        DM.IFCSetEnergy(float(params.get("energy_offset_eV", 0.0)))
        DM.IFCSetActiveDispersions(int(params.get("dispersion_idx", 0)))
        slit_w = float(params.get("slit_width_eV", 10.0))
        if slit_w > 0:
            DM.IFCSetSlitWidth(slit_w)
            DM.IFCSetSlitIn(1)
        else:
            DM.IFCSetSlitIn(0)
        return {
            "success":          True,
            "energy_loss_eV":   DM.IFGetEnergyLoss(0),
            "slit_width_eV":    DM.IFCGetSlitWidth(),
            "in_eels_mode":     DM.IFIsInEELSMode(),
        }

    if func == "EELS_Acquire":
        cam = DM.CM_GetCurrentCamera()
        acq = DM.CM_CreateAcquisitionParameters_FullCCD(
            cam, 3, float(params.get("exposure", 1.0)), 1, 1
        )
        if params.get("full_vertical_binning", True):
            DM.CM_SetBinning(acq, 1, 2048)
        DM.CM_Validate_AcquisitionParameters(cam, acq)
        spec = DM.CM_AcquireImage(cam, acq)
        spec.SetName("EELS_Spectrum")
        spec.ShowImage()
        return {"success": True, **_image_to_dict(spec, params.get("include_data", False))}

    if func == "IFSetImageMode":
        DM.IFSetImageMode()
        return {"success": True, "in_image_mode": DM.IFIsInImageMode()}

    # ── Utility ────────────────────────────────────────────────────────────
    if func == "GetFrontImage":
        img = DM.GetFrontImage()
        return {"success": True, **_image_to_dict(img, params.get("include_data", False))}

    if func == "LiveProcessingJobStart":
        job_type = str(params.get("job_type", ""))
        if job_type not in {"radial_profile", "difference", "fft_map", "filtered_view", "maximum_spot_mapping"}:
            return {
                "success": False,
                "error": "job_type must be 'radial_profile', 'difference', 'fft_map', 'filtered_view', or 'maximum_spot_mapping'.",
            }

        source_image = DM.GetFrontImage()
        source_data = np.asarray(source_image.GetNumArray())
        if job_type == "maximum_spot_mapping":
            try:
                _resolve_4dstem_array(source_image)
            except Exception as exc:
                return {"success": False, "error": str(exc)}
        elif source_data.ndim != 2:
            return {
                "success": False,
                "error": "Live-processing jobs currently require a 2D front-most image.",
            }

        job_id = os.urandom(6).hex()
        stop_event = threading.Event()
        job = {
            "job_id": job_id,
            "job_type": job_type,
            "params": dict(params),
            "poll_interval_s": float(params.get("poll_interval_s", 0.5)),
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
            name=f"gms-dm-live-job-{job_id}",
        )
        job["thread"] = thread

        with _live_jobs_lock:
            _live_jobs[job_id] = job

        thread.start()
        return {
            "success": True,
            "job": {
                "job_id": job_id,
                "job_type": job_type,
                "backend": "bridge",
                "status": "starting",
                "poll_interval_s": float(params.get("poll_interval_s", 0.5)),
                "show_result": bool(params.get("show_result", False)),
                "source_image_name": source_image.GetName(),
            },
        }

    if func == "LiveProcessingJobStatus":
        job = _get_live_job(str(params.get("job_id", "")))
        return {"success": True, "job": _job_status_payload(job)}

    if func == "LiveProcessingJobResult":
        job = _get_live_job(str(params.get("job_id", "")))
        latest_result = job.get("latest_result")
        if not isinstance(latest_result, dict):
            return {"success": False, "error": "Live-processing job has not produced a result yet."}

        data = latest_result.get("data")
        summary = latest_result.get("summary")
        if not isinstance(data, np.ndarray) or not isinstance(summary, dict):
            return {"success": False, "error": "Live-processing job result is malformed."}

        result_payload = dict(summary)
        result_payload.update(_summarize_array(data))
        include_data = bool(params.get("include_data", False)) or bool(job["params"].get("include_result_data", False))
        if include_data:
            result_payload.update(_encode_array_b64(data))

        return {
            "success": True,
            "job": _job_status_payload(job),
            "result": result_payload,
        }

    if func == "LiveProcessingJobStop":
        job = _get_live_job(str(params.get("job_id", "")))
        stop_event_obj = job.get("stop_event")
        thread_obj = job.get("thread")
        if isinstance(stop_event_obj, threading.Event):
            stop_event_obj.set()
        if isinstance(thread_obj, threading.Thread):
            timeout_s = max(2.0, float(job.get("poll_interval_s", 0.5)) * 3.0)
            thread_obj.join(timeout=timeout_s)
        return {"success": True, "job": _job_status_payload(job)}

    if func == "SaveImage":
        img = DM.GetFrontImage()
        path = params.get("path", "C:\\MCP_Export\\image.dm4")
        DM.SaveImage(img, path)
        return {"success": True, "path": path}

    if func == "Ping":
        return {"success": True, "message": "GMS DM bridge alive", "time": time.time()}

    # ── Unknown ────────────────────────────────────────────────────────────
    return {
        "success": False,
        "error":   f"Unknown function: {func!r}",
        "hint":    "Check gms_mcp.dm_plugin._dispatch for supported commands.",
    }


# ---------------------------------------------------------------------------
# Bridge thread
# ---------------------------------------------------------------------------

_zmq_context: zmq.Context | None = None
_zmq_socket: zmq.Socket | None   = None
_bridge_thread: threading.Thread | None = None
_running = threading.Event()
_bridge_ready = threading.Event()
_bridge_error: str | None = None
_bridge_state_lock = threading.Lock()


def _set_bridge_error(message: str | None) -> None:
    global _bridge_error
    with _bridge_state_lock:
        _bridge_error = message


def _get_bridge_error() -> str | None:
    with _bridge_state_lock:
        return _bridge_error


def _bridge_loop() -> None:
    """Main ZeroMQ REP loop — runs in a daemon thread."""
    global _zmq_context, _zmq_socket

    try:
        _zmq_context = zmq.Context()
        _zmq_socket = _zmq_context.socket(zmq.REP)
        _zmq_socket.setsockopt(zmq.RCVTIMEO, 500)   # 500 ms poll timeout
        _zmq_socket.bind(ZMQ_BIND)
    except Exception as exc:
        _set_bridge_error(str(exc))
        _running.clear()
        _bridge_ready.set()
        if _zmq_socket is not None:
            _zmq_socket.close(linger=0)
            _zmq_socket = None
        if _zmq_context is not None:
            _zmq_context.term()
            _zmq_context = None
        return

    _bridge_ready.set()

    while _running.is_set():
        try:
            msg_bytes = _zmq_socket.recv()
        except zmq.Again:
            continue
        except zmq.ZMQError:
            break

        try:
            cmd    = json.loads(msg_bytes.decode("utf-8"))
            result = _dispatch(cmd)
            result = _to_json_safe(result)
        except Exception as exc:
            result = {"success": False, "error": str(exc)}

        try:
            _zmq_socket.send(json.dumps(result).encode("utf-8"))
        except zmq.ZMQError:
            break

    if _zmq_socket is not None:
        _zmq_socket.close(linger=0)
        _zmq_socket = None
    if _zmq_context is not None:
        _zmq_context.term()
        _zmq_context = None


def start_bridge(bind: str = ZMQ_BIND) -> None:
    """Start the ZeroMQ bridge in a background daemon thread."""
    global _bridge_thread, ZMQ_BIND

    if _running.is_set():
        DM.Result("[GMS-MCP] Bridge is already running.\n")
        return

    ZMQ_BIND = bind
    _set_bridge_error(None)
    _bridge_ready.clear()
    _running.set()
    _bridge_thread = threading.Thread(target=_bridge_loop, daemon=True, name="gms-mcp-bridge")
    _bridge_thread.start()

    if not _bridge_ready.wait(timeout=2.0):
        DM.Result(f"[GMS-MCP] Bridge thread started → {bind}\n")
        DM.Result("[GMS-MCP] Waiting for ZeroMQ socket bind confirmation...\n")
        return

    error = _get_bridge_error()
    if error:
        DM.Result(f"[GMS-MCP] Failed to start bridge on {bind}: {error}\n")
        return

    DM.Result(f"[GMS-MCP] DM bridge ready on {bind}\n")
    DM.Result(f"[GMS-MCP] Bridge thread started → {bind}\n")


def stop_bridge() -> None:
    """Signal the bridge thread to stop and wait for it to exit."""
    _running.clear()
    if _bridge_thread and _bridge_thread.is_alive():
        _bridge_thread.join(timeout=3.0)
    DM.Result("[GMS-MCP] Bridge stopped.\n")


# ---------------------------------------------------------------------------
# Auto-start when exec()'d inside GMS
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    start_bridge()
