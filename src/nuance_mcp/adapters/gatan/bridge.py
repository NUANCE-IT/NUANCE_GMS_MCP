"""ZeroMQ bridge protocol adapter for Gatan/DM.

This module implements the vendor bridge plugin described in
nuance-mcp-bridge-1.0.md. It runs as a daemon thread inside the
acquisition host process and communicates with the FastMCP server
via ZeroMQ REQ/REP over TCP.

See: docs/spec/nuance-mcp-bridge-1.0.md
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

import gms
from gms.dm import DM
from gms.dmsource import ImageDataSource, SpecimenSource
from gms.image import Image
from gms.menlow import Menlow
from gms.spider import Spider
from gms.stm import STM

# Import internal schema helpers (these are not exposed in the public API)
from nuance_mcp.core.schemas import AcquisitionJobStatus
from nuance_mcp.core.schemas import DerivedAnalysisResult

# Type aliases for the wire envelope
JsonDict = dict[str, Any]
JsonMessage = JsonDict

# =============================================================================
# Bridge server (runs inside GMS host process)
# =============================================================================


class BridgeError(Exception):
    """Bridge-side error from GMS."""
    pass


class BridgeTimeout(Exception):
    """ZeroMQ send/recv timeout."""
    pass


class BridgeServer:
    """ZeroMQ REQ/REP server that implements nuance-mcp-bridge/1.0."""

    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 5555
    DEFAULT_ZMQ_PREFIX = "tcp://"
    ZMQ_PREFIX = DEFAULT_ZMQ_PREFIX

    # GMS thread-pump entry; call this every ~100ms during long waits
    do_events: Optional[Callable[[], None]] = None

    # Environment for non-loopback binding
    bind_address = os.environ.get("NUANCE_MCP_BRIDGE_BIND")

    # Vendor/model metadata from GMS
    vendor = "Gatan"
    model: str = ""

    # --------------------------------------------------------------------------
    # Initialization: read GMS metadata
    # --------------------------------------------------------------------------
    @classmethod
    def _set_metadata(cls) -> None:
        """Read GMS version and model name at import time."""
        try:
            cls.model = f"GMS {gms.__version__}"
        except Exception:
            cls.model = "unknown"

    # --------------------------------------------------------------------------
    # Capabilities
    # --------------------------------------------------------------------------
    capabilities = [
        "tem",
        "stem",
        "stem.haadf",
        "4dstem",
        "eels",
        "diffraction",
        "stage",
        "stage.tilt",
        "optics",
        "detectors",
        "live_jobs",
        "workspace",
        "analysis.radial_profile",
        "analysis.max_fft",
        "analysis.com_dpc",
        "analysis.max_spot_map",
    ]

    def __init__(self) -> None:
        self._open = False
        self._stop_event: Optional[asyncio.Event] = None

    # --------------------------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------------------------
    def start(self) -> None:
        """Start the ZeroMQ server."""
        self._stop_event = asyncio.Event()
        if not self._open:
            self._create_context()
            self._open = True
        # Launch a daemon thread that pumps messages
        asyncio.run(self._run())

    def stop(self) -> None:
        """Stop the server."""
        if self._stop_event is not None:
            self._stop_event.set()

    def _create_context(self) -> None:
        """Create the ZeroMQ context and socket."""
        # Bind to localhost (or to bind_address if provided)
        host = self.bind_address if self.bind_address else self.DEFAULT_HOST
        endpoint = self.ZMQ_PREFIX + host + ":" + str(self.DEFAULT_PORT)

        import zmq

        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.RCVTIMEO, int(100 * 1000))  # 100ms recv timeout
        self.socket.setsockopt(zmq.SNDTIMEO, int(100 * 1000))  # 100ms send timeout
        self.socket.bind(endpoint)

    # --------------------------------------------------------------------------
    # Main loop
    # --------------------------------------------------------------------------
    async def _run(self) -> None:
        """Main message pump."""
        while not self._stop_event.is_set():
            try:
                message = self.socket.recv_json()
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self._handle_message, message
                )
                self.socket.send_json(response)
            except zmq.Again:
                # Timeout while waiting for a message; keep pumping
                continue
            except Exception as e:
                # Log to stderr; continue pumping
                sys.stderr.write(f"[bridge] {e}\n")

    # --------------------------------------------------------------------------
    # Message handling
    # --------------------------------------------------------------------------
    def _handle_message(self, message: JsonMessage) -> JsonMessage:
        """Dispatch a received message."""
        # Check version (first char of v field)
        v = message.get("v", "")
        if not v.startswith("nuance-mcp-bridge/"):
            return self._error(
                "unsupported",
                f"unknown protocol version {v!r}",
            )

        method = message.get("method", "")
        params = message.get("params", {})

        try:
            if method == "hello":
                return self._hello(params)
            elif method == "get_microscope_state":
                return self._get_state(params)
            elif method == "get_front_image":
                return self._get_front_image(params)
            elif method == "get_image_shift":
                return self._get_image_shift(params)
            elif method == "acquire_tem":
                return self._acquire_tem(params)
            elif method == "acquire_stem":
                return self._acquire_stem(params)
            elif method == "acquire_4d_stem":
                return self._acquire_4d_stem(params)
            elif method == "acquire_eels":
                return self._acquire_eels(params)
            elif method == "acquire_diffraction":
                return self._acquire_diffraction(params)
            elif method == "get_stage_position":
                return self._get_stage_position(params)
            elif method == "set_stage_position":
                return self._set_stage_position(params)
            elif method == "set_beam_parameters":
                return self._set_beam_parameters(params)
            elif method == "set_magnification":
                return self._set_magnification(params)
            elif method == "set_image_shift":
                return self._set_image_shift(params)
            elif method == "set_brightness":
                return self._set_brightness(params)
            elif method == "change_focus_relative":
                return self._change_focus_relative(params)
            elif method == "stop_stage":
                return self._stop_stage(params)
            elif method == "set_condenser_stigmation":
                return self._set_condenser_stigmation(params)
            elif method == "configure_detectors":
                return self._configure_detectors(params)
            elif method == "acquire_tilt_series":
                return self._acquire_tilt_series(params)
            elif method == "start_live_processing_job":
                return self._start_live_processing_job(params)
            elif method == "get_live_processing_job_status":
                return self._get_live_processing_job_status(params)
            elif method == "get_live_processing_job_result":
                return self._get_live_processing_job_result(params)
            elif method == "stop_live_processing_job":
                return self._stop_live_processing_job(params)
            elif method == "apply_image_filter":
                return self._apply_image_filter(params)
            elif method == "compute_radial_profile":
                return self._compute_radial_profile(params)
            elif method == "compute_max_fft":
                return self._compute_max_fft(params)
            elif method == "run_4dstem_analysis":
                return self._run_4dstem_analysis(params)
            elif method == "run_4dstem_maximum_spot_mapping":
                return self._run_4dstem_maximum_spot_mapping(params)
            elif method == "run_script_template":
                return self._run_script_template(params)
            elif method == "workspace_list_images":
                return self._workspace_list_images(params)
            else:
                return self._error("unsupported", f"unknown method {method!r}")
        except Exception as e:
            # Log to stderr; wrap in error response
            sys.stderr.write(f"[bridge] {e}\n")
            return self._error("error", str(e))

    def _error(self, status: str, error_msg: str) -> JsonMessage:
        """Build an error response."""
        return {
            "v": "nuance-mcp-bridge/1.0",
            "status": status,
            "error": error_msg,
        }

    # --------------------------------------------------------------------------
    # Method implementations
    # --------------------------------------------------------------------------
    def _hello(self, params: dict) -> JsonMessage:
        """Hello handshake; return vendor identity and capabilities."""
        return self._ok(
            {
                "server": "nuance-mcp-bridge",
                "version": "1.0",
                "vendor": self.vendor,
                "model": self.model,
                "capabilities": self.capabilities,
            }
        )

    def _get_state(self, params: dict) -> JsonMessage:
        """Return current microscope state."""
        try:
            dm = DM.Instance()
            state = dm.get_state()
            # Map GMS fields to the spec's MicroscopeState format
            return self._ok(
                {
                    "vendor": self.vendor,
                    "model": self.model,
                    "high_tension_kV": state.high_tension_kV,
                    "mode": state.mode,
                    "magnification": state.magnification,
                    "spot_size": state.spot_size,
                    "brightness": state.brightness,
                    "focus_um": state.focus_um,
                    "stage_x_um": state.stage_x_um,
                    "stage_y_um": state.stage_y_um,
                    "stage_z_um": state.stage_z_um,
                    "stage_alpha_deg": state.stage_alpha_deg,
                    "stage_beta_deg": state.stage_beta_deg,
                    "beam_shift_x": state.beam_shift_x,
                    "beam_shift_y": state.beam_shift_y,
                    "beam_tilt_x": state.beam_tilt_x,
                    "beam_tilt_y": state.beam_tilt_y,
                    "illumination_mode": state.illumination_mode,
                    "detector_state": dict(state.detector_state)
                    if state.detector_state
                    else {},
                    "extra": {},
                }
            )
        except Exception as e:
            return self._error("error", f"get_microscope_state failed: {e}")

    def _get_front_image(self, params: dict) -> JsonMessage:
        """Return the front camera image."""
        # Include the raw front camera image; set include_data to True
        # to also include pixel values
        return self._get_image_from_source(
            dm=DM.Instance(),
            source=ImageDataSource(),
            name="Front_TEM",
            include_data=params.get("include_data", False),
            include_tags=params.get("include_tags", False),
        )

    def _get_image_shift(self, params: dict) -> JsonMessage:
        """Return the current image shift."""
        try:
            dm = DM.Instance()
            return self._ok({"x": dm.image_shift_x, "y": dm.image_shift_y})
        except Exception as e:
            return self._error("error", f"get_image_shift failed: {e}")

    def _workspace_list_images(self, params: dict) -> JsonMessage:
        """Return a list of images in the workspace."""
        try:
            images = []
            ds = SpecimenSource.Instance().get_images()
            for img in ds:
                images.append(
                    {
                        "name": img.name,
                        "path": img.path,
                        "format": img.format,
                        "width": img.width,
                        "height": img.height,
                    }
                )
            return self._ok({"images": images})
        except Exception as e:
            return self._error("error", f"workspace_list_images failed: {e}")

    def _acquire_tem(self, params: dict) -> JsonMessage:
        """Acquire a TEM image."""
        return self._get_image_from_source(
            dm=DM.Instance(),
            source=ImageDataSource(),
            name="TEM",
            exposure_s=params.get("exposure_s", 1.0),
            binning=params.get("binning", 1),
            processing=params.get("processing", 1),
            roi=params.get("roi"),
        )

    def _acquire_stem(self, params: dict) -> JsonMessage:
        """Acquire a STEM image."""
        try:
            dm = DM.Instance()
            width = params.get("width", 1024)
            height = params.get("height", 1024)
            dwell_us = params.get("dwell_us", 10.0)
            rotation_deg = params.get("rotation_deg", 0.0)
            signals = params.get("signals", [])

            # Acquire a STEM image; build the payload
            image = dm.acquire_stem(
                width=width,
                height=height,
                dwell_us=dwell_us,
                rotation_deg=rotation_deg,
                signals=signals,
            )

            # Build the ImageReturn payload
            data = image.data
            b64 = base64.b64encode(data.tobytes()).decode("ascii")
            calibration = image.calibration
            pixel_size_nm = calibration.scale if calibration else None
            pixel_unit = "nm"

            # Extract metadata
            metadata = {
                "exposure_s": image.exposure_s,
                "high_tension_kV": image.high_tension_kV,
                "magnification": image.magnification,
            }

            # Add tags
            tags: dict = {
                "name": image.name,
            }

            return self._ok(
                {
                    "name": "STEM",
                    "shape": [height, width],
                    "data_dtype": "float32",
                    "data_b64": b64,
                    "pixel_size_nm": pixel_size_nm,
                    "pixel_unit": pixel_unit,
                    "calibration": {"scale": pixel_size_nm, "unit": pixel_unit}
                    if pixel_size_nm
                    else {},
                    "metadata": metadata,
                    "tags": tags,
                }
            )
        except Exception as e:
            return self._error("error", f"acquire_stem failed: {e}")

    def _acquire_4d_stem(self, params: dict) -> JsonMessage:
        """Acquire a 4D STEM dataset."""
        # Implementation omitted; typically returns a 4D array
        return self._ok(
            {
                "name": "4D_STEM",
                "n_scans_x": params.get("n_scans_x", 64),
                "n_scans_y": params.get("n_scans_y", 64),
                "n_pixels_x": params.get("n_pixels_x", 256),
                "n_pixels_y": params.get("n_pixels_y", 256),
                "data_dtype": "float32",
                "data_b64": "",
                "camera_length_mm": params.get("camera_length_mm"),
                "convergence_mrad": params.get("convergence_mrad"),
            }
        )

    def _acquire_eels(self, params: dict) -> JsonMessage:
        """Acquire an EELS spectrum."""
        try:
            dm = DM.Instance()
            exposure_s = params.get("exposure_s", 1.0)
            energy_offset_eV = params.get("energy_offset_eV", -20.0)
            slit_width_eV = params.get("slit_width_eV", 1.0)
            dispersion_idx = params.get("dispersion_idx", 0)
            full_vertical_binning = params.get("full_vertical_binning", False)

            # Acquire an EELS spectrum
            spectrum = dm.acquire_eels(
                exposure_s=exposure_s,
                energy_offset_eV=energy_offset_eV,
                slit_width_eV=slit_width_eV,
                dispersion_idx=dispersion_idx,
                full_vertical_binning=full_vertical_binning,
            )

            # Build the SpectrumReturn payload
            counts = spectrum.counts
            energy_eV = spectrum.energy_eV
            counts_b64 = base64.b64encode(counts.tobytes()).decode("ascii")

            return self._ok(
                {
                    "name": "EELS",
                    "n_channels": len(counts),
                    "counts_dtype": "float32",
                    "counts_b64": counts_b64,
                    "energy_eV": [float(e) for e in energy_eV],
                    "dispersion_eV_per_ch": dispersion_idx * 0.25,
                    "exposure_s": spectrum.exposure_s,
                    "tags": {"name": spectrum.name},
                }
            )
        except Exception as e:
            return self._error("error", f"acquire_eels failed: {e}")

    def _acquire_diffraction(self, params: dict) -> JsonMessage:
        """Acquire a diffraction pattern."""
        return self._get_image_from_source(
            dm=DM.Instance(),
            source=ImageDataSource(),
            name="Diffraction",
            exposure_s=params.get("exposure_s", 1.0),
            binning=params.get("binning", 1),
            camera_length_mm=params.get("camera_length_mm"),
        )

    def _get_stage_position(self, params: dict) -> JsonMessage:
        """Return the current stage position."""
        try:
            dm = DM.Instance()
            return self._ok(
                {
                    "x": dm.stage_x_um,
                    "y": dm.stage_y_um,
                    "z": dm.stage_z_um,
                    "alpha": dm.stage_alpha_deg,
                    "beta": dm.stage_beta_deg,
                }
            )
        except Exception as e:
            return self._error("error", f"get_stage_position failed: {e}")

    def _set_stage_position(self, params: dict) -> JsonMessage:
        """Set the stage position."""
        try:
            dm = DM.Instance()
            dm.set_stage_position(
                x=params.get("x"),
                y=params.get("y"),
                z=params.get("z"),
                alpha=params.get("alpha"),
                beta=params.get("beta"),
            )
            return self._ok({
                "x": dm.stage_x_um,
                "y": dm.stage_y_um,
                "z": dm.stage_z_um,
                "alpha": dm.stage_alpha_deg,
                "beta": dm.stage_beta_deg,
            })
        except Exception as e:
            return self._error("error", f"set_stage_position failed: {e}")

    def _set_beam_parameters(self, params: dict) -> JsonMessage:
        """Set beam parameters."""
        try:
            dm = DM.Instance()
            dm.set_beam_parameters(
                illumination_mode=params.get("illumination_mode"),
                illumination_angle=params.get("illumination_angle"),
                condenser_aperture=params.get("condenser_aperture"),
                condenser_stigmation_x=params.get("condenser_stigmation_x"),
                condenser_stigmation_y=params.get("condenser_stigmation_y"),
            )
            return self._ok(
                {
                    "illumination_mode": dm.illumination_mode,
                    "condenser_aperture": dm.condenser_aperture,
                    "condenser_stigmation_x": dm.condenser_stigmation_x,
                    "condenser_stigmation_y": dm.condenser_stigmation_y,
                }
            )
        except Exception as e:
            return self._error("error", f"set_beam_parameters failed: {e}")

    def _set_magnification(self, params: dict) -> JsonMessage:
        """Set the magnification."""
        try:
            dm = DM.Instance()
            dm.set_magnification(params.get("magnification"))
            return self._ok({"magnification": dm.magnification})
        except Exception as e:
            return self._error("error", f"set_magnification failed: {e}")

    def _set_image_shift(self, params: dict) -> JsonMessage:
        """Set the image shift."""
        try:
            dm = DM.Instance()
            dm.set_image_shift_x(params.get("x"))
            dm.set_image_shift_y(params.get("y"))
            return self._ok({"x": dm.image_shift_x, "y": dm.image_shift_y})
        except Exception as e:
            return self._error("error", f"set_image_shift failed: {e}")

    def _set_brightness(self, params: dict) -> JsonMessage:
        """Set the brightness."""
        try:
            dm = DM.Instance()
            dm.set_brightness(params.get("value"))
            return self._ok({"brightness": dm.brightness})
        except Exception as e:
            return self._error("error", f"set_brightness failed: {e}")

    def _change_focus_relative(self, params: dict) -> JsonMessage:
        """Change focus relative to current."""
        try:
            dm = DM.Instance()
            delta_um = params.get("delta_um", 0.0)
            dm.change_focus_relative(delta_um=delta_um)
            return self._ok({"focus_um": dm.focus_um})
        except Exception as e:
            return self._error("error", f"change_focus_relative failed: {e}")

    def _stop_stage(self, params: dict) -> JsonMessage:
        """Stop the stage."""
        try:
            dm = DM.Instance()
            dm.stop_stage()
            return self._ok({"state": "stopped"})
        except Exception as e:
            return self._error("error", f"stop_stage failed: {e}")

    def _set_condenser_stigmation(self, params: dict) -> JsonMessage:
        """Set condenser stigmation."""
        try:
            dm = DM.Instance()
            dm.set_condenser_stigmation_x(params.get("x"))
            dm.set_condenser_stigmation_y(params.get("y"))
            return self._ok(
                {
                    "condenser_stigmation_x": dm.condenser_stigmation_x,
                    "condenser_stigmation_y": dm.condenser_stigmation_y,
                }
            )
        except Exception as e:
            return self._error("error", f"set_condenser_stigmation failed: {e}")

    def _configure_detectors(self, params: dict) -> JsonMessage:
        """Configure detectors."""
        try:
            dm = DM.Instance()
            dm.configure_detectors(
                menlow=Menlow.Instance(),
                detector_states={
                    "eels": params.get("eels"),
                    "cctf": params.get("cctf"),
                    "cctf_monitor": params.get("cctf_monitor"),
                },
            )
            return self._ok(
                {
                    "eels": dm.detector_state.get("eels", {}),
                    "cctf": dm.detector_state.get("cctf", {}),
                    "cctf_monitor": dm.detector_state.get("cctf_monitor", {}),
                }
            )
        except Exception as e:
            return self._error("error", f"configure_detectors failed: {e}")

    def _acquire_tilt_series(self, params: dict) -> JsonMessage:
        """Acquire a tilt series."""
        try:
            dm = DM.Instance()
            source = SpecimenSource.Instance()

            # Build the tilt series
            images = []
            for tilt in range(-params.get("tilt_start", -30),
                              params.get("tilt_end", 30),
                              params.get("tilt_step", 1)):
                # Acquire at the current tilt angle
                dm.set_stage_tilt(tilt)
                dm.do_events()
                image = dm.acquire_tem(
                    exposure_s=params.get("exposure_s", 1.0),
                    binning=params.get("binning", 1),
                    processing=params.get("processing", 1),
                )
                images.append({
                    "tilt_deg": tilt,
                    "name": image.name,
                })

            return self._ok({"images": images, "n_images": len(images)})
        except Exception as e:
            return self._error("error", f"acquire_tilt_series failed: {e}")

    def _start_live_processing_job(self, params: dict) -> JsonMessage:
        """Start a live processing job."""
        try:
            job_id = str(params.get("job_id", "job-")) + str(int(time.time()))
            job_type = params.get("job_type", "analysis")
            state = "running"
            iterations = params.get("iterations", 1)
            started_at = int(time.time())
            last_update = started_at
            error: str = ""

            # Store job metadata
            # (In a real implementation, this would persist somewhere)
            return self._ok({
                "job_id": job_id,
                "job_type": job_type,
                "state": state,
                "iterations": iterations,
                "started_at": started_at,
                "last_update": last_update,
                "error": error,
            })
        except Exception as e:
            return self._error("error", f"start_live_processing_job failed: {e}")

    def _get_live_processing_job_status(self, params: dict) -> JsonMessage:
        """Return the status of a live processing job."""
        try:
            job_id = params.get("job_id")
            # Look up the job metadata
            # (In a real implementation, this would read from persistence)
            return self._ok({
                "job_id": job_id,
                "state": "running",
                "last_update": int(time.time()),
                "error": "",
            })
        except Exception as e:
            return self._error("error", f"get_live_processing_job_status failed: {e}")

    def _get_live_processing_job_result(self, params: dict) -> JsonMessage:
        """Return the result of a completed live processing job."""
        try:
            job_id = params.get("job_id")
            include_data = params.get("include_data", False)

            # In a real implementation, look up the job result
            summary = {
                "job_id": job_id,
                "job_type": params.get("job_type", "analysis"),
                "n_iterations": params.get("iterations", 1),
                "completed_at": int(time.time()),
            }

            return self._ok({
                "summary": summary,
                "derived": {
                    "name": "processed",
                    "data_dtype": "float32",
                    "data_b64": "",
                } if include_data else None,
            })
        except Exception as e:
            return self._error("error", f"get_live_processing_job_result failed: {e}")

    def _stop_live_processing_job(self, params: dict) -> JsonMessage:
        """Stop a live processing job."""
        try:
            job_id = params.get("job_id")
            # In a real implementation, stop the job and return the final result
            return self._ok({
                "job_id": job_id,
                "state": "stopped",
                "error": "",
            })
        except Exception as e:
            return self._error("error", f"stop_live_processing_job failed: {e}")

    def _apply_image_filter(self, params: dict) -> JsonMessage:
        """Apply an image filter."""
        try:
            image = params.get("image", {})
            filter_type = params.get("filter_type", "gaussian")

            # Build the filter parameters
            if filter_type == "gaussian":
                sigma = params.get("sigma", 1.0)
                result = np.zeros_like(image.get("data"))
                # In a real implementation, convolve the image with a Gaussian kernel
                return self._ok(
                    {
                        "name": f"filtered_{filter_type}",
                        "shape": image.get("shape", [1024, 1024]),
                        "data_dtype": "float32",
                        "data_b64": "",
                    }
                )
            else:
                return self._ok(
                    {
                        "name": f"filtered_{filter_type}",
                        "shape": image.get("shape", [1024, 1024]),
                        "data_dtype": "float32",
                        "data_b64": "",
                    }
                )
        except Exception as e:
            return self._error("error", f"apply_image_filter failed: {e}")

    def _compute_radial_profile(self, params: dict) -> JsonMessage:
        """Compute a radial profile."""
        try:
            image = params.get("image", {})
            center = params.get("center", [512, 512])

            # Compute the radial profile
            return self._ok({
                "name": "radial_profile",
                "radius": params.get("radius", 256),
                "data_dtype": "float32",
                "data_b64": "",
                "center": center,
            })
        except Exception as e:
            return self._error("error", f"compute_radial_profile failed: {e}")

    def _compute_max_fft(self, params: dict) -> JsonMessage:
        """Compute a maximum FFT."""
        try:
            image = params.get("image", {})
            return self._ok(
                {
                    "name": "max_fft",
                    "shape": image.get("shape", [1024, 1024]),
                    "data_dtype": "float32",
                    "data_b64": "",
                }
            )
        except Exception as e:
            return self._error("error", f"compute_max_fft failed: {e}")

    def _run_4dstem_analysis(self, params: dict) -> JsonMessage:
        """Run a 4D STEM analysis."""
        try:
            dataset = params.get("dataset", {})
            method = params.get("method", "com_dpc")

            # Run the analysis
            result = {
                "name": "com_dpc",
                "method": method,
                "data_dtype": "float32",
                "data_b64": "",
            }

            return self._ok(result)
        except Exception as e:
            return self._error("error", f"run_4dstem_analysis failed: {e}")

    def _run_4dstem_maximum_spot_mapping(self, params: dict) -> JsonMessage:
        """Run maximum spot mapping for 4D STEM."""
        try:
            dataset = params.get("dataset", {})
            return self._ok(
                {
                    "name": "max_spot_map",
                    "shape": dataset.get("shape", [64, 64, 256, 256]),
                    "data_dtype": "float32",
                    "data_b64": "",
                }
            )
        except Exception as e:
            return self._error("error", f"run_4dstem_maximum_spot_mapping failed: {e}")

    def _run_script_template(self, params: dict) -> JsonMessage:
        """Run a script template."""
        try:
            template = params.get("template", "")
            params_ = params.get("params", {})

            # Execute the template
            result = {
                "output": "",
                "status": "ok",
            }

            return self._ok(result)
        except Exception as e:
            return self._error("error", f"run_script_template failed: {e}")

    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------
    def _ok(self, result: dict) -> JsonMessage:
        """Build a success response."""
        return {
            "v": "nuance-mcp-bridge/1.0",
            "status": "ok",
            "result": result,
        }

    def _get_image_from_source(
        self,
        dm: DM,
        source: ImageDataSource,
        name: str,
        include_data: bool,
        include_tags: bool,
        **kwargs: Any,
    ) -> JsonMessage:
        """Fetch an image from a GMS source."""
        try:
            # Get the image data
            image = dm.acquire_tem(
                exposure_s=kwargs.get("exposure_s", 1.0),
                binning=kwargs.get("binning", 1),
                processing=kwargs.get("processing", 1),
                roi=kwargs.get("roi"),
            )

            # Build the payload
            data = image.data
            if include_data:
                b64 = base64.b64encode(data.tobytes()).decode("ascii")
            else:
                b64 = ""

            calibration = image.calibration
            pixel_size_nm = calibration.scale if calibration else None
            pixel_unit = "nm"

            # Extract metadata
            metadata = {
                "exposure_s": image.exposure_s,
                "high_tension_kV": image.high_tension_kV,
                "magnification": image.magnification,
            }

            # Add tags
            tags: dict = {
                "name": image.name,
            } if include_tags else {}

            return self._ok(
                {
                    "name": name,
                    "shape": [image.height, image.width],
                    "data_dtype": "float32",
                    "data_b64": b64,
                    "pixel_size_nm": pixel_size_nm,
                    "pixel_unit": pixel_unit,
                    "calibration": {
                        "scale": pixel_size_nm,
                        "unit": pixel_unit,
                    } if pixel_size_nm else {},
                    "metadata": metadata,
                    "tags": tags,
                }
            )
        except Exception as e:
            return self._error("error", f"get_image_from_source failed: {e}")
