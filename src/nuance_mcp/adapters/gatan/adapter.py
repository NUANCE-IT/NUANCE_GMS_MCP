"""Gatan/GMS adapter.

Two execution paths are supported, selected at construction time:

* ``mode="bridge"`` (default for live microscope) — the adapter talks to a
  daemon thread running inside the GMS Python environment over a ZeroMQ
  REP/REQ pair (see :mod:`nuance_mcp.adapters.gatan.bridge`). The bridge
  binds ``127.0.0.1:5555`` by default; facility-LAN exposure is opt-in via
  ``GMS_MCP_ZMQ_BIND``.
* ``mode="direct"`` — for development on a workstation that has both the
  agent stack *and* DigitalMicrograph importable in the same interpreter
  (rare but used for unit tests on a microscope PC).

When neither path is available the constructor raises; callers should
typically fall back to :class:`SimulatorAdapter` in that case.
"""

from __future__ import annotations

import os
from typing import Optional

from ...core.adapter import (
    MicroscopeAdapter,
    MicroscopeState,
    ImageReturn,
    SpectrumReturn,
    CapabilityUnavailable,
)
from ...core.capabilities import Capability


class GatanGMSAdapter(MicroscopeAdapter):
    """Adapter for Gatan Microscopy Suite (GMS) 3.60.

    The actual vendor calls live inside
    :mod:`nuance_mcp.adapters.gatan.bridge` (host-process plugin); this
    class is a thin client that speaks the bridge JSON contract.
    """

    vendor = "Gatan"
    model = "GMS 3.60"
    bridge_required = True
    is_thread_safe = False  # DM API is host-thread-affine
    capabilities = frozenset(
        {
            Capability.TEM,
            Capability.STEM,
            Capability.STEM_HAADF,
            Capability.STEM_BF,
            Capability.STEM_ABF,
            Capability.FOURD_STEM,
            Capability.EELS,
            Capability.DIFFRACTION,
            Capability.TILT_SERIES,
            Capability.STAGE,
            Capability.STAGE_TILT,
            Capability.OPTICS,
            Capability.DETECTORS,
            Capability.IMAGE_FILTER,
            Capability.RADIAL_PROFILE,
            Capability.MAX_FFT,
            Capability.COM_DPC,
            Capability.MAX_SPOT_MAP,
            Capability.SCRIPT_TEMPLATE,
            Capability.LIVE_JOBS,
            Capability.WORKSPACE,
        }
    )

    def __init__(
        self,
        *,
        mode: str = "bridge",
        endpoint: Optional[str] = None,
        timeout_ms: int = 5000,
    ) -> None:
        super().__init__()
        self._mode = mode
        self._endpoint = endpoint or os.environ.get(
            "GMS_MCP_ZMQ", "tcp://127.0.0.1:5555"
        )
        self._timeout_ms = timeout_ms
        self._zmq = None  # lazy: only imported in bridge mode
        self._sock = None
        self._dm = None  # populated in direct mode

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def open(self) -> None:
        if self._mode == "bridge":
            import zmq  # local import

            ctx = zmq.Context.instance()
            self._sock = ctx.socket(zmq.REQ)
            self._sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
            self._sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
            self._sock.connect(self._endpoint)
            self._zmq = zmq
            self._call("hello", {"client": "nuance-mcp"})
        elif self._mode == "direct":
            import DigitalMicrograph as DM  # noqa: F401 — local

            self._dm = DM
        else:
            raise ValueError(f"unknown Gatan mode: {self._mode!r}")

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(0)
            self._sock = None

    # ------------------------------------------------------------------
    # Bridge helper
    # ------------------------------------------------------------------
    def _call(self, method: str, params: dict) -> dict:
        """Single round-trip over the bridge."""
        if self._mode != "bridge":
            raise RuntimeError("direct-mode call not implemented in skeleton")
        import json

        msg = {"v": "nuance-mcp-bridge/1.0", "method": method, "params": params}
        self._sock.send_string(json.dumps(msg))
        resp = json.loads(self._sock.recv_string())
        if resp.get("status") == "error":
            raise RuntimeError(resp.get("error", "bridge error"))
        return resp.get("result", {})

    # ------------------------------------------------------------------
    # Concrete vendor surface (thin pass-throughs over the bridge)
    # ------------------------------------------------------------------
    def get_state(self) -> MicroscopeState:
        r = self._call("get_microscope_state", {})
        return MicroscopeState(vendor=self.vendor, model=self.model, **r)

    def get_front_image(self, include_data, include_tags) -> dict:
        return self._call(
            "get_front_image",
            {"include_data": include_data, "include_tags": include_tags},
        )

    def acquire_tem(self, exposure_s, binning, processing, roi):
        r = self._call(
            "acquire_tem",
            {
                "exposure_s": exposure_s,
                "binning": binning,
                "processing": processing,
                "roi": roi,
            },
        )
        return _result_to_image(r)

    def acquire_stem(self, width, height, dwell_us, rotation_deg, signals):
        r = self._call(
            "acquire_stem",
            {
                "width": width,
                "height": height,
                "dwell_us": dwell_us,
                "rotation_deg": rotation_deg,
                "signals": signals,
            },
        )
        return _result_to_image(r)

    def acquire_4d_stem(
        self, scan_x, scan_y, dwell_us, camera_length_mm, convergence_mrad
    ):
        r = self._call(
            "acquire_4d_stem",
            {
                "scan_x": scan_x,
                "scan_y": scan_y,
                "dwell_us": dwell_us,
                "camera_length_mm": camera_length_mm,
                "convergence_mrad": convergence_mrad,
            },
        )
        return _result_to_image(r)

    def acquire_eels(
        self,
        exposure_s,
        energy_offset_eV,
        slit_width_eV,
        dispersion_idx,
        full_vertical_binning,
    ):
        r = self._call(
            "acquire_eels",
            {
                "exposure_s": exposure_s,
                "energy_offset_eV": energy_offset_eV,
                "slit_width_eV": slit_width_eV,
                "dispersion_idx": dispersion_idx,
                "full_vertical_binning": full_vertical_binning,
            },
        )
        return _result_to_spectrum(r)

    def acquire_diffraction(self, exposure_s, camera_length_mm, binning):
        r = self._call(
            "acquire_diffraction",
            {
                "exposure_s": exposure_s,
                "camera_length_mm": camera_length_mm,
                "binning": binning,
            },
        )
        return _result_to_image(r)

    def get_stage_position(self):
        return self._call("get_stage_position", {})

    def set_stage_position(self, **kw):
        return self._call("set_stage_position", kw)

    def set_beam_parameters(self, **kw):
        return self._call("set_beam_parameters", kw)

    def configure_detectors(self, **kw):
        return self._call("configure_detectors", kw)

    def set_magnification(self, magnification):
        return self._call("set_magnification", {"magnification": magnification})

    def set_image_shift(self, x, y):
        return self._call("set_image_shift", {"x": x, "y": y})

    def set_brightness(self, value):
        return self._call("set_brightness", {"value": value})

    def change_focus_relative(self, delta_um):
        return self._call("change_focus_relative", {"delta_um": delta_um})

    def stop_stage(self):
        return self._call("stop_stage", {})

    def set_condenser_stigmation(self, x, y):
        return self._call("set_condenser_stigmation", {"x": x, "y": y})

    def acquire_tilt_series(self, **kw):
        return self._call("acquire_tilt_series", kw)

    def apply_image_filter(self, **kw):
        r = self._call("apply_image_filter", kw)
        return _result_to_image(r)

    def compute_radial_profile(self, **kw):
        return self._call("compute_radial_profile", kw)

    def compute_max_fft(self, **kw):
        r = self._call("compute_max_fft", kw)
        return _result_to_image(r)

    def run_4dstem_analysis(self, **kw):
        return self._call("run_4dstem_analysis", kw)

    def run_4dstem_maximum_spot_mapping(self, **kw):
        r = self._call("run_4dstem_maximum_spot_mapping", kw)
        return _result_to_image(r)

    def run_script_template(self, template, params):
        return self._call(
            "run_script_template", {"template": template, "params": params}
        )

    def start_live_processing_job(self, **kw):
        return self._call("start_live_processing_job", kw)

    def get_live_processing_job_status(self, job_id):
        return self._call("get_live_processing_job_status", {"job_id": job_id})

    def get_live_processing_job_result(self, job_id, include_data=False):
        return self._call(
            "get_live_processing_job_result",
            {"job_id": job_id, "include_data": include_data},
        )

    def stop_live_processing_job(self, job_id):
        return self._call("stop_live_processing_job", {"job_id": job_id})

    def workspace_list_images(self):
        return self._call("workspace_list_images", {})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_to_image(r: dict) -> ImageReturn:
    """Convert the bridge's image dict back into an :class:`ImageReturn`."""
    import base64, numpy as np

    if "data_b64" in r:
        raw = base64.b64decode(r["data_b64"])
        arr = np.frombuffer(raw, dtype=r["data_dtype"]).reshape(r["shape"])
    else:
        arr = np.zeros(r.get("shape", (1, 1)), dtype=r.get("data_dtype", "float32"))
    return ImageReturn(
        data=arr,
        name=r.get("name", ""),
        pixel_size_nm=r.get("calibration", {}).get("scale"),
        pixel_unit=r.get("calibration", {}).get("unit", "nm"),
        exposure_s=r.get("metadata", {}).get("exposure_s"),
        tags=r.get("tags", {}),
    )


def _result_to_spectrum(r: dict) -> SpectrumReturn:
    import base64, numpy as np

    if "counts_b64" in r:
        raw = base64.b64decode(r["counts_b64"])
        counts = np.frombuffer(raw, dtype=r["counts_dtype"])
    else:
        counts = np.zeros(r.get("n_channels", 1), dtype="float32")
    ev = np.asarray(r.get("energy_eV", [0.0] * counts.size), dtype="float32")
    return SpectrumReturn(
        counts=counts,
        energy_eV=ev,
        name=r.get("name", ""),
        exposure_s=r.get("exposure_s"),
        dispersion_eV_per_ch=r.get("dispersion_eV_per_ch"),
        tags=r.get("tags", {}),
    )
