"""Vendor-neutral physics-plausible simulator backend.

The simulator implements :class:`MicroscopeAdapter` directly. It is the
zero-dependency baseline that the CI suite runs against, and the default
adapter when no vendor backend is selected on the command line. It uses
``numpy.random.default_rng`` with explicit seeds so that all
hardware-independent tests are bit-stable across machines for a fixed
NumPy version.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from .adapter import (
    MicroscopeAdapter, MicroscopeState, ImageReturn, SpectrumReturn,
    CapabilityUnavailable,
)
from .capabilities import Capability
from .lifecycle import JobRegistry, JobState


@dataclass
class _SimulatedColumn:
    """Mutable column state owned by the simulator."""
    high_tension_kV: float = 200.0
    mode: str = "TEM"
    magnification: float = 50_000.0
    spot_size: int = 3
    brightness: float = 0.5
    focus_um: float = 0.0
    stage: list[float] = field(default_factory=lambda: [0., 0., 0., 0., 0.])
    image_shift: tuple[float, float] = (0.0, 0.0)


class SimulatorAdapter(MicroscopeAdapter):
    """Default adapter: a physics-plausible simulator with no hardware deps."""

    vendor = "simulator"
    model = "MicroscopeSimulator-v1"
    bridge_required = False
    is_thread_safe = True
    capabilities = frozenset({
        Capability.TEM, Capability.STEM, Capability.STEM_HAADF,
        Capability.STEM_BF, Capability.STEM_ABF,
        Capability.FOURD_STEM, Capability.EELS, Capability.DIFFRACTION,
        Capability.STAGE, Capability.STAGE_TILT,
        Capability.OPTICS, Capability.DETECTORS,
        Capability.RADIAL_PROFILE, Capability.MAX_FFT,
        Capability.IMAGE_FILTER, Capability.COM_DPC,
        Capability.MAX_SPOT_MAP, Capability.LIVE_JOBS,
        Capability.WORKSPACE, Capability.TILT_SERIES,
    })

    def __init__(self, *, seed: int = 0) -> None:
        super().__init__()
        self._rng = np.random.default_rng(seed)
        self._col = _SimulatedColumn()
        self._front: Optional[ImageReturn] = None
        self._workspace: list[ImageReturn] = []
        self._jobs = JobRegistry()

    def open(self) -> None: pass
    def close(self) -> None: pass

    # --- diagnostics --------------------------------------------------

    def get_state(self) -> MicroscopeState:
        c = self._col
        return MicroscopeState(
            vendor=self.vendor, model=self.model,
            high_tension_kV=c.high_tension_kV, mode=c.mode,
            magnification=c.magnification, spot_size=c.spot_size,
            brightness=c.brightness, focus_um=c.focus_um,
            stage_x_um=c.stage[0], stage_y_um=c.stage[1], stage_z_um=c.stage[2],
            stage_alpha_deg=c.stage[3], stage_beta_deg=c.stage[4],
            illumination_mode="Parallel",
        )

    def get_front_image(self, include_data: bool, include_tags: bool) -> dict:
        if self._front is None:
            return {"error": "no front image", "shape": None}
        arr = self._front.data
        out = {
            "name": self._front.name,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "statistics": {
                "min": float(arr.min()), "max": float(arr.max()),
                "mean": float(arr.mean()), "std": float(arr.std()),
            },
            "calibration": {
                "scale": self._front.pixel_size_nm,
                "unit": self._front.pixel_unit,
            },
        }
        if include_tags:
            out["tags"] = self._front.tags
        if include_data:
            import base64
            out["data_b64"] = base64.b64encode(arr.tobytes()).decode()
            out["data_dtype"] = str(arr.dtype)
        return out

    def workspace_list_images(self) -> list[dict]:
        return [
            {"name": im.name, "shape": list(im.data.shape)}
            for im in self._workspace
        ]

    # --- acquisition --------------------------------------------------

    def _new_hrtem(self, N: int = 512) -> np.ndarray:
        x = np.arange(N)
        xx, yy = np.meshgrid(x, x)
        img = np.zeros_like(xx, dtype=np.float32)
        for k in [(0.45, 0.0), (0.225, 0.39), (-0.225, 0.39)]:
            img += np.cos(2 * np.pi * (k[0] * xx + k[1] * yy) / 4.5)
        img += 0.4 * self._rng.standard_normal(img.shape)
        return img

    def acquire_tem(self, exposure_s, binning, processing, roi) -> ImageReturn:
        arr = self._new_hrtem()
        if roi is not None:
            t, l, b, r = roi
            arr = arr[t:b, l:r]
        img = ImageReturn(
            data=arr, name="SimTEM",
            pixel_size_nm=0.05 * binning, exposure_s=exposure_s,
            tags={"processing": processing, "binning": binning},
        )
        self._front = img
        self._workspace.append(img)
        return img

    def acquire_stem(self, width, height, dwell_us, rotation_deg, signals):
        rng = self._rng
        arr = 0.05 * rng.poisson(20, (height, width)).astype(np.float32)
        for _ in range(28):
            cx, cy = rng.integers(40, min(height, width) - 40, 2)
            r = rng.integers(8, 22)
            yy, xx = np.ogrid[:height, :width]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
            arr[mask] += rng.uniform(2.0, 4.5)
        img = ImageReturn(data=gaussian_filter(arr, 0.7), name="SimSTEM",
                          pixel_size_nm=0.02, exposure_s=dwell_us * width * height * 1e-6,
                          tags={"rotation_deg": rotation_deg, "signals": signals})
        self._front = img
        self._workspace.append(img)
        return img

    def acquire_4d_stem(self, scan_x, scan_y, dwell_us,
                        camera_length_mm, convergence_mrad):
        rng = self._rng
        det = 64
        cbed = np.zeros((scan_y, scan_x, det, det), dtype=np.float32)
        yy, xx = np.indices((det, det))
        yi, xi = np.indices((scan_y, scan_x))
        sx = 6 * np.sin(2 * np.pi * xi / max(scan_x, 1))
        sy = 6 * np.cos(2 * np.pi * yi / max(scan_y, 1))
        for i in range(scan_y):
            for j in range(scan_x):
                rr = (xx - det/2 - sx[i, j])**2 + (yy - det/2 - sy[i, j])**2
                cbed[i, j] = (rr < 9**2).astype(np.float32) * 100.0
        cbed += 0.5 * rng.poisson(0.5, cbed.shape)
        img = ImageReturn(
            data=cbed, name="Sim4DSTEM",
            pixel_size_nm=0.05, exposure_s=dwell_us * scan_x * scan_y * 1e-6,
            tags={"camera_length_mm": camera_length_mm,
                  "convergence_mrad": convergence_mrad},
        )
        self._front = img
        self._workspace.append(img)
        return img

    def acquire_eels(self, exposure_s, energy_offset_eV, slit_width_eV,
                     dispersion_idx, full_vertical_binning):
        n = 1024
        dispersions = [0.1, 0.25, 0.5, 1.0]
        dispersion = dispersions[dispersion_idx]
        ev = energy_offset_eV + (np.arange(n) - n / 2) * dispersion
        eels = 1.0 + 0.2 * self._rng.standard_normal(n)
        for centre, amp, width in [(0, 1500, 0.5), (23, 280, 6),
                                   (285, 95, 4), (453, 70, 3.5), (532, 90, 4)]:
            eels += amp * np.exp(-0.5 * ((ev - centre) / width)**2)
        return SpectrumReturn(
            counts=eels, energy_eV=ev, name="SimEELS",
            exposure_s=exposure_s, dispersion_eV_per_ch=dispersion,
            tags={"slit_width_eV": slit_width_eV,
                  "full_vertical_binning": full_vertical_binning},
        )

    def acquire_diffraction(self, exposure_s, camera_length_mm, binning):
        N = 512
        rng = self._rng
        yy, xx = np.indices((N, N))
        rr = np.hypot(xx - N/2, yy - N/2)
        D = np.zeros((N, N), dtype=np.float32)
        for r0 in [55, 92, 122, 158, 190]:
            D += np.exp(-0.5 * ((rr - r0) / 2.5)**2) * (4.0 + 0.05 * r0)
        D += 0.5 * rng.poisson(0.7, D.shape)
        D += 200.0 * np.exp(-0.5 * (rr / 4.0)**2)
        img = ImageReturn(data=gaussian_filter(D, 0.5), name="SimDiffraction",
                          pixel_size_nm=None, exposure_s=exposure_s,
                          tags={"camera_length_mm": camera_length_mm,
                                "binning": binning})
        self._front = img
        self._workspace.append(img)
        return img

    # --- stage / optics ----------------------------------------------

    def get_stage_position(self) -> dict:
        x, y, z, a, b = self._col.stage
        return {"x_um": x, "y_um": y, "z_um": z,
                "alpha_deg": a, "beta_deg": b}

    def set_stage_position(self, **kw):
        cur = self._col.stage
        mapping = {"x_um": 0, "y_um": 1, "z_um": 2,
                   "alpha_deg": 3, "beta_deg": 4}
        for k, idx in mapping.items():
            v = kw.get(k)
            if v is not None:
                cur[idx] = float(v)
        return self.get_stage_position()

    def set_beam_parameters(self, **kw):
        if (v := kw.get("spot_size")) is not None: self._col.spot_size = int(v)
        if (v := kw.get("focus_um")) is not None: self._col.focus_um = float(v)
        return {"spot_size": self._col.spot_size, "focus_um": self._col.focus_um}

    def configure_detectors(self, **kw):
        return {"applied": {k: v for k, v in kw.items() if v is not None}}

    def set_magnification(self, magnification):
        self._col.magnification = float(magnification)
        return {"magnification": self._col.magnification}

    def set_image_shift(self, x, y):
        self._col.image_shift = (float(x), float(y))
        return {"image_shift": self._col.image_shift}

    def set_brightness(self, value):
        self._col.brightness = float(value)
        return {"brightness": self._col.brightness}

    def change_focus_relative(self, delta_um):
        self._col.focus_um += float(delta_um)
        return {"focus_um": self._col.focus_um}

    def stop_stage(self) -> dict:
        return {"stopped": True}

    def set_condenser_stigmation(self, x, y):
        return {"condenser_stigmation": [float(x), float(y)]}

    # --- live jobs (in-process) --------------------------------------

    def start_live_processing_job(self, **kwargs) -> dict:
        rec = self._jobs.new_job(kwargs["job_type"], dict(kwargs))
        # In the simulator we do not actually spawn a thread; we just mark
        # the job READY and treat get_result as on-demand compute.
        rec.state = JobState.RESULT_READY
        rec.iterations = 1
        return rec.summary()

    def get_live_processing_job_status(self, job_id):
        return self._jobs.status(job_id)

    def get_live_processing_job_result(self, job_id, include_data=False):
        rec = self._jobs._get(job_id)
        return {"summary": rec.summary(), "data_available": include_data}

    def stop_live_processing_job(self, job_id):
        return self._jobs.stop(job_id)
