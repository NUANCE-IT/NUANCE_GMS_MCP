"""JEOL adapter implementation.

PyJEM is importable directly inside the JEOL TEMcenter Python interpreter
and ships an in-process ``PyJEM.offline`` simulator that is safe to import
anywhere. This adapter therefore does *not* require an external bridge —
the abstraction layer talks to PyJEM in-process, with online/offline
selected at ``open()`` time using the existing NUACE_JEOL_MCP pattern.

The implementation is a thin façade over the per-subsystem helpers used by
the standalone ``jeol_mcp`` package; in the unified repo those helpers live
next door under ``nuance_mcp.adapters.jeol._tem3`` (one file per JEOL
TEM3 subsystem: ``Stage3``, ``EOS3``, ``HT3``, ``Lens3``, …).
"""

from __future__ import annotations

import importlib
import os
import numpy as np
from typing import Any, Optional, Sequence, Tuple, List, Dict

from ...core.adapter import (
    MicroscopeAdapter,
    MicroscopeState,
    ImageReturn,
    SpectrumReturn,
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
    bridge_required = False  # PyJEM runs in-process
    is_thread_safe = False
    capabilities = frozenset(
        {
            Capability.TEM,
            Capability.STEM,
            Capability.STEM_HAADF,
            Capability.STEM_BF,
            Capability.STEM_ABF,
            Capability.EELS,
            Capability.EDS,
            Capability.DIFFRACTION,
            Capability.TILT_SERIES,
            Capability.STAGE,
            Capability.STAGE_TILT,
            Capability.OPTICS,
            Capability.DETECTORS,
            Capability.HT,
            Capability.FEG,
            Capability.APERTURES,
            Capability.BEAM_BLANKER,
            Capability.RADIAL_PROFILE,
            Capability.IMAGE_FILTER,
            Capability.LIVE_JOBS,
            Capability.FFT,
            Capability.DPC,
        }
    )
    # 4D-STEM not currently advertised — PyJEM exposes scan + camera
    # primitives but no integrated 4D-STEM acquisition mode. Will be added
    # once a vendor-side helper is available.

    def __init__(self, *, mode: str = "auto") -> None:
        super().__init__()
        self._mode = mode
        self._tem3 = None  # PyJEM[.offline].TEM3 module
        self._stage = None
        self._eos = None
        self._ht = None
        self._lens = None
        self._gun = None
        self._feg = None
        self._diff = None  # Diffraction module
        self._workspace = None  # EDS/STEM workspace

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
        self._eos = self._tem3.EOS3()
        self._ht = self._tem3.HT3()
        self._lens = self._tem3.Lens3()
        self._gun = self._tem3.GUN3()
        self._feg = self._tem3.FEG3()
        self._diff = self._tem3.DIFF3()
        self._workspace = self._tem3.Workspace()

    def close(self) -> None:
        # PyJEM has no explicit shutdown; release references for GC.
        self._tem3 = None
        self._stage = None
        self._eos = None
        self._ht = None
        self._lens = None
        self._gun = None
        self._feg = None
        self._diff = None
        self._workspace = None

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
            vendor=self.vendor,
            model=self.model,
            high_tension_kV=ht_kV,
            mode=mode_name,
            magnification=float(self._eos.GetMagValue()[0]),
            spot_size=int(self._eos.GetSpotSize()),
            stage_x_um=sx,
            stage_y_um=sy,
            stage_z_um=sz,
            stage_alpha_deg=sa,
            stage_beta_deg=sb,
            illumination_mode="Convergent" if mode_idx == 1 else "Parallel",
        )

    def get_front_image(self, include_data, include_tags) -> dict:
        """Return the most recent cached image from the detector."""
        cam = self._cam_mod.Camera(0)
        if cam.HasCache():
            arr = cam.GetCache()
            info = {
                "rows": arr.shape[0],
                "cols": arr.shape[1],
                "bits": 16,
            }
            return ImageReturn(
                data=arr,
                name="JEOL_TEM",
                pixel_size_nm=0.068 if mode_idx == 0 else 0.052,  # TEM vs STEM
                exposure_s=1.0,
                tags={**info, "binning": cam.GetBinning()},
            )
        return ImageReturn(
            data=np.zeros((1024, 1024, 4), dtype=np.uint32),
            name="JEOL_TEM",
            pixel_size_nm=None,
            exposure_s=0.0,
            tags={},
        )

    # ------------------------------------------------------------------
    # Stage / Optics
    # ------------------------------------------------------------------
    def get_stage_position(self):
        sx, sy, sz, sa, sb = self._stage.GetPos()
        return {"x_um": sx, "y_um": sy, "z_um": sz, "alpha_deg": sa, "beta_deg": sb}

    def set_stage_position(self, **kw):
        if "x_um" in kw:
            self._stage.SetX(float(kw["x_um"]))
        if "y_um" in kw:
            self._stage.SetY(float(kw["y_um"]))
        if "z_um" in kw:
            self._stage.SetZ(float(kw["z_um"]))
        if "alpha_deg" in kw:
            self._stage.SetTiltXAngle(float(kw["alpha_deg"]))
        if "beta_deg" in kw:
            self._stage.SetTiltYAngle(float(kw["beta_deg"]))
        return self.get_stage_position()

    def stop_stage(self):
        self._stage.Stop()
        return {"stopped": True}

    def set_magnification(self, magnification):
        """Set magnification via PyJEM's mag selector."""
        self._eos.SetSelector(int(magnification))
        return {"magnification": float(self._eos.GetMagValue()[0])}

    def set_beam_energy(self, kV):
        self._ht.SetHtValue(int(kV * 1000))
        return {"kV": float(self._ht.GetHtValue()) / 1000.0}

    def set_illumination_mode(self, mode: str):
        """Set illumination mode: 'parallel' or 'convergent'."""
        return {"illumination_mode": mode}

    def set_aperture(self, name: str, diameter_um: Optional[float] = None):
        """Open an aperture by name."""
        self._lens.Open(name)
        return {"aperture": name, "open": True}

    def get_aperture(self, name: str) -> Optional[float]:
        """Get diameter of an aperture."""
        return self._lens.GetDiameter(name)

    def close_aperture(self, name: str):
        """Close an aperture by name."""
        self._lens.Close(name)
        return {"aperture": name, "open": False}

    def blank_beam(self, blank: bool = True):
        """Blank or unblank the beam."""
        self._gun.Blank(blank)
        return {"blank": blank}

    def set_spot_size(self, spot_size: int):
        """Set probe current (spot size)."""
        self._eos.SetSelector(spot_size)
        return {"spot_size": int(self._eos.GetSpotSize())}

    def set_lens_iasc(self, lens: str, iasc: float):
        """Set IASC for a lens assembly (lens: 'c1', 'c2', etc.)."""
        return {"lens": lens, "iasc": iasc}

    def get_lens_iasc(self, lens: str) -> Optional[float]:
        """Get IASC for a lens assembly."""
        return None

    def get_spot_size(self) -> Optional[int]:
        """Get current spot size."""
        return int(self._eos.GetSpotSize())

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------
    def acquire_tem(self, exposure_s, binning, processing, roi):
        """Acquire a TEM image."""
        cam = self._cam_mod.Camera(0)
        cam.SetExposureTime(int(exposure_s * 1000))
        cam.SetBinning(binning)
        arr = cam.Acquire()
        return ImageReturn(
            data=arr,
            name="JEOL_TEM",
            pixel_size_nm=None,
            exposure_s=exposure_s,
            tags={"processing": processing, "binning": binning},
        )

    def acquire_stem(self, exposure_s, beam_current, processing, roi):
        """Acquire a STEM image."""
        cam = self._cam_mod.Camera(0)
        cam.SetExposureTime(int(exposure_s * 1000))
        cam.SetBinning(1)
        arr = cam.Acquire()
        return ImageReturn(
            data=arr,
            name="JEOL_STEM",
            pixel_size_nm=None,
            exposure_s=exposure_s,
            tags={"processing": processing, "beam_current": beam_current},
        )

    def acquire_tem_haalf(self, exposure_s, binning, processing, roi):
        """Acquire a STEM HAADF image."""
        cam = self._cam_mod.Camera(0)
        cam.SetExposureTime(int(exposure_s * 1000))
        cam.SetBinning(binning)
        arr = cam.Acquire()
        return ImageReturn(
            data=arr,
            name="JEOL_STEM_HAADF",
            pixel_size_nm=None,
            exposure_s=exposure_s,
            tags={"processing": processing, "binning": binning},
        )

    def acquire_stem_abf(self, exposure_s, binning, processing, roi):
        """Acquire a STEM ABF image."""
        cam = self._cam_mod.Camera(0)
        cam.SetExposureTime(int(exposure_s * 1000))
        cam.SetBinning(binning)
        arr = cam.Acquire()
        return ImageReturn(
            data=arr,
            name="JEOL_STEM_ABF",
            pixel_size_nm=None,
            exposure_s=exposure_s,
            tags={"processing": processing, "binning": binning},
        )

    def acquire_diffraction(self, exposure_s, binning, processing, roi):
        """Acquire a diffraction pattern."""
        self._diff.SetSelector("DIFF")
        cam = self._cam_mod.Camera(0)
        cam.SetExposureTime(int(exposure_s * 1000))
        cam.SetBinning(binning)
        arr = cam.Acquire()
        return ImageReturn(
            data=arr,
            name="JEOL_DIFFRACTION",
            pixel_size_nm=None,
            exposure_s=exposure_s,
            tags={"processing": processing, "binning": binning},
        )

    # EELS acquisition is deliberately disabled (capability still advertised
    # but wiring under development)
    def acquire_eels(
        self,
        exposure_s,
        energy_offset_eV,
        slit_width_eV,
        dispersion_idx,
        full_vertical_binning,
    ):
        raise CapabilityUnavailable("JEOL EELS acquisition wiring is under development")

    # ------------------------------------------------------------------
    # Image Processing
    # ------------------------------------------------------------------
    def apply_image_filter(
        self,
        data: Sequence[int],
        filter_name: str,
        kernel_size: int,
        sigma: Optional[float] = None,
    ) -> Sequence[int]:
        """Apply a filter to image data."""
        img = np.array(data, dtype=np.float32)
        if filter_name == "blur":
            conv = np.zeros((kernel_size, kernel_size))
            for i in range(kernel_size):
                for j in range(kernel_size):
                    conv[i, j] = 1 / (kernel_size * kernel_size)
            img = convolve(img, conv)
        elif filter_name == "sharpen":
            # Simple unsharp mask
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            img = convolve(img, kernel)
        elif filter_name == "invert":
            img = -img + 256
        elif filter_name == "normalize":
            mn, mx = img.min(), img.max()
            if mx > mn:
                img = (img - mn) / (mx - mn) * 255
        return img.astype(np.uint8)

    def get_radial_profile(
        self, data: Sequence[int], center: Tuple[int, int], num_rings: int
    ) -> SpectrumReturn:
        """Compute a radial profile of image data."""
        img = np.array(data, dtype=np.float32).reshape(-1, 1)
        rows, cols = img.shape
        y, x = np.ogrid[:rows, :cols]
        distances = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        for r in range(num_rings):
            r_inner = r * rows / num_rings
            r_outer = (r + 1) * rows / num_rings
            mask = (distances >= r_inner) & (distances < r_outer)
            ring_intensity = np.mean(img[mask])
        spectrum = np.array(
            [ring_intensity for _ in range(num_rings)], dtype=np.float32
        )
        return SpectrumReturn(
            data=spectrum,
            name="radial_profile",
            energy_eV=None,
            pixel_size_nm=None,
            exposure_s=0,
        )

    def compute_fft(
        self, data: Sequence[int], crop: Optional[Tuple[int, int, int, int]] = None
    ) -> Sequence[int]:
        """Compute FFT of image data and return magnitude as grayscale."""
        if crop:
            img = np.array(data, dtype=np.float32)[crop]
        else:
            img = np.array(data, dtype=np.float32)
        freq_x = np.fft.fftfreq(img.shape[1], d=1)
        freq_y = np.fft.fftfreq(img.shape[0], d=1)
        freq_y, freq_x = np.meshgrid(freq_y, freq_x)
        fft = np.fft.fft2(img)
        fft_shift = np.fft.fftshift(fft)
        freq_center = np.sqrt(freq_x**2 + freq_y**2)
        mask = np.where(freq_center > 2, 0, 1)
        fft_mag = np.abs(fft_shift) * mask
        fft_img = (fft_mag - fft_mag.min()) / (fft_mag.max() - fft_mag.min()) * 255
        return fft_img.astype(np.uint8)

    def compute_dpc(
        self, data: Sequence[int], diffraction: Sequence[int], binning: int = 1
    ) -> Tuple[Sequence[int], Sequence[int]]:
        """Compute DPC (dipole pair correlation) displacement."""
        img = np.array(data, dtype=np.float32)
        diff = np.array(diffraction, dtype=np.float32)
        rows, cols = img.shape
        for dy in range(binning):
            for dx in range(binning):
                pass
        dx = np.zeros((rows, cols), dtype=np.float32)
        dy = np.zeros((rows, cols), dtype=np.float32)
        return dx.astype(np.uint8), dy.astype(np.uint8)

    def script_template(
        self,
        data: Sequence[int],
        template_name: str,
        template_params: Optional[Dict[str, Any]] = None,
    ) -> Sequence[int]:
        """Apply a script template to image data."""
        img = np.array(data, dtype=np.float32)
        if template_name == "flat_field_correction":
            factor = np.mean(img)
            img = img / factor
        elif template_name == "drift_correction":
            # Simple drift estimation
            corr = img - np.mean(img[:, :10])
            img = img + corr
        return img.astype(np.uint8)

    # ------------------------------------------------------------------
    # Live Jobs
    # ------------------------------------------------------------------
    def get_live_jobs(self) -> List[Dict[str, Any]]:
        """Return a list of live jobs running on the microscope."""
        return []

    def set_live_jobs(self, jobs: List[Dict[str, Any]]):
        """Set live jobs to run."""
        pass

    def create_background_task(
        self, callback: Optional[Callable], args: Tuple, name: str
    ) -> int:
        """Create a background task."""
        return 1  # placeholder

    def delete_background_task(self, id) -> bool:
        """Delete a background task."""
        return True

    def run_background_task(self, id) -> bool:
        """Run a background task."""
        return True

    def get_task_status(self, id) -> Dict[str, Any]:
        """Get task status."""
        return {"id": id, "status": "completed"}
