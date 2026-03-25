"""
dm_simulator.py
================
Faithful in-process simulator of the DigitalMicrograph (DM) Python API.

When the GMS MCP server is started outside GMS (e.g. for local Ollama
testing), `import DigitalMicrograph as DM` fails.  This module provides a
drop-in replacement with realistic physics-plausible return values for every
function used by the MCP server, making it possible to develop and test the
full MCP ↔ Ollama pipeline on any workstation — no microscope required.

Usage (automatic, inside gms_mcp_server.py):
    try:
        import DigitalMicrograph as DM
    except ImportError:
        from dm_simulator import DMSimulator
        DM = DMSimulator()
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Simulated image / tag objects
# ---------------------------------------------------------------------------

class SimTagGroup:
    """Minimal simulation of a DM TagGroup (hierarchical metadata store)."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def SetTagAsString(self, key: str, value: str) -> None:
        self._store[key] = value

    def SetTagAsFloat(self, key: str, value: float) -> None:
        self._store[key] = float(value)

    def SetTagAsLong(self, key: str, value: int) -> None:
        self._store[key] = int(value)

    def GetTagAsString(self, key: str) -> tuple[bool, str]:
        v = self._store.get(key)
        return (True, str(v)) if v is not None else (False, "")

    def GetTagAsFloat(self, key: str) -> tuple[bool, float]:
        v = self._store.get(key)
        return (True, float(v)) if v is not None else (False, 0.0)

    def GetTagAsLong(self, key: str) -> tuple[bool, int]:
        v = self._store.get(key)
        return (True, int(v)) if v is not None else (False, 0)

    def GetTagAsTagGroup(self, key: str) -> tuple[bool, "SimTagGroup"]:
        sub = SimTagGroup()
        return (True, sub)

    def OpenBrowserWindow(self, modal: bool = False) -> None:
        pass

    def to_dict(self) -> dict:
        return dict(self._store)


class SimImage:
    """Simulation of a DM Py_Image object."""

    _id_counter = 1

    def __init__(self, data: np.ndarray, name: str = "SimImage") -> None:
        self._data = data.copy()
        self._name = name
        self._id = SimImage._id_counter
        SimImage._id_counter += 1
        self._tags = SimTagGroup()
        # seed calibration: 0.01 nm/px
        self._tags.SetTagAsFloat("Calibration:PixelSizeX", 0.01)
        self._tags.SetTagAsFloat("Calibration:PixelSizeY", 0.01)
        self._tags.SetTagAsString("Calibration:Unit", "nm")

    def GetNumArray(self) -> np.ndarray:
        """Return a writeable view (mirrors real DM memory-mapped behaviour)."""
        return self._data

    def GetTagGroup(self) -> SimTagGroup:
        return self._tags

    def SetName(self, name: str) -> None:
        self._name = name

    def GetName(self) -> str:
        return self._name

    def GetID(self) -> int:
        return self._id

    def ShowImage(self) -> None:
        pass  # no-op outside GMS

    def UpdateImage(self) -> None:
        pass

    def GetDimensionCalibration(self, axis: int, _param: int = 0) -> tuple:
        """Return (origin, scale, unit) for the requested axis."""
        return (0.0, 0.01, "nm")

    def SetDimensionCalibration(self, axis: int, origin: float,
                                 scale: float, unit: str, _param: int) -> None:
        self._tags.SetTagAsFloat(f"Cal:axis{axis}:scale", scale)
        self._tags.SetTagAsString(f"Cal:axis{axis}:unit", unit)

    def to_summary(self) -> dict:
        return {
            "id": self._id,
            "name": self._name,
            "shape": list(self._data.shape),
            "dtype": str(self._data.dtype),
            "min": float(self._data.min()),
            "max": float(self._data.max()),
            "mean": float(self._data.mean()),
        }

    def to_b64(self) -> str:
        """Serialise pixel data as base64-encoded raw bytes."""
        return base64.b64encode(self._data.tobytes()).decode()


# ---------------------------------------------------------------------------
# Camera acquisition parameter object
# ---------------------------------------------------------------------------

@dataclass
class SimAcqParams:
    processing: int = 3      # 1=raw 2=dark 3=dark+gain
    exposure: float = 1.0    # seconds
    binning_x: int = 1
    binning_y: int = 1
    ccd_top: int = 0
    ccd_left: int = 0
    ccd_bottom: int = 2048
    ccd_right: int = 2048
    continuous: bool = False
    validated: bool = False


# ---------------------------------------------------------------------------
# Simulated camera object
# ---------------------------------------------------------------------------

class SimCamera:
    """Simulation of a Gatan camera (OneView / K3 / Rio)."""

    def __init__(self, name: str = "SimCamera-OneView") -> None:
        self._name = name
        self._inserted = True
        self._temp_target = -25.0
        self._temp_actual = -24.8
        self._ccd_w = 2048
        self._ccd_h = 2048
        self._pixel_um = 15.0    # µm physical pixel pitch

    def get_name(self) -> str:
        return self._name


# ---------------------------------------------------------------------------
# Microscope state (the "virtual microscope column")
# ---------------------------------------------------------------------------

@dataclass
class MicroscopeState:
    # Beam / optics
    high_tension_V: float = 200_000.0
    spot_size: int = 3
    brightness: float = 0.5
    focus: float = 0.0
    mag_index: int = 10
    magnification: float = 50_000.0
    operation_mode: str = "TEM"          # TEM | STEM | DIFFRACTION
    illumination_mode: str = "Parallel"  # Parallel | Convergent

    # Beam shift / tilt
    beam_shift_x: float = 0.0
    beam_shift_y: float = 0.0
    beam_tilt_x: float = 0.0
    beam_tilt_y: float = 0.0
    image_shift_x: float = 0.0
    image_shift_y: float = 0.0

    # Stigmators
    obj_stig_x: float = 0.0
    obj_stig_y: float = 0.0
    cond_stig_x: float = 0.0
    cond_stig_y: float = 0.0

    # Stage
    stage_x_um: float = 0.0
    stage_y_um: float = 0.0
    stage_z_um: float = 0.0
    stage_alpha_deg: float = 0.0
    stage_beta_deg: float = 0.0

    # GIF / EELS
    energy_loss_eV: float = 0.0
    slit_width_eV: float = 10.0
    slit_inserted: bool = False
    dispersion_index: int = 0
    drift_tube_V: float = 0.0
    drift_tube_on: bool = False

    # Camera length (diffraction / STEM)
    camera_length_mm: float = 100.0

    # DigiScan
    ds_frame_w: int = 512
    ds_frame_h: int = 512
    ds_pixel_time_us: float = 10.0
    ds_rotation_deg: float = 0.0
    ds_flyback_us: float = 500.0
    ds_signals_enabled: dict = field(default_factory=lambda: {0: True, 1: True, 2: False})
    ds_continuous: bool = False

    # HT offset
    ht_offset_V: float = 0.0
    ht_offset_enabled: bool = False


# ---------------------------------------------------------------------------
# The DMSimulator — the main drop-in for `import DigitalMicrograph as DM`
# ---------------------------------------------------------------------------

class DMSimulator:
    """
    Drop-in replacement for `DigitalMicrograph` (DM) Python module.

    All functions return realistic, physics-plausible values so that the
    GMS MCP server can be tested end-to-end without a real microscope.
    """

    def __init__(self) -> None:
        self._state = MicroscopeState()
        self._camera = SimCamera()
        self._front_image: SimImage | None = None
        self._images: dict[int, SimImage] = {}
        self._persistent_tags = SimTagGroup()

        # seed a 512×512 HRTEM-like front image
        self._front_image = self._make_hrtem_image(512, 512)
        self._images[self._front_image.GetID()] = self._front_image

    # ------------------------------------------------------------------
    # Image generation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_hrtem_image(w: int, h: int) -> SimImage:
        """Generate a synthetic HRTEM image with periodic lattice fringes."""
        x = np.linspace(0, 4 * np.pi, w)
        y = np.linspace(0, 4 * np.pi, h)
        xx, yy = np.meshgrid(x, y)
        lattice = (np.sin(xx * 3) * np.cos(yy * 3) +
                   np.cos(xx * 5 - yy * 2) * 0.5)
        noise = np.random.normal(0, 0.05, lattice.shape)
        data = ((lattice + noise - lattice.min()) /
                (lattice.max() - lattice.min()) * 4096).astype(np.float32)
        img = SimImage(data, "HRTEM_sim")
        return img

    @staticmethod
    def _make_haadf_image(w: int, h: int) -> SimImage:
        """Synthetic HAADF-STEM image: bright nanoparticles on dark background."""
        data = np.random.poisson(50, (h, w)).astype(np.float32)
        for _ in range(np.random.randint(5, 15)):
            cx = np.random.randint(20, w - 20)
            cy = np.random.randint(20, h - 20)
            r  = np.random.randint(5, 20)
            yy, xx = np.ogrid[:h, :w]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
            data[mask] += np.random.uniform(200, 1000)
        return SimImage(data, "HAADF_sim")

    @staticmethod
    def _make_diffraction_image(w: int, h: int) -> SimImage:
        """Synthetic polycrystalline electron diffraction pattern."""
        data = np.zeros((h, w), dtype=np.float32)
        cx, cy = w // 2, h // 2
        d_spacings = [2.1, 1.8, 1.5, 1.2, 0.9]    # Å
        pixel_scale = 0.05                           # 1/Å per pixel
        yy, xx = np.ogrid[:h, :w]
        for d in d_spacings:
            q_px = (1.0 / d) / pixel_scale
            ring = np.abs(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - q_px)
            data += np.exp(-(ring ** 2) / (2 * 1.5 ** 2)) * 3000
        data += np.random.poisson(20, data.shape).astype(np.float32)
        # bright direct beam
        beam = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 5 ** 2))
        data += beam * 10000
        return SimImage(data, "SAED_sim")

    @staticmethod
    def _make_eels_spectrum(channels: int = 2048) -> SimImage:
        """Synthetic EELS spectrum: zero-loss + plasmon + core-loss edges."""
        energy = np.linspace(-20, 1000, channels)
        spec = np.zeros(channels, dtype=np.float32)
        # zero-loss peak
        spec += 50000 * np.exp(-(energy ** 2) / (2 * 0.5 ** 2))
        # plasmon (~25 eV)
        spec += 5000 * np.exp(-((energy - 25) ** 2) / (2 * 3 ** 2))
        spec += 500  * np.exp(-((energy - 50) ** 2) / (2 * 5 ** 2))
        # Ti L2,3 edge at ~460 eV
        spec[energy > 450] += 200 * np.exp(-(energy[energy > 450] - 460) / 20)
        # Noise
        spec = np.random.poisson(np.maximum(spec, 0)).astype(np.float32)
        return SimImage(spec.reshape(1, channels), "EELS_sim")

    @staticmethod
    def _make_4d_stem(scan_x: int = 16, scan_y: int = 16,
                      det_x: int = 64, det_y: int = 64) -> SimImage:
        """Synthetic 4D-STEM dataset (scan_y × scan_x × det_y × det_x)."""
        data = np.zeros((scan_y, scan_x, det_y, det_x), dtype=np.float32)
        cx, cy = det_x // 2, det_y // 2
        yy, xx = np.ogrid[:det_y, :det_x]
        for iy in range(scan_y):
            for ix in range(scan_x):
                shift_x = (ix - scan_x / 2) * 0.3
                shift_y = (iy - scan_y / 2) * 0.3
                disk = np.exp(-((xx - cx - shift_x) ** 2 +
                                (yy - cy - shift_y) ** 2) / (2 * 5 ** 2))
                noise = np.random.poisson(2, (det_y, det_x)).astype(np.float32)
                data[iy, ix] = disk * 1000 + noise
        return SimImage(data.reshape(scan_y, scan_x * det_y * det_x),
                        "4DSTEM_sim")  # flattened for SimImage

    # ------------------------------------------------------------------
    # Core image management
    # ------------------------------------------------------------------

    def GetFrontImage(self) -> SimImage:
        if self._front_image is None:
            self._front_image = self._make_hrtem_image(1024, 1024)
        return self._front_image

    def FindImageByName(self, name: str) -> SimImage | None:
        for img in self._images.values():
            if img.GetName() == name:
                return img
        return None

    def FindImageByID(self, id_: int) -> SimImage | None:
        return self._images.get(id_)

    def CreateImage(self, arr: np.ndarray) -> SimImage:
        img = SimImage(arr)
        self._images[img.GetID()] = img
        self._front_image = img
        return img

    def CreateReal2DImage(self, name: str, _bytes: int, w: int, h: int) -> SimImage:
        data = np.zeros((h, w), dtype=np.float32)
        img = SimImage(data, name)
        self._images[img.GetID()] = img
        return img

    def OpenImage(self, path: str) -> SimImage:
        img = self._make_hrtem_image(512, 512)
        img.SetName(path.split("\\")[-1])
        self._images[img.GetID()] = img
        self._front_image = img
        return img

    def SaveImage(self, img: SimImage, path: str) -> None:
        pass  # no-op in simulation

    def GetPersistentTagGroup(self) -> SimTagGroup:
        return self._persistent_tags

    def Result(self, text: str) -> None:
        print(f"[DM] {text}", end="")

    def DoEvents(self) -> None:
        pass

    def ExecuteScriptString(self, script: str) -> None:
        pass  # no-op: DM-Script bridge not available in simulation

    # ------------------------------------------------------------------
    # Camera Manager (CM_*) — TEM / camera acquisition
    # ------------------------------------------------------------------

    def CM_GetCurrentCamera(self) -> SimCamera:
        return self._camera

    def CM_GetCameraName(self, camera: SimCamera) -> str:
        return camera.get_name()

    def CM_GetCameraIdentifier(self, camera: SimCamera) -> str:
        return "SIM-001"

    def CM_GetCameraControllerClass(self, camera: SimCamera) -> str:
        return "SimulatedController"

    def CM_IsCameraRetractable(self, camera: SimCamera) -> bool:
        return True

    def CM_GetCameraInserted(self, camera: SimCamera) -> bool:
        return camera._inserted

    def CM_SetCameraInserted(self, camera: SimCamera, inserted: int) -> None:
        camera._inserted = bool(inserted)

    def CM_GetActualTemperature_C(self, camera: SimCamera) -> float:
        return camera._temp_actual

    def CM_SetTargetTemperature_C(self, camera: SimCamera,
                                   use_target: int, temp: float) -> None:
        camera._temp_target = temp

    def CM_IsTemperatureStable(self, camera: SimCamera, temp: float) -> bool:
        return abs(temp - camera._temp_actual) < 1.0

    def CM_CCD_GetSize(self, camera: SimCamera,
                        w: int, h: int) -> tuple[int, int]:
        return (camera._ccd_w, camera._ccd_h)

    def CM_CCD_GetPixelSize_um(self, camera: SimCamera,
                                 pw: float, ph: float) -> tuple[float, float]:
        return (camera._pixel_um, camera._pixel_um)

    def CM_CreateAcquisitionParameters_FullCCD(
        self, camera: SimCamera, processing: int,
        exposure: float, bin_x: int, bin_y: int
    ) -> SimAcqParams:
        return SimAcqParams(
            processing=processing,
            exposure=exposure,
            binning_x=bin_x,
            binning_y=bin_y,
        )

    def CM_GetCameraAcquisitionParameterSet(
        self, camera: SimCamera, mode: str, style: str,
        set_name: str, create: int
    ) -> SimAcqParams:
        return SimAcqParams(exposure=1.0, binning_x=1, binning_y=1)

    def CM_SetExposure(self, acq: SimAcqParams, t: float) -> None:
        acq.exposure = t

    def CM_SetBinning(self, acq: SimAcqParams, bx: int, by: int) -> None:
        acq.binning_x = bx
        acq.binning_y = by

    def CM_SetCCDReadArea(self, acq: SimAcqParams,
                           top: int, left: int, bot: int, right: int) -> None:
        acq.ccd_top = top; acq.ccd_left = left
        acq.ccd_bottom = bot; acq.ccd_right = right

    def CM_SetProcessing(self, acq: SimAcqParams, proc: int) -> None:
        acq.processing = proc

    def CM_SetDoContinuousReadout(self, acq: SimAcqParams, cont: int) -> None:
        acq.continuous = bool(cont)

    def CM_SetStandardParameters(
        self, acq: SimAcqParams, proc: int, exp: float,
        bx: int, by: int, top: int, left: int, bot: int, right: int
    ) -> None:
        acq.processing = proc; acq.exposure = exp
        acq.binning_x = bx; acq.binning_y = by
        acq.ccd_top = top; acq.ccd_left = left
        acq.ccd_bottom = bot; acq.ccd_right = right

    def CM_Validate_AcquisitionParameters(
        self, camera: SimCamera, acq: SimAcqParams
    ) -> None:
        acq.validated = True

    def CM_AcquireImage(
        self, camera: SimCamera, acq: SimAcqParams
    ) -> SimImage:
        """Simulate a camera acquisition with appropriate image type."""
        mode = self._state.operation_mode
        w = (acq.ccd_right - acq.ccd_left) // acq.binning_x
        h = (acq.ccd_bottom - acq.ccd_top) // acq.binning_y

        if mode == "DIFFRACTION":
            img = self._make_diffraction_image(w, h)
        elif mode == "EELS":
            img = self._make_eels_spectrum()
        else:
            img = self._make_hrtem_image(w, h)

        tags = img.GetTagGroup()
        tags.SetTagAsFloat("Acquisition:ExposureTime", acq.exposure)
        tags.SetTagAsLong("Acquisition:BinningX", acq.binning_x)
        tags.SetTagAsLong("Acquisition:BinningY", acq.binning_y)
        tags.SetTagAsFloat("Microscope:HighTension_kV",
                           self._state.high_tension_V / 1000)
        tags.SetTagAsFloat("Microscope:Magnification",
                           self._state.magnification)

        self._images[img.GetID()] = img
        self._front_image = img
        return img

    def CM_AcquireDarkReference(
        self, camera: SimCamera, acq: SimAcqParams,
        dark_img: SimImage, frame_info: Any
    ) -> None:
        data = dark_img.GetNumArray()
        data[:] = np.random.normal(100, 5, data.shape).astype(data.dtype)

    def CM_CreateImageForAcquire(
        self, camera: SimCamera, acq: SimAcqParams, name: str
    ) -> SimImage:
        w = (acq.ccd_right - acq.ccd_left) // acq.binning_x
        h = (acq.ccd_bottom - acq.ccd_top) // acq.binning_y
        return self.CreateReal2DImage(name, 4, w, h)

    def CM_GetCameraManager(self) -> object:
        return object()

    def CM_GetCameras(self, mgr: object) -> list:
        return [self._camera]

    # ------------------------------------------------------------------
    # DigiScan (DS*) — STEM scanning
    # ------------------------------------------------------------------

    def DSGetNumberOfSignals(self) -> int:
        return 4

    def DSSetFrameSize(self, w: int, h: int) -> None:
        self._state.ds_frame_w = w
        self._state.ds_frame_h = h

    def DSSetPixelTime(self, us: float) -> None:
        self._state.ds_pixel_time_us = us

    def DSSetRotation(self, deg: float) -> None:
        self._state.ds_rotation_deg = deg

    def DSSetFlybackTime(self, us: float) -> None:
        self._state.ds_flyback_us = us

    def DSSetLineSync(self, enabled: int) -> None:
        pass

    def DSSetSignalEnabled(self, channel: int, enabled: int) -> None:
        self._state.ds_signals_enabled[channel] = bool(enabled)

    def DSGetSignalEnabled(self, channel: int) -> bool:
        return self._state.ds_signals_enabled.get(channel, False)

    def DSSetContinuousMode(self, cont: int) -> None:
        self._state.ds_continuous = bool(cont)

    def DSStartAcquisition(self) -> None:
        w = self._state.ds_frame_w
        h = self._state.ds_frame_h
        img = self._make_haadf_image(w, h)
        self._images[img.GetID()] = img
        self._front_image = img

    def DSStopAcquisition(self) -> None:
        pass

    def DSWaitUntilFinished(self) -> None:
        pass

    def DSSetBeamPosition(self, x: int, y: int) -> None:
        pass

    def DSSetBeamBlanked(self, blanked: int) -> None:
        pass

    # ------------------------------------------------------------------
    # Imaging Filter / GIF (IF*, IFC*) — EELS
    # ------------------------------------------------------------------

    def IFSetEELSMode(self) -> None:
        self._state.operation_mode = "EELS"

    def IFSetImageMode(self) -> None:
        self._state.operation_mode = "TEM"

    def IFSetEnergyLoss(self, eV: float) -> None:
        self._state.energy_loss_eV = eV

    def IFGetEnergyLoss(self, _: int = 0) -> float:
        return self._state.energy_loss_eV

    def IFSetSlitWidth(self, eV: float) -> None:
        self._state.slit_width_eV = eV

    def IFSetSlitIn(self, inserted: int) -> None:
        self._state.slit_inserted = bool(inserted)

    def IFIsInEELSMode(self) -> bool:
        return self._state.operation_mode == "EELS"

    def IFIsInImageMode(self) -> bool:
        return self._state.operation_mode == "TEM"

    def IFCGetNumberofDispersions(self) -> int:
        return 4

    def IFCGetSlitWidth(self) -> float:
        return self._state.slit_width_eV

    def IFCSetEnergy(self, eV: float) -> None:
        self._state.energy_loss_eV = eV

    def IFCSetSlitWidth(self, eV: float) -> None:
        self._state.slit_width_eV = eV

    def IFCSetSlitIn(self, inserted: int) -> None:
        self._state.slit_inserted = bool(inserted)

    def IFCSetDriftTubeVoltage(self, V: float) -> None:
        self._state.drift_tube_V = V

    def IFCSetDriftTubeOn(self, on: int) -> None:
        self._state.drift_tube_on = bool(on)

    def IFCSetActiveDispersions(self, idx: int) -> None:
        self._state.dispersion_index = idx

    def IFCSetAperture(self, idx: int) -> None:
        pass

    # ------------------------------------------------------------------
    # Microscope control (EM*) — Optics, stage, HT
    # ------------------------------------------------------------------

    # --- High tension ---
    def EMCanGetHighTension(self) -> bool:
        return True

    def EMGetHighTension(self) -> float:
        return self._state.high_tension_V

    def EMHasHighTensionOffset(self) -> bool:
        return True

    def EMSetHighTensionOffset(self, V: float) -> None:
        self._state.ht_offset_V = V

    def EMSetHighTensionOffsetEnabled(self, enabled: bool) -> None:
        self._state.ht_offset_enabled = enabled

    # --- Focus ---
    def EMGetFocus(self) -> float:
        return self._state.focus

    def EMSetFocus(self, v: float) -> None:
        self._state.focus = v

    def EMChangeFocus(self, delta: float) -> None:
        self._state.focus += delta

    # --- Spot size and brightness ---
    def EMGetSpotSize(self) -> int:
        return self._state.spot_size

    def EMSetSpotSize(self, size: int) -> None:
        self._state.spot_size = max(1, min(11, size))

    def EMGetBrightness(self) -> float:
        return self._state.brightness

    def EMSetBrightness(self, v: float) -> None:
        self._state.brightness = max(0.0, min(1.0, v))

    # --- Magnification ---
    def EMCanGetMagnification(self) -> bool:
        return True

    def EMGetMagnification(self) -> float:
        return self._state.magnification

    def EMGetMagIndex(self) -> int:
        return self._state.mag_index

    def EMSetMagIndex(self, idx: int) -> None:
        self._state.mag_index = idx
        # Simulate magnification stepping
        self._state.magnification = 1000 * (2 ** max(0, idx - 1))

    # --- Beam shift / tilt ---
    def EMSetBeamShift(self, x: float, y: float) -> None:
        self._state.beam_shift_x = x
        self._state.beam_shift_y = y

    def EMGetBeamShift(self, x: float, y: float) -> tuple[float, float]:
        return (self._state.beam_shift_x, self._state.beam_shift_y)

    def EMSetCalibratedBeamShift(self, x: float, y: float) -> None:
        self._state.beam_shift_x = x
        self._state.beam_shift_y = y

    def EMChangeCalibratedBeamShift(self, dx: float, dy: float) -> None:
        self._state.beam_shift_x += dx
        self._state.beam_shift_y += dy

    def EMSetBeamTilt(self, x: float, y: float) -> None:
        self._state.beam_tilt_x = x
        self._state.beam_tilt_y = y

    def EMChangeCalibratedBeamTilt(self, dx: float, dy: float) -> None:
        self._state.beam_tilt_x += dx
        self._state.beam_tilt_y += dy

    # --- Image shift ---
    def EMSetImageShift(self, x: float, y: float) -> None:
        self._state.image_shift_x = x
        self._state.image_shift_y = y

    def EMChangeCalibratedImageShift(self, dx: float, dy: float) -> None:
        self._state.image_shift_x += dx
        self._state.image_shift_y += dy

    # --- Stigmation ---
    def EMSetObjectiveStigmation(self, x: float, y: float) -> None:
        self._state.obj_stig_x = x
        self._state.obj_stig_y = y

    def EMChangeCondensorStigmation(self, dx: float, dy: float) -> None:
        self._state.cond_stig_x += dx
        self._state.cond_stig_y += dy

    # --- Operation mode ---
    def EMGetOperationMode(self) -> str:
        return self._state.operation_mode

    def EMGetIlluminationMode(self) -> str:
        return self._state.illumination_mode

    def EMGetIlluminationModes(self) -> SimTagGroup:
        tg = SimTagGroup()
        tg.SetTagAsString("0", "Parallel")
        tg.SetTagAsString("1", "Convergent")
        return tg

    # --- Camera length ---
    def EMCanGetCameraLength(self) -> bool:
        return True

    def EMGetCameraLength(self) -> float:
        return self._state.camera_length_mm

    def EMSetCameraLength(self, mm: float) -> None:
        self._state.camera_length_mm = mm

    # --- Stage ---
    def EMGetStageX(self) -> float:
        return self._state.stage_x_um

    def EMGetStageY(self) -> float:
        return self._state.stage_y_um

    def EMGetStageZ(self) -> float:
        return self._state.stage_z_um

    def EMGetStageAlpha(self) -> float:
        return self._state.stage_alpha_deg

    def EMGetStageBeta(self) -> float:
        return self._state.stage_beta_deg

    def EMSetStageX(self, um: float) -> None:
        self._state.stage_x_um = um

    def EMSetStageY(self, um: float) -> None:
        self._state.stage_y_um = um

    def EMSetStageXY(self, x: float, y: float) -> None:
        self._state.stage_x_um = x
        self._state.stage_y_um = y

    def EMSetStageAlpha(self, deg: float) -> None:
        self._state.stage_alpha_deg = max(-80.0, min(80.0, deg))

    def EMSetStageBeta(self, deg: float) -> None:
        self._state.stage_beta_deg = max(-30.0, min(30.0, deg))

    def EMSetStagePositions(self, flags: int, x: float, y: float,
                              z: float, alpha: float, beta: float) -> None:
        if flags & 1:  self._state.stage_x_um = x
        if flags & 2:  self._state.stage_y_um = y
        if flags & 4:  self._state.stage_z_um = z
        if flags & 8:  self._state.stage_alpha_deg = max(-80.0, min(80.0, alpha))
        if flags & 16: self._state.stage_beta_deg = max(-30.0, min(30.0, beta))

    def EMGetStagePositions(
        self, flags: int,
        x: float, y: float, z: float, alpha: float, beta: float
    ) -> tuple:
        return (
            self._state.stage_x_um,
            self._state.stage_y_um,
            self._state.stage_z_um,
            self._state.stage_alpha_deg,
            self._state.stage_beta_deg,
        )

    def EMWaitUntilReady(self) -> None:
        pass  # instant in simulation

    def EMStopStage(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Utility / convenience
    # ------------------------------------------------------------------

    def get_state_dict(self) -> dict:
        """Expose full microscope state for diagnostics."""
        s = self._state
        return {
            "high_tension_kV": s.high_tension_V / 1000,
            "spot_size":        s.spot_size,
            "brightness":       s.brightness,
            "focus_um":         s.focus,
            "magnification":    s.magnification,
            "operation_mode":   s.operation_mode,
            "camera_length_mm": s.camera_length_mm,
            "stage": {
                "x_um": s.stage_x_um,
                "y_um": s.stage_y_um,
                "z_um": s.stage_z_um,
                "alpha_deg": s.stage_alpha_deg,
                "beta_deg": s.stage_beta_deg,
            },
            "beam": {
                "shift_x": s.beam_shift_x,
                "shift_y": s.beam_shift_y,
                "tilt_x":  s.beam_tilt_x,
                "tilt_y":  s.beam_tilt_y,
            },
            "eels": {
                "energy_loss_eV": s.energy_loss_eV,
                "slit_width_eV":  s.slit_width_eV,
                "slit_inserted":  s.slit_inserted,
                "dispersion_idx": s.dispersion_index,
            },
            "digiscan": {
                "frame_w": s.ds_frame_w,
                "frame_h": s.ds_frame_h,
                "dwell_us": s.ds_pixel_time_us,
                "rotation_deg": s.ds_rotation_deg,
                "signals": s.ds_signals_enabled,
            },
        }
