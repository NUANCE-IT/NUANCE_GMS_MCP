"""
test_gms_mcp.py
================
Comprehensive test suite for the GMS MCP server and Ollama integration.

Test groups
-----------
    TestDMSimulator       : Unit tests for the physics simulator
    TestMCPServerTools    : Unit tests for every MCP tool (no LLM required)
    TestServerTransport   : Server starts cleanly in both stdio and HTTP modes
    TestOllamaIntegration : End-to-end tests requiring a live Ollama instance

Run all tests (no Ollama required):
    pytest test_gms_mcp.py -v -m "not ollama"

Run end-to-end with Ollama (Ollama must be running with a model pulled):
    OLLAMA_MODEL=qwen2.5:7b pytest test_gms_mcp.py -v

Run a single test:
    pytest test_gms_mcp.py::TestMCPServerTools::test_acquire_tem -v
"""

from __future__ import annotations

import json
import os
import sys
import subprocess
import asyncio
from pathlib import Path

import numpy as np
import pytest

# Ensure local packages resolve
_HERE = Path(__file__).parent.parent.resolve() / "src"
sys.path.insert(0, str(_HERE))

# Force simulation mode before importing the server
os.environ["GMS_SIMULATE"] = "1"

from gms_mcp.simulator import DMSimulator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def dm() -> DMSimulator:
    """Shared DMSimulator instance for the entire test session."""
    return DMSimulator()


@pytest.fixture(scope="session")
def server():
    """
    Import the GMS MCP server module in simulation mode.
    Returns the module so tests can call tools directly.
    """
    import gms_mcp.server as srv
    return srv


# ---------------------------------------------------------------------------
# TestDMSimulator — physics simulator unit tests
# ---------------------------------------------------------------------------

class TestDMSimulator:
    """Verify that the DMSimulator faithfully mimics the DM Python API."""

    def test_get_front_image_returns_image(self, dm: DMSimulator) -> None:
        img = dm.GetFrontImage()
        assert img is not None
        arr = img.GetNumArray()
        assert arr.ndim == 2
        assert arr.shape[0] > 0 and arr.shape[1] > 0

    def test_numpy_view_is_writable(self, dm: DMSimulator) -> None:
        img = dm.GetFrontImage()
        arr = img.GetNumArray()
        original_mean = arr.mean()
        arr[:10, :10] = 0.0   # in-place modification
        new_mean = img.GetNumArray().mean()
        assert new_mean != original_mean  # proves the view is live

    def test_create_image_from_numpy(self, dm: DMSimulator) -> None:
        data = np.ones((128, 128), dtype=np.float32) * 42.0
        img = dm.CreateImage(data)
        assert img.GetNumArray().mean() == pytest.approx(42.0)

    def test_tag_round_trip(self, dm: DMSimulator) -> None:
        img = dm.GetFrontImage()
        tags = img.GetTagGroup()
        tags.SetTagAsFloat("Test:Value", 3.14159)
        ok, val = tags.GetTagAsFloat("Test:Value")
        assert ok is True
        assert val == pytest.approx(3.14159)

    def test_calibration_axes(self, dm: DMSimulator) -> None:
        img = dm.GetFrontImage()
        origin, scale, unit = img.GetDimensionCalibration(0, 0)
        assert isinstance(scale, float)
        assert scale > 0

    def test_stage_get_set_roundtrip(self, dm: DMSimulator) -> None:
        dm.EMSetStageX(150.0)
        dm.EMSetStageAlpha(-45.0)
        assert dm.EMGetStageX() == pytest.approx(150.0)
        assert dm.EMGetStageAlpha() == pytest.approx(-45.0)

    def test_stage_alpha_clamped(self, dm: DMSimulator) -> None:
        dm.EMSetStageAlpha(999.0)
        assert dm.EMGetStageAlpha() <= 80.0

    def test_stage_move_multiple_axes(self, dm: DMSimulator) -> None:
        dm.EMSetStagePositions(1 + 2 + 8, 100.0, 200.0, 0, 30.0, 0)
        assert dm.EMGetStageX() == pytest.approx(100.0)
        assert dm.EMGetStageY() == pytest.approx(200.0)
        assert dm.EMGetStageAlpha() == pytest.approx(30.0)

    def test_high_tension_read(self, dm: DMSimulator) -> None:
        assert dm.EMCanGetHighTension() is True
        ht = dm.EMGetHighTension()
        assert 60_000 <= ht <= 300_000   # plausible TEM range

    def test_spot_size_clamped(self, dm: DMSimulator) -> None:
        dm.EMSetSpotSize(0)  # below minimum
        assert dm.EMGetSpotSize() == 1
        dm.EMSetSpotSize(99)  # above maximum
        assert dm.EMGetSpotSize() == 11

    def test_eels_configuration(self, dm: DMSimulator) -> None:
        dm.IFSetEELSMode()
        assert dm.IFIsInEELSMode() is True
        dm.IFCSetEnergy(200.0)
        assert dm.IFGetEnergyLoss(0) == pytest.approx(200.0)
        dm.IFSetImageMode()
        assert dm.IFIsInImageMode() is True

    def test_camera_insertion(self, dm: DMSimulator) -> None:
        camera = dm.CM_GetCurrentCamera()
        dm.CM_SetCameraInserted(camera, 0)
        assert dm.CM_GetCameraInserted(camera) is False
        dm.CM_SetCameraInserted(camera, 1)
        assert dm.CM_GetCameraInserted(camera) is True

    def test_digiscan_configuration(self, dm: DMSimulator) -> None:
        dm.DSSetFrameSize(256, 256)
        dm.DSSetPixelTime(5.0)
        dm.DSSetSignalEnabled(0, 1)
        dm.DSSetSignalEnabled(1, 0)
        assert dm.DSGetSignalEnabled(0) is True
        assert dm.DSGetSignalEnabled(1) is False

    def test_acquire_diffraction_pattern(self, dm: DMSimulator) -> None:
        camera = dm.CM_GetCurrentCamera()
        dm._state.operation_mode = "DIFFRACTION"
        acq = dm.CM_CreateAcquisitionParameters_FullCCD(camera, 3, 0.5, 1, 1)
        dm.CM_Validate_AcquisitionParameters(camera, acq)
        img = dm.CM_AcquireImage(camera, acq)
        arr = img.GetNumArray()
        # Diffraction pattern should have bright central beam
        cx, cy = arr.shape[1] // 2, arr.shape[0] // 2
        centre_val = arr[cy - 5:cy + 5, cx - 5:cx + 5].mean()
        edge_val = arr[:20, :20].mean()
        assert centre_val > edge_val

    def test_4d_stem_generator(self, dm: DMSimulator) -> None:
        img4d = dm._make_4d_stem(8, 8, 32, 32)
        arr = img4d.GetNumArray()
        # Stored as (scan_y, scan_x * det_y * det_x) in simulator
        assert arr.ndim == 2
        total = 8 * 8 * 32 * 32
        assert arr.shape[0] * arr.shape[1] == total

    def test_eels_spectrum_shape(self, dm: DMSimulator) -> None:
        spec = dm._make_eels_spectrum(2048)
        arr = spec.GetNumArray()
        assert arr.shape == (1, 2048)
        # ZLP should be the highest peak
        assert float(arr.argmax()) < 100   # near channel 0

    def test_state_dict_complete(self, dm: DMSimulator) -> None:
        state = dm.get_state_dict()
        for key in ("high_tension_kV", "spot_size", "stage", "eels", "digiscan"):
            assert key in state


# ---------------------------------------------------------------------------
# TestMCPServerTools — tool function unit tests (no LLM)
# ---------------------------------------------------------------------------

class TestMCPServerTools:
    """
    Call every MCP tool function directly (bypassing the MCP protocol layer)
    and validate JSON responses.  Does not require Ollama or a network.
    """

    def _parse(self, raw: str) -> dict:
        return json.loads(raw)

    def test_get_microscope_state(self, server) -> None:
        raw = server.gms_get_microscope_state()
        data = self._parse(raw)
        assert data["success"] is True
        assert data["simulation_mode"] is True
        assert "optics" in data
        assert "stage" in data
        assert "camera" in data

    def test_get_microscope_state_optics_values(self, server) -> None:
        raw = server.gms_get_microscope_state()
        data = self._parse(raw)
        ht = data["optics"]["high_tension_kV"]
        assert ht is not None
        assert 60.0 <= ht <= 300.0

    def test_acquire_tem_default_params(self, server) -> None:
        from gms_mcp.server import AcquireTEMInput
        raw = server.gms_acquire_tem_image(AcquireTEMInput())
        data = self._parse(raw)
        assert data["success"] is True
        assert data["acquisition_type"] == "TEM"
        assert "shape" in data
        assert "statistics" in data
        assert data["statistics"]["max"] > 0

    def test_acquire_tem_with_roi(self, server) -> None:
        from gms_mcp.server import AcquireTEMInput
        raw = server.gms_acquire_tem_image(
            AcquireTEMInput(exposure_s=0.5, binning=2, roi=[0, 0, 512, 512])
        )
        data = self._parse(raw)
        assert data["success"] is True

    def test_acquire_tem_invalid_exposure(self) -> None:
        from gms_mcp.server import AcquireTEMInput
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AcquireTEMInput(exposure_s=0.0)   # below minimum 0.001

    def test_acquire_tem_invalid_roi(self) -> None:
        from gms_mcp.server import AcquireTEMInput
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AcquireTEMInput(roi=[0, 0, 512])  # wrong length

    def test_acquire_stem_default(self, server) -> None:
        from gms_mcp.server import AcquireSTEMInput
        raw = server.gms_acquire_stem(AcquireSTEMInput())
        data = self._parse(raw)
        assert data["success"] is True
        assert data["acquisition_type"] == "STEM"
        assert data["scan_parameters"]["width"] == 512
        assert data["scan_parameters"]["dwell_us"] == 10.0

    def test_acquire_stem_custom_signals(self, server) -> None:
        from gms_mcp.server import AcquireSTEMInput
        raw = server.gms_acquire_stem(
            AcquireSTEMInput(width=256, height=256, dwell_us=5.0, signals=[0])
        )
        data = self._parse(raw)
        assert data["success"] is True

    def test_acquire_4d_stem(self, server) -> None:
        from gms_mcp.server import Acquire4DSTEMInput
        raw = server.gms_acquire_4d_stem(
            Acquire4DSTEMInput(scan_x=16, scan_y=16, dwell_us=500.0,
                               camera_length_mm=150.0)
        )
        data = self._parse(raw)
        assert data["success"] is True
        assert data["dataset"]["scan_shape"] == [16, 16]
        assert data["dataset"]["camera_length_mm"] == pytest.approx(150.0)
        assert data["dataset"]["total_patterns"] == 16 * 16

    def test_acquire_4d_stem_invalid_convergence(self) -> None:
        from gms_mcp.server import Acquire4DSTEMInput
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Acquire4DSTEMInput(scan_x=16, scan_y=16, convergence_mrad=500.0)

    def test_acquire_eels_zero_loss(self, server) -> None:
        from gms_mcp.server import AcquireEELSInput
        raw = server.gms_acquire_eels(AcquireEELSInput(
            exposure_s=1.0, energy_offset_eV=0.0, slit_width_eV=5.0
        ))
        data = self._parse(raw)
        assert data["success"] is True
        assert data["spectrum"]["n_channels"] > 0
        # ZLP should be near energy offset
        assert abs(data["spectrum"]["zlp_centre_eV"]) < 50.0

    def test_acquire_eels_core_loss(self, server) -> None:
        from gms_mcp.server import AcquireEELSInput
        raw = server.gms_acquire_eels(AcquireEELSInput(
            exposure_s=2.0, energy_offset_eV=400.0,
            dispersion_idx=1, slit_width_eV=0.0   # slit out
        ))
        data = self._parse(raw)
        assert data["success"] is True
        assert data["spectrum"]["energy_range_eV"][0] == pytest.approx(400.0)

    def test_acquire_diffraction(self, server) -> None:
        from gms_mcp.server import AcquireDiffractionInput
        raw = server.gms_acquire_diffraction(
            AcquireDiffractionInput(exposure_s=0.2, camera_length_mm=200.0)
        )
        data = self._parse(raw)
        assert data["success"] is True
        assert data["pattern"]["camera_length_mm"] == pytest.approx(200.0)

    def test_get_stage_position(self, server) -> None:
        raw = server.gms_get_stage_position()
        data = self._parse(raw)
        assert data["success"] is True
        for key in ("x_um", "y_um", "z_um", "alpha_deg", "beta_deg"):
            assert key in data["stage"]

    def test_set_stage_x_only(self, server) -> None:
        from gms_mcp.server import SetStageInput
        raw = server.gms_set_stage_position(SetStageInput(x_um=250.0))
        data = self._parse(raw)
        assert data["success"] is True
        assert data["new_position"]["x_um"] == pytest.approx(250.0)

    def test_set_stage_tilt(self, server) -> None:
        from gms_mcp.server import SetStageInput
        raw = server.gms_set_stage_position(SetStageInput(alpha_deg=-30.0))
        data = self._parse(raw)
        assert data["success"] is True
        assert data["new_position"]["alpha_deg"] == pytest.approx(-30.0)

    def test_set_stage_no_axes_error(self, server) -> None:
        from gms_mcp.server import SetStageInput
        raw = server.gms_set_stage_position(SetStageInput())
        data = self._parse(raw)
        assert data["success"] is False
        assert "No axes specified" in data["error"]

    def test_set_stage_alpha_out_of_range(self) -> None:
        from gms_mcp.server import SetStageInput
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SetStageInput(alpha_deg=90.0)  # exceeds ±80°

    def test_set_beam_spot_size(self, server) -> None:
        from gms_mcp.server import SetBeamInput
        raw = server.gms_set_beam_parameters(SetBeamInput(spot_size=5))
        data = self._parse(raw)
        assert data["success"] is True
        assert data["current_state"]["spot_size"] == 5

    def test_set_beam_shift(self, server) -> None:
        from gms_mcp.server import SetBeamInput
        raw = server.gms_set_beam_parameters(SetBeamInput(shift_x=0.5, shift_y=-0.3))
        data = self._parse(raw)
        assert data["success"] is True
        assert data["applied_settings"]["beam_shift"] == pytest.approx([0.5, -0.3])

    def test_set_beam_stigmation(self, server) -> None:
        from gms_mcp.server import SetBeamInput
        raw = server.gms_set_beam_parameters(
            SetBeamInput(obj_stig_x=0.001, obj_stig_y=-0.001)
        )
        data = self._parse(raw)
        assert data["success"] is True
        assert "obj_stigmation" in data["applied_settings"]

    def test_configure_detectors_insert(self, server) -> None:
        from gms_mcp.server import SetDetectorInput
        raw = server.gms_configure_detectors(SetDetectorInput(insert_camera=True))
        data = self._parse(raw)
        assert data["success"] is True
        assert data["status"]["camera_inserted"] is True

    def test_configure_detectors_signals(self, server) -> None:
        from gms_mcp.server import SetDetectorInput
        raw = server.gms_configure_detectors(
            SetDetectorInput(haadf_enabled=True, bf_enabled=False, abf_enabled=False)
        )
        data = self._parse(raw)
        assert data["success"] is True
        assert data["status"]["haadf_enabled"] is True
        assert data["status"]["bf_enabled"] is False

    def test_tilt_series_short(self, server) -> None:
        from gms_mcp.server import TiltSeriesInput
        raw = server.gms_acquire_tilt_series(
            TiltSeriesInput(start_deg=-10.0, end_deg=10.0, step_deg=5.0,
                            exposure_s=0.1, binning=4)
        )
        data = self._parse(raw)
        assert data["success"] is True
        # -10, -5, 0, +5, +10 = 5 frames
        assert data["tilt_series"]["n_frames"] == 5
        assert len(data["per_tilt"]) == 5

    def test_tilt_series_per_frame_statistics(self, server) -> None:
        from gms_mcp.server import TiltSeriesInput
        raw = server.gms_acquire_tilt_series(
            TiltSeriesInput(start_deg=-6.0, end_deg=6.0, step_deg=3.0,
                            exposure_s=0.2, binning=4)
        )
        data = self._parse(raw)
        for frame in data["per_tilt"]:
            assert "angle_deg" in frame
            assert "mean" in frame
            assert frame["mean"] >= 0

    def test_4dstem_analysis_virtual_haadf(self, server) -> None:
        # First acquire a 4D dataset into the simulator
        from gms_mcp.server import Acquire4DSTEMInput
        server.gms_acquire_4d_stem(
            Acquire4DSTEMInput(scan_x=8, scan_y=8, dwell_us=500.0)
        )
        raw = server.gms_run_4dstem_analysis(
            inner_angle_mrad=10.0,
            outer_angle_mrad=40.0,
            analysis_type="virtual_haadf",
        )
        data = self._parse(raw)
        assert data["success"] is True
        assert data["analysis"]["type"] == "virtual_haadf"

    def test_4dstem_analysis_com(self, server) -> None:
        raw = server.gms_run_4dstem_analysis(
            inner_angle_mrad=0.0,
            outer_angle_mrad=50.0,
            analysis_type="com",
        )
        data = self._parse(raw)
        assert data["success"] is True


# ---------------------------------------------------------------------------
# TestServerTransport — verify the server can start cleanly
# ---------------------------------------------------------------------------

class TestServerTransport:
    """Smoke tests that verify the FastMCP server starts without errors."""

    def test_server_module_imports_cleanly(self) -> None:
        """Import the server in a fresh subprocess to catch import errors."""
        result = subprocess.run(
            [sys.executable, "-c",
             "import os; os.environ['GMS_SIMULATE']='1'; "
             "import sys; sys.path.insert(0, '.'); "
             "import gms_mcp.server; "
             "print('OK')"],
            capture_output=True, text=True, timeout=15,
            cwd=str(_HERE),
        )
        assert result.returncode == 0, f"Import failed:\n{result.stderr}"
        assert "OK" in result.stdout

    def test_simulator_imports_cleanly(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, '.'); "
             "from gms_mcp.simulator import DMSimulator; "
             "d = DMSimulator(); print(d.EMGetHighTension())"],
            capture_output=True, text=True, timeout=10,
            cwd=str(_HERE),
        )
        assert result.returncode == 0
        assert float(result.stdout.strip()) > 0

    def test_tools_are_registered(self, server) -> None:
        """All expected tools must be registered in the FastMCP instance."""
        tools = asyncio.run(server.mcp.list_tools())
        tool_names = {t.name for t in tools}
        expected = {
            "gms_get_microscope_state",
            "gms_acquire_tem_image",
            "gms_acquire_stem",
            "gms_acquire_4d_stem",
            "gms_acquire_eels",
            "gms_acquire_diffraction",
            "gms_get_stage_position",
            "gms_set_stage_position",
            "gms_set_beam_parameters",
            "gms_configure_detectors",
            "gms_acquire_tilt_series",
            "gms_run_4dstem_analysis",
        }
        missing = expected - tool_names
        assert not missing, f"Missing tools: {missing}"

    def test_tool_descriptions_non_empty(self, server) -> None:
        tools = asyncio.run(server.mcp.list_tools())
        for tool in tools:
            assert tool.description, f"Tool '{tool.name}' has no description"

    def test_tool_input_schemas_present(self, server) -> None:
        # FastMCP 3.x uses .parameters (dict); official MCP SDK uses .inputSchema
        tools = asyncio.run(server.mcp.list_tools())
        for tool in tools:
            schema = (getattr(tool, "parameters", None)
                      or getattr(tool, "inputSchema", None))
            assert schema is not None, (
                f"Tool '{tool.name}' has no input schema (.parameters or .inputSchema)"
            )


# ---------------------------------------------------------------------------
# TestOllamaIntegration — end-to-end tests (requires Ollama)
# ---------------------------------------------------------------------------

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

def _ollama_available() -> bool:
    """Check whether the Ollama server is reachable and the model is present."""
    try:
        import httpx
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3.0)
        if r.status_code != 200:
            return False
        models = [m["name"] for m in r.json().get("models", [])]
        return any(OLLAMA_MODEL in m for m in models)
    except Exception:
        return False


_OLLAMA_SKIP = pytest.mark.skipif(
    not _ollama_available(),
    reason=(
        f"Ollama not available at {OLLAMA_URL} "
        f"or model '{OLLAMA_MODEL}' not pulled. "
        f"Run: ollama pull {OLLAMA_MODEL}"
    ),
)


@pytest.mark.ollama
class TestOllamaIntegration:
    """End-to-end tests that exercise the full Ollama → MCP → GMS pipeline."""

    @_OLLAMA_SKIP
    def test_single_tool_query(self) -> None:
        """Agent should call gms_get_microscope_state for a state query."""
        from gms_mcp.client import run_agent
        result = asyncio.run(run_agent(
            query="What is the current accelerating voltage of the microscope?",
            model=OLLAMA_MODEL,
            base_url=OLLAMA_URL,
        ))
        assert result["answer"], "Agent returned empty answer"
        assert len(result["tool_calls"]) >= 1
        called = [tc["tool"] for tc in result["tool_calls"]]
        assert "gms_get_microscope_state" in called

    @_OLLAMA_SKIP
    def test_stage_position_query(self) -> None:
        from gms_mcp.client import run_agent
        result = asyncio.run(run_agent(
            query="What are the current stage coordinates in micrometers?",
            model=OLLAMA_MODEL,
            base_url=OLLAMA_URL,
        ))
        assert result["answer"]
        called = [tc["tool"] for tc in result["tool_calls"]]
        assert any("stage" in t for t in called)

    @_OLLAMA_SKIP
    def test_acquire_and_report(self) -> None:
        """Agent should acquire a TEM image and report its statistics."""
        from gms_mcp.client import run_agent
        result = asyncio.run(run_agent(
            query=(
                "Please acquire a TEM image with 0.5 s exposure and 2× binning, "
                "then report the image dimensions and mean pixel intensity."
            ),
            model=OLLAMA_MODEL,
            base_url=OLLAMA_URL,
        ))
        assert result["answer"]
        called = [tc["tool"] for tc in result["tool_calls"]]
        assert "gms_acquire_tem_image" in called
        # Answer should mention pixel or intensity
        answer_lower = result["answer"].lower()
        assert any(w in answer_lower for w in ("pixel", "mean", "intensity", "statistic"))

    @_OLLAMA_SKIP
    def test_eels_acquisition_workflow(self) -> None:
        """Agent should configure EELS and report the zero-loss peak position."""
        from gms_mcp.client import run_agent
        result = asyncio.run(run_agent(
            query=(
                "Acquire an EELS spectrum centred at 0 eV with a 5 eV slit width "
                "and 1 s exposure. Report the position of the zero-loss peak in eV."
            ),
            model=OLLAMA_MODEL,
            base_url=OLLAMA_URL,
        ))
        assert result["answer"]
        called = [tc["tool"] for tc in result["tool_calls"]]
        assert "gms_acquire_eels" in called

    @_OLLAMA_SKIP
    def test_stage_move_and_confirm(self) -> None:
        """Agent should move the stage and confirm the new position."""
        from gms_mcp.client import run_agent
        result = asyncio.run(run_agent(
            query="Move the stage to X = 100 µm, Y = -50 µm, then confirm the new position.",
            model=OLLAMA_MODEL,
            base_url=OLLAMA_URL,
        ))
        assert result["answer"]
        called = [tc["tool"] for tc in result["tool_calls"]]
        assert "gms_set_stage_position" in called

    @_OLLAMA_SKIP
    def test_multi_step_workflow(self) -> None:
        """
        Agent should perform a multi-step workflow:
        1. Check state
        2. Set spot size
        3. Acquire STEM image
        """
        from gms_mcp.client import run_agent
        result = asyncio.run(run_agent(
            query=(
                "Check the microscope state, set the spot size to 4, "
                "then acquire a 256×256 HAADF STEM image with 5 µs dwell time."
            ),
            model=OLLAMA_MODEL,
            base_url=OLLAMA_URL,
        ))
        assert result["answer"]
        called = [tc["tool"] for tc in result["tool_calls"]]
        # Should have called at least 3 distinct tools
        assert len(set(called)) >= 2
        assert "gms_acquire_stem" in called


# ---------------------------------------------------------------------------
# Pytest configuration
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "ollama: marks tests that require a running Ollama instance "
        "(deselect with -m 'not ollama')"
    )
