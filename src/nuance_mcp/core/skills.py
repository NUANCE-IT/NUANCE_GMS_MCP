"""Vendor-portable skills (MCP prompts).

Each skill is a declarative, parameterised protocol. Skills reference the
*generic* tool names defined in :mod:`nuance_mcp.tools` (e.g.\
``acquire_eels``, ``get_microscope_state``) — never vendor-prefixed ones.
The same skill therefore unrolls into identical tool calls regardless of
which adapter is mounted.

Skills are best-effort: they include capability checks so that an adapter
that does not advertise (say) EELS receives a clear ``CapabilityUnavailable``
when the skill tries to call ``acquire_eels``.
"""

from __future__ import annotations

from fastmcp import FastMCP


def register_skills(mcp: FastMCP) -> None:
    @mcp.prompt(
        name="eels_survey",
        description="Full EELS characterisation: ZLP + core-loss + edge ID.",
    )
    def eels_survey(material: str = "unknown", core_loss_eV: str = "500") -> str:
        loss = int(core_loss_eV)
        return f"""You are running a full EELS survey on a specimen of {material}.

Step 1 — Verify capabilities
  Call: get_capabilities
  Check: 'eels' must be present. If not, stop and report.

Step 2 — Verify instrument state
  Call: get_microscope_state
  Check: mode must include EELS readiness.

Step 3 — Acquire ZLP reference
  Call: acquire_eels
    energy_offset_eV: 0
    slit_width_eV: 0
    exposure_s: 0.1
    dispersion_idx: 0
    full_vertical_binning: true
  Record ZLP FWHM as the energy resolution estimate.

Step 4 — Acquire core-loss spectrum at {loss} eV
  Call: acquire_eels
    energy_offset_eV: {loss}
    exposure_s: 1.0
    dispersion_idx: 1
    full_vertical_binning: true
  Identify peaks above 3σ.

Step 5 — Report
  Summarize ZLP FWHM, core-loss range, detected edges, tentative element
  assignments for {material}, recommended next step.
"""

    @mcp.prompt(
        name="tilt_series_protocol",
        description="Automated tilt series with pre/post quality checks.",
    )
    def tilt_series(
        start_deg: str = "-60",
        end_deg: str = "60",
        step_deg: str = "2",
        save_dir: str = "",
    ) -> str:
        return f"""Run a tomographic tilt series from {start_deg}° to {end_deg}°
in {step_deg}° steps.

Step 1 — Verify capabilities: 'tilt_series' must be present.
Step 2 — Pre-flight: call get_microscope_state and get_stage_position;
         confirm specimen is at α=0° and centred.
Step 3 — Call acquire_tilt_series with the requested range, step, exposure.
         If save_dir is non-empty, save individual frames there: {save_dir!r}.
Step 4 — Post-flight: re-read state; report frame count, mean intensity
         stability across tilt, and any per-frame anomalies.
"""

    @mcp.prompt(
        name="4dstem_characterization",
        description="vBF/HAADF + CoM + DPC + (optional) orientation map.",
    )
    def fourdstem(
        scan_size: str = "64", material: str = "unknown", convergence_mrad: str = "10"
    ) -> str:
        return f"""Run a 4D-STEM characterisation on {material}.

Step 1 — Verify capabilities: '4dstem' and 'analysis.com_dpc'.
Step 2 — Call get_microscope_state.
Step 3 — Call acquire_4d_stem with scan_x={scan_size}, scan_y={scan_size},
         convergence_mrad={convergence_mrad}.
Step 4 — Call run_4dstem_analysis to produce vBF, HAADF, CoM, DPC.
Step 5 — If material has expected symmetry, call
         run_4dstem_maximum_spot_mapping.
Step 6 — Report key statistics for each derived map.
"""

    @mcp.prompt(
        name="beam_alignment",
        description="Systematic beam centring, stigmation, focus verification.",
    )
    def beam_alignment() -> str:
        return """Step 1 — Call get_microscope_state.
Step 2 — Centre the beam using set_image_shift/set_beam_parameters as needed.
Step 3 — Inspect a HRTEM frame and its FFT (acquire_tem then compute_max_fft).
Step 4 — If FFT shows astigmatism, apply small set_condenser_stigmation
         adjustments and re-image.
Step 5 — Report final focus, stigmation, and FFT isotropy estimate.
"""

    @mcp.prompt(
        name="hrtem_imaging",
        description="Survey → HRTEM → FFT → d-spacing extraction.",
    )
    def hrtem(material: str = "unknown", zone_axis: str = "") -> str:
        return f"""HRTEM survey on {material} (zone axis: {zone_axis or "any"}).

Step 1 — get_microscope_state.
Step 2 — acquire_tem at low magnification for context.
Step 3 — Acquire a HRTEM frame at high magnification, short exposure.
Step 4 — compute_radial_profile mode='fft' to extract d-spacings.
Step 5 — Match against known phases of {material}; report best match.
"""

    @mcp.prompt(
        name="diffraction_survey",
        description="Diffraction pattern + radial profile + phase ID.",
    )
    def diffraction(material: str = "unknown", camera_length_mm: str = "300") -> str:
        return f"""Crystallographic survey on {material}.

Step 1 — Verify 'diffraction' capability.
Step 2 — get_microscope_state.
Step 3 — acquire_diffraction with camera_length_mm={camera_length_mm}.
Step 4 — compute_radial_profile mode='diffraction'.
Step 5 — Compare ring positions to {material} reference d-spacings, report
         the best phase match and any unindexed rings.
"""
