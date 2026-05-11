# Tool reference

All 30 typed tools, grouped by domain. Every tool validates its
argument object against a Pydantic v2 schema **before** the adapter is
touched. Out-of-bounds arguments raise `ValidationError` at the
server; capability mismatches return `{"status": "UNSUPPORTED",
"reason": "..."}` from the tool itself.

`Effectful` tools mutate instrument or simulator state or produce
derived images; `Read-only` tools do not.

## Diagnostics (4)

| Tool                       | Type      | Purpose                                              |
|----------------------------|-----------|------------------------------------------------------|
| `get_microscope_state`     | Read-only | Snapshot of column, optics, stage, detectors         |
| `get_capabilities`         | Read-only | Adapter vendor/model and advertised capability list  |
| `get_front_image`          | Read-only | Inspect the current front image (metadata, tags)     |
| `get_image_shift`          | Read-only | Read calibrated image-shift coils                    |
| `workspace_list_images`    | Read-only | Enumerate images in the vendor workspace             |

## Acquisition (5)

| Tool                       | Type      | Bounds (input)                                                                                  |
|----------------------------|-----------|-------------------------------------------------------------------------------------------------|
| `acquire_tem_image`        | Effectful | `exposure_s ∈ [10⁻³, 60]`, `binning ∈ {1..8}`, `processing ∈ {1,2,3}`                          |
| `acquire_stem`             | Effectful | `width, height ∈ [64, 4096]`, `dwell_us ∈ [0.5, 10000]`                                        |
| `acquire_4d_stem`          | Effectful | `scan_x, scan_y ∈ [8, 512]`, `dwell_us ∈ [100, 10⁵]`                                           |
| `acquire_eels`             | Effectful | `exposure_s ∈ [10⁻³, 60]`, `energy_offset_eV ∈ [-200, 3000]`, `dispersion_idx ∈ {0..3}`        |
| `acquire_diffraction`      | Effectful | `exposure_s ∈ [10⁻³, 60]`, `camera_length_mm ∈ [20, 2000]`                                     |

## Stage / Optics / Beam (10)

| Tool                            | Type      | Bounds (input)                                              |
|---------------------------------|-----------|-------------------------------------------------------------|
| `get_stage_position`            | Read-only | —                                                           |
| `set_stage_position`            | Effectful | `α ∈ [-80°, +80°]`, `β ∈ [-30°, +30°]`, `x,y ∈ [-5000, 5000] µm` |
| `set_beam_parameters`           | Effectful | `spot_size ∈ {1..11}`                                       |
| `configure_detectors`           | Effectful | `target_temp_c ∈ [-60, 30]`                                 |
| `set_magnification`             | Effectful | `magnification ∈ [10, 2·10⁶]`                               |
| `set_image_shift`               | Effectful | calibrated units                                            |
| `set_brightness`                | Effectful | `value ∈ [0, 1]`                                            |
| `change_focus_relative`         | Effectful | `delta_um ∈ [-100, +100]`                                   |
| `stop_stage`                    | Effectful | —                                                           |
| `set_condenser_stigmation`      | Effectful | `x, y ∈ [-1, 1]`                                            |

## Workflow / lifecycle (5)

| Tool                                  | Type      | Purpose                                                    |
|---------------------------------------|-----------|------------------------------------------------------------|
| `acquire_tilt_series`                 | Effectful | Automated tomographic tilt series                          |
| `start_live_processing_job`           | Effectful | Begin a persistent live job (5 types — see below)          |
| `get_live_processing_job_status`      | Read-only | Poll job state                                             |
| `get_live_processing_job_result`      | Read-only | Retrieve latest derived payload                            |
| `stop_live_processing_job`            | Effectful | Terminate a live job                                       |

Live-job types: `radial_profile`, `difference`, `fft_map`,
`filtered_view`, `maximum_spot_mapping`.

## Derived analyses (6)

| Tool                                   | Type      | Purpose                                                |
|----------------------------------------|-----------|--------------------------------------------------------|
| `apply_image_filter`                   | Effectful | Median + Gaussian derived image                        |
| `compute_radial_profile`               | Read-only | 1-D radial profile from diffraction or FFT             |
| `compute_max_fft`                      | Effectful | Local maximum-FFT map over the front image             |
| `run_4dstem_analysis`                  | Read-only | Virtual BF/HAADF + CoM + DPC                           |
| `run_4dstem_maximum_spot_mapping`      | Effectful | RGB max-spot map (θ or r) from a 4D-STEM dataset       |
| `run_script_template`                  | Effectful | Run a named DM-script template (Gatan adapter only)    |

## Bounded execution at the boundary

Every tool returns one of three outcomes:

1. **`{"status": "OK", "result": ...}`** — the adapter returned a
   structured payload.
2. **`{"status": "UNSUPPORTED", "reason": "..."}`** — the adapter does
   not declare the capability.
3. **`ValidationError`** — the argument object failed the Pydantic
   schema; no adapter call was made.

These outcomes are independent of how the calling agent phrased the
request: a malformed alpha-tilt is rejected the same way whether the
agent typed `alpha_deg=95.0` or got there via a multi-step skill.
This is what we mean by "bounded execution at the tool boundary"; see
[`safety.md`](safety.md) for the broader safety discussion.
