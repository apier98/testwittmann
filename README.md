# TestWittmann

CLI test project for the current MoldPilot-like live monitoring flow.

## What it does

- Uses the same Baumer camera service from the sibling `ARIA_MoldPilot` project
- Uses the same ONNX inference service and the latest MoldPilot defect detector bundle
- Hardcodes the MoldPilot local runtime inputs:
  - configuration: `cfg_wittman__stampo_giotto__giotto__nomaterial_v0_1_3`
  - model bundle: `Surface_Defect_Detector-v1.0.2`
- Shows:
  - a live stream window
  - a replay window after a triggered segment has been captured and inferred
  - per-defect frame ratios on the replay overlay (`frames_with_defect / component_span_frames`)
  - a bundled suggestion-model step that pauses monitoring after each processed component when a supported defect is detected
- **Global Sensitivity:** Displays global parameter trends alongside local slopes to prevent optimization stalls (marked with `*` when local gradients vanish).
- **Simulation Mode:** Run the entire decision logic offline without a camera or detector.

## Runtime assumptions

This project currently expects the existing MoldPilot runtime data to be present under:

`C:\Users\<your-user>\AppData\Local\ARIA\MoldPilot`

It reads:

- `configs\published_configurations.json`
- `models\registry.json`
- `models\bundles\Surface_Defect_Detector-v1.0.2`

It also imports runtime code from the sibling repository:

`C:\Users\aria-\dev\ARIA_MoldPilot`

The suggestion model is bundled directly in this package under `src\testwittmann\assets\suggestion_model`, so it ships with the CLI instead of depending on a separate runtime folder.

## How to run

From the project root:

```powershell
Set-Location C:\Users\aria-\dev\TestWittmann
uv sync --group dev
uv run testwittmann
```

## CLI options

Show the available options with:

```powershell
uv run testwittmann --help
```

Current useful flags:

```powershell
uv run testwittmann --trigger-threshold 0.05 --trigger-frames 3 --pre-roll 10 --post-roll 10 --max-segment-frames 160
```

The CLI uses separate per-class score thresholds by default:

```powershell
uv run testwittmann --component-score-threshold 0.7 --defect-score-threshold 0.45
```

For meaningful suggestions, also pass the current process setpoints:

```powershell
uv run testwittmann --t-melt 260 --t-mold 65 --inj-speed 45 --pack-pressure 750
```

### Simulation Mode (Offline Testing)
Run the entire decision logic manually without hardware:
```bash
uv run testwittmann --simulation
```
- Enter mock defect ratios when prompted (e.g. `sink_mark=0.4`).
- Interact with the terminal to test suggestions and slopes.

## Controls

- `q` or `Esc`: stop the app
- `r`: restart the current replay from frame 0
- terminal `1`, `2`, `3`, ...: choose the defect, choose parameters, then choose one of the first 3 suggestions
- terminal multi-select like `1,3` or `all`: choose which process parameters the suggestion engine may vary
- terminal custom numeric target: choose the defect-ratio threshold to reach for the current round
- terminal `b`: go back to the previous decision step
- terminal `s`: skip the current component and resume monitoring without applying a suggestion

## Current flow

1. Start the process camera using the hardcoded MoldPilot camera profile.
2. Show the live stream in an OpenCV window.
3. Buffer frames and trigger a segment when the contrast-based presence signal crosses the threshold.
4. Run buffered ONNX inference on that captured segment.
5. Compute per-defect frame ratios as `frames_with_defect / component_span_frames`, without any tracking.
6. If a supported defect is present, pause capture/inference on the current component and let the operator use the terminal shell to:
   - pick which detected defect to adjust
   - see the local parameter slopes vs. **Global Trends** in `x0`
   - choose one or more parameters that the model is allowed to vary
   - choose the target defect-ratio threshold for that round
   - search only inside the notebook-style local delta window, clipped to model bounds:
     `t_mold ±5`, `t_melt ±5`, `pack_pressure ±100`, `inj_speed ±10`
   - review the first 3 ranked process-setting suggestions from the bundled model
   - accept one suggestion or skip the round
7. Resume capture/inference after the operator decides; accepted suggestions become the next baseline setpoints for future rounds.
