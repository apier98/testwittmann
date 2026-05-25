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

Suggestion-model output is not implemented yet.

## Runtime assumptions

This project currently expects the existing MoldPilot runtime data to be present under:

`C:\Users\<your-user>\AppData\Local\ARIA\MoldPilot`

It reads:

- `configs\published_configurations.json`
- `models\registry.json`
- `models\bundles\Surface_Defect_Detector-v1.0.2`

It also imports runtime code from the sibling repository:

`C:\Users\aria-\dev\ARIA_MoldPilot`

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

You can also override the detector score threshold:

```powershell
uv run testwittmann --score-threshold 0.4
```

## Controls

- `q` or `Esc`: stop the app
- `r`: restart the current replay from frame 0

## Current flow

1. Start the process camera using the hardcoded MoldPilot camera profile.
2. Show the live stream in an OpenCV window.
3. Buffer frames and trigger a segment when the contrast-based presence signal crosses the threshold.
4. Run buffered ONNX inference on that captured segment.
5. Play the inferred segment back in the replay window with detection overlays.
