from __future__ import annotations

from pathlib import Path

WINDOW_NAME_LIVE = "TestWittmann Live"
WINDOW_NAME_REPLAY = "TestWittmann Replay"

HARDCODED_CONFIGURATION_ID = "cfg_wittman__stampo_giotto__giotto__nomaterial_v0_1_3"
HARDCODED_CONFIGURATION_VERSION = "v0.1.3"
HARDCODED_MODEL_BUNDLE_ID = "Surface_Defect_Detector-v1.0.2"

DEFAULT_RUNTIME_ROOT_SUFFIX = Path("ARIA") / "MoldPilot"
DEFAULT_BUNDLE_RELATIVE_PATH = Path("bundles") / HARDCODED_MODEL_BUNDLE_ID
