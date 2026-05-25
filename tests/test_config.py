from __future__ import annotations

import json
from pathlib import Path

from testwittmann.config import load_runtime_assets


def test_load_runtime_assets_uses_hardcoded_configuration_and_bundle(tmp_path: Path) -> None:
    runtime_root = tmp_path / "ARIA" / "MoldPilot"
    configs_root = runtime_root / "configs"
    models_root = runtime_root / "models"
    bundle_root = models_root / "bundles" / "Surface_Defect_Detector-v1.0.2"

    configs_root.mkdir(parents=True)
    bundle_root.mkdir(parents=True)

    (configs_root / "published_configurations.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "reference": {
                            "configuration_id": "cfg_wittman__stampo_giotto__giotto__nomaterial_v0_1_3",
                        },
                        "process_camera_profile": {
                            "exposure_time_us": 1000.0,
                            "gain_db": 30.0,
                            "frame_rate_fps": 39.0,
                            "pixel_format": "BayerRG8",
                            "trigger_mode": "Off",
                            "roi_width": 2048,
                            "roi_height": 1536,
                            "offset_x": 0,
                            "offset_y": 0,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    (models_root / "registry.json").write_text(
        json.dumps(
            {
                "bundles": [
                    {
                        "bundle_id": "Surface_Defect_Detector-v1.0.2",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assets = load_runtime_assets(runtime_root=runtime_root)

    assert assets.config_id == "cfg_wittman__stampo_giotto__giotto__nomaterial_v0_1_3"
    assert assets.model_bundle_id == "Surface_Defect_Detector-v1.0.2"
    assert assets.process_camera_profile.pixel_format == "BayerRG8"
    assert assets.model_bundle_path == bundle_root
