from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from testwittmann.constants import (
    DEFAULT_BUNDLE_RELATIVE_PATH,
    DEFAULT_RUNTIME_ROOT_SUFFIX,
    HARDCODED_CONFIGURATION_ID,
    HARDCODED_CONFIGURATION_VERSION,
    HARDCODED_MODEL_BUNDLE_ID,
)


@dataclass(frozen=True)
class CameraProfileSettings:
    exposure_time_us: float
    gain_db: float
    frame_rate_fps: float
    pixel_format: str
    trigger_mode: str
    roi_width: int
    roi_height: int
    offset_x: int
    offset_y: int


@dataclass(frozen=True)
class RuntimeAssets:
    runtime_root: Path
    config_id: str
    config_version: str
    model_bundle_id: str
    process_camera_profile: CameraProfileSettings
    model_bundle_path: Path
    configs_root: Path


def resolve_runtime_root() -> Path:
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata) / DEFAULT_RUNTIME_ROOT_SUFFIX
    return Path.home() / ".aria" / "moldpilot"


def load_runtime_assets(runtime_root: Path | None = None) -> RuntimeAssets:
    root = runtime_root or resolve_runtime_root()
    configs_root = root / "configs"
    published_path = configs_root / "published_configurations.json"
    registry_path = root / "models" / "registry.json"
    bundle_path = root / "models" / DEFAULT_BUNDLE_RELATIVE_PATH

    published_payload = _read_json_object(published_path)
    items = published_payload.get("items", [])
    if not isinstance(items, list):
        raise RuntimeError(f"Invalid published configurations payload: {published_path}")

    config_item = next(
        (
            item
            for item in items
            if isinstance(item, dict)
            and isinstance(item.get("reference"), dict)
            and item["reference"].get("configuration_id") == HARDCODED_CONFIGURATION_ID
        ),
        None,
    )
    if config_item is None:
        raise RuntimeError(
            "Hardcoded MoldPilot camera configuration was not found: "
            f"{HARDCODED_CONFIGURATION_ID}"
        )

    process_profile_payload = config_item.get("process_camera_profile")
    if not isinstance(process_profile_payload, dict):
        raise RuntimeError(
            "The hardcoded MoldPilot configuration is missing process_camera_profile: "
            f"{HARDCODED_CONFIGURATION_ID}"
        )

    registry_payload = _read_json_object(registry_path)
    bundles = registry_payload.get("bundles", [])
    if not isinstance(bundles, list):
        raise RuntimeError(f"Invalid model registry payload: {registry_path}")

    if not any(
        isinstance(entry, dict) and entry.get("bundle_id") == HARDCODED_MODEL_BUNDLE_ID
        for entry in bundles
    ):
        raise RuntimeError(
            "Hardcoded defect detector bundle was not found in the MoldPilot registry: "
            f"{HARDCODED_MODEL_BUNDLE_ID}"
        )

    if not bundle_path.exists():
        raise RuntimeError(f"Detector bundle path does not exist: {bundle_path}")

    return RuntimeAssets(
        runtime_root=root,
        config_id=HARDCODED_CONFIGURATION_ID,
        config_version=HARDCODED_CONFIGURATION_VERSION,
        model_bundle_id=HARDCODED_MODEL_BUNDLE_ID,
        process_camera_profile=CameraProfileSettings(
            exposure_time_us=float(process_profile_payload["exposure_time_us"]),
            gain_db=float(process_profile_payload["gain_db"]),
            frame_rate_fps=float(process_profile_payload["frame_rate_fps"]),
            pixel_format=str(process_profile_payload["pixel_format"]),
            trigger_mode=str(process_profile_payload["trigger_mode"]),
            roi_width=int(process_profile_payload["roi_width"]),
            roi_height=int(process_profile_payload["roi_height"]),
            offset_x=int(process_profile_payload["offset_x"]),
            offset_y=int(process_profile_payload["offset_y"]),
        ),
        model_bundle_path=bundle_path,
        configs_root=configs_root,
    )


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required MoldPilot runtime file was not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in MoldPilot runtime file: {path}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in: {path}")
    return payload
