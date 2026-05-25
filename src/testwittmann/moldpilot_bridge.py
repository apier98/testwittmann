from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MoldPilotImports:
    BaumerCameraRuntimeConfig: Any
    StaticNetworkConfig: Any
    CameraProfile: Any
    BaumerCameraService: Any
    OnnxInferenceService: Any
    ComponentTracker: Any
    DefectTracker: Any
    SeverityEngine: Any
    apply_machine_camera_bindings: Any
    load_machine_camera_bindings: Any


def load_moldpilot_imports() -> MoldPilotImports:
    src_path = _resolve_moldpilot_src_path()
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from aria_moldpilot.domain.camera import BaumerCameraRuntimeConfig, StaticNetworkConfig
    from aria_moldpilot.domain.configuration import CameraProfile
    from aria_moldpilot.infrastructure.baumer_camera import BaumerCameraService
    from aria_moldpilot.infrastructure.camera_binding_store import (
        apply_machine_camera_bindings,
        load_machine_camera_bindings,
    )
    from aria_moldpilot.infrastructure.iou_tracker import ComponentTracker, DefectTracker
    from aria_moldpilot.infrastructure.onnx_inference import OnnxInferenceService
    from aria_moldpilot.infrastructure.severity_engine import SeverityEngine

    return MoldPilotImports(
        BaumerCameraRuntimeConfig=BaumerCameraRuntimeConfig,
        StaticNetworkConfig=StaticNetworkConfig,
        CameraProfile=CameraProfile,
        BaumerCameraService=BaumerCameraService,
        OnnxInferenceService=OnnxInferenceService,
        ComponentTracker=ComponentTracker,
        DefectTracker=DefectTracker,
        SeverityEngine=SeverityEngine,
        apply_machine_camera_bindings=apply_machine_camera_bindings,
        load_machine_camera_bindings=load_machine_camera_bindings,
    )


def _resolve_moldpilot_src_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    moldpilot_root = repo_root.parent / "ARIA_MoldPilot"
    src_path = moldpilot_root / "src"
    if not src_path.exists():
        raise RuntimeError(
            "Unable to locate the sibling ARIA_MoldPilot source tree at "
            f"{src_path}"
        )
    return src_path
