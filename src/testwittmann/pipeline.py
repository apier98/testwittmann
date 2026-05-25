from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class TriggerConfig:
    threshold: float = 0.05
    consecutive_frames: int = 3
    pre_roll_count: int = 10
    post_roll_count: int = 10
    max_active_frames: int = 160


@dataclass(frozen=True)
class CapturedSegment:
    frames: tuple[np.ndarray, ...]
    timestamps: tuple[float, ...]
    trigger_metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalyzedSegment:
    replay_frames: tuple[np.ndarray, ...]
    detections: tuple[tuple[Any, ...], ...]
    timestamps: tuple[float, ...]
    total_detections: int
    total_defects: int
    frame_ratios: dict[str, float] = field(default_factory=dict)
    overall_severity: float = 0.0
    primary_component_track_id: int | None = None


class SegmentTrigger:
    def __init__(self, config: TriggerConfig) -> None:
        self._config = config
        self._pre_roll: deque[tuple[float, np.ndarray]] = deque(maxlen=config.pre_roll_count)
        self._active_frames: list[np.ndarray] = []
        self._active_timestamps: list[float] = []
        self._is_capturing = False
        self._consecutive_present = 0
        self._post_roll_countdown = 0

    def push_frame(self, timestamp: float, frame: np.ndarray) -> CapturedSegment | None:
        value = _contrast_signal(frame)
        is_present = value > self._config.threshold

        if is_present:
            self._consecutive_present += 1
        else:
            self._consecutive_present = 0

        if not self._is_capturing:
            if self._consecutive_present >= self._config.consecutive_frames:
                self._is_capturing = True
                self._active_frames = [stored_frame.copy() for _, stored_frame in self._pre_roll]
                self._active_timestamps = [stored_ts for stored_ts, _ in self._pre_roll]
                self._active_frames.append(frame.copy())
                self._active_timestamps.append(timestamp)
            else:
                self._pre_roll.append((timestamp, frame.copy()))
            return None

        self._active_frames.append(frame.copy())
        self._active_timestamps.append(timestamp)

        if len(self._active_frames) >= self._config.max_active_frames:
            return self._finalize_segment(value, forced_split=True)

        if self._consecutive_present == 0:
            if self._post_roll_countdown == 0:
                self._post_roll_countdown = self._config.post_roll_count
            self._post_roll_countdown -= 1
            if self._post_roll_countdown <= 0:
                return self._finalize_segment(value, forced_split=False)
        else:
            self._post_roll_countdown = 0

        return None

    def reset(self) -> None:
        self._pre_roll.clear()
        self._active_frames = []
        self._active_timestamps = []
        self._is_capturing = False
        self._consecutive_present = 0
        self._post_roll_countdown = 0

    def _finalize_segment(self, value: float, *, forced_split: bool) -> CapturedSegment:
        segment = CapturedSegment(
            frames=tuple(self._active_frames),
            timestamps=tuple(self._active_timestamps),
            trigger_metadata={
                "mean_intensity": float(value),
                "forced_split": forced_split,
            },
        )
        tail = self._active_frames[-self._config.pre_roll_count :]
        tail_ts = self._active_timestamps[-self._config.pre_roll_count :]
        self._pre_roll = deque(
            [(ts, frame.copy()) for ts, frame in zip(tail_ts, tail, strict=False)],
            maxlen=self._config.pre_roll_count,
        )
        self._active_frames = []
        self._active_timestamps = []
        self._is_capturing = False
        self._consecutive_present = 0
        self._post_roll_countdown = 0
        return segment


def analyze_segment(
    segment: CapturedSegment,
    inference_service: Any,
    *,
    tracking_imports: Any,
    pixel_format: str,
    component_score_threshold: float = 0.7,
    defect_score_threshold: float = 0.45,
    replay_height: int = 480,
) -> AnalyzedSegment:
    del tracking_imports
    replay_frames: list[np.ndarray] = []
    detections_per_frame: list[tuple[Any, ...]] = []
    total_detections = 0
    total_defects = 0
    frame_records: list[dict[str, Any]] = []

    defect_labels = _defect_labels_from_inference_service(inference_service)

    for frame_index, frame in enumerate(segment.frames):
        frame_bgr = frame_to_bgr(frame, pixel_format=pixel_format)
        if frame_bgr is None:
            continue

        tensor, lbmeta = inference_service.preprocess(frame_bgr)
        output_map = inference_service.run_session(tensor)
        frame_h, frame_w = frame_bgr.shape[:2]
        detections = tuple(
            _filter_detections_by_class_thresholds(
                inference_service.postprocess(output_map, frame_w, frame_h, lbmeta),
                component_score_threshold=component_score_threshold,
                defect_score_threshold=defect_score_threshold,
            )
        )
        total_detections += len(detections)
        total_defects += sum(1 for detection in detections if getattr(detection, "class_id", -1) > 0)
        detections_per_frame.append(detections)
        replay_frames.append(_resize_for_replay(_draw_detections(frame_bgr, detections), replay_height))
        frame_records.append(_build_metric_frame_record(frame_index=frame_index, detections=detections))

    primary_summary = summarize_frame_ratios(
        frame_records=frame_records,
        defect_labels=defect_labels,
    )

    return AnalyzedSegment(
        replay_frames=tuple(replay_frames),
        detections=tuple(detections_per_frame),
        timestamps=segment.timestamps,
        total_detections=total_detections,
        total_defects=total_defects,
        frame_ratios=primary_summary.frame_ratios,
        overall_severity=primary_summary.overall_severity,
        primary_component_track_id=primary_summary.component_track_id,
    )


def frame_to_display(frame: object, *, pixel_format: str, max_width: int = 960) -> np.ndarray | None:
    frame_bgr = frame_to_bgr(frame, pixel_format=pixel_format)
    if frame_bgr is None:
        return None
    height, width = frame_bgr.shape[:2]
    if width <= max_width:
        return frame_bgr
    scale = max_width / width
    resized_height = int(round(height * scale))
    return cv2.resize(frame_bgr, (max_width, resized_height), interpolation=cv2.INTER_LINEAR)


def frame_to_bgr(frame: object, *, pixel_format: str = "BayerRG8") -> np.ndarray | None:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    if arr.ndim == 2:
        fmt = str(pixel_format).upper()
        code = cv2.COLOR_BayerBG2BGR
        if "BAYERGR" in fmt or fmt == "9":
            code = cv2.COLOR_BayerGR2BGR
        try:
            return cv2.cvtColor(arr, code)
        except cv2.error:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    if arr.ndim == 3 and arr.shape[2] == 3:
        return np.ascontiguousarray(arr[:, :, ::-1])

    if arr.ndim == 3 and arr.shape[2] == 1:
        return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)

    return None


def replay_interval_ms(segment: AnalyzedSegment) -> int:
    timestamps = segment.timestamps
    if len(timestamps) < 2:
        return 26
    deltas = [
        timestamps[index] - timestamps[index - 1]
        for index in range(1, len(timestamps))
        if timestamps[index] > timestamps[index - 1]
    ]
    if not deltas:
        return 26
    return max(1, int(round((sum(deltas) / len(deltas)) * 1000.0)))


def monotonic_seconds() -> float:
    return time.monotonic()


@dataclass(frozen=True)
class FrameRatioSummary:
    component_track_id: int | None
    overall_severity: float
    frame_ratios: dict[str, float]


def summarize_frame_ratios(
    *,
    frame_records: list[dict[str, Any]],
    defect_labels: list[str],
) -> FrameRatioSummary:
    aggregate_frame_ratios: dict[str, float] = {label: 0.0 for label in defect_labels}
    component_span_frames = _component_span_frames(frame_records)
    component_frame_records = [
        record
        for record in frame_records
        if record.get("component_bbox") is not None
    ]
    relevant_records = component_frame_records if component_frame_records else frame_records

    for label in defect_labels:
        frames_with_defect = sum(
            1
            for record in relevant_records
            if isinstance(record.get("by_label"), dict) and record["by_label"].get(label)
        )
        aggregate_frame_ratios[label] = float(frames_with_defect) / float(component_span_frames) if component_span_frames > 0 else 0.0

    return FrameRatioSummary(
        component_track_id=None,
        overall_severity=max(aggregate_frame_ratios.values(), default=0.0),
        frame_ratios=aggregate_frame_ratios,
    )


def _filter_detections_by_class_thresholds(
    detections: list[Any] | tuple[Any, ...],
    *,
    component_score_threshold: float,
    defect_score_threshold: float,
) -> list[Any]:
    filtered: list[Any] = []
    for detection in detections:
        class_id = int(getattr(detection, "class_id", -1))
        score = float(getattr(detection, "score", 0.0) or 0.0)
        threshold = component_score_threshold if class_id == 0 else defect_score_threshold
        if score >= threshold:
            filtered.append(detection)
    return filtered


def _contrast_signal(frame: np.ndarray) -> float:
    height, width = frame.shape[:2]
    margin_y = int(height * 0.15)
    margin_x = int(width * 0.15)
    roi = frame[margin_y : height - margin_y, margin_x : width - margin_x]

    if roi.ndim == 3 and roi.shape[2] >= 3:
        return float(np.std(roi[:, :, 1]) / 255.0)
    return float(np.std(roi) / 255.0)


def _draw_detections(frame_bgr: np.ndarray, detections: tuple[Any, ...]) -> np.ndarray:
    output = frame_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in getattr(detection, "bbox_xyxy", (0, 0, 0, 0))]
        class_id = int(getattr(detection, "class_id", -1))
        label = str(getattr(detection, "label", "unknown"))
        score = float(getattr(detection, "score", 0.0))
        color = (0, 200, 0) if class_id == 0 else (0, 64, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            f"{label} {score:.2f}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def _resize_for_replay(frame_bgr: np.ndarray, replay_height: int) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    if height <= replay_height:
        return frame_bgr
    scale = replay_height / height
    replay_width = int(round(width * scale))
    return cv2.resize(frame_bgr, (replay_width, replay_height), interpolation=cv2.INTER_LINEAR)


def _defect_labels_from_inference_service(inference_service: Any) -> list[str]:
    manifest = getattr(inference_service, "_manifest", {})
    classes = manifest.get("classes", {}) if isinstance(manifest, dict) else {}
    if isinstance(classes, dict):
        labels = [
            value
            for key, value in sorted(classes.items(), key=lambda item: int(item[0]))
            if int(key) > 0
        ]
        if labels:
            return [str(label) for label in labels]
    class_map = getattr(inference_service, "_class_map", {})
    if isinstance(class_map, dict):
        labels = [value for key, value in sorted(class_map.items()) if int(key) > 0]
        if labels:
            return [str(label) for label in labels]
    return []


def _get_confirmed_tracks(tracker: Any) -> list[Any]:
    getter = getattr(tracker, "get_confirmed", None)
    if callable(getter):
        return list(getter())

    legacy_getter = getattr(tracker, "get_all_confirmed", None)
    if callable(legacy_getter):
        return list(legacy_getter())

    raise AttributeError("Tracker does not expose get_confirmed()")


def _build_metric_frame_record(*, frame_index: int, detections: tuple[Any, ...]) -> dict[str, Any]:
    component_bbox = _select_primary_component_bbox(detections)
    crop_w = None
    crop_h = None
    by_label: dict[str, list[dict[str, Any]]] = {}
    if component_bbox is not None:
        crop_w = max(0.0, component_bbox[2] - component_bbox[0])
        crop_h = max(0.0, component_bbox[3] - component_bbox[1])
        for detection in detections:
            if int(getattr(detection, "class_id", -1)) <= 0:
                continue
            defect_bbox = tuple(float(value) for value in getattr(detection, "bbox_xyxy", (0, 0, 0, 0)))
            if len(defect_bbox) != 4:
                continue
            if not _bbox_assigned_to_component(defect_bbox, component_bbox):
                continue
            label = str(getattr(detection, "label", "unknown"))
            by_label.setdefault(label, []).append(
                {
                    "score": float(getattr(detection, "score", 0.0)),
                    "bbox_xyxy": defect_bbox,
                    "crop_w": crop_w if crop_w > 0 else None,
                    "crop_h": crop_h if crop_h > 0 else None,
                }
            )
    return {
        "frame_idx": int(frame_index),
        "sample_index": int(frame_index),
        "component_bbox": component_bbox,
        "crop_w": crop_w,
        "crop_h": crop_h,
        "by_label": by_label,
    }


def _select_primary_component_bbox(detections: tuple[Any, ...]) -> tuple[float, float, float, float] | None:
    candidates: list[tuple[float, float, tuple[float, float, float, float]]] = []
    for detection in detections:
        if int(getattr(detection, "class_id", -1)) != 0:
            continue
        bbox = tuple(float(value) for value in getattr(detection, "bbox_xyxy", (0, 0, 0, 0)))
        if len(bbox) != 4:
            continue
        area = _bbox_area(bbox)
        score = float(getattr(detection, "score", 0.0))
        candidates.append((score, area, bbox))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2][0], item[2][1]))
    return candidates[0][2]
def _component_span_frames(frame_records: list[dict[str, Any]]) -> int:
    component_frames = [
        int(record["frame_idx"])
        for record in frame_records
        if record.get("component_bbox") is not None
    ]
    if component_frames:
        return max(1, max(component_frames) - min(component_frames) + 1)
    if frame_records:
        frame_indices = [int(record["frame_idx"]) for record in frame_records]
        return max(1, max(frame_indices) - min(frame_indices) + 1)
    return 1


def _bbox_assigned_to_component(
    defect_bbox: tuple[float, float, float, float],
    component_bbox: tuple[float, float, float, float],
) -> bool:
    cx_px = (defect_bbox[0] + defect_bbox[2]) / 2.0
    cy_px = (defect_bbox[1] + defect_bbox[3]) / 2.0
    component_width = max(1.0, component_bbox[2] - component_bbox[0])
    component_height = max(1.0, component_bbox[3] - component_bbox[1])
    margin_x = component_width * 0.1
    margin_y = component_height * 0.1
    return (
        (component_bbox[0] - margin_x) <= cx_px <= (component_bbox[2] + margin_x)
        and (component_bbox[1] - margin_y) <= cy_px <= (component_bbox[3] + margin_y)
    )


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _analyze_tracking_frame(
    *,
    frame_index: int,
    detections: tuple[Any, ...],
    frame_time_seconds: float,
    component_tracker: Any,
    defect_tracker: Any,
    severity_engine: Any,
) -> tuple[list[Any], list[Any], dict[int, int], list[Any]]:
    component_detections = [detection for detection in detections if int(detection.class_id) == 0]
    defect_detections = [detection for detection in detections if int(detection.class_id) > 0]

    confirmed_components = component_tracker.update(frame_index, component_detections)
    confirmed_defects = defect_tracker.update(frame_index, defect_detections)
    track_assignments = _assign_defects_to_components(confirmed_components, confirmed_defects)
    metrics_list, _finalized = severity_engine.update(
        frame_index,
        frame_time_seconds,
        confirmed_components,
        confirmed_defects,
        track_assignments,
    )
    return confirmed_components, confirmed_defects, track_assignments, metrics_list


def _assign_defects_to_components(
    confirmed_components: list[Any],
    confirmed_defects: list[Any],
) -> dict[int, int]:
    track_assignments: dict[int, int] = {}
    for defect_track in confirmed_defects:
        x1, y1, x2, y2 = defect_track.bbox_xyxy
        cx_px = (x1 + x2) / 2.0
        cy_px = (y1 + y2) / 2.0
        best_component = None
        best_overlap = 0.0
        for component_track in confirmed_components:
            cx1, cy1, cx2, cy2 = component_track.bbox_xyxy
            component_width = max(1, cx2 - cx1)
            component_height = max(1, cy2 - cy1)
            margin_x = component_width * 0.1
            margin_y = component_height * 0.1

            if (cx1 - margin_x) <= cx_px <= (cx2 + margin_x) and (cy1 - margin_y) <= cy_px <= (cy2 + margin_y):
                overlap_ratio = _bbox_overlap_ratio(defect_track.bbox_xyxy, component_track.bbox_xyxy)
                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_component = component_track
        if best_component is not None:
            track_assignments[int(defect_track.track_id)] = int(best_component.track_id)
    return track_assignments


def _bbox_overlap_ratio(
    defect_bbox: tuple[int, int, int, int],
    component_bbox: tuple[int, int, int, int],
) -> float:
    intersection_x1 = max(defect_bbox[0], component_bbox[0])
    intersection_y1 = max(defect_bbox[1], component_bbox[1])
    intersection_x2 = min(defect_bbox[2], component_bbox[2])
    intersection_y2 = min(defect_bbox[3], component_bbox[3])
    if intersection_x2 <= intersection_x1 or intersection_y2 <= intersection_y1:
        return 0.0
    intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
    defect_area = (defect_bbox[2] - defect_bbox[0]) * (defect_bbox[3] - defect_bbox[1])
    return intersection_area / defect_area if defect_area > 0 else 0.0
