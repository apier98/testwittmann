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
    pixel_format: str,
    replay_height: int = 480,
) -> AnalyzedSegment:
    replay_frames: list[np.ndarray] = []
    detections_per_frame: list[tuple[Any, ...]] = []
    total_detections = 0
    total_defects = 0

    for frame in segment.frames:
        frame_bgr = frame_to_bgr(frame, pixel_format=pixel_format)
        if frame_bgr is None:
            continue

        tensor, lbmeta = inference_service.preprocess(frame_bgr)
        output_map = inference_service.run_session(tensor)
        frame_h, frame_w = frame_bgr.shape[:2]
        detections = tuple(inference_service.postprocess(output_map, frame_w, frame_h, lbmeta))

        total_detections += len(detections)
        total_defects += sum(1 for detection in detections if getattr(detection, "class_id", -1) > 0)
        detections_per_frame.append(detections)
        replay_frames.append(_resize_for_replay(_draw_detections(frame_bgr, detections), replay_height))

    return AnalyzedSegment(
        replay_frames=tuple(replay_frames),
        detections=tuple(detections_per_frame),
        timestamps=segment.timestamps,
        total_detections=total_detections,
        total_defects=total_defects,
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
