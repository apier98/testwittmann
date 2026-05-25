from __future__ import annotations

import numpy as np

from testwittmann.pipeline import (
    SegmentTrigger,
    TriggerConfig,
    _filter_detections_by_class_thresholds,
    _get_confirmed_tracks,
    summarize_frame_ratios,
)


def test_segment_trigger_captures_pre_and_post_roll() -> None:
    trigger = SegmentTrigger(
        TriggerConfig(
            threshold=0.01,
            consecutive_frames=2,
            pre_roll_count=2,
            post_roll_count=2,
            max_active_frames=10,
        )
    )

    blank = np.zeros((20, 20, 3), dtype=np.uint8)
    textured = np.zeros((20, 20, 3), dtype=np.uint8)
    textured[:, ::2, 1] = 255

    segment = None
    timeline = [
        blank,
        blank,
        textured,
        textured,
        textured,
        blank,
        blank,
    ]
    for index, frame in enumerate(timeline):
        segment = trigger.push_frame(float(index), frame)

    assert segment is not None
    assert len(segment.frames) == 6
    assert segment.timestamps[0] == 1.0
    assert segment.timestamps[-1] == 6.0
    assert segment.trigger_metadata["forced_split"] is False


def test_segment_trigger_forces_split_when_segment_grows_too_long() -> None:
    trigger = SegmentTrigger(
        TriggerConfig(
            threshold=0.01,
            consecutive_frames=1,
            pre_roll_count=1,
            post_roll_count=1,
            max_active_frames=3,
        )
    )

    textured = np.zeros((20, 20, 3), dtype=np.uint8)
    textured[:, ::2, 1] = 255

    segment = None
    for index in range(3):
        segment = trigger.push_frame(float(index), textured)

    assert segment is not None
    assert len(segment.frames) == 3
    assert segment.trigger_metadata["forced_split"] is True


def test_summarize_frame_ratios_uses_frames_with_defect_over_component_span() -> None:
    summary = summarize_frame_ratios(
        frame_records=[
            {
                "frame_idx": frame_index,
                "sample_index": frame_index,
                "component_bbox": (0.0, 0.0, 100.0, 100.0),
                "crop_w": 100.0,
                "crop_h": 100.0,
                "by_label": {
                    "Weld_Line": [
                        {
                            "score": 0.9,
                            "bbox_xyxy": (10.0, 10.0, 20.0, 20.0),
                            "crop_w": 100.0,
                            "crop_h": 100.0,
                        }
                    ]
                }
                if frame_index in {0, 1, 3}
                else {},
            }
            for frame_index in range(10)
        ],
        defect_labels=["Weld_Line", "Sink_Mark", "Flash"],
    )

    assert summary.component_track_id is None
    assert summary.overall_severity == 0.3
    assert summary.frame_ratios == {
        "Weld_Line": 0.3,
        "Sink_Mark": 0.0,
        "Flash": 0.0,
    }


def test_get_confirmed_tracks_supports_moldpilot_tracker_api() -> None:
    class Tracker:
        def get_confirmed(self) -> list[int]:
            return [1, 2]

    assert _get_confirmed_tracks(Tracker()) == [1, 2]


def test_summarize_frame_ratios_counts_short_visible_defects_too() -> None:
    summary = summarize_frame_ratios(
        frame_records=[
            {
                "frame_idx": frame_index,
                "sample_index": frame_index,
                "component_bbox": (0.0, 0.0, 100.0, 100.0),
                "crop_w": 100.0,
                "crop_h": 100.0,
                "by_label": {
                    "Sink_Mark": [
                        {
                            "score": 0.8,
                            "bbox_xyxy": (30.0, 30.0, 40.0, 40.0),
                            "crop_w": 100.0,
                            "crop_h": 100.0,
                        }
                    ]
                }
                if frame_index in {0, 1}
                else {},
            }
            for frame_index in range(6)
        ],
        defect_labels=["Weld_Line", "Sink_Mark"],
    )

    assert summary.component_track_id is None
    assert summary.overall_severity == (2 / 6)
    assert summary.frame_ratios == {
        "Weld_Line": 0.0,
        "Sink_Mark": (2 / 6),
    }


def test_filter_detections_by_class_thresholds_uses_separate_component_and_defect_thresholds() -> None:
    component_low = type("Detection", (), {"class_id": 0, "score": 0.69})()
    component_ok = type("Detection", (), {"class_id": 0, "score": 0.70})()
    defect_low = type("Detection", (), {"class_id": 2, "score": 0.44})()
    defect_ok = type("Detection", (), {"class_id": 2, "score": 0.45})()

    filtered = _filter_detections_by_class_thresholds(
        [component_low, component_ok, defect_low, defect_ok],
        component_score_threshold=0.7,
        defect_score_threshold=0.45,
    )

    assert filtered == [component_ok, defect_ok]
