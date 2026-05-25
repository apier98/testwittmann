from __future__ import annotations

import numpy as np

from testwittmann.pipeline import SegmentTrigger, TriggerConfig


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
