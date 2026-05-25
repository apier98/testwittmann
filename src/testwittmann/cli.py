from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2

from testwittmann.config import RuntimeAssets, load_runtime_assets
from testwittmann.constants import WINDOW_NAME_LIVE, WINDOW_NAME_REPLAY
from testwittmann.moldpilot_bridge import MoldPilotImports, load_moldpilot_imports
from testwittmann.pipeline import (
    AnalyzedSegment,
    SegmentTrigger,
    TriggerConfig,
    analyze_segment,
    frame_to_display,
    monotonic_seconds,
    replay_interval_ms,
)


@dataclass
class ReplayState:
    segment: AnalyzedSegment
    frame_index: int = 0
    next_frame_at: float = 0.0
    interval_ms: int = 26


class LiveDetectorCli:
    def __init__(
        self,
        assets: RuntimeAssets,
        imports: MoldPilotImports,
        *,
        trigger_config: TriggerConfig,
        score_threshold: float | None,
    ) -> None:
        self._assets = assets
        self._imports = imports
        self._trigger_config = trigger_config
        self._score_threshold = score_threshold
        self._log = logging.getLogger("testwittmann")

        self._stop_event = threading.Event()
        self._subscription_queue: queue.Queue[tuple[float, Any]] = queue.Queue(maxsize=200)
        self._segment_queue: queue.Queue = queue.Queue(maxsize=4)
        self._replay_queue: queue.Queue[AnalyzedSegment] = queue.Queue(maxsize=2)

        self._capture_thread: threading.Thread | None = None
        self._segment_thread: threading.Thread | None = None

        self._camera_service: Any | None = None
        self._inference_service: Any | None = None
        self._replay_state: ReplayState | None = None

    def run(self) -> int:
        self._setup_runtime()
        self._start_workers()

        print(
            "Starting CLI monitoring with MoldPilot assets:\n"
            f"- config: {self._assets.config_id} ({self._assets.config_version})\n"
            f"- detector: {self._assets.model_bundle_id}\n"
            f"- bundle path: {self._assets.model_bundle_path}\n"
            "Controls: press 'q' or ESC in the OpenCV window to stop."
        )

        cv2.namedWindow(WINDOW_NAME_LIVE, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WINDOW_NAME_REPLAY, cv2.WINDOW_NORMAL)

        try:
            while not self._stop_event.is_set():
                self._show_live_frame()
                self._maybe_start_latest_replay()
                self._show_replay_frame()

                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q"), ord("Q")}:
                    break
                if key in {ord("r"), ord("R")} and self._replay_state is not None:
                    self._replay_state.frame_index = 0
                    self._replay_state.next_frame_at = 0.0

                time.sleep(0.005)
        finally:
            self.shutdown()
        return 0

    def shutdown(self) -> None:
        self._stop_event.set()

        if self._camera_service is not None:
            try:
                self._camera_service.unsubscribe("process", self._subscription_queue)
            except Exception:
                pass

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        if self._segment_thread is not None:
            self._segment_thread.join(timeout=2.0)
            self._segment_thread = None

        if self._inference_service is not None:
            try:
                self._inference_service.unload()
            except Exception:
                pass
            self._inference_service = None

        if self._camera_service is not None:
            try:
                self._camera_service.stop()
            except Exception:
                pass
            self._camera_service = None

        cv2.destroyAllWindows()

    def _setup_runtime(self) -> None:
        profile = self._imports.CameraProfile(
            self._assets.process_camera_profile.exposure_time_us,
            self._assets.process_camera_profile.gain_db,
            self._assets.process_camera_profile.frame_rate_fps,
            self._assets.process_camera_profile.pixel_format,
            self._assets.process_camera_profile.trigger_mode,
            self._assets.process_camera_profile.roi_width,
            self._assets.process_camera_profile.roi_height,
            self._assets.process_camera_profile.offset_x,
            self._assets.process_camera_profile.offset_y,
        )

        runtime_config = self._imports.BaumerCameraRuntimeConfig(
            camera_id="camera_primary",
            model_name="Baumer VCXG.2-32C",
            display_label="Component View",
            network=self._imports.StaticNetworkConfig(
                host_ip="192.168.10.10",
                monitor_camera_ip="192.168.10.11",
                process_camera_ip="192.168.10.12",
                subnet_mask="255.255.255.0",
            ),
            profile=profile,
            heartbeat_timeout_ms=3000,
            reconnect_backoff_seconds=2.0,
            active_recovery_window_seconds=45.0,
            passive_reconnect_poll_seconds=10.0,
            force_ip_enabled=True,
        )

        runtime_config = self._imports.apply_machine_camera_bindings(
            runtime_config,
            self._imports.load_machine_camera_bindings(self._assets.configs_root),
        )

        self._camera_service = self._imports.BaumerCameraService(runtime_config, logger=self._log)
        self._camera_service.start()
        self._camera_service.apply_camera_profile(profile, source_id="process")
        self._camera_service.subscribe("process", self._subscription_queue)

        self._inference_service = self._imports.OnnxInferenceService()
        self._inference_service.load_bundle(self._assets.model_bundle_path)
        self._inference_service.set_score_threshold_override(self._score_threshold)

    def _start_workers(self) -> None:
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="testwittmann-capture",
            daemon=True,
        )
        self._capture_thread.start()

        self._segment_thread = threading.Thread(
            target=self._segment_loop,
            name="testwittmann-segment",
            daemon=True,
        )
        self._segment_thread.start()

    def _capture_loop(self) -> None:
        trigger = SegmentTrigger(self._trigger_config)
        while not self._stop_event.is_set():
            try:
                timestamp, frame = self._subscription_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            segment = trigger.push_frame(timestamp, frame)
            if segment is None:
                continue

            try:
                self._segment_queue.put(segment, timeout=0.1)
            except queue.Full:
                self._log.warning("Segment queue full; dropping one captured segment.")

    def _segment_loop(self) -> None:
        assert self._inference_service is not None
        pixel_format = self._assets.process_camera_profile.pixel_format

        while not self._stop_event.is_set():
            try:
                segment = self._segment_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                analyzed = analyze_segment(
                    segment,
                    self._inference_service,
                    pixel_format=pixel_format,
                )
            except Exception as exc:
                self._log.exception("Segment inference failed: %s", exc)
                continue

            if not analyzed.replay_frames:
                continue

            self._log.info(
                "Segment analyzed: frames=%d detections=%d defects=%d latency=%.1fms provider=%s",
                len(analyzed.replay_frames),
                analyzed.total_detections,
                analyzed.total_defects,
                float(self._inference_service.latency_ms),
                self._inference_service.provider_used,
            )

            while not self._replay_queue.empty():
                try:
                    self._replay_queue.get_nowait()
                except queue.Empty:
                    break
            self._replay_queue.put_nowait(analyzed)

    def _show_live_frame(self) -> None:
        assert self._camera_service is not None
        frame = self._camera_service.get_latest_frame("process")
        if frame is None:
            return

        display_frame = frame_to_display(
            frame,
            pixel_format=self._assets.process_camera_profile.pixel_format,
        )
        if display_frame is not None:
            cv2.imshow(WINDOW_NAME_LIVE, display_frame)

    def _maybe_start_latest_replay(self) -> None:
        try:
            segment = self._replay_queue.get_nowait()
        except queue.Empty:
            return

        self._replay_state = ReplayState(
            segment=segment,
            frame_index=0,
            next_frame_at=0.0,
            interval_ms=replay_interval_ms(segment),
        )

    def _show_replay_frame(self) -> None:
        if self._replay_state is None:
            return

        replay_frames = self._replay_state.segment.replay_frames
        if not replay_frames:
            return

        now = monotonic_seconds()
        if self._replay_state.next_frame_at == 0.0 or now >= self._replay_state.next_frame_at:
            frame = replay_frames[self._replay_state.frame_index]
            cv2.imshow(WINDOW_NAME_REPLAY, frame)

            self._replay_state.frame_index += 1
            if self._replay_state.frame_index >= len(replay_frames):
                self._replay_state.frame_index = 0

            self._replay_state.next_frame_at = now + (self._replay_state.interval_ms / 1000.0)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="testwittmann",
        description="Run the MoldPilot-like live stream, trigger, inference and replay flow from the CLI.",
    )
    parser.add_argument(
        "--trigger-threshold",
        type=float,
        default=0.05,
        help="Binary contrast trigger threshold copied from the MoldPilot segmented capture flow.",
    )
    parser.add_argument(
        "--trigger-frames",
        type=int,
        default=3,
        help="How many consecutive present frames are required before capture starts.",
    )
    parser.add_argument(
        "--pre-roll",
        type=int,
        default=10,
        help="How many frames to retain before the trigger fires.",
    )
    parser.add_argument(
        "--post-roll",
        type=int,
        default=10,
        help="How many trailing frames to keep after the trigger falls inactive.",
    )
    parser.add_argument(
        "--max-segment-frames",
        type=int,
        default=160,
        help="Maximum buffered frames per segment before forcing a split.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Optional detector score threshold override.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = build_argument_parser()
    args = parser.parse_args(argv)

    assets = load_runtime_assets()
    imports = load_moldpilot_imports()
    app = LiveDetectorCli(
        assets,
        imports,
        trigger_config=TriggerConfig(
            threshold=args.trigger_threshold,
            consecutive_frames=args.trigger_frames,
            pre_roll_count=args.pre_roll,
            post_roll_count=args.post_roll,
            max_active_frames=args.max_segment_frames,
        ),
        score_threshold=args.score_threshold,
    )
    return app.run()
