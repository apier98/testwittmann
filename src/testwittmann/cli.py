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
from testwittmann.suggestions import (
    DefectObservation,
    ParameterSlope,
    SuggestionCandidate,
    SuggestionModelBundle,
    default_process_parameter_values,
    load_bundled_suggestion_model,
)

_DEFAULT_PROCESS_PARAMETERS = default_process_parameter_values()


@dataclass
class ReplayState:
    segment: AnalyzedSegment
    frame_index: int = 0
    next_frame_at: float = 0.0
    interval_ms: int = 26


@dataclass(frozen=True)
class ReplayPayload:
    segment: AnalyzedSegment
    suggestion_defects: tuple[DefectObservation, ...] = ()


@dataclass
class SuggestionDecisionState:
    round_id: int
    defects: tuple[DefectObservation, ...]
    current_parameters: dict[str, float]


@dataclass(frozen=True)
class SuggestionDecisionResult:
    round_id: int
    selected_defect: DefectObservation | None
    suggestion: SuggestionCandidate | None
    skipped: bool


class LiveDetectorCli:
    def __init__(
        self,
        assets: RuntimeAssets,
        imports: MoldPilotImports,
        *,
        trigger_config: TriggerConfig,
        component_score_threshold: float,
        defect_score_threshold: float,
        suggestion_model: SuggestionModelBundle,
        current_process_parameters: dict[str, float],
        suggestion_target_ratio: float,
        suggestion_sample_count: int,
    ) -> None:
        self._assets = assets
        self._imports = imports
        self._trigger_config = trigger_config
        self._component_score_threshold = component_score_threshold
        self._defect_score_threshold = defect_score_threshold
        self._suggestion_model = suggestion_model
        self._current_process_parameters = {
            str(key): float(value) for key, value in current_process_parameters.items()
        }
        self._suggestion_target_ratio = float(suggestion_target_ratio)
        self._suggestion_sample_count = int(suggestion_sample_count)
        self._log = logging.getLogger("testwittmann")

        self._stop_event = threading.Event()
        self._pause_processing_event = threading.Event()
        self._subscription_queue: queue.Queue[tuple[float, Any]] = queue.Queue(maxsize=200)
        self._segment_queue: queue.Queue = queue.Queue(maxsize=4)
        self._replay_queue: queue.Queue[ReplayPayload] = queue.Queue(maxsize=2)
        self._decision_request_queue: queue.Queue[SuggestionDecisionState] = queue.Queue(maxsize=1)
        self._decision_result_queue: queue.Queue[SuggestionDecisionResult] = queue.Queue(maxsize=1)

        self._capture_thread: threading.Thread | None = None
        self._segment_thread: threading.Thread | None = None
        self._decision_thread: threading.Thread | None = None

        self._camera_service: Any | None = None
        self._inference_service: Any | None = None
        self._replay_state: ReplayState | None = None
        self._decision_state: SuggestionDecisionState | None = None
        self._decision_round_id = 0

    def run(self) -> int:
        self._setup_runtime()
        self._start_workers()

        print(
            "Starting CLI monitoring with MoldPilot assets:\n"
            f"- config: {self._assets.config_id} ({self._assets.config_version})\n"
            f"- detector: {self._assets.model_bundle_id}\n"
            f"- bundle path: {self._assets.model_bundle_path}\n"
            f"- component threshold: {self._component_score_threshold:.2f}\n"
            f"- defect threshold: {self._defect_score_threshold:.2f}\n"
            f"- suggestion model defects: {', '.join(self._suggestion_model.supported_defect_keys)}\n"
            f"- suggestion current setpoints: {_format_parameter_summary(self._current_process_parameters)}\n"
            "Controls: press 'q' or ESC in the OpenCV window to stop, 'r' to restart replay, "
            "and use the terminal when a suggestion round opens to choose defect, parameters, target ratio and suggestion."
        )

        cv2.namedWindow(WINDOW_NAME_LIVE, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WINDOW_NAME_REPLAY, cv2.WINDOW_NORMAL)

        try:
            while not self._stop_event.is_set():
                self._show_live_frame()
                self._maybe_start_latest_replay()
                self._maybe_apply_decision_result()
                self._show_replay_frame()

                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q"), ord("Q")}:
                    break
                if key in {ord("r"), ord("R")} and self._replay_state is not None:
                    self._replay_state.frame_index = 0
                    self._replay_state.next_frame_at = 0.0
                    continue

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

        if self._decision_thread is not None:
            self._decision_thread.join(timeout=0.1)
            self._decision_thread = None

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
        self._inference_service.set_score_threshold_override(
            min(self._component_score_threshold, self._defect_score_threshold)
        )

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

        self._decision_thread = threading.Thread(
            target=self._decision_loop,
            name="testwittmann-decision",
            daemon=True,
        )
        self._decision_thread.start()

    def _capture_loop(self) -> None:
        trigger = SegmentTrigger(self._trigger_config)
        while not self._stop_event.is_set():
            try:
                timestamp, frame = self._subscription_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._pause_processing_event.is_set():
                trigger.reset()
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
            if self._pause_processing_event.is_set():
                self._drain_queue(self._segment_queue)
                time.sleep(0.05)
                continue

            try:
                segment = self._segment_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                analyzed = analyze_segment(
                    segment,
                    self._inference_service,
                    tracking_imports=self._imports,
                    pixel_format=pixel_format,
                    component_score_threshold=self._component_score_threshold,
                    defect_score_threshold=self._defect_score_threshold,
                )
            except Exception as exc:
                self._log.exception("Segment inference failed: %s", exc)
                continue

            if not analyzed.replay_frames:
                continue

            self._log.info(
                "Segment analyzed: frames=%d detections=%d defects=%d ratios=%s max_ratio=%.3f latency=%.1fms provider=%s",
                len(analyzed.replay_frames),
                analyzed.total_detections,
                analyzed.total_defects,
                ", ".join(
                    f"{label}={ratio:.3f}" for label, ratio in analyzed.frame_ratios.items()
                ) or "none",
                analyzed.overall_severity,
                float(self._inference_service.latency_ms),
                self._inference_service.provider_used,
            )

            suggestion_defects = self._suggestion_model.available_defect_observations(
                analyzed.frame_ratios
            )
            if suggestion_defects:
                self._pause_processing_event.set()
                self._drain_queue(self._segment_queue)

            while not self._replay_queue.empty():
                try:
                    self._replay_queue.get_nowait()
                except queue.Empty:
                    break
            self._replay_queue.put_nowait(
                ReplayPayload(
                    segment=analyzed,
                    suggestion_defects=suggestion_defects,
                )
            )

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
            payload = self._replay_queue.get_nowait()
        except queue.Empty:
            return

        self._replay_state = ReplayState(
            segment=payload.segment,
            frame_index=0,
            next_frame_at=0.0,
            interval_ms=replay_interval_ms(payload.segment),
        )
        if payload.suggestion_defects:
            self._decision_round_id += 1
            self._decision_state = SuggestionDecisionState(
                round_id=self._decision_round_id,
                defects=payload.suggestion_defects,
                current_parameters=dict(self._current_process_parameters),
            )
            self._log.info(
                "Suggestion round opened for defects: %s",
                ", ".join(
                    f"{item.defect_label}={item.measured_ratio:.3f}"
                    for item in payload.suggestion_defects
                ),
            )
            self._queue_decision_prompt(self._decision_state)
        else:
            self._decision_state = None

    def _show_replay_frame(self) -> None:
        if self._replay_state is None:
            return

        replay_frames = self._replay_state.segment.replay_frames
        if not replay_frames:
            return

        now = monotonic_seconds()
        if self._replay_state.next_frame_at == 0.0 or now >= self._replay_state.next_frame_at:
            frame = replay_frames[self._replay_state.frame_index].copy()
            _draw_replay_metrics_overlay(frame, self._replay_state.segment)
            if self._decision_state is not None:
                _draw_terminal_prompt_overlay(
                    frame,
                    self._decision_state,
                )
            cv2.imshow(WINDOW_NAME_REPLAY, frame)

            self._replay_state.frame_index += 1
            if self._replay_state.frame_index >= len(replay_frames):
                self._replay_state.frame_index = 0

            self._replay_state.next_frame_at = now + (self._replay_state.interval_ms / 1000.0)

    def _decision_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                state = self._decision_request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            result = self._run_terminal_decision_round(state)
            if result is None:
                continue
            while not self._decision_result_queue.empty():
                try:
                    self._decision_result_queue.get_nowait()
                except queue.Empty:
                    break
            self._decision_result_queue.put_nowait(result)

    def _run_terminal_decision_round(
        self,
        state: SuggestionDecisionState,
    ) -> SuggestionDecisionResult | None:
        print()
        print("=" * 72)
        print(f"Suggestion round {state.round_id} paused for the current component.")
        print(f"Current setpoints: {_format_parameter_summary(state.current_parameters)}")
        print("Replay stays visible. Choose in the terminal to continue.")

        while not self._stop_event.is_set():
            print()
            print("Detected supported defects:")
            for index, defect in enumerate(state.defects, start=1):
                print(f"  {index}. {defect.defect_label} ({defect.measured_ratio:.3f})")
            raw_choice = _read_console_choice(
                "Select defect [1-9] or 's' to skip: ",
                stop_event=self._stop_event,
            )
            if raw_choice is None:
                return None
            if raw_choice in {"s", "skip", "0"}:
                return SuggestionDecisionResult(
                    round_id=state.round_id,
                    selected_defect=None,
                    suggestion=None,
                    skipped=True,
                )
            if not raw_choice.isdigit():
                print("Invalid choice.")
                continue
            selected_index = int(raw_choice) - 1
            if selected_index < 0 or selected_index >= len(state.defects):
                print("Invalid choice.")
                continue

            defect = state.defects[selected_index]
            slopes = self._suggestion_model.local_parameter_slopes(
                current_parameters=state.current_parameters,
                defect_label=defect.defect_key,
            )

            while not self._stop_event.is_set():
                print()
                print(f"Pendenze numeriche in x0 per {defect.defect_label} (ordine decrescente):")
                _print_parameter_slopes(slopes)
                raw_feature_choice = _read_console_choice(
                    "Seleziona uno o piu parametri [es. 1,3], 'all', 'b' indietro o 's' skip: ",
                    stop_event=self._stop_event,
                )
                if raw_feature_choice is None:
                    return None
                if raw_feature_choice in {"b", "back"}:
                    break
                if raw_feature_choice in {"s", "skip", "0"}:
                    return SuggestionDecisionResult(
                        round_id=state.round_id,
                        selected_defect=defect,
                        suggestion=None,
                        skipped=True,
                    )

                selected_feature_keys = _parse_feature_selection(
                    raw_feature_choice,
                    ranked_slopes=slopes,
                )
                if selected_feature_keys is None:
                    print("Selezione parametri non valida.")
                    continue

                while not self._stop_event.is_set():
                    raw_target_ratio = _read_console_choice(
                        f"Soglia target da raggiungere [{self._suggestion_target_ratio:.3f}] "
                        "(Invio=default, 'b' indietro, 's' skip): ",
                        stop_event=self._stop_event,
                    )
                    if raw_target_ratio is None:
                        return None
                    if raw_target_ratio in {"b", "back"}:
                        break
                    if raw_target_ratio in {"s", "skip", "0"}:
                        return SuggestionDecisionResult(
                            round_id=state.round_id,
                            selected_defect=defect,
                            suggestion=None,
                            skipped=True,
                        )

                    target_ratio = _parse_target_ratio(
                        raw_target_ratio,
                        default_value=self._suggestion_target_ratio,
                    )
                    if target_ratio is None:
                        print("Soglia non valida.")
                        continue

                    suggestions = self._suggestion_model.top_suggestions(
                        current_parameters=state.current_parameters,
                        defect_label=defect.defect_key,
                        measured_ratio=defect.measured_ratio,
                        target_ratio=target_ratio,
                        top_k=3,
                        sample_count=self._suggestion_sample_count,
                        feature_keys_to_optimize=selected_feature_keys,
                    )
                    self._log.info(
                        "Top suggestions for %s (measured=%.3f, target=%.3f, features=%s): %s",
                        defect.defect_label,
                        defect.measured_ratio,
                        target_ratio,
                        ", ".join(selected_feature_keys),
                        " | ".join(
                            _format_suggestion_summary(
                                suggestion,
                                current_parameters=state.current_parameters,
                            )
                            for suggestion in suggestions
                        ),
                    )
                    print()
                    print(
                        f"Top {len(suggestions)} suggestions for {defect.defect_label} "
                        f"({defect.measured_ratio:.3f} -> {target_ratio:.3f}) "
                        f"muovendo: {', '.join(selected_feature_keys)}"
                    )
                    for suggestion in suggestions:
                        _print_suggestion_candidate(
                            suggestion,
                            current_parameters=state.current_parameters,
                        )

                    while not self._stop_event.is_set():
                        raw_suggestion_choice = _read_console_choice(
                            "Scegli suggerimento [1-3], 'b' indietro o 's' skip: ",
                            stop_event=self._stop_event,
                        )
                        if raw_suggestion_choice is None:
                            return None
                        if raw_suggestion_choice in {"b", "back"}:
                            break
                        if raw_suggestion_choice in {"s", "skip", "0"}:
                            return SuggestionDecisionResult(
                                round_id=state.round_id,
                                selected_defect=defect,
                                suggestion=None,
                                skipped=True,
                            )
                        if not raw_suggestion_choice.isdigit():
                            print("Invalid choice.")
                            continue
                        suggestion_index = int(raw_suggestion_choice) - 1
                        if suggestion_index < 0 or suggestion_index >= len(suggestions):
                            print("Invalid choice.")
                            continue
                        chosen_suggestion = suggestions[suggestion_index]
                        return SuggestionDecisionResult(
                            round_id=state.round_id,
                            selected_defect=defect,
                            suggestion=chosen_suggestion,
                            skipped=False,
                        )

        return None

    def _maybe_apply_decision_result(self) -> None:
        if self._decision_state is None:
            return
        try:
            result = self._decision_result_queue.get_nowait()
        except queue.Empty:
            return
        if result.round_id != self._decision_state.round_id:
            return
        self._finish_decision_result(result)

    def _finish_decision_result(self, result: SuggestionDecisionResult) -> None:
        if result.skipped or result.suggestion is None:
            self._log.info(
                "Suggestion round skipped. Current setpoints remain: %s",
                _format_parameter_summary(self._current_process_parameters),
            )
            print("Suggestion round skipped. Monitoring resumed.")
        else:
            self._current_process_parameters = dict(result.suggestion.parameter_values)
            self._log.info(
                "Accepted suggestion %d for %s. New current setpoints: %s",
                result.suggestion.rank,
                result.suggestion.defect_label,
                _format_parameter_summary(self._current_process_parameters),
            )
            print(
                "Accepted suggestion "
                f"{result.suggestion.rank} for {result.suggestion.defect_label}. "
                f"New setpoints: {_format_parameter_summary(self._current_process_parameters)}"
            )

        self._decision_state = None
        self._pause_processing_event.clear()

    def _queue_decision_prompt(self, state: SuggestionDecisionState) -> None:
        while not self._decision_request_queue.empty():
            try:
                self._decision_request_queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._decision_request_queue.put_nowait(state)
        except queue.Full:
            self._log.warning("Decision prompt queue full; dropping the older prompt state.")

    @staticmethod
    def _drain_queue(target_queue: queue.Queue) -> None:
        while True:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                return


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="testwittmann",
        description="Run the MoldPilot-like live stream, trigger, inference, replay and suggestion flow from the CLI.",
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
        "--defect-score-threshold",
        type=float,
        default=0.45,
        help="Minimum score for defect classes (class_id > 0). Default: 0.45.",
    )
    parser.add_argument(
        "--component-score-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the component class (class_id == 0). Default: 0.70.",
    )
    parser.add_argument(
        "--t-melt",
        type=float,
        default=_DEFAULT_PROCESS_PARAMETERS["t_melt"],
        help="Current melt temperature used as the baseline for suggestion search.",
    )
    parser.add_argument(
        "--t-mold",
        type=float,
        default=_DEFAULT_PROCESS_PARAMETERS["t_mold"],
        help="Current mold temperature used as the baseline for suggestion search.",
    )
    parser.add_argument(
        "--inj-speed",
        type=float,
        default=_DEFAULT_PROCESS_PARAMETERS["inj_speed"],
        help="Current injection speed used as the baseline for suggestion search.",
    )
    parser.add_argument(
        "--pack-pressure",
        type=float,
        default=_DEFAULT_PROCESS_PARAMETERS["pack_pressure"],
        help="Current pack pressure used as the baseline for suggestion search.",
    )
    parser.add_argument(
        "--suggestion-target-ratio",
        type=float,
        default=0.0,
        help="Desired defect frame ratio after the operator applies a suggestion. Default: 0.0.",
    )
    parser.add_argument(
        "--suggestion-samples",
        type=int,
        default=512,
        help="How many random candidate setpoints to explore per suggestion round. Default: 512.",
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
    suggestion_model = load_bundled_suggestion_model()
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
        component_score_threshold=args.component_score_threshold,
        defect_score_threshold=args.defect_score_threshold,
        suggestion_model=suggestion_model,
        current_process_parameters={
            "t_melt": args.t_melt,
            "t_mold": args.t_mold,
            "inj_speed": args.inj_speed,
            "pack_pressure": args.pack_pressure,
        },
        suggestion_target_ratio=args.suggestion_target_ratio,
        suggestion_sample_count=args.suggestion_samples,
    )
    return app.run()


def _draw_replay_metrics_overlay(frame: Any, segment: AnalyzedSegment) -> None:
    lines = [
        "Frame ratio (frames with defect / component span)",
        f"Max frame ratio: {segment.overall_severity:.3f}",
    ]
    if segment.primary_component_track_id is not None:
        lines.append(f"Component track: {segment.primary_component_track_id}")
    for label, ratio in segment.frame_ratios.items():
        lines.append(f"{label}: {ratio:.3f}")

    padding = 12
    line_height = 24
    box_height = padding * 2 + (line_height * len(lines))
    box_width = 360
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + box_width, 10 + box_height), (16, 16, 16), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0.0, frame)

    y = 10 + padding + 16
    for index, line in enumerate(lines):
        color = (255, 255, 255) if index == 0 else (200, 230, 255)
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
        y += line_height


def _draw_terminal_prompt_overlay(
    frame: Any,
    decision_state: SuggestionDecisionState,
) -> None:
    lines = [
        "Suggestion round paused",
        f"Round {decision_state.round_id}: check the terminal prompt",
        "Replay stays active while inference waits for your choice",
        "Terminal flow: defect -> slopes -> parameters -> target -> suggestion",
    ]
    padding = 12
    line_height = 22
    box_width = min(max(frame.shape[1] - 40, 420), 680)
    box_height = min(frame.shape[0] - 20, padding * 2 + line_height * len(lines))
    overlay = frame.copy()
    top_left = (10, max(10, frame.shape[0] - box_height - 10))
    bottom_right = (10 + box_width, top_left[1] + box_height)
    cv2.rectangle(overlay, top_left, bottom_right, (24, 24, 24), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0.0, frame)

    y = top_left[1] + padding + 16
    for index, line in enumerate(lines):
        color = (255, 255, 255) if index < 2 else (200, 240, 200)
        cv2.putText(
            frame,
            line,
            (top_left[0] + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        y += line_height
        if y >= bottom_right[1] - 4:
            break


def _print_suggestion_candidate(
    suggestion: SuggestionCandidate,
    *,
    current_parameters: dict[str, float],
) -> None:
    changed = suggestion.changed_parameters(current_parameters)
    delta_text = ", ".join(
        f"{name}={suggestion.parameter_values[name]:.1f} ({delta:+.1f})"
        for name, delta in changed.items()
    ) or "no parameter change"
    print(
        f"  {suggestion.rank}. expected={suggestion.expected_ratio:.3f} "
        f"pred={suggestion.predicted_ratio:.3f} err={suggestion.ranking_error:.3f}"
    )
    print(f"     {delta_text}")


def _print_parameter_slopes(slopes: tuple[ParameterSlope, ...]) -> None:
    for index, item in enumerate(slopes, start=1):
        print(f"  {index}. {item.feature_key}: slope={item.slope:.6f}")


def _format_suggestion_summary(
    suggestion: SuggestionCandidate,
    *,
    current_parameters: dict[str, float],
) -> str:
    changed = suggestion.changed_parameters(current_parameters)
    deltas = ", ".join(f"{name}{delta:+.1f}" for name, delta in changed.items()) or "no-op"
    return (
        f"#{suggestion.rank} expected={suggestion.expected_ratio:.3f} "
        f"pred={suggestion.predicted_ratio:.3f} {deltas}"
    )


def _format_parameter_summary(current_parameters: dict[str, float]) -> str:
    return ", ".join(f"{name}={value:.1f}" for name, value in current_parameters.items())


def _parse_feature_selection(
    raw_choice: str,
    *,
    ranked_slopes: tuple[ParameterSlope, ...],
) -> tuple[str, ...] | None:
    if raw_choice in {"all", "*"}:
        return tuple(item.feature_key for item in ranked_slopes)
    tokens = [token.strip().lower() for token in raw_choice.replace(";", ",").split(",") if token.strip()]
    if not tokens:
        return None

    ranked_feature_keys = [item.feature_key for item in ranked_slopes]
    normalized_feature_keys = {
        item.feature_key.lower().replace("-", "_"): item.feature_key for item in ranked_slopes
    }
    selected: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token.isdigit():
            index = int(token) - 1
            if index < 0 or index >= len(ranked_feature_keys):
                return None
            feature_key = ranked_feature_keys[index]
        else:
            feature_key = normalized_feature_keys.get(token.replace(" ", "_"))
            if feature_key is None:
                return None
        if feature_key in seen:
            continue
        selected.append(feature_key)
        seen.add(feature_key)
    return tuple(selected) if selected else None


def _parse_target_ratio(raw_value: str, *, default_value: float) -> float | None:
    if raw_value == "":
        return float(default_value)
    try:
        return float(raw_value)
    except ValueError:
        return None


def _read_console_choice(prompt: str, *, stop_event: threading.Event) -> str | None:
    if stop_event.is_set():
        return None
    try:
        return input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        stop_event.set()
        return None
