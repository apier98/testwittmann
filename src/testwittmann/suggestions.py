from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib.resources import as_file, files
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np

_BUNDLED_MODEL_RESOURCE = ("assets", "suggestion_model")
_FEATURE_ALIASES = {
    "p_pack": "pack_pressure",
    "q": "inj_speed",
}


@dataclass(frozen=True)
class DefectObservation:
    defect_key: str
    defect_label: str
    measured_ratio: float


@dataclass(frozen=True)
class SuggestionCandidate:
    rank: int
    defect_key: str
    defect_label: str
    parameter_values: dict[str, float]
    predicted_ratio: float
    expected_ratio: float
    delta_model: float
    ranking_error: float

    @property
    def target_gain(self) -> float:
        return self.predicted_ratio - self.expected_ratio

    def changed_parameters(self, current_parameters: dict[str, float]) -> dict[str, float]:
        return {
            key: float(self.parameter_values[key]) - float(current_parameters[key])
            for key in self.parameter_values
            if abs(float(self.parameter_values[key]) - float(current_parameters[key])) > 1e-9
        }


@dataclass(frozen=True)
class ParameterSlope:
    feature_key: str
    slope: float
    global_slope: float | None = None

    @property
    def absolute_slope(self) -> float:
        return abs(self.slope)


class SuggestionModelBundle:
    def __init__(self, model_root: Path) -> None:
        self._model_root = Path(model_root)
        self._model_info = _read_json_object(self._model_root / "model_info.json")
        self._feature_keys = tuple(str(key) for key in self._model_info["feature_cols"])
        targets = self._model_info["targets"]
        self._targets = {
            str(defect_key): _LoadedSuggestionTarget(
                defect_key=str(defect_key),
                trained_on_column=str(target_info["trained_on_column"]),
                model=joblib.load(self._resolve_bundle_path(str(target_info["model_file"]))),
                scaler=joblib.load(self._resolve_bundle_path(str(target_info["x_scaler_file"]))),
                poly_features=(
                    joblib.load(self._resolve_bundle_path(str(target_info["poly_features_file"])))
                    if "poly_features_file" in target_info
                    else None
                ),
            )
            for defect_key, target_info in targets.items()
        }
        self._feature_bounds = {
            str(name): (
                float(bounds["min"]),
                float(bounds["max"]),
            )
            for name, bounds in self._model_info["x_min_max"].items()
        }
        self._feature_deltas = {
            str(name): float(bounds.get("delta_max", float("inf")))
            for name, bounds in self._model_info["x_min_max"].items()
        }
        self._global_slope_cache: dict[str, dict[str, float]] = {}
        self._precompute_global_slopes()

    def _precompute_global_slopes(self, n_samples: int = 1000, seed: int = 123) -> None:
        """Estimate global sensitivity by averaging slopes across the parameter space."""
        rng = np.random.default_rng(seed)
        for defect_key in self._targets:
            cumulative_slopes = {feat: 0.0 for feat in self._feature_keys}
            
            for _ in range(n_samples):
                x_rand = {
                    feat: float(rng.uniform(self._feature_bounds[feat][0], self._feature_bounds[feat][1]))
                    for feat in self._feature_keys
                }
                local_slopes = self.local_parameter_slopes(
                    current_parameters=x_rand,
                    defect_label=defect_key,
                    step_size=0.1
                )
                for s in local_slopes:
                    cumulative_slopes[s.feature_key] += s.slope
            
            self._global_slope_cache[defect_key] = {
                feat: val / n_samples for feat, val in cumulative_slopes.items()
            }

    @property
    def feature_keys(self) -> tuple[str, ...]:
        return self._feature_keys

    @property
    def feature_bounds(self) -> dict[str, tuple[float, float]]:
        return dict(self._feature_bounds)

    @property
    def supported_defect_keys(self) -> tuple[str, ...]:
        return tuple(self._targets)

    def default_parameter_values(self) -> dict[str, float]:
        return {
            key: (bounds[0] + bounds[1]) / 2.0
            for key, bounds in self._feature_bounds.items()
        }

    def available_defect_observations(
        self,
        frame_ratios: dict[str, float],
    ) -> tuple[DefectObservation, ...]:
        observations: list[DefectObservation] = []
        for defect_label, ratio in frame_ratios.items():
            defect_key = self.resolve_defect_key(defect_label)
            if defect_key is None:
                continue
            measured_ratio = float(ratio)
            if measured_ratio <= 0.0:
                continue
            observations.append(
                DefectObservation(
                    defect_key=defect_key,
                    defect_label=str(defect_label),
                    measured_ratio=measured_ratio,
                )
            )
        observations.sort(key=lambda item: (-item.measured_ratio, item.defect_label.lower()))
        return tuple(observations)

    def resolve_defect_key(self, defect_label: str) -> str | None:
        token = _normalize_token(defect_label)
        if token in self._targets:
            return token
        return None

    def predict_ratio(self, current_parameters: dict[str, float], defect_label: str) -> float:
        defect_key = self.resolve_defect_key(defect_label)
        if defect_key is None:
            raise KeyError(f"Unsupported defect label for suggestion model: {defect_label}")
        vector = self._feature_vector(current_parameters)
        target = self._targets[defect_key]
        scaled = target.scaler.transform(vector)
        
        if target.poly_features is not None:
            scaled = target.poly_features.transform(scaled)
            
        prediction = target.model.predict(scaled)
        return float(np.asarray(prediction).reshape(-1)[0])

    def local_parameter_slopes(
        self,
        *,
        current_parameters: dict[str, float],
        defect_label: str,
        step_size: float = 0.1,
    ) -> tuple[ParameterSlope, ...]:
        defect_key = self.resolve_defect_key(defect_label)
        if defect_key is None:
            raise KeyError(f"Unsupported defect label for suggestion model: {defect_label}")

        current = self._canonicalize_parameters(current_parameters)
        current_vector = self._feature_vector(current)
        target = self._targets[defect_key]
        current_scaled = target.scaler.transform(current_vector)
        slopes: list[ParameterSlope] = []
        normalized_half_span = max(float(step_size), 1e-6)
        
        global_trends = self._global_slope_cache.get(defect_key, {})

        for feature_index, feature_key in enumerate(self._feature_keys):
            lower_bound, upper_bound = self._feature_bounds[feature_key]

            lower_vector = current_vector.copy()
            lower_vector[0, feature_index] = lower_bound
            lower_scaled = float(target.scaler.transform(lower_vector)[0, feature_index])

            upper_vector = current_vector.copy()
            upper_vector[0, feature_index] = upper_bound
            upper_scaled = float(target.scaler.transform(upper_vector)[0, feature_index])

            center_scaled = float(current_scaled[0, feature_index])
            probe_low = max(lower_scaled, center_scaled - normalized_half_span)
            probe_high = min(upper_scaled, center_scaled + normalized_half_span)
            if probe_high <= probe_low:
                slope = 0.0
            else:
                low_scaled = current_scaled.copy()
                high_scaled = current_scaled.copy()
                low_scaled[0, feature_index] = probe_low
                high_scaled[0, feature_index] = probe_high
                y_low = self._predict_ratio_from_scaled(low_scaled, defect_key)
                y_high = self._predict_ratio_from_scaled(high_scaled, defect_key)
                slope = (y_high - y_low) / (probe_high - probe_low)
            
            slopes.append(
                ParameterSlope(
                    feature_key=feature_key, 
                    slope=float(slope),
                    global_slope=global_trends.get(feature_key)
                )
            )

        slopes.sort(key=lambda item: (-item.absolute_slope, item.feature_key))
        return tuple(slopes)

    def top_suggestions(
        self,
        *,
        current_parameters: dict[str, float],
        defect_label: str,
        measured_ratio: float,
        target_ratio: float = 0.0,
        top_k: int = 3,
        sample_count: int = 512,
        random_seed: int = 7,
        feature_keys_to_optimize: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[SuggestionCandidate, ...]:
        defect_key = self.resolve_defect_key(defect_label)
        if defect_key is None:
            raise KeyError(f"Unsupported defect label for suggestion model: {defect_label}")

        current = self._canonicalize_parameters(current_parameters)
        selected_feature_keys = self._normalize_selected_feature_keys(feature_keys_to_optimize)
        baseline_prediction = self.predict_ratio(current, defect_key)
        delta_target = float(target_ratio) - float(measured_ratio)

        candidate_rows: list[dict[str, Any]] = []
        seen: set[tuple[float, ...]] = set()
        for parameters in self._iter_candidate_parameters(
            current_parameters=current,
            sample_count=sample_count,
            random_seed=random_seed,
            feature_keys_to_optimize=selected_feature_keys,
        ):
            candidate_key = tuple(round(float(parameters[name]), 6) for name in self._feature_keys)
            if candidate_key in seen:
                continue
            seen.add(candidate_key)

            if all(abs(float(parameters[name]) - float(current[name])) <= 1e-9 for name in self._feature_keys):
                continue

            predicted_ratio = self.predict_ratio(parameters, defect_key)
            delta_model = predicted_ratio - baseline_prediction
            expected_ratio = float(measured_ratio) + delta_model
            ranking_error = abs(delta_model - delta_target)
            candidate_rows.append(
                {
                    "parameter_values": parameters,
                    "predicted_ratio": predicted_ratio,
                    "expected_ratio": expected_ratio,
                    "delta_model": delta_model,
                    "ranking_error": ranking_error,
                }
            )

        improving_rows = [
            row
            for row in candidate_rows
            if row["predicted_ratio"] <= baseline_prediction or row["expected_ratio"] <= float(measured_ratio)
        ]
        ranked_rows = improving_rows or candidate_rows
        ranked_rows.sort(
            key=lambda row: (
                row["ranking_error"],
                row["expected_ratio"],
                row["predicted_ratio"],
                sum(
                    1
                    for name in self._feature_keys
                    if abs(float(row["parameter_values"][name]) - float(current[name])) > 1e-9
                ),
            )
        )

        top_rows = ranked_rows[: max(1, top_k)]
        defect_display = self._display_label_for_key(defect_key, fallback_label=defect_label)
        return tuple(
            SuggestionCandidate(
                rank=index,
                defect_key=defect_key,
                defect_label=defect_display,
                parameter_values={
                    name: float(row["parameter_values"][name]) for name in self._feature_keys
                },
                predicted_ratio=float(row["predicted_ratio"]),
                expected_ratio=float(row["expected_ratio"]),
                delta_model=float(row["delta_model"]),
                ranking_error=float(row["ranking_error"]),
            )
            for index, row in enumerate(top_rows, start=1)
        )

    def _feature_vector(self, current_parameters: dict[str, float]) -> np.ndarray:
        canonical = self._canonicalize_parameters(current_parameters)
        return np.asarray(
            [[float(canonical[feature_key]) for feature_key in self._feature_keys]],
            dtype=float,
        )

    def _predict_ratio_from_scaled(self, scaled_vector: np.ndarray, defect_key: str) -> float:
        target = self._targets[defect_key]
        
        vector = scaled_vector
        if target.poly_features is not None:
            vector = target.poly_features.transform(vector)
            
        prediction = target.model.predict(vector)
        return float(np.asarray(prediction).reshape(-1)[0])

    def _canonicalize_parameters(self, current_parameters: dict[str, float]) -> dict[str, float]:
        canonical = {str(key): float(value) for key, value in current_parameters.items()}
        for old_key, new_key in _FEATURE_ALIASES.items():
            if old_key in canonical and new_key not in canonical:
                canonical[new_key] = canonical[old_key]
        missing = [feature_key for feature_key in self._feature_keys if feature_key not in canonical]
        if missing:
            raise ValueError(
                "Missing current process parameters required by the suggestion model: "
                f"{missing}"
            )
        return {
            feature_key: float(canonical[feature_key])
            for feature_key in self._feature_keys
        }

    def _iter_candidate_parameters(
        self,
        *,
        current_parameters: dict[str, float],
        sample_count: int,
        random_seed: int,
        feature_keys_to_optimize: tuple[str, ...],
    ) -> list[dict[str, float]]:
        candidates: list[dict[str, float]] = []
        
        # Determine the effective bounds for each feature based on delta_max
        effective_bounds = {}
        for name in self._feature_keys:
            low, high = self._feature_bounds[name]
            delta = self._feature_deltas[name]
            curr = current_parameters[name]
            
            # Clip the global bounds by the current value +/- delta_max
            eff_low = max(low, curr - delta)
            eff_high = min(high, curr + delta)
            effective_bounds[name] = (eff_low, eff_high)

        midpoints = {
            name: (bounds[0] + bounds[1]) / 2.0
            for name, bounds in effective_bounds.items()
        }

        candidates.append(dict(current_parameters))
        
        # Midpoint candidate (within effective bounds)
        midpoint_candidate = dict(current_parameters)
        for feature_key in feature_keys_to_optimize:
            midpoint_candidate[feature_key] = midpoints[feature_key]
        candidates.append(midpoint_candidate)

        # Boundary candidates (within effective bounds)
        for feature_key in feature_keys_to_optimize:
            eff_low, eff_high = effective_bounds[feature_key]
            for boundary_value in (eff_low, eff_high, midpoints[feature_key]):
                candidate = dict(current_parameters)
                candidate[feature_key] = boundary_value
                candidates.append(candidate)

        # Corner candidates (within effective bounds)
        for corner_values in product((0, 1), repeat=len(feature_keys_to_optimize)):
            candidate = dict(current_parameters)
            for index, feature_key in enumerate(feature_keys_to_optimize):
                eff_low, eff_high = effective_bounds[feature_key]
                candidate[feature_key] = eff_low if corner_values[index] == 0 else eff_high
            candidates.append(candidate)

        # Random samples (within effective bounds)
        rng = np.random.default_rng(random_seed)
        for _ in range(max(0, sample_count)):
            candidate = dict(current_parameters)
            for feature_key in feature_keys_to_optimize:
                eff_low, eff_high = effective_bounds[feature_key]
                candidate[feature_key] = float(rng.uniform(eff_low, eff_high))
            candidates.append(candidate)

        return candidates

    def _normalize_selected_feature_keys(
        self,
        feature_keys_to_optimize: list[str] | tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        if feature_keys_to_optimize is None:
            return self._feature_keys
        selected: list[str] = []
        seen: set[str] = set()
        for raw_key in feature_keys_to_optimize:
            feature_key = str(raw_key).strip()
            if feature_key not in self._feature_keys:
                raise KeyError(f"Unknown feature selected for optimization: {feature_key}")
            if feature_key in seen:
                continue
            selected.append(feature_key)
            seen.add(feature_key)
        if not selected:
            raise ValueError("At least one feature must be selected for optimization.")
        return tuple(selected)

    def _display_label_for_key(self, defect_key: str, *, fallback_label: str) -> str:
        return fallback_label if self.resolve_defect_key(fallback_label) == defect_key else defect_key

    def _resolve_bundle_path(self, raw_path: str) -> Path:
        candidate = self._model_root / Path(raw_path)
        if candidate.exists():
            return candidate
        normalized = Path(raw_path)
        fallback = self._model_root / normalized.parent.name / normalized.name
        if fallback.exists():
            return fallback
        return candidate


@dataclass(frozen=True)
class _LoadedSuggestionTarget:
    defect_key: str
    trained_on_column: str
    model: Any
    scaler: Any
    poly_features: Any | None = None


def load_bundled_suggestion_model() -> SuggestionModelBundle:
    resource = files("testwittmann")
    for chunk in _BUNDLED_MODEL_RESOURCE:
        resource = resource.joinpath(chunk)
    with as_file(resource) as model_root:
        return SuggestionModelBundle(Path(model_root))


def default_process_parameter_values() -> dict[str, float]:
    return load_bundled_suggestion_model().default_parameter_values()


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in suggestion model metadata: {path}")
    return payload


def _normalize_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return token.strip("_")
