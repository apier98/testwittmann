from __future__ import annotations

from testwittmann.suggestions import load_bundled_suggestion_model


def test_bundled_suggestion_model_loads_expected_metadata() -> None:
    bundle = load_bundled_suggestion_model()

    assert bundle.feature_keys == ("t_melt", "t_mold", "inj_speed", "pack_pressure")
    assert set(bundle.supported_defect_keys) == {"sink_mark", "weld_line"}
    assert bundle.max_absolute_deltas == {
        "t_melt": 5.0,
        "t_mold": 5.0,
        "inj_speed": 10.0,
        "pack_pressure": 100.0,
    }


def test_available_defect_observations_match_supported_labels_only() -> None:
    bundle = load_bundled_suggestion_model()

    observations = bundle.available_defect_observations(
        {
            "Sink_Mark": 0.22,
            "weld line": 0.11,
            "flash": 0.35,
            "sink_mark": 0.0,
        }
    )

    assert [(item.defect_key, item.defect_label, item.measured_ratio) for item in observations] == [
        ("sink_mark", "Sink_Mark", 0.22),
        ("weld_line", "weld line", 0.11),
    ]


def test_top_suggestions_returns_three_ranked_candidates() -> None:
    bundle = load_bundled_suggestion_model()
    current = {
        "t_melt": 250.0,
        "t_mold": 75.0,
        "inj_speed": 62.5,
        "pack_pressure": 825.0,
    }

    suggestions = bundle.top_suggestions(
        current_parameters=current,
        defect_label="Sink_Mark",
        measured_ratio=0.20,
        target_ratio=0.0,
        top_k=3,
        sample_count=64,
        random_seed=123,
    )

    assert len(suggestions) == 3
    assert [item.rank for item in suggestions] == [1, 2, 3]
    assert suggestions[0].ranking_error <= suggestions[1].ranking_error <= suggestions[2].ranking_error
    assert all(item.expected_ratio <= 0.20 for item in suggestions)
    assert any(
        abs(item.parameter_values[key] - current[key]) > 1e-9
        for item in suggestions
        for key in current
    )


def test_local_parameter_slopes_are_sorted_by_absolute_value() -> None:
    bundle = load_bundled_suggestion_model()
    current = {
        "t_melt": 250.0,
        "t_mold": 75.0,
        "inj_speed": 62.5,
        "pack_pressure": 825.0,
    }

    slopes = bundle.local_parameter_slopes(
        current_parameters=current,
        defect_label="Sink_Mark",
    )

    assert {item.feature_key for item in slopes} == {
        "t_melt",
        "t_mold",
        "inj_speed",
        "pack_pressure",
    }
    assert [item.absolute_slope for item in slopes] == sorted(
        (item.absolute_slope for item in slopes),
        reverse=True,
    )


def test_top_suggestions_only_changes_selected_parameters() -> None:
    bundle = load_bundled_suggestion_model()
    current = {
        "t_melt": 250.0,
        "t_mold": 75.0,
        "inj_speed": 62.5,
        "pack_pressure": 825.0,
    }

    suggestions = bundle.top_suggestions(
        current_parameters=current,
        defect_label="Sink_Mark",
        measured_ratio=0.20,
        target_ratio=0.05,
        top_k=3,
        sample_count=64,
        random_seed=123,
        feature_keys_to_optimize=("pack_pressure", "t_mold"),
    )

    assert len(suggestions) == 3
    for suggestion in suggestions:
        changed = suggestion.changed_parameters(current)
        assert set(changed).issubset({"pack_pressure", "t_mold"})


def test_top_suggestions_respect_absolute_delta_limits_and_bounds() -> None:
    bundle = load_bundled_suggestion_model()
    current = {
        "t_melt": 258.0,
        "t_mold": 66.0,
        "inj_speed": 82.0,
        "pack_pressure": 1000.0,
    }

    suggestions = bundle.top_suggestions(
        current_parameters=current,
        defect_label="Sink_Mark",
        measured_ratio=0.20,
        target_ratio=0.05,
        top_k=3,
        sample_count=64,
        random_seed=123,
    )

    assert len(suggestions) == 3
    for suggestion in suggestions:
        assert 253.0 <= suggestion.parameter_values["t_melt"] <= 260.0
        assert 65.0 <= suggestion.parameter_values["t_mold"] <= 71.0
        assert 72.0 <= suggestion.parameter_values["inj_speed"] <= 85.0
        assert 900.0 <= suggestion.parameter_values["pack_pressure"] <= 1050.0
