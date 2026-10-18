import json

import numpy as np
import pandas as pd
import pytest

from mouse_tracking_utils import (
    MotorMovementData,
    check_columns,
    exclude_participants,
    exclude_trials,
    extract_distance,
    filter_df,
    generate_demo_data,
    load_demo_data,
    preprocess_mouse_tracking_data,
    plot_binary_trials,
    plot_trials,
)
from mouse_tracking_utils.preprocessing import DEFAULT_COLUMNS


def _raw(images=("a.png", "b.png"), reps=2, names=("L", "R"), **overrides):
    """Raw 800x400 trials; left/right buttons are 100x50 at y-bottom 200."""
    rects = [[100, 200, 100, 50], [600, 200, 100, 50]][: len(names)]
    rows = []
    for image in images:
        for _ in range(reps):
            path = [{"x": 400 + 10 * i, "y": 300 - 10 * i, "t": 100 * i, "event": "mousemove"} for i in range(4)]
            rows.append(
                {
                    "trial_type": "html-button-response",
                    "event": "main_trial",
                    "response": 0,
                    "accuracy": 1,
                    "stimulus": f"./stim/{image}",
                    "mouse_tracking_data": json.dumps(path),
                    "response_button_names": json.dumps(list(names)),
                    "screen_width": 800,
                    "screen_height": 400,
                    "response_button_positions": json.dumps(rects),
                    "starting_button_position": json.dumps([380, 330, 40, 40]),
                }
            )
    return pd.DataFrame(rows).assign(**overrides)


def test_demo_data_end_to_end():
    for count in (2, 4, 8):
        raw = load_demo_data(count)
        out = preprocess_mouse_tracking_data(raw)
        assert set(out.columns) <= set(DEFAULT_COLUMNS)
        eu = extract_distance(out, metric="euclidean")
        co = extract_distance(out, metric="cosine")
        assert eu.dimensions == ("participants", "repetitions", "images", "targets", "timepoints")
        assert eu.values.shape[3] == count == co.values.shape[3]
        pos = extract_distance(out, metric="position", position_col="y_std")
        assert pos.dimensions == ("participants", "repetitions", "images", "timepoints")
    assert len(generate_demo_data(4, n_ppts=1, n_images=1, n_repetitions=1)) == 1
    with pytest.raises(ValueError):
        load_demo_data(3)


def test_output_has_standard_columns_and_no_empty_additions():
    raw = _raw().drop(columns=["trial_type", "event"])
    out = preprocess_mouse_tracking_data(raw)
    assert "mouse_tracking_targets" not in out.columns
    assert "mouse_tracking_data" not in out.columns
    assert list(out.columns) == [c for c in DEFAULT_COLUMNS if c in out.columns]
    assert out["stimulus"].tolist() == ["a"] * 2 + ["b"] * 2
    assert out["repetition"].tolist() == [1, 2, 1, 2]
    assert (out["ppt_number"] == 1).all()
    assert out["response_button_names"].iloc[0] == ["L", "R"]


def test_standardisation_of_trajectory_and_buttons():
    out = preprocess_mouse_tracking_data(_raw(), invert_y=False)
    np.testing.assert_allclose(out["x_std"].iloc[0][:2], [0.0, 0.025])
    np.testing.assert_allclose(out["y_std"].iloc[0][0], 0.5)
    # left button centre: (150, 175) -> (-0.625, -0.125)
    np.testing.assert_allclose(out["response_button_positions_std"].iloc[0]["L"], (-0.625, -0.125))
    np.testing.assert_allclose(out["starting_button_position_std"].iloc[0], (0.0, 0.55))
    flipped = preprocess_mouse_tracking_data(_raw())
    np.testing.assert_allclose(flipped["y_std"].iloc[0][0], -0.5)
    np.testing.assert_allclose(flipped["response_button_positions_std"].iloc[0]["L"], (-0.625, 0.125))


def test_check_columns_detects_maps_and_errors():
    raw = _raw().rename(columns={"accuracy": "correct", "response_button_names": "targets"})
    found = check_columns(raw)
    assert found["accuracy"] == "correct" and found["response_button_names"] == "targets"

    # explicit mapping wins, and a clashing empty 'accuracy' column is dropped
    raw = _raw().rename(columns={"accuracy": "correct"})
    raw["accuracy"] = np.nan
    assert check_columns(raw)["accuracy"] == "correct"
    out = preprocess_mouse_tracking_data(raw.rename(columns={"correct": "hit"}), column_map={"accuracy": "hit"})
    assert out["accuracy"].tolist() == [1] * 4

    with pytest.raises(ValueError, match="accuracy"):
        check_columns(_raw().drop(columns=["accuracy"]))
    with pytest.raises(ValueError, match="not in the dataframe"):
        check_columns(_raw(), {"accuracy": "nope"})
    with pytest.raises(ValueError, match="Unknown"):
        check_columns(_raw(), {"foo": "response"})


def test_screen_size_from_screen_dimensions_json():
    raw = _raw().drop(columns=["screen_width", "screen_height"])
    raw["screen_dimensions"] = json.dumps({"window_width": 800, "window_height": 400})
    out = preprocess_mouse_tracking_data(raw)
    assert out["screen_width"].iloc[0] == 800


def test_list_input_adds_participant_numbers_and_filters_rows():
    extra = _raw(reps=1, images=("a.png",))
    extra.loc[0, "event"] = "training_trial"
    out = preprocess_mouse_tracking_data([_raw(), extra, _raw(reps=1)])
    assert out["ppt_number"].tolist() == [1] * 4 + [3] * 2
    assert out.index.is_unique


def test_extract_distance_orders_targets_and_requires_preprocessing():
    out = preprocess_mouse_tracking_data(_raw(), invert_y=False)
    data = extract_distance(out, target_names=["r", "L"])
    assert data.coordinates["targets"] == ["R", "L"]
    np.testing.assert_allclose(
        data.values[0, 0, 0, 1, 0], np.hypot(0.625, 0.625)
    )
    assert extract_distance(out, target_names=[1, 0]).coordinates["targets"] == ["R", "L"]
    with pytest.raises(ValueError, match="preprocess"):
        extract_distance(_raw())


def test_single_repetition_is_kept():
    out = preprocess_mouse_tracking_data(_raw(reps=1))
    assert extract_distance(out).dimensions == ("participants", "repetitions", "images", "targets", "timepoints")


def test_target_order_uses_case_insensitive_button_names():
    out = preprocess_mouse_tracking_data(_raw(names=("Animate", "Inanimate")), invert_y=False)
    data = extract_distance(out, target_names=["inanimate", "animate"])
    assert data.coordinates["targets"] == ["Inanimate", "Animate"]


def test_plot_trials_colors_excluded_and_draws_button_boxes():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = preprocess_mouse_tracking_data(_raw(reps=2, images=("animate.png",)))
    out["category"] = ["animate", "inanimate"]
    ax = plot_trials(
        out,
        color_by="category",
        palette={"animate": "red", "inanimate": "blue"},
        excluded_indices=[out.index[1]],
    )
    assert len(ax.lines) == 2
    assert [line.get_color() for line in ax.lines] == ["red", "gray"]
    assert len(ax.patches) == 3
    assert ax.get_legend().get_title().get_text() == "category"
    plt.close(ax.figure)


def test_general_plot_preserves_each_trials_screen_layout():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = _raw(reps=2, images=("animate.png",))
    raw.loc[1, "response_button_names"] = json.dumps(["R", "L"])
    raw.loc[1, "response_button_positions"] = json.dumps(
        [[100, 200, 100, 50], [600, 200, 100, 50]]
    )
    out = preprocess_mouse_tracking_data(raw)

    ax = plot_trials(out)
    assert len(ax.patches) == 5
    assert [line.get_xdata()[0] for line in ax.lines] == pytest.approx([0, 0])
    assert [patch.get_x() for patch in ax.patches[1:]] == pytest.approx(
        [-0.75, 0.5, -0.75, 0.5]
    )
    plt.close(ax.figure)


def test_binary_plot_mirrors_trials_to_align_response_buttons():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = _raw(reps=2, images=("animate.png",))
    raw.loc[1, "response_button_names"] = json.dumps(["R", "L"])
    raw.loc[1, "response_button_positions"] = json.dumps(
        [[100, 200, 100, 50], [600, 200, 100, 50]]
    )
    out = preprocess_mouse_tracking_data(raw)

    ax = plot_binary_trials(out)
    assert len(ax.patches) == 3
    assert ax.lines[1].get_xdata() == pytest.approx(-np.asarray(out["x_std"].iloc[1]))
    assert [patch.get_x() for patch in ax.patches[1:]] == pytest.approx([-0.75, 0.5])
    plt.close(ax.figure)

    ax = plot_binary_trials(out, response_button_order=["R", "L"])
    assert ax.lines[1].get_xdata() == pytest.approx(np.asarray(out["x_std"].iloc[1]))
    plt.close(ax.figure)

    with pytest.raises(ValueError, match="both response buttons exactly once"):
        plot_binary_trials(out, response_button_order=["L"])


def test_binary_plot_rejects_more_than_two_buttons():
    import matplotlib
    matplotlib.use("Agg")

    out = preprocess_mouse_tracking_data(generate_demo_data(4, n_ppts=1, n_images=1, n_repetitions=1))
    with pytest.raises(ValueError, match="two response-button positions"):
        plot_binary_trials(out)


def test_data_object_describes_and_averages_itself():
    data = extract_distance(preprocess_mouse_tracking_data(_raw(), invert_y=False))
    text = str(data)
    assert "metric" in text and "repetitions" in text and "images" in text
    assert data.describe_dimensions()["images"]["labels"] == ["a", "b"]
    assert data.axis_of("targets") == 3
    averaged = data.average("repetitions")
    assert averaged.dimensions == ("participants", "images", "targets", "timepoints")
    assert averaged.metadata["averaged_over"] == ["repetitions"]
    assert repr(averaged).startswith("MotorMovementData(")


def test_named_average_and_npz_round_trip(tmp_path):
    data = MotorMovementData(
        np.array([[[1.0, 3.0], [5.0, 7.0]], [[3.0, 5.0], [7.0, 9.0]]]),
        ("participants", "images", "timepoints"),
        {"participants": ["p1", "p2"], "images": ["a", "b"], "timepoints": [0, 1]},
    )
    average = data.average(axis="participants")
    assert average.dimensions == ("images", "timepoints")
    np.testing.assert_allclose(average.values, [[2.0, 4.0], [6.0, 8.0]])
    loaded = MotorMovementData.load(data.save(tmp_path / "trajectories"))
    assert loaded.dimensions == data.dimensions
    assert loaded.coordinates == data.coordinates
    np.testing.assert_array_equal(loaded, data.values)


def test_filter_df_and_exclusions():
    raw = _raw(reps=1, images=("a.png",))
    raw = pd.concat([raw, raw.assign(trial_type="survey", mouse_tracking_data=None)], ignore_index=True)
    standard = raw.rename(columns={})
    assert len(filter_df(standard, trial_type_value="html-button-response")) == 1
    out = preprocess_mouse_tracking_data(raw)
    assert len(out) == 1
    assert len(exclude_trials(out)) == 1
    assert len(exclude_participants(out, min_accuracy=0.5)) == 1
    assert len(exclude_participants(out, min_completed_trials=2)) == 0


def test_rows_with_missing_or_training_events_are_filtered():
    raw = _raw(reps=1, images=("a.png", "b.png", "c.png"))
    raw["event"] = pd.array([None, "training_trial", "trial"], dtype="string")
    out = preprocess_mouse_tracking_data(raw)
    assert out["stimulus"].tolist() == ["a", "c"]


def _interp_df(times, xs):
    """One preprocessed trial with given sample times (ms) and x pixels (y fixed)."""
    path = [{"x": x, "y": 200, "t": t, "event": "mousemove"} for t, x in zip(times, xs)]
    raw = _raw(images=("a.png",), reps=1, mouse_tracking_data=json.dumps(path))
    return preprocess_mouse_tracking_data(raw)


def test_samples_are_linearly_interpolated_and_edges_stay_nan():
    out = _interp_df([100, 300, 600], [400, 500, 800])  # x_std 0, .25, 1
    mm = extract_distance(out, metric="position", position_col="x_std", duration=800, sampling_interval=100)
    series = mm.values[0, 0, 0]
    assert mm.coordinates["timepoints"] == [0, 100, 200, 300, 400, 500, 600, 700]
    assert np.isnan(series[0])
    assert series[1:7] == pytest.approx([0, 0.125, 0.25, 0.25 + 0.75 / 3, 0.25 + 1.5 / 3, 1.0])
    assert np.isnan(series[7])


def test_fill_start_uses_start_button_centre_and_duration_default():
    out = _interp_df([100, 300], [400, 500])
    start_x = out["starting_button_position_std"].iloc[0][0]
    mm = extract_distance(out, metric="position", fill_start=True, sampling_interval=100)
    series = mm.values[0, 0, 0]
    assert len(series) == 4  # up to the latest time (300), inclusive
    assert series[0] == pytest.approx(start_x)
    assert series[1] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        extract_distance(out.drop(columns="starting_button_position_std"), metric="position", fill_start=True)


def test_distances_follow_interpolated_positions_and_ignore_nan_padding():
    out = _interp_df([100, 300], [400, 500])
    mm = extract_distance(out, duration=500, sampling_interval=100)
    assert np.isnan(mm.values[0, 0, 0, :, 0]).all()
    assert np.isfinite(mm.values[0, 0, 0, :, 1:4]).all()
    assert np.isnan(mm.values[0, 0, 0, :, 4]).all()
    assert mm.metadata["time_unit"] == "ms"


def test_save_with_downsampling(tmp_path):
    out = _interp_df([0, 1000], [400, 500])
    mm = extract_distance(out, metric="position")
    assert mm.shape[-1] == 1001
    path = mm.save(tmp_path / "x", downsample_hz=250)
    loaded = MotorMovementData.load(path)
    assert loaded.shape[-1] == 251
    assert loaded.coordinates["timepoints"][:2] == [0, 4]
    assert loaded.metadata["sampling_interval"] == 4
    assert mm.shape[-1] == 1001
    with pytest.raises(ValueError):
        mm.downsample(5000)


def test_circular_buttons_are_drawn_as_ellipses():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.patches import Ellipse

    out = _interp_df([0, 100], [400, 500])
    ax = plot_trials(out, button_shape="circle", start_shape="circle")
    assert any(isinstance(p, Ellipse) for p in ax.patches)
    with pytest.raises(ValueError):
        plot_trials(out, button_shape="star")
