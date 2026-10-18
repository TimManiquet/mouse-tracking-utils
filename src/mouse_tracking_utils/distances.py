"""Time-resolved distance extraction from preprocessed mouse trajectories.

Works on the output of :func:`preprocess_mouse_tracking_data`, which uses
standard column names: ``ppt_number``, ``repetition``, ``stimulus``,
``x_std``, ``y_std``, ``response_button_names``,
``response_button_positions_std`` and ``starting_button_position_std``.
"""

import numpy as np
import pandas as pd

from .data_object import MotorMovementData

_BASE_COLUMNS = ("ppt_number", "repetition", "stimulus", "x_std", "y_std", "time")


def _array(value):
    if value is None:
        return np.array([], dtype=float)
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.array([], dtype=float)


def _level_values(series):
    return list(dict.fromkeys(series.tolist()))


def _resample(time, series, grid, fill_value=None):
    """Linearly interpolate ``series`` (sampled at ``time``) onto ``grid``.

    Grid points before the first sample are NaN, unless ``fill_value`` is given:
    it is then used as the value at time 0 and the gap up to the first sample
    is interpolated. Grid points after the last sample are always NaN.
    """
    time, series = _array(time), _array(series)
    keep = np.isfinite(time) & np.isfinite(series)
    time, series = time[keep], series[keep]
    if len(time) == 0:
        return np.full(len(grid), np.nan)
    order = np.argsort(time, kind="stable")
    time, series = time[order], series[order]
    # duplicated timestamps: keep the last sample
    last = np.append(time[1:] != time[:-1], True)
    time, series = time[last], series[last]
    if fill_value is not None and time[0] > 0:
        time, series = np.insert(time, 0, 0.0), np.insert(series, 0, fill_value)
    return np.interp(grid, time, series, left=np.nan, right=np.nan)


def _cosine_distance(x, y, start, target):
    movement = np.column_stack((x - start[0], y - start[1]))
    direction = np.asarray(target, dtype=float) - np.asarray(start, dtype=float)
    norms = np.linalg.norm(movement, axis=1) * np.linalg.norm(direction)
    similarity = np.divide(movement @ direction, norms, out=np.zeros(len(movement)), where=norms > 0)
    return 1.0 - np.clip(similarity, -1.0, 1.0)


def _norm(name):
    """Comparison key for button names: case, spaces and punctuation are ignored."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _resolve_order(available, target_names):
    """Order of the targets: ``target_names`` holds names (case-insensitive) or indices."""
    if target_names is None:
        return list(available)
    lowered = {_norm(name): name for name in available}
    order = []
    for item in target_names:
        if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
            if not -len(available) <= item < len(available):
                raise ValueError(f"target_names index {item} is out of range for buttons {list(available)!r}.")
            order.append(available[item])
        elif _norm(item) in lowered:
            order.append(lowered[_norm(item)])
        else:
            raise ValueError(f"target_names contains unknown button {item!r}; available buttons: {list(available)!r}.")
    if len(set(order)) != len(order):
        raise ValueError(f"target_names lists a button twice: {order!r}.")
    return order


def extract_distance(
    input_df,
    *,
    metric="euclidean",
    target_names=None,
    position_col="x_std",
    duration=None,
    sampling_interval=1.0,
    fill_start=False,
):
    """Extract trajectory distances as a metadata-aware :class:`MotorMovementData`.

    ``input_df`` must be the output of :func:`preprocess_mouse_tracking_data`.

    Metrics:

    - ``'euclidean'``: distance of each time point to each response button.
    - ``'cosine'``: cosine distance between the movement direction (from the
      start button) and the direction of each response button.
    - ``'position'``: the position itself, taken from ``position_col``
      (``'x_std'``, ``'y_std'``, ...); there is no targets dimension.

    Dimensions: ``participants``, ``repetitions``, ``images``, [``targets``],
    ``timepoints``. Missing trials and padding samples are NaN.

    Time axis and interpolation: mouse samples are irregular, so the x and y
    positions of each trial are linearly interpolated onto a regular time grid
    (from the ``time`` column, in ms) before the distances are computed. The
    ``timepoints`` coordinate holds the grid times in ms.

    - ``duration``: length of the grid in ms (e.g. ``1000``). By default the
      latest time point found in the whole dataset is used. Samples after
      ``duration`` are ignored.
    - ``sampling_interval``: grid step in ms (default 1, i.e. one array entry
      per ms). Increase it to get smaller arrays.
    - ``fill_start``: before a trial's first sample the position is unknown, so
      the array is NaN there by default. With ``fill_start=True`` the centre of
      the start button (``starting_button_position_std``) is taken as the
      position at time 0 and interpolated up to the first sample. Time after
      the last sample (the trial has ended) is always NaN.

    Buttons and positions are read from ``response_button_names`` and
    ``response_button_positions_std``. ``target_names`` is the desired order
    of the target dimension: provide button names (matched ignoring case, spaces and punctuation, so 'firehydrant' matches 'fire hydrant')
    or indices into the names in the data. By default the order of the data is
    used. ``target_positions`` is not an argument; positions always come from
    the preprocessed dataframe.
    """
    if metric not in {"euclidean", "cosine", "position"}:
        raise ValueError("metric must be 'euclidean', 'cosine', or 'position'.")
    if not isinstance(input_df, pd.DataFrame):
        raise TypeError("input_df must be a pandas DataFrame.")
    if input_df.empty:
        raise ValueError("input_df must contain at least one trial.")
    if sampling_interval <= 0:
        raise ValueError("sampling_interval must be positive (ms).")
    if duration is not None and duration <= 0:
        raise ValueError("duration must be positive (ms).")

    needed = list(_BASE_COLUMNS)
    if fill_start:
        needed.append("starting_button_position_std")
    if metric == "position":
        needed.append(position_col)
    else:
        needed.append("response_button_names")
        needed.append("response_button_positions_std")
        if metric == "cosine":
            needed.append("starting_button_position_std")
    missing = [column for column in dict.fromkeys(needed) if column not in input_df.columns]
    if missing:
        raise ValueError(
            f"Column(s) {missing!r} not found. Run preprocess_mouse_tracking_data first and "
            "pass its output; it creates the standard columns."
        )
    df = input_df

    participants = _level_values(df["ppt_number"])
    repetitions = _level_values(df["repetition"])
    images = _level_values(df["stimulus"])

    row_positions = []
    if metric != "position":
        for _, row in df.iterrows():
            value = row["response_button_positions_std"]
            row_positions.append(value if isinstance(value, dict) else {})
        available = next(
            (row["response_button_names"] for _, row in df.iterrows() if isinstance(row["response_button_names"], list)),
            None,
        ) or next((list(found) for found in row_positions if found), None)
        if not available:
            raise ValueError("No response button names found in 'response_button_names'.")
        target_names = _resolve_order([str(name) for name in available], target_names)
        canonical = {_norm(name): name for name in target_names}
        row_positions = [{canonical.get(_norm(k), k): v for k, v in found.items()} for found in row_positions]
        for row_index, found in enumerate(row_positions):
            absent = [name for name in target_names if name not in found]
            if absent:
                raise ValueError(
                    f"Button positions are missing for {absent!r} in trial at row index "
                    f"{df.index[row_index]!r}; positions exist for {list(found)!r}."
                )

    times = [_array(value) for value in df["time"]]
    finite_max = [t[np.isfinite(t)].max() for t in times if np.isfinite(t).any()]
    if not finite_max:
        raise ValueError("No trajectory samples were found.")
    if duration is None:
        duration = max(finite_max) + sampling_interval
    grid = np.arange(0.0, duration, sampling_interval)
    if len(grid) == 0:
        raise ValueError("duration is shorter than sampling_interval.")
    if fill_start and metric == "position" and position_col not in ("x_std", "y_std"):
        raise ValueError("fill_start is only available for position_col 'x_std' or 'y_std'.")
    max_timepoints = len(grid)
    dimensions = ["participants", "repetitions", "images"]
    shape = [len(participants), len(repetitions), len(images)]
    if metric == "position":
        dimensions.append("timepoints")
        shape.append(max_timepoints)
    else:
        dimensions.extend(("targets", "timepoints"))
        shape.extend((len(target_names), max_timepoints))
    values = np.full(shape, np.nan, dtype=float)

    p_index = {value: index for index, value in enumerate(participants)}
    r_index = {value: index for index, value in enumerate(repetitions)}
    i_index = {value: index for index, value in enumerate(images)}
    button_position_metadata = []
    for row_number, (_, row) in enumerate(df.iterrows()):
        time = times[row_number]
        raw_x, raw_y = _array(row["x_std"]), _array(row["y_std"])
        if len(raw_x) != len(raw_y) or len(raw_x) != len(time):
            raise ValueError("x, y and time must have equal lengths.")
        trial_positions = row_positions[row_number] if row_positions else {}
        start = row["starting_button_position_std"] if "starting_button_position_std" in df.columns else None
        start = tuple(float(v) for v in start) if isinstance(start, (tuple, list, np.ndarray)) and len(start) == 2 else None
        if metric == "cosine" and start is None:
            raise ValueError(f"No starting button position for the trial at row index {df.index[row_number]!r}.")
        if trial_positions or start is not None:
            button_position_metadata.append(
                {
                    "participant": row["ppt_number"],
                    "repetition": row["repetition"],
                    "image": row["stimulus"],
                    "start_position": None if start is None else list(start),
                    "positions": {name: list(point) for name, point in trial_positions.items()},
                }
            )
        origin = start if fill_start else None
        if fill_start and start is None:
            raise ValueError(f"No starting button position for the trial at row index {df.index[row_number]!r}.")
        x = _resample(time, raw_x, grid, None if origin is None else origin[0])
        y = _resample(time, raw_y, grid, None if origin is None else origin[1])
        index = (p_index[row["ppt_number"]], r_index[row["repetition"]], i_index[row["stimulus"]])
        if metric == "position":
            if position_col in ("x_std", "y_std"):
                values[index] = x if position_col == "x_std" else y
            else:
                values[index] = _resample(time, row[position_col], grid)
            continue
        for target_index, name in enumerate(target_names):
            target = trial_positions[name]
            if metric == "euclidean":
                series = np.hypot(x - target[0], y - target[1])
            else:
                series = _cosine_distance(x, y, start, target)
            series = np.where(np.isfinite(x) & np.isfinite(y), series, np.nan)
            values[index + (target_index, slice(None))] = series

    coordinates = {
        "participants": participants,
        "repetitions": repetitions,
        "images": images,
        "timepoints": grid.tolist(),
    }
    if metric != "position":
        coordinates["targets"] = target_names
    return MotorMovementData(
        values,
        dimensions,
        coordinates,
        metadata={
            "metric": metric,
            "time_unit": "ms",
            "sampling_interval": float(sampling_interval),
            "duration": float(duration),
            "fill_start": bool(fill_start),
            "button_positions": {
                "available": bool(button_position_metadata),
                "available_for_all_trials": len(button_position_metadata) == len(df),
                "per_trial": button_position_metadata,
            },
        },
    )
