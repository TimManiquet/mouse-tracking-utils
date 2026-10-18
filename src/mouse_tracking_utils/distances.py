"""Time-resolved distance extraction from preprocessed mouse trajectories.

Works on the output of :func:`preprocess_mouse_tracking_data`, which uses
standard column names: ``ppt_number``, ``repetition``, ``stimulus``,
``x_std``, ``y_std``, ``response_button_names``,
``response_button_positions_std`` and ``starting_button_position_std``.
"""

import numpy as np
import pandas as pd

from .data_object import MotorMovementData

_BASE_COLUMNS = ("ppt_number", "repetition", "stimulus", "x_std", "y_std")


def _array(value):
    if value is None:
        return np.array([], dtype=float)
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.array([], dtype=float)


def _level_values(series):
    return list(dict.fromkeys(series.tolist()))


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

    needed = list(_BASE_COLUMNS)
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

    trajectory_col = position_col if metric == "position" else "x_std"
    max_timepoints = max(len(_array(value)) for value in df[trajectory_col])
    if max_timepoints == 0:
        raise ValueError("No trajectory samples were found.")
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
        x, y = _array(row["x_std"]), _array(row["y_std"])
        if len(x) != len(y):
            raise ValueError("x and y trajectories must have equal lengths.")
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
        index = (p_index[row["ppt_number"]], r_index[row["repetition"]], i_index[row["stimulus"]])
        if metric == "position":
            series = _array(row[position_col])
            values[index + (slice(0, len(series)),)] = series
            continue
        for target_index, name in enumerate(target_names):
            target = trial_positions[name]
            if metric == "euclidean":
                series = np.hypot(x - target[0], y - target[1])
            else:
                series = _cosine_distance(x, y, start, target)
            values[index + (target_index, slice(0, len(series)))] = series

    coordinates = {
        "participants": participants,
        "repetitions": repetitions,
        "images": images,
        "timepoints": list(range(max_timepoints)),
    }
    if metric != "position":
        coordinates["targets"] = target_names
    return MotorMovementData(
        values,
        dimensions,
        coordinates,
        metadata={
            "metric": metric,
            "button_positions": {
                "available": bool(button_position_metadata),
                "available_for_all_trials": len(button_position_metadata) == len(df),
                "per_trial": button_position_metadata,
            },
        },
    )
