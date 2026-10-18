"""Column checking, filtering and preprocessing of raw mouse-tracking trial data.

Raw files from different experiments name their columns differently.
:func:`check_columns` finds (or is told) which raw column holds what, and
:func:`preprocess_mouse_tracking_data` renames them to the standard names
listed in ``DEFAULT_COLUMNS``, which the rest of the package relies on.

Conventions used throughout:

- Button rectangles are ``(bottom-left x, bottom-left y, width, height)`` in
  pixels, with the origin at the top left of the window (the system of the raw
  mouse coordinates). The "bottom" edge is therefore the larger y value.
- Standardised coordinates (``*_std``) span ``[-1, 1]`` on both axes.
"""

import json
import os
import re
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

# Columns kept at the end of preprocessing, under their standard names. A column
# that does not exist in the data is simply left out (never added empty).
DEFAULT_COLUMNS = (
    "ppt_number",
    "repetition",
    "stimulus",
    "response",
    "accuracy",
    "mouse_tracking_targets",
    "response_button_names",
    "screen_width",
    "screen_height",
    "starting_button_position",
    "starting_button_position_std",
    "response_button_positions",
    "response_button_positions_std",
    "time",
    "x_std",
    "y_std",
)

# Input columns the preprocessing needs (under any of their usual names).
REQUIRED_INPUT_COLUMNS = (
    "response",
    "accuracy",
    "stimulus",
    "mouse_tracking_data",
    "response_button_names",
    "screen_width",
    "screen_height",
    "response_button_positions",
    "starting_button_position",
)

# Input columns used when available. ``trial_type`` and ``event`` only serve to
# select rows (mouse-tracking and main-task trials), which is optional.
OPTIONAL_INPUT_COLUMNS = (
    "trial_type",
    "event",
    "mouse_tracking_targets",
    "repetition",
    "ppt_number",
)

# Names under which each standard column is looked for, in order of preference.
_COLUMN_ALIASES = {
    "trial_type": ("trial_type",),
    "event": ("event",),
    "response": ("response", "pressed_button", "button_pressed"),
    "accuracy": ("accuracy", "acc", "correct", "is_correct"),
    "stimulus": ("stimulus", "stimulus_name", "filename", "image"),
    "mouse_tracking_data": ("mouse_tracking_data", "mouse_data"),
    "mouse_tracking_targets": ("mouse_tracking_targets",),
    "response_button_names": ("response_button_names", "response_buttons", "button_names", "targets"),
    "screen_width": ("screen_width", "window_width"),
    "screen_height": ("screen_height", "window_height"),
    "response_button_positions": ("response_button_positions", "button_positions", "target_positions"),
    "starting_button_position": ("starting_button_position", "starting_box", "start_button_position"),
    "repetition": ("repetition",),
    "ppt_number": ("ppt_number", "ppt_id", "participant_id", "participant", "subject", "ppt"),
}

_COMPUTED_COLUMNS = (
    "time",
    "x_std",
    "y_std",
    "response_button_positions_std",
    "starting_button_position_std",
)
_MAIN_PATTERN = re.compile(r"(?<![a-z])main(?![a-z])", re.IGNORECASE)
_TRAINING_PATTERN = re.compile(
    r"(?<![a-z])(?:train(?:ing)?|practice|habituation)(?![a-z])",
    re.IGNORECASE,
)
_IMAGE_PATTERN = re.compile(r"([^/\\\"'<>=\s]+)\.(?:png|jpe?g|gif|bmp|webp|tiff?)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------

def _parse_json(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _is_missing(value):
    return value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))


def _is_number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _is_rectangle(value):
    """A rectangle is (bottom-left x, bottom-left y, width, height)."""
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str)
        and len(value) == 4
        and all(_is_number(item) for item in value)
    )


def _rectangle_center(rectangle):
    """Centre of a rectangle; the bottom edge is the larger y value (top-left origin)."""
    x, y, width, height = (float(item) for item in rectangle)
    return (x + width / 2.0, y - height / 2.0)


def _image_id(value):
    """Image ID from a stimulus: the image file name without extension, also inside html."""
    if not isinstance(value, str):
        return value
    match = _IMAGE_PATTERN.search(value)
    return match.group(1) if match else os.path.splitext(os.path.basename(value))[0]


def _trajectory_arrays(value):
    value = _parse_json(value)
    if not isinstance(value, list):
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)
    valid = [point for point in value if isinstance(point, Mapping)]
    x = np.asarray([point["x"] for point in valid if "x" in point], dtype=float)
    y = np.asarray([point["y"] for point in valid if "y" in point], dtype=float)
    time = np.asarray([point["t"] for point in valid if "t" in point], dtype=float)
    return x, y, time


def _parse_names(value):
    """Response-button names as a list of strings, or None if unavailable."""
    value = _parse_json(value)
    if isinstance(value, Mapping):
        return [str(name) for name in value]
    if isinstance(value, Sequence) and not isinstance(value, str) and len(value) > 0:
        return [str(name) for name in value]
    return None


def _parse_rectangles(value):
    """Response-button rectangles as a list of (x, y, width, height) tuples when possible."""
    if _is_missing(value):
        return None
    value = _parse_json(value)
    if isinstance(value, Sequence) and not isinstance(value, str):
        if _is_rectangle(value):
            return [tuple(float(item) for item in value)]
        if len(value) > 0 and all(_is_rectangle(item) for item in value):
            return [tuple(float(item) for item in rectangle) for rectangle in value]
    return value


def _parse_start_rectangle(value):
    """Start button as an (x, y, width, height) tuple, or None.

    Also accepts a browser bounding box (``{"x", "y", "width", "height"}``,
    top-left corner), which is converted to the bottom-left convention.
    """
    if _is_missing(value):
        return None
    value = _parse_json(value)
    if isinstance(value, Mapping) and {"x", "y", "width", "height"} <= set(value):
        return (
            float(value["x"]),
            float(value["y"]) + float(value["height"]),
            float(value["width"]),
            float(value["height"]),
        )
    if _is_rectangle(value):
        return tuple(float(item) for item in value)
    return None


def _center(item):
    """Centre (x, y) in pixels of a rectangle, bounding box or point, or None."""
    item = _parse_json(item)
    if isinstance(item, Mapping):
        if {"x", "y", "width", "height"} <= set(item):
            return (float(item["x"]) + float(item["width"]) / 2.0, float(item["y"]) + float(item["height"]) / 2.0)
        if "x" in item and "y" in item:
            return (float(item["x"]), float(item["y"]))
        if "position" in item:
            return _center(item["position"])
        return None
    if _is_rectangle(item):
        return _rectangle_center(item)
    if isinstance(item, Sequence) and not isinstance(item, str) and len(item) == 2 and all(_is_number(v) for v in item):
        return (float(item[0]), float(item[1]))
    return None


def _target_position_map(value, names=None):
    """Button centres in pixels as ``{name: (x, y)}``.

    ``value`` is a list of rectangles (named with ``names``, or by index) or a
    mapping from name to rectangle / point.
    """
    if _is_missing(value):
        return {}
    value = _parse_json(value)
    if isinstance(value, Mapping):
        centers = {str(name): _center(position) for name, position in value.items()}
        return {name: point for name, point in centers.items() if point is not None}
    if isinstance(value, Sequence) and not isinstance(value, str):
        items = [value] if _is_rectangle(value) else value
        result = {}
        for index, item in enumerate(items):
            point = _center(item)
            if point is not None:
                name = names[index] if names is not None and index < len(names) else str(index)
                result[name] = point
        return result
    return {}


def _screen_size(row):
    width, height = row.get("screen_width"), row.get("screen_height")
    if _is_missing(width) or _is_missing(height) or not (float(width) > 0 and float(height) > 0):
        raise ValueError(f"Invalid screen size ({width!r}, {height!r}) in trial {row.name!r}.")
    return float(width), float(height)


def _add_screen_size_columns(df):
    """Create screen_width / screen_height from a ``screen_dimensions`` JSON column if needed."""
    if "screen_dimensions" not in df.columns:
        return df
    wanted = (
        ("screen_width", "window_width", ("window_width", "width")),
        ("screen_height", "window_height", ("window_height", "height")),
    )
    output = df
    dimensions = None
    for name, alias, keys in wanted:
        if name in df.columns or alias in df.columns:
            continue
        if dimensions is None:
            dimensions = df["screen_dimensions"].map(_parse_json)
            output = df.copy()
        output[name] = dimensions.map(
            lambda value: next((value[key] for key in keys if key in value), np.nan)
            if isinstance(value, Mapping)
            else np.nan
        )
    return output


# ---------------------------------------------------------------------------
# Column checking and renaming
# ---------------------------------------------------------------------------

def _detect_column(df, lowered, name):
    """First usual name of ``name`` found in the data; names holding data come first."""
    present = [lowered[alias.lower()] for alias in _COLUMN_ALIASES[name] if alias.lower() in lowered]
    with_data = [column for column in present if df[column].notna().any()]
    return (with_data or present or [None])[0]


def check_columns(input_df, column_map=None, required=None):
    """Check which input column holds which information, before preprocessing.

    Expected columns, under their standard names (any of their usual aliases
    is also recognised, e.g. ``correct`` for ``accuracy``, ``targets`` for
    ``response_button_names``):

    - required: ``response``, ``accuracy``, ``stimulus``, ``mouse_tracking_data``,
      ``response_button_names``, ``screen_width``, ``screen_height``,
      ``response_button_positions``, ``starting_button_position``
    - optional: ``trial_type``, ``event`` (to select main-task mouse-tracking
      trials), ``mouse_tracking_targets``, ``repetition``, ``ppt_number``

    ``screen_width`` / ``screen_height`` are also created from a
    ``screen_dimensions`` JSON column (``window_width`` / ``window_height``)
    when absent, as :func:`preprocess_mouse_tracking_data` does.

    Args:
        input_df: raw dataframe.
        column_map: optional ``{standard_name: column_in_your_data}``, e.g.
            ``{"accuracy": "correct"}``. Columns not mentioned are detected.
        required: standard names that must be found; defaults to the required
            columns above.

    Returns:
        dict mapping each standard name found to the column holding it.

    Raises:
        ValueError: if a required column cannot be found, or a mapping is invalid.
    """
    if not isinstance(input_df, pd.DataFrame):
        raise TypeError("input_df must be a pandas DataFrame.")
    required = REQUIRED_INPUT_COLUMNS if required is None else tuple(required)
    known = REQUIRED_INPUT_COLUMNS + OPTIONAL_INPUT_COLUMNS
    column_map = dict(column_map or {})
    unknown = sorted(set(column_map) - set(known))
    if unknown:
        raise ValueError(f"Unknown standard column names {unknown!r}; choose from {list(known)!r}.")

    df = _add_screen_size_columns(input_df)
    lowered = {str(column).lower(): column for column in df.columns}
    mapping, missing = {}, []
    for name in known:
        if name in column_map:
            if column_map[name] not in df.columns:
                raise ValueError(f"Column {column_map[name]!r}, given for {name!r}, is not in the dataframe.")
            mapping[name] = column_map[name]
            continue
        source = _detect_column(df, lowered, name)
        if source is not None:
            mapping[name] = source
        elif name in required:
            missing.append(name)
    if missing:
        raise ValueError(
            f"Could not find the required column(s) {missing!r}. Rename them in your data, or "
            f"pass column_map={{{', '.join(repr(name) + ': <your column>' for name in missing)}}}."
        )
    sources = list(mapping.values())
    duplicated = sorted({str(source) for source in sources if sources.count(source) > 1})
    if duplicated:
        raise ValueError(f"Column(s) {duplicated!r} were assigned to more than one standard name.")
    return mapping


def _rename_to_standard(df, mapping):
    """Rename source columns to their standard names (dropping clashing leftovers)."""
    sources = set(mapping.values())
    clashing = [name for name, source in mapping.items() if source != name and name in df.columns and name not in sources]
    return df.drop(columns=clashing).rename(
        columns={source: name for name, source in mapping.items() if source != name}
    )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _filter_rows(df, trial_type_value=None, main_event=None, filter_training=True):
    """Keep the mouse-tracking trials of the main task (uses standard column names)."""
    output = df
    if "mouse_tracking_data" in output.columns:
        output = output.loc[output["mouse_tracking_data"].notna()]
    if trial_type_value is not None and "trial_type" in output.columns:
        output = output.loc[output["trial_type"] == trial_type_value]
    if filter_training and "event" in output.columns:
        # missing events become "" (astype(str) keeps NaN under pandas' string dtype)
        events = output["event"].astype(object).where(output["event"].notna(), "").astype(str)
        is_main = events.map(lambda item: bool(_MAIN_PATTERN.search(item)))
        is_training = events.map(lambda item: bool(_TRAINING_PATTERN.search(item)))
        if main_event is not None:
            output = output.loc[events.str.contains(main_event, case=False, regex=False)]
        elif is_main.any():
            output = output.loc[is_main]
        elif is_training.any():
            output = output.loc[~is_training]
    return output


def _select_columns(df, columns=None):
    """Keep the listed columns that exist, in the listed order (nothing is added)."""
    columns = DEFAULT_COLUMNS if columns is None else tuple(columns)
    selected = [column for column in columns if column in df.columns]
    return df.loc[:, selected].copy()


def filter_df(
    input_df,
    *,
    trial_type_value=None,
    main_event=None,
    keep_columns=None,
    filter_training=True,
):
    """Select mouse-tracking trials and keep only the default columns.

    Expects the standard column names (see :func:`check_columns`). Rows without
    ``mouse_tracking_data`` are dropped. ``trial_type_value`` additionally keeps
    one trial type, and when an ``event`` column holds main-task and training
    markers only the main-task rows are kept (``main_event`` sets a custom
    marker). Columns are then restricted to ``keep_columns``
    (``DEFAULT_COLUMNS`` by default); those missing from the data are skipped.
    """
    if not isinstance(input_df, pd.DataFrame):
        raise TypeError("input_df must be a pandas DataFrame.")
    output = _filter_rows(input_df, trial_type_value, main_event, filter_training)
    return _select_columns(output, keep_columns)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_mouse_tracking_data(
    input_df,
    *,
    column_map=None,
    required_columns=None,
    invert_y=True,
    filter_data=True,
    filter_kwargs=None,
):
    """Rename, filter, parse and standardise raw trial data in one step.

    Steps: (1) :func:`check_columns` finds the input columns (``column_map``
    overrides the detection) and they are renamed to the standard names;
    (2) mouse-tracking trials of the main task are kept (``filter_data``,
    ``filter_kwargs`` is forwarded to the row filter); (3) trajectories are
    parsed and scaled to ``[-1, 1]`` by the screen size (y is flipped so up is
    positive when ``invert_y``); (4) only the columns of ``DEFAULT_COLUMNS``
    that exist are kept.

    Output columns (standard names):

    - ``ppt_number``: participant (1 if the data has none),
    - ``repetition``: how many times the participant has seen the image so far,
    - ``stimulus``: image ID (file name without extension),
    - ``response``, ``accuracy``, ``mouse_tracking_targets``,
    - ``response_button_names``: list of button labels,
    - ``screen_width``, ``screen_height``,
    - ``starting_button_position`` and ``response_button_positions``: pixel
      rectangles (bottom-left x, bottom-left y, width, height), with their
      standardised centres in ``starting_button_position_std`` and
      ``response_button_positions_std`` (``{name: (x, y)}``),
    - ``time``, ``x_std``, ``y_std``: the trajectory samples.

    ``input_df`` can also be a list of dataframes (one per participant); each
    is preprocessed separately, gets ``ppt_number`` 1, 2, ... in list order,
    and the results are concatenated with a fresh index.
    """
    options = dict(
        column_map=column_map,
        required_columns=required_columns,
        invert_y=invert_y,
        filter_data=filter_data,
        filter_kwargs=filter_kwargs,
    )
    if isinstance(input_df, (list, tuple)):
        if len(input_df) == 0:
            raise ValueError("input_df is an empty list.")
        processed = []
        for ppt_number, single_df in enumerate(input_df, start=1):
            result = preprocess_mouse_tracking_data(single_df, **options)
            result["ppt_number"] = ppt_number
            processed.append(result)
        return pd.concat(processed, ignore_index=True)
    if not isinstance(input_df, pd.DataFrame):
        raise TypeError("input_df must be a pandas DataFrame or a list of DataFrames.")

    # 1. standard column names
    df = _add_screen_size_columns(input_df)
    mapping = check_columns(df, column_map, required_columns)
    df = _rename_to_standard(df, mapping)

    # 2. trials to keep
    if filter_data:
        df = _filter_rows(df, **(filter_kwargs or {}))
    output = df.copy()
    # columns computed below must not carry over stale values from the input
    output = output.drop(columns=list(_COMPUTED_COLUMNS), errors="ignore")

    # 3. participant, image and repetition
    if "ppt_number" not in output.columns:
        output["ppt_number"] = 1
    output["stimulus"] = output["stimulus"].map(_image_id)
    if "repetition" not in output.columns:
        output["repetition"] = output.groupby(["ppt_number", "stimulus"], dropna=False).cumcount() + 1

    # 4. button metadata
    output["response_button_names"] = output["response_button_names"].map(_parse_names)
    if "response_button_positions" in output.columns:
        output["response_button_positions"] = output["response_button_positions"].map(_parse_rectangles)
    if "starting_button_position" in output.columns:
        output["starting_button_position"] = output["starting_button_position"].map(_parse_start_rectangle)

    # 5. trajectories
    unpacked = output["mouse_tracking_data"].map(_trajectory_arrays)
    xs = unpacked.map(lambda trajectory: trajectory[0])
    ys = unpacked.map(lambda trajectory: trajectory[1])
    times = unpacked.map(lambda trajectory: trajectory[2])
    mismatched = output.index[xs.map(len) != ys.map(len)]
    if len(mismatched):
        raise ValueError(f"x and y trajectories must have the same number of samples (rows: {list(mismatched)!r}).")
    mismatched = output.index[(times.map(len) != 0) & (times.map(len) != xs.map(len))]
    if len(mismatched):
        raise ValueError(f"time and trajectory arrays must have the same number of samples (rows: {list(mismatched)!r}).")
    times = pd.Series(
        [t if len(t) else np.arange(len(x), dtype=float) for t, x in zip(times, xs)], index=output.index, dtype=object
    )

    flip = -1.0 if invert_y else 1.0

    def standardise(point, width, height):
        return (2.0 * point[0] / width - 1.0, flip * (2.0 * point[1] / height - 1.0))

    x_std, y_std, button_std, start_std = [], [], [], []
    for (_, row), x, y in zip(output.iterrows(), xs, ys):
        width, height = _screen_size(row)
        x_std.append(2.0 * x / width - 1.0)
        y_std.append(flip * (2.0 * y / height - 1.0))
        centers = _target_position_map(row.get("response_button_positions"), row["response_button_names"])
        button_std.append({name: standardise(point, width, height) for name, point in centers.items()})
        start = _center(row["starting_button_position"]) if "starting_button_position" in output.columns else None
        start_std.append(None if start is None else standardise(start, width, height))
    output["time"] = times
    output["x_std"] = pd.Series(x_std, index=output.index, dtype=object)
    output["y_std"] = pd.Series(y_std, index=output.index, dtype=object)
    if "response_button_positions" in output.columns:
        output["response_button_positions_std"] = pd.Series(button_std, index=output.index, dtype=object)
    if "starting_button_position" in output.columns:
        output["starting_button_position_std"] = pd.Series(start_std, index=output.index, dtype=object)

    # 6. keep the standard columns only
    return _select_columns(output)
