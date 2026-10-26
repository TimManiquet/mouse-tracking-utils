"""Column checking and preprocessing of raw mouse-tracking trial data.

Raw files from different experiments name their columns differently.
:func:`check_columns` finds (or is told) which raw column holds what, and
:func:`preprocess_mouse_tracking_data` renames them to the standard names
listed in ``DEFAULT_COLUMNS``, which the rest of the package relies on.

Conventions used throughout:

- Button rectangles are ``(bottom-left x, bottom-left y, width, height)`` in
  pixels, with the origin at the top left of the window (the system of the raw
  mouse coordinates). The "bottom" edge is therefore the larger y value.
- Standardised coordinates (``*_std``) have their origin at the screen centre
  and, by default, one unit is half of the shorter screen side on both axes.
"""

import json
import os
import re
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ._progress import track

# Standard names of the columns the package works with (used by filter_df).
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

# Input columns used when available. ``trial_type`` and ``event`` are not used
# by the pipeline itself; they help to write a ``row_mask``.
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
    scaling="min_side",
    row_mask=None,
    progress=True,
):
    """Parse and standardise raw trial data, keeping all rows and columns.

    The output has the same rows and columns as the input, plus the columns
    computed here. Nothing is dropped or filtered; only the rows that can be
    processed are filled in, the others get empty values in the new columns.
    A row is processed when it has mouse-tracking samples, a stimulus, a
    parsable list of response-button names, a valid screen size and (when
    those columns exist) parsable response-button and start-button positions.
    Rows that have mouse-tracking samples but fail another check are left
    unprocessed and reported in a warning. :func:`processed_trial_mask` gives
    the processed rows of the output.

    Steps: (1) :func:`check_columns` finds the columns holding the needed
    information (``column_map`` overrides the detection). Where a column has
    another name than the standard one, a copy is added under the standard
    name; your own column is kept. (2) In the processed rows, the stimulus
    becomes an image ID, button names and positions are parsed, and
    trajectories are standardised (see ``scaling``).

    Added or filled-in columns (standard names):

    - ``ppt_number``: participant (1 if the data has none),
    - ``repetition``: how many times the participant has seen the image so far,
    - ``stimulus``: image ID (file name without extension),
    - ``response_button_names``: list of button labels,
    - ``starting_button_position`` and ``response_button_positions``: pixel
      rectangles (bottom-left x, bottom-left y, width, height), with their
      standardised centres in ``starting_button_position_std`` and
      ``response_button_positions_std`` (``{name: (x, y)}``),
    - ``time``, ``x_std``, ``y_std``: the trajectory samples,
    - ``std_scale``: pixels per standardised unit along (x, y) in that trial.

    Args:
        input_df: raw dataframe, or a list of dataframes (one per
            participant): each is preprocessed separately, gets ``ppt_number``
            1, 2, ... in list order, and the results are concatenated with a
            fresh index.
        column_map: optional ``{standard_name: your_column}`` saying where the
            needed information is, e.g. ``{"accuracy": "correct"}``.
        required_columns: standard names that must be found (see
            :func:`check_columns`).
        invert_y: flip y so that up is positive.
        scaling: how pixels become standardised units, with the origin at the
            screen centre. ``"min_side"`` (default) divides both axes by half
            of the shorter screen side, so distances and angles keep their
            true proportions (a circle of buttons stays a circle; on a
            landscape screen x spans ``[-width/height, width/height]`` and y
            ``[-1, 1]``). ``"stretch"`` maps both axes to ``[-1, 1]``, which
            squeezes the longer side.
        progress: show a progress bar (per row, or per participant for a list
            of dataframes). Needs tqdm; silently skipped without it.
        row_mask: optionally restrict which rows are processed, e.g.
            ``lambda df: df["event"] != "training_trial"``. A function of the
            dataframe (after step 1) or a boolean array; for a list of
            dataframes it must be a function.
    """
    if scaling not in ("min_side", "stretch"):
        raise ValueError("scaling must be 'min_side' or 'stretch'.")
    options = dict(
        column_map=column_map,
        required_columns=required_columns,
        invert_y=invert_y,
        scaling=scaling,
        row_mask=row_mask,
        progress=False,
    )
    if isinstance(input_df, (list, tuple)) and row_mask is not None and not callable(row_mask):
        raise TypeError("For a list of dataframes, row_mask must be a function of the dataframe.")
    if isinstance(input_df, (list, tuple)):
        if len(input_df) == 0:
            raise ValueError("input_df is an empty list.")
        processed = []
        for ppt_number, single_df in track(
            enumerate(input_df, start=1), progress, total=len(input_df), desc="Preprocessing", unit="participant"
        ):
            result = preprocess_mouse_tracking_data(single_df, **options)
            result["ppt_number"] = ppt_number
            processed.append(result)
        return pd.concat(processed, ignore_index=True)
    if not isinstance(input_df, pd.DataFrame):
        raise TypeError("input_df must be a pandas DataFrame or a list of DataFrames.")

    # 1. standard names: add a copy of each mapped column under its standard name
    df = _add_screen_size_columns(input_df)
    mapping = check_columns(df, column_map, required_columns)
    output = df.copy()
    for name, source in mapping.items():
        if source == name:
            continue
        if name in output.columns and name not in mapping.values():
            output = output.rename(columns={name: f"{name}_original"})
        output[name] = output[source]
    # computed columns must not carry over stale values from the input
    output = output.drop(columns=list(_COMPUTED_COLUMNS) + ["std_scale"], errors="ignore")

    # 2. rows that can be processed
    if row_mask is not None:
        wanted = row_mask(output) if callable(row_mask) else row_mask
        wanted = np.asarray(wanted, dtype=bool)
        if wanted.shape != (len(output),):
            raise ValueError("row_mask must give one boolean per row.")
    else:
        wanted = np.ones(len(output), dtype=bool)

    if "ppt_number" not in output.columns:
        output["ppt_number"] = 1
    n_rows = len(output)
    columns = {
        name: [None] * n_rows
        for name in ("time", "x_std", "y_std", "response_button_positions_std", "starting_button_position_std", "std_scale")
    }
    stimulus = output["stimulus"].tolist()
    names_col = output["response_button_names"].tolist()
    positions_col = output["response_button_positions"].tolist() if "response_button_positions" in output else None
    start_col = output["starting_button_position"].tolist() if "starting_button_position" in output else None
    processed = np.zeros(n_rows, dtype=bool)
    skipped = []
    flip = -1.0 if invert_y else 1.0

    for i, (_, row) in track(
        enumerate(output.iterrows()), progress, total=n_rows, desc="Preprocessing", unit="row"
    ):
        x, y, t = _trajectory_arrays(row["mouse_tracking_data"])
        if len(x) == 0 or not wanted[i]:
            continue
        names = _parse_names(row["response_button_names"])
        try:
            width, height = _screen_size(row)
        except ValueError:
            width = None
        if width is None or names is None or _is_missing(row["stimulus"]) or len(x) != len(y) or (
            len(t) and len(t) != len(x)
        ):
            skipped.append(output.index[i])
            continue
        if scaling == "min_side":
            scale_x = scale_y = min(width, height) / 2.0
        else:
            scale_x, scale_y = width / 2.0, height / 2.0

        def standardise(point, scale_x=scale_x, scale_y=scale_y, width=width, height=height):
            return ((point[0] - width / 2.0) / scale_x, flip * (point[1] - height / 2.0) / scale_y)

        stimulus[i] = _image_id(row["stimulus"])
        names_col[i] = names
        columns["time"][i] = t if len(t) else np.arange(len(x), dtype=float)
        columns["x_std"][i] = (x - width / 2.0) / scale_x
        columns["y_std"][i] = flip * (y - height / 2.0) / scale_y
        columns["std_scale"][i] = (scale_x, scale_y)
        if positions_col is not None:
            positions_col[i] = _parse_rectangles(row["response_button_positions"])
            centers = _target_position_map(row["response_button_positions"], names)
            columns["response_button_positions_std"][i] = {
                name: standardise(point) for name, point in centers.items()
            }
        if start_col is not None:
            start_col[i] = _parse_start_rectangle(row["starting_button_position"])
            center = _center(row["starting_button_position"])
            columns["starting_button_position_std"][i] = None if center is None else standardise(center)
        processed[i] = True

    if skipped:
        warnings.warn(
            f"{len(skipped)} row(s) with mouse-tracking data were left unprocessed because a needed value "
            f"(stimulus, button names, screen size or matching sample counts) is missing or invalid; "
            f"row labels: {skipped[:10]!r}{'...' if len(skipped) > 10 else ''}.",
            stacklevel=2,
        )

    output["stimulus"] = pd.Series(stimulus, index=output.index, dtype=object)
    output["response_button_names"] = pd.Series(names_col, index=output.index, dtype=object)
    if positions_col is not None:
        output["response_button_positions"] = pd.Series(positions_col, index=output.index, dtype=object)
    if start_col is not None:
        output["starting_button_position"] = pd.Series(start_col, index=output.index, dtype=object)
    for name, values in columns.items():
        if name == "response_button_positions_std" and positions_col is None:
            continue
        if name == "starting_button_position_std" and start_col is None:
            continue
        output[name] = pd.Series(values, index=output.index, dtype=object)

    # repetition: how often each participant has seen each image (processed rows only)
    if "repetition" not in output.columns:
        counts = output.loc[processed].groupby(["ppt_number", "stimulus"], dropna=False).cumcount() + 1
        output["repetition"] = counts.reindex(output.index)
    return output


def processed_trial_mask(df):
    """Boolean mask of the rows holding a preprocessed trial (they have a trajectory)."""
    if "x_std" not in df.columns:
        raise ValueError("Column 'x_std' not found; run preprocess_mouse_tracking_data first.")
    return df["x_std"].map(lambda value: isinstance(value, np.ndarray) and len(value) > 0).to_numpy(dtype=bool)
