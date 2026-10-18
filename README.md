# mouse-tracking-utils

## To-do's

 - [ ] have a nice print function for the mousetrackign data object
 - [ ]

`mouse-tracking-utils` preprocesses mouse-tracking trial data and computes
time-resolved trajectory distances to response targets. It supports raw CSV
files and dataframes, and includes synthetic 2-, 4-, and 8-button demo data.

## Installation

```bash
pip install mouse-tracking-utils
```

For development from a clone:

```bash
uv sync --group dev
```

## Usage

Workflow: preprocessing (which includes row/column filtering), optional
exclusions, then distance extraction.

`check_columns(df, column_map=None)` verifies that your raw dataframe has the
columns the pipeline needs, detecting them by their usual names (e.g.
`correct` for `accuracy`, `targets` for `response_button_names`) or through
`column_map={"accuracy": "correct"}`. It raises an error listing what is
missing. Required input columns: `response`, `accuracy`, `stimulus`,
`mouse_tracking_data`, `response_button_names`, `screen_width`,
`screen_height`, `response_button_positions`, `starting_button_position`.
`trial_type` / `event` are optional and only used to keep main-task rows.
Button rectangles are `(bottom-left x, bottom-left y, width, height)` in
pixels with the origin at the top left of the window.

`preprocess_mouse_tracking_data` renames columns to standard names, keeps
mouse-tracking trials, standardises trajectories to `[-1, 1]` and returns only
these columns (those that exist): `ppt_number`, `repetition`, `stimulus`,
`response`, `accuracy`, `mouse_tracking_targets`, `response_button_names`,
`screen_width`, `screen_height`, `starting_button_position`,
`starting_button_position_std`, `response_button_positions`,
`response_button_positions_std`, `time`, `x_std`, `y_std`. It accepts a list
of dataframes (one per participant) and numbers them from 1 in `ppt_number`.
`extract_distance` relies on these names only.

```python
from mouse_tracking_utils import (
    MotorMovementData,
    extract_distance,
    load_demo_data,
    preprocess_mouse_tracking_data,
)

preprocessed = preprocess_mouse_tracking_data(load_demo_data(4))

data = extract_distance(preprocessed, metric="euclidean")  # or "cosine"
print(data)
data = extract_distance(preprocessed, target_names=["button_2", "button_1"])

position = extract_distance(preprocessed, metric="position", position_col="x_std")

group_average = data.average("participants")
data.save("distances.npz")
reloaded = MotorMovementData.load("distances.npz")
```

Distance results are `MotorMovementData` objects. Euclidean and cosine metrics
have dimensions `(participants, repetitions, images, targets, timepoints)`
(`repetitions` is retained even when there is only one); position results omit
`targets`. `target_names` sets the target dimension order by button names
(case-insensitive) or button indices; button names and positions always come
from the preprocessed dataframe. `average(axis=...)` accepts a dimension name
or numeric axis. The `.npz` format stores the array and JSON metadata without
pickle.

`plot_trials` displays one or more preprocessed trajectories in their original
screen coordinates. For binary tasks, `plot_binary_trials` mirrors trials
horizontally as needed so response buttons line up with the first trial's
left-to-right button order; use `response_button_order` to set that order.
Both functions accept `color_by` with an optional `{value: color}` palette,
and `excluded_indices` to draw excluded rows in gray. Start and response
buttons are drawn in gray behind the trajectories.

```python
from mouse_tracking_utils import plot_binary_trials, plot_trials

ax = plot_trials(preprocessed, color_by="accuracy", palette={0: "salmon", 1: "teal"})
ax.figure.show()

binary_ax = plot_binary_trials(preprocessed, response_button_order=["inanimate", "animate"])
binary_ax.figure.show()
```

## Development

```bash
uv run pytest
uv build
```
