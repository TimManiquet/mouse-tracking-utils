# Distances

`extract_distance(preprocessed_df, metric="euclidean", target_names=None, position_col=None, duration=None, sampling_interval=1.0, fill_start=False, progress=True)`

returns a [`MotorMovementData`](data-object.md).

## Metrics

| `metric` | Value | Dimensions |
|---|---|---|
| `"euclidean"` | distance to each response button (standardised space) | participants, repetitions, images, targets, timepoints |
| `"cosine"` | cosine distance between movement direction (from the start button) and each button direction | same |
| `"position"` | x and y positions by default; set `position_col="x_std"` or `"y_std"` to return only one column | participants, repetitions, images, coordinates, timepoints by default; no targets |

The repetitions dimension is always kept, also with a single repetition.
For the default position output, `coordinates` has labels `["x_std", "y_std"]`.

## Target order

Button names and positions come from the preprocessed dataframe. `target_names`
sets the order of the targets dimension, as names (ignoring case, spaces and
punctuation) or indices. Default: the order found in the data.

## Time axis and interpolation

Mouse samples are irregular (about every 16 ms). Per trial, x and y are
**linearly interpolated** onto a regular grid (`time` column, ms), then the
distances are computed.

- `duration`: length of the array in ms (e.g. `1000`). Default: the latest time
  point in the whole dataset.
- `sampling_interval`: grid step in ms. Default 1 (one entry per ms). Use e.g.
  `4` for 250 Hz to get smaller arrays.
- Before a trial's first sample the position is unknown: the array is NaN there.
  `fill_start=True` instead assumes the cursor was at the start-button centre
  at time 0 and interpolates up to the first sample.
- After the trial's last sample (trial ended) the array is always NaN.

The `timepoints` coordinate holds the grid times in ms.

## Progress bar

A tqdm progress bar over trials is shown by default (`progress=False` hides
it). `preprocess_mouse_tracking_data` has the same option.
