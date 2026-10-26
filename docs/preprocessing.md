# Preprocessing and exclusions

## `preprocess_mouse_tracking_data(df_or_list, ...)`

Hands-off: **all rows and all columns are kept**. The output is your dataframe
plus the columns computed here; only the rows that can be processed get them
filled in, the others (surveys, instructions, rows without a trajectory) keep
empty values. `processed_trial_mask(df)` tells which rows were processed.

1. `check_columns` finds the columns holding the needed information. Where
   your column has another name, a copy is added under the standard name (yours
   is kept; `column_map={"accuracy": "correct"}` says where things are).
2. A row is processed when it has mouse samples, a stimulus, button names, a
   valid screen size and parsable button positions. Rows with samples that fail
   one of these are reported in a warning. `row_mask` restricts the rows
   further, e.g. `row_mask=lambda d: ~d["event"].fillna("").str.contains("training")`
   to leave out training trials (they are processed by default).
3. Processed rows get: `stimulus` as image ID, parsed button names and
   positions, `time`, `x_std`, `y_std`, `starting_button_position_std`,
   `response_button_positions_std`, `repetition` (if absent), `std_scale`.

Given a list of dataframes it concatenates them and adds `ppt_number` (1, 2, ...).
A tqdm progress bar is shown (per row, or per participant for a list); pass `progress=False` to hide it.

### Standardised coordinates

The origin is the screen centre. With `scaling="min_side"` (default) both axes
are divided by **half the shorter screen side**, so distances and angles keep
their true proportions: buttons placed on a circle around the start button stay
equidistant from it, and `x_std` spans `[-width/height, width/height]` on a
landscape screen while `y_std` spans `[-1, 1]`. `scaling="stretch"` instead maps
both axes to `[-1, 1]`, which squeezes the longer side (fine for two buttons on
a horizontal line, wrong for circular layouts). `invert_y=True` (default) makes
up positive. `std_scale` holds the pixels per standardised unit in each trial.

## Exclusions

```python
kept, excluded_idx = exclude_trials(
    pre, min_endpoint=0.3, max_endpoint=1.5, min_distance=0.5, max_distance=4,
    return_indices=True,
)
```

Rows without a trajectory are never excluded. Thresholds you leave out are not applied. `exclude_trials` prints how many
trials were excluded for each reason.
