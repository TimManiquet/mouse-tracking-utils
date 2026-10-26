# Input data

One row per trial; one dataframe (CSV) per participant. Mouse samples live in a
`mouse_tracking_data` column as a list (or JSON string) of
`{"x": ..., "y": ..., "t": <ms>, "event": ...}`.

## Required columns

| Standard name | Meaning |
|---|---|
| `response` | response given (used to detect completed trials) |
| `accuracy` | correct / incorrect |
| `stimulus` | image shown (a path or html containing it) |
| `mouse_tracking_data` | the samples described above |
| `response_button_names` | list of button labels |
| `screen_width`, `screen_height` | window size in px |
| `response_button_positions` | one rectangle per button |
| `starting_button_position` | rectangle of the start button |

`trial_type` and `event` are optional and not used by the pipeline; they are handy to write a `row_mask` (e.g. to leave out training trials).

A **rectangle** is `(bottom-left x, bottom-left y, width, height)` in pixels,
origin at the top left of the window (so "bottom" is the larger y).

If your columns are named differently, `check_columns` detects common aliases
or accepts a mapping, and raises an error listing what is missing:

```python
check_columns(df, column_map={"accuracy": "correct"})
preprocess_mouse_tracking_data(df, column_map={"accuracy": "correct"})
```

## Output columns

All your columns are kept. Added or filled in (standard names, only where the
source information exists): `ppt_number`, `repetition`, `stimulus` (image ID),
`response_button_names`, `starting_button_position`,
`starting_button_position_std`, `response_button_positions`,
`response_button_positions_std`, `time`, `x_std`, `y_std`, `std_scale`, plus a
standard-named copy of any mapped column (`response`, `accuracy`, ...).
