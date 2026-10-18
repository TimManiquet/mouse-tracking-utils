# Preprocessing and exclusions

## `preprocess_mouse_tracking_data(df_or_list, ...)`

1. checks and renames columns to the standard names,
2. keeps mouse-tracking trials (drops rows without samples and training trials),
3. standardises trajectories: `x_std = 2x/width - 1`, `y_std = 2y/height - 1`
   (sign flipped by default so that up is positive; `invert_y=False` keeps
   screen orientation), both in `[-1, 1]`,
4. standardises start and response button centres the same way,
5. keeps only the standard columns.

Given a list of dataframes it concatenates them and adds `ppt_number` (1, 2, ...).

## Exclusions

```python
kept, excluded_idx = exclude_trials(
    pre, min_endpoint=0.3, max_endpoint=1.5, min_distance=0.5, max_distance=4,
    return_indices=True,
)
pre = exclude_participants(pre, min_trials=100, min_accuracy=0.7)
```

Thresholds you leave out are not applied. `exclude_trials` prints how many
trials were excluded for each reason. `exclude_participants` uses completed
trials (a response was given) and mean accuracy.
