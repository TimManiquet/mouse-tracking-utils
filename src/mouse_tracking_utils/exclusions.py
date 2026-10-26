"""Trial quality filters."""

import numpy as np
import pandas as pd

from .preprocessing import processed_trial_mask


def exclude_trials(
    input_df,
    *,
    x_col="x_std",
    y_col="y_std",
    origin=(0.0, 0.0),
    min_endpoint=None,
    max_endpoint=None,
    min_distance=None,
    max_distance=None,
    verbose=True,
    return_indices=False,
):
    """Exclude trials based on where they end and how far the mouse travelled.

    Expects a preprocessed dataframe whose trajectories have been standardised.
    A threshold left as None is not applied. Trials without any valid
    trajectory sample are kept, as no criterion can be evaluated on them.

    Args:
        input_df: preprocessed dataframe, one trial per row.
        x_col, y_col: columns holding the x and y trajectories.
        origin: (x, y) point the endpoint distance is measured from.
        min_endpoint: exclude trials ending closer than this to the origin.
        max_endpoint: exclude trials ending farther than this from the origin.
        min_distance: exclude trials whose path length is shorter than this.
        max_distance: exclude trials whose path length is longer than this.
        verbose: print a summary of the exclusions.
        return_indices: if True, also return the index labels of the excluded
            trials, as a tuple (output, excluded_indices).

    Returns:
        A copy of input_df without the excluded trials (plus the index labels
        of the excluded trials if return_indices is True). The per-reason counts
        are stored in the `exclusion_summary` entry of the output's `attrs`.
        A trial failing several criteria is counted under each of them, but
        only once in the total.
    """
    if not isinstance(input_df, pd.DataFrame):
        raise TypeError("input_df must be a pandas DataFrame.")

    # Check for necessary columns: x_col, y_col
    if x_col not in input_df.columns:
        raise ValueError(f"Missing x trajectory column {x_col!r}.")
    if y_col not in input_df.columns:
        raise ValueError(f"Missing y trajectory column {y_col!r}.")
    if len(origin) != 2:
        raise ValueError("origin must be an (x, y) pair.")

    # Validate distance thresholds
    if min_distance is not None and min_distance < 0:
        raise ValueError("min_distance must be non-negative.")
    if max_distance is not None and max_distance < 0:
        raise ValueError("max_distance must be non-negative.")
    if min_distance is not None and max_distance is not None and min_distance > max_distance:
        raise ValueError("min_distance cannot exceed max_distance.")

    # Validate endpoint thresholds
    if min_endpoint is not None and min_endpoint < 0:
        raise ValueError("min_endpoint must be non-negative.")
    if max_endpoint is not None and max_endpoint < 0:
        raise ValueError("max_endpoint must be non-negative.")
    if min_endpoint is not None and max_endpoint is not None and min_endpoint > max_endpoint:
        raise ValueError("min_endpoint cannot exceed max_endpoint.")

    reasons = ["min_endpoint", "max_endpoint", "min_distance", "max_distance"]
    flags = {reason: np.zeros(len(input_df), dtype=bool) for reason in reasons}

    for i, (x, y) in enumerate(zip(input_df[x_col], input_df[y_col])):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        valid = (len(x) == len(y)) and len(x) > 0
        if valid:
            finite = np.isfinite(x) & np.isfinite(y)
            x, y = x[finite], y[finite]
            valid = len(x) > 0
        if not valid:
            continue

        endpoint = np.hypot(x[-1] - origin[0], y[-1] - origin[1])
        path_length = np.hypot(np.diff(x), np.diff(y)).sum()

        flags["min_endpoint"][i] = min_endpoint is not None and endpoint < min_endpoint
        flags["max_endpoint"][i] = max_endpoint is not None and endpoint > max_endpoint
        flags["min_distance"][i] = min_distance is not None and path_length < min_distance
        flags["max_distance"][i] = max_distance is not None and path_length > max_distance

    excluded = np.logical_or.reduce(list(flags.values()))
    output = input_df.loc[~excluded].copy()

    summary = {reason: int(flag.sum()) for reason, flag in flags.items()}
    summary["total_excluded"] = int(excluded.sum())
    summary["total_trials"] = len(input_df)
    summary["excluded_indices"] = input_df.index[excluded].tolist()
    output.attrs["exclusion_summary"] = summary

    if verbose:
        details = ", ".join(
            f"{summary[reason]} {reason.replace('_', ' ')}" for reason in reasons
        )
        print(
            f"Excluded {summary['total_excluded']} of {summary['total_trials']} trials "
            f"({details})."
        )

    if return_indices:
        return output, input_df.index[excluded]
    return output
