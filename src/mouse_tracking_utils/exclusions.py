"""Trial and participant quality filters."""

import numpy as np
import pandas as pd


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


_ACCURACY_CANDIDATES = ("accuracy", "acc", "ACC", "correct", "is_correct", "correct_response_given")
_RESPONSE_CANDIDATES = ("response", "pressed_button", "button_pressed", "response_button")


def _find_column(df, candidates, user_choice, what):
    """Return the user's column, or the first candidate that exists and holds data."""
    if user_choice is not None:
        if user_choice not in df.columns:
            raise ValueError(f"Missing {what} column {user_choice!r}.")
        return user_choice
    lowered = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        column = candidate if candidate in df.columns else lowered.get(candidate.lower())
        if column is not None and df[column].notna().any():
            return column
    return None


def exclude_participants(
    input_df,
    *,
    participant_col="ppt_number",
    min_completed_trials=None,
    min_accuracy=None,
    response_col=None,
    accuracy_col=None,
    verbose=True,
):
    """Exclude participants with too few completed trials or too low accuracy.

    Expects a dataframe with one trial per row and a participant column. A
    threshold left as None is not applied.

    Args:
        input_df: dataframe with a participant column.
        participant_col: column identifying the participant of each trial.
        min_completed_trials: exclude participants with fewer completed trials.
        min_accuracy: exclude participants whose mean accuracy (between 0 and
            1) over their completed trials is lower than this.
        response_col: column flagging that a response button was pressed (a
            trial is completed when it is not empty). Detected automatically
            among ``response``, ``pressed_button``... when None.
        accuracy_col: column holding trial accuracy (True/False or 0/1).
            Detected automatically among ``accuracy``, ``acc``, ``correct``...
            (first one containing data) when None.
        verbose: print a summary of the exclusions.

    Returns:
        A copy of input_df without the excluded participants. The excluded
        participants and the per-reason counts are stored in the
        `exclusion_summary` entry of the output's `attrs`.
    """
    if not isinstance(input_df, pd.DataFrame):
        raise TypeError("input_df must be a pandas DataFrame.")
    if participant_col not in input_df.columns:
        raise ValueError(f"Missing participant identifier column {participant_col!r}.")
    if min_completed_trials is not None and min_completed_trials < 0:
        raise ValueError("min_completed_trials must be non-negative.")
    if min_accuracy is not None and not 0 <= min_accuracy <= 1:
        raise ValueError("min_accuracy must be between zero and one.")

    completed = None
    if min_completed_trials is not None or min_accuracy is not None:
        response_col = _find_column(input_df, _RESPONSE_CANDIDATES, response_col, "response")
        if response_col is None:
            raise ValueError("Could not find a response column; specify response_col.")
        completed = input_df[response_col].notna()

    accuracy = None
    if min_accuracy is not None:
        accuracy_col = _find_column(input_df, _ACCURACY_CANDIDATES, accuracy_col, "accuracy")
        if accuracy_col is None:
            raise ValueError("Could not find an accuracy column with data; specify accuracy_col.")
        accuracy = input_df[accuracy_col].map(
            lambda value: np.nan if pd.isna(value) else float(value in (True, 1, "True", "true", "1", 1.0))
        )

    too_few_trials, too_inaccurate = [], []
    for participant, trials in input_df.groupby(participant_col, dropna=False):
        done = completed.loc[trials.index]
        if min_completed_trials is not None and done.sum() < min_completed_trials:
            too_few_trials.append(participant)
        if min_accuracy is not None:
            mean_accuracy = accuracy.loc[trials.index][done].mean()
            if not mean_accuracy >= min_accuracy:
                too_inaccurate.append(participant)

    rejected = set(too_few_trials) | set(too_inaccurate)
    output = input_df.loc[~input_df[participant_col].isin(rejected)].copy()

    n_participants = input_df[participant_col].nunique(dropna=False)
    output.attrs["exclusion_summary"] = {
        "min_completed_trials": too_few_trials,
        "min_accuracy": too_inaccurate,
        "excluded_participants": sorted(rejected, key=str),
        "total_participants": n_participants,
    }
    if verbose:
        print(
            f"Excluded {len(rejected)} of {n_participants} participants "
            f"({len(too_few_trials)} too few completed trials, "
            f"{len(too_inaccurate)} too low accuracy)."
        )
    return output
