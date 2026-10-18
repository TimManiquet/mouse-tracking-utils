"""Plotting functions for preprocessed mouse-tracking data."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle


def _draw_box(ax, center, width, height, shape="rectangle"):
    """Gray background box; a circle or oval is inscribed in the rectangle."""
    if shape == "rectangle":
        patch = Rectangle((center[0] - width / 2, center[1] - height / 2), width, height)
    else:
        patch = Ellipse(center, width, height)
    patch.set(facecolor="lightgray", edgecolor="gray", zorder=0)
    ax.add_patch(patch)


def _check_shape(name, value):
    if value not in ("rectangle", "circle"):
        raise ValueError(f"{name} must be 'rectangle' or 'circle', not {value!r}.")


def _button_rectangles(row):
    """Map button names to pixel rectangles using each trial's own button order."""
    rectangles = row.get("response_button_positions")
    centers = row.get("response_button_positions_std")
    if not isinstance(centers, dict):
        return {}
    if isinstance(rectangles, dict):
        return {str(name).casefold(): (center, rectangles.get(name)) for name, center in centers.items()}
    names = row.get("response_button_names")
    if not isinstance(names, list):
        names = list(centers)
    by_name = {
        str(name).casefold(): rectangle
        for name, rectangle in zip(names, rectangles if isinstance(rectangles, list) else ())
    }
    return {
        str(name).casefold(): (center, by_name.get(str(name).casefold()))
        for name, center in centers.items()
    }


def _mirror_flags(df, response_button_order):
    """Return which binary trials need horizontal reflection to match a left-right order."""
    first = next(
        (
            row
            for _, row in df.iterrows()
            if isinstance(row.get("response_button_positions_std"), dict)
            and len(row["response_button_positions_std"]) == 2
        ),
        None,
    )
    if first is None:
        raise ValueError("Binary trial alignment requires two response-button positions per trial.")
    first_positions = _button_rectangles(first)
    if response_button_order is None:
        ordered = sorted(first_positions, key=lambda name: first_positions[name][0][0])
        order = ordered
    else:
        order = [str(name).casefold() for name in response_button_order]
        if len(order) != 2 or len(set(order)) != 2 or set(order) != set(first_positions):
            available = [name for name, _ in sorted(first_positions.items(), key=lambda item: item[1][0][0])]
            raise ValueError(
                "response_button_order must name both response buttons exactly once; "
                f"available buttons: {available!r}."
            )

    mirrors = {}
    for index, row in df.iterrows():
        positions = _button_rectangles(row)
        if set(positions) != set(order):
            raise ValueError(
                f"Binary trial at row index {index!r} must have the same two response buttons "
                f"{order!r}; found {list(positions)!r}."
            )
        if any(positions[name][0] is None for name in order):
            raise ValueError(f"Binary trial at row index {index!r} is missing a standardised button position.")
        mirrors[index] = positions[order[0]][0][0] > positions[order[1]][0][0]
    return mirrors, order


def _draw_buttons(ax, df, mirror_flags=None, binary_order=None, button_shape="rectangle", start_shape="rectangle"):
    """Draw button boxes in screen coordinates, optionally mirroring binary trials."""
    drawn = set()
    for _, row in df.iterrows():
        width, height = row.get("screen_width"), row.get("screen_height")
        if not width or not height:
            continue
        mirror = bool(mirror_flags and mirror_flags.get(row.name, False))

        def transform(center):
            return (-center[0], center[1]) if mirror else center

        start = row.get("starting_button_position")
        start_center = row.get("starting_button_position_std")
        if start is not None and start_center is not None:
            center = transform(start_center)
            key = ("start", tuple(center), start[2], start[3])
            if key not in drawn:
                drawn.add(key)
                _draw_box(ax, center, 2 * start[2] / width, 2 * start[3] / height, start_shape)

        buttons = _button_rectangles(row)
        names = binary_order if binary_order is not None else buttons
        for name in names:
            key = str(name).casefold()
            item = buttons.get(key)
            if item is None:
                continue
            center, rectangle = item
            if rectangle is None or len(rectangle) != 4:
                continue
            center = transform(center)
            draw_key = (key, tuple(center), rectangle[2], rectangle[3])
            if draw_key not in drawn:
                drawn.add(draw_key)
                _draw_box(ax, center, 2 * rectangle[2] / width, 2 * rectangle[3] / height, button_shape)


def _plot_trials(
    df,
    *,
    color_by,
    palette,
    excluded_indices,
    ax,
    alpha,
    show_buttons,
    invert_yaxis,
    mirror_flags=None,
    binary_order=None,
    button_shape="rectangle",
    start_shape="rectangle",
):
    _check_shape("button_shape", button_shape)
    _check_shape("start_shape", start_shape)
    missing = [column for column in ("x_std", "y_std") if column not in df.columns]
    if missing:
        raise ValueError(f"Missing trajectory column(s) {missing!r}; preprocess the data first.")
    if color_by is not None and color_by not in df.columns:
        raise ValueError(f"Color column {color_by!r} was not found in the dataframe.")
    if ax is None:
        _, ax = plt.subplots()
    palette = dict(palette or {})
    values = list(dict.fromkeys(df[color_by].dropna())) if color_by else [None]
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for value in values:
        palette.setdefault(value, "black" if value is None else cycle[len(palette) % len(cycle)])

    if show_buttons:
        _draw_buttons(ax, df, mirror_flags, binary_order, button_shape, start_shape)

    excluded = set(() if excluded_indices is None else excluded_indices)
    legend = {}
    for trial_idx, row in df.iterrows():
        is_excluded = trial_idx in excluded
        value = row[color_by] if color_by else None
        label = "excluded" if is_excluded else (str(value) if color_by else None)
        color = "gray" if is_excluded else palette.get(value, "black")
        x = np.asarray(row["x_std"], dtype=float)
        y = np.asarray(row["y_std"], dtype=float)
        if mirror_flags and mirror_flags.get(trial_idx, False):
            x = -x
        (line,) = ax.plot(x, y, color=color, alpha=alpha, label=label)
        if label is not None:
            legend.setdefault(label, line)
    if legend:
        ax.legend(legend.values(), legend.keys(), title=color_by)
    ax.set_xlim(-1, 1)
    ax.set_ylim(*((1, -1) if invert_yaxis else (-1, 1)))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (standardised)")
    ax.set_ylabel("y (standardised)")
    return ax


def plot_trials(
    df,
    color_by=None,
    palette=None,
    excluded_indices=None,
    ax=None,
    alpha=0.4,
    show_buttons=True,
    invert_yaxis=True,
    button_shape="rectangle",
    start_shape="rectangle",
):
    """Plot one or more trials in their original, unmirrored screen coordinates.

    ``button_shape`` / ``start_shape`` are ``'rectangle'`` (default) or
    ``'circle'``; a circle is drawn as an oval inscribed in the button
    rectangle when its width and height differ.

    Use :func:`plot_binary_trials` instead when binary trials should be mirrored
    to align the semantic response buttons left-to-right.
    """
    return _plot_trials(
        df,
        color_by=color_by,
        palette=palette,
        excluded_indices=excluded_indices,
        ax=ax,
        alpha=alpha,
        show_buttons=show_buttons,
        invert_yaxis=invert_yaxis,
        button_shape=button_shape,
        start_shape=start_shape,
    )


def plot_binary_trials(
    df,
    color_by=None,
    palette=None,
    excluded_indices=None,
    response_button_order=None,
    ax=None,
    alpha=0.4,
    show_buttons=True,
    invert_yaxis=True,
    button_shape="rectangle",
    start_shape="rectangle",
):
    """Plot binary-task trials, mirroring trials to align the response buttons.

    By default, the left-to-right button order from the first trial is used as
    the canonical order. Pass ``response_button_order`` to choose the desired
    left-to-right order explicitly. Each trial's trajectories and button boxes
    are reflected horizontally as needed. This is intended for exactly two
    response buttons. ``button_shape`` / ``start_shape`` work as in
    :func:`plot_trials`.
    """
    mirrors, order = _mirror_flags(df, response_button_order)
    return _plot_trials(
        df,
        color_by=color_by,
        palette=palette,
        excluded_indices=excluded_indices,
        ax=ax,
        alpha=alpha,
        show_buttons=show_buttons,
        invert_yaxis=invert_yaxis,
        mirror_flags=mirrors,
        binary_order=order,
        button_shape=button_shape,
        start_shape=start_shape,
    )
