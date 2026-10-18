"""Bundled synthetic mouse-tracking datasets and their generator."""

import json
from importlib.resources import files

import numpy as np
import pandas as pd


def load_demo_data(button_count=None):
    """Load bundled demo data for 2-, 4-, or 8-button tasks."""
    demo_files = {
        2: "mouse_tracking_demo_2_buttons.csv",
        4: "mouse_tracking_demo_4_buttons.csv",
        8: "mouse_tracking_demo_8_buttons.csv",
    }
    resources = files("mouse_tracking_utils").joinpath("demo_data")
    if button_count is None:
        return {
            count: pd.read_csv(resources.joinpath(filename))
            for count, filename in demo_files.items()
        }
    if button_count not in demo_files:
        raise ValueError(
            f"Unsupported demo dataset size {button_count}. Choose one of: {sorted(demo_files)}."
        )
    return pd.read_csv(resources.joinpath(demo_files[button_count]))


_SCREEN = (1280, 800)
_BUTTON = (120, 60)
_START = (40, 40)


def _layout(button_count):
    """Button names and rectangles (bottom-left x, bottom-left y, w, h) on a circle."""
    centre = np.array([_SCREEN[0] / 2, _SCREEN[1] / 2])
    angles = np.linspace(0, 2 * np.pi, button_count, endpoint=False)
    names = [f"button_{index + 1}" for index in range(button_count)]
    rectangles = [
        [
            float(centre[0] + 300 * np.cos(angle) - _BUTTON[0] / 2),
            float(centre[1] + 300 * np.sin(angle) + _BUTTON[1] / 2),
            _BUTTON[0],
            _BUTTON[1],
        ]
        for angle in angles
    ]
    start = [float(centre[0] - _START[0] / 2), float(centre[1] + _START[1] / 2), _START[0], _START[1]]
    return names, rectangles, start


def _generate_trial(button_count, participant, repetition, image, names, rectangles, start):
    samples = np.linspace(0.0, 1.0, 20)
    chosen = (repetition + participant) % button_count
    x0, y0, w, h = rectangles[chosen]
    target = np.array([x0 + w / 2, y0 - h / 2])
    origin = np.array([start[0] + start[2] / 2, start[1] - start[3] / 2])
    path = origin + (target - origin) * samples[:, None] ** 1.5
    wobble = np.column_stack(
        (np.sin(np.arange(1, 21) * 0.7 + participant), np.cos(np.arange(1, 21) * 0.9 + repetition))
    )
    path = path + wobble * 20 * (1 - samples[:, None])
    trajectory = [
        {"x": float(px), "y": float(py), "t": float(t * 1000), "event": "mousemove"}
        for (px, py), t in zip(path, samples)
    ]
    return {
        "ppt_number": participant,
        "trial_type": "html-button-response",
        "event": "main_trial",
        "response": chosen,
        "accuracy": int(chosen == image % button_count),
        "stimulus": f"./stimuli/image_{image}.png",
        "mouse_tracking_data": json.dumps(trajectory),
        "response_button_names": json.dumps(names),
        "screen_width": _SCREEN[0],
        "screen_height": _SCREEN[1],
        "response_button_positions": json.dumps(rectangles),
        "starting_button_position": json.dumps(start),
    }


def generate_demo_data(button_count=2, n_ppts=2, n_images=2, n_repetitions=2):
    """Create a synthetic raw dataframe (input of the preprocessing) for 2, 4 or 8 buttons.

    All participants are in one dataframe, identified by ``ppt_number``.
    """
    if button_count not in {2, 4, 8}:
        raise ValueError("button_count must be one of {2, 4, 8}.")
    if min(n_ppts, n_images, n_repetitions) < 1:
        raise ValueError("n_ppts, n_images, and n_repetitions must be positive.")
    names, rectangles, start = _layout(button_count)
    return pd.DataFrame(
        [
            _generate_trial(button_count, participant, repetition, image, names, rectangles, start)
            for participant in range(1, n_ppts + 1)
            for image in range(1, n_images + 1)
            for repetition in range(1, n_repetitions + 1)
        ]
    )
