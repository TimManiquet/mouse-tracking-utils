"""Tools for filtering, preprocessing, and analysing mouse-tracking data."""

from .data_object import MotorMovementData, MouseTrackingData
from .distances import extract_distance
from .exclusions import exclude_trials
from .preprocessing import check_columns, filter_df, preprocess_mouse_tracking_data, processed_trial_mask
from .plotting import plot_binary_trials, plot_trials
from .demo_data import generate_demo_data, load_demo_data

__all__ = [
    "MotorMovementData",
    "MouseTrackingData",
    "check_columns",
    "exclude_trials",
    "extract_distance",
    "filter_df",
    "generate_demo_data",
    "load_demo_data",
    "plot_trials",
    "plot_binary_trials",
    "preprocess_mouse_tracking_data",
    "processed_trial_mask",
]
