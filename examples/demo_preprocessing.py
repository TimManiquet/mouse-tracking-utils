# %% [markdown]
# # Demo 1: what preprocessing does
#
# Raw trials in, standardised trajectories out. Run cell by cell (VS Code,
# Spyder, PyCharm) or with `python examples/demo_preprocessing.py`.

# %%
import json

import matplotlib.pyplot as plt
import numpy as np

from mouse_tracking_utils import (
    exclude_trials,
    generate_demo_data,
    plot_trials,
    preprocess_mouse_tracking_data,
    processed_trial_mask,
)

# %% [markdown]
# ## 1. The raw data
# One row per trial. Mouse samples are a list of `{x, y, t}` in screen pixels,
# with the origin at the top left of the window. Here: an 8-button task, 2
# participants, 4 images, 2 repetitions.

# %%
raw = generate_demo_data(8, n_ppts=2, n_images=4, n_repetitions=2)
print(raw.shape)
print(raw.columns.tolist())
print(raw.loc[0, "mouse_tracking_data"][:80], "...")

# %% [markdown]
# ## 2. After preprocessing
# All rows and columns are kept. New columns hold the standardised trajectories
# (`x_std`, `y_std`, `time`), the standardised button positions and
# `repetition`. `processed_trial_mask` tells which rows hold a trajectory.

# %%
pre = preprocess_mouse_tracking_data(raw)
print(pre.shape, "- rows with a trajectory:", processed_trial_mask(pre).sum())
print([column for column in pre.columns if column not in raw.columns])
row = pre.iloc[0]
samples = json.loads(raw.loc[0, "mouse_tracking_data"])
print("x pixels :", np.round([p["x"] for p in samples[:3]], 1))
print("x_std    :", np.round(row["x_std"][:3], 3))
print("std_scale (pixels per unit, x and y):", row["std_scale"])

# %% [markdown]
# ## 3. Rescaling: `min_side` vs `stretch`
# Screens are wider than tall. `stretch` maps both axes to [-1, 1], which
# squeezes x: with buttons on a circle, they are no longer equidistant from the
# start button. `min_side` (default) uses the same unit on both axes.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, scaling in zip(axes, ("stretch", "min_side")):
    result = preprocess_mouse_tracking_data(raw, scaling=scaling, progress=False)
    plot_trials(result, ax=ax)
    ax.set_title(f'scaling="{scaling}"')
    first = result[processed_trial_mask(result)].iloc[0]
    start = np.asarray(first["starting_button_position_std"])
    distances = [np.linalg.norm(np.asarray(p) - start) for p in first["response_button_positions_std"].values()]
    print(f"{scaling:9s} start-to-button distances:", np.round(distances, 2))
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Choosing rows: `row_mask`
# Nothing is dropped, but you can restrict which rows are processed, for
# example to leave out training trials. Unprocessed rows keep empty new columns.

# %%
raw_with_training = raw.copy()
raw_with_training.loc[:3, "event"] = "training_trial"
pre_main = preprocess_mouse_tracking_data(
    raw_with_training, row_mask=lambda d: d["event"] != "training_trial", progress=False
)
print("rows:", len(pre_main), "- processed:", processed_trial_mask(pre_main).sum())

# %% [markdown]
# ## 5. Excluding trials
# Thresholds are in standardised units (1 = half the shorter screen side).
# The excluded indices can be used to colour those trials in the plot. The demo
# trials are very regular, so the threshold is chosen just to exclude a few.

# %%
kept, excluded = exclude_trials(pre, min_endpoint=0.5, min_distance=0.86, return_indices=True)
plot_trials(pre, color_by="stimulus", excluded_indices=excluded)
plt.title(f"{len(excluded)} excluded trials in gray")
plt.show()
