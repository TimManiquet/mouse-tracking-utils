# %% [markdown]
# # Demo 2: what distance extraction gives you
#
# The same trials, seen through different metrics. Run cell by cell or with
# `python examples/demo_distances.py`.

# %%
import matplotlib.pyplot as plt
import numpy as np

from mouse_tracking_utils import (
    MotorMovementData,
    extract_distance,
    generate_demo_data,
    preprocess_mouse_tracking_data,
)

raw = generate_demo_data(4, n_ppts=3, n_images=4, n_repetitions=2)
pre = preprocess_mouse_tracking_data(raw)

# %% [markdown]
# ## 1. Euclidean distance to each button
# Output dimensions: participants x repetitions x images x targets x timepoints.

# %%
euclid = extract_distance(pre, metric="euclidean", duration=1000, sampling_interval=5)
print(euclid)
print("axes:", euclid.axes)
print("targets:", euclid.targets)

# %% [markdown]
# ## 2. Euclidean vs cosine
# For one trial: euclidean distance falls towards the chosen button and grows
# for the others. Cosine distance compares the movement *direction* with each
# button's direction, so it tells which button the hand is heading to, even
# early in the movement.

# %%
cosine = extract_distance(pre, metric="cosine", duration=1000, sampling_interval=5)
trial = dict(participants=1, repetitions=1, images=euclid.images[0])
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
for ax, data in zip(axes, (euclid, cosine)):
    one = data.select(**trial)
    for target, series in zip(one.targets, one.values):
        ax.plot(one.timepoints, series, label=target)
    ax.set_title(f"{data.metric} (image {trial['images']})")
    ax.set_xlabel("time (ms)")
axes[0].set_ylabel("distance")
axes[0].legend(title="target")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Position over time
# `metric="position"` returns x and y; use `position_col="x_std"` for one.

# %%
position = extract_distance(pre, metric="position", duration=1000, sampling_interval=5)
print(position.dimensions, position["coordinates"])
one = position.select(**trial)
plt.plot(one.timepoints, one.values.T, label=one["coordinates"])
plt.legend()
plt.xlabel("time (ms)")
plt.show()

# %% [markdown]
# ## 4. Choosing the target order
# Distances come in the order you ask for.

# %%
names = euclid.targets
reordered = extract_distance(pre, target_names=names[::-1], duration=1000, sampling_interval=5)
print(names, "->", reordered.targets)
np.testing.assert_allclose(reordered.values, euclid.values[..., ::-1, :], equal_nan=True)

# %% [markdown]
# ## 5. Start of the trial: NaN or filled in
# Before the first mouse sample the position is unknown. By default the array is
# NaN there; `fill_start=True` places the cursor at the start button. The demo
# trials start moving at t=0, so here the movement is delayed by 200 ms.

# %%
late = pre.copy()
late["time"] = [None if t is None else t + 200 for t in late["time"]]
nan_start = extract_distance(late, duration=1200, sampling_interval=5, progress=False)
filled = extract_distance(late, duration=1200, sampling_interval=5, fill_start=True, progress=False)
series = lambda d: d.select(**trial).values[0]
print("first values, default   :", np.round(series(nan_start)[:3], 2), "...")
print("first values, fill_start:", np.round(series(filled)[:3], 2), "...")

# %% [markdown]
# ## 6. Working with the result
# Dimensions are renumbered when one disappears; `axes` always tells where
# each one is now.

# %%
by_image = euclid.average("repetitions").average("participants")
print(by_image.axes)
one_participant = euclid.select(participants=2)
print(one_participant.axes, one_participant.shape)
path = euclid.save("demo_distances.npz", downsample_hz=100)
print(MotorMovementData.load(path).shape)
path.unlink()
