# `MotorMovementData`

A NumPy array with named dimensions, labels and metadata.

```python
print(data)                  # nice summary: metric, shape, dimensions + labels, NaN %, layout
data.dimensions              # ('participants', 'repetitions', 'images', 'targets', 'timepoints')
data.describe_dimensions()   # axis, length and labels per dimension
data.image_ids               # image of each entry on the images axis
data.metric                  # 'euclidean', 'cosine' or 'position'
data.trial_info(participants=1, images="image_1")  # button + start positions of those trials
```

| Metadata | Where |
|---|---|
| distance type | `data.metric` |
| image IDs | `data.coordinates["images"]` |
| response / start button positions | `data.trial_info(...)`, `data.metadata["button_positions"]` |
| meaning of each dimension | `data.dimensions`, `data.describe_dimensions()` |
| time unit and step | `data.metadata["time_unit"]`, `["sampling_interval"]` |

## Methods

- `data.average("repetitions")` or `data.average(["participants", "repetitions"])`:
  NaN-aware mean over named dimensions (or axes).
- `data.downsample(250)`: keep every n-th time point to reach about 250 Hz.
- `data.save("file.npz", downsample_hz=None)`: compressed `.npz` (no pickle).
  `downsample_hz=250` or `500` downsamples first; the object is untouched.
- `MotorMovementData.load("file.npz")`.
