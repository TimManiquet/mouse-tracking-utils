# `MotorMovementData`

A NumPy array with named dimensions, labels and metadata.

```python
print(data)                  # nice summary: metric, shape, dimensions + labels, NaN %, layout
data.dimensions              # ('participants', 'repetitions', 'images', 'targets', 'timepoints')
data.describe_dimensions()   # axis, length and labels per dimension
data.axes                    # current axis of each dimension, e.g. {'participants': 0, ...}
data.axis_of("repetitions")  # axis of one dimension (updated after select / average)
data.images                  # or data["images"]: image of each entry on the images axis
data.targets                 # or data["targets"]: response buttons, in the order of the data
data.participants, data.repetitions, data.timepoints   # same for the other dimensions
data["coordinates"]          # x/y labels (data.coordinates is the dict of all labels)
data.metric                  # 'euclidean', 'cosine' or 'position'
data.trial_info(participants=1, images="image_1")  # button + start positions of those trials
```

| Metadata | Where |
|---|---|
| distance type | `data.metric` |
| image IDs | `data.images`, `data.coordinates["images"]` |
| response / start button positions | `data.trial_info(...)`, `data.metadata["button_positions"]` |
| meaning of each dimension | `data.dimensions`, `data.axes`, `data.describe_dimensions()` |
| targets (response buttons) | `data.targets` |
| time unit and step | `data.metadata["time_unit"]`, `["sampling_interval"]` |

## Methods

- `data.average("repetitions")` or `data.average(["participants", "repetitions"])`:
  NaN-aware mean over named dimensions (or axes).
- `data.select(participants=2)` or `data.select(images=["b", "a"], repetitions=1)`:
  select by label. A single label removes that dimension; a list keeps it, in
  the order given. Singular names work (`participant=2`).
- `data.downsample(250)`: keep every n-th time point to reach about 250 Hz.
- `data.save("file.npz", downsample_hz=None)`: compressed `.npz` (no pickle).
  `downsample_hz=250` or `500` downsamples first; the object is untouched.
- `MotorMovementData.load("file.npz")`.

## Dimensions move when you select or average

`select` with a single label and `average` remove a dimension, so the
remaining ones are renumbered. `data.axes` and `data.axis_of(name)` always
describe the current array. Asking for the labels of a removed dimension (e.g.
`one.participants` after `select(participants=2)`) raises an error that says how
it was removed.
