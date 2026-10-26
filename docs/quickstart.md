# Quickstart

```bash
pip install mouse-tracking-utils
```

```python
from mouse_tracking_utils import (
    extract_distance, load_demo_data, preprocess_mouse_tracking_data, plot_trials,
)

raw = load_demo_data(4)                        # demo task with 4 response buttons (2 and 8 also exist)
pre = preprocess_mouse_tracking_data(raw)      # keeps all rows/columns; or pass a list of dataframes, one per participant
data = extract_distance(pre, metric="euclidean", duration=1000)
print(data)                                    # dimensions, labels, metadata
by_image = data.average("participants")
data.save("distances.npz", downsample_hz=250)
plot_trials(pre, color_by="stimulus")
```

Next: the [workflow](workflow.md) explains what happens at each stage, and the
demos in [`examples/`](https://github.com/TimManiquet/mouse-tracking-utils/tree/main/examples)
show the results of [preprocessing](demos.md) and of distance extraction.
