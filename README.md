# mouse-tracking-utils

`mouse-tracking-utils` preprocesses mouse-tracking trial data and computes
time-resolved trajectory distances to response targets. It supports raw CSV
files and dataframes, and includes synthetic 2-, 4-, and 8-button demo data.

## Installation

```bash
pip install mouse-tracking-utils
```

For development from a clone:

```bash
uv sync --group dev
```

## Usage

```python
from mouse_tracking_utils import extract_distance, load_demo_data, preprocess_mouse_tracking_data

pre = preprocess_mouse_tracking_data(load_demo_data(4))
data = extract_distance(pre, metric="euclidean", duration=1000, fill_start=True)
print(data)
data.save("distances.npz", downsample_hz=250)
```

Two runnable demos are in [`examples/`](examples/): `demo_preprocessing.py` (what
standardisation and rescaling do) and `demo_distances.py` (euclidean vs cosine vs
position). Full documentation is in [`docs/`](docs/index.md): input data format,
workflow, preprocessing and exclusions, distances (metrics, ordering, interpolation),
the data object and plotting.

## Development

```bash
uv run pytest
uv build
```
