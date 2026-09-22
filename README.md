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

`preprocess_data` accepts a dataframe, a CSV path, a directory of CSVs, or a
list of dataframe/CSV inputs. It normalises common trial columns and, by
default, adds centered `x_std` and `y_std` coordinate arrays.

```python
from mouse_tracking_utils import extract_distances, load_demo_data, preprocess_data

df = load_demo_data(2)
preprocessed = preprocess_data(df)

# Euclidean distance to each button at each time point
arr = extract_distances(df, metric='euclidean', target_names=['0', '1'])
print(arr.shape)

# Position on screen (x coordinate over time)
pos = extract_distances(df, metric='position', position_col='x')
print(pos.shape)
```

Distance arrays have shape
`(participants, repetitions, images, targets, timepoints)` for the
`euclidean` and `cosine` metrics. Position mode omits the target axis.
Singleton participant, repetition, image, and target dimensions are removed by
default; pass `squeeze_singleton=False` to preserve the full shape.

## Development

```bash
uv run pytest
uv build
```
