# Demos

Two runnable scripts in [`examples/`](https://github.com/TimManiquet/mouse-tracking-utils/tree/main/examples)
use the built-in demo data (no download needed). They are written as `# %%`
cells, so they run top to bottom in VS Code, Spyder or PyCharm, or with
`python examples/<name>.py` (figures are shown one after the other).

| Script | Shows |
|---|---|
| `demo_preprocessing.py` | the raw dataframe before and after preprocessing, what the new columns contain, how `min_side` and `stretch` scaling change the trajectories for 2 and 8 buttons, and `row_mask` and `exclude_trials` |
| `demo_distances.py` | euclidean vs cosine vs position output for the same trials, target ordering, the effect of `fill_start` and `sampling_interval`, and averaging, selecting and saving the result |

Both are a starting point: edit the demo-data call to load your own data.
