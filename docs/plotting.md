# Plotting

Both functions take a preprocessed dataframe (one or many trials) and return a
matplotlib `Axes`; call `plt.show()` yourself.

```python
ax = plot_trials(pre, color_by="accuracy", palette={0: "salmon", 1: "teal"},
                 excluded_indices=excluded_idx, button_shape="circle")
```

| Argument | Meaning |
|---|---|
| `color_by`, `palette` | column used for colours and optional `{value: colour}` |
| `excluded_indices` | these rows are drawn gray and labelled "excluded" |
| `button_shape`, `start_shape` | `"rectangle"` (default) or `"circle"` (an oval inscribed in the button rectangle if width != height) |
| `show_buttons` | draw start/response buttons in gray in the background |
| `invert_yaxis`, `alpha`, `ax` | axis orientation, line transparency, axes to draw on |

`plot_trials` keeps each trial's own screen coordinates (any number of buttons).
`plot_binary_trials` is for two-button tasks: it mirrors trials left-right so
the buttons line up with the first trial's order, or with
`response_button_order=["inanimate", "animate"]`.
