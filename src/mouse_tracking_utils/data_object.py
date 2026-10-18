"""Array container with named dimensions and serializable coordinate metadata.

Metadata conventions used by :func:`mouse_tracking_utils.extract_distance`:

- ``metadata["metric"]``: type of values stored (euclidean, cosine, position).
- ``coordinates["images"]``: image ID of each position along the images axis.
- ``metadata["button_positions"]["per_trial"]``: one record per trial with its
  participant / repetition / image labels, its ``positions`` (response-button
  centres) and its ``start_position`` (start-button centre).
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


class MotorMovementData:
    """Numerical mouse-tracking results with named dimensions and coordinates."""

    def __init__(self, values, dimensions, coordinates=None, *, metadata=None):
        self.values = np.asarray(values)
        self.dimensions = tuple(dimensions)
        if self.values.ndim != len(self.dimensions):
            raise ValueError(
                f"Got {self.values.ndim} array dimensions but {len(self.dimensions)} dimension names."
            )
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError("Dimension names must be unique.")
        self.coordinates = {
            name: list(values) for name, values in (coordinates or {}).items()
        }
        unknown = set(self.coordinates) - set(self.dimensions)
        if unknown:
            raise ValueError(f"Coordinates supplied for unknown dimensions: {sorted(unknown)!r}.")
        for axis, name in enumerate(self.dimensions):
            if name in self.coordinates and len(self.coordinates[name]) != self.values.shape[axis]:
                raise ValueError(
                    f"Dimension {name!r} has length {self.values.shape[axis]} but "
                    f"{len(self.coordinates[name])} coordinate labels."
                )
        self.metadata = dict(metadata or {})

    @property
    def metric(self):
        """Type of values stored (euclidean, cosine, position...), if known."""
        return self.metadata.get("metric")

    @property
    def image_ids(self):
        """Image ID along the images dimension, if the object has one."""
        return list(self.coordinates.get("images", []))

    def describe_dimensions(self):
        """Describe each dimension: its axis, length and labels (if any).

        Returns:
            dict mapping dimension name to {"axis", "length", "labels"}.
        """
        return {
            name: {
                "axis": axis,
                "length": self.values.shape[axis],
                "labels": list(self.coordinates.get(name, [])),
            }
            for axis, name in enumerate(self.dimensions)
        }

    def axis_of(self, name):
        """Numeric axis of a named dimension."""
        if name not in self.dimensions:
            raise ValueError(f"Unknown dimension {name!r}; choose from {self.dimensions!r}.")
        return self.dimensions.index(name)

    def trial_info(self, **labels):
        """Trial records (button positions, start position) matching dimension labels.

        Example: ``data.trial_info(participants=1, images="image_1")``. Each
        record holds the labels of the trial, its response-button positions
        and its start-button position, so a time series can be related to
        the screen layout it was recorded with. Records describe the original
        trials, also after averaging.
        """
        key_for = {"participants": "participant", "repetitions": "repetition", "images": "image"}
        unknown = set(labels) - set(key_for)
        if unknown:
            raise ValueError(f"Cannot select trials by {sorted(unknown)!r}; use {sorted(key_for)!r}.")
        records = self.metadata.get("button_positions", {}).get("per_trial", [])
        return [
            record
            for record in records
            if all(record.get(key_for[name]) == value for name, value in labels.items())
        ]

    def __repr__(self):
        return f"{type(self).__name__}(shape={self.shape}, dimensions={self.dimensions}, metric={self.metric!r})"

    def __str__(self):
        lines = [f"{type(self).__name__}", f"  metric:     {self.metric or 'unknown'}"]
        lines.append(f"  shape:      {self.shape}  ({self.dtype})")
        lines.append("  dimensions:")
        for axis, name in enumerate(self.dimensions):
            labels = self.coordinates.get(name)
            if labels is None:
                summary = "no labels"
            elif len(labels) > 4:
                summary = f"{labels[0]!r}, {labels[1]!r}, ... {labels[-1]!r}"
            else:
                summary = ", ".join(repr(label) for label in labels)
            lines.append(f"    [{axis}] {name} ({self.values.shape[axis]}): {summary}")
        finite = np.isfinite(self.values) if np.issubdtype(self.values.dtype, np.number) else None
        if finite is not None:
            lines.append(f"  missing:    {100 * (1 - finite.mean()):.1f}% NaN")
        trials = self.metadata.get("button_positions", {}).get("per_trial", [])
        n_buttons = len(trials[0]["positions"]) if trials else 0
        has_start = any(record.get("start_position") is not None for record in trials)
        lines.append(
            f"  screen layout: {len(trials)} trials with response-button positions "
            f"({n_buttons} buttons), start-button position {'available' if has_start else 'not available'}"
        )
        if self.metadata.get("averaged_over"):
            lines.append(f"  averaged over: {', '.join(self.metadata['averaged_over'])}")
        return "\n".join(lines)

    @property
    def array(self):
        """Alias for the underlying NumPy array."""
        return self.values

    @property
    def shape(self):
        return self.values.shape

    @property
    def ndim(self):
        return self.values.ndim

    @property
    def dtype(self):
        return self.values.dtype

    def __array__(self, dtype=None, copy=None):
        array = np.asarray(self.values, dtype=dtype)
        if copy:
            array = array.copy()
        return array

    def __getitem__(self, key):
        return self.values[key]

    def average(self, axis):
        """Average along one or more named dimensions or numeric axes.

        Missing combinations are represented by NaN in extracted data; masked
        averaging omits those values and keeps all-NaN slices as NaN.
        """
        axes = self._normalise_axes(axis)
        averaged = np.ma.masked_invalid(self.values).mean(axis=axes).filled(np.nan)
        averaged = np.asarray(averaged)
        removed = {self.dimensions[index] for index in axes}
        dimensions = tuple(name for name in self.dimensions if name not in removed)
        coordinates = {
            name: labels for name, labels in self.coordinates.items() if name not in removed
        }
        metadata = dict(self.metadata)
        metadata["averaged_over"] = list(self.metadata.get("averaged_over", [])) + [
            self.dimensions[index] for index in axes
        ]
        return type(self)(averaged, dimensions, coordinates, metadata=metadata)

    def downsample(self, rate_hz):
        """Return a copy resampled to ``rate_hz`` by keeping every n-th time point.

        The ``timepoints`` dimension must be on a regular grid in ms (as
        created by :func:`extract_distance`, see ``metadata["sampling_interval"]``).
        ``rate_hz`` must not exceed the current rate; the step is rounded to a
        whole number of samples, so e.g. 400 Hz from 1000 Hz becomes 333 Hz.
        """
        if "timepoints" not in self.dimensions:
            raise ValueError("Downsampling needs a 'timepoints' dimension.")
        interval = self.metadata.get("sampling_interval")
        if not interval:
            raise ValueError("The sampling interval (ms) is unknown; cannot downsample.")
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive.")
        current_rate = 1000.0 / interval
        if rate_hz > current_rate + 1e-9:
            raise ValueError(f"rate_hz ({rate_hz}) is higher than the current rate ({current_rate:g} Hz).")
        step = max(1, int(round(current_rate / rate_hz)))
        axis = self.axis_of("timepoints")
        slicer = [slice(None)] * self.ndim
        slicer[axis] = slice(None, None, step)
        coordinates = dict(self.coordinates)
        if "timepoints" in coordinates:
            coordinates["timepoints"] = coordinates["timepoints"][::step]
        metadata = dict(self.metadata)
        metadata["sampling_interval"] = interval * step
        metadata["downsampled_from_hz"] = current_rate
        return type(self)(self.values[tuple(slicer)], self.dimensions, coordinates, metadata=metadata)

    def save(self, path, downsample_hz=None):
        """Save values and JSON metadata to a compressed ``.npz`` archive.

        ``downsample_hz`` (e.g. 250 or 500) optionally downsamples the time
        axis first (see :meth:`downsample`); the object itself is unchanged.
        Returns the path written. Reload with :meth:`load`.
        """
        if downsample_hz is not None:
            return self.downsample(downsample_hz).save(path)
        path = Path(path)
        if path.suffix.lower() != ".npz":
            path = path.with_suffix(".npz")
        metadata = {
            "dimensions": self.dimensions,
            "coordinates": self.coordinates,
            "metadata": self.metadata,
        }
        with path.open("wb") as archive:
            np.savez_compressed(
                archive,
                values=self.values,
                metadata=np.asarray(json.dumps(metadata, default=_json_default)),
            )
        return path

    @classmethod
    def load(cls, path):
        """Load a ``MotorMovementData`` archive without enabling pickle."""
        with np.load(path, allow_pickle=False) as archive:
            values = archive["values"]
            metadata = json.loads(str(archive["metadata"].item()))
        return cls(
            values,
            metadata["dimensions"],
            metadata.get("coordinates"),
            metadata=metadata.get("metadata"),
        )

    def _normalise_axes(self, axis):
        if isinstance(axis, str):
            axis = (axis,)
        elif isinstance(axis, Sequence):
            axis = tuple(axis)
        else:
            axis = (axis,)
        normalised = []
        for item in axis:
            if isinstance(item, str):
                if item not in self.dimensions:
                    raise ValueError(f"Unknown dimension {item!r}; choose from {self.dimensions!r}.")
                index = self.dimensions.index(item)
            elif isinstance(item, (int, np.integer)):
                index = int(item)
                if index < 0:
                    index += self.ndim
                if index < 0 or index >= self.ndim:
                    raise ValueError(f"Axis {item} is out of bounds for {self.ndim} dimensions.")
            else:
                raise TypeError("axis must be a dimension name, integer, or sequence of these.")
            if index in normalised:
                raise ValueError(f"Axis {item!r} was specified more than once.")
            normalised.append(index)
        if not normalised:
            raise ValueError("At least one axis must be specified.")
        return tuple(normalised)


# Backwards-compatible name
MouseTrackingData = MotorMovementData


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Value of type {type(value).__name__} cannot be saved in metadata.")
