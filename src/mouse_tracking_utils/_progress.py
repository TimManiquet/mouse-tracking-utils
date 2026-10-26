"""Optional progress bars (tqdm), shared by the long-running functions."""

try:
    from tqdm.auto import tqdm
except ImportError:  # progress bars are a convenience, never a requirement
    tqdm = None


def track(iterable, enabled, total=None, desc=None, unit="trial"):
    """Wrap ``iterable`` in a progress bar when ``enabled`` and tqdm is available."""
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit, leave=False)
