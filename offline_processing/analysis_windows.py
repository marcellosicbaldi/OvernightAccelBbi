"""Explicit, end-exclusive analysis intervals shared by acceleration and HR."""

import numpy as np
import pandas as pd


def validate_intervals(intervals):
    """Return sorted UTC bounds; preserve touching intervals as separate runs."""
    out = intervals[["start", "end"]].copy()
    for column in ("start", "end"):
        times = pd.DatetimeIndex(out[column])
        if times.hasnans or (len(times) and times.tz is None):
            raise ValueError("Analysis intervals need timezone-aware, nonmissing bounds")
        out[column] = pd.to_datetime(times, utc=True)
    out = out.sort_values("start").reset_index(drop=True)
    if (out.end <= out.start).any():
        raise ValueError("Analysis interval end must follow start")
    if len(out) > 1 and (out.start.iloc[1:].array < out.end.iloc[:-1].array).any():
        raise ValueError("Analysis intervals must not overlap")
    out.index.name = "analysis_interval_id"
    return out


def interval_ids(times, intervals):
    """Assign timestamps to [start, end), or -1 outside the selected intervals."""
    bounds = validate_intervals(intervals)
    times = pd.DatetimeIndex(times)
    if times.hasnans or (len(times) and times.tz is None):
        raise ValueError("Sample times must be timezone-aware and nonmissing")
    ids = np.full(len(times), -1, dtype=int)
    if len(bounds):
        positions = pd.DatetimeIndex(bounds.start).searchsorted(times, side="right") - 1
        valid = (positions >= 0) & (times < pd.DatetimeIndex(bounds.end).take(np.maximum(positions, 0)))
        ids[valid] = positions[valid]
    return ids


def select_observations(observations, intervals):
    """Filter HR rows and mark every interval boundary as an interpolation break."""
    ids = interval_ids(observations.index, intervals)
    selected = observations.loc[ids >= 0].copy()
    ids = ids[ids >= 0]
    selected["analysis_interval_id"] = ids
    if len(selected):
        selected["break_before"] = selected.break_before.to_numpy(dtype=bool) | np.r_[True, np.diff(ids) != 0]
    return selected
