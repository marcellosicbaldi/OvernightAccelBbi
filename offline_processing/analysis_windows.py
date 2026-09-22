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


def first_overlapping_interval(starts, ends, intervals):
    """Index of the first positive-duration overlap, or -1; touching is not overlap."""
    bounds = validate_intervals(intervals)
    starts, ends = pd.DatetimeIndex(starts), pd.DatetimeIndex(ends)
    if (starts.hasnans or ends.hasnans or len(starts) != len(ends)
            or (len(starts) and (starts.tz is None or ends.tz is None))
            or (ends <= starts).any()):
        raise ValueError("Events need valid timezone-aware start < end bounds")
    ids = np.full(len(starts), -1, dtype=int)
    if len(bounds):
        positions = pd.DatetimeIndex(bounds.end).searchsorted(starts, side="right")
        valid = ((positions < len(bounds))
                 & (pd.DatetimeIndex(bounds.start).take(np.minimum(positions, len(bounds) - 1)) < ends))
        ids[valid] = positions[valid]
    return ids


def intersect_intervals(intervals, coverage):
    """Intersect two interval sets without joining across either set's gaps."""
    intervals, coverage = validate_intervals(intervals), validate_intervals(coverage)
    pieces = []
    i = j = 0
    while i < len(intervals) and j < len(coverage):
        a, b = intervals.iloc[i], coverage.iloc[j]
        start, end = max(a.start, b.start), min(a.end, b.end)
        if start < end:
            pieces.append((start, end))
        if a.end <= b.end:
            i += 1
        else:
            j += 1
    return validate_intervals(pd.DataFrame(pieces, columns=["start", "end"]))


def classify_bursts(bursts, sleep_intervals, wake_intervals):
    """Label entire bursts: ANY counted-wake overlap wins over sleep.

    Sleep requires full containment in a scored sleep interval. Other events are
    unclassified. No clipping, splitting, duration threshold, or onset-only rule
    is applied. Nanosecond comparisons preserve even sub-millisecond overlaps.
    """
    sleep = validate_intervals(sleep_intervals)
    wake = validate_intervals(wake_intervals)
    if not intersect_intervals(sleep, wake).empty:
        raise ValueError("Sleep and wake intervals must not overlap")
    out = bursts.copy()
    wake_id = first_overlapping_interval(out.start, out.end, wake)
    sleep_id = interval_ids(out.start, sleep)
    full_sleep = np.zeros(len(out), dtype=bool)
    valid = sleep_id >= 0
    full_sleep[valid] = pd.DatetimeIndex(out.end)[valid] <= pd.DatetimeIndex(sleep.end).take(sleep_id[valid])
    out["overlaps_wake"] = wake_id >= 0
    out["sleep_wake"] = np.where(wake_id >= 0, "wake", np.where(full_sleep, "sleep", "unclassified"))
    return out
