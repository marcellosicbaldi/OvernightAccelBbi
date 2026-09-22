"""End-exclusive analysis intervals shared by acceleration, HR and HRV."""

import numpy as np
import pandas as pd


def validate_intervals(intervals, recording_start, recording_end):
    """Clip sorted, disjoint intervals to the recording; keep touching ones separate.

    None selects the whole recording; an empty table selects nothing. Clocks
    must be compatible with the bounds (localize naive wall times beforehand).
    """
    begin, finish = pd.Timestamp(recording_start), pd.Timestamp(recording_end)
    if (begin.tz is None) != (finish.tz is None):
        raise ValueError("Recording bounds must use the same timezone convention")
    if pd.isna(begin) or pd.isna(finish) or finish <= begin:
        raise ValueError("Recording end must follow its valid start")
    if intervals is None:
        intervals = pd.DataFrame({"start": [begin], "end": [finish]})
    result = intervals[["start", "end"]].copy()
    for column in ("start", "end"):
        index = pd.DatetimeIndex(result[column])
        if not len(index):
            result[column] = pd.Series(index=result.index, dtype=pd.DatetimeIndex([begin]).dtype)
            continue
        if (index.tz is None) != (begin.tz is None):
            raise ValueError("Use the same timezone convention for analysis intervals and bounds")
        result[column] = index.tz_convert(begin.tz) if begin.tz is not None else index
    if result.isna().any().any() or (result.end <= result.start).any():
        raise ValueError("Analysis intervals need valid, increasing start/end bounds")
    result = result.sort_values("start", kind="stable")
    if (result.start.iloc[1:].reset_index(drop=True)
            < result.end.iloc[:-1].reset_index(drop=True)).any():
        raise ValueError("Analysis intervals must not overlap")
    result["start"] = result.start.clip(lower=begin)
    result["end"] = result.end.clip(upper=finish)
    result = result.loc[result.start < result.end].reset_index(drop=True)
    result.index.name = "analysis_interval_id"
    return result


def interval_ids(times, intervals):
    """Assign each time to one [start, end) interval, or -1 outside selection."""
    times = pd.DatetimeIndex(times)
    if times.hasnans:
        raise ValueError("Missing timestamps cannot be assigned to analysis intervals")
    if intervals.empty:
        return np.full(len(times), -1, dtype=int)
    starts, ends = pd.DatetimeIndex(intervals.start), pd.DatetimeIndex(intervals.end)
    if (times.tz is None) != (starts.tz is None):
        raise ValueError("Use the same timezone convention for times and analysis intervals")
    ids = starts.searchsorted(times, side="right") - 1
    inside = (ids >= 0) & (times < ends.take(np.maximum(ids, 0)))
    return np.where(inside, ids, -1)


def select_observations(observations, intervals):
    """Keep selected HR rows and mark even sub-second excluded gaps as barriers."""
    ids = interval_ids(observations.index, intervals)
    result = observations.loc[ids >= 0].copy()
    result["analysis_interval_id"] = ids[ids >= 0]
    result["break_before"] = (result.break_before
                              | result.analysis_interval_id.ne(result.analysis_interval_id.shift()))
    return result


def detect_bursts_in_intervals(accel_df, intervals, *, sampling_rate=100.0,
                              input_unit="mg", alfa=0.040, max_gap_s=0.25):
    """Regularize and detect independently in each selected raw-XYZ interval.

    Returns bursts, per-interval signals, and QC. Very short intervals that
    cannot support filter padding are reported, with no invented detections.
    No magnitude filtering, extrema interpolation, or merging spans intervals.
    """
    from detect_acc_bursts import detect_bursts, prepare_acceleration

    times = pd.DatetimeIndex(accel_df.sample_time)
    period = pd.Timedelta(seconds=1 / sampling_rate)
    intervals = validate_intervals(intervals, times.min(), times.max() + period)
    ids = interval_ids(times, intervals)
    parts, signals, quality = [], [], []
    for interval_id, interval in intervals.iterrows():
        raw = accel_df.loc[ids == interval_id]
        if raw.sample_time.nunique() < 2:
            quality.append({"analysis_interval_id": interval_id, "input_samples": len(raw),
                            "status": "too_short_for_filter"})
            continue
        magnitude, qc = prepare_acceleration(raw, sampling_rate, input_unit, max_gap_s)
        # Only complete sample periods can contribute AUC or detector bounds.
        magnitude = magnitude.loc[magnitude.index + period <= interval.end]
        qc.update(analysis_interval_id=interval_id, regular_samples=len(magnitude), status="ok")
        if len(magnitude) <= 51:
            qc["status"] = "too_short_for_filter"
            quality.append(qc)
            continue
        if not 0.5 <= qc["median_magnitude_g"] <= 1.5:
            raise ValueError("Median magnitude is not near 1 g. Check input_unit and calibration.")
        bursts, diagnostic = detect_bursts(magnitude, sampling_rate, alfa=alfa, return_signals=True)
        bursts["analysis_interval_id"] = interval_id
        parts.append(bursts)
        signals.append({**diagnostic, "magnitude": magnitude, "analysis_interval_id": interval_id})
        quality.append(qc)
    empty = pd.DataFrame({"start": pd.Series(dtype=times.dtype), "end": pd.Series(dtype=times.dtype),
                          "duration": pd.Series(dtype="timedelta64[ns]"),
                          "peak-to-peak": pd.Series(dtype=float), "AUC": pd.Series(dtype=float),
                          "analysis_interval_id": pd.Series(dtype=int)})
    bursts = pd.concat(parts, ignore_index=True) if parts else empty
    bursts.index.name = "burst_id"
    return bursts, signals, pd.DataFrame(quality)
