"""VH2015 z-angle sleep/wake for an already diary-cropped Garmin XYZ table.

Core pandas rolling median, GGIR epoch average, and stable-angle classifier
ported from the user's sleep_menghini/vh2015_sleep.py. No diary selection,
GENEActiv page handling, 15-minute trimming, or PSG alignment is performed.
Garmin timestamps are regularized separately inside each acquisition run.
"""

import numpy as np
import pandas as pd

from analysis_windows import validate_intervals


def _r_seq_round_indices(length: int, by: float) -> np.ndarray:
    """Return zero-based Python indices equivalent to R seq(1, length, by)."""

    n_select = int(np.floor(((float(length) - 1.0) / by) + 1e-10)) + 1
    select = 1.0 + (np.arange(n_select, dtype=np.float64) * by)
    r_indices = np.rint(select).astype(np.int64)
    r_indices = r_indices[(r_indices >= 1) & (r_indices <= length)]
    return r_indices - 1


def ggir_average_per_epoch(x: np.ndarray, sample_rate: float, epoch_seconds: int) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    csum = np.concatenate(([0.0], np.cumsum(x64)))
    idx = _r_seq_round_indices(len(csum), by=sample_rate * epoch_seconds)
    values = csum[idx]
    denom = np.abs(np.diff(idx + 1))
    return np.diff(values) / denom


def _replace_edge_zeros_like_ggir(xm: np.ndarray, sample_rate: float) -> np.ndarray:
    xm = np.asarray(xm, dtype=np.float64).copy()
    if len(xm) == 0:
        return xm

    s2check = int(max(sample_rate * 60, 1000))
    head_n = min(len(xm), s2check)
    head = xm[:head_n]
    head_nonzero = np.flatnonzero(head != 0)
    if len(head_nonzero) > 0:
        head[head == 0] = head[head_nonzero[0]]
        xm[:head_n] = head

    ln = len(xm)
    tail_check = min(ln - 1, s2check)
    tail_start = max(0, ln - tail_check - 1)
    tail = xm[tail_start:ln]
    tail_nonzero = np.flatnonzero(tail != 0)
    if len(tail_nonzero) > 0:
        lastvalue = tail[tail_nonzero[0]]
    else:
        all_nonzero = np.flatnonzero(xm != 0)
        lastvalue = xm[all_nonzero[0]] if len(all_nonzero) > 0 else 0.0
    tail[tail == 0] = lastvalue
    xm[tail_start:ln] = tail
    return xm


def ggir_roll_median_axis_pandas(
    x: np.ndarray,
    sample_rate: float,
    match_ggir_resize_bug: bool = True,
) -> np.ndarray:
    """Rolling-median axis pre-processing from GGIR g.applymetrics.

    The odd-looking resize branch mirrors GGIR 3.3-6 exactly. In that version,
    when the repeated downsampled median is longer than the original vector, the
    original vector is returned.
    """

    x = np.asarray(x, dtype=np.float64)
    stepsize = max(int(np.floor(sample_rate / 10)), 1)
    newsf = 10 if stepsize > 1 else sample_rate
    winsi = int(round(newsf * 5))
    if round(winsi / 2) == (winsi / 2):
        winsi += 1
    if (winsi % 2) == 0:
        winsi += 1

    down = x[::stepsize]
    xm = (
        pd.Series(down)
        .rolling(window=winsi, center=True, min_periods=winsi)
        .median()
        .fillna(0)
        .to_numpy(dtype=np.float64)
    )
    xm = _replace_edge_zeros_like_ggir(xm, sample_rate)
    xm = np.repeat(xm, stepsize)
    if len(xm) > len(x):
        return x.copy() if match_ggir_resize_bug else xm[: len(x)]
    if len(xm) < len(x):
        xm = np.concatenate((xm, np.repeat(xm[-1], len(x) - len(xm))))
    return xm


def vh2015_sleep_wake_from_anglez(
    anglez: np.ndarray,
    epoch_seconds: int = 5,
    timethreshold_minutes: int = 5,
    anglethreshold_degrees: float = 5.0,
) -> np.ndarray:
    """Classify sustained flat anglez periods as sleep.

    A sleep bout is any contiguous run of epochs where adjacent anglez changes
    stay within `anglethreshold_degrees` for at least `timethreshold_minutes`.
    Edge runs at the start or end of the analysed window are included.
    """

    anglez = np.asarray(anglez, dtype=np.float64)
    sleep = np.zeros(len(anglez), dtype=np.int8)
    if len(anglez) == 0:
        return sleep

    min_sleep_epochs = int(np.ceil((timethreshold_minutes * 60) / epoch_seconds))
    min_sleep_epochs = max(min_sleep_epochs, 1)
    large_change = np.abs(np.diff(anglez)) > anglethreshold_degrees

    run_start = 0
    for change_idx in np.flatnonzero(large_change):
        run_stop = change_idx + 1
        if run_stop - run_start >= min_sleep_epochs:
            sleep[run_start:run_stop] = 1
        run_start = change_idx + 1

    if len(anglez) - run_start >= min_sleep_epochs:
        sleep[run_start:] = 1

    return sleep


def compute_anglez(acc, sample_rate, epoch_seconds=5, decimals=4,
                   match_ggir_resize_bug=True):
    """Pandas branch of the reference compute_anglez, with the same defaults."""
    x, y, z = (ggir_roll_median_axis_pandas(acc[:, i], sample_rate, match_ggir_resize_bug)
               for i in range(3))
    with np.errstate(divide="ignore", invalid="ignore"):
        angle = np.degrees(np.arctan(z / np.sqrt(x * x + y * y)))
    return np.round(ggir_average_per_epoch(angle, sample_rate, epoch_seconds), decimals)


def _positive(value, name, allow_zero=False):
    if (not np.isscalar(value) or isinstance(value, (str, bool)) or not np.isfinite(value)
            or (value < 0 if allow_zero else value <= 0)):
        raise ValueError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}")


def _runs(values):
    edges = np.r_[0, np.flatnonzero(np.diff(values) != 0) + 1, len(values)]
    return zip(edges[:-1], edges[1:]) if len(values) else []


def apply_minimum_wake(epochs, min_wake_episode_seconds=None):
    """Reclassify short raw wake runs as sleep; None retains every wake run.

    Durations use full five-second epochs. The comparison is inclusive: an
    episode exactly as long as the minimum counts. Runs never cross acquisition
    boundaries. Raw labels and every raw wake episode remain available for QC.
    """
    if min_wake_episode_seconds is not None:
        _positive(min_wake_episode_seconds, "min_wake_episode_seconds", allow_zero=True)
    result = epochs.copy().reset_index(drop=True)
    result["wake"] = result.wake_raw.astype(bool)
    episodes = []
    for segment_id, group in result.groupby("source_segment_id", sort=False):
        values = group.wake_raw.to_numpy(dtype=bool)
        for first, stop in _runs(values):
            if not values[first]:
                continue
            start, end = group.start.iloc[first], group.end.iloc[stop - 1]
            duration = (end - start).total_seconds()
            counted = min_wake_episode_seconds is None or duration >= min_wake_episode_seconds
            episodes.append((start, end, duration, segment_id, counted))
            if not counted:
                result.loc[group.index[first:stop], "wake"] = False
    result["sleep"] = ~result.wake
    episodes = pd.DataFrame(episodes, columns=["start", "end", "duration_s", "source_segment_id", "counted_wake"])
    for column in ("start", "end"):
        episodes[column] = pd.to_datetime(episodes[column], utc=True)
    episodes["counted_wake"] = episodes.counted_wake.astype(bool)
    return result, episodes


def detect_sleep_wake(accel_df, *, sampling_rate=100.0, naive_timezone="UTC",
                      max_gap_s=0.25, min_wake_episode_seconds=None):
    """Score ONLY the supplied cropped table; return epochs, episodes and coverage.

    XYZ may be in mg or g: common scale cancels in the angle. Sort timestamps,
    average duplicate XYZ samples, interpolate jitter within acquisition runs,
    and split gaps >max_gap_s. Five-second epochs start at each run's first sample.
    Partial trailing epochs are unclassified (kept by whole-night analysis only).
    Nonfinite/zero-vector inputs or undefined epoch angles raise an error instead
    of silently becoming sleep. Time bounds are UTC and end-exclusive.
    """
    _positive(sampling_rate, "sampling_rate")
    _positive(max_gap_s, "max_gap_s")
    if min_wake_episode_seconds is not None:
        _positive(min_wake_episode_seconds, "min_wake_episode_seconds", allow_zero=True)
    columns = ["accel_x", "accel_y", "accel_z"]
    xyz = accel_df[columns].to_numpy(dtype=float)
    if len(xyz) < 2 or not np.isfinite(xyz).all() or (np.linalg.norm(xyz, axis=1) == 0).any():
        raise ValueError("Need at least two finite, nonzero XYZ samples")
    times = pd.DatetimeIndex(pd.to_datetime(accel_df.sample_time))
    if times.hasnans:
        raise ValueError("sample_time contains missing timestamps")
    if times.tz is None:
        times = times.tz_localize(naive_timezone, ambiguous="raise", nonexistent="raise")
    times = times.tz_convert("UTC")
    raw = pd.DataFrame(xyz, index=times, columns=columns).groupby(level=0, sort=True).mean()
    period = pd.Timedelta(seconds=1 / sampling_rate)
    if period.value < 1:
        raise ValueError("sampling_rate exceeds timestamp resolution")
    gaps = (raw.index[1:] - raw.index[:-1]).total_seconds().to_numpy()
    boundaries = np.r_[0, np.flatnonzero(gaps > max_gap_s) + 1, len(raw)]
    stepsize = max(int(np.floor(sampling_rate / 10)), 1)
    median_window = int(round((10 if stepsize > 1 else sampling_rate) * 5))
    if median_window % 2 == 0:
        median_window += 1
    coverage, parts = [], []
    for segment_id, (first, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        run = raw.iloc[first:stop]
        start, end = run.index[0], run.index[-1] + period
        coverage.append((start, end))
        if (end - start).total_seconds() < 5:
            continue
        grid = pd.date_range(start, run.index[-1], freq=period)
        if len(grid[::stepsize]) < median_window:
            continue  # The reference requires a full odd rolling-median window.
        elapsed = (run.index - start).total_seconds().to_numpy()
        target = (grid - start).total_seconds().to_numpy()
        regular = np.column_stack([np.interp(target, elapsed, run[c].to_numpy()) for c in columns])
        # Fix the reference's optional resize bug for arbitrary Garmin crop lengths:
        # retain the rolling medians when their repeated length overshoots by <10 samples.
        anglez = compute_anglez(regular, sampling_rate, match_ggir_resize_bug=False)
        if not np.isfinite(anglez).all():
            raise ValueError("Undefined z-angle epochs; inspect XYZ data before classifying sleep")
        sleep = vh2015_sleep_wake_from_anglez(anglez)
        epoch_starts = pd.date_range(start, periods=len(anglez), freq="5s")
        parts.append(pd.DataFrame({"start": epoch_starts, "end": epoch_starts + pd.Timedelta(seconds=5),
                                   "anglez": anglez, "sleep_raw": sleep.astype(bool),
                                   "wake_raw": ~sleep.astype(bool), "source_segment_id": segment_id}))
    epochs = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["start", "end", "anglez", "sleep_raw", "wake_raw", "source_segment_id"])
    for column in ("start", "end"):
        epochs[column] = pd.to_datetime(epochs[column], utc=True)
    epochs, episodes = apply_minimum_wake(epochs, min_wake_episode_seconds)
    coverage = validate_intervals(pd.DataFrame(coverage, columns=["start", "end"]))
    report = {"method": "VH2015 z-angle", "epoch_seconds": 5,
              "sleep_min_minutes": 5, "angle_threshold_degrees": 5.0,
              "sampling_rate_hz": sampling_rate, "max_acceleration_gap_s": max_gap_s,
              "min_wake_episode_seconds": min_wake_episode_seconds,
              "duplicate_timestamps_averaged": int(times.duplicated().sum()),
              "acquisition_runs": len(coverage), "scored_epochs": len(epochs),
              "raw_wake_episodes": len(episodes), "counted_wake_episodes": int(episodes.counted_wake.sum()),
              "unclassified_seconds": float((coverage.end - coverage.start).dt.total_seconds().sum() - 5 * len(epochs)),
              "match_ggir_resize_bug": False}
    return {"epochs": epochs, "wake_episodes": episodes, "coverage": coverage, "report": report}


def select_analysis_intervals(result, mode="whole"):
    """Choose the diary-cropped coverage, scored sleep, or counted wake.

    'whole' is the complete supplied diary night, without a second SPT estimate.
    Short wake runs rejected by the optional minimum are included in sleep_only.
    Unclassified partial epochs and acquisition gaps never become sleep/wake.
    """
    if mode not in ("whole", "sleep_only", "wake_only"):
        raise ValueError("mode must be 'whole', 'sleep_only', or 'wake_only'")
    if mode == "whole":
        return validate_intervals(result["coverage"])
    state = "sleep" if mode == "sleep_only" else "wake"
    bounds = []
    for _, group in result["epochs"].groupby("source_segment_id", sort=False):
        selected = group[state].to_numpy(dtype=bool)
        for first, stop in _runs(selected):
            if selected[first]:
                bounds.append((group.start.iloc[first], group.end.iloc[stop - 1]))
    return validate_intervals(pd.DataFrame(bounds, columns=["start", "end"]))
