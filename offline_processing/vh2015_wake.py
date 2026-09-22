"""VH2015 z-angle sleep/wake labels for an already diary-cropped Garmin table.

Algorithmic functions ported from sleep_menghini/vh2015_sleep.py:
GGIR five-second rolling medians, mean z-angle per five-second epoch,
and sustained adjacent-angle-change classification. No diary selection,
GENEActiv resampling, 15-minute trimming, or SPT estimation happens here.
The GGIR 3.3-6 resize bug is disabled for arbitrary Garmin crop lengths.
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


def compute_anglez(
    acc: np.ndarray,
    sample_rate: float,
    epoch_seconds: int = 5,
    decimals: int = 4,
    match_ggir_resize_bug: bool = False,
) -> np.ndarray:
    roller = ggir_roll_median_axis_pandas

    x = roller(acc[:, 0], sample_rate, match_ggir_resize_bug)
    y = roller(acc[:, 1], sample_rate, match_ggir_resize_bug)
    z = roller(acc[:, 2], sample_rate, match_ggir_resize_bug)
    with np.errstate(divide="ignore", invalid="ignore"):
        angle = np.degrees(np.arctan(z / np.sqrt((x * x) + (y * y))))
    if not np.isfinite(angle).all():
        raise ValueError("Rolling XYZ has an undefined z-angle; need nonzero orientation and >5 s of data")
    anglez = ggir_average_per_epoch(angle, sample_rate, epoch_seconds)
    return np.round(anglez, decimals=decimals)


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
    if anglez.ndim != 1 or not np.isfinite(anglez).all():
        raise ValueError("anglez must be a finite one-dimensional array")
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


def _positive(value, name, allow_zero=False):
    if (isinstance(value, (str, bool)) or not np.isscalar(value)
            or not np.isfinite(value) or (value < 0 if allow_zero else value <= 0)):
        raise ValueError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}")


def _runs(values):
    boundaries = np.r_[0, np.flatnonzero(np.diff(values)) + 1, len(values)]
    return zip(boundaries[:-1], boundaries[1:]) if len(values) else []


def classify_epochs(anglez, start, *, epoch_seconds=5, timethreshold_minutes=5,
                    anglethreshold_degrees=5.0, min_wake_duration_s=None):
    """Label complete epochs, preserving raw wake before optional bout suppression.

    Wake bouts strictly shorter than the minimum become effective sleep.
    None (default) and zero retain every detected wake bout, including edges.
    """
    for name, value in (("epoch_seconds", epoch_seconds),
                        ("timethreshold_minutes", timethreshold_minutes),
                        ("anglethreshold_degrees", anglethreshold_degrees)):
        _positive(value, name)
    if min_wake_duration_s is not None:
        _positive(min_wake_duration_s, "min_wake_duration_s", allow_zero=True)
    raw_sleep = vh2015_sleep_wake_from_anglez(
        anglez, epoch_seconds, timethreshold_minutes, anglethreshold_degrees,
    ).astype(bool)
    wake = ~raw_sleep
    suppressed = np.zeros(len(wake), dtype=bool)
    if min_wake_duration_s is not None:
        for first, stop in _runs(wake):
            if wake[first] and (stop - first) * epoch_seconds < min_wake_duration_s:
                suppressed[first:stop] = True
        wake[suppressed] = False
    times = pd.date_range(start, periods=len(raw_sleep), freq=pd.Timedelta(seconds=epoch_seconds))
    return pd.DataFrame({"start": times, "end": times + pd.Timedelta(seconds=epoch_seconds),
                         "anglez": np.asarray(anglez, dtype=float), "raw_wake": ~raw_sleep,
                         "wake": wake, "sleep": ~wake, "wake_suppressed": suppressed})


def state_episodes(epochs, state="wake"):
    """Return contiguous effective or raw state bouts with [start, end) bounds."""
    rows = []
    for first, stop in _runs(epochs[state].to_numpy(dtype=bool)):
        if epochs[state].iloc[first]:
            rows.append((epochs.start.iloc[first], epochs.end.iloc[stop - 1]))
    result = pd.DataFrame({"start": pd.Series([r[0] for r in rows], dtype=epochs.start.dtype),
                           "end": pd.Series([r[1] for r in rows], dtype=epochs.end.dtype)})
    result["duration_s"] = (result.end - result.start).dt.total_seconds()
    return result


def detect_vh2015(accel_df, *, sampling_rate=100.0, max_gap_s=0.25,
                  min_wake_duration_s=None, timethreshold_minutes=5,
                  anglethreshold_degrees=5.0):
    """Use raw Garmin sample_time/accel_x/y/z within the caller's diary crop.

    Sort and average duplicate XYZ timestamps, then interpolate axes on a
    regular grid. Units cancel in the angle; all axes must use the same unit.
    Reject nonfinite XYZ and acquisition gaps over max_gap_s, like the burst
    preparation. Complete five-second epochs begin at the first input sample;
    the final incomplete epoch is unclassified, never silently called sleep.
    Whole-night selection still includes this tail. Input tables are untouched.
    """
    _positive(sampling_rate, "sampling_rate")
    _positive(max_gap_s, "max_gap_s")
    times = pd.DatetimeIndex(accel_df.sample_time)
    xyz = accel_df[["accel_x", "accel_y", "accel_z"]].to_numpy(dtype=float)
    if times.hasnans or len(times) < 2 or not np.isfinite(xyz).all():
        raise ValueError("Need at least two finite XYZ samples with valid timestamps")
    raw = pd.DataFrame(xyz, index=times).groupby(level=0, sort=True).mean()
    if len(raw) < 2:
        raise ValueError("Need at least two distinct timestamps")
    elapsed = (raw.index - raw.index[0]).total_seconds().to_numpy()
    gaps = np.diff(elapsed)
    if gaps.max() > max_gap_s + 1e-9:
        raise ValueError("Acceleration gap exceeds max_gap_s; analyze continuous recordings separately")
    period = pd.Timedelta(seconds=1 / sampling_rate)
    if period.value <= 0:
        raise ValueError("sampling_rate exceeds timestamp resolution")
    grid = pd.date_range(raw.index[0], raw.index[-1], freq=period)
    query = (grid - grid[0]).total_seconds().to_numpy()
    regular = np.column_stack([np.interp(query, elapsed, raw[axis]) for axis in raw])
    # The source rolling median needs a full centered five-second window.
    # Do not infer wake from a recording too short to compute that metric.
    stepsize = max(int(np.floor(sampling_rate / 10)), 1)
    newsf = 10 if stepsize > 1 else sampling_rate
    window = int(round(newsf * 5))
    window += int(window % 2 == 0)
    anglez = (compute_anglez(regular, sampling_rate) if len(regular[::stepsize]) >= window
              else np.array([], dtype=float))
    epochs = classify_epochs(anglez, grid[0], min_wake_duration_s=min_wake_duration_s,
                             timethreshold_minutes=timethreshold_minutes,
                             anglethreshold_degrees=anglethreshold_degrees)
    end = raw.index[-1] + period
    classified_end = epochs.end.iloc[-1] if len(epochs) else grid[0]
    report = {"algorithm": "VH2015 from sleep_menghini/vh2015_sleep.py",
              "epoch_seconds": 5, "timethreshold_minutes": timethreshold_minutes,
              "anglethreshold_degrees": anglethreshold_degrees,
              "min_wake_duration_s": min_wake_duration_s, "match_ggir_resize_bug": False,
              "sampling_rate_hz": sampling_rate, "max_gap_s": max_gap_s,
              "duplicate_timestamps_averaged": int(times.duplicated().sum()),
              "unclassified_tail_s": (end - classified_end).total_seconds(),
              "suppressed_wake_s": float(epochs.wake_suppressed.sum() * 5)}
    return {"epochs": epochs, "wake_episodes": state_episodes(epochs),
            "raw_wake_episodes": state_episodes(epochs, "raw_wake"),
            "recording_start": grid[0], "recording_end": end, "report": report}


def select_analysis_intervals(result, mode="whole_spt"):
    """whole_spt means the entire diary crop, not an automatically inferred SPT."""
    if mode not in ("whole_spt", "sleep_only", "wake_only"):
        raise ValueError("mode must be 'whole_spt', 'sleep_only', or 'wake_only'")
    intervals = None if mode == "whole_spt" else state_episodes(
        result["epochs"], "sleep" if mode == "sleep_only" else "wake",
    )
    return validate_intervals(intervals, result["recording_start"], result["recording_end"])


