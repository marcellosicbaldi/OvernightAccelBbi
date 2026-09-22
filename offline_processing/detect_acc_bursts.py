"""Offline movement-burst detection from acceleration magnitude in g.

The original 0.1-10 Hz, order-8 Butterworth filter, groups of ten extrema,
strict threshold crossing, and <5 s merging rule are retained. The supplied
wrist threshold is 0.020 g for the envelope, not for raw magnitude or STD.
"""

import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from analysis_windows import validate_intervals


def _finite_values(acc):
    if not isinstance(acc, pd.Series):
        raise TypeError("acc must be a pandas Series")
    values = acc.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Acceleration contains missing or non-finite samples")
    return values


def _positive_number(value, name, allow_zero=False):
    if value is None or not np.isscalar(value) or isinstance(value, (str, bool)):
        raise ValueError(f"{name} must be a finite {'non-negative' if allow_zero else 'positive'} number")
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"Invalid {name}: {value}")


def prepare_acceleration(accel_df, sampling_rate=100.0, input_unit="mg", max_gap_s=0.25):
    """Build a regular magnitude Series in g from the notebook's FIT table.

    Sort timestamps, average magnitudes at duplicate timestamps, and linearly
    interpolate magnitude onto a regular grid. Never interpolate across a gap
    larger than max_gap_s; split such recordings before calling this function.
    The input unit is explicit because FIT field labels can disagree with the
    stored values. This is unit conversion, not a sensor calibration.

    Returns (magnitude_g, quality), including duplicate count and largest gap.
    """
    _positive_number(sampling_rate, "sampling_rate")
    _positive_number(max_gap_s, "max_gap_s")
    if input_unit not in ("g", "mg"):
        raise ValueError("input_unit must be 'g' or 'mg'")
    xyz = accel_df[["accel_x", "accel_y", "accel_z"]].to_numpy(dtype=float)
    if len(xyz) < 2 or not np.isfinite(xyz).all():
        raise ValueError("Need at least two finite XYZ samples; do not silently drop missing data")
    index = pd.DatetimeIndex(pd.to_datetime(accel_df["sample_time"]))
    if index.hasnans:
        raise ValueError("sample_time contains missing timestamps")
    magnitude = np.linalg.norm(xyz, axis=1) / (1000.0 if input_unit == "mg" else 1.0)
    raw = pd.Series(magnitude, index=index, name="magnitude_g")
    duplicates = int(index.duplicated().sum())
    raw = raw.groupby(level=0, sort=True).mean()
    if len(raw) < 2:
        raise ValueError("Need at least two distinct timestamps")
    elapsed = (raw.index - raw.index[0]).total_seconds().to_numpy()
    gaps = np.diff(elapsed)
    if gaps.max() > max_gap_s:
        raise ValueError(
            f"Largest timestamp gap is {gaps.max():.3f} s (limit {max_gap_s:.3f} s). "
            "Split into continuous segments instead of filtering across missing data."
        )
    period = pd.to_timedelta(1.0 / sampling_rate, unit="s")
    if period.value <= 0:
        raise ValueError("sampling_rate exceeds timestamp resolution")
    grid = pd.date_range(raw.index[0], raw.index[-1], freq=period)
    regular_s = (grid - raw.index[0]).total_seconds().to_numpy()
    regular = pd.Series(np.interp(regular_s, elapsed, raw.to_numpy()), index=grid, name=raw.name)
    quality = {
        "input_samples": len(accel_df),
        "duplicate_timestamps_averaged": duplicates,
        "median_input_interval_ms": float(np.median(gaps) * 1000),
        "max_input_gap_s": float(gaps.max()),
        "input_unit": input_unit,
        "median_magnitude_g": float(raw.median()),
        "sampling_rate_hz": sampling_rate,
        "regular_samples": len(regular),
    }
    return regular, quality


def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """Return (minimum_indices, maximum_indices), grouped by extrema count.

    dmin/dmax are counts of extrema, NOT window lengths in samples or seconds.
    """
    s = np.asarray(s, dtype=float)
    if s.ndim != 1 or not np.isfinite(s).all():
        raise ValueError("s must be a finite one-dimensional signal")
    for size in (dmin, dmax):
        if not isinstance(size, (int, np.integer)) or isinstance(size, bool) or size < 1:
            raise ValueError("dmin and dmax must be positive integers")
    turning = np.diff(np.sign(np.diff(s)))
    lmin = np.flatnonzero(turning > 0) + 1
    lmax = np.flatnonzero(turning < 0) + 1
    if split and s.size:
        midpoint = s.mean()
        lmin = lmin[s[lmin] < midpoint]
        lmax = lmax[s[lmax] > midpoint]
    lmin = np.array([chunk[np.argmin(s[chunk])] for chunk in
                     (lmin[i:i + dmin] for i in range(0, len(lmin), dmin))], dtype=int)
    lmax = np.array([chunk[np.argmax(s[chunk])] for chunk in
                     (lmax[i:i + dmax] for i in range(0, len(lmax), dmax))], dtype=int)
    return lmin, lmax


def compute_envelope(acc, resample=True):
    """Upper-minus-lower envelope of an already filtered magnitude Series.

    Interpolate the two envelopes independently at matching sample positions;
    unequal numbers of extrema do not require dropping either envelope's ends.
    Non-oscillating signals without both extrema types return a zero envelope.
    With resample=False, evaluate both envelopes at the upper extrema only.
    Constant extension outside the outer extrema is retained from the original
    implementation, so recording-edge detections need particular scrutiny.
    """
    values = _finite_values(acc)
    lmin, lmax = hl_envelopes_idx(values, dmin=10, dmax=10)
    if not len(lmin) or not len(lmax):
        return pd.Series(0.0, index=acc.index, name="envelope_g")
    positions = np.arange(len(acc)) if resample else lmax
    upper = np.interp(positions, lmax, values[lmax])
    lower = np.interp(positions, lmin, values[lmin])
    return pd.Series(np.maximum(upper - lower, 0.0), index=acc.index[positions], name="envelope_g")


def _burst_ranges(score, threshold, record_start, record_end, merge_gap_s):
    # Padding detects runs touching either recording boundary without NaT edits.
    active = score.to_numpy() > threshold
    changes = np.diff(np.r_[False, active, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    merged = []
    for first, stop in zip(starts, stops):
        start = max(score.index[first], record_start)
        end = score.index[stop] if stop < len(score) else record_end
        end = min(end, record_end)
        if merged and (start - merged[-1][1]).total_seconds() < merge_gap_s:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return merged


def detect_bursts(acc, sampling_rate, envelope=True, resample_envelope=True, alfa=None,
                  *, merge_gap_s=5.0, return_signals=False):
    """Detect bursts from UNFILTERED acceleration magnitude in g.

    acc must be finite and regularly sampled with a unique, increasing
    DatetimeIndex. Use prepare_acceleration for jittered FIT timestamps. The
    filter is applied here exactly once: 0.1-10 Hz, order-8, zero-phase
    Butterworth (NeuroKit's SOS implementation). sampling_rate must exceed
    20 Hz and the input must contain more than 51 samples for filter padding.

    With envelope=True, alfa is an absolute threshold in g (wrist: 0.020).
    The legacy envelope=False branch uses alfa as a multiplier of the 10th
    percentile of 1-second standard deviations, NOT an absolute g threshold.
    Bursts use score > threshold and merge only gaps strictly < merge_gap_s.

    Returns a DataFrame with start, end, duration (Timedelta), peak-to-peak
    (filtered g), and AUC (score integrated in g*s, including merged gaps).
    Boundaries are [start, end): end is the first below-threshold timestamp,
    or one sample beyond the final input timestamp. AUC interpolates at these
    boundaries and extends the final score over the last sample interval.
    Timezones are preserved. Empty results have the same columns and dtypes.

    return_signals=True returns (bursts, diagnostics), where diagnostics holds
    filtered, score, and threshold for plotting without filtering twice.
    Edge transients and extrema-count smoothing remain methodological limits;
    these fixes do not independently validate the supplied wrist threshold.
    """
    values = _finite_values(acc)
    _positive_number(sampling_rate, "sampling_rate")
    _positive_number(alfa, "alfa")
    _positive_number(merge_gap_s, "merge_gap_s", allow_zero=True)
    if sampling_rate <= 20:
        raise ValueError("sampling_rate must exceed 20 Hz for the 10 Hz upper cutoff")
    if not isinstance(acc.index, pd.DatetimeIndex):
        raise TypeError("acc must have a DatetimeIndex")
    if acc.index.hasnans or not acc.index.is_unique or not acc.index.is_monotonic_increasing:
        raise ValueError("Timestamps must be present, unique, and increasing")
    if len(acc) <= 51:
        raise ValueError("Need more than 51 samples for order-8 band-pass filter padding")
    intervals = (acc.index[1:] - acc.index[:-1]).total_seconds().to_numpy()
    if not np.allclose(intervals, 1.0 / sampling_rate, rtol=0.01, atol=1e-9):
        raise ValueError("Timestamps do not match sampling_rate; regularize or split the signal first")

    filtered = pd.Series(nk.signal_filter(values, sampling_rate=sampling_rate,
                         lowcut=0.1, highcut=10, method="butterworth", order=8),
                         index=acc.index, name="filtered_g")
    if envelope:
        score = compute_envelope(filtered, resample=resample_envelope)
        threshold = float(alfa)
    else:
        score = filtered.resample("1s").std().rename("std_g")
        valid = score.dropna()
        if valid.empty:
            raise ValueError("Not enough samples in one-second bins for STD detection")
        threshold = float(np.percentile(valid, 10) * alfa)

    record_end = acc.index[-1] + pd.to_timedelta(1.0 / sampling_rate, unit="s")
    ranges = _burst_ranges(score, threshold, acc.index[0], record_end, merge_gap_s)
    rows = []
    for start, end in ranges:
        first, stop = filtered.index.searchsorted([start, end])
        peak_to_peak = float(np.ptp(filtered.iloc[first:stop].to_numpy()))
        # Use elapsed seconds, not sample count or epoch-sized floating timestamps.
        first, stop = score.index.searchsorted([start, end], side="right")
        inside = score.iloc[first:stop]
        inside = inside.loc[inside.index < end].dropna()
        boundary_positions = score.index.searchsorted([start, end])
        left = max(0, int(boundary_positions[0]) - 1)
        right = min(len(score), int(boundary_positions[1]) + 1)
        nearby = score.iloc[left:right].dropna()
        boundary_values = np.interp([0.0, (end - start).total_seconds()],
                                   (nearby.index - start).total_seconds(), nearby.to_numpy())
        times = np.r_[0.0, (inside.index - start).total_seconds(), (end - start).total_seconds()]
        heights = np.r_[boundary_values[0], inside.to_numpy(), boundary_values[1]]
        rows.append((start, end, end - start, peak_to_peak, float(trapezoid(heights, x=times))))
    bursts = pd.DataFrame(rows, columns=["start", "end", "duration", "peak-to-peak", "AUC"])
    bursts = bursts.astype({"start": acc.index.dtype, "end": acc.index.dtype,
                            "duration": "timedelta64[ns]", "peak-to-peak": float, "AUC": float})
    bursts.attrs.update({"threshold": threshold, "score_unit": "g", "AUC_unit": "g*s",
                         "sampling_rate_hz": sampling_rate, "merge_gap_s": merge_gap_s})
    if return_signals:
        return bursts, {"filtered": filtered, "score": score, "threshold": threshold}
    return bursts


def detect_bursts_in_intervals(accel_df, intervals, *, sampling_rate=100.0,
                               input_unit="mg", max_gap_s=0.25, alfa=0.040,
                               merge_gap_s=5.0):
    """Regularize/filter/merge independently within each selected UTC interval.

    Input sample_time must be timezone-aware. Short intervals that cannot support
    the band-pass filter are reported and omitted from returned analysis bounds.
    Returned signal parts stay separate, including for plotting. Global burst
    IDs and analysis_interval_id connect bursts to the actual processed bounds.
    """
    intervals = validate_intervals(intervals)
    times = pd.DatetimeIndex(accel_df.sample_time)
    if times.hasnans or times.tz is None:
        raise ValueError("sample_time must contain valid timezone-aware timestamps")
    times = times.tz_convert("UTC")
    source = accel_df.copy()
    source["sample_time"] = times
    source = source.sort_values("sample_time")
    times = pd.DatetimeIndex(source.sample_time)
    _positive_number(sampling_rate, "sampling_rate")
    period = pd.Timedelta(seconds=1 / sampling_rate)
    parts, signals, checks, used = [], [], [], []
    for requested_id, interval in intervals.iterrows():
        first, stop = times.searchsorted([interval.start, interval.end])
        frame = source.iloc[first:stop]
        if frame.sample_time.nunique() < 2:
            checks.append({"requested_interval_id": requested_id, "status": "too_few_samples"})
            continue
        magnitude, quality = prepare_acceleration(frame, sampling_rate, input_unit, max_gap_s)
        magnitude = magnitude.loc[magnitude.index + period <= interval.end]
        if len(magnitude) <= 51:
            checks.append({**quality, "requested_interval_id": requested_id, "status": "too_short_for_filter"})
            continue
        if not 0.5 <= quality["median_magnitude_g"] <= 1.5:
            raise ValueError("Median magnitude is not near 1 g. Check input_unit and calibration.")
        interval_id = len(used)
        bursts, diagnostic = detect_bursts(magnitude, sampling_rate, alfa=alfa,
                                           merge_gap_s=merge_gap_s, return_signals=True)
        bursts["analysis_interval_id"] = interval_id
        parts.append(bursts)
        used.append((magnitude.index[0], magnitude.index[-1] + period))
        signals.append({**diagnostic, "magnitude": magnitude, "analysis_interval_id": interval_id})
        checks.append({**quality, "requested_interval_id": requested_id,
                       "analysis_interval_id": interval_id, "status": "processed"})
    if parts:
        bursts = pd.concat(parts, ignore_index=True)
    else:
        bursts = pd.DataFrame({"start": pd.Series(dtype="datetime64[ns, UTC]"),
                               "end": pd.Series(dtype="datetime64[ns, UTC]"),
                               "duration": pd.Series(dtype="timedelta64[ns]"),
                               "peak-to-peak": pd.Series(dtype=float), "AUC": pd.Series(dtype=float),
                               "analysis_interval_id": pd.Series(dtype=int)})
    return {"bursts": bursts, "signals": signals, "quality": pd.DataFrame(checks),
            "intervals": validate_intervals(pd.DataFrame(used, columns=["start", "end"]))}
