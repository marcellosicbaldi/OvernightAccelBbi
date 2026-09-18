"""Exploratory RMSSD and PIP from numbered BBIs, without resampling or gap repair.

Window assignment uses callback ARRIVAL times, not exact beat times. Sequence
numbers describe delivered intervals, not necessarily all physiological beats.
Bounds screening cannot establish normal-to-normal (NN) intervals or source.

This is the previous strict, uncorrected HRV implementation, retained for
reference and regression checks. The main notebook now uses gp_hrv.py, which
shares this module's input validation, quiet-segment helper, and PIP formula.
"""

import numpy as np
import pandas as pd


def _inflection_mask(differences, adjacent_pairs):
    # Two consecutive differences require three contiguous valid intervals.
    triplets = adjacent_pairs[:-1] & adjacent_pairs[1:]
    return triplets & (differences[:-1] * differences[1:] <= 1e-10)


def compute_HRF(ppi):
    """Return PIP as a fraction, matching the user's formula on complete data.

    ppi is a 1D, contiguous sequence of positive finite BBIs in milliseconds.
    The denominator is N intervals, not N-2 evaluable triplets. Products of
    successive differences <= 1e-10 count, including zero*zero. Consequently
    a flat series has PIP=(N-2)/N. No detrending or interpolation is performed.
    Fewer than three intervals return NaN (no evaluable inflection point).
    """
    ppi = np.asarray(ppi, dtype=float)
    if ppi.ndim != 1 or not np.isfinite(ppi).all() or (ppi <= 0).any():
        raise ValueError("PPI must be a 1D sequence of positive finite intervals in ms")
    if len(ppi) < 3:
        return np.nan
    differences = np.diff(ppi)
    points = _inflection_mask(differences, np.ones(len(differences), dtype=bool))
    return float(points.sum() / len(ppi))


def _utc_bound(value):
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is None:
        raise ValueError("Time bounds must be timezone-aware")
    return value.tz_convert("UTC")


def burst_free_segments(bursts, recording_start, recording_end, guard_s=0.0):
    """Complement of ALL bursts in [start, end), including quiet recording edges.

    Touching/overlapping bursts are merged. guard_s optionally excludes an
    additional margin on both sides; no physiological recovery period is assumed.
    """
    start, end = _utc_bound(recording_start), _utc_bound(recording_end)
    if end <= start or not np.isfinite(guard_s) or guard_s < 0:
        raise ValueError("Need increasing bounds and a finite, non-negative burst guard")
    if not {"start", "end"}.issubset(bursts.columns):
        raise ValueError("Bursts need start/end columns, even when empty")
    guard = pd.Timedelta(seconds=guard_s)
    blocked = []
    for a, b in bursts[["start", "end"]].itertuples(index=False, name=None):
        a, b = _utc_bound(a), _utc_bound(b)
        if b <= a:
            raise ValueError("Each burst must end after it starts")
        a, b = max(start, a - guard), min(end, b + guard)
        if a < b:
            blocked.append((a, b))
    cursor, gaps = start, []
    for a, b in sorted(blocked):
        if a > cursor:
            gaps.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < end:
        gaps.append((cursor, end))
    result = pd.DataFrame(gaps, columns=["start", "end"])
    for column in ("start", "end"):
        result[column] = pd.to_datetime(result[column], utc=True)
    result["duration_s"] = (result.end - result.start).dt.total_seconds()
    result.index.name = "segment_id"
    return result


def prepare_bbi_intervals(rows, bbi_limits_ms=(300.0, 2000.0)):
    """Preserve every delivered interval, including invalid values and tied times."""
    low, high = bbi_limits_ms
    if not (np.isfinite(low) and np.isfinite(high) and 0 < low < high):
        raise ValueError("BBI limits must be finite, positive and increasing")
    columns = ["sequence", "bbi_ms", "callback_time_utc_approx"]
    data = pd.DataFrame(rows).copy()
    if data.empty:
        data = pd.DataFrame(columns=columns)
    if not set(columns).issubset(data.columns):
        raise ValueError("Need decoded numbered BBIs, not hr_observations or record-level HR")
    data = data[columns]
    sequence = pd.to_numeric(data.sequence, errors="coerce").to_numpy(dtype=float)
    if (not np.isfinite(sequence).all() or (sequence < 1).any()
            or (sequence != np.floor(sequence)).any() or pd.Series(sequence).duplicated().any()):
        raise ValueError("BBI sequence numbers must be unique positive integers; decode snapshots first")
    data["sequence"] = sequence.astype(np.int64)
    data = data.sort_values("sequence").reset_index(drop=True)
    data["callback_time_utc_approx"] = pd.to_datetime(data.callback_time_utc_approx, utc=True, format="ISO8601")
    if data.callback_time_utc_approx.isna().any() or not data.callback_time_utc_approx.is_monotonic_increasing:
        raise ValueError("BBI arrival times must be present and nondecreasing in sequence order")
    data["bbi_ms"] = pd.to_numeric(data.bbi_ms, errors="coerce").astype(float)
    data["valid_bbi"] = np.isfinite(data.bbi_ms) & data.bbi_ms.between(low, high)
    return data


def interburst_rmssd(rows, bursts, recording_start, recording_end, *,
                    window_s=300.0, step_s=60.0, guard_s=0.0,
                    bbi_limits_ms=(300.0, 2000.0), max_callback_gap_s=3.0,
                    min_pairs=100, duration_fraction_limits=(0.9, 1.1)):
    """Return interval, segment and window tables, with strict exploratory QC.

    Windows restart at each quiet segment's beginning and are end-exclusive.
    Successive differences NEVER cross invalid values, missing sequence numbers
    or callback gaps longer than max_callback_gap_s. Same-callback intervals are
    kept individually. Nothing is interpolated, averaged by callback or repaired.

    rmssd_ms, pip and pip_pct are supplied only for windows passing all QC: sufficient pairs, no
    invalid intervals/sequence holes, bounded arrival gaps including edges, and
    plausible sum(BBI)/window duration. The latter is only a consistency check,
    not proven beat coverage. rmssd_available_pairs_ms is a diagnostic even for
    rejected windows, NOT a substitute for a complete 5-minute HRV estimate.
    These configurable thresholds are not validated artifact/ectopy detection.

    PIP uses the supplied <= 1e-10 product rule and divides the inflection count
    by the total interval count N, as in compute_HRF. pip is a fraction; pip_pct
    is 100*pip. PIP additionally needs an evaluable triplet, otherwise it is NaN.
    n_valid_triplets and n_inflection_points are diagnostic counts even when a
    window is rejected; neither joins differences across barriers. The public
    function name is retained for existing RMSSD callers.
    """
    if not all(np.isfinite(x) and x > 0 for x in (window_s, step_s, max_callback_gap_s)):
        raise ValueError("Window, step and callback gap must be finite and positive")
    if step_s > window_s or not isinstance(min_pairs, (int, np.integer)) or min_pairs < 1:
        raise ValueError("Step cannot exceed window; min_pairs must be a positive integer")
    frac_low, frac_high = duration_fraction_limits
    if not (np.isfinite(frac_low) and np.isfinite(frac_high) and 0 <= frac_low < frac_high):
        raise ValueError("Duration-fraction limits must be finite, non-negative and increasing")
    intervals = prepare_bbi_intervals(rows, bbi_limits_ms)
    segments = burst_free_segments(bursts, recording_start, recording_end, guard_s)
    times = pd.DatetimeIndex(intervals.callback_time_utc_approx)
    bbi = intervals.bbi_ms.to_numpy()
    valid = intervals.valid_bbi.to_numpy()
    sequence = intervals.sequence.to_numpy()
    window, step = pd.Timedelta(seconds=window_s), pd.Timedelta(seconds=step_s)
    if min(window.value, step.value) < 1:
        raise ValueError("Window and step must be at least one nanosecond")
    results = []
    for segment_id, segment in segments.iterrows():
        a = segment.start
        while a + window <= segment.end:
            b = a + window
            left, right = times.searchsorted([a, b], side="left")
            values, good = bbi[left:right], valid[left:right]
            seq_steps = np.diff(sequence[left:right])
            arrival_steps = np.diff(times[left:right].asi8) / 1e9
            pairs = good[:-1] & good[1:] & (seq_steps == 1) & (arrival_steps <= max_callback_gap_s)
            all_differences = np.diff(np.where(good, values, np.nan))
            differences = all_differences[pairs]
            rmssd = float(np.sqrt(np.mean(differences ** 2))) if len(differences) else np.nan
            n_triplets = int((pairs[:-1] & pairs[1:]).sum())
            n_inflection = int(_inflection_mask(all_differences, pairs).sum())
            pip = float(n_inflection / len(values)) if n_triplets else np.nan
            n_invalid = int((~good).sum())
            missing = int(np.maximum(seq_steps - 1, 0).sum())
            # Edge silence matters too: a short dense batch is not a 5-minute window.
            gaps = np.diff(np.r_[a.value, times[left:right].asi8, b.value]) / 1e9
            max_gap = float(gaps.max())
            duration_fraction = float(values[good].sum() / (1000 * window_s))
            reasons = []
            if len(differences) < min_pairs:
                reasons.append("too_few_adjacent_pairs")
            if n_invalid:
                reasons.append("invalid_bbi")
            if missing:
                reasons.append("missing_sequences")
            if max_gap > max_callback_gap_s:
                reasons.append("callback_gap_or_edge_silence")
            if not frac_low <= duration_fraction <= frac_high:
                reasons.append("bbi_duration_inconsistent_with_window")
            results.append({
                "segment_id": segment_id, "start": a, "end": b, "center": a + window / 2,
                "n_intervals": len(values), "n_valid_intervals": int(good.sum()),
                "n_adjacent_pairs": len(differences), "n_invalid_intervals": n_invalid,
                "missing_sequences": missing, "max_callback_gap_s": max_gap,
                "bbi_duration_fraction": duration_fraction,
                "rmssd_available_pairs_ms": rmssd, "rmssd_ms": rmssd if not reasons else np.nan,
                "n_valid_triplets": n_triplets, "n_inflection_points": n_inflection,
                "pip": pip if not reasons else np.nan,
                "pip_pct": 100 * pip if not reasons else np.nan,
                "included": not reasons, "exclusion_reason": ";".join(reasons),
            })
            a += step
    columns = ["segment_id", "start", "end", "center", "n_intervals", "n_valid_intervals",
               "n_adjacent_pairs", "n_invalid_intervals", "missing_sequences", "max_callback_gap_s",
               "bbi_duration_fraction", "rmssd_available_pairs_ms", "rmssd_ms",
               "n_valid_triplets", "n_inflection_points", "pip", "pip_pct", "included", "exclusion_reason"]
    windows = pd.DataFrame(results, columns=columns)
    for column in ("start", "end", "center"):
        windows[column] = pd.to_datetime(windows[column], utc=True)
    windows["included"] = windows.included.astype(bool)
    windows.index.name = "window_id"
    return {"intervals": intervals, "segments": segments, "windows": windows}
