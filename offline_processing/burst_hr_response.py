"""Movement-locked HR analysis from BBIs or FIT-recorded heart rate.

Reference: https://www.nature.com/articles/s41598-025-29723-7
Uses an adapted -19..+54 s epoch, a 14-s baseline [-19, -5), and
post-onset HR maximum. Callback/record-timed HR is NOT ECG beat-timed HR. AUC
tertiles, plausibility screening, and dropout guards are local adaptations.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

RELATIVE_SECONDS = np.arange(-19, 55)
BASELINE = (RELATIVE_SECONDS >= -19) & (RELATIVE_SECONDS < -5)
POST = RELATIVE_SECONDS > 0
TERTILES = ["Low", "Medium", "High"]


def _utc(values):
    """FIT naive timestamps denote UTC, not the computer's local timezone."""
    times = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    if times.hasnans:
        raise ValueError("Missing timestamps are not allowed")
    return times


def record_times_to_utc(values, naive_timezone="UTC"):
    """Normalize record/local display times without relabeling local time as UTC.

    Raw FIT times are UTC by default. Callers that previously converted to local
    wall time and removed the timezone must explicitly supply that timezone.
    Ambiguous/nonexistent DST times are rejected instead of silently guessed.
    """
    times = pd.DatetimeIndex(pd.to_datetime(values))
    if times.hasnans:
        raise ValueError("Missing record timestamps are not allowed")
    if times.tz is None:
        times = times.tz_localize(naive_timezone, ambiguous="raise", nonexistent="raise")
    return times.tz_convert("UTC")


def record_hr_to_observations(records, naive_timezone="UTC", hr_limits_bpm=(30.0, 200.0)):
    """Adapt FIT-recorded HR, not BBI-derived HR, to the event-analysis input.

    Accept the notebook's bbi_df with timestamp/heart_rate columns, or its
    timestamp-indexed, hr_bpm-renamed variant. Preserve the caller's table.
    Duplicate timestamps are averaged, but any invalid HR at that timestamp
    makes it unusable. Missing/invalid rows remain interpolation barriers.
    Recorded HR does not establish the sensor source or upstream processing.
    """
    low, high = hr_limits_bpm
    if not np.isfinite([low, high]).all() or not 0 < low < high:
        raise ValueError("HR limits must be finite, positive, and increasing")
    columns = [name for name in ("heart_rate", "hr_bpm") if name in records.columns]
    if len(columns) != 1:
        raise ValueError("Provide exactly one recorded HR column: heart_rate or hr_bpm")
    if "timestamp" in records.columns:
        times = records["timestamp"]
    elif isinstance(records.index, pd.DatetimeIndex):
        times = records.index
    else:
        raise ValueError("Recorded HR needs a timestamp column or DatetimeIndex")
    index = record_times_to_utc(times, naive_timezone=naive_timezone)
    values = pd.to_numeric(records[columns[0]], errors="raise").to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(values) & (values >= low) & (values <= high)
    samples = pd.DataFrame({"hr_bpm": np.where(valid, values, np.nan), "invalid_hr": ~valid}, index=index)
    observations = samples.groupby(level=0, sort=True).agg(
        hr_bpm=("hr_bpm", "mean"), n_records=("hr_bpm", "size"), invalid_hr=("invalid_hr", "sum"),
    )
    observations.loc[observations.invalid_hr > 0, "hr_bpm"] = np.nan
    observations["break_before"] = False
    observations.index.name = "record_time_utc"
    gaps = observations.index.to_series().diff().dt.total_seconds()
    report = {
        "source": "FIT-recorded HR; not reconstructed from BBIs", "value_column": columns[0],
        "input_rows": len(records), "invalid_or_out_of_range_hr": int((~valid).sum()),
        "unique_timestamps": len(observations), "duplicate_timestamps_averaged": int(index.duplicated().sum()),
        "valid_timestamps": int(observations.hr_bpm.notna().sum()),
        "largest_record_gap_s": float(gaps.max()) if len(gaps) > 1 else None,
        "hr_limits_bpm": [low, high], "naive_timestamp_timezone": naive_timezone,
        "timing": "FIT record timestamps; not exact beat timestamps",
    }
    return observations, report


def bbi_to_hr(rows, bbi_limits_ms=(300.0, 2000.0)):
    """Decode EVERY recovered BBI into HR, then average HR within callbacks.

    Input: rows from fit_bbi.decode_snapshots/load_bbi. Equal-valued intervals
    with distinct sequence numbers are retained. A callback containing an
    invalid/out-of-range BBI is marked unusable as a whole. Sequence holes are
    interpolation barriers. Bounds are configurable QC, not paper criteria.
    No native one-second heart_rate field or cumulative beat-time reconstruction
    is used. The callback UTC clock has a coarse anchor and unknown beat delay.
    """
    low, high = bbi_limits_ms
    if not np.isfinite([low, high]).all() or not 0 < low < high:
        raise ValueError("BBI limits must be finite, positive, and increasing")
    raw = pd.DataFrame(rows)
    if raw.empty:
        observations = pd.DataFrame(columns=["hr_bpm", "n_bbi", "first_sequence",
                                             "last_sequence", "invalid_bbi", "break_before"],
                                    index=pd.DatetimeIndex([], tz="UTC", name="callback_time"))
        return observations, {"received_bbi": 0, "invalid_or_out_of_range_bbi": 0,
                              "valid_callbacks": 0, "timing": "callback arrival proxy"}
    sequences = raw["sequence"].to_numpy(dtype=float)
    if (not np.isfinite(sequences).all() or (sequences < 1).any()
            or (sequences != np.floor(sequences)).any() or (np.diff(sequences) <= 0).any()):
        raise ValueError("BBI sequence numbers must be positive, unique, and increasing")
    times = _utc(raw["callback_time_utc_approx"])
    if not times.is_monotonic_increasing:
        raise ValueError("Callback times must increase with BBI sequence")
    intervals = raw["bbi_ms"].to_numpy(dtype=float)
    valid = np.isfinite(intervals) & (intervals >= low) & (intervals <= high)
    hr = np.full(len(intervals), np.nan)
    hr[valid] = 60000.0 / intervals[valid]
    samples = pd.DataFrame({"hr_bpm": hr, "sequence": sequences.astype(np.int64),
                            "invalid_bbi": ~valid,
                            "sequence_hole": np.r_[False, np.diff(sequences) != 1]}, index=times)
    observations = samples.groupby(level=0, sort=True).agg(
        hr_bpm=("hr_bpm", "mean"), n_bbi=("sequence", "size"),
        first_sequence=("sequence", "first"), last_sequence=("sequence", "last"),
        invalid_bbi=("invalid_bbi", "sum"), break_before=("sequence_hole", "any"),
    )
    # A missing sequence inside a batch also prevents using that batch's mean.
    internal_hole = (observations.last_sequence - observations.first_sequence + 1) != observations.n_bbi
    observations.loc[(observations.invalid_bbi > 0) | internal_hole, "hr_bpm"] = np.nan
    observations.index.name = "callback_time"
    gaps = observations.index.to_series().diff().dt.total_seconds()
    report = {"received_bbi": len(raw), "invalid_or_out_of_range_bbi": int((~valid).sum()),
              "callbacks": len(observations), "valid_callbacks": int(observations.hr_bpm.notna().sum()),
              "sequence_holes": int(samples.sequence_hole.sum()),
              "largest_callback_gap_s": float(gaps.max()) if len(gaps) > 1 else None,
              "bbi_limits_ms": [low, high], "timing": "callback arrival proxy"}
    return observations, report


def _artifact_intervals(artifacts):
    intervals = []
    for start, end in artifacts:
        start, end = _utc([start, end])
        if end <= start:
            raise ValueError("Artifact end must follow start")
        intervals.append((start, end))
    intervals.sort()
    if any(b[0] < a[1] for a, b in zip(intervals, intervals[1:])):
        raise ValueError("Merge overlapping artifact annotations first")
    return intervals


def sample_hr(observations, query_times, max_gap_s=3.0, artifacts=()):
    """Linear HR interpolation with no extrapolation or silent long-gap fill.

    max_gap_s limits unannotated observation gaps. Manually annotated artifacts
    shorter than 10 s can be repaired with a cubic spline using two clean
    observations on each side; >=10 s artifacts and unsupported repairs remain
    missing. Repairs cannot cross lost sequence numbers or long acquisition
    gaps. Returns HR, bracketing gap seconds, and cubic-repair flags.
    """
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be positive and finite")
    query = _utc(query_times)
    index = _utc(observations.index)
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("HR observations must have unique, increasing timestamps")
    hr = np.full(len(query), np.nan)
    gap = np.full(len(query), np.nan)
    repaired = np.zeros(len(query), dtype=bool)
    annotations = _artifact_intervals(artifacts)
    if len(index) == 0:
        return hr, gap, repaired
    origin = index[0]
    x = (index - origin).total_seconds().to_numpy()
    q = (query - origin).total_seconds().to_numpy()
    y = observations.hr_bpm.to_numpy(dtype=float).copy()
    if (np.isfinite(y) & (y <= 0)).any():
        raise ValueError("Finite HR observations must be positive")
    broken = observations.break_before.to_numpy(dtype=bool)
    intervals = [((start - origin).total_seconds(), (end - origin).total_seconds())
                 for start, end in annotations]
    for start, end in intervals:
        y[(x >= start) & (x < end)] = np.nan
    right = np.searchsorted(x, q, side="left")
    exact = (right < len(x)) & (x[np.minimum(right, len(x) - 1)] == q)
    hr[exact] = y[right[exact]]
    gap[exact] = 0.0
    bracketed = (~exact) & (right > 0) & (right < len(x))
    positions = np.flatnonzero(bracketed)
    r = right[positions]
    l = r - 1
    width = x[r] - x[l]
    gap[positions] = width
    allowed = (width <= max_gap_s + 1e-9) & np.isfinite(y[l]) & np.isfinite(y[r]) & ~broken[r]
    for start, end in intervals:
        allowed &= ~((x[l] < end) & (x[r] >= start))
    p, l, r = positions[allowed], l[allowed], r[allowed]
    hr[p] = y[l] + (y[r] - y[l]) * (q[p] - x[l]) / (x[r] - x[l])

    for start, end in intervals:
        if end - start >= 10.0 - 1e-9:
            continue
        left = np.flatnonzero((x < start) & np.isfinite(y))[-2:]
        right = np.flatnonzero((x >= end) & np.isfinite(y))[:2]
        if len(left) < 2 or len(right) < 2:
            continue
        anchors = np.r_[left, right]
        a, b = left[-1], right[0]
        if (start - x[a] > max_gap_s or x[b] - end > max_gap_s
                or np.any(broken[left[0] + 1:right[-1] + 1])
                or np.any(np.diff(x[left[0]:right[-1] + 1]) > max_gap_s + 1e-9)
                or x[left[1]] - x[left[0]] > max_gap_s
                or x[right[1]] - x[right[0]] > max_gap_s):
            continue
        # Do not let a repair extend into a separate annotation.
        if any((s, e) != (start, end) and x[a] < e and x[b] >= s for s, e in intervals):
            continue
        selected = (q > x[a]) & (q < x[b])
        values = CubicSpline(x[anchors], y[anchors], extrapolate=False)(q[selected])
        values[~np.isfinite(values) | (values <= 0)] = np.nan
        hr[selected] = values
        repaired[selected] = np.isfinite(values)
    return hr, gap, repaired


def assign_auc_tertiles(auc):
    """Cut all candidate bursts at 1/3 and 2/3 quantiles, before HR exclusions.

    Ties stay together: <=q1 Low, q1<value<=q2 Medium, >q2 High. Equal cutoffs
    may leave groups empty; observations are never split by arbitrary rank.
    """
    values = np.asarray(auc, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Burst AUC must be finite and non-negative (g*s)")
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3]) if len(values) else (np.nan, np.nan)
    groups = np.where(values <= q1, "Low", np.where(values <= q2, "Medium", "High"))
    return pd.Categorical(groups, categories=TERTILES, ordered=True), (float(q1), float(q2))


def analyze_burst_hr(bursts, observations, recording_start, recording_end,
                     max_gap_s=3.0, artifacts=(), isolation_s=30.0, exclude_late_overlap=False):
    """Return per-burst metrics, epochs, AUC-group summaries, curves, and QC.

    Isolation uses clear offset-to-onset gaps on both sides of the target
    movement (>=30 s), considering ALL candidates, including HR-invalid ones.
    All epochs are retained; only complete, isolated epochs enter summaries.
    Features for non-isolated but HR-complete epochs are descriptive only.
    exclude_late_overlap optionally applies a stricter full-epoch movement guard.
    SEM is across events in ONE recording, not across independent subjects.
    """
    if not np.isfinite(isolation_s) or isolation_s < 0:
        raise ValueError("isolation_s must be non-negative and finite")
    events = bursts.copy().reset_index(drop=True)
    events.index.name = "burst_id"
    events["start"] = _utc(events["start"])
    events["end"] = _utc(events["end"])
    if (events.end <= events.start).any():
        raise ValueError("Every burst must end after its start")
    events["auc_tertile"], cutoffs = assign_auc_tertiles(events.AUC)
    begin, finish = _utc([recording_start, recording_end])
    if finish <= begin:
        raise ValueError("Recording end must follow start")
    events = events.sort_values("start", kind="stable")
    previous_end = events.end.cummax().shift(1)
    next_start = events.start.shift(-1)
    before = (events.start - previous_end).dt.total_seconds()
    after = (next_start - events.end).dt.total_seconds()
    events["gap_before_s"] = before
    events["gap_after_s"] = after
    events["isolated"] = (before.isna() | (before >= isolation_s)) & (after.isna() | (after >= isolation_s))
    events["isolation_observed"] = ((events.start - pd.Timedelta(seconds=isolation_s) >= begin)
                                    & (events.end + pd.Timedelta(seconds=isolation_s) <= finish))
    events["other_movement_in_epoch"] = next_start.notna() & (next_start <= events.start + pd.Timedelta(seconds=int(RELATIVE_SECONDS[-1])))
    events = events.sort_index()

    epochs_bpm = pd.DataFrame(np.nan, index=events.index, columns=RELATIVE_SECONDS)
    epochs_pct = epochs_bpm.copy()
    metrics = []
    annotations = list(artifacts)
    for burst_id, event in events.iterrows():
        times = event.start + pd.to_timedelta(RELATIVE_SECONDS, unit="s")
        hr, gap, repaired = sample_hr(observations, times, max_gap_s=max_gap_s, artifacts=annotations)
        within_recording = times[0] >= begin and times[-1] <= finish
        baseline_ok = np.isfinite(hr[BASELINE]).all()
        baseline = float(hr[BASELINE].mean()) if baseline_ok else np.nan
        complete = bool(np.isfinite(hr).all() and np.isfinite(baseline) and baseline > 0 and within_recording)
        reasons = []
        if not within_recording:
            reasons.append("recording_edge")
        if not event.isolated or not event.isolation_observed:
            reasons.append("not_isolated_30s" if isolation_s == 30 else "not_isolated")
        if exclude_late_overlap and event.other_movement_in_epoch:
            reasons.append("other_movement_in_hr_epoch")
        if not np.isfinite(hr).all():
            reasons.append("insufficient_bbi_or_artifact")
        epochs_bpm.loc[burst_id] = hr
        row = {"hr_complete": complete, "included": complete and not reasons,
               "exclusion_reason": "; ".join(reasons), "baseline_bpm": baseline,
               "hr_coverage_pct": float(np.isfinite(hr).mean() * 100),
               "baseline_coverage_pct": float(np.isfinite(hr[BASELINE]).mean() * 100),
               "max_hr_bracket_gap_s": float(np.nanmax(gap)) if np.isfinite(gap).any() else np.nan,
               "cubic_repaired_samples": int(repaired.sum()),
               "hr_post_max_pct": np.nan, "hr_peak_increase_pct": np.nan,
               "hr_peak_latency_s": np.nan, "hr_post_min_pct": np.nan,
               "hr_trough_latency_s": np.nan, "hr_post_mean_pct": np.nan}
        if complete:
            normalized = (hr / baseline - 1) * 100
            epochs_pct.loc[burst_id] = normalized
            post = normalized[POST]
            row.update({"hr_post_max_pct": float(post.max()),
                        "hr_peak_increase_pct": float(max(0, post.max())),
                        "hr_peak_latency_s": float(RELATIVE_SECONDS[POST][post.argmax()]) if post.max() > 0 else np.nan,
                        "hr_post_min_pct": float(post.min()),
                        "hr_trough_latency_s": float(RELATIVE_SECONDS[POST][post.argmin()]),
                        "hr_post_mean_pct": float(post.mean())})
        metrics.append(row)
    metric_types = {
        "hr_complete": bool, "included": bool, "exclusion_reason": str,
        "baseline_bpm": float, "hr_coverage_pct": float, "baseline_coverage_pct": float,
        "max_hr_bracket_gap_s": float, "cubic_repaired_samples": int,
        "hr_post_max_pct": float, "hr_peak_increase_pct": float, "hr_peak_latency_s": float,
        "hr_post_min_pct": float, "hr_trough_latency_s": float, "hr_post_mean_pct": float,
    }
    events = events.join(pd.DataFrame(metrics, index=events.index, columns=metric_types).astype(metric_types))
    summaries, curves = [], []
    for group in TERTILES:
        subset = events.loc[events.auc_tertile == group]
        selected = subset.loc[subset.included]
        epochs = epochs_pct.loc[selected.index]
        summaries.append({"auc_tertile": group, "total_bursts": len(subset),
                          "hr_complete_bursts": int(subset.hr_complete.sum()),
                          "included_bursts": len(selected), "excluded_bursts": len(subset) - len(selected),
                          "auc_median_g_s": float(subset.AUC.median()),
                          "hr_peak_mean_pct": float(selected.hr_peak_increase_pct.mean()),
                          "hr_peak_median_pct": float(selected.hr_peak_increase_pct.dropna().median()),
                          "peak_latency_median_s": float(selected.hr_peak_latency_s.dropna().median())})
        for t in RELATIVE_SECONDS:
            values = epochs[t]
            curves.append({"auc_tertile": group, "relative_s": int(t), "n": len(values),
                           "mean_pct": float(values.mean()), "sem_pct": float(values.sem())})
    report = {"auc_tertile_cutoffs_g_s": list(cutoffs), "tertile_population": "all candidate bursts before HR exclusions",
              "total_bursts": len(events), "included_bursts": int(events.included.sum()),
              "hr_complete_bursts": int(events.hr_complete.sum()), "max_unannotated_gap_s": max_gap_s,
              "annotated_artifacts": len(annotations), "isolation_s": isolation_s,
              "exclude_late_overlap": exclude_late_overlap,
              "included_with_other_movement_in_epoch": int(events.loc[events.included, "other_movement_in_epoch"].sum()),
              "baseline_seconds": "[-19, -5): fourteen 1-Hz samples", "response_seconds": "-19 through +54 inclusive",
              "notes": ["HR uses callback arrival times, not exact beat timestamps; latency is approximate.",
                        "No raw ECG/PPG morphology is available for artifact validation.",
                        "No artifact annotations means unreviewed, not artifact-free.",
                        "AUC groups and extra QC are adaptations, not the paper's movement magnitude metric.",
                        "Within-night event SEM is descriptive, not between-subject uncertainty.",
                        "The 30-s isolation rule can leave another movement within the HR epoch; inspect other_movement_in_epoch."]}
    return {"events": events, "epochs_bpm": epochs_bpm, "epochs_pct": epochs_pct,
            "summary": pd.DataFrame(summaries).set_index("auc_tertile"),
            "curves": pd.DataFrame(curves), "report": report}
