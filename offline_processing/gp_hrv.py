"""GP_pipeline quiet-period HRV adapted to numbered Garmin BBIs.

Keep the GP variable windows, single-pass artifact classification, linear
interval cleaning, and time-domain metrics. Garmin sequence/callback gaps are
hard boundaries for cleaning; callback timestamps remain approximate arrival
times. No raw PPG is available and no synthetic first interval is needed.
"""

import numpy as np
import pandas as pd

from gp_hrv_artifacts import _find_artifacts
from interburst_hrv import burst_free_segments, compute_HRF, prepare_bbi_intervals


ARTIFACT_CLASSES = ("ectopic", "missed", "extra", "longshort")


def build_variable_windows(qstart, qend, min_window, max_window, step):
    """GP window geometry, including the final window anchored to segment end.

    Segments shorter than min_window are skipped. Up to max_window, use the
    entire segment. Longer segments use max_window with a fixed step and an
    extra final window if needed. Intervals are assigned with [start, end).
    """
    duration = qend - qstart
    if duration < min_window:
        return []
    if duration <= max_window:
        return [(qstart, qend)]
    starts = []
    current, last = qstart, qend - max_window
    while current <= last:
        starts.append(current)
        current += step
    if starts[-1] < last:
        starts.append(last)
    return [(start, start + max_window) for start in starts]


def _runs(mask):
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))


def clean_interval_run(values_ms, bbi_limits_ms=(300.0, 2000.0)):
    """Classify once, mask artifacts/range failures, then interpolate by order.

    This is GP's interval cleaning, including endpoint filling. The caller must
    split delivery gaps BEFORE calling. Classifier runs never cross nonfinite
    or nonpositive raw values. No artifact correction of peak positions is used.
    """
    values = np.asarray(values_ms, dtype=float)
    flags = {name: np.zeros(len(values), dtype=bool) for name in ARTIFACT_CLASSES}
    finite_positive = np.isfinite(values) & (values > 0)
    for first, stop in _runs(finite_positive):
        if stop - first < 3:
            continue
        # GP passes IBIs in seconds; sampling_rate is unused in this classifier.
        artifacts, _ = _find_artifacts(values[first:stop].copy() / 1000.0)
        for name in ARTIFACT_CLASSES:
            indices = np.asarray(artifacts[name], dtype=int)
            flags[name][first + indices] = True
    artifact = np.logical_or.reduce(list(flags.values()))
    low, high = bbi_limits_ms
    invalid = ~finite_positive | (values < low) | (values > high)
    masked = values.copy()
    masked[artifact | invalid] = np.nan
    cleaned = pd.Series(masked).interpolate(method="linear", limit_direction="both").to_numpy()
    return {**flags, "artifact": artifact, "invalid": invalid,
            "cleaned": cleaned, "interpolated": ~np.isfinite(masked) & np.isfinite(cleaned)}


def interburst_hrv(rows, bursts, recording_start, recording_end, *,
                  min_window_s=60.0, max_window_s=300.0, step_s=60.0,
                  min_burst_duration_s=2.0, post_burst_guard_s=1.0,
                  min_beats=30, bbi_limits_ms=(300.0, 2000.0),
                  max_callback_gap_s=3.0, duration_fraction_limits=(0.9, 1.1),
                  max_interpolated_fraction=1.0):
    """Return intervals, GP quiet segments, variable windows, and provenance.

    GP defaults: ignore bursts <2 s for HRV segmentation, add 1 s after retained
    bursts, use 1-5 min windows with a 1-min step, and require >=30 intervals.
    Existing Garmin movement detections are reused without changing the HR
    response analysis. The term quiet follows this configurable burst policy.

    All candidate windows remain inspectable. Metrics are NaN if a window has
    too few usable intervals, a sequence hole, a callback/edge gap >3 s, remaining
    invalid values, or implausible sum(cleaned BBI)/window duration. Cleaning never
    crosses sequence holes or long callback gaps. Same-callback BBIs stay distinct.
    GP has no repaired-fraction rejection; the default 1.0 retains that behavior
    while exposing raw/artifact/repaired counts and a configurable stricter limit.
    """
    positive = (min_window_s, max_window_s, step_s, max_callback_gap_s)
    if not all(np.isfinite(x) and x > 0 for x in positive):
        raise ValueError("Window lengths, step, and callback gap must be finite and positive")
    if min_window_s > max_window_s or step_s > max_window_s:
        raise ValueError("Minimum window and step cannot exceed maximum window")
    if not all(np.isfinite(x) and x >= 0 for x in (min_burst_duration_s, post_burst_guard_s)):
        raise ValueError("Burst duration and post-burst guard must be finite and non-negative")
    if not isinstance(min_beats, (int, np.integer)) or isinstance(min_beats, bool) or min_beats < 3:
        raise ValueError("min_beats must be an integer >=3")
    low_fraction, high_fraction = duration_fraction_limits
    if not (np.isfinite(low_fraction) and np.isfinite(high_fraction) and 0 <= low_fraction < high_fraction):
        raise ValueError("Duration-fraction limits must be finite, non-negative and increasing")
    if not np.isfinite(max_interpolated_fraction) or not 0 <= max_interpolated_fraction <= 1:
        raise ValueError("max_interpolated_fraction must be between 0 and 1")

    intervals = prepare_bbi_intervals(rows, bbi_limits_ms)
    # Validate all supplied bursts, including ones the GP duration policy ignores.
    burst_free_segments(bursts, recording_start, recording_end)
    used_bursts = bursts[["start", "end"]].copy()
    for name in ("start", "end"):
        used_bursts[name] = pd.to_datetime(used_bursts[name], utc=True)
    retained = (used_bursts.end - used_bursts.start).dt.total_seconds() >= min_burst_duration_s
    ignored_count = int((~retained).sum())
    used_bursts = used_bursts.loc[retained].copy()
    used_bursts["end"] += pd.Timedelta(seconds=post_burst_guard_s)
    segments = burst_free_segments(used_bursts, recording_start, recording_end)

    intervals["segment_id"] = -1
    intervals["delivery_run_id"] = -1
    intervals["bbi_clean_ms"] = np.nan
    for name in (*ARTIFACT_CLASSES, "artifact", "invalid", "interpolated"):
        intervals[name] = False
    times = pd.DatetimeIndex(intervals.callback_time_utc_approx)
    sequence = intervals.sequence.to_numpy()
    values = intervals.bbi_ms.to_numpy()
    run_id = 0
    for segment_id, segment in segments.iterrows():
        left, right = times.searchsorted([segment.start, segment.end], side="left")
        if left == right:
            continue
        # No interpolation or classifier differences cross a delivery barrier.
        breaks = ((np.diff(sequence[left:right]) != 1)
                  | (np.diff(times[left:right].asi8) / 1e9 > max_callback_gap_s))
        boundaries = np.r_[left, left + np.flatnonzero(breaks) + 1, right]
        for first, stop in zip(boundaries[:-1], boundaries[1:]):
            clean = clean_interval_run(values[first:stop], bbi_limits_ms)
            selected = intervals.index[first:stop]
            intervals.loc[selected, "segment_id"] = segment_id
            intervals.loc[selected, "delivery_run_id"] = run_id
            intervals.loc[selected, "bbi_clean_ms"] = clean["cleaned"]
            for name in (*ARTIFACT_CLASSES, "artifact", "invalid", "interpolated"):
                intervals.loc[selected, name] = clean[name]
            run_id += 1

    periods = [pd.Timedelta(seconds=x) for x in (min_window_s, max_window_s, step_s)]
    if min(p.value for p in periods) < 1:
        raise ValueError("Window lengths and step must be at least one nanosecond")
    results = []
    for segment_id, segment in segments.iterrows():
        segment_rows = intervals.loc[intervals.segment_id == segment_id]
        for start, end in build_variable_windows(segment.start, segment.end, *periods):
            left, right = times.searchsorted([start, end], side="left")
            selected = intervals.iloc[left:right]
            clean = selected.bbi_clean_ms.to_numpy()
            finite = np.isfinite(clean)
            missing = int(np.maximum(np.diff(sequence[left:right]) - 1, 0).sum())
            max_gap = float(np.diff(np.r_[start.value, times[left:right].asi8, end.value]).max() / 1e9)
            duration_s = (end - start).total_seconds()
            duration_fraction = float(clean[finite].sum() / (1000 * duration_s))
            n_interpolated = int(selected.interpolated.sum())
            fraction = n_interpolated / len(clean) if len(clean) else np.nan
            reasons = []
            if int(finite.sum()) < min_beats:
                reasons.append("too_few_intervals")
            if not finite.all():
                reasons.append("unresolved_invalid_bbi")
            if missing:
                reasons.append("missing_sequences")
            if max_gap > max_callback_gap_s:
                reasons.append("callback_gap_or_edge_silence")
            if not low_fraction <= duration_fraction <= high_fraction:
                reasons.append("bbi_duration_inconsistent_with_window")
            if fraction > max_interpolated_fraction:
                reasons.append("too_many_interpolated_intervals")
            metrics = dict(mean_hr_bpm=np.nan, rmssd_ms=np.nan, sdnn_ms=np.nan,
                           pip=np.nan, pip_pct=np.nan)
            if not reasons:
                differences = np.diff(clean)
                pip = compute_HRF(clean)
                metrics.update(mean_hr_bpm=float(np.mean(60000 / clean)),
                               rmssd_ms=float(np.sqrt(np.mean(differences ** 2))),
                               sdnn_ms=float(np.std(clean, ddof=1)), pip=pip, pip_pct=100 * pip)
            results.append({"segment_id": segment_id, "start": start, "end": end,
                            "center": start + (end - start) / 2, "window_length_s": duration_s,
                            "quiet_segment_length_s": segment.duration_s,
                            "n_intervals": len(clean), "n_usable_intervals": int(finite.sum()),
                            "n_artifacts": int(selected.artifact.sum()),
                            "n_artifacts_segment": int(segment_rows.artifact.sum()),
                            "n_invalid_intervals": int(selected.invalid.sum()),
                            "n_interpolated": n_interpolated, "interpolated_fraction": fraction,
                            "missing_sequences": missing, "max_callback_gap_s": max_gap,
                            "bbi_duration_fraction": duration_fraction, **metrics,
                            "included": not reasons, "exclusion_reason": ";".join(reasons)})
    columns = ["segment_id", "start", "end", "center", "window_length_s", "quiet_segment_length_s",
               "n_intervals", "n_usable_intervals", "n_artifacts", "n_artifacts_segment",
               "n_invalid_intervals", "n_interpolated", "interpolated_fraction", "missing_sequences",
               "max_callback_gap_s", "bbi_duration_fraction", "mean_hr_bpm", "rmssd_ms", "sdnn_ms",
               "pip", "pip_pct", "included", "exclusion_reason"]
    windows = pd.DataFrame(results, columns=columns)
    for name in ("start", "end", "center"):
        windows[name] = pd.to_datetime(windows[name], utc=True)
    windows["included"] = windows.included.astype(bool)
    windows.index.name = "window_id"
    report = {"pipeline": "GP_pipeline adapted to Garmin BBI", "min_window_s": min_window_s,
              "max_window_s": max_window_s, "step_s": step_s, "min_beats": min_beats,
              "min_burst_duration_s": min_burst_duration_s, "post_burst_guard_s": post_burst_guard_s,
              "ignored_short_bursts": ignored_count, "retained_bursts": len(used_bursts),
              "bbi_limits_ms": list(bbi_limits_ms), "max_callback_gap_s": max_callback_gap_s,
              "duration_fraction_limits": list(duration_fraction_limits),
              "max_interpolated_fraction": max_interpolated_fraction,
              "n_artifacts": int(intervals.artifact.sum()), "n_interpolated": int(intervals.interpolated.sum()),
              "window_bounds": "[start, end)", "timing": "callback arrival proxy, not beat timestamps",
              "cleaning": "single-pass GP classifier; linear interval-order interpolation within delivery runs",
              "notes": ["Short bursts below the configured duration remain inside GP quiet periods.",
                        "No PPG beat detection, invented first interval, or peak-position correction.",
                        "Garmin transport QC and classifier adaptation are not physiological validation.",
                        "Variable-duration windows and overlapping windows are not independent or interchangeable."]}
    return {"intervals": intervals, "segments": segments, "windows": windows,
            "blocked_bursts": used_bursts, "report": report}
