"""Manual awake/invalid periods override analysis selection, not VH2015 labels."""

import re

import pandas as pd

from analysis_windows import first_overlapping_interval, subtract_intervals, validate_intervals


def prepare_exclusions(annotations, recording_start, recording_end, timezone="Europe/Rome"):
    """Validate dated (start, end, reason) entries; merge and clip to this night.

    Naive datetimes denote LOCAL wall time. DST ambiguity/nonexistent times are
    errors; explicit UTC offsets can disambiguate them. Other nights are ignored.
    Overlapping/touching annotations form a union so time is counted only once.
    """
    recording = validate_intervals(pd.DataFrame({"start": [recording_start], "end": [recording_end]}))
    start, end = recording.iloc[0]
    rows = []
    for annotation in annotations:
        if isinstance(annotation, (str, dict)) or len(annotation) != 3:
            raise ValueError("Each exclusion needs (dated start, dated end, reason)")
        first, stop, reason = annotation
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Each manual exclusion needs a nonempty reason")
        bounds = []
        for value in (first, stop):
            if isinstance(value, str) and not re.match(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", value):
                raise ValueError("Use full dates and times, e.g. '2025-01-02 03:15'")
            stamp = pd.Timestamp(value)
            if pd.isna(stamp):
                raise ValueError("Manual exclusion timestamps must not be missing")
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
            bounds.append(stamp.tz_convert("UTC"))
        first, stop = bounds
        if stop <= first:
            raise ValueError("Manual exclusion end must follow start; include the next date across midnight")
        first, stop = max(start, first), min(end, stop)
        if first < stop:
            rows.append((first, stop, reason.strip()))
    merged = []
    for first, stop, reason in sorted(rows):
        if merged and first <= merged[-1][1]:
            previous = merged[-1]
            previous[1] = max(previous[1], stop)
            if reason not in previous[2]:
                previous[2].append(reason)
        else:
            merged.append([first, stop, [reason]])
    result = pd.DataFrame([(a, b, "; ".join(reasons)) for a, b, reasons in merged],
                          columns=["start", "end", "reason"])
    for column in ("start", "end"):
        result[column] = pd.to_datetime(result[column], utc=True)
    return result


def exclusion_report(exclusions):
    """JSON-safe provenance for local exports."""
    return {
        "intervals": [{"start": row.start.isoformat(), "end": row.end.isoformat(), "reason": row.reason}
                      for row in exclusions.itertuples(index=False)],
        "excluded_seconds": float((exclusions.end - exclusions.start).dt.total_seconds().sum()),
        "burst_policy": "exclude whole candidate on any positive manual-period overlap",
        "boundary_policy": "end exclusive; HR epochs and HRV windows cannot cross exclusions",
    }


def apply_exclusions(detection, exclusions):
    """Flag all candidates, remove overlapping whole bursts, and split signals.

    Apply AFTER once-per-acquisition detection and sleep/wake selection. Full
    candidate bounds/AUC and VH2015 labels remain intact for inspection and as
    movement barriers for HR isolation and HRV. No sleep-side fragment is made.
    """
    intervals = subtract_intervals(detection["intervals"], exclusions)
    all_bursts = detection["all_bursts"].copy()
    all_bursts["excluded_manually"] = (
        first_overlapping_interval(all_bursts.start, all_bursts.end, exclusions) >= 0
    )
    all_bursts["manual_exclusion_reason"] = ""
    for period in exclusions.itertuples(index=False):
        overlaps = (all_bursts.start < period.end) & (all_bursts.end > period.start)
        previous = all_bursts.loc[overlaps, "manual_exclusion_reason"]
        all_bursts.loc[overlaps, "manual_exclusion_reason"] = previous.map(
            lambda text: f"{text}; {period.reason}" if text else period.reason
        )
    flags = all_bursts.set_index("candidate_id")
    bursts = detection["bursts"].copy()
    for column in ("excluded_manually", "manual_exclusion_reason"):
        bursts[column] = bursts.candidate_id.map(flags[column])
    bursts = bursts.loc[~bursts.excluded_manually].copy().reset_index(drop=True)
    bursts["analysis_interval_id"] = first_overlapping_interval(bursts.start, bursts.end, intervals)
    signals = []
    for interval_id, interval in intervals.iterrows():
        for part in detection["signals"]:
            selected = {"analysis_interval_id": interval_id, "threshold": part["threshold"]}
            for name in ("magnitude", "filtered", "score"):
                values = part[name]
                selected[name] = values.loc[(values.index >= interval.start) & (values.index < interval.end)]
            if not selected["magnitude"].empty:
                signals.append(selected)
    return {**detection, "bursts": bursts, "all_bursts": all_bursts,
            "signals": signals, "intervals": intervals}
