"""Fixed-window HR/RMSSD profiles from saved SPT and GP quiet-period exports."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from burst_hr_response import record_hr_to_observations, record_times_to_utc, sample_hr


def fixed_hrv_windows(intervals, segments, start, end, length_s=60,
                      max_interpolated_fraction=0.05):
    """SPT-anchored windows every 60 s; never join quiet or delivery runs.

    Keep rejected windows with reasons. Five-minute windows are centred on the
    same minute centres as one-minute windows; incomplete edges are omitted.
    Reuse saved GP cleaning, not a second artifact-classification pass.
    """
    if length_s not in (60, 300):
        raise ValueError("Use 60 or 300 second windows")
    if not 0 <= max_interpolated_fraction <= 1:
        raise ValueError("Interpolation fraction must be in [0, 1]")
    data = intervals.sort_values(["callback_time_utc_approx", "sequence"])
    times = pd.DatetimeIndex(data.callback_time_utc_approx)
    quiet = segments.sort_values("start")
    if (quiet.end <= quiet.start).any() or (quiet.start.iloc[1:].to_numpy() < quiet.end.iloc[:-1].to_numpy()).any():
        raise ValueError("Quiet segments must have positive duration and not overlap")
    rows = []
    n_minutes = int((end - start).total_seconds() // 60)
    for minute in range(n_minutes):
        center = start + pd.Timedelta(seconds=60 * minute + 30)
        left = center - pd.Timedelta(seconds=length_s / 2)
        right = center + pd.Timedelta(seconds=length_s / 2)
        if left < start or right > end:
            continue
        matching = quiet.loc[(quiet.start <= left) & (quiet.end >= right)]
        a, b = times.searchsorted([left, right])
        part = data.iloc[a:b]
        values = part.bbi_clean_ms.to_numpy(dtype=float)
        finite = np.isfinite(values) & (values > 0)
        fraction = float(part.interpolated.mean()) if len(part) else np.nan
        gap = float(np.diff(np.r_[left.value, times[a:b].asi8, right.value]).max() / 1e9)
        duration_fraction = float(values[finite].sum() / (1000 * length_s))
        reasons = []
        if len(matching) != 1:
            reasons.append("not_wholly_in_one_quiet_segment")
        elif not (part.segment_id == matching.index[0]).all():
            reasons.append("quiet_segment_mismatch")
        if len(part) < max(30, int(length_s / 2)):
            reasons.append("too_few_intervals")
        if not finite.all():
            reasons.append("unresolved_bbi")
        if len(part) and (part.delivery_run_id.nunique() != 1 or (part.delivery_run_id < 0).any()):
            reasons.append("delivery_boundary")
        if len(part) > 1 and (np.diff(part.sequence) != 1).any():
            reasons.append("sequence_discontinuity")
        if gap > 5:
            reasons.append("callback_gap_or_edge_silence")
        if not .9 <= duration_fraction <= 1.1:
            reasons.append("bbi_duration_inconsistent")
        if fraction > max_interpolated_fraction:
            reasons.append("too_many_interpolated")
        rmssd = float(np.sqrt(np.mean(np.diff(values) ** 2))) if not reasons else np.nan
        rows.append({"minute": minute, "start": left, "end": right, "center": center,
                     "window_s": length_s, "quiet": len(matching) == 1,
                     "n_intervals": len(part), "interpolated_fraction": fraction,
                     "max_callback_gap_s": gap, "bbi_duration_fraction": duration_fraction,
                     "included": not reasons, "exclusion_reason": ";".join(reasons),
                     "rmssd_ms": rmssd, "lnrmssd": np.log(rmssd) if rmssd > 0 else np.nan,
                     "bbi_hr_bpm": float(np.mean(60000 / values)) if not reasons else np.nan})
    return pd.DataFrame(rows, columns=["minute", "start", "end", "center", "window_s", "quiet",
                                      "n_intervals", "interpolated_fraction", "max_callback_gap_s",
                                      "bbi_duration_fraction", "included", "exclusion_reason",
                                      "rmssd_ms", "lnrmssd", "bbi_hr_bpm"])


def minute_hr(records, start, end, naive_timezone="Europe/Rome", min_coverage=.9):
    """Mean/median recorded HR on a 1-Hz grid; gaps >3 s stay missing."""
    observations, _ = record_hr_to_observations(records, naive_timezone)
    observations = observations.loc[(observations.index >= start) & (observations.index < end)]
    n = int((end - start).total_seconds() // 60)
    query = pd.date_range(start, periods=n * 60, freq="s")
    hr, _, _ = sample_hr(observations, query, max_gap_s=3)
    frame = pd.DataFrame({"minute": np.repeat(np.arange(n), 60), "hr": hr})
    grouped = frame.groupby("minute").hr
    result = grouped.agg(hr_mean_bpm="mean", hr_median_bpm="median", valid_seconds="count").reset_index()
    result["hr_coverage"] = result.valid_seconds / 60
    result.loc[result.hr_coverage < min_coverage, ["hr_mean_bpm", "hr_median_bpm"]] = np.nan
    return result


def analyze_night(folder, naive_timezone="Europe/Rome", max_interpolated_fraction=.05):
    folder = Path(folder)
    times = record_times_to_utc(pd.read_parquet(folder / "accelerometer_samples.parquet",
                                               columns=["sample_time"]).sample_time, naive_timezone)
    start, end = times.min(), times.max() + pd.Timedelta(milliseconds=10)
    intervals = pd.read_parquet(folder / "gp_bbi_intervals.parquet")
    segments = pd.read_parquet(folder / "gp_hrv_segments.parquet")
    settings = json.loads((folder / "gp_hrv_settings.json").read_text(encoding="utf-8"))
    hr = minute_hr(pd.read_parquet(folder / "record_bbi_fields.parquet"), start, end, naive_timezone)
    one = fixed_hrv_windows(intervals, segments, start, end, 60, max_interpolated_fraction).merge(hr, on="minute", how="left")
    five = fixed_hrv_windows(intervals, segments, start, end, 300, max_interpolated_fraction)
    one["quiet_hr_bpm"] = one.hr_mean_bpm.where(one.quiet)
    for frame in [one, five]:
        frame["night"] = folder.name
        frame["elapsed_min"] = (frame.center - start).dt.total_seconds() / 60
        frame["before_wake_min"] = (frame.center - end).dt.total_seconds() / 60
        frame["spt_pct"] = 100 * (frame.center - start).dt.total_seconds() / (end - start).total_seconds()
    info = {"night": folder.name, "spt_start": start, "spt_end": end,
            "spt_minutes": (end - start).total_seconds() / 60,
            "complete_minutes": len(one), "quiet_minutes": int(one.quiet.sum()),
            "valid_hr_minutes": int(one.hr_mean_bpm.count()), "valid_rmssd_minutes": int(one.rmssd_ms.count()),
            "valid_five_min_windows": int(five.rmssd_ms.count()),
            "rmssd_spt_coverage_pct": one.rmssd_ms.count() / ((end - start).total_seconds() / 60) * 100,
            "source_min_burst_s": settings.get("min_burst_duration_s"),
            "source_post_burst_guard_s": settings.get("post_burst_guard_s"),
            "source_analysis_window": settings.get("analysis_window", "unknown")}
    return one, five, info, settings


METRICS = ["hr_mean_bpm", "quiet_hr_bpm", "rmssd_ms", "lnrmssd"]


def night_summary(one):
    """Fixed one-minute values get equal weight; never average rolling windows."""
    rows = []
    for night, frame in one.groupby("night", sort=True):
        row = {"night": night}
        for metric in METRICS:
            series = frame[metric]
            row.update({f"{metric}_{stat}": value for stat, value in
                        {"mean": series.mean(), "median": series.median(), "sd": series.std(),
                         "q25": series.quantile(.25), "q75": series.quantile(.75), "n": series.count()}.items()})
            early = frame.loc[frame.elapsed_min < 60, metric]
            late = frame.loc[frame.before_wake_min >= -60, metric]
            row[f"{metric}_first_hour_n"] = early.count()
            row[f"{metric}_last_hour_n"] = late.count()
            row[f"{metric}_last_minus_first"] = late.mean() - early.mean() if min(early.count(), late.count()) >= 15 else np.nan
            for label, lo, hi in [("early", 0, 100/3), ("middle", 100/3, 200/3), ("late", 200/3, 100)]:
                values = frame.loc[(frame.spt_pct >= lo) & (frame.spt_pct < hi), metric]
                row[f"{metric}_{label}_mean"] = values.mean()
                row[f"{metric}_{label}_n"] = values.count()
        rows.append(row)
    return pd.DataFrame(rows).set_index("night")


def aligned_profiles(one, metric, alignment):
    """Bin within each night first, then give available nights equal weight.

    SPT percentage uses 1% bins, not minutes. No missing-bin interpolation.
    """
    column = {"onset": "elapsed_min", "wake": "before_wake_min", "percent": "spt_pct"}[alignment]
    frame = one[["night", column, metric]].copy()
    frame["bin"] = np.floor(frame[column]).astype(int)
    individual = frame.groupby(["night", "bin"])[metric].mean().unstack("night")
    if not individual.empty:
        individual = individual.reindex(range(individual.index.min(), individual.index.max() + 1))
    summary = pd.DataFrame({"mean": individual.mean(axis=1), "sd": individual.std(axis=1),
                            "n_nights": individual.count(axis=1)})
    return individual, summary


def analyze_subject(root, **kwargs):
    folders = sorted(p for p in Path(root).iterdir() if p.is_dir() and (p / "accelerometer_samples.parquet").exists())
    if not folders:
        raise ValueError("No processed night folders")
    minutes, rolling, metadata, provenance = [], [], [], {}
    for folder in folders:
        print(f"Processing {folder.name}", flush=True)
        one, five, info, settings = analyze_night(folder, **kwargs)
        minutes.append(one)
        rolling.append(five)
        metadata.append(info)
        provenance[folder.name] = settings
    one, five = pd.concat(minutes, ignore_index=True), pd.concat(rolling, ignore_index=True)
    return {"minutes": one, "five_minutes": five,
            "nights": pd.DataFrame(metadata).set_index("night"),
            "summary": night_summary(one), "source_settings": provenance}
