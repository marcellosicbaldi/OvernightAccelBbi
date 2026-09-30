"""Single-subject SPT movement and recorded-HR summaries across nights."""

from pathlib import Path

import numpy as np
import pandas as pd

from burst_hr_response import (RELATIVE_SECONDS, TERTILES, analyze_burst_hr,
                               assign_auc_tertiles, record_hr_to_observations,
                               record_times_to_utc)
from detect_acc_bursts import detect_bursts_in_intervals


def process_night(folder, naive_timezone="Europe/Rome", sampling_rate=100.0,
                  max_accel_gap_s=0.25, threshold_g=0.040):
    """Use the saved crop as SPT; split acquisition gaps before filtering.

    SPT is estimated from first through last saved acceleration sample plus one
    sample period. Gaps stay in the denominator, but never bridge HR epochs.
    Recorded HR can cover the full recording; analysis bounds enforce the crop.
    """
    folder = Path(folder)
    acc = pd.read_parquet(folder / "accelerometer_samples.parquet")
    acc["sample_time"] = record_times_to_utc(acc.sample_time, naive_timezone)
    acc = acc.sort_values("sample_time").reset_index(drop=True)
    times = pd.DatetimeIndex(acc.sample_time.drop_duplicates())
    if len(times) < 2:
        raise ValueError(f"{folder.name}: fewer than two acceleration timestamps")
    period = pd.Timedelta(seconds=1 / sampling_rate)
    gaps = np.flatnonzero(np.diff(times.asi8) / 1e9 > max_accel_gap_s)
    starts = np.r_[0, gaps + 1]
    ends = np.r_[gaps, len(times) - 1]
    bounds = pd.DataFrame({"start": times[starts], "end": times[ends] + period})
    detection = detect_bursts_in_intervals(
        acc, bounds, sampling_rate=sampling_rate, input_unit="mg",
        max_gap_s=max_accel_gap_s, alfa=threshold_g, merge_gap_s=5.0)
    hr, hr_qc = record_hr_to_observations(
        pd.read_parquet(folder / "record_bbi_fields.parquet"), naive_timezone)
    result = analyze_burst_hr(
        detection["bursts"], hr, times[0], times[-1] + period,
        analysis_intervals=detection["intervals"], max_gap_s=3.0,
        isolation_s=30.0, exclude_late_overlap=True)
    events = result["events"].copy()
    events["night"] = folder.name
    events["duration_s"] = events.duration.dt.total_seconds()
    epochs = result["epochs_pct"].copy()
    epochs["night"] = folder.name
    epochs = epochs.set_index("night", append=True).reorder_levels([1, 0])
    events = events.reset_index().set_index(["night", "burst_id"])
    spt_s = (times[-1] + period - times[0]).total_seconds()
    usable = detection["intervals"]
    covered_s = (usable.end - usable.start).dt.total_seconds().sum()
    info = {"night": folder.name, "spt_start": times[0], "spt_end": times[-1] + period,
            "spt_s": spt_s, "accel_coverage_pct": covered_s / spt_s * 100,
            "acquisition_runs": len(bounds), "hr_invalid_records": hr_qc["invalid_or_out_of_range_hr"],
            "burst_count": len(events), "hr_included": int(events.included.sum())}
    return events, epochs, info


def summarize(events, epochs, nights):
    """Fit common cutoffs to ALL candidate bursts, then apply HR eligibility."""
    events = events.copy()
    events["auc_tertile"], cutoffs = assign_auc_tertiles(events.AUC)
    rows = []
    for night in nights.index:
        part = events.loc[events.index.get_level_values("night") == night]
        for group in ["All", *TERTILES]:
            subset = part if group == "All" else part.loc[part.auc_tertile == group]
            valid = subset.loc[subset.included]
            rows.append({"night": night, "group": group, "burst_count": len(subset),
                         "hr_n": len(valid), "p2p_mean_g": subset["peak-to-peak"].mean(),
                         "auc_mean_g_s": subset.AUC.mean(), "duration_mean_s": subset.duration_s.mean(),
                         "duration_total_s": subset.duration_s.sum(),
                         "burst_spt_pct": subset.duration_s.sum() / nights.loc[night, "spt_s"] * 100,
                         "hr_peak_mean_pct": valid.hr_peak_increase_pct.mean(),
                         "hr_latency_mean_s": valid.hr_peak_latency_s.mean(),
                         "hr_latency_n": valid.hr_peak_latency_s.count()})
    nightly = pd.DataFrame(rows)
    included = events.loc[events.included]
    curve_rows = []
    for (night, group), subset in included.groupby(["night", "auc_tertile"], observed=True):
        mean = epochs.loc[subset.index, RELATIVE_SECONDS].mean()
        curve_rows.extend({"night": night, "group": group, "relative_s": int(t),
                           "mean_pct": value, "events": len(subset)} for t, value in mean.items())
    curves = pd.DataFrame(curve_rows, columns=["night", "group", "relative_s", "mean_pct", "events"])
    grand = curves.groupby(["group", "relative_s"]).mean_pct.agg(["mean", "std", "count", "sem"]).reset_index()
    # Paired nightly averages use precisely the same eligible bursts for X and Y.
    pairs = included.reset_index()
    pairs["size_group"] = np.where(pairs.auc_tertile == "Low", "Small", "Medium + large")
    night_pairs = pairs.groupby(["night", "size_group"], observed=True).agg(
        AUC=("AUC", "mean"), **{"peak-to-peak": ("peak-to-peak", "mean")},
        hr_peak_increase_pct=("hr_peak_increase_pct", "mean"), n_bursts=("burst_id", "size")).reset_index()
    correlations = []
    for level, table in [("Burst", pairs), ("Night", night_pairs)]:
        for group in ["Small", "Medium + large"]:
            selected = table.loc[table.size_group == group]
            for metric in ["AUC", "peak-to-peak"]:
                values = selected[[metric, "hr_peak_increase_pct"]].replace([np.inf, -np.inf], np.nan).dropna()
                usable = len(values) >= 3 and (values.nunique() > 1).all()
                correlations.append({"level": level, "group": group, "metric": metric,
                                     "n": len(values), "n_nights": selected.night.nunique(),
                                     "pearson_r": values.corr().iloc[0, 1] if usable else np.nan,
                                     "spearman_rho": values.corr(method="spearman").iloc[0, 1] if usable else np.nan})
    return {"events": events, "nightly": nightly, "night_curves": curves,
            "grand_curves": grand, "burst_pairs": pairs, "night_pairs": night_pairs,
            "correlations": pd.DataFrame(correlations), "cutoffs": cutoffs}


def analyze_subject(root, **kwargs):
    folders = sorted(p for p in Path(root).iterdir() if p.is_dir() and (p / "accelerometer_samples.parquet").exists())
    if not folders:
        raise ValueError("No processed night folders found")
    events, epochs, metadata = [], [], []
    for folder in folders:
        print(f"Processing {folder.name}", flush=True)
        e, h, info = process_night(folder, **kwargs)
        events.append(e)
        epochs.append(h)
        metadata.append(info)
    nights = pd.DataFrame(metadata).set_index("night")
    epoch_table = pd.concat(epochs)
    return {**summarize(pd.concat(events), epoch_table, nights), "nights": nights, "epochs": epoch_table}
