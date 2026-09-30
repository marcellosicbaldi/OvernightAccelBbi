"""Single-subject SPT movement and recorded-HR summaries across nights.

Version 2 adds a continuous movement-intensity representation and nightly
movement-adjusted cardiac-reactivity metrics. AUC tertiles are retained only
for descriptive/visual compatibility with the original notebook.
"""

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


def _fit_simple_ols(x, y, min_n=5):
    """Fit y = intercept + slope*x using finite values only.

    Returns slope/intercept, R^2 and classical OLS standard errors. This is a
    descriptive within-night fit; it is not an independent-event inferential
    model.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)

    out = {"n": n, "intercept": np.nan, "slope": np.nan, "r2": np.nan,
           "intercept_se": np.nan, "slope_se": np.nan}
    if n < min_n or np.unique(x).size < 2:
        return out

    x_mean = x.mean()
    y_mean = y.mean()
    sxx = np.sum((x - x_mean) ** 2)
    if sxx <= 0:
        return out

    slope = np.sum((x - x_mean) * (y - y_mean)) / sxx
    intercept = y_mean - slope * x_mean
    fitted = intercept + slope * x
    residual = y - fitted
    sse = np.sum(residual ** 2)
    sst = np.sum((y - y_mean) ** 2)
    r2 = 1 - sse / sst if sst > 0 else np.nan

    if n > 2:
        mse = sse / (n - 2)
        slope_se = np.sqrt(mse / sxx)
        intercept_se = np.sqrt(mse * (1 / n + x_mean ** 2 / sxx))
    else:
        slope_se = np.nan
        intercept_se = np.nan

    out.update({"intercept": intercept, "slope": slope, "r2": r2,
                "intercept_se": intercept_se, "slope_se": slope_se})
    return out


def _continuous_auc_bins(included, n_bins=15):
    """Quantile-bin the continuous AUC-HR relationship for visualization."""
    data = included[["AUC", "log10_auc", "hr_peak_increase_pct"]].replace(
        [np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return pd.DataFrame(columns=["bin", "n", "auc_median_g_s", "log10_auc_mean",
                                     "hr_mean_pct", "hr_std_pct", "hr_sem_pct"])

    # qcut may drop bins when AUC ties are common; that is fine and explicit.
    data = data.copy()
    data["bin"] = pd.qcut(data["log10_auc"], q=min(n_bins, len(data)),
                          duplicates="drop")
    bins = data.groupby("bin", observed=True).agg(
        n=("hr_peak_increase_pct", "size"),
        auc_median_g_s=("AUC", "median"),
        log10_auc_mean=("log10_auc", "mean"),
        hr_mean_pct=("hr_peak_increase_pct", "mean"),
        hr_std_pct=("hr_peak_increase_pct", "std"),
    ).reset_index(drop=False)
    bins["hr_sem_pct"] = bins.hr_std_pct / np.sqrt(bins.n)
    bins["bin"] = np.arange(len(bins))
    return bins


def _nightly_reactivity(events, nights, reference_log10_auc, min_events=5):
    """Estimate movement-adjusted HR reactivity separately for every night.

    The predictor is log10(AUC) centered at a subject-wide reference AUC.
    Therefore:
      * slope = HR percentage-point change per 10x increase in movement AUC;
      * intercept = expected HR peak increase at the reference movement AUC.
    """
    included = events.loc[events.included].copy()
    rows = []
    for night in nights.index:
        part = included.loc[included.index.get_level_values("night") == night]
        fit = _fit_simple_ols(part["log10_auc_centered"],
                              part["hr_peak_increase_pct"], min_n=min_events)
        rows.append({
            "night": night,
            "hr_eligible_n": fit["n"],
            "reactivity_slope_pct_per_decade": fit["slope"],
            "reactivity_slope_se": fit["slope_se"],
            "hr_at_reference_auc_pct": fit["intercept"],
            "hr_at_reference_auc_se": fit["intercept_se"],
            "reactivity_r2": fit["r2"],
            "mean_hr_peak_pct_unadjusted": part.hr_peak_increase_pct.mean(),
            "mean_auc_g_s_eligible": part.AUC.mean(),
            "median_auc_g_s_eligible": part.AUC.median(),
        })
    return pd.DataFrame(rows).set_index("night")


def _nightly_motor_burden(events, nights):
    """Summarize movement amount using all candidate bursts, regardless of HR QC."""
    rows = []
    for night in nights.index:
        part = events.loc[events.index.get_level_values("night") == night]
        hours = nights.loc[night, "spt_s"] / 3600.0
        total_duration = part.duration_s.sum()
        total_auc = part.AUC.sum()
        rows.append({
            "night": night,
            "burst_count_all": len(part),
            "bursts_per_hour": len(part) / hours if hours > 0 else np.nan,
            "movement_duration_s": total_duration,
            "movement_burden_spt_pct": total_duration / nights.loc[night, "spt_s"] * 100,
            "auc_total_g_s": total_auc,
            "auc_per_hour_g_s": total_auc / hours if hours > 0 else np.nan,
            "auc_median_g_s_all": part.AUC.median(),
            "auc_mean_g_s_all": part.AUC.mean(),
        })
    return pd.DataFrame(rows).set_index("night")


def summarize(events, epochs, nights, reactivity_min_events=5, continuous_bins=15):
    """Summarize movement/HR while treating AUC as the primary continuous variable.

    Common AUC tertiles are retained for descriptive plots only. The primary
    movement predictor is log10(AUC), and each night receives a reactivity fit
    after centering log10(AUC) at the subject-wide median eligible AUC.
    """
    events = events.copy()

    # Keep the original common tertiles for visualization/backward compatibility.
    events["auc_tertile"], cutoffs = assign_auc_tertiles(events.AUC)

    # Primary continuous movement-intensity representation.
    positive_auc = events.AUC.where(events.AUC > 0)
    events["log10_auc"] = np.log10(positive_auc)
    eligible_log_auc = events.loc[events.included, "log10_auc"].replace(
        [np.inf, -np.inf], np.nan).dropna()
    if eligible_log_auc.empty:
        reference_log10_auc = np.nan
        reference_auc = np.nan
    else:
        reference_log10_auc = float(eligible_log_auc.median())
        reference_auc = float(10 ** reference_log10_auc)
    events["log10_auc_centered"] = events.log10_auc - reference_log10_auc

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

    # Original grouped correlations, kept for comparison.
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

    # Continuous pooled relationship (descriptive; events are clustered within nights).
    global_fit = _fit_simple_ols(included.log10_auc_centered,
                                 included.hr_peak_increase_pct,
                                 min_n=reactivity_min_events)
    global_reactivity = pd.Series({
        "reference_auc_g_s": reference_auc,
        "reference_log10_auc": reference_log10_auc,
        "n": global_fit["n"],
        "hr_at_reference_auc_pct": global_fit["intercept"],
        "hr_at_reference_auc_se": global_fit["intercept_se"],
        "reactivity_slope_pct_per_decade": global_fit["slope"],
        "reactivity_slope_se": global_fit["slope_se"],
        "r2": global_fit["r2"],
    }, name="value")
    continuous = _continuous_auc_bins(included, n_bins=continuous_bins)

    # Point 2: nightly movement-adjusted reactivity.
    reactivity = _nightly_reactivity(events, nights, reference_log10_auc,
                                     min_events=reactivity_min_events)

    # Point 3: separate motor burden (all bursts) from cardiac reactivity (eligible HR).
    motor = _nightly_motor_burden(events, nights)
    phenotype = nights.join(motor, how="left").join(reactivity, how="left")

    return {"events": events, "nightly": nightly, "night_curves": curves,
            "grand_curves": grand, "burst_pairs": pairs, "night_pairs": night_pairs,
            "correlations": pd.DataFrame(correlations), "cutoffs": cutoffs,
            "continuous_auc_bins": continuous,
            "global_reactivity": global_reactivity,
            "nightly_reactivity": reactivity,
            "nightly_motor_burden": motor,
            "nightly_phenotype": phenotype,
            "reference_auc_g_s": reference_auc,
            "reference_log10_auc": reference_log10_auc}


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
