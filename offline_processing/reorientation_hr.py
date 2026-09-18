"""Descriptive HR comparisons and outcome-blind, within-tertile AUC matching.

Events from one night are not independent participants. These outputs do not
establish causality, sleep stage, body position, or population-level inference.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from burst_hr_response import TERTILES
from wrist_reorientation import CHANGED, CLASSES, LITTLE

PAIR_COLUMNS = [
    "auc_tertile", "changed_id", "little_id", "changed_auc_g_s", "little_auc_g_s",
    "auc_ratio", "changed_tilt_deg", "little_tilt_deg", "changed_duration_s",
    "little_duration_s", "time_separation_min", "changed_peak_pct", "little_peak_pct",
    "peak_difference_pp", "post_mean_difference_pp",
]


def join_hr_orientation(events, orientation):
    """Join by burst ID AND verify event times, never silently align by row order."""
    if not events.index.is_unique or not orientation.index.is_unique:
        raise ValueError("Burst IDs must be unique")
    if len(events) != len(orientation) or not events.index.isin(orientation.index).all():
        raise ValueError("HR and orientation must describe exactly the same burst IDs")
    orientation = orientation.reindex(events.index)
    for left, right in [("start", "burst_start"), ("end", "burst_end")]:
        a = pd.to_datetime(events[left], utc=True)
        b = pd.to_datetime(orientation[right], utc=True)
        if a.isna().any() or b.isna().any() or not np.array_equal(a.to_numpy(), b.to_numpy()):
            raise ValueError("Burst timestamps do not match between HR and orientation")
    result = events.join(orientation)
    result["comparison_included"] = (
        result.included & result.orientation_usable
        & result.reorientation_class.isin([LITTLE, CHANGED])
    )
    reasons = []
    for _, row in result.iterrows():
        parts = []
        if not row.included:
            parts.append("HR: " + (str(row.exclusion_reason) or "excluded"))
        if not row.orientation_usable or row.reorientation_class not in [LITTLE, CHANGED]:
            parts.append("orientation: " + (str(row.orientation_reason) or str(row.reorientation_class)))
        reasons.append("; ".join(parts))
    result["comparison_exclusion_reason"] = reasons
    result.attrs["orientation_settings"] = orientation.attrs.copy()
    return result


def summarize_groups(events, epochs_pct):
    """Count every class; average only complete, comparison-eligible HR epochs."""
    if not epochs_pct.index.is_unique or not events.index.isin(epochs_pct.index).all():
        raise ValueError("HR epochs must contain every unique burst ID")
    selected_epochs = epochs_pct.loc[events.index[events.comparison_included]]
    if not np.isfinite(selected_epochs.to_numpy(dtype=float)).all():
        raise ValueError("Comparison-eligible HR epochs must be complete")
    summaries, curves = [], []
    for tertile in TERTILES:
        for label in CLASSES:
            candidates = events.loc[(events.auc_tertile == tertile) & (events.reorientation_class == label)]
            selected = candidates.loc[candidates.comparison_included]
            summaries.append({
                "auc_tertile": tertile, "reorientation_class": label,
                "candidate_n": len(candidates), "hr_eligible_n": int(candidates.included.sum()),
                "comparison_n": len(selected), "auc_median_g_s": selected.AUC.median(),
                "tilt_median_deg": selected.tilt_change_deg.median(),
                "peak_mean_pct": selected.hr_peak_increase_pct.mean(),
                "peak_median_pct": selected.hr_peak_increase_pct.median(),
                "post_mean_pct": selected.hr_post_mean_pct.mean(),
            })
            if label not in [LITTLE, CHANGED]:
                continue
            values = epochs_pct.loc[selected.index]
            mean, sem = values.mean(), values.sem()
            for t in epochs_pct.columns:
                curves.append({"auc_tertile": tertile, "reorientation_class": label,
                               "relative_s": t, "n": len(values),
                               "mean_pct": mean[t], "sem_pct": sem[t]})
    return pd.DataFrame(summaries), pd.DataFrame(curves)


def match_by_auc(events, max_auc_ratio=1.25):
    """Maximum-cardinality, minimum-log-distance one-to-one matching.

    Match only eligible opposite classes within the original AUC tertile and
    the specified max(AUC)/min(AUC) caliper. No event is reused. HR outcomes are
    not used in selecting pairs. Zero AUC is excluded because ratios are undefined.
    Duration, time of night and sleep state are NOT controlled by this matching.
    """
    if not np.isfinite(max_auc_ratio) or max_auc_ratio < 1:
        raise ValueError("max_auc_ratio must be finite and >= 1")
    if not events.index.is_unique:
        raise ValueError("Burst IDs must be unique")
    candidates = events.loc[events.comparison_included & np.isfinite(events.AUC) & (events.AUC > 0)]
    rows = []
    for tertile in TERTILES:
        subset = candidates.loc[candidates.auc_tertile == tertile]
        changed = subset.loc[subset.reorientation_class == CHANGED]
        little = subset.loc[subset.reorientation_class == LITTLE]
        n, m = len(changed), len(little)
        if not n or not m:
            continue
        distance = np.abs(np.log(changed.AUC.to_numpy())[:, None] - np.log(little.AUC.to_numpy())[None, :])
        caliper = np.log(max_auc_ratio)
        allowed = distance <= caliper + 1e-12
        # Dummy columns allow unmatched rows. A lost pair costs more than the
        # sum of all allowed distances, so cardinality precedes closeness.
        penalty = n + 1.0
        cost = np.full((n, m + n), penalty)
        cost[:, :m] = np.where(allowed, distance / max(caliper, 1e-12), penalty * (n + 2))
        ii, jj = linear_sum_assignment(cost)
        for i, j in zip(ii, jj):
            if j >= m or not allowed[i, j]:
                continue
            a, b = changed.iloc[i], little.iloc[j]
            a_start, b_start = pd.Timestamp(a.start), pd.Timestamp(b.start)
            rows.append({
                "auc_tertile": tertile, "changed_id": changed.index[i], "little_id": little.index[j],
                "changed_auc_g_s": a.AUC, "little_auc_g_s": b.AUC,
                "auc_ratio": max(a.AUC, b.AUC) / min(a.AUC, b.AUC),
                "changed_tilt_deg": a.tilt_change_deg, "little_tilt_deg": b.tilt_change_deg,
                "changed_duration_s": (pd.Timestamp(a.end) - a_start).total_seconds(),
                "little_duration_s": (pd.Timestamp(b.end) - b_start).total_seconds(),
                "time_separation_min": abs((a_start - b_start).total_seconds()) / 60,
                "changed_peak_pct": a.hr_peak_increase_pct, "little_peak_pct": b.hr_peak_increase_pct,
                "peak_difference_pp": a.hr_peak_increase_pct - b.hr_peak_increase_pct,
                "post_mean_difference_pp": a.hr_post_mean_pct - b.hr_post_mean_pct,
            })
    return pd.DataFrame(rows, columns=PAIR_COLUMNS).rename_axis("pair_id")


def summarize_pairs(pairs, epochs_pct):
    """Return paired HR curves and descriptive effects in percentage points."""
    if not epochs_pct.index.is_unique:
        raise ValueError("HR epoch IDs must be unique")
    ids = pd.concat([pairs.changed_id, pairs.little_id])
    if ids.duplicated().any() or not ids.isin(epochs_pct.index).all():
        raise ValueError("Pairs must use distinct, existing burst IDs")
    summaries, curves = [], []
    for tertile in TERTILES:
        subset = pairs.loc[pairs.auc_tertile == tertile]
        a = epochs_pct.loc[subset.changed_id].to_numpy(dtype=float)
        b = epochs_pct.loc[subset.little_id].to_numpy(dtype=float)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Matched HR epochs must be complete")
        n = len(subset)
        summaries.append({
            "auc_tertile": tertile, "pairs_n": n,
            "auc_ratio_median": pd.to_numeric(subset.auc_ratio).median(),
            "peak_difference_mean_pp": pd.to_numeric(subset.peak_difference_pp).mean(),
            "peak_difference_median_pp": pd.to_numeric(subset.peak_difference_pp).median(),
            "post_mean_difference_pp": pd.to_numeric(subset.post_mean_difference_pp).mean(),
        })
        for name, values in [(CHANGED, a), (LITTLE, b), ("Paired difference", a - b)]:
            mean = values.mean(axis=0) if n else np.full(len(epochs_pct.columns), np.nan)
            sem = values.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(len(epochs_pct.columns), np.nan)
            for k, t in enumerate(epochs_pct.columns):
                curves.append({"auc_tertile": tertile, "series": name, "relative_s": t,
                               "n": n, "mean_pct": mean[k], "sem_pct": sem[k]})
    return pd.DataFrame(summaries), pd.DataFrame(curves)
