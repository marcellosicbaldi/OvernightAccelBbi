"""Measure stable changes of wrist gravity direction, NOT sleeping position.

Only tilt relative to gravity is observable here; rotation about gravity can
remain invisible. Defaults are exploratory QC/angle settings, not validated
sleep-position thresholds. Raw acceleration retains its gravity component.
"""

import numpy as np
import pandas as pd

LITTLE = "Little reorientation"
CHANGED = "Sustained reorientation"
INTERMEDIATE = "Intermediate"
UNKNOWN = "Unknown"
CLASSES = [LITTLE, INTERMEDIATE, CHANGED, UNKNOWN]


def angle_degrees(a, b):
    """Angle between vectors, with zero/nonfinite vectors producing NaN."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    norms = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    dot = np.sum(a * b, axis=-1)
    cosine = np.divide(dot, norms, out=np.full(np.broadcast_shapes(dot.shape, norms.shape), np.nan),
                       where=np.isfinite(norms) & (norms > 0))
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def prepare_wrist_axes(accel_df, input_unit="mg"):
    """Convert XYZ to g, sort UTC timestamps, and average duplicate samples.

    No high-pass filtering, magnitude-only reduction, or gap interpolation is
    applied. If any duplicate row has invalid XYZ, its timestamp remains invalid.
    """
    if input_unit not in ("g", "mg"):
        raise ValueError("input_unit must be 'g' or 'mg'")
    times = pd.DatetimeIndex(pd.to_datetime(accel_df["sample_time"], utc=True))
    if times.hasnans:
        raise ValueError("Missing accelerometer timestamps")
    values = accel_df[["accel_x", "accel_y", "accel_z"]].to_numpy(dtype=float)
    bad = ~np.isfinite(values).all(axis=1)
    values = values / (1000.0 if input_unit == "mg" else 1.0)
    axes = pd.DataFrame(values, index=times, columns=["x_g", "y_g", "z_g"])
    axes["invalid"] = bad
    groups = axes.groupby(level=0, sort=True)
    result = groups[["x_g", "y_g", "z_g"]].mean()
    result.loc[groups.invalid.any()] = np.nan
    result.index.name = "time_utc"
    result.attrs.update({"input_unit": input_unit, "output_unit": "g",
                         "duplicate_timestamps_averaged": int(times.duplicated().sum()),
                         "invalid_input_rows": int(bad.sum())})
    return result


def _window_features(axes, start, end, sampling_rate, min_coverage, max_gap_s,
                     max_rms_g, max_spread_deg, gravity_limits_g):
    segment = axes.iloc[axes.index.searchsorted(start):axes.index.searchsorted(end)]
    values = segment.to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=1)
    clean = values[finite]
    times = segment.index[finite]
    duration = (end - start).total_seconds()
    coverage = min(1.0, len(clean) / (duration * sampling_rate))
    elapsed = (times - start).total_seconds().to_numpy()
    largest_gap = float(np.diff(np.r_[0.0, elapsed, duration]).max())
    data = {"coverage": coverage, "max_gap_s": largest_gap, "norm_g": np.nan,
            "rms_g": np.nan, "spread_deg": np.nan, "x_g": np.nan, "y_g": np.nan, "z_g": np.nan}
    reasons = []
    if coverage < min_coverage or largest_gap > max_gap_s + 1e-9:
        reasons.append("insufficient_samples")
    if len(clean):
        center = np.median(clean, axis=0)
        norm = float(np.linalg.norm(center))
        rms = float(np.sqrt(np.mean(np.sum((clean - center) ** 2, axis=1))))
        spread = float(np.percentile(angle_degrees(clean, center), 95)) if norm > 0 and (np.linalg.norm(clean, axis=1) > 0).all() else np.nan
        data.update(dict(zip(["x_g", "y_g", "z_g"], center)))
        data.update({"norm_g": norm, "rms_g": rms, "spread_deg": spread})
        if not gravity_limits_g[0] <= norm <= gravity_limits_g[1]:
            reasons.append("gravity_scale")
        if rms > max_rms_g:
            reasons.append("dynamic_acceleration")
        if not np.isfinite(spread) or spread > max_spread_deg:
            reasons.append("unstable_direction")
    else:
        reasons.append("no_valid_samples")
    return data, reasons


def classify_reorientation(features, little_angle_deg=10.0, change_angle_deg=30.0,
                           persistence_angle_deg=10.0):
    """Reclassify cached measurements for transparent threshold sensitivity.

    Both post windows must meet a category threshold. The persistence guard
    compares the two post-window centers. Intermediate/Unknown are never folded
    into the little-change control group.
    """
    if (not np.isfinite([little_angle_deg, change_angle_deg, persistence_angle_deg]).all()
            or not 0 <= little_angle_deg < change_angle_deg <= 180
            or not 0 <= persistence_angle_deg <= 180):
        raise ValueError("Require 0 <= little < change <= 180 and a valid persistence angle")
    result = features.copy()
    numeric = result[["pre_post_angle_deg", "pre_hold_angle_deg", "post_drift_deg"]]
    valid = result.window_qc_pass & np.isfinite(numeric).all(axis=1)
    persistent = result.post_drift_deg <= persistence_angle_deg + 1e-9
    usable = valid & persistent
    labels = np.full(len(result), UNKNOWN, dtype=object)
    labels[usable] = INTERMEDIATE
    labels[usable & (numeric[["pre_post_angle_deg", "pre_hold_angle_deg"]].max(axis=1) <= little_angle_deg + 1e-9)] = LITTLE
    labels[usable & (numeric[["pre_post_angle_deg", "pre_hold_angle_deg"]].min(axis=1) >= change_angle_deg - 1e-9)] = CHANGED
    result["orientation_usable"] = usable
    result["reorientation_class"] = pd.Categorical(labels, categories=CLASSES, ordered=True)
    result["orientation_reason"] = result.window_qc_reason
    result.loc[valid & ~persistent, "orientation_reason"] = "post_direction_not_sustained"
    result.loc[usable & (result.reorientation_class == INTERMEDIATE), "orientation_reason"] = "between_angle_thresholds"
    result.attrs.update(features.attrs)
    result.attrs.update({"little_angle_deg": little_angle_deg, "change_angle_deg": change_angle_deg,
                         "persistence_angle_deg": persistence_angle_deg})
    return result


def measure_reorientation(axes_g, bursts, sampling_rate=100.0, window_s=10.0,
                          buffer_s=2.0, hold_s=10.0, min_coverage=0.95,
                          max_gap_s=0.25, max_rms_g=0.05, max_spread_deg=10.0,
                          gravity_limits_g=(0.8, 1.2), little_angle_deg=10.0,
                          change_angle_deg=30.0, persistence_angle_deg=10.0):
    """Measure three stable raw-XYZ windows around each burst.

    Defaults: pre [start-12,start-2), post [end+2,end+12), hold [end+12,end+22).
    Component medians estimate gravity in low-motion windows; raw-vector RMS
    and the 95th percentile angular deviation check stability. Windows with
    inadequate coverage, scale, or another detected burst are rejected. The
    measured change is between pre and the mean of the two unit post vectors.
    Sustained refers ONLY to the observed post windows, not the rest of the night.
    """
    positive = [sampling_rate, window_s, hold_s, max_gap_s, max_rms_g]
    if not np.isfinite(positive).all() or min(positive) <= 0:
        raise ValueError("Sampling rate, window lengths, gap and RMS limits must be positive")
    if not np.isfinite([buffer_s, min_coverage, max_spread_deg]).all() or buffer_s < 0 or not 0 < min_coverage <= 1 or not 0 < max_spread_deg <= 180:
        raise ValueError("Invalid buffer, coverage, or angular-spread setting")
    if not np.isfinite(gravity_limits_g).all() or not 0 < gravity_limits_g[0] < gravity_limits_g[1]:
        raise ValueError("Invalid gravity limits")
    axes = axes_g[["x_g", "y_g", "z_g"]].copy()
    axes.index = pd.DatetimeIndex(pd.to_datetime(axes.index, utc=True))
    if axes.index.hasnans or not axes.index.is_unique or not axes.index.is_monotonic_increasing:
        raise ValueError("XYZ samples must have unique, increasing timestamps")
    if not bursts.index.is_unique:
        raise ValueError("Burst identifiers must be unique")
    starts = pd.DatetimeIndex(pd.to_datetime(bursts.start, utc=True))
    ends = pd.DatetimeIndex(pd.to_datetime(bursts.end, utc=True))
    if starts.hasnans or ends.hasnans or (ends <= starts).any():
        raise ValueError("Every burst needs valid start/end timestamps")
    rows = []
    for i, (start, end) in enumerate(zip(starts, ends)):
        windows = {
            "pre": (start - pd.Timedelta(seconds=buffer_s + window_s), start - pd.Timedelta(seconds=buffer_s)),
            "post": (end + pd.Timedelta(seconds=buffer_s), end + pd.Timedelta(seconds=buffer_s + window_s)),
            "hold": (end + pd.Timedelta(seconds=buffer_s + window_s), end + pd.Timedelta(seconds=buffer_s + window_s + hold_s)),
        }
        row = {"burst_start": start, "burst_end": end}
        reasons, centers = [], {}
        for name, (a, b) in windows.items():
            values, failed = _window_features(axes, a, b, sampling_rate, min_coverage, max_gap_s,
                                               max_rms_g, max_spread_deg, gravity_limits_g)
            row.update({name + "_" + key: value for key, value in values.items()})
            row[name + "_start"], row[name + "_end"] = a, b
            other = (starts < b) & (ends > a)
            other[i] = False
            if other.any():
                failed.append("other_burst")
            reasons.extend(name + ":" + reason for reason in failed)
            centers[name] = np.array([values[key] for key in ("x_g", "y_g", "z_g")])
        post, hold = centers["post"], centers["hold"]
        row["pre_post_angle_deg"] = float(angle_degrees(centers["pre"], post))
        row["pre_hold_angle_deg"] = float(angle_degrees(centers["pre"], hold))
        row["post_drift_deg"] = float(angle_degrees(post, hold))
        if np.isfinite([post, hold]).all() and np.linalg.norm(post) > 0 and np.linalg.norm(hold) > 0:
            combined = post / np.linalg.norm(post) + hold / np.linalg.norm(hold)
            row["tilt_change_deg"] = float(angle_degrees(centers["pre"], combined))
        else:
            row["tilt_change_deg"] = np.nan
        row["window_qc_pass"] = not reasons
        row["window_qc_reason"] = "; ".join(reasons)
        rows.append(row)
    result = pd.DataFrame(rows, index=bursts.index)
    if not len(result):
        result = pd.DataFrame(index=bursts.index, columns=["burst_start", "burst_end", "pre_post_angle_deg", "pre_hold_angle_deg",
                              "post_drift_deg", "tilt_change_deg", "window_qc_pass", "window_qc_reason"])
        result["window_qc_pass"] = result.window_qc_pass.astype(bool)
        for column in ["pre_post_angle_deg", "pre_hold_angle_deg", "post_drift_deg", "tilt_change_deg"]:
            result[column] = result[column].astype(float)
    result.attrs.update({"sampling_rate_hz": sampling_rate, "window_s": window_s, "buffer_s": buffer_s,
                         "hold_s": hold_s, "min_coverage": min_coverage, "max_gap_s": max_gap_s,
                         "max_rms_g": max_rms_g, "max_spread_deg": max_spread_deg,
                         "gravity_limits_g": list(gravity_limits_g)})
    return classify_reorientation(result, little_angle_deg, change_angle_deg, persistence_angle_deg)
