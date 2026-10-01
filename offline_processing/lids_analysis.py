"""Garmin envelope adaptation of LIDS, with reference projection cosine fitting.

Reference: https://github.com/marcellosicbaldi/lids-analysis (cosine_fit.py).
Summed minute-mean envelope in mg is not calibrated ZCM activity counts; amplitudes/MRI are specific
to this adaptation. Inputs must already be cropped to sleep_period_time.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from detect_acc_bursts import bandpass_acceleration, compute_envelope, prepare_acceleration


@dataclass(frozen=True)
class LIDSConfig:
    timezone: str = "Europe/Rome"
    input_unit: str = "mg"
    sampling_rate: float = 100.0
    min_coverage: float = 0.90
    max_gap_s: float = 0.25
    period_min: float = 30.0
    period_max: float = 180.0
    period_step: float = 5.0
    fit_method: str = "projection"

    def __post_init__(self):
        if self.input_unit not in ("g", "mg"):
            raise ValueError("input_unit must be g or mg")
        if not np.isfinite(self.sampling_rate) or self.sampling_rate <= 20:
            raise ValueError("sampling_rate must exceed 20 Hz for the band-pass filter")
        if not np.isfinite(self.max_gap_s) or self.max_gap_s <= 0:
            raise ValueError("max_gap_s must be positive and finite")
        if not 0 < self.min_coverage <= 1:
            raise ValueError("min_coverage must be in (0, 1]")
        bounds = [self.period_min, self.period_max, self.period_step]
        if not np.isfinite(bounds).all() or not 20 < self.period_min <= self.period_max or self.period_step <= 0:
            raise ValueError("Periods must exceed the 20-minute Nyquist limit; invalid search bounds")
        if self.fit_method not in ("projection", "least_squares"):
            raise ValueError("fit_method must be projection or least_squares")


def minute_activity(acc: pd.DataFrame, config=LIDSConfig()):
    """Average the burst-analysis envelope each minute, in mg.

    The shared burst helpers regularize magnitude, band-pass 0.1–10 Hz (order
    8), then subtract lower from upper envelopes (groups of ten extrema).
    Process continuous finite runs separately; never filter across invalid XYZ
    or gaps > max_gap_s. Envelopes are continuous scores, without thresholding.
    Minute coverage requires both observed unique and processed regular samples.
    """
    columns = ["accel_x", "accel_y", "accel_z"]
    times = pd.DatetimeIndex(pd.to_datetime(acc["sample_time"]))
    if times.hasnans or len(times) < 2:
        raise ValueError("Need at least two nonmissing acceleration timestamps")
    times = (times.tz_localize(config.timezone, ambiguous="raise", nonexistent="raise")
             if times.tz is None else times.tz_convert(config.timezone))
    source = acc[columns].copy()
    source["sample_time"] = times
    source = source.sort_values("sample_time", kind="stable").reset_index(drop=True)
    times = pd.DatetimeIndex(source.sample_time)
    unique_times = times.drop_duplicates()
    if len(unique_times) < 2:
        raise ValueError("Need at least two distinct timestamps")
    delta_s = np.diff(unique_times.asi8) / 1e9
    observed_hz = 1 / np.median(delta_s)
    if not np.isclose(observed_hz, config.sampling_rate, rtol=0.25):
        raise ValueError(f"Median sampling rate {observed_hz:.1f} Hz disagrees with configured {config.sampling_rate:g} Hz")
    finite = np.isfinite(source[columns].to_numpy(dtype=float)).all(axis=1)
    # Invalidate every duplicate at an invalid timestamp, rather than hiding it
    # by averaging with the remaining finite rows.
    if not finite.all():
        finite &= ~times.isin(times[~finite])
    gap_before = np.r_[True, np.diff(times.asi8) / 1e9 > config.max_gap_s]
    starts = np.flatnonzero(finite & (gap_before | ~np.r_[False, finite[:-1]]))
    ends = np.flatnonzero(finite & (np.r_[gap_before[1:], True] | ~np.r_[finite[1:], False]))
    start = times[0]
    end = times[-1] + pd.Timedelta(seconds=1 / config.sampling_rate)
    minute_index = pd.date_range(start, periods=int((times[-1] - start).total_seconds() // 60) + 1, freq="1min")
    observed = pd.Series(1, index=times[finite].drop_duplicates()).resample("1min", origin=start).sum()
    totals, skipped = [], 0
    for first, last in zip(starts, ends):
        part = source.iloc[first:last + 1]
        if part.sample_time.nunique() < 2:
            skipped += 1
            continue
        magnitude, _ = prepare_acceleration(part, config.sampling_rate, config.input_unit, config.max_gap_s)
        if len(magnitude) <= 51:
            skipped += 1
            continue
        envelope_mg = 1000 * compute_envelope(bandpass_acceleration(magnitude, config.sampling_rate))
        groups = envelope_mg.resample("1min", origin=start)
        totals.append(pd.DataFrame({"sum": groups.sum(), "processed_samples": groups.count()}))
    if totals:
        aggregated = pd.concat(totals).groupby(level=0).sum().reindex(minute_index, fill_value=0)
    else:
        aggregated = pd.DataFrame(0., index=minute_index, columns=["sum", "processed_samples"])
    minutes = pd.DataFrame(index=minute_index)
    minutes.index.name = "timestamp"
    minutes["observed_samples"] = observed.reindex(minute_index, fill_value=0)
    minutes["processed_samples"] = aggregated.processed_samples
    minutes["coverage"] = np.minimum(minutes.observed_samples, minutes.processed_samples) / (60 * config.sampling_rate)
    minutes["valid"] = minutes.coverage >= config.min_coverage
    minutes["envelope_mg"] = (aggregated["sum"] / aggregated.processed_samples.where(aggregated.processed_samples > 0)).where(minutes.valid)
    minutes["external_min"] = (minutes.index - start).total_seconds() / 60
    qc = {"spt_start": start.isoformat(), "spt_end": end.isoformat(),
          "duration_min": (end - start).total_seconds() / 60,
          "duplicate_samples": int(times.duplicated().sum()), "median_sampling_hz": observed_hz,
          "max_sample_gap_s": float(delta_s.max()), "acquisition_runs": len(starts),
          "runs_too_short": skipped, "minutes_total": len(minutes),
          "minutes_valid": int(minutes.valid.sum()),
          "sample_coverage": float(minutes.observed_samples.sum() / ((end - start).total_seconds() * config.sampling_rate))}
    return minutes, qc


def activity_to_lids(minutes):
    """Sum ten valid one-minute means, transform, then smooth three bins.

    All ten minutes must pass coverage; missing/partial bins stay missing.
    Sum of minute-mean mg values is an activity score, not calibrated counts.
    """
    groups = minutes.resample("10min", origin=minutes.index[0])
    bins = pd.DataFrame({"activity_sum_mg": groups.envelope_mg.sum(min_count=10),
                         "valid_minutes": groups.envelope_mg.count(),
                         "coverage": groups.coverage.sum() / 10})
    bins["valid"] = bins.valid_minutes == 10
    bins["lids_raw"] = (100 / (1 + bins.activity_sum_mg)).where(bins.valid)
    bins["lids"] = np.nan
    run_ids = (~bins.valid).cumsum()
    for _, run in bins.loc[bins.valid].groupby(run_ids[bins.valid]):
        bins.loc[run.index, "lids"] = run.lids_raw.rolling(3, center=True, min_periods=1).mean()
    bins["external_min"] = (bins.index - minutes.index[0]).total_seconds() / 60
    return bins


def acceleration_to_lids(acc: pd.DataFrame, config=LIDSConfig()):
    """Convenience wrapper returning 10-minute LIDS and acquisition quality."""
    minutes, qc = minute_activity(acc, config)
    bins = activity_to_lids(minutes)
    qc.update(bins_total=len(bins), bins_valid=int(bins.valid.sum()))
    return bins, qc


def fit_lids(values, times_min, config=LIDSConfig()):
    """Scan 30:5:180 min for maximum MRI = 2 * amplitude * Pearson r.

    Default projection exactly follows the referenced analysis: a=2 mean(y cos),
    b=2 mean(y sin), offset=mean(y). Model = offset + A*cos(2*pi*t/P - phase).
    Least-squares is an explicit alternative for noninteger cycles/missing bins;
    the projection basis is not orthogonal there and R2 can be negative.
    No p-value is reported: smoothing and period selection preclude interpreting
    a naive Pearson test as a test of rhythmicity.
    """
    y, t = np.asarray(values, dtype=float), np.asarray(times_min, dtype=float)
    if y.ndim != 1 or y.shape != t.shape:
        raise ValueError("values and times_min must be equal-length 1D arrays")
    mask = np.isfinite(y) & np.isfinite(t)
    yy, tt = y[mask], t[mask]
    empty = {key: np.nan for key in ("period_min", "amplitude", "phase_rad", "phase_deg",
             "peak_time_min", "offset", "pearson_r", "r2", "rmse", "mri")}
    empty.update(status="insufficient_data", period_at_boundary=False)
    if len(yy) < 6 or np.ptp(tt) < 60:
        return empty, np.full(y.shape, np.nan)
    if np.allclose(yy, yy[0], atol=1e-10, rtol=1e-10):
        empty["status"] = "constant_signal"
        return empty, np.full(y.shape, np.nan)
    periods = np.arange(config.period_min, config.period_max + 1e-9, config.period_step)
    best, best_curve = None, None
    for period in periods:
        angle = 2 * np.pi * tt / period
        co, si = np.cos(angle), np.sin(angle)
        if config.fit_method == "projection":
            a, b, offset = 2 * np.mean(yy * co), 2 * np.mean(yy * si), yy.mean()
        else:
            design = np.column_stack([co, si, np.ones(len(tt))])
            coefficients, _, rank, _ = np.linalg.lstsq(design, yy, rcond=None)
            if rank < 3:
                continue
            a, b, offset = coefficients
        amplitude = float(np.hypot(a, b))
        fitted = a * co + b * si + offset
        if np.allclose(fitted, fitted[0], atol=1e-10, rtol=1e-10):
            continue
        r = float(np.corrcoef(yy, fitted)[0, 1])
        mri = 2 * amplitude * r
        if not np.isfinite(mri) or (best is not None and mri <= best["mri"]):
            continue
        phase = float(np.arctan2(b, a))
        error = yy - fitted
        best = {"status": "ok", "period_min": float(period), "amplitude": amplitude,
                "phase_rad": phase, "phase_deg": float(np.degrees(phase)),
                "peak_time_min": float((phase % (2 * np.pi)) * period / (2 * np.pi)),
                "offset": float(offset), "pearson_r": r,
                "r2": float(1 - np.sum(error**2) / np.sum((yy - yy.mean())**2)),
                "rmse": float(np.sqrt(np.mean(error**2))), "mri": float(mri),
                "period_at_boundary": bool(period == periods[0] or period == periods[-1])}
        full_angle = 2 * np.pi * t / period
        best_curve = a * np.cos(full_angle) + b * np.sin(full_angle) + offset
    if best is None:
        empty["status"] = "no_finite_fit"
        return empty, np.full(y.shape, np.nan)
    return best, best_curve


def internal_profile(bouts, summaries):
    """Interpolate observed LIDS at t_int=t_ext/optPer*90, every 10 min.

    Equal weight per available night, no phase alignment, amplitude scaling,
    extrapolation, zero-padding, or interpolation across missing bins.
    Grid labels 0..350 are the 36 ten-minute positions in four 90-min cycles.
    """
    grid = np.arange(0.0, 360.0, 10.0)
    normalized = pd.DataFrame(index=pd.Index(grid, name="internal_min"))
    for night, bins in bouts.items():
        row = summaries.loc[night]
        if row.status != "ok":
            continue
        t = bins.external_min.to_numpy() / row.period_min * 90
        y = bins.lids.to_numpy()
        result = np.full(len(grid), np.nan)
        valid = np.isfinite(y)
        starts = np.flatnonzero(valid & ~np.r_[False, valid[:-1]])
        ends = np.flatnonzero(valid & ~np.r_[valid[1:], False])
        for start, end in zip(starts, ends):
            inside = (grid >= t[start]) & (grid <= t[end])
            result[inside] = np.interp(grid[inside], t[start:end+1], y[start:end+1])
        normalized[night] = result
    profile = pd.DataFrame({"mean": normalized.mean(axis=1),
                            "sd": normalized.std(axis=1, ddof=1),
                            "n_nights": normalized.count(axis=1)})
    profile["sem"] = profile.sd / np.sqrt(profile.n_nights.where(profile.n_nights > 0))
    profile["cycle"] = (profile.index // 90 + 1).astype(int)
    cycles = []
    for cycle in range(1, 5):
        part = normalized.loc[(grid >= (cycle - 1) * 90) & (grid < cycle * 90)]
        cycles.append({"cycle": cycle, "n_any": int(part.notna().any().sum()),
                       "n_complete": int(part.notna().all().sum()),
                       "min_n_per_bin": int(part.count(axis=1).min()),
                       "max_n_per_bin": int(part.count(axis=1).max())})
    return normalized, profile, pd.DataFrame(cycles).set_index("cycle")


def analyze_nights(root, config=LIDSConfig(), progress=None):
    """Read one crop at a time. Keep excluded/unreadable nights in the summary."""
    paths = sorted(Path(root).glob("*/accelerometer_samples.parquet"))
    if not paths:
        raise FileNotFoundError(f"No */accelerometer_samples.parquet under {root}")
    bouts, minute_bouts, rows = {}, {}, []
    for path in paths:
        if progress:
            progress(f"LIDS: {path.parent.name}")
        row = {"night": path.parent.name, "activity_metric": "sum_10_minute_mean_envelope_mg",
               "fit_method": config.fit_method}
        try:
            acc = pd.read_parquet(path, columns=["sample_time", "accel_x", "accel_y", "accel_z"])
            minutes, qc = minute_activity(acc, config)
            bins = activity_to_lids(minutes)
            qc.update(bins_total=len(bins), bins_valid=int(bins.valid.sum()))
            minute_bouts[path.parent.name] = minutes
            del acc
            fit, bins["fitted"] = fit_lids(bins.lids, bins.external_min, config)
            row.update(qc)
            row.update(fit)
            bouts[path.parent.name] = bins
        except (ValueError, OSError, KeyError, TypeError) as error:
            row.update(status="input_error", error=str(error))
        rows.append(row)
    summaries = pd.DataFrame(rows).set_index("night")
    normalized, profile, cycles = internal_profile(bouts, summaries)
    return {"bouts": bouts, "minutes": minute_bouts, "summary": summaries, "normalized": normalized,
            "profile": profile, "cycles": cycles, "config": config}


def plot_night(night, bins, row):
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter

    fig, ax = plt.subplots(figsize=(11, 4.3), layout="constrained")
    ax.plot(bins.index, bins.lids_raw, color="#b6c4cc", lw=1, label="10-min LIDS")
    ax.plot(bins.index, bins.lids, color="#137d8d", lw=2, label="30-min smoothed")
    if row.status == "ok":
        dense_min = np.linspace(bins.external_min.iloc[0], bins.external_min.iloc[-1], 800)
        dense_time = bins.index[0] + pd.to_timedelta(dense_min, unit="min")
        curve = row.offset + row.amplitude * np.cos(2 * np.pi * dense_min / row.period_min - row.phase_rad)
        ax.plot(dense_time, curve, "--", color="#c56432", lw=1.7, label="Cosine fit")
        detail = (f"Period {row.period_min:.0f} min | amplitude {row.amplitude:.2f} | "
                  f"phase {row.phase_deg:.1f} deg | r {row.pearson_r:.2f} | "
                  f"R² {row.r2:.2f} | MRI {row.mri:.2f}")
        if row.period_at_boundary:
            detail += " | boundary period"
    else:
        detail = f"Fit unavailable: {row.status}"
    ax.set_title(f"{night}  ·  Garmin envelope–LIDS\n{detail}", fontsize=11, loc="left")
    ax.set_ylabel("LIDS (higher = less movement)")
    ax.set_xlabel(f"Local time ({bins.index.tz}) · {row.fit_method} fit")
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M", tz=bins.index.tz))
    ax.grid(alpha=0.18)
    ax.legend(loc="best", fontsize=9)
    return fig


def plot_average(result):
    import matplotlib.pyplot as plt

    profile = result["profile"]
    x = profile.index.to_numpy()
    fig, (ax, counts) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                    height_ratios=[3, 1], layout="constrained")
    ax.plot(x, result["normalized"], color="#c5d5d8", alpha=0.5, lw=0.7)
    ax.fill_between(x, profile["mean"] - profile["sem"], profile["mean"] + profile["sem"],
                    color="#137d8d", alpha=0.20, label="Mean ± SEM across nights")
    ax.plot(x, profile["mean"], color="#137d8d", lw=2.5, label="Equal-night mean")
    ax.set_title("Average Garmin envelope–LIDS · four internal cycles", loc="left", pad=38)
    ax.set_ylabel("LIDS (higher = less movement)")
    ax.legend(fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False)
    counts.step(x, profile.n_nights, where="mid", color="#137d8d")
    counts.set_ylabel("Nights")
    counts.set_ylim(bottom=0)
    counts.set_xlabel("Internal time (min) = elapsed time / fitted period × 90")
    counts.set_xticks(np.arange(0, 361, 30))
    counts.set_xlim(0, 360)
    for axis in (ax, counts):
        for boundary in [90, 180, 270, 360]:
            axis.axvline(boundary, color="#879399", lw=0.8, ls=":")
        axis.grid(alpha=0.15)
    for cycle in range(4):
        ax.text((cycle + 0.5) / 4, 0.97, f"Cycle {cycle + 1}", transform=ax.transAxes,
                ha="center", va="top", fontsize=9)
    return fig


def save_results(result, output):
    """Save private tables, settings, nightly PNGs and average PNG/SVG."""
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for key in ("summary", "normalized", "profile", "cycles"):
        result[key].to_csv(output / f"lids_{key}.csv")
    (output / "settings.json").write_text(json.dumps({
        **asdict(result["config"]), "activity_metric": "sum_10_minute_mean_envelope_mg",
        "minute_epoch_seconds": 60, "ten_minute_aggregation": "sum (10 valid minutes required)",
        "activity_unit": "sum of minute-mean envelope mg",
        "bandpass_hz": [0.1, 10], "filter_order": 8, "envelope_extrema_group": 10,
        "bin_minutes": 10, "smoothing_minutes": 30,
        "internal_cycle_minutes": 90, "max_cycles": 4,
        "reference": "https://github.com/marcellosicbaldi/lids-analysis",
        "phase_convention": "offset + amplitude*cos(2*pi*t/period - phase_rad)",
    }, indent=2), encoding="utf-8")
    for night, bins in result["bouts"].items():
        result["minutes"][night].to_csv(output / f"{night}_minute_activity.csv")
        bins.to_csv(output / f"{night}_lids.csv")
        fig = plot_night(night, bins, result["summary"].loc[night])
        fig.savefig(output / f"{night}_lids.png", dpi=160)
        plt.close(fig)
    fig = plot_average(result)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"lids_average.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Folder containing nightly SPT crop folders")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs" / "lids_envelope")
    parser.add_argument("--fit-method", choices=["projection", "least_squares"], default="projection")
    args = parser.parse_args()
    result = analyze_nights(args.root, LIDSConfig(fit_method=args.fit_method), progress=print)
    save_results(result, args.output)
    print(result["summary"].to_string())
    print(result["cycles"].to_string())
    print(f"Results: {args.output.resolve()}")
    if not result["summary"].status.eq("ok").any():
        raise SystemExit("No nights yielded a fit; inspect lids_summary.csv")


if __name__ == "__main__":
    main()
