"""Quiet-period Polar HR/PP versus PWA CCF; see polar_quiet_ccf.ipynb.

No ECG RR or blood pressure is inferred. All times share the Polar sensor clock.
Only complete windows are analyzed; recording edges are not interburst periods.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from analysis_windows import intersect_intervals, validate_intervals
from gp_hrv_artifacts import _find_artifacts
from interburst_hrv import burst_free_segments
from ppg_fiducials import get_peak_onset


@dataclass(frozen=True)
class CCFConfig:
    window_s: float = 300
    step_s: float = 60
    fs: float = 4
    max_lag_s: float = 25
    lowpass_hz: float = .15
    filter_order: int = 10
    filter_pad_s: float = 30
    pre_burst_s: float = 2
    post_burst_s: float = 5
    max_beat_gap_s: float = 3
    min_beats: int = 100
    max_bad_fraction: float = 0
    pp_min_s: float = .3
    pp_max_s: float = 2

    def __post_init__(self):
        positive = (self.window_s, self.step_s, self.fs, self.max_lag_s,
                    self.lowpass_hz, self.max_beat_gap_s, self.pp_min_s, self.pp_max_s)
        if not all(np.isfinite(x) and x > 0 for x in positive):
            raise ValueError("CCF lengths, frequencies, gaps and PP bounds must be positive")
        if (self.step_s > self.window_s or self.max_lag_s >= self.window_s
                or self.lowpass_hz >= self.fs / 2 or self.pp_min_s >= self.pp_max_s):
            raise ValueError("Inconsistent window, lag, cutoff or PP bounds")
        if any(not np.isclose(x * self.fs, round(x * self.fs))
               for x in (self.window_s, self.step_s, self.max_lag_s, self.filter_pad_s)):
            raise ValueError("Lengths must represent integer sample counts")
        if (not isinstance(self.filter_order, int) or isinstance(self.filter_order, bool)
                or self.filter_order < 1 or not isinstance(self.min_beats, int)
                or isinstance(self.min_beats, bool) or self.min_beats < 3):
            raise ValueError("Need positive integer order and min_beats >=3")
        if not all(np.isfinite(x) and x >= 0 for x in
                   (self.pre_burst_s, self.post_burst_s, self.filter_pad_s)):
            raise ValueError("Buffers and padding must be finite and nonnegative")
        if self.filter_pad_s >= self.window_s or not 0 <= self.max_bad_fraction <= 1:
            raise ValueError("Padding must be shorter than window; bad fraction must be in [0,1]")


def sensor_clock_index(index, timezone="UTC"):
    """Attach a declared zone to naive sensor timestamps; do not change offsets.

    UTC is a bookkeeping label for the loader's naive sensor clock. It does not
    establish agreement with civil time. Use the same choice for PPG and ACC.
    """
    index = pd.DatetimeIndex(index)
    if index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Sensor times must be present, unique and increasing")
    return index.tz_localize(timezone).tz_convert("UTC") if index.tz is None else index.tz_convert("UTC")


def quiet_periods(bursts, coverage, config=CCFConfig()):
    """Buffered complement strictly BETWEEN bursts, intersected with coverage.

    Reuses the shared complement helper, with asymmetric expansion performed
    first. Overlapping expanded bursts merge. Separate acquisition/state bounds
    stay separate; no recording-edge intervals are admitted.
    """
    coverage = validate_intervals(coverage)
    events = bursts[["start", "end"]].copy()
    for name in ("start", "end"):
        times = pd.DatetimeIndex(events[name])
        if times.hasnans or (len(times) and times.tz is None):
            raise ValueError("Burst bounds must be timezone-aware and present")
        events[name] = pd.to_datetime(times, utc=True)
    events = events.sort_values("start")
    if (events.end <= events.start).any():
        raise ValueError("Each burst must end after it starts")
    pieces = []
    if len(events) >= 2:
        outer_start, outer_end = events.start.min(), events.end.max()
        events["start"] -= pd.Timedelta(seconds=config.pre_burst_s)
        events["end"] += pd.Timedelta(seconds=config.post_burst_s)
        gaps = burst_free_segments(events, outer_start, outer_end)
        pieces = intersect_intervals(gaps, coverage)
    else:
        pieces = validate_intervals(pd.DataFrame(columns=["start", "end"]))
    pieces["duration_s"] = (pieces.end - pieces.start).dt.total_seconds()
    pieces["long_enough"] = pieces.duration_s >= config.window_s
    pieces["geometric_windows"] = np.where(
        pieces.long_enough, np.floor(np.maximum(0, pieces.duration_s - config.window_s)
                                     / config.step_s).astype(int) + 1, 0)
    pieces.index.name = "quiet_period_id"
    return pieces


def extract_beats(ppg, *, fs=55, config=CCFConfig(), polarity=-1,
                  max_raw_gap_s=.1, edge_guard_s=5):
    """Reuse polar.ipynb's 0.5-8 Hz/order-3 preprocessing and paired fiducials.

    Regularize the waveform at fs on its original clock before detection.
    PP is onset-to-onset, timestamped at the interval MIDPOINT. PWA is the
    filtered peak minus its paired foot, timestamped at that PEAK. Retain these
    distinct physical times for independent interpolation onto one 4 Hz grid.
    No invented first PP, rolling mean, peak correction, or clock shift is used.
    Interval classifier flags are retained, never repaired. PPG PP is not ECG RR.
    """
    if not isinstance(ppg, pd.Series) or not isinstance(ppg.index, pd.DatetimeIndex):
        raise ValueError("PPG needs a datetime-indexed Series")
    index = sensor_clock_index(ppg.index)
    values = ppg.to_numpy(dtype=float)
    if (len(values) < fs * 10 or not np.isfinite(values).all() or np.ptp(values) == 0
            or not np.isfinite(fs) or fs <= 20 or polarity not in (-1, 1)
            or not np.isfinite(max_raw_gap_s) or max_raw_gap_s <= 0
            or not np.isfinite(edge_guard_s) or edge_guard_s < 0):
        raise ValueError("Need >=10 s finite, nonconstant PPG and valid extraction settings")
    elapsed = (index - index[0]).total_seconds().to_numpy()
    gaps = np.diff(elapsed)
    if gaps.max() > max_raw_gap_s:
        raise ValueError("Raw PPG gap exceeds limit; select continuous acquisition runs")
    t = np.arange(int(np.floor(elapsed[-1] * fs)) + 1) / fs
    grid = index[0] + pd.to_timedelta(t, unit="s")
    regular = np.interp(t, elapsed, values)
    sos = butter(3, [.5, 8], btype="bandpass", fs=fs, output="sos")
    filtered = pd.Series(polarity * sosfiltfilt(sos, regular), index=grid, name="ppg_filtered")
    peaks, feet = get_peak_onset(filtered.to_numpy(), fs)
    if len(peaks) < 3:
        raise ValueError("Too few paired pulses; inspect channel, polarity and waveform")
    # PP for pulse i spans foot[i-1] to foot[i]; PWA belongs to pulse i.
    pp = np.diff(grid[feet].asi8) / 1e9
    midpoint = grid[feet[:-1]] + (grid[feet[1:]] - grid[feet[:-1]]) / 2
    amplitude = filtered.iloc[peaks].to_numpy() - filtered.iloc[feet].to_numpy()
    morphology = (feet < peaks) & np.r_[peaks[:-1] < feet[1:], True] & (amplitude > 0)
    artifact = np.zeros(len(pp), dtype=bool)
    flags, _ = _find_artifacts(pp.copy())
    for indices in flags.values():
        artifact[np.asarray(indices, dtype=int)] = True
    valid = ((pp >= config.pp_min_s) & (pp <= config.pp_max_s)
             & morphology[:-1] & morphology[1:] & ~artifact)
    edge = ((grid[feet[:-1]] >= grid[0] + pd.Timedelta(seconds=edge_guard_s))
            & (grid[peaks[1:]] <= grid[-1] - pd.Timedelta(seconds=edge_guard_s)))
    beats = pd.DataFrame({
        "pp_time": midpoint, "pp_start": grid[feet[:-1]], "pp_end": grid[feet[1:]],
        "pwa_time": grid[peaks[1:]], "foot_time": grid[feet[1:]],
        "pp_s": pp, "hr_bpm": 60 / pp, "pwa": amplitude[1:],
        "artifact": artifact, "valid": valid & edge,
    })
    quality = {"ppg_samples": len(ppg), "ppg_observed_median_hz": 1 / np.median(gaps),
               "ppg_max_gap_s": gaps.max(), "regularization_hz": fs,
               "paired_pulses": len(peaks), "beat_rows": len(beats),
               "bad_beat_rows": int((~beats.valid).sum()), "polarity": polarity,
               "edge_guard_s": edge_guard_s,
               "pwa_peak_minus_foot_median_s": float(np.median((peaks - feet) / fs)),
               "hr_pp_time_assignment": "onset interval midpoint; PWA at paired peak"}
    return beats, filtered, quality


def normalized_ccf(x, y, *, fs=4, max_lag_s=25):
    """corr(X(t), Y(t+lag)); positive lag means Y follows X.

    Subtract full-window means and divide each lagged dot product by the full
    window energy product (MATLAB xcorr 'coeff' style). Use only overlapping
    pairs in the numerator, no circular wrapping, per-lag re-centering, or
    N/(N-|lag|) correction. Autocorrelation at zero lag is exactly one.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (x.ndim != 1 or x.shape != y.shape or not np.isfinite(x).all()
            or not np.isfinite(y).all() or not np.isfinite(fs) or fs <= 0
            or not np.isfinite(max_lag_s) or max_lag_s <= 0
            or not np.isclose(fs * max_lag_s, round(fs * max_lag_s))):
        raise ValueError("Need paired finite vectors and valid sample-aligned lags")
    limit = int(round(fs * max_lag_s))
    if len(x) <= limit:
        raise ValueError("Window must exceed maximum lag")
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if not np.isfinite(denominator) or denominator == 0:
        raise ValueError("Constant or numerically degenerate signal")
    lags = np.arange(-limit, limit + 1)
    curve = [np.dot(x[:len(x) - k], y[k:]) if k >= 0
             else np.dot(x[-k:], y[:len(y) + k]) for k in lags]
    return lags / fs, np.asarray(curve) / denominator


def lowpass(values, config=CCFConfig()):
    """10th order PER PASS, forward/reverse SOS, zero phase, odd reflection.

    Default 30 s reflection padding uses only this window. Double passing
    squares the magnitude response (effective order 20; -6 dB at 0.15 Hz).
    Padding reduces but cannot eliminate finite-window edge effects.
    """
    sos = butter(config.filter_order, config.lowpass_hz, fs=config.fs, output="sos")
    return sosfiltfilt(sos, values, padtype="odd", padlen=int(config.filter_pad_s * config.fs), axis=0)


def _inside(beats, period):
    # Both ends of the measured interval and the paired PWA peak must be quiet.
    return beats.loc[(beats.pp_start >= period.start) & (beats.pp_end < period.end)
                     & (beats.pwa_time < period.end)].copy()


def _sample_window(beats, start, config):
    grid = start + pd.to_timedelta(np.arange(round(config.window_s * config.fs)) / config.fs, unit="s")
    data = {}
    max_gap = 0.0
    for name, time_column in (("hr_bpm", "pp_time"), ("pp_s", "pp_time"), ("pwa", "pwa_time")):
        times = pd.DatetimeIndex(beats[time_column])
        if not times.is_monotonic_increasing or not times.is_unique:
            raise ValueError("Beat times must be unique and increasing for interpolation")
        left = max(0, times.searchsorted(grid[0], side="right") - 1)
        stop = min(len(times), times.searchsorted(grid[-1], side="left") + 1)
        times, values = times[left:stop], beats[name].to_numpy()[left:stop]
        if len(times) < 2 or times[0] > grid[0] or times[-1] < grid[-1]:
            raise ValueError("unbracketed_window")
        seconds = (times - grid[0]).total_seconds().to_numpy()
        max_gap = max(max_gap, float(np.diff(seconds).max()))
        data[name] = np.interp(np.arange(len(grid)) / config.fs, seconds, values)
    raw = pd.DataFrame(data, index=grid)
    return raw, pd.DataFrame(lowpass(raw.to_numpy(), config), index=grid, columns=raw.columns), max_gap


def analyze_quiet_ccf(beats, periods, config=CCFConfig()):
    """Return all candidate-window QC, eligible curves and aggregation tables.

    Window starts are anchored to common within-period beat support, not to an
    extrapolated edge. Geometric (movement-only) availability remains reported.
    Invalid beats are never repaired. Defaults reject any window touching a
    flagged beat; optional relaxed QC still prohibits long interpolation gaps.
    """
    required = {"pp_time", "pp_start", "pp_end", "pwa_time", "pp_s", "hr_bpm", "pwa", "valid"}
    if not required.issubset(beats):
        raise ValueError("Need the paired beat table from extract_beats")
    beats = beats.copy()
    for name in ("pp_time", "pp_start", "pp_end", "pwa_time"):
        times = pd.DatetimeIndex(beats[name])
        if times.hasnans or times.tz is None or not times.is_monotonic_increasing or not times.is_unique:
            raise ValueError("Beat times must be timezone-aware, unique and increasing")
    if (beats.pp_end <= beats.pp_start).any():
        raise ValueError("PP interval must have positive duration")
    if not np.isfinite(beats[["pp_s", "hr_bpm", "pwa"]].to_numpy()).all():
        raise ValueError("Beat values must be finite; mark physiological failures with valid=False")
    if not np.allclose(beats.hr_bpm * beats.pp_s, 60):
        raise ValueError("HR and PP must come from the same intervals")
    bounds = validate_intervals(periods)
    period_table = periods.copy().reset_index(drop=True)
    period_table.index.name = "quiet_period_id"
    # Require the supplied IDs/order to agree with validated, nonoverlapping bounds.
    if not bounds[["start", "end"]].equals(period_table[["start", "end"]]):
        raise ValueError("Quiet periods must be sorted, nonoverlapping UTC bounds")
    records, curves = [], []
    duration, step = pd.Timedelta(seconds=config.window_s), pd.Timedelta(seconds=config.step_s)
    for quiet_id, period in period_table.iterrows():
        part = _inside(beats, period)
        good = part.loc[part.valid]
        if len(good) < 2:
            period_table.loc[quiet_id, "signal_duration_s"] = 0
            period_table.loc[quiet_id, "candidate_windows"] = 0
            continue
        start = max(period.start, good.pp_time.iloc[0], good.pwa_time.iloc[0])
        end = min(period.end, good.pp_time.iloc[-1] + pd.Timedelta(seconds=1 / config.fs),
                  good.pwa_time.iloc[-1] + pd.Timedelta(seconds=1 / config.fs))
        period_table.loc[quiet_id, "signal_duration_s"] = max(0, (end - start).total_seconds())
        period_table.loc[quiet_id, "candidate_windows"] = 0
        current = start
        while current + duration <= end:
            window_id = len(records)
            period_table.loc[quiet_id, "candidate_windows"] += 1
            # Include bad observations used as bracketing support by either series.
            used = part.loc[(part.pwa_time >= current - pd.Timedelta(seconds=config.max_beat_gap_s))
                            & (part.pp_time <= current + duration)]
            n_bad = int((~used.valid).sum())
            bad_fraction = n_bad / len(used) if len(used) else 1.
            row = {"window_id": window_id, "quiet_period_id": quiet_id,
                   "start": current, "end": current + duration,
                   "n_beats": len(used), "n_bad_beats": n_bad,
                   "bad_fraction": bad_fraction, "max_interpolation_gap_s": np.nan,
                   "n_grid_samples": round(config.window_s * config.fs),
                   "included": False, "exclusion_reason": ""}
            reasons = []
            if len(used.loc[used.valid]) < config.min_beats:
                reasons.append("too_few_beats")
            if bad_fraction > config.max_bad_fraction:
                reasons.append("flagged_beats")
            try:
                raw, filtered, gap = _sample_window(good, current, config)
                row["max_interpolation_gap_s"] = gap
                for name in raw:
                    row[f"{name}_mean"] = raw[name].mean()
                    row[f"{name}_sd"] = raw[name].std(ddof=0)
                    row[f"{name}_lowpass_sd"] = filtered[name].std(ddof=0)
                if gap > config.max_beat_gap_s:
                    reasons.append("interpolation_gap")
                staged = []
                for mode, signals in (("unfiltered", raw), ("lowpass", filtered)):
                    for signal in ("hr_bpm", "pp_s"):
                        lags, ccf = normalized_ccf(signals[signal], signals.pwa, fs=config.fs,
                                                   max_lag_s=config.max_lag_s)
                        prefix = f"{signal}_{mode}"
                        row.update(extrema(ccf, lags, prefix))
                        staged.extend({"window_id": window_id, "quiet_period_id": quiet_id,
                                       "signal": signal, "mode": mode, "lag_s": lag, "ccf": c}
                                      for lag, c in zip(lags, ccf))
                if not reasons:
                    row["included"] = True
                    curves.extend(staged)
            except ValueError as error:
                reasons.append(str(error))
            # Excluded estimates must not be confused with usable coefficients.
            if reasons:
                for key in list(row):
                    if "_r_" in key or "_lag_" in key:
                        row[key] = np.nan
            row["exclusion_reason"] = ";".join(dict.fromkeys(reasons))
            records.append(row)
            current += step
    columns = ["window_id", "quiet_period_id", "start", "end", "n_beats", "n_bad_beats",
               "bad_fraction", "max_interpolation_gap_s", "n_grid_samples", "included", "exclusion_reason"]
    for signal in ("hr_bpm", "pp_s", "pwa"):
        columns.extend(f"{signal}_{stat}" for stat in ("mean", "sd", "lowpass_sd"))
    for signal in ("hr_bpm", "pp_s"):
        for mode in ("unfiltered", "lowpass"):
            columns.extend(f"{signal}_{mode}_{metric}" for metric in
                           ("r_min", "lag_min_s", "r_max", "lag_max_s",
                            "min_at_lag_boundary", "max_at_lag_boundary"))
    windows = pd.DataFrame(records).reindex(columns=columns)
    for name in ("start", "end"):
        windows[name] = pd.to_datetime(windows[name], utc=True)
    windows["included"] = windows.included.astype(bool)
    curve_table = pd.DataFrame(curves, columns=["window_id", "quiet_period_id", "signal", "mode", "lag_s", "ccf"])
    summaries, period_curves = aggregate_curves(curve_table)
    return {"periods": period_table, "windows": windows, "curves": curve_table,
            "summary_curves": summaries, "period_curves": period_curves,
            "config": asdict(config)}


def extrema(ccf, lags, prefix=""):
    """Global min/max, including endpoints; tied extrema use first (lowest) lag."""
    prefix = prefix + "_" if prefix else ""
    a, b = int(np.argmin(ccf)), int(np.argmax(ccf))
    return {f"{prefix}r_min": ccf[a], f"{prefix}lag_min_s": lags[a],
            f"{prefix}r_max": ccf[b], f"{prefix}lag_max_s": lags[b],
            f"{prefix}min_at_lag_boundary": a in (0, len(ccf) - 1),
            f"{prefix}max_at_lag_boundary": b in (0, len(ccf) - 1)}


def aggregate_curves(curves):
    """Average windows within periods first; never use window SEM for inference."""
    keys = ["signal", "mode", "lag_s"]
    period_curves = curves.groupby(["quiet_period_id", *keys], as_index=False).ccf.mean()
    tables = []
    for unit, frame in (("overlapping_windows_descriptive", curves), ("quiet_periods", period_curves)):
        out = frame.groupby(keys).ccf.agg(mean="mean", sd="std", n="count").reset_index()
        out["sem"] = out.sd / np.sqrt(out.n)  # NaN for n=1, never pretend SEM=0.
        out["aggregation"] = unit
        tables.append(out)
    return pd.concat(tables, ignore_index=True), period_curves


def union_duration_s(windows):
    if windows.empty:
        return 0.
    pairs = windows.sort_values("start")[["start", "end"]].itertuples(index=False, name=None)
    merged = []
    for a, b in pairs:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    return sum((b - a).total_seconds() for a, b in merged)


def availability(result):
    periods, windows = result["periods"], result["windows"]
    accepted = windows.loc[windows.included]
    return pd.Series({
        "interburst_periods": len(periods),
        "periods_at_least_one_full_window": int(periods.long_enough.sum()),
        "retained_quiet_duration_min": periods.loc[periods.long_enough, "duration_s"].sum() / 60,
        "movement_only_geometric_windows": int(periods.geometric_windows.sum()),
        "signal_supported_candidate_windows": len(windows),
        "eligible_windows": len(accepted),
        "distinct_contributing_quiet_periods": accepted.quiet_period_id.nunique(),
        "eligible_window_union_min": union_duration_s(accepted) / 60,
        "eligible_window_sum_min_counts_overlap": len(accepted) * result["config"]["window_s"] / 60,
        "participants": 1,
    }, name="availability")


def representative_signals(beats, result, window_id=None):
    windows = result["windows"].loc[result["windows"].included]
    if windows.empty:
        return None
    # A deterministic quality-based choice, never chosen for a strong CCF.
    row = (windows.sort_values(["bad_fraction", "max_interpolation_gap_s", "start"]).iloc[0]
           if window_id is None else windows.loc[windows.window_id == window_id].iloc[0])
    part = _inside(beats, result["periods"].loc[row.quiet_period_id])
    raw, filtered, _ = _sample_window(part.loc[part.valid], row.start, CCFConfig(**result["config"]))
    return row, raw, filtered


def plot_overview(detection, result, selected_window_ids=()):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(15, 8))
    for part in detection["signals"]:
        for ax, name in zip(axes[:2], ("magnitude", "score")):
            series = part[name].iloc[::10]
            ax.plot(series.index, series, lw=.5, color=".3")
        axes[1].axhline(part["threshold"], color="tab:red", ls=":", lw=.8)
    for burst in detection["bursts"].itertuples():
        for ax in axes[:2]:
            ax.axvspan(burst.start, burst.end, color="tab:red", alpha=.3)
    for quiet_id, period in result["periods"].iterrows():
        if period.long_enough:
            axes[2].broken_barh([(period.start, period.end - period.start)], (.65, .25), color="tab:green", alpha=.4)
    accepted = result["windows"].loc[result["windows"].included]
    for row in accepted.itertuples():
        y = (row.window_id % 4) * .12
        axes[2].plot([row.start, row.end], [y, y], color="tab:blue", alpha=.35, lw=2)
        if row.window_id in selected_window_ids:
            for ax in axes:
                ax.axvspan(row.start, row.end, color="gold", alpha=.3)
    axes[0].set(ylabel="Magnitude (g)", title="Polar whole recording; sleep unverified")
    axes[1].set(ylabel="Movement envelope (g)")
    axes[2].set(yticks=[.2, .78], yticklabels=["Eligible windows", "Quiet ≥5 min"], ylim=(-.1, 1),
                xlabel="Polar sensor time (declared UTC clock)")
    axes[0].legend(handles=[Patch(color="tab:red", alpha=.3, label="Movement bursts"),
                            Patch(color="gold", alpha=.3, label="Selected representative window")])
    fig.tight_layout()
    return fig


def plot_representative(representative):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(13, 8))
    if representative is None:
        axes[0].text(.5, .5, "No eligible complete window", ha="center", transform=axes[0].transAxes)
    else:
        row, raw, filtered = representative
        for name, color in (("hr_bpm", "tab:blue"), ("pwa", "tab:red"), ("pp_s", "tab:green")):
            for ax, frame in zip(axes[:2], (raw, filtered)):
                z = (frame[name] - frame[name].mean()) / frame[name].std(ddof=0)
                ax.plot((frame.index - row.start).total_seconds(), z, lw=.9, color=color, label=name)
        axes[0].set_title(f"Representative window {row.window_id}, quiet period {row.quiet_period_id}: {row.start}")
        axes[0].legend(ncol=3)
        axes[1].set_title("Low-pass <0.15 Hz, zero phase; each displayed series standardized separately")
        axes[2].plot((raw.index - row.start).total_seconds(), raw.pwa, color="tab:red", lw=.7)
        axes[2].set_title("PWA in original amplitude units (no visualization scaling)")
    axes[0].set_ylabel("Unfiltered (z)")
    axes[1].set_ylabel("Low-pass (z)")
    axes[2].set(ylabel="PWA (a.u.)", xlabel="Seconds within window")
    fig.tight_layout()
    return fig


def plot_ccf(result):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(13, 8))
    summary = result["summary_curves"]
    for row, aggregation in enumerate(("overlapping_windows_descriptive", "quiet_periods")):
        for col, signal in enumerate(("hr_bpm", "pp_s")):
            ax = axes[row, col]
            for mode, color in (("unfiltered", "tab:blue"), ("lowpass", "tab:red")):
                frame = summary.loc[(summary.aggregation == aggregation) & (summary.signal == signal)
                                    & (summary["mode"] == mode)].sort_values("lag_s")
                if frame.empty:
                    continue
                n = int(frame.n.iloc[0])
                ax.plot(frame.lag_s, frame["mean"], color=color, label=f"{mode}, n={n}")
                ax.fill_between(frame.lag_s, frame["mean"] - frame["sem"],
                                frame["mean"] + frame["sem"], color=color, alpha=.18)
            ax.axhline(0, color=".5", lw=.7)
            ax.axvline(0, color=".5", lw=.7, ls="--")
            ax.set(title=f"{signal} / PWA — " + ("window mean ± SEM (descriptive)" if row == 0
                                              else "equal-period mean ± SEM"), ylim=(-1, 1))
            if not summary.empty:
                ax.legend(fontsize=9)
            else:
                ax.text(.5, .5, "No eligible windows", ha="center", transform=ax.transAxes)
    for ax in axes[-1]:
        ax.set_xlabel("Lag τ (s): positive = PWA follows HR or PP")
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized CCF")
    fig.suptitle("No inferential tests on overlapping windows; periods share one recording/participant")
    fig.tight_layout()
    return fig


def plot_extrema(result):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    frame = result["windows"].loc[result["windows"].included]
    for row, signal in enumerate(("hr_bpm", "pp_s")):
        for col, metric in enumerate(("r_min", "r_max", "lag_min_s", "lag_max_s")):
            ax = axes[row, col]
            for mode, color in (("unfiltered", "tab:blue"), ("lowpass", "tab:red")):
                if frame.empty:
                    continue
                values = frame[f"{signal}_{mode}_{metric}"]
                limits = (-1, 1) if metric.startswith("r_") else (-result["config"]["max_lag_s"], result["config"]["max_lag_s"])
                ax.hist(values, bins=np.linspace(*limits, 17), histtype="step", color=color, label=mode, lw=1.5)
            ax.set(title=f"{signal}: {metric}", ylabel="Windows (overlapping)")
            ax.set_xlabel("Correlation" if metric.startswith("r_") else "Lag (s)")
    if not frame.empty:
        axes[0, 0].legend()
    fig.suptitle("Window extrema are descriptive; boundary extrema are flagged in CSV")
    fig.tight_layout()
    return fig
