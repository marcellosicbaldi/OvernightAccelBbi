"""Three PPG respiratory surrogates in shared HRV quiet periods.

RIIV = baseline wander, RIAV = pulse amplitude, RIFV = pulse period (RSA proxy).
These are respiratory surrogates, not reference respiration or validated rates.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, periodogram, sosfiltfilt

from analysis_windows import validate_intervals
from gp_hrv import build_variable_windows
from interburst_hrv import burst_free_segments
from polar_ccf import CCFConfig, extract_beats, sensor_clock_index, union_duration_s


METHODS = ("RIIV", "RIAV", "RIFV")
LABELS = {"RIIV": "Baseline wander", "RIAV": "Pulse amplitude", "RIFV": "Pulse period / RSA proxy"}
TIME_COLUMNS = {"RIIV": "pwa_time", "RIAV": "pwa_time", "RIFV": "pp_time"}
COLORS = {"RIIV": "tab:blue", "RIAV": "tab:orange", "RIFV": "tab:green"}


@dataclass(frozen=True)
class RespirationConfig:
    min_burst_duration_s: float = 2
    post_burst_guard_s: float = 1
    include_recording_edges: bool = True  # Matches gp_hrv's complement helper.
    min_window_s: float = 60
    max_window_s: float = 300
    step_s: float = 60
    fs: float = 4
    ppg_fs: float = 55
    polarity: int = -1
    raw_lowpass_hz: float = 8
    max_raw_gap_s: float = .1
    pulse_edge_guard_s: float = 5
    respiratory_band_hz: tuple = (.1, .5)
    filter_order: int = 3
    filter_pad_s: float = 30
    filter_edge_guard_s: float = 10
    max_beat_gap_s: float = 3
    max_bad_fraction: float = .05
    min_beats: int = 30
    peak_half_width_hz: float = .025
    min_peak_power_fraction: float = .2  # Exploratory spectral concentration screen.
    beat_nyquist_safety: float = .9

    def __post_init__(self):
        positive = (self.min_window_s, self.max_window_s, self.step_s, self.fs, self.ppg_fs,
                    self.raw_lowpass_hz, self.max_raw_gap_s, self.max_beat_gap_s,
                    self.filter_pad_s, self.peak_half_width_hz)
        if not all(np.isfinite(x) and x > 0 for x in positive):
            raise ValueError("Frequencies, window lengths, padding and gaps must be finite and positive")
        if not all(np.isfinite(x) and x >= 0 for x in (self.min_burst_duration_s,
                   self.post_burst_guard_s, self.filter_edge_guard_s, self.pulse_edge_guard_s)):
            raise ValueError("Burst durations and guards must be finite and nonnegative")
        low, high = self.respiratory_band_hz
        if not (np.isfinite(low) and np.isfinite(high) and 0 < low < high < self.fs / 2):
            raise ValueError("Respiratory band must lie below resampling Nyquist")
        if (self.min_window_s > self.max_window_s or self.step_s > self.max_window_s
                or self.ppg_fs <= 20 or not high < self.raw_lowpass_hz < self.ppg_fs / 2):
            raise ValueError("Inconsistent windows or waveform frequencies")
        for name in ("filter_order", "min_beats"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < (3 if name == "min_beats" else 1):
                raise ValueError(f"Invalid {name}")
        for name in ("max_bad_fraction", "min_peak_power_fraction"):
            if not np.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if not 0 < self.beat_nyquist_safety <= 1 or self.polarity not in (-1, 1):
            raise ValueError("Invalid beat Nyquist safety factor or waveform polarity")
        if not isinstance(self.include_recording_edges, bool):
            raise ValueError("include_recording_edges must be Boolean")
        for length in (self.min_window_s, self.max_window_s, self.step_s, self.filter_edge_guard_s):
            if not np.isclose(length * self.fs, round(length * self.fs)):
                raise ValueError("Window and edge lengths must align to the resampling grid")


def respiration_quiet_periods(bursts, coverage, config=RespirationConfig()):
    """Match HRV's >=2 s burst / +1 s guard policy and optional quiet edges.

    Coverage boundaries are always barriers. Short movements remain inside GP
    quiet periods, deliberately matching HRV; all original bursts stay available
    for diagnostics. No movement detection is duplicated here.
    """
    coverage = validate_intervals(coverage)
    events = bursts[["start", "end"]].copy()
    for name in ("start", "end"):
        times = pd.DatetimeIndex(events[name])
        if times.hasnans or (len(times) and times.tz is None):
            raise ValueError("Bursts need timezone-aware nonmissing bounds")
        events[name] = pd.to_datetime(times, utc=True)
    if (events.end <= events.start).any():
        raise ValueError("Burst end must follow start")
    retained = (events.end - events.start).dt.total_seconds() >= config.min_burst_duration_s
    blocked = events.loc[retained].copy()
    original_start = blocked.start.min() if len(blocked) else None
    original_end = blocked.end.max() if len(blocked) else None
    blocked["end"] += pd.Timedelta(seconds=config.post_burst_guard_s)
    pieces = []
    for coverage_id, interval in coverage.iterrows():
        a, b = interval.start, interval.end
        if not config.include_recording_edges:
            if len(blocked) < 2:
                continue
            a, b = max(a, original_start), min(b, original_end)
        if b <= a:
            continue
        part = burst_free_segments(blocked, a, b)
        part["coverage_id"] = coverage_id
        pieces.append(part)
    quiet = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        {"start": pd.Series(dtype="datetime64[ns, UTC]"), "end": pd.Series(dtype="datetime64[ns, UTC]"),
         "duration_s": pd.Series(dtype=float), "coverage_id": pd.Series(dtype=int)})
    quiet["long_enough"] = quiet.duration_s >= config.min_window_s
    quiet["geometric_windows"] = [len(_windows(row.start, row.end, config)) for row in quiet.itertuples()]
    quiet.index.name = "quiet_period_id"
    report = {"all_bursts": len(events), "retained_bursts": int(retained.sum()),
              "ignored_short_bursts": int((~retained).sum()), "quiet_periods": len(quiet),
              "quiet_periods_ge_minimum": int(quiet.long_enough.sum()),
              "retained_quiet_min": quiet.loc[quiet.long_enough, "duration_s"].sum() / 60,
              "geometric_windows": int(quiet.geometric_windows.sum())}
    return quiet, report


def _windows(start, end, config):
    return build_variable_windows(start, end, pd.Timedelta(seconds=config.min_window_s),
                                  pd.Timedelta(seconds=config.max_window_s),
                                  pd.Timedelta(seconds=config.step_s))


def derive_features(waveform, beats):
    """Sample BASELINE-PRESERVING PPG at shared detected fiducials.

    BW: mean of the raw peak and preceding foot; AM: their difference. Both
    retain the peak timestamp. FM: onset-to-onset PP in seconds at its midpoint,
    reusing the existing pipeline's interval convention (not ECG RR).
    """
    if not isinstance(waveform, pd.Series) or not isinstance(waveform.index, pd.DatetimeIndex):
        raise ValueError("Waveform needs a datetime-indexed Series")
    sensor_clock_index(waveform.index)
    if waveform.index.tz is None or not np.isfinite(waveform.to_numpy()).all():
        raise ValueError("Waveform must be finite with timezone-aware timestamps")
    result = beats.copy()
    for name in ("pwa_time", "foot_time", "pp_time", "pp_start", "pp_end"):
        times = pd.DatetimeIndex(result[name])
        if (times.hasnans or times.tz is None or not times.is_unique
                or not times.is_monotonic_increasing):
            raise ValueError("Fiducial times must be valid, timezone-aware, unique and increasing")
        if len(times) and (times[0] < waveform.index[0] or times[-1] > waveform.index[-1]):
            raise ValueError("Fiducials must lie within observed waveform support")
    midpoint = result.pp_start + (result.pp_end - result.pp_start) / 2
    if ((result.foot_time >= result.pwa_time).any() or (result.pp_start >= result.pp_end).any()
            or (result.foot_time != result.pp_end).any() or (result.pp_time != midpoint).any()
            or not np.allclose((result.pp_end - result.pp_start).dt.total_seconds(), result.pp_s)):
        raise ValueError("Inconsistent pulse pairing or PP intervals")
    origin = waveform.index[0]
    seconds = (waveform.index - origin).total_seconds().to_numpy()
    for name, time_column in (("raw_peak", "pwa_time"), ("raw_foot", "foot_time")):
        t = (pd.DatetimeIndex(result[time_column]) - origin).total_seconds()
        result[name] = np.interp(t, seconds, waveform.to_numpy())
    result["RIIV"] = (result.raw_peak + result.raw_foot) / 2
    result["RIAV"] = result.raw_peak - result.raw_foot
    result["RIFV"] = result.pp_s
    result["valid"] = (result.valid & np.isfinite(result[list(METHODS)]).all(axis=1)
                       & (result.RIAV > 0) & (result.RIFV > 0))
    return result


def extract_period_features(ppg, config=RespirationConfig()):
    """Use shared pulse detection, but never its high-passed values for BW/AM."""
    beat_config = CCFConfig(pp_min_s=.3, pp_max_s=2)
    beats, pulse_waveform, quality = extract_beats(
        ppg, fs=config.ppg_fs, config=beat_config, polarity=config.polarity,
        max_raw_gap_s=config.max_raw_gap_s, edge_guard_s=config.pulse_edge_guard_s)
    source_time = (ppg.index - ppg.index[0]).total_seconds().to_numpy()
    grid_time = (pulse_waveform.index - ppg.index[0]).total_seconds().to_numpy()
    raw = config.polarity * np.interp(grid_time, source_time, ppg.to_numpy(dtype=float))
    # Remove only high-frequency waveform noise. DC/respiratory baseline is kept.
    sos = butter(3, config.raw_lowpass_hz, fs=config.ppg_fs, output="sos")
    preserved = pd.Series(sosfiltfilt(sos, raw), index=pulse_waveform.index)
    return derive_features(preserved, beats), quality


def respiratory_runs(features, config=RespirationConfig()):
    """Common clean support for fair comparisons; no long-gap filtering/bridging.

    Short masked gaps <= max_beat_gap_s are linearly interpolated and reported.
    Break if either PP or amplitude timestamps exceed that gap. The three
    modulations keep their distinct timestamps before common 4 Hz resampling.
    """
    good = features.loc[features.valid].copy()
    if len(good) < 2:
        return
    jumps = np.zeros(len(good) - 1, dtype=bool)
    for time_column in set(TIME_COLUMNS.values()):
        jumps |= np.diff(pd.DatetimeIndex(good[time_column]).asi8) / 1e9 > config.max_beat_gap_s
    boundaries = np.r_[0, np.flatnonzero(jumps) + 1, len(good)]
    band_sos = butter(config.filter_order, config.respiratory_band_hz, btype="bandpass", fs=config.fs, output="sos")
    for first, stop in zip(boundaries[:-1], boundaries[1:]):
        part = good.iloc[first:stop]
        if len(part) < 2:
            continue
        start = max(part.pp_time.iloc[0], part.pwa_time.iloc[0])
        end = min(part.pp_time.iloc[-1], part.pwa_time.iloc[-1])
        duration_s = (end - start).total_seconds()
        guard = int(round(config.filter_edge_guard_s * config.fs))
        count = int(np.floor(duration_s * config.fs)) + 1
        if count <= 2 * guard + 1:
            continue
        grid = start + pd.to_timedelta(np.arange(count) / config.fs, unit="s")
        raw = pd.DataFrame(index=grid)
        for method in METHODS:
            times = (pd.DatetimeIndex(part[TIME_COLUMNS[method]]) - start).total_seconds()
            raw[method] = np.interp(np.arange(count) / config.fs, times, part[method])
        padding = min(round(config.filter_pad_s * config.fs), len(raw) - 1)
        # Remove means before filtering to reduce floating-point cancellation for DC-rich BW.
        values = raw.to_numpy() - raw.to_numpy().mean(axis=0)
        filtered = pd.DataFrame(sosfiltfilt(band_sos, values, axis=0, padtype="odd", padlen=padding),
                                columns=METHODS, index=grid)
        keep = slice(guard, -guard if guard else None)
        yield raw.iloc[keep], filtered.iloc[keep]


def rate_spectrum(values, config=RespirationConfig()):
    """Hann periodogram of UNFILTERED, linearly detrended modulation.

    Frequency-bin resolution = 60/window duration breaths/min; no zero padding.
    The separate respiratory bandpass is for waveform visualization, not for
    selecting the rate peak, so its transfer function does not favor band center.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < config.min_window_s * config.fs or not np.isfinite(values).all():
        raise ValueError("Need a complete finite respiratory window")
    centered = detrend(values - values.mean(), type="linear")
    sd = float(np.std(centered))
    threshold = np.finfo(float).eps * max(1., np.max(abs(values))) * 100
    if sd <= threshold:
        raise ValueError("no_modulation_variance")
    frequency, power = periodogram(centered, fs=config.fs, window="hann", detrend=False)
    low, high = config.respiratory_band_hz
    band = (frequency >= low - 1e-12) & (frequency <= high + 1e-12)
    f, p = frequency[band], power[band]
    if len(f) < 3 or p.sum() <= 0:
        raise ValueError("insufficient_respiratory_spectrum")
    peak = int(np.argmax(p))
    near = abs(f - f[peak]) <= config.peak_half_width_hz + 1e-12
    metrics = {"candidate_rate_bpm": float(f[peak] * 60), "peak_hz": float(f[peak]),
               "peak_power_fraction": float(p[near].sum() / p.sum()),
               "peak_at_band_boundary": peak in (0, len(f) - 1),
               "modulation_sd": sd, "resolution_bpm": config.fs / len(values) * 60}
    return metrics, pd.DataFrame({"frequency_hz": f, "power": p})


def _window_support(features, start, end, config):
    """Original beat rows bracketing the actual resampling support of a window."""
    features = features.reset_index(drop=True)
    good = features.loc[features.valid]
    first, last, max_gap = len(features), -1, 0.
    last_grid = end - pd.Timedelta(seconds=1 / config.fs)
    for time_column in set(TIME_COLUMNS.values()):
        times = pd.DatetimeIndex(good[time_column])
        left = max(0, times.searchsorted(start, side="right") - 1)
        stop = min(len(times), times.searchsorted(last_grid, side="left") + 1)
        part = good.iloc[left:stop]
        if part.empty:
            continue
        first, last = min(first, int(part.index[0])), max(last, int(part.index[-1]))
        max_gap = max(max_gap, float((np.diff(pd.DatetimeIndex(part[time_column]).asi8) / 1e9).max(initial=0)))
    return features.iloc[first:last + 1], max_gap


def analyze_respiration(ppg, quiet, config=RespirationConfig(), feature_extractor=None):
    """Extract/filter independently in each quiet period and continuous beat run.

    Short and rejected periods stay in diagnostics. No period is concatenated,
    and windows never span a motion boundary or long beat gap. Method-specific
    quality flags are retained alongside candidate and quality-screened rates.
    """
    if not isinstance(ppg, pd.Series):
        raise ValueError("PPG must be a Series")
    ppg = ppg.copy()
    ppg.index = sensor_clock_index(ppg.index)
    if not np.isfinite(ppg.to_numpy()).all():
        raise ValueError("PPG contains nonfinite samples")
    validated = validate_intervals(quiet)
    if not validated[["start", "end"]].equals(quiet.reset_index(drop=True)[["start", "end"]]):
        raise ValueError("Quiet periods must be sorted nonoverlapping UTC bounds")
    extractor = feature_extractor or extract_period_features
    periods = quiet.copy().reset_index(drop=True)
    periods.index.name = "quiet_period_id"
    all_features, run_frames, spectrum_frames, records = [], [], [], []
    next_window_id = 0
    for quiet_id, period in periods.iterrows():
        periods.loc[quiet_id, "status"] = "shorter_than_minimum_window"
        periods.loc[quiet_id, "signal_runs"] = 0
        periods.loc[quiet_id, "candidate_windows"] = 0
        if (period.end - period.start).total_seconds() < config.min_window_s:
            continue
        raw = ppg.loc[(ppg.index >= period.start) & (ppg.index < period.end)]
        try:
            features, _ = extractor(raw, config)
        except ValueError as error:
            periods.loc[quiet_id, "status"] = f"pulse_extraction_failed: {error}"
            continue
        features["quiet_period_id"] = quiet_id
        all_features.append(features)
        periods.loc[quiet_id, "n_beats"] = len(features)
        periods.loc[quiet_id, "bad_fraction"] = 1 - features.valid.mean()
        periods.loc[quiet_id, "status"] = "insufficient_continuous_clean_support"
        for raw_run, filtered_run in respiratory_runs(features, config):
            run_id = len(run_frames)
            period_windows = _windows(raw_run.index[0], raw_run.index[-1] + pd.Timedelta(seconds=1 / config.fs), config)
            if not period_windows:
                continue
            periods.loc[quiet_id, "signal_runs"] += 1
            periods.loc[quiet_id, "candidate_windows"] += len(period_windows)
            periods.loc[quiet_id, "status"] = "analyzed"
            run_frame = raw_run.add_suffix("_raw").join(filtered_run.add_suffix("_resp"))
            run_frame["quiet_period_id"], run_frame["signal_run_id"] = quiet_id, run_id
            run_frames.append(run_frame.rename_axis("time").reset_index())
            for start, end in period_windows:
                window_id = next_window_id
                next_window_id += 1
                support, max_gap = _window_support(features, start, end, config)
                good = support.loc[support.valid]
                bad_fraction = 1 - support.valid.mean() if len(support) else 1.
                median_pp = float(good.pp_s.median()) if len(good) else np.nan
                window = raw_run.loc[(raw_run.index >= start) & (raw_run.index < end)]
                common_reasons = []
                if len(good) < config.min_beats:
                    common_reasons.append("too_few_beats")
                if bad_fraction > config.max_bad_fraction + 1e-12:
                    common_reasons.append("too_many_flagged_beats")
                for method in METHODS:
                    row = {"window_id": window_id, "quiet_period_id": quiet_id, "signal_run_id": run_id,
                           "method": method, "start": start, "end": end,
                           "duration_s": (end - start).total_seconds(), "n_grid_samples": len(window),
                           "n_beats_support": len(support), "n_bad_beats_support": int((~support.valid).sum()),
                           "bad_fraction": bad_fraction, "max_beat_gap_s": max_gap, "median_pp_s": median_pp,
                           "beat_nyquist_hz_approx": .5 / median_pp,
                           "candidate_rate_bpm": np.nan, "rate_bpm": np.nan,
                           "peak_power_fraction": np.nan, "included": False}
                    reasons = common_reasons.copy()
                    try:
                        metrics, spectrum = rate_spectrum(window[method], config)
                        row.update(metrics)
                        if metrics["peak_power_fraction"] < config.min_peak_power_fraction:
                            reasons.append("weak_spectral_concentration")
                        if metrics["peak_at_band_boundary"]:
                            reasons.append("peak_at_search_boundary")
                        if metrics["peak_hz"] >= config.beat_nyquist_safety * row["beat_nyquist_hz_approx"]:
                            reasons.append("peak_near_or_above_beat_nyquist")
                        spectrum["window_id"], spectrum["method"] = window_id, method
                        spectrum_frames.append(spectrum)
                    except ValueError as error:
                        reasons.append(str(error))
                    row["included"] = not reasons
                    if row["included"]:
                        row["rate_bpm"] = row["candidate_rate_bpm"]
                    row["exclusion_reason"] = ";".join(reasons)
                    records.append(row)
    columns = ["window_id", "quiet_period_id", "signal_run_id", "method", "start", "end", "duration_s",
               "n_grid_samples", "n_beats_support", "n_bad_beats_support", "bad_fraction", "max_beat_gap_s",
               "median_pp_s", "beat_nyquist_hz_approx", "candidate_rate_bpm", "rate_bpm", "peak_hz",
               "peak_power_fraction", "peak_at_band_boundary", "modulation_sd", "resolution_bpm",
               "included", "exclusion_reason"]
    windows = pd.DataFrame(records).reindex(columns=columns)
    windows["included"] = windows.included.astype(bool)
    for name in ("start", "end"):
        windows[name] = pd.to_datetime(windows[name], utc=True)
    signals = pd.concat(run_frames, ignore_index=True) if run_frames else pd.DataFrame(
        columns=["time", *[m + suffix for suffix in ("_raw", "_resp") for m in METHODS],
                 "quiet_period_id", "signal_run_id"])
    spectra = pd.concat(spectrum_frames, ignore_index=True) if spectrum_frames else pd.DataFrame(
        columns=["frequency_hz", "power", "window_id", "method"])
    period_rates = windows.loc[windows.included].groupby(["quiet_period_id", "method"], as_index=False).agg(
        rate_bpm=("rate_bpm", "median"), n_windows=("window_id", "size"))
    comparison = windows.pivot(index="window_id", columns="method", values="rate_bpm").reindex(columns=METHODS)
    comparison["all_methods_included"] = comparison.notna().all(axis=1)
    return {"periods": periods, "features": pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame(),
            "signals": signals, "windows": windows, "spectra": spectra,
            "period_rates": period_rates, "comparison": comparison, "config": asdict(config)}


def respiration_availability(result):
    rows = []
    for method in METHODS:
        windows = result["windows"].loc[result["windows"].method == method]
        included = windows.loc[windows.included]
        period_rates = result["period_rates"].loc[result["period_rates"].method == method]
        rows.append({"method": method, "candidate_windows": len(windows), "included_windows": len(included),
                     "contributing_quiet_periods": included.quiet_period_id.nunique(),
                     "unique_covered_min": union_duration_s(included) / 60,
                     "summed_window_min_counts_overlap": included.duration_s.sum() / 60,
                     "median_of_quiet_period_rates_bpm": period_rates.rate_bpm.median()})
    return pd.DataFrame(rows)
