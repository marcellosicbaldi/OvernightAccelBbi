"""Per-window HR-adaptive low-pass extraction of RIIV directly from raw PPG.

Custom filtering experiment, not a reproduction of a published RR algorithm.
Pulse intervals provide HR only; the respiratory waveform is not beat-sampled.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, periodogram, sosfiltfilt, sosfreqz

from polar_ccf import sensor_clock_index
from ppg_respiration import METHODS, RespirationConfig, rate_spectrum


DIRECT_METHOD = "RIIV_direct"


@dataclass(frozen=True)
class DirectRIIVConfig:
    cutoff_hr_fraction: float = .7
    hr_quantile: float = .1
    max_cutoff_hz: float = 1.
    order: int = 8  # per pass; forward/reverse gives an effective order of 16
    context_s: float = 30
    peak_cutoff_fraction: float = .85
    min_cardiac_attenuation_db: float = 30

    def __post_init__(self):
        if not 0 < self.cutoff_hr_fraction < 1 or not 0 <= self.hr_quantile <= .5:
            raise ValueError("Need cutoff below HR and a lower-half HR quantile")
        if not 0 < self.peak_cutoff_fraction < 1:
            raise ValueError("Peak cutoff fraction must be in (0,1)")
        if not all(np.isfinite(x) and x > 0 for x in (self.max_cutoff_hz, self.context_s)):
            raise ValueError("Maximum cutoff and context must be finite and positive")
        if (not isinstance(self.order, int) or isinstance(self.order, bool) or self.order < 1
                or not np.isfinite(self.min_cardiac_attenuation_db) or self.min_cardiac_attenuation_db < 0):
            raise ValueError("Invalid filter order or attenuation threshold")


def direct_riiv_window(ppg, start, end, quiet_start, quiet_end, hr_bpm,
                       config=RespirationConfig(), direct=DirectRIIVConfig()):
    """Return native-filtered 4 Hz waveforms, metadata, and native PPG spectra.

    Clip context to THIS quiet period, regularize at the native working rate,
    center, low-pass in SOS forward/reverse form, THEN downsample to 4 Hz.
    Odd-reflection padding never reads another quiet period. Filter the same
    4 Hz context in the respiratory band for display, then crop the window.
    Overlapping windows can have different filters and stay separate in output.
    """
    if not isinstance(ppg, pd.Series) or len(ppg) < 2:
        raise ValueError("Need raw datetime-indexed PPG")
    sensor_clock_index(ppg.index)
    if ppg.index.tz is None:
        raise ValueError("Raw PPG must have timezone-aware timestamps")
    for bound in (start, end, quiet_start, quiet_end):
        if pd.isna(bound) or pd.Timestamp(bound).tzinfo is None:
            raise ValueError("Need timezone-aware window and quiet bounds")
    if not quiet_start <= start < end <= quiet_end:
        raise ValueError("Window must lie entirely in one quiet period")
    hr = np.asarray(hr_bpm, dtype=float)
    if hr.ndim != 1 or len(hr) < 3 or not np.isfinite(hr).all() or (hr <= 0).any():
        raise ValueError("Need at least three finite positive within-window HR values")
    if not direct.max_cutoff_hz < config.fs / 2:
        raise ValueError("Maximum direct cutoff must be below the 4 Hz resampling Nyquist")
    cardiac_hz = float(np.quantile(hr, direct.hr_quantile) / 60)
    if cardiac_hz >= config.ppg_fs / 2:
        raise ValueError("Cardiac reference exceeds native working-rate Nyquist")
    cutoff = min(direct.cutoff_hr_fraction * cardiac_hz, direct.max_cutoff_hz)
    if not 0 < cutoff < config.ppg_fs / 2:
        raise ValueError("Invalid adaptive low-pass cutoff")
    if cutoff * direct.peak_cutoff_fraction <= config.respiratory_band_hz[0]:
        raise ValueError("no_separated_respiratory_band")
    context = pd.Timedelta(seconds=direct.context_s)
    a, b = max(quiet_start, start - context), min(quiet_end, end + context)
    raw = ppg.loc[(ppg.index >= a) & (ppg.index < b)]
    if len(raw) < 2 or not np.isfinite(raw.to_numpy()).all():
        raise ValueError("missing_or_nonfinite_raw_ppg")
    elapsed = (raw.index - raw.index[0]).total_seconds().to_numpy()
    max_gap = float(np.diff(elapsed).max())
    if max_gap > config.max_raw_gap_s:
        raise ValueError("raw_ppg_gap_in_filter_context")
    t = np.arange(int(np.floor(elapsed[-1] * config.ppg_fs)) + 1) / config.ppg_fs
    native_times = raw.index[0] + pd.to_timedelta(t, unit="s")
    values = config.polarity * np.interp(t, elapsed, raw.to_numpy(dtype=float))
    offset = float(values.mean())
    values -= offset
    sos = butter(direct.order, cutoff, fs=config.ppg_fs, output="sos")
    padding = min(round(direct.context_s * config.ppg_fs), len(values) - 1)
    lowpass = sosfiltfilt(sos, values, padtype="odd", padlen=padding)
    # A common grid anchored to the existing window, bracketed by native samples.
    left = int(np.floor((start - native_times[0]).total_seconds() * config.fs))
    right = int(np.floor((native_times[-1] - start).total_seconds() * config.fs))
    grid_seconds = np.arange(-left, right + 1) / config.fs
    grid = start + pd.to_timedelta(grid_seconds, unit="s")
    native_seconds = (native_times - start).total_seconds().to_numpy()
    if grid[0] < native_times[0] or grid[-1] > native_times[-1]:
        raise ValueError("unbracketed_raw_grid")
    sampled = np.interp(grid_seconds, native_seconds, lowpass)
    band = butter(config.filter_order, config.respiratory_band_hz, btype="bandpass", fs=config.fs, output="sos")
    display_values = sosfiltfilt(band, sampled - sampled.mean(), padtype="odd",
                               padlen=min(round(config.filter_pad_s * config.fs), len(sampled) - 1))
    keep = (grid >= start) & (grid < end)
    frame = pd.DataFrame({"time": grid[keep], "RIIV_direct_raw": sampled[keep],
                          "RIIV_direct_resp": display_values[keep]})
    expected = round((end - start).total_seconds() * config.fs)
    if len(frame) != expected or frame.empty or frame.time.iloc[0] != start:
        raise ValueError("raw_ppg_does_not_bracket_complete_window")
    _, gain = sosfreqz(sos, worN=[cardiac_hz], fs=config.ppg_fs)
    attenuation = float(-40 * np.log10(max(abs(gain[0]), np.finfo(float).tiny)))
    native_keep = (native_times >= start) & (native_times < end)
    f, before = periodogram(values[native_keep], fs=config.ppg_fs, window="hann", detrend="linear")
    _, after = periodogram(lowpass[native_keep], fs=config.ppg_fs, window="hann", detrend="linear")
    shown = f <= min(4, config.ppg_fs / 2)
    spectra = pd.DataFrame({"frequency_hz": f[shown], "raw_power": before[shown], "lowpass_power": after[shown]})
    metadata = {"hr_median_bpm": float(np.median(hr)), "hr_filter_reference_bpm": cardiac_hz * 60,
                "n_hr_intervals": len(hr), "lowpass_cutoff_hz": cutoff,
                "cardiac_attenuation_db_theoretical": attenuation, "raw_context_start": raw.index[0],
                "raw_context_end": raw.index[-1], "raw_max_gap_s": max_gap,
                "raw_optical_offset_removed": offset,
                "respiratory_flat_band_max_hz": direct.peak_cutoff_fraction * cutoff}
    return frame, metadata, spectra


def add_direct_riiv(result, ppg, direct=DirectRIIVConfig()):
    """Append a direct-RIIV estimate on EXACTLY the existing candidate windows.

    This comparison inherits window availability from beat-derived methods; it
    does not claim to measure all raw-data-only coverage. Pulse flags assess HR
    reliability. A beat-sampling Nyquist rejection is not applied to native PPG.
    Re-running replaces this variant rather than duplicating its rows.
    """
    config = RespirationConfig(**result["config"])
    base = result["windows"].loc[result["windows"].method.isin(METHODS)].copy()
    windows = base.drop_duplicates("window_id")
    methods = (*METHODS, DIRECT_METHOD)
    features = result["features"]
    rows, signals, spectra, native_spectra = [], [], [], []
    for source in windows.to_dict("records"):
        row = source.copy()
        row.update(method=DIRECT_METHOD, candidate_rate_bpm=np.nan, rate_bpm=np.nan, peak_hz=np.nan,
                   peak_power_fraction=np.nan, peak_at_band_boundary=False, modulation_sd=np.nan,
                   included=False, exclusion_reason="", beat_nyquist_hz_approx=np.nan)
        for name in ("hr_median_bpm", "hr_filter_reference_bpm", "n_hr_intervals", "lowpass_cutoff_hz",
                     "cardiac_attenuation_db_theoretical", "raw_max_gap_s", "raw_optical_offset_removed",
                     "respiratory_flat_band_max_hz"):
            row[name] = np.nan
        reasons = []
        period = result["periods"].loc[source["quiet_period_id"]]
        selected = features.loc[(features.quiet_period_id == source["quiet_period_id"])
                                & (features.pp_time >= source["start"]) & (features.pp_time < source["end"])]
        good = selected.loc[selected.valid]
        if len(good) < config.min_beats:
            reasons.append("too_few_hr_intervals")
        if source["bad_fraction"] > config.max_bad_fraction + 1e-12:
            reasons.append("too_many_flagged_beats_for_hr")
        try:
            frame, metadata, native = direct_riiv_window(
                ppg, source["start"], source["end"], period.start, period.end,
                good.hr_bpm.to_numpy(), config, direct)
            row.update(metadata)
            for table in (frame, native):
                table["window_id"] = source["window_id"]
                table["quiet_period_id"] = source["quiet_period_id"]
            signals.append(frame)
            native_spectra.append(native)
            metrics, spectrum = rate_spectrum(frame.RIIV_direct_raw, config)
            row.update(metrics)
            if metadata["cardiac_attenuation_db_theoretical"] < direct.min_cardiac_attenuation_db:
                reasons.append("insufficient_cardiac_attenuation")
            if metrics["peak_hz"] >= metadata["respiratory_flat_band_max_hz"]:
                reasons.append("respiratory_peak_near_lowpass_cutoff")
            if metrics["peak_at_band_boundary"]:
                reasons.append("peak_at_search_boundary")
            if metrics["peak_power_fraction"] < config.min_peak_power_fraction:
                reasons.append("weak_spectral_concentration")
            spectrum["window_id"] = source["window_id"]
            spectrum["quiet_period_id"] = source["quiet_period_id"]
            spectrum["method"] = DIRECT_METHOD
            spectra.append(spectrum)
        except ValueError as error:
            reasons.append(str(error))
        row["included"] = not reasons
        row["rate_bpm"] = row["candidate_rate_bpm"] if row["included"] else np.nan
        row["exclusion_reason"] = ";".join(reasons)
        rows.append(row)
    out = dict(result)
    out["methods"], out["direct_config"] = methods, asdict(direct)
    out["windows"] = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    out["spectra"] = pd.concat([result["spectra"].loc[result["spectra"].method.isin(METHODS)],
                                *spectra], ignore_index=True)
    out["direct_signals"] = pd.concat(signals, ignore_index=True) if signals else pd.DataFrame(
        columns=["time", "RIIV_direct_raw", "RIIV_direct_resp", "window_id", "quiet_period_id"])
    out["direct_native_spectra"] = pd.concat(native_spectra, ignore_index=True) if native_spectra else pd.DataFrame(
        columns=["frequency_hz", "raw_power", "lowpass_power", "window_id", "quiet_period_id"])
    out["period_rates"] = out["windows"].loc[out["windows"].included].groupby(
        ["quiet_period_id", "method"], as_index=False).agg(rate_bpm=("rate_bpm", "median"), n_windows=("window_id", "size"))
    out["comparison"] = out["windows"].pivot(index="window_id", columns="method", values="rate_bpm").reindex(columns=methods)
    out["comparison"]["all_feature_methods_included"] = out["comparison"][list(METHODS)].notna().all(axis=1)
    out["comparison"]["all_methods_included"] = out["comparison"][list(methods)].notna().all(axis=1)
    return out
