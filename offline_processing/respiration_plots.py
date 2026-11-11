"""Plotting for respiration.ipynb; no extraction or quality decisions here."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ppg_respiration import COLORS, LABELS, METHODS
from direct_riiv import DIRECT_METHOD


def representative_window(result, window_id=None):
    windows = result["windows"]
    if windows.empty:
        return None
    ranking = windows.groupby("window_id").agg(n_included=("included", "sum"),
                                               bad_fraction=("bad_fraction", "max"),
                                               start=("start", "first"))
    if window_id is None:
        # No selection on inter-method agreement or particular respiratory rate.
        window_id = ranking.sort_values(["n_included", "bad_fraction", "start"],
                                        ascending=[False, True, True]).index[0]
    selected = windows.loc[windows.window_id == window_id]
    if selected.empty:
        raise ValueError("Unknown representative window ID")
    return selected


def plot_recording(detection, result):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(15, 8))
    for part in detection["signals"]:
        series = part["magnitude"].iloc[::10]
        axes[0].plot(series.index, series, color=".35", lw=.5)
    for burst in detection["bursts"].itertuples():
        axes[0].axvspan(burst.start, burst.end, color="tab:red", alpha=.3)
    for quiet in result["periods"].itertuples():
        if quiet.long_enough:
            axes[1].axvspan(quiet.start, quiet.end, color="tab:blue", alpha=.15)
    for run_id, run in result["signals"].groupby("signal_run_id"):
        axes[1].plot([run.time.iloc[0], run.time.iloc[-1]], [.3 + .3 * (run_id % 2)] * 2,
                     color="tab:green", lw=3)
    for method in result.get("methods", METHODS):
        part = result["windows"].loc[(result["windows"].method == method) & result["windows"].included]
        axes[2].scatter(part.start + (part.end - part.start) / 2, part.rate_bpm,
                        s=14, alpha=.65, color=COLORS[method], label=method)
    axes[0].set(ylabel="ACC magnitude (g)", title="Whole Polar recording; sleep unverified — all detected bursts in red")
    axes[1].set(yticks=[], ylim=(0, 1), ylabel="Quiet/support",
                title="Blue: HRV quiet periods ≥minimum; green: continuous support after pulse QC / filter-edge trimming")
    axes[2].set(ylabel="Estimated breaths/min", xlabel="Polar sensor clock (declared UTC)",
                title="Quality-screened window estimates; methods may have different coverage")
    axes[2].legend(ncol=len(result.get("methods", METHODS)))
    fig.tight_layout()
    return fig


def plot_example(raw_ppg, result, selected=None):
    fig, axes = plt.subplots(4, 1, figsize=(14, 11))
    if selected is None:
        selected = representative_window(result)
    if selected is None:
        axes[0].text(.5, .5, "No complete quiet window available", ha="center", transform=axes[0].transAxes)
        fig.tight_layout()
        return fig
    row = selected.iloc[0]
    preview_end = min(row.start + pd.Timedelta(seconds=30), row.end)
    preview = raw_ppg.loc[row.start:preview_end] * result["config"]["polarity"]
    features = result["features"]
    points = features.loc[(features.quiet_period_id == row.quiet_period_id)
                          & (features.foot_time >= row.start) & (features.pwa_time < preview_end)]
    axes[0].plot((preview.index - row.start).total_seconds(), preview, color=".4", lw=.7, label="Raw PPG (chosen polarity)")
    for value, time, marker in (("raw_peak", "pwa_time", "^"), ("raw_foot", "foot_time", "v")):
        axes[0].scatter((points[time] - row.start).dt.total_seconds(), points[value], marker=marker, s=20, label=value)
    axes[0].plot((points.pwa_time - row.start).dt.total_seconds(), points.RIIV, "o-", ms=2, label="RIIV midline")
    axes[0].set(title=f"Window {row.window_id}, quiet period {row.quiet_period_id}: {row.start:%Y-%m-%d %H:%M:%S} — raw 30 s detail",
                ylabel="PPG optical units", xlabel="Seconds from window start")
    axes[0].legend(ncol=4, fontsize=9)
    data = result["signals"]
    window = data.loc[(data.signal_run_id == row.signal_run_id) & (data.time >= row.start) & (data.time < row.end)]
    for method in result.get("methods", METHODS):
        method_window = (result["direct_signals"].loc[result["direct_signals"].window_id == row.window_id]
                         if method == DIRECT_METHOD else window)
        seconds = (method_window.time - row.start).dt.total_seconds()
        for ax, suffix in ((axes[1], "_raw"), (axes[2], "_resp")):
            values = method_window[method + suffix]
            sd = values.std(ddof=0)
            if sd > 0:
                ax.plot(seconds, (values - values.mean()) / sd, color=COLORS[method], lw=.9, label=method)
        spectrum = result["spectra"].loc[(result["spectra"].window_id == row.window_id)
                                         & (result["spectra"].method == method)]
        if not spectrum.empty:
            area = spectrum.power.sum()
            metrics = selected.loc[selected.method == method].iloc[0]
            label = f"{method}: {metrics.candidate_rate_bpm:.1f}/min" + ("" if metrics.included else " (QC rejected)")
            axes[3].plot(spectrum.frequency_hz * 60, spectrum.power / area, color=COLORS[method], label=label)
    axes[1].set(title="Modulations before final respiratory bandpass; separately standardized (direct RIIV already low-pass filtered)",
                ylabel="Unfiltered (z)", xlabel="Seconds within window")
    axes[2].set(title="Respiratory bandpass; zero phase — polarity/phase do not label inspiration",
                ylabel="Respiratory surrogate (z)", xlabel="Seconds within window")
    axes[1].legend(ncol=3)
    axes[3].set(title="Hann spectra from unfiltered, detrended modulations; normalized to in-band power",
                xlabel="Frequency (breaths/min)", ylabel="Fraction of band power/bin")
    axes[3].legend()
    fig.tight_layout()
    return fig


def plot_waveforms(result, quiet_period_id=None, max_minutes=10):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(14, 7))
    signals = result["signals"]
    if quiet_period_id is None:
        example = representative_window(result)
        quiet_period_id = None if example is None else example.quiet_period_id.iloc[0]
    part = signals.loc[signals.quiet_period_id == quiet_period_id]
    if part.empty:
        axes[0].text(.5, .5, "No continuous respiratory signal", ha="center", transform=axes[0].transAxes)
    else:
        origin = part.time.iloc[0]
        part = part.loc[part.time <= origin + pd.Timedelta(minutes=max_minutes)]
        for ax, method in zip(axes, METHODS):
            for _, run in part.groupby("signal_run_id"):
                y = run[method + "_resp"]
                sd = y.std(ddof=0)
                if sd > 0:
                    ax.plot((run.time - origin).dt.total_seconds(), (y - y.mean()) / sd,
                            color=COLORS[method], lw=.8)
            ax.set_ylabel(method + " (z)")
        axes[0].set_title(f"Quiet period {quiet_period_id}: separate methods; gaps remain visible")
    axes[-1].set_xlabel("Seconds from selected quiet period's first clean support")
    fig.tight_layout()
    return fig


def plot_comparison(result):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    common = result["comparison"].loc[result["comparison"].all_methods_included]
    pairs = ([(DIRECT_METHOD, method) for method in METHODS] if DIRECT_METHOD in result.get("methods", ())
             else [("RIIV", "RIAV"), ("RIIV", "RIFV"), ("RIAV", "RIFV")])
    for col, (a, b) in enumerate(pairs):
        axes[0, col].scatter(common[a], common[b], s=20, alpha=.65)
        limits = np.asarray(result["config"]["respiratory_band_hz"]) * 60
        axes[0, col].plot(limits, limits, "--", color=".5", lw=.8)
        axes[0, col].set(xlabel=f"{a} breaths/min", ylabel=f"{b} breaths/min", title=f"Common windows: n={len(common)}")
        axes[1, col].scatter((common[a] + common[b]) / 2, common[b] - common[a], s=20, alpha=.65)
        axes[1, col].axhline(0, color=".5", lw=.8)
        axes[1, col].set(xlabel="Mean estimated breaths/min", ylabel=f"{b} − {a} breaths/min")
    fig.suptitle("Matched windows passing all configured methods; descriptive agreement, no reference accuracy or independent-window inference")
    fig.tight_layout()
    return fig


def plot_direct_filter(raw_ppg, result, selected=None):
    """Inspect raw filtering, the two RIIV waveforms and cardiac suppression."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    selected = representative_window(result) if selected is None else selected
    metrics = (pd.DataFrame() if selected is None else selected.loc[selected.method == DIRECT_METHOD])
    if metrics.empty or pd.isna(metrics.lowpass_cutoff_hz.iloc[0]):
        axes[0].text(.5, .5, "No direct-RIIV filter available", ha="center", transform=axes[0].transAxes)
        fig.tight_layout()
        return fig
    row = metrics.iloc[0]
    direct = result["direct_signals"].loc[result["direct_signals"].window_id == row.window_id]
    end_preview = min(row.end, row.start + pd.Timedelta(seconds=60))
    raw = raw_ppg.loc[row.start:end_preview] * result["config"]["polarity"] - row.raw_optical_offset_removed
    preview = direct.loc[direct.time <= end_preview]
    axes[0].plot((raw.index - row.start).total_seconds(), raw, color=".7", lw=.6, label="Raw PPG, context mean removed")
    axes[0].plot((preview.time - row.start).dt.total_seconds(), preview.RIIV_direct_raw,
                 color=COLORS[DIRECT_METHOD], lw=1.3, label="HR-adaptive low-pass output")
    axes[0].set(ylabel="Optical units", xlabel="Seconds within window", title=(
        f"Window {row.window_id}: reference HR {row.hr_filter_reference_bpm:.1f} bpm; "
        f"low-pass cutoff {row.lowpass_cutoff_hz:.3f} Hz; theoretical cardiac attenuation {row.cardiac_attenuation_db_theoretical:.1f} dB"))
    axes[0].legend()
    features = result["signals"]
    features = features.loc[(features.signal_run_id == row.signal_run_id)
                            & (features.time >= row.start) & (features.time < row.end)]
    for frame, name in ((features, "RIIV"), (direct, DIRECT_METHOD)):
        y = frame[name + "_resp"]
        sd = y.std(ddof=0)
        if sd > 0:
            axes[1].plot((frame.time - row.start).dt.total_seconds(), (y - y.mean()) / sd,
                         label=name, color=COLORS[name], lw=1)
    axes[1].set(ylabel="Respiratory waveform (z)", xlabel="Seconds within window",
                title="Feature midline RIIV versus direct-filter RIIV, on the same window")
    axes[1].legend()
    spectrum = result["direct_native_spectra"].loc[result["direct_native_spectra"].window_id == row.window_id]
    axes[2].semilogy(spectrum.frequency_hz, spectrum.raw_power.clip(lower=1e-15), color=".5", label="Raw PPG")
    axes[2].semilogy(spectrum.frequency_hz, spectrum.lowpass_power.clip(lower=1e-15),
                     color=COLORS[DIRECT_METHOD], label="Adaptive low-pass PPG")
    axes[2].axvline(row.lowpass_cutoff_hz, ls="--", color=COLORS[DIRECT_METHOD], label="Low-pass cutoff")
    axes[2].axvline(row.hr_filter_reference_bpm / 60, ls=":", color="k", label="Cardiac reference frequency")
    low, high = result["config"]["respiratory_band_hz"]
    axes[2].axvspan(low, high, alpha=.1, color="tab:blue", label="Respiratory search band")
    axes[2].set(xlabel="Frequency (Hz)", ylabel="PPG PSD (optical units²/Hz)",
                xlim=(0, max(high * 1.5, min(3, row.hr_filter_reference_bpm / 30))),
                title="Native-rate spectra before downsampling; theoretical attenuation is not a measured artifact-removal guarantee")
    axes[2].legend(ncol=2)
    fig.tight_layout()
    return fig
