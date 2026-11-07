"""Reusable version of the Aboy++-style detector in polar.ipynb.

Retains its 75 Hz working grid, adaptive 10 s / 8 s windows, derivative
threshold (75th percentile), and onset rules. Implementation defects in
the exploratory cell are corrected: use a digital Butterworth SOS bandpass (the cell used
analog numerator coefficients without their denominator), return an onset's
sample position rather than its ordinal index, and process the final window.
Use a centered five-point derivative, and omit the cell's dimensionally
inconsistent HR-variability update (comparing BPM with sample intervals).
This is the repository's adapted detector, not a validated pyPPG release.
"""
from types import SimpleNamespace

import numpy as np
from scipy import signal


def def_bandpass(sig, fs, lower_cutoff, upper_cutoff):
    # Interpret bounds as passband cutoffs. A direct Chebyshev-II replacement
    # interprets them as STOPband edges and can suppress the sleep pulse rate.
    sos = signal.butter(5, [lower_cutoff, upper_cutoff], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, sig)


def detect_maxima(sig, percentile, hr_win, peak_detector):
    sig = np.asarray(sig)
    maxima = signal.find_peaks(sig)[0]
    if peak_detector == "ABD" or hr_win == 0:
        return maxima[sig[maxima] > np.percentile(sig, percentile)]
    maxima = signal.find_peaks(sig, distance=max(1, hr_win))[0]
    minima = signal.find_peaks(-sig, distance=max(1, hr_win))[0]
    if not len(maxima) or not len(minima):
        return np.array([], dtype=int)
    amplitudes = [sig[p] - sig[minima[np.argmin(abs(p - minima))]] for p in maxima]
    return signal.find_peaks(sig, prominence=max(0, np.mean(amplitudes) * .25),
                             distance=max(1, hr_win))[0]


def estimate_HR(sig, fs, up, hr_past):
    f, power = signal.periodogram(sig, fs, np.blackman(len(sig)))
    low, high = ((up.fl_hz, up.fh_hz) if not up.fl <= hr_past <= up.fh
                 else (hr_past / 60 * .5, hr_past / 60 * 1.4))
    selected = (f >= low) & (f <= high)
    return int(f[selected][np.argmax(power[selected])] * 60)


def estimate_deriv(sig):
    # The cell's five-point Savitzky-Golay first derivative, centered.
    return signal.savgol_filter(sig, 5, 1, deriv=1)


def find_pulse_peaks(p2, p3):
    positions = np.searchsorted(p3, p2, side="right")
    return np.asarray(p3)[positions[positions < len(p3)]]


def find_onsets(sig, fs, up, peaks, med_interval_samples):
    smooth = def_bandpass(sig, fs, .9 * up.fl_hz, 3 * up.fh_hz)
    minima = signal.find_peaks(-smooth, distance=max(1, med_interval_samples * .3))[0]
    onsets, kept = [], []
    for peak in peaks:
        if not kept:
            candidates = minima[minima < peak]
        else:
            candidates = minima[(minima > onsets[-1]) & (peak - minima > fs * .12)
                                & (minima - kept[-1] > fs * .3) & (minima < peak)]
        if len(candidates):
            onsets.append(candidates[-1])
            kept.append(peak)
    return np.asarray(onsets, dtype=int), np.asarray(kept, dtype=int)


def get_peak_onset(ppg, fs, peak_detector="PPGdet"):
    """Return paired (peak, preceding onset) indices on the input sample grid."""
    ppg = np.asarray(ppg, dtype=float)
    if (ppg.ndim != 1 or not np.isfinite(ppg).all() or not np.isfinite(fs)
            or fs <= 20 or len(ppg) < 10 * fs):
        raise ValueError("Need >=10 s of finite regular PPG and fs >20 Hz")
    if peak_detector not in ("PPGdet", "ABD"):
        raise ValueError("peak_detector must be PPGdet or ABD")
    work_fs, window = 75, 750
    x = signal.resample(ppg, int(round(len(ppg) * work_fs / fs)))
    if np.ptp(x) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    up = SimpleNamespace(fl=30, fh=200, fl_hz=.5, fh_hz=200 / 60)
    starts = np.unique(np.r_[np.arange(0, len(x) - window + 1, 600), len(x) - window])
    all_peaks, heart_rates = [], []
    hr_past, hrvi = 0, 0
    for start in starts:
        current = x[start:start + window]
        y1 = def_bandpass(current, work_fs, .45, 10)
        hr = estimate_HR(y1, work_fs, up, hr_past)
        hr_past = hr
        heart_rates.append(hr)
        if peak_detector == "PPGdet" and hr > 40:
            if start == 0:
                intervals = np.diff(detect_maxima(y1, 0, 0, peak_detector))
                if len(intervals):
                    intervals = intervals[intervals >= np.median(intervals)]
                    hrvi = np.std(intervals) / np.mean(intervals) * 5
            hr_win = work_fs / ((1 + hrvi) * 3)
        else:
            hr_win = 0
        y2 = def_bandpass(current, work_fs, .45, 2.5 * hr / 60)
        p2 = detect_maxima(estimate_deriv(y2), 75, hr_win, peak_detector)
        y3 = def_bandpass(current, work_fs, .45, 10 * hr / 60)
        p3 = detect_maxima(y3, 50, hr_win, peak_detector)
        p4 = np.unique(find_pulse_peaks(p2, p3))
        all_peaks.extend(start + p4)
    peaks = np.unique((np.asarray(all_peaks) / work_fs * fs).astype(int))
    if not len(peaks):
        return peaks, peaks.copy()
    onsets, peaks = find_onsets(ppg, fs, up, peaks, 60 / np.median(heart_rates) * fs)
    return peaks, onsets
