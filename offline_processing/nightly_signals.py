"""Load diary-defined nights and compare raw H10 ECG and two PPG recordings.

Run ``python offline_processing/nightly_signals.py PATH_TO_EXPORT`` to open one
Qt5Agg window per night. No filtering, normalization, or clock correction is
applied. The overview retains extrema; zooming redraws from the original samples.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SIGNALS = {
    "h10_ecg": ("H10 ECG", "ecg", "#b33d35"),
    "verity_sense_ppg": ("Verity Sense PPG", "ppg", "#197a61"),
    "empatica_ppg": ("EmbracePlus PPG", "ppg", "#315aaf"),
}


@dataclass
class Night:
    subject: str
    diary_date: str
    start: pd.Timestamp
    end: pd.Timestamp
    timezone: str
    signals: dict[str, pd.Series]
    units: dict[str, str]


def load_nights(directory, subject=None):
    """Read raw samples and inclusive sleep windows from export metadata.

    Naive timestamps are localized using the export's declared source timezone;
    aware timestamps are converted to the diary timezone. Missing sensor data
    produces an empty Series for that night. Duplicate timestamps are retained.
    """
    directory = Path(directory).expanduser()
    subjects = sorted(
        p.name.removesuffix("_h10_ecg.parquet")
        for p in directory.glob("*_h10_ecg.parquet")
        if not p.name.startswith("._")
    )
    if subject is None:
        if len(subjects) != 1:
            raise ValueError(f"Expected one subject in {directory}; found {subjects}. Specify subject.")
        subject = subjects[0]

    recordings, units = {}, {}
    windows = timezone = None
    for key, (_, column, _) in SIGNALS.items():
        path = directory / f"{subject}_{key}.parquet"
        parquet = pq.ParquetFile(path)
        metadata = parquet.schema_arrow.metadata or {}
        if b"night_data_export" not in metadata:
            raise ValueError(f"Missing night_data_export metadata: {path.name}")
        export = json.loads(metadata[b"night_data_export"])
        if export["subject"] != subject or export["signal"] != key:
            raise ValueError(f"Subject/signal metadata disagrees with filename: {path.name}")
        if windows is None:
            windows = export["sleep_windows_local"]
            timezone = export["diary_timezone"]
        elif windows != export["sleep_windows_local"] or timezone != export["diary_timezone"]:
            raise ValueError(f"Sleep windows/timezone disagree across exports: {path.name}")

        time_column = export["source_timestamp_field"]
        frame = parquet.read(columns=[time_column, column]).to_pandas(ignore_metadata=True)
        index = pd.DatetimeIndex(frame[time_column])
        if index.hasnans:
            raise ValueError(f"Invalid sample timestamps: {path.name}")
        if index.tz is None:
            source_timezone = export.get("naive_source_timestamp_timezone")
            if not source_timezone:
                raise ValueError(f"No timezone declared for naive timestamps: {path.name}")
            index = index.tz_localize(source_timezone, ambiguous="raise", nonexistent="raise")
        index = index.tz_convert(timezone).rename("time")
        series = pd.Series(frame[column].to_numpy(), index=index, name=key)
        if not series.index.is_monotonic_increasing:
            series = series.sort_index(kind="stable")
        recordings[key] = series
        attrs = json.loads(metadata.get(b"PANDAS_ATTRS", b"{}"))
        units[key] = attrs.get("units", {}).get(column, "export units")

    if not windows:
        raise ValueError("No sleep windows in the export metadata.")
    nights = []
    for window in windows:
        start, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
        if start.tzinfo is None or end.tzinfo is None or start >= end:
            raise ValueError(f"Invalid sleep window: {window}")
        start, end = start.tz_convert(timezone), end.tz_convert(timezone)
        nights.append(Night(
            subject, window["diary_date"], start, end, timezone,
            {key: series.loc[start:end] for key, series in recordings.items()}, units.copy(),
        ))
    return nights


def overview_indices(values, max_points=80_000):
    """Select chronological block minima/maxima, endpoints, and NaN boundaries.

    This affects display only. Unlike a fixed stride, it retains narrow ECG
    peaks, and the plot requests the original samples again after every zoom.
    """
    if max_points < 4:
        raise ValueError("max_points must be at least 4")
    count = len(values)
    if count <= max_points:
        return np.arange(count)
    width = int(np.ceil(count / (max_points // 2)))
    rows = count // width
    blocks = np.asarray(values[:rows * width]).reshape(rows, width)
    finite = np.isfinite(blocks)
    offsets = np.arange(rows) * width
    minima = offsets + np.where(finite, blocks, np.inf).argmin(axis=1)
    maxima = offsets + np.where(finite, blocks, -np.inf).argmax(axis=1)
    tail = np.asarray(values[rows * width:])
    tail_indices = np.array([], dtype=int)
    if len(tail):
        valid = np.isfinite(tail)
        tail_indices = rows * width + np.array([
            np.where(valid, tail, np.inf).argmin(),
            np.where(valid, tail, -np.inf).argmax(),
        ])
    boundaries = np.flatnonzero(np.diff(np.isfinite(values)))
    return np.unique(np.concatenate((
        [0, count - 1], minima, maxima, tail_indices,
        boundaries, boundaries + 1,
    )))


class RawSignalView:
    """Keep overview drawing small while retaining raw samples for pan/zoom."""

    def __init__(self, axis, signal, color, max_points):
        self.axis, self.signal, self.max_points = axis, signal, max_points
        self.line, = axis.plot([], [], color=color, linewidth=0.6)
        steps = np.diff(signal.index.asi8)
        positive = steps[:10_000][steps[:10_000] > 0]
        self.sample_interval_ns = float(np.median(positive)) if len(positive) else None
        threshold = max(1e9, 5 * self.sample_interval_ns) if len(positive) else 1e9
        self.gaps = np.flatnonzero(steps > threshold) + 1
        self.refresh(axis)
        axis.relim()
        axis.autoscale_view(scalex=False)
        axis.callbacks.connect("xlim_changed", self.refresh)

    def refresh(self, axis):
        import matplotlib.dates as mdates

        left, right = sorted(axis.get_xlim())
        lo = max(0, self.signal.index.searchsorted(pd.Timestamp(mdates.num2date(left))) - 1)
        hi = min(len(self.signal), self.signal.index.searchsorted(
            pd.Timestamp(mdates.num2date(right)), side="right") + 1)
        values = self.signal.to_numpy()
        selected = overview_indices(values[lo:hi], self.max_points) + lo
        # Keep both sides of gaps even when they fall inside a display block.
        gaps = self.gaps[(self.gaps > lo) & (self.gaps < hi)]
        selected = np.unique(np.concatenate((selected, gaps - 1, gaps)))
        times = self.signal.index[selected].tz_convert("UTC").tz_localize(None).to_numpy()
        x = mdates.date2num(times)
        y = values[selected].astype(float)
        breaks = np.flatnonzero(np.diff(np.searchsorted(self.gaps, selected, side="right"))) + 1
        self.line.set_data(np.insert(x, breaks, np.nan), np.insert(y, breaks, np.nan))
        axis.figure.canvas.draw_idle()


def plot_night(night, max_points=80_000):
    """Three shared-time subplots; return a figure with live raw-data zooming."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    if max_points < 4:
        raise ValueError("max_points must be at least 4")
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(15, 9), layout="constrained")
    figure.canvas.manager.set_window_title(f"{night.subject} | night {night.diary_date}")
    figure.suptitle(
        f"{night.subject} | night {night.diary_date}\n"
        f"{night.start:%Y-%m-%d %H:%M} - {night.end:%Y-%m-%d %H:%M} ({night.timezone})"
    )
    axes[0].set_xlim(night.start, night.end)
    views = []
    for axis, (key, (label, _, color)) in zip(axes, SIGNALS.items()):
        series = night.signals[key]
        unit = night.units[key].replace("uV", "µV")
        axis.set_ylabel(f"{label}\n({unit})")
        axis.grid(alpha=0.2)
        if series.empty:
            axis.text(0.5, 0.5, "No samples in this sleep window",
                      transform=axis.transAxes, ha="center", va="center", color="0.4")
        else:
            view = RawSignalView(axis, series, color, max_points)
            views.append(view)
            rate = 1e9 / view.sample_interval_ns if view.sample_interval_ns else None
            title = f"{len(series):,} samples" + (f" | ~{rate:.1f} Hz" if rate else "")
            axis.set_title(title, loc="left", fontsize=9, color="0.35")
    timezone = ZoneInfo(night.timezone)
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(tz=timezone, minticks=4, maxticks=10))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone))
    axes[-1].set_xlabel(f"Local time ({night.timezone}); zoom/pan to inspect original samples")
    # Matplotlib holds bound callbacks weakly. Retain the views with the figure.
    figure._raw_signal_views = views
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, help="Folder containing the six nightly exports")
    parser.add_argument("--subject", help="Subject prefix; auto-detected when only one is present")
    parser.add_argument("--save-dir", type=Path, help="Also save full-night PNGs locally")
    args = parser.parse_args()
    if args.directory is None:
        from notebook_paths import input_path
        args.directory = input_path("night_data_export_path")

    import matplotlib as mpl
    mpl.use("Qt5Agg")
    import matplotlib.pyplot as plt

    nights = load_nights(args.directory, args.subject)
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
    for night in nights:
        counts = ", ".join(f"{key}={len(values):,}" for key, values in night.signals.items())
        print(f"{night.diary_date}: {counts}", flush=True)
        figure = plot_night(night)
        if args.save_dir:
            figure.savefig(args.save_dir / f"{night.subject}_{night.diary_date}_signals.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
