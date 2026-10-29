"""Raw FIT export helpers for batch_raw_export.ipynb; no analysis or cleaning."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from fitparse import FitFile

from fit_bbi import decode_snapshots
from fit_recording import expand_acceleration_batch
from sleep_diary import select_sleep_interval


OUTPUT_NAMES = ("accelerometer_samples.parquet", "raw_bbi.parquet")
_BBI_CONTEXT = {"timestamp", "timestamp_ms", "logger_meta", "logger_schema",
                "elapsed_ms", "start_unix_s"}


def select_recordings(directory, *, start_from=None, recording_names=None):
    """Select sorted FIT files, including start_from and all later filenames.

    An optional explicit filename list further restricts that range. Validate
    both selectors against the full folder so a typo never silently selects
    a different starting point. Filename order does not perform diary matching.
    """
    files = sorted((p for p in Path(directory).iterdir()
                    if p.is_file() and p.suffix.lower() == ".fit"),
                   key=lambda p: p.name.casefold())
    names = [p.name.casefold() for p in files]
    if recording_names is not None:
        if isinstance(recording_names, str):
            raise TypeError("RECORDING_NAMES must be a list of filenames, or None")
        selected = {name.casefold() for name in recording_names}
        missing = selected - set(names)
        if missing:
            raise FileNotFoundError(f"Requested recordings not found: {sorted(missing)}")
    if start_from is not None:
        if start_from.casefold() not in names:
            raise FileNotFoundError(f"START_FROM recording not found: {start_from}")
        files = files[names.index(start_from.casefold()):]
    if recording_names is not None:
        files = [p for p in files if p.name.casefold() in selected]
    if not files:
        raise FileNotFoundError(f"No FIT recordings selected in {directory}")
    return files


def read_raw_fit(path):
    """Read once, preserving XYZ values and original record/session BBI fields.

    BBI snapshots are deliberately NOT decoded for export: even inconsistent
    snapshots, repeated values, invalid slots and final session fields survive.
    Their arrays are native Parquet list columns, not JSON strings.
    """
    batches, clocks, snapshots, units = [], [], [], {}
    for message_index, message in enumerate(FitFile(str(path)).get_messages()):
        if message.name == "accelerometer_data":
            times, xyz = expand_acceleration_batch(message.get_values())
            clocks.append(times)
            batches.append(xyz)
            for field in message:
                if field.name.startswith("calibrated_accel_"):
                    units.setdefault(field.name, set()).add(str(field.units))
        elif message.name in ("record", "session"):
            fields = message.get_values()
            if not any("bbi" in name for name in fields):
                continue
            row = {"message_index": message_index, "message_type": message.name}
            for name, value in fields.items():
                unprefixed = name.removeprefix("final_")
                if "bbi" in name or unprefixed in _BBI_CONTEXT:
                    row[name] = list(value) if isinstance(value, tuple) else value
            snapshots.append(row)
    if not batches or not any(len(batch) for batch in batches):
        raise ValueError("No calibrated accelerometer samples found")
    accel = pd.DataFrame(np.concatenate(batches), columns=["accel_x", "accel_y", "accel_z"])
    accel.insert(0, "sample_time", pd.to_datetime(np.concatenate(clocks), unit="ns", utc=True))
    raw_bbi = pd.DataFrame(snapshots)
    if raw_bbi.empty:
        raw_bbi = pd.DataFrame({"message_index": pd.Series(dtype="int64"),
                                "message_type": pd.Series(dtype="string"),
                                "timestamp": pd.Series(dtype="datetime64[ns, UTC]")})
    elif "timestamp" in raw_bbi:
        raw_bbi["timestamp"] = pd.to_datetime(raw_bbi["timestamp"], utc=True)
    return accel, raw_bbi.convert_dtypes(), {key: sorted(value) for key, value in units.items()}


def crop_to_diary(accel, diary, timezone="Europe/Rome"):
    """Match actual UTC coverage to lightsoff -> wakeup; retain [start, end)."""
    required = ["Date", "lightsoff", "wakeup"]
    if not set(required).issubset(diary.columns):
        raise ValueError(f"Diary needs columns {required}")
    times = pd.DatetimeIndex(accel["sample_time"])
    if times.tz is None or times.hasnans or len(times) < 2:
        raise ValueError("Acceleration needs at least two valid timezone-aware timestamps")
    steps = times.to_series().diff()
    period = steps[steps > pd.Timedelta(0)].median()
    if pd.isna(period):
        raise ValueError("Cannot determine the accelerometer sample period")
    start, end = times.min(), times.max() + period
    row, skipped = select_sleep_interval(diary[required], start, end, timezone=timezone)
    lights, wakeup = row["lightsoff_at"], row["wakeup_at"]
    mask = (times >= lights) & (times < wakeup)
    cropped = accel.loc[mask].reset_index(drop=True)
    if cropped.empty:
        raise ValueError("Matched diary interval contains no accelerometer samples")
    window = {"diary_excel_row": int(row["excel_row"]),
              "lightsoff": lights, "wakeup": wakeup,
              "recording_start": start, "recording_end_exclusive": end,
              "full_diary_interval_recorded": bool(row["full_diary_interval_recorded"]),
              "accel_samples_full": len(accel), "accel_samples_saved": len(cropped),
              "incomplete_diary_rows": skipped["excel_row"].tolist()}
    return cropped, window


def snapshot_messages(raw_bbi):
    """Restore decoder input from either fresh or Parquet-loaded raw snapshots."""
    for row in raw_bbi.to_dict("records"):
        fields = {}
        for name, value in row.items():
            if name in ("message_index", "message_type") or value is None or value is pd.NA or value is pd.NaT:
                continue
            if isinstance(value, (float, np.floating)) and np.isnan(value):
                continue
            fields[name] = value.tolist() if isinstance(value, np.ndarray) else value
        yield row["message_type"], fields


def bbi_preview(raw_bbi):
    """Decode only for display; exported snapshots stay untouched.

    Legacy slots are shown at their record timestamps without reconstructing
    beat times or claiming that repeated record snapshots are unique beats.
    """
    empty = pd.DataFrame({"time": pd.Series(dtype="datetime64[ns, UTC]"),
                          "bbi_ms": pd.Series(dtype="float64")})
    try:
        rows, report = decode_snapshots(snapshot_messages(raw_bbi))
    except (ValueError, KeyError, TypeError) as exc:
        return empty, f"BBI preview unavailable: {exc}. Original snapshots will still be saved."
    if report["schema"] is not None:
        if not rows:
            return empty, "No intervals recovered; original snapshots will still be saved."
        points = pd.DataFrame(rows)
        points["time"] = pd.to_datetime(points["callback_time_utc_approx"], utc=True)
        return points[["time", "bbi_ms"]], (
            f"Numbered intervals at approximate callback arrival times; "
            f"{report['missing_received_intervals']} missing, "
            f"{report['invalid_recovered_intervals']} invalid; no cleaning/interpolation"
        )
    points = []
    for name, fields in snapshot_messages(raw_bbi):
        if name != "record" or fields.get("timestamp") is None:
            continue
        slots = [f"bbi_{i}" for i in range(1, 9) if f"bbi_{i}" in fields]
        if slots and fields.get("bbi_count") is not None:
            slots = slots[:max(0, int(fields["bbi_count"]))]
        if not slots and "bbi_latest" in fields:
            slots = ["bbi_latest"]
        for slot in slots:
            points.append({"time": fields["timestamp"], "bbi_ms": fields[slot]})
    if not points:
        return empty, "No BBI values available for preview."
    points = pd.DataFrame(points)
    points["time"] = pd.to_datetime(points["time"], utc=True)
    return points, "Legacy BBI slots at FIT record times; repeated snapshots may be present"


def plot_raw_preview(accel, raw_bbi, window, title, timezone="Europe/Rome", max_acc_points=200_000):
    """Full recording above BBI, with shared local time and diary overlays."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    if not isinstance(max_acc_points, int) or max_acc_points < 1:
        raise ValueError("max_acc_points must be a positive integer")
    stride = max(1, int(np.ceil(len(accel) / max_acc_points)))
    shown = accel.iloc[::stride]
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, layout="constrained")
    times = shown["sample_time"].dt.tz_convert(timezone)
    for column in ("accel_x", "accel_y", "accel_z"):
        axes[0].plot(times, shown[column], lw=0.5, alpha=0.8, label=column)
    axes[0].set_ylabel("Acceleration\n(original FIT scale)")
    axes[0].set_title(f"{title} | full recording; display stride {stride}; export keeps every cropped sample")
    points, note = bbi_preview(raw_bbi)
    if len(points):
        axes[1].plot(points["time"].dt.tz_convert(timezone), points["bbi_ms"],
                     linestyle="none", marker=".", ms=2, alpha=0.6)
    axes[1].set_ylabel("BBI (ms)")
    axes[1].set_title(note, fontsize=10)
    for axis in axes:
        axis.axvspan(window["lightsoff"], window["wakeup"], color="teal", alpha=0.08)
        axis.axvline(window["lightsoff"], color="green", ls="--", label="Diary lightsoff")
        axis.axvline(window["wakeup"], color="crimson", ls="--", label="Diary wakeup")
        axis.grid(alpha=0.25)
    axes[0].legend(loc="upper right", ncol=5)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M", tz=ZoneInfo(timezone)))
    axes[1].set_xlabel(f"Local time ({timezone}); BBI timestamps are arrival/record proxies, not beat times")
    return fig


def save_raw_tables(output_dir, accel, raw_bbi, window, source, units, overwrite=False):
    """Write only two raw Parquet files; provenance is embedded in each file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_dir, source = Path(output_dir), Path(source)
    targets = [output_dir / name for name in OUTPUT_NAMES]
    if not overwrite and any(path.exists() for path in targets):
        raise FileExistsError(f"Raw export already exists in {output_dir}; set OVERWRITE=True to replace it")
    metadata = {"export_version": 2, "source_fit": source.name,
                "source_fit_size": source.stat().st_size,
                "source_fit_mtime_ns": source.stat().st_mtime_ns,
                "window": window, "fit_acceleration_units": units,
                "acceleration": "Original FIT scale, cropped to [lightsoff, wakeup); no filtering/resampling",
                "bbi": "Original record/session snapshots for the FULL recording; no decoding/cleaning"}
    encoded = json.dumps(metadata, default=str).encode("utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Finish both writes before replacing either destination; a failed write
    # leaves existing exports intact and removes its temporary files.
    with TemporaryDirectory(prefix=".raw-export-", dir=output_dir) as staging:
        for name, frame in zip(OUTPUT_NAMES, (accel, raw_bbi)):
            table = pa.Table.from_pandas(frame, preserve_index=False)
            schema_metadata = dict(table.schema.metadata or {})
            schema_metadata[b"overnight_raw_export"] = encoded
            table = table.replace_schema_metadata(schema_metadata)
            pq.write_table(table, Path(staging) / name, compression="zstd")
        for name, target in zip(OUTPUT_NAMES, targets):
            os.replace(Path(staging) / name, target)
    return targets
