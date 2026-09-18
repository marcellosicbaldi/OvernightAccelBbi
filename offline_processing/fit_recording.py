"""Load raw XYZ acceleration and numbered BBI snapshots in one FIT pass."""

from pathlib import Path

import numpy as np
import pandas as pd
from fitparse import FitFile

from fit_bbi import decode_snapshots


def expand_acceleration_batch(fields):
    """Return UTC nanoseconds and an Nx3 array; never invent missing timing."""
    names = ["calibrated_accel_" + axis for axis in "xyz"]
    if any(fields.get(name) is None for name in names):
        raise ValueError("FIT batch lacks calibrated XYZ acceleration")
    axes = [np.atleast_1d(np.asarray(fields[name], dtype=float)) for name in names]
    if any(axis.ndim != 1 for axis in axes) or len({len(axis) for axis in axes}) != 1:
        raise ValueError("FIT XYZ batch lengths must match")
    if fields.get("sample_time_offset") is None or fields.get("timestamp") is None:
        raise ValueError("FIT batch lacks sample timing; cannot align movement and HR")
    offsets = np.atleast_1d(np.asarray(fields["sample_time_offset"], dtype=float))
    if offsets.ndim != 1 or len(offsets) != len(axes[0]) or not np.isfinite(offsets).all() or (offsets < 0).any():
        raise ValueError("FIT sample offsets must be finite, non-negative, and match XYZ lengths")
    timestamp_ms = fields.get("timestamp_ms") or 0
    if not np.isfinite(timestamp_ms) or timestamp_ms < 0:
        raise ValueError("Invalid FIT millisecond timestamp")
    start = pd.Timestamp(fields["timestamp"])
    if pd.isna(start):
        raise ValueError("Missing FIT timestamp")
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    times = start.value + np.rint((timestamp_ms + offsets) * 1e6).astype(np.int64)
    return times, np.column_stack(axes)


def load_recording(path):
    """Load a logger recording without the old notebook's per-sample loop.

    Values retain the file's numerical scale. Callers must choose g versus mg
    explicitly; calibrated field labels are not sufficient for this Garmin.
    No dependency on another notebook's variables or saved outputs is needed.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    timestamps, batches, snapshots = [], [], []
    units = {}
    for message in FitFile(str(path)).get_messages():
        if message.name == "accelerometer_data":
            times, xyz = expand_acceleration_batch(message.get_values())
            timestamps.append(times)
            batches.append(xyz)
            for field in message:
                if field.name.startswith("calibrated_accel_"):
                    units.setdefault(field.name, set()).add(str(field.units))
        elif message.name in ("record", "session"):
            snapshots.append((message.name, message.get_values()))
    if not batches or not any(len(batch) for batch in batches):
        raise ValueError("No calibrated accelerometer batches found in FIT file")
    xyz = np.concatenate(batches)
    accel = pd.DataFrame(xyz, columns=["accel_x", "accel_y", "accel_z"])
    accel.insert(0, "sample_time", pd.to_datetime(np.concatenate(timestamps), unit="ns", utc=True))
    bbi_rows, bbi_report = decode_snapshots(snapshots)
    return {"path": path, "accel_df": accel, "bbi_rows": bbi_rows, "bbi_report": bbi_report,
            "fit_field_units": {key: sorted(value) for key, value in units.items()},
            "acceleration_batches": len(batches)}
