"""Match a lightsoff-dated diary to actual recording bounds."""

from datetime import datetime, time, timedelta
from numbers import Real

import pandas as pd


def _clock_offset(value):
    if isinstance(value, (datetime, pd.Timestamp)):
        value = value.time()
    if isinstance(value, str):
        for fmt in ("%H:%M:%S.%f", "%H:%M:%S", "%H:%M"):
            try:
                value = datetime.strptime(value.strip(), fmt).time()
                break
            except ValueError:
                pass
        else:
            raise ValueError(f"Invalid diary clock time: {value!r}")
    if isinstance(value, time):
        if value.tzinfo is not None:
            raise ValueError("Diary clock times must be local wall times")
        result = pd.Timedelta(hours=value.hour, minutes=value.minute,
                              seconds=value.second, microseconds=value.microsecond)
    elif isinstance(value, (timedelta, pd.Timedelta)):
        result = pd.Timedelta(value)
    elif isinstance(value, Real) and 0 <= value < 1:
        result = pd.Timedelta(days=float(value)).round("us")
    else:
        raise ValueError(f"Invalid diary clock time: {value!r}")
    if not pd.Timedelta(0) <= result < pd.Timedelta(days=1):
        raise ValueError("Diary clock times must be within one calendar day")
    return result


def select_sleep_interval(diary, recording_start, recording_end, timezone="Europe/Rome"):
    """Return (matched row with aware local bounds, skipped incomplete rows).

    Date ALWAYS means the date of lightsoff, including times after midnight.
    Wakeup rolls to the next calendar day only if its clock time is earlier.
    Recording bounds are timezone-aware and end-exclusive. Match by positive
    time overlap, never by FIT filename, row number or the current date.
    """
    required = ["Date", "lightsoff", "wakeup"]
    if not set(required).issubset(diary.columns):
        raise ValueError(f"Diary needs columns {required}")
    start, end = pd.Timestamp(recording_start), pd.Timestamp(recording_end)
    if pd.isna(start) or pd.isna(end) or start.tzinfo is None or end.tzinfo is None or end <= start:
        raise ValueError("Recording bounds must be timezone-aware, with end after start")
    start, end = start.tz_convert(timezone), end.tz_convert(timezone)
    candidates, skipped = [], []
    for position, (_, row) in enumerate(diary.iterrows(), start=2):
        if row[required].isna().any():
            skipped.append({"excel_row": position, "reason": "incomplete Date/lightsoff/wakeup"})
            continue
        try:
            day = pd.Timestamp(row["Date"])
            if pd.isna(day) or day.tzinfo is not None or isinstance(row["Date"], Real):
                raise ValueError("Date must be a calendar date, not a numeric serial or aware timestamp")
            day = day.normalize()
            lights, wake = _clock_offset(row["lightsoff"]), _clock_offset(row["wakeup"])
            if lights == wake:
                raise ValueError("Identical lightsoff and wakeup times are ambiguous (0 or 24 hours)")
            # Build both local calendar dates BEFORE applying DST offsets.
            lights_at = (day + lights).tz_localize(timezone, ambiguous="raise", nonexistent="raise")
            wake_day = day + pd.Timedelta(days=int(wake < lights))
            wake_at = (wake_day + wake).tz_localize(timezone, ambiguous="raise", nonexistent="raise")
        except Exception as exc:
            raise ValueError(f"Diary Excel row {position}: {exc}") from exc
        overlap = max(0.0, (min(end, wake_at) - max(start, lights_at)).total_seconds())
        candidates.append({**row.to_dict(), "excel_row": position, "lightsoff_at": lights_at,
                           "wakeup_at": wake_at, "overlap_seconds": overlap})
    matches = [row for row in candidates if row["overlap_seconds"] > 0]
    if not matches:
        raise ValueError(
            f"No complete diary interval overlaps {start} to {end}. "
            "Date must be the calendar date of lightsoff, even when lightsoff is after midnight."
        )
    if len(matches) > 1:
        rows = [row["excel_row"] for row in matches]
        raise ValueError(f"Multiple diary intervals overlap this recording (Excel rows {rows}); check the diary or recording bounds")
    selected = pd.Series(matches[0])
    selected["recorded_start"] = max(start, selected["lightsoff_at"])
    selected["recorded_end"] = min(end, selected["wakeup_at"])
    selected["full_diary_interval_recorded"] = (
        start <= selected["lightsoff_at"] and end >= selected["wakeup_at"]
    )
    return selected, pd.DataFrame(skipped, columns=["excel_row", "reason"])
