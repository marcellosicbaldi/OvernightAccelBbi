"""Decode logger v1.1/v1.1.1 snapshots, preserving gaps and arrival times.

Run: python offline_processing/fit_bbi.py path/to/recording.fit
Notebook: rows, report = load_bbi(FIT_PATH)
"""

import argparse
from datetime import datetime, timedelta, timezone
import json

CAPACITY = 24


def decode_snapshots(messages):
    """Accept (message_name, field_dictionary) pairs from ONE activity session.

    Sequence numbers enumerate intervals delivered by Garmin, not all heartbeats.
    callback_elapsed_ms is an arrival timestamp, never a beat timestamp.
    """
    recovered = {}
    observed_total = 0
    final_total = None
    start_unix = None
    detected_schema = None
    for name, fields in messages:
        if name not in ("record", "session"):
            continue
        prefix = "final_" if name == "session" else ""
        metadata = fields.get(prefix + "logger_meta")
        if metadata is not None:
            if not isinstance(metadata, (tuple, list)) or len(metadata) != 8:
                raise ValueError("Unexpected compact metadata array size")
            if any(not isinstance(value, int) or value < 0 for value in metadata):
                raise ValueError("Invalid compact metadata value")
            if metadata[0] != 3:
                raise ValueError(f"Unsupported compact logger schema: {metadata[0]}")
            fields = dict(fields)
            fields.update({prefix + key: value for key, value in zip(
                ("logger_schema", "bbi_history_count", "callback_count",
                 "bbi_empty_callbacks", "bbi_invalid_total", "logged_accel_samples",
                 "elapsed_ms", "start_unix_s"), metadata
            )})
        schema = fields.get(prefix + "logger_schema")
        if schema is None:
            continue
        if schema not in (2, 3) or (schema == 3 and metadata is None):
            raise ValueError(f"Unsupported logger schema: {schema}")
        if detected_schema is not None and schema != detected_schema:
            raise ValueError("Mixed logger schemas in one session")
        detected_schema = schema
        total = int(fields[prefix + "bbi_total"])
        count = int(fields[prefix + "bbi_history_count"])
        values = fields[prefix + "bbi_history"]
        times = fields[prefix + "bbi_rx_ms"]
        anchor = int(fields[prefix + "start_unix_s"])
        if total < 0 or count != min(total, CAPACITY):
            raise ValueError("Inconsistent BBI snapshot count")
        if len(values) != CAPACITY or len(times) != CAPACITY:
            raise ValueError("Unexpected BBI snapshot array size")
        if start_unix is not None and anchor != start_unix:
            raise ValueError("Multiple sessions or inconsistent start-time anchors")
        start_unix = anchor
        observed_total = max(observed_total, total)
        if name == "session":
            if final_total is not None and final_total != total:
                raise ValueError("Multiple inconsistent final summaries")
            final_total = total
        for sequence in range(total - count + 1, total + 1):
            slot = (sequence - 1) % CAPACITY
            value, received_ms = values[slot], times[slot]
            value = int(value) if value is not None and 0 < value < 65535 else None
            if received_ms is None or received_ms < 0:
                raise ValueError("Missing or invalid callback arrival time")
            sample = (value, int(received_ms))
            if sequence in recovered and recovered[sequence] != sample:
                raise ValueError(f"Conflicting repeated snapshot for interval {sequence}")
            recovered[sequence] = sample

    if final_total is not None and final_total < observed_total:
        raise ValueError("Final interval total is smaller than a record total")
    missing_ranges = []
    previous = 0
    rows = []
    for sequence, (value, received_ms) in sorted(recovered.items()):
        if sequence > previous + 1:
            missing_ranges.append((previous + 1, sequence - 1))
        previous = sequence
        rows.append({
            "sequence": sequence,
            "bbi_ms": value,
            "callback_elapsed_ms": received_ms,
            "callback_time_utc_approx": (
                datetime.fromtimestamp(start_unix, timezone.utc)
                + timedelta(milliseconds=received_ms)
            ).isoformat(),
        })
    if previous < observed_total:
        missing_ranges.append((previous + 1, observed_total))
    report = {
        "schema": detected_schema,
        "final_summary_present": final_total is not None,
        "final_received_total": final_total,
        "largest_observed_total": observed_total,
        "recovered_intervals": len(rows),
        "invalid_recovered_intervals": sum(row["bbi_ms"] is None for row in rows),
        "missing_received_intervals": sum(end - start + 1 for start, end in missing_ranges),
        "missing_sequence_ranges": missing_ranges,
        "notes": [
            "Counts describe intervals received by the app, not all physiological beats.",
            "Arrival times are not beat times; UTC anchor has one-second resolution.",
        ],
    }
    if detected_schema is None:
        report["notes"].append("No numbered snapshots. Use the existing notebook to inspect legacy fields.")
    elif final_total is None:
        report["notes"].append("No final summary: the end of recording cannot be checked.")
    if detected_schema is not None and observed_total == 0:
        report["notes"].append("No BBI received by the app; HR alone does not establish BBI availability.")
    return rows, report


def load_bbi(path):
    from fitparse import FitFile

    return decode_snapshots(
        (message.name, message.get_values())
        for message in FitFile(str(path)).get_messages()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit_file")
    args = parser.parse_args()
    _, summary = load_bbi(args.fit_file)
    print(json.dumps(summary, indent=2))
