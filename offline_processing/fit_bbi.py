"""Decode logger snapshots, preserving gaps and arrival times.

Run: python offline_processing/fit_bbi.py path/to/recording.fit
Notebook: rows, report = load_bbi(FIT_PATH)
"""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json

CAPACITY = 24


def _sequence_ranges(sequences):
    ranges = []
    for sequence in sorted(sequences):
        if ranges and sequence == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], sequence)
        else:
            ranges.append((sequence, sequence))
    return ranges


def decode_snapshots(messages, *, strict=False):
    """Accept (message_name, field_dictionary) pairs from ONE activity session.

    Sequence numbers enumerate intervals delivered by Garmin, not all heartbeats.
    callback_elapsed_ms is an arrival timestamp, never a beat timestamp.

    Schema 3 wrote related fields separately. By default, reject its inconsistent
    snapshots and omit intervals whose remaining copies disagree. Never choose a
    conflicting value by recency or majority. strict=True disables this recovery.
    Schemas 2 and 4 retain fail-fast validation.
    """
    recovered = {}
    observed_total = 0
    final_total = None
    start_unix = None
    detected_schema = None
    rejected = Counter()
    conflicting = set()
    final_snapshot_valid = False
    for name, fields in messages:
        if name not in ("record", "session"):
            continue
        prefix = "final_" if name == "session" else ""
        metadata = fields.get(prefix + "logger_meta")
        packed = fields.get(prefix + "bbi_snapshot")
        if packed is not None:
            if not isinstance(packed, (tuple, list)) or len(packed) != 8 + 2 * CAPACITY:
                raise ValueError("Unexpected packed BBI snapshot array size")
            if any(not isinstance(value, int) or not 0 <= value < 4294967295 for value in packed):
                raise ValueError("Invalid packed BBI snapshot value")
            if packed[0] != 4:
                raise ValueError(f"Unsupported packed logger schema: {packed[0]}")
            fields = dict(fields)
            fields.update({prefix + key: value for key, value in {
                "logger_schema": 4, "bbi_total": packed[1],
                "bbi_history_count": min(packed[1], CAPACITY),
                "elapsed_ms": packed[6], "start_unix_s": packed[7],
                "bbi_history": packed[8:8 + CAPACITY],
                "bbi_rx_ms": packed[8 + CAPACITY:],
            }.items()})
        elif metadata is not None:
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
        if (schema not in (2, 3, 4) or (schema == 3 and metadata is None)
                or (schema == 4 and packed is None)):
            raise ValueError(f"Unsupported logger schema: {schema}")
        if detected_schema is not None and schema != detected_schema:
            raise ValueError("Mixed logger schemas in one session")
        detected_schema = schema
        total = int(fields[prefix + "bbi_total"])
        count = int(fields[prefix + "bbi_history_count"])
        values = fields[prefix + "bbi_history"]
        times = fields[prefix + "bbi_rx_ms"]
        anchor = int(fields[prefix + "start_unix_s"])
        if total < 0 or not 0 <= count <= CAPACITY:
            raise ValueError("Invalid BBI snapshot count")
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
        # Validate a whole snapshot before accepting any of its slots. Once the
        # ring is full, a stale slot wraps back by 24 sequences and its callback
        # time precedes the slot before it, even though the count still matches.
        recoverable = schema == 3 and not strict
        reason = None
        samples = []
        if count != min(total, CAPACITY):
            reason = "inconsistent_count"
        elapsed = fields.get(prefix + "elapsed_ms")
        sequences = range(total - count + 1, total + 1) if reason is None else ()
        for sequence in sequences:
            slot = (sequence - 1) % CAPACITY
            value, received_ms = values[slot], times[slot]
            value = int(value) if value is not None and 0 < value < 65535 else None
            if not isinstance(received_ms, int) or received_ms < 0:
                reason = "invalid_callback_time"
                break
            if samples and received_ms < samples[-1][1][1]:
                reason = "nonmonotonic_callback_times"
                break
            if elapsed is not None and received_ms > elapsed:
                reason = "callback_after_snapshot"
                break
            sample = (value, int(received_ms))
            samples.append((sequence, sample))
        if reason is not None:
            if not recoverable:
                raise ValueError(f"Inconsistent BBI snapshot: {reason}")
            rejected[reason] += 1
            continue
        if name == "session":
            final_snapshot_valid = True
        for sequence, sample in samples:
            if sequence in recovered and recovered[sequence] != sample:
                if not recoverable:
                    raise ValueError(f"Conflicting repeated snapshot for interval {sequence}")
                conflicting.add(sequence)
            else:
                recovered[sequence] = sample

    if final_total is not None and final_total < observed_total:
        raise ValueError("Final interval total is smaller than a record total")
    missing_ranges = []
    previous = 0
    rows = []
    for sequence, (value, received_ms) in sorted(recovered.items()):
        if sequence in conflicting:
            continue
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
        "final_snapshot_valid": final_snapshot_valid,
        "final_received_total": final_total,
        "largest_observed_total": observed_total,
        "recovered_intervals": len(rows),
        "invalid_recovered_intervals": sum(row["bbi_ms"] is None for row in rows),
        "missing_received_intervals": sum(end - start + 1 for start, end in missing_ranges),
        "missing_sequence_ranges": missing_ranges,
        "skipped_inconsistent_snapshots": sum(rejected.values()),
        "snapshot_rejections": dict(rejected),
        "conflicting_intervals": len(conflicting),
        "conflicting_sequence_ranges": _sequence_ranges(conflicting),
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
    if rejected:
        report["notes"].append("Inconsistent schema 3 snapshots were skipped; repeated consistent snapshots can recover their intervals.")
    if conflicting:
        report["notes"].append("Intervals with conflicting consistent copies were omitted and counted as missing; no values were guessed.")
    return rows, report


def load_bbi(path, *, strict=False):
    from fitparse import FitFile

    return decode_snapshots(
        ((message.name, message.get_values())
         for message in FitFile(str(path)).get_messages()), strict=strict
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit_file")
    parser.add_argument("--strict", action="store_true", help="Fail instead of recovering inconsistent schema 3 snapshots")
    args = parser.parse_args()
    _, summary = load_bbi(args.fit_file, strict=args.strict)
    print(json.dumps(summary, indent=2))
