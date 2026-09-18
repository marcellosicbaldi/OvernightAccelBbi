import unittest

from fit_bbi import CAPACITY, decode_snapshots


def snapshot(total, name="record", overrides=None):
    prefix = "final_" if name == "session" else ""
    values = [0] * CAPACITY
    times = [0] * CAPACITY
    for seq in range(max(1, total - CAPACITY + 1), total + 1):
        slot = (seq - 1) % CAPACITY
        values[slot] = 1000
        times[slot] = seq * 1000
    fields = {
        "logger_schema": 2, "bbi_total": total,
        "bbi_history_count": min(total, CAPACITY),
        "bbi_history": values, "bbi_rx_ms": times,
        "start_unix_s": 1781526925,
    }
    fields.update(overrides or {})
    return name, {prefix + key: value for key, value in fields.items()}


class DecodeTests(unittest.TestCase):
    def test_compact_schema_recovers_records_and_final_tail(self):
        messages = []
        for total, name in ((10, "record"), (30, "record"), (35, "session")):
            _, fields = snapshot(total, name)
            prefix = "final_" if name == "session" else ""
            fields.pop(prefix + "logger_schema")
            count = fields.pop(prefix + "bbi_history_count")
            anchor = fields.pop(prefix + "start_unix_s")
            fields[prefix + "logger_meta"] = (3, count, 30, 0, 0, 3000, 35000, anchor)
            messages.append((name, fields))
        rows, report = decode_snapshots(messages)
        self.assertEqual(report["schema"], 3)
        self.assertEqual(len(rows), 35)
        self.assertTrue(report["final_summary_present"])
        self.assertEqual(report["missing_received_intervals"], 0)

    def test_invalid_compact_metadata_rejected(self):
        for meta in ((3,), (4, 0, 0, 0, 0, 0, 0, 0), (3, None, 0, 0, 0, 0, 0, 0)):
            with self.subTest(meta=meta), self.assertRaises(ValueError):
                decode_snapshots([("record", {"logger_meta": meta})])

    def test_equal_intervals_are_not_deduplicated_by_value(self):
        rows, report = decode_snapshots([snapshot(2), snapshot(2), snapshot(3, "session")])
        self.assertEqual([row["sequence"] for row in rows], [1, 2, 3])
        self.assertEqual(report["missing_received_intervals"], 0)
        self.assertTrue(report["final_summary_present"])

    def test_skipped_records_recovered_from_history_and_tail(self):
        rows, report = decode_snapshots([snapshot(10), snapshot(30), snapshot(35, "session")])
        self.assertEqual(len(rows), 35)
        self.assertEqual(rows[-1]["callback_elapsed_ms"], 35000)
        self.assertEqual(report["missing_received_intervals"], 0)

    def test_history_overrun_is_reported(self):
        _, report = decode_snapshots([snapshot(2), snapshot(30, "session")])
        self.assertEqual(report["missing_sequence_ranges"], [(3, 6)])
        self.assertEqual(report["missing_received_intervals"], 4)

    def test_lost_initial_records(self):
        _, report = decode_snapshots([snapshot(30, "session")])
        self.assertEqual(report["missing_sequence_ranges"], [(1, 6)])

    def test_no_bbi_is_not_legacy(self):
        rows, report = decode_snapshots([snapshot(0), snapshot(0, "session")])
        self.assertEqual(rows, [])
        self.assertEqual(report["schema"], 2)
        self.assertEqual(report["final_received_total"], 0)

    def test_missing_final_summary(self):
        _, report = decode_snapshots([snapshot(4)])
        self.assertFalse(report["final_summary_present"])

    def test_legacy(self):
        _, report = decode_snapshots([("record", {"bbi_total": 0})])
        self.assertIsNone(report["schema"])

    def test_invalid_interval_preserves_sequence(self):
        name, fields = snapshot(2)
        fields["bbi_history"][0] = 0
        rows, report = decode_snapshots([(name, fields)])
        self.assertIsNone(rows[0]["bbi_ms"])
        self.assertEqual(report["invalid_recovered_intervals"], 1)
        self.assertEqual(report["missing_received_intervals"], 0)

    def test_conflicting_snapshot_is_rejected(self):
        name, fields = snapshot(2)
        fields["bbi_history"][0] = 999
        with self.assertRaises(ValueError):
            decode_snapshots([snapshot(2), (name, fields)])

    def test_bad_schema_shape_count_and_multiple_sessions(self):
        for overrides in ({"logger_schema": 3}, {"bbi_history": [1]},
                          {"bbi_history_count": 1}, {"start_unix_s": 0}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                decode_snapshots([snapshot(2), snapshot(2, overrides=overrides)])


if __name__ == "__main__":
    unittest.main()
