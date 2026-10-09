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


def compact_snapshot(total, name="record", overrides=None):
    name, fields = snapshot(total, name, overrides)
    prefix = "final_" if name == "session" else ""
    fields.pop(prefix + "logger_schema")
    count = fields.pop(prefix + "bbi_history_count")
    anchor = fields.pop(prefix + "start_unix_s")
    fields[prefix + "logger_meta"] = (3, count, total, 0, 0, 0, total * 1000, anchor)
    return name, fields


def packed_snapshot(total, name="record"):
    _, fields = snapshot(total)
    packed = [4, total, total, 0, 0, 0, total * 1000, fields["start_unix_s"]]
    packed += fields["bbi_history"] + fields["bbi_rx_ms"]
    prefix = "final_" if name == "session" else ""
    # The separately published scalar may belong to another callback.
    return name, {prefix + "bbi_snapshot": packed, prefix + "bbi_total": total + 2}


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

    def test_torn_count_recovered_from_next_snapshot(self):
        torn = compact_snapshot(6)
        torn[1]["bbi_total"] = 8
        rows, report = decode_snapshots(iter([compact_snapshot(5), torn, compact_snapshot(8, "session")]))
        self.assertEqual([row["sequence"] for row in rows], list(range(1, 9)))
        self.assertEqual(report["snapshot_rejections"], {"inconsistent_count": 1})
        self.assertEqual(report["missing_received_intervals"], 0)

    def test_stale_wrapped_slot_rejected_before_accepting_any_samples(self):
        torn = compact_snapshot(25)
        torn[1]["bbi_total"] = 26  # Slot 1 still contains interval 2, not 26.
        rows, report = decode_snapshots([compact_snapshot(24), torn, compact_snapshot(26, "session")])
        self.assertEqual(len(rows), 26)
        self.assertEqual(rows[-1]["callback_elapsed_ms"], 26000)
        self.assertEqual(report["snapshot_rejections"], {"nonmonotonic_callback_times": 1})
        self.assertEqual(report["conflicting_intervals"], 0)

    def test_stale_metadata_rejected_and_same_callback_times_allowed(self):
        torn = compact_snapshot(3)
        torn[1]["logger_meta"] = (3, 3, 2, 0, 0, 0, 2000, 1781526925)
        valid = compact_snapshot(3, "session", {"bbi_rx_ms": [2000, 2000, 2000] + [0] * 21})
        rows, report = decode_snapshots([torn, valid])
        self.assertEqual(len(rows), 3)
        self.assertEqual(report["snapshot_rejections"], {"callback_after_snapshot": 1})

    def test_ambiguous_values_omitted_even_if_majority_agrees(self):
        changed = compact_snapshot(2, overrides={"bbi_history": [1000, 999] + [0] * 22})
        for messages in ([compact_snapshot(2)] * 3 + [changed],
                         [changed] + [compact_snapshot(2)] * 3):
            rows, report = decode_snapshots(messages)
            self.assertEqual([row["sequence"] for row in rows], [1])
            self.assertEqual(report["conflicting_sequence_ranges"], [(2, 2)])
            self.assertEqual(report["missing_sequence_ranges"], [(2, 2)])

    def test_skipped_final_snapshot_keeps_total_and_reports_missing_tail(self):
        torn = compact_snapshot(2, "session")
        torn[1]["final_bbi_total"] = 3
        rows, report = decode_snapshots([compact_snapshot(1), torn])
        self.assertEqual(len(rows), 1)
        self.assertEqual(report["final_received_total"], 3)
        self.assertTrue(report["final_summary_present"])
        self.assertFalse(report["final_snapshot_valid"])
        self.assertEqual(report["missing_sequence_ranges"], [(2, 3)])

    def test_strict_mode_rejects_torn_or_conflicting_compact_snapshots(self):
        torn_count = compact_snapshot(2)
        torn_count[1]["bbi_total"] = 3
        torn_slots = compact_snapshot(25)
        torn_slots[1]["bbi_total"] = 26
        changed = compact_snapshot(2, overrides={"bbi_history": [1000, 999] + [0] * 22})
        for messages in ([torn_count], [torn_slots], [compact_snapshot(2), changed]):
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                decode_snapshots(messages, strict=True)

    def test_recovery_still_rejects_mixed_sessions_and_malformed_arrays(self):
        for other in (compact_snapshot(2, overrides={"start_unix_s": 0}),
                      compact_snapshot(2, overrides={"bbi_history": [1]}),
                      packed_snapshot(2)):
            with self.subTest(other=other), self.assertRaises(ValueError):
                decode_snapshots([compact_snapshot(2), other])

    def test_packed_schema_uses_embedded_total_and_recovers_tail(self):
        rows, report = decode_snapshots([packed_snapshot(10), packed_snapshot(30), packed_snapshot(35, "session")])
        self.assertEqual(report["schema"], 4)
        self.assertEqual(report["final_received_total"], 35)
        self.assertEqual(len(rows), 35)
        self.assertEqual(report["missing_received_intervals"], 0)
        self.assertTrue(report["final_snapshot_valid"])

    def test_packed_zero_bbi_and_invalid_values_preserve_sequence(self):
        rows, report = decode_snapshots([packed_snapshot(0, "session")])
        self.assertEqual(rows, [])
        self.assertEqual(report["schema"], 4)
        invalid = packed_snapshot(2)
        invalid[1]["bbi_snapshot"][8] = 0
        rows, report = decode_snapshots([invalid])
        self.assertEqual(rows[0]["sequence"], 1)
        self.assertIsNone(rows[0]["bbi_ms"])

    def test_packed_schema_rejects_corrupt_payload(self):
        for index, value in ((0, 5), (8, None), (32, 3000)):
            bad = packed_snapshot(2)
            bad[1]["bbi_snapshot"][index] = value
            with self.subTest(index=index), self.assertRaises(ValueError):
                decode_snapshots([bad])
        with self.assertRaises(ValueError):
            decode_snapshots([("record", {"bbi_snapshot": [4]})])


if __name__ == "__main__":
    unittest.main()
