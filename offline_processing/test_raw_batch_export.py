"""Synthetic raw-export regressions; no personal FIT files or notebook execution."""

from datetime import time
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from fit_bbi import decode_snapshots
from raw_batch_export import (OUTPUT_NAMES, bbi_preview, crop_to_diary,
                              plot_raw_preview, read_raw_fit, save_raw_tables,
                              select_recordings, snapshot_messages)


class Message:
    def __init__(self, name, values):
        self.name, self.values = name, values

    def get_values(self):
        return self.values

    def __iter__(self):
        return iter(())


class RawBatchExportTests(unittest.TestCase):
    def accel(self, start="2026-10-14 21:00", periods=10, freq="h"):
        return pd.DataFrame({"sample_time": pd.date_range(start, periods=periods, freq=freq, tz="UTC"),
                             "accel_x": np.arange(periods), "accel_y": 0.0, "accel_z": 1000.0})

    def diary(self, day="2026-10-14", lights=time(23, 30), wakeup=time(7)):
        return pd.DataFrame({"Date": [pd.Timestamp(day)], "lightsoff": [lights],
                             "wakeup": [wakeup], "gotup": [time(8)], "bedtime": [None]})

    def test_crop_uses_wakeup_not_later_gotup_and_excludes_end(self):
        accel = self.accel()
        cropped, window = crop_to_diary(accel, self.diary())
        self.assertEqual(cropped.accel_x.tolist(), list(range(1, 8)))
        self.assertEqual(window["wakeup"], pd.Timestamp("2026-10-15 07:00", tz="Europe/Rome"))
        self.assertNotIn("gotup", window)
        self.assertTrue(window["full_diary_interval_recorded"])
        pd.testing.assert_frame_equal(accel, self.accel())

    def test_after_midnight_lights_use_the_given_calendar_date(self):
        _, window = crop_to_diary(self.accel(), self.diary("2026-10-15", time(0, 30)))
        self.assertEqual(window["lightsoff"], pd.Timestamp("2026-10-15 00:30", tz="Europe/Rome"))

    def test_dst_and_partial_coverage(self):
        # Rome's autumn clock change adds an elapsed hour to this diary window.
        accel = self.accel("2026-10-24 21:00", 12)
        cropped, window = crop_to_diary(accel, self.diary("2026-10-24", time(23), time(7)))
        self.assertEqual(len(cropped), 9)
        self.assertEqual((window["wakeup"] - window["lightsoff"]).total_seconds(), 9 * 3600)
        cropped, window = crop_to_diary(self.accel(periods=4), self.diary())
        self.assertFalse(window["full_diary_interval_recorded"])
        self.assertEqual(len(cropped), 3)

    def test_ambiguous_or_missing_diary_match_is_not_guessed(self):
        diary = self.diary()
        with self.assertRaisesRegex(ValueError, "Multiple"):
            crop_to_diary(self.accel(), pd.concat([diary, diary], ignore_index=True))
        with self.assertRaisesRegex(ValueError, "No complete"):
            crop_to_diary(self.accel(), self.diary("2026-10-20"))
        with self.assertRaisesRegex(ValueError, "wakeup"):
            crop_to_diary(self.accel(), diary.drop(columns="wakeup"))
        diary["wakeup"] = None
        with self.assertRaisesRegex(ValueError, "No complete"):
            crop_to_diary(self.accel(), diary)

    def test_recording_selection_starts_inclusively_and_keeps_filename_order(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            names = ["2026-01-01-23-00-00.fit", "2026-01-02-23-00-00.FIT", "2026-01-03-23-00-00.fit"]
            for name in reversed(names):
                (folder / name).touch()
            (folder / "notes.txt").touch()
            (folder / "excluded.fit").mkdir()
            self.assertEqual([p.name for p in select_recordings(folder)], names)
            self.assertEqual([p.name for p in select_recordings(folder, start_from=names[1].lower())], names[1:])
            self.assertEqual([p.name for p in select_recordings(
                folder, start_from=names[1], recording_names=[names[0], names[2]])], [names[2]])
            with self.assertRaisesRegex(FileNotFoundError, "START_FROM"):
                select_recordings(folder, start_from="missing.fit")
            with self.assertRaisesRegex(FileNotFoundError, "Requested recordings"):
                select_recordings(folder, recording_names=["missing.fit"])
            with self.assertRaisesRegex(TypeError, "list"):
                select_recordings(folder, recording_names=names[0])

    def test_loader_preserves_legacy_and_final_raw_arrays_without_decoding(self):
        fields = {"timestamp": pd.Timestamp("2026-10-14 21:00"), "timestamp_ms": 10,
                  "sample_time_offset": [0, 10], "calibrated_accel_x": [123, 456],
                  "calibrated_accel_y": [0, 0], "calibrated_accel_z": [1000, 999]}
        messages = [Message("accelerometer_data", fields),
                    Message("record", {"timestamp": fields["timestamp"], "bbi_count": 2,
                                       "bbi_1": 0, "bbi_2": 900}),
                    Message("session", {"timestamp": fields["timestamp"],
                                        "final_bbi_history": (0, 65535, 900),
                                        "final_bbi_rx_ms": (10, 20, 30)})]
        with patch("raw_batch_export.FitFile") as fit, patch("raw_batch_export.decode_snapshots") as decode:
            fit.return_value.get_messages.return_value = iter(messages)
            accel, raw, _ = read_raw_fit("synthetic.fit")
            decode.assert_not_called()
        self.assertEqual(accel.accel_x.tolist(), [123, 456])
        self.assertEqual(raw.iloc[0].bbi_1, 0)
        self.assertEqual(raw.iloc[1].final_bbi_history, [0, 65535, 900])
        self.assertEqual(raw.message_type.tolist(), ["record", "session"])
        self.assertEqual(str(accel.sample_time.dt.tz), "UTC")

    def test_legacy_preview_retains_equal_values_at_record_times(self):
        raw = pd.DataFrame({"message_type": ["record"],
                            "timestamp": pd.to_datetime(["2026-10-14 21:00"], utc=True),
                            "bbi_count": [2], "bbi_1": [900], "bbi_2": [900]})
        points, note = bbi_preview(raw)
        self.assertEqual(points.bbi_ms.tolist(), [900, 900])
        self.assertEqual(points.time.nunique(), 1)
        self.assertIn("Legacy", note)

    def test_corrupt_decode_only_disables_preview_not_raw_export(self):
        raw = pd.DataFrame({"message_type": ["record"], "bbi_snapshot": [[4]]})
        points, note = bbi_preview(raw)
        self.assertTrue(points.empty)
        self.assertIn("Original snapshots will still be saved", note)

    def test_all_numbered_schemas_decode_after_raw_parquet_roundtrip(self):
        from test_fit_bbi import snapshot, compact_snapshot, packed_snapshot

        for fixture in (snapshot, compact_snapshot, packed_snapshot):
            with self.subTest(schema=fixture.__name__), TemporaryDirectory() as directory:
                messages = [fixture(2), fixture(3, "session")]
                raw = pd.DataFrame([{"message_type": name, **fields} for name, fields in messages])
                path = Path(directory) / "raw_bbi.parquet"
                raw.to_parquet(path, index=False)
                loaded = pd.read_parquet(path)
                rows, report = decode_snapshots(snapshot_messages(loaded))
                self.assertEqual([r["sequence"] for r in rows], [1, 2, 3])
                self.assertTrue(report["final_summary_present"])
                self.assertEqual(report["missing_received_intervals"], 0)
                points, _ = bbi_preview(loaded)
                self.assertEqual(len(points), 3)

    def test_empty_bbi_still_exports_a_readable_raw_table(self):
        with patch("raw_batch_export.FitFile") as fit:
            fit.return_value.get_messages.return_value = iter([
                Message("accelerometer_data", {"timestamp": pd.Timestamp("2026-10-14 21:00"),
                    "sample_time_offset": [0, 10], "calibrated_accel_x": [0, 1],
                    "calibrated_accel_y": [0, 0], "calibrated_accel_z": [1000, 1000]})])
            _, raw, _ = read_raw_fit("synthetic.fit")
        self.assertTrue(raw.empty)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "empty_bbi.parquet"
            raw.to_parquet(path, index=False)
            self.assertTrue(pd.read_parquet(path).empty)
            self.assertIn("timestamp", pd.read_parquet(path))

    def test_two_raw_parquets_roundtrip_and_existing_files_are_protected(self):
        cropped, window = crop_to_diary(self.accel(), self.diary())
        raw = pd.DataFrame({"message_index": [1], "message_type": ["session"],
                            "timestamp": pd.to_datetime(["2026-10-15 08:00"], utc=True),
                            "final_bbi_history": [[0, 65535, 900]], "final_bbi_rx_ms": [[10, 20, 30]]})
        with TemporaryDirectory() as directory:
            source = Path(directory) / "synthetic.fit"
            source.write_bytes(b"synthetic fixture, not a FIT recording")
            output = Path(directory) / "garmin_processed_new" / source.stem
            paths = save_raw_tables(output, cropped, raw, window, source, {})
            self.assertEqual(sorted(p.name for p in output.iterdir()), sorted(OUTPUT_NAMES))
            pd.testing.assert_frame_equal(pd.read_parquet(paths[0]), cropped)
            self.assertEqual(pd.read_parquet(paths[1]).iloc[0].final_bbi_history.tolist(), [0, 65535, 900])
            # BBI outside the diary window remains in the raw full-recording table.
            self.assertGreater(pd.read_parquet(paths[1]).timestamp.iloc[0], window["wakeup"])
            metadata = json.loads(pq.read_schema(paths[0]).metadata[b"overnight_raw_export"])
            self.assertEqual(metadata["window"]["diary_excel_row"], 2)
            self.assertEqual(metadata["export_version"], 2)
            self.assertIn("wakeup", metadata["window"])
            self.assertNotIn("gotup", metadata["window"])
            self.assertIn("FULL recording", metadata["bbi"])
            before = [p.read_bytes() for p in paths]
            with self.assertRaises(FileExistsError):
                save_raw_tables(output, cropped, raw, window, source, {})
            self.assertEqual([p.read_bytes() for p in paths], before)

    def test_preview_has_two_axes_and_both_diary_boundaries(self):
        import matplotlib.pyplot as plt

        accel = self.accel()
        _, window = crop_to_diary(accel, self.diary())
        fig = plot_raw_preview(accel, pd.DataFrame(), window, "Synthetic recording", max_acc_points=4)
        try:
            self.assertEqual(len(fig.axes), 2)
            for axis in fig.axes:
                self.assertTrue({"Diary lightsoff", "Diary wakeup"}.issubset(
                    {line.get_label() for line in axis.lines}))
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
