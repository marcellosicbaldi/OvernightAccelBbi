"""FIT batch timing and one-pass loading regressions."""

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

import numpy as np
import pandas as pd

from fit_recording import expand_acceleration_batch, load_recording


class FitRecordingTests(unittest.TestCase):
    def fields(self):
        return {"timestamp": pd.Timestamp("2026-01-01"), "timestamp_ms": 120,
                "sample_time_offset": [10, 20], "calibrated_accel_x": [0, 1],
                "calibrated_accel_y": [0, 0], "calibrated_accel_z": [1000, 999]}

    def test_sample_offsets_add_to_batch_timestamp(self):
        times, xyz = expand_acceleration_batch(self.fields())
        base = pd.Timestamp("2026-01-01", tz="UTC").value
        np.testing.assert_array_equal(times - base, [130_000_000, 140_000_000])
        np.testing.assert_array_equal(xyz[:, 2], [1000, 999])

    def test_missing_timing_rejected(self):
        for key in ["timestamp", "sample_time_offset", "calibrated_accel_x"]:
            fields = self.fields()
            fields.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                expand_acceleration_batch(fields)

    def test_bad_batch_rejected(self):
        for key, value in [("sample_time_offset", [1]), ("sample_time_offset", [0, np.nan]),
                           ("calibrated_accel_z", [1000]), ("timestamp_ms", -1)]:
            fields = self.fields()
            fields[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                expand_acceleration_batch(fields)

    def test_no_accelerometer_messages(self):
        with patch("fit_recording.FitFile") as fit:
            fit.return_value.get_messages.return_value = []
            with self.assertRaisesRegex(ValueError, "No calibrated"):
                load_recording(Path(__file__))

    def test_empty_accelerometer_batch_is_not_a_recording(self):
        fields = self.fields()
        for key in ["sample_time_offset", "calibrated_accel_x", "calibrated_accel_y", "calibrated_accel_z"]:
            fields[key] = []
        message = MagicMock()
        message.name = "accelerometer_data"
        message.get_values.return_value = fields
        message.__iter__.return_value = iter([])
        with patch("fit_recording.FitFile") as fit:
            fit.return_value.get_messages.return_value = [message]
            with self.assertRaisesRegex(ValueError, "No calibrated"):
                load_recording(Path(__file__))

    def test_one_pass_preserves_record_and_session_bbi(self):
        class Message:
            def __init__(self, name, fields):
                self.name, self.fields = name, fields

            def get_values(self):
                return self.fields

            def __iter__(self):
                return iter([])

        with patch("fit_recording.FitFile") as fit, patch("fit_recording.decode_snapshots", return_value=([], {})) as decode:
            fit.return_value.get_messages.return_value = iter([
                Message("accelerometer_data", self.fields()), Message("record", {"id": 1}), Message("session", {"id": 2})])
            result = load_recording(Path(__file__))
            self.assertEqual(len(result["accel_df"]), 2)
            fit.assert_called_once()
            decode.assert_called_once_with([("record", {"id": 1}), ("session", {"id": 2})])


if __name__ == "__main__":
    unittest.main()
