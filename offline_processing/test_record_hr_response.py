"""Regression checks for the notebook's FIT-recorded-HR input path."""

import unittest

import numpy as np
import pandas as pd

from burst_hr_response import analyze_burst_hr, record_hr_to_observations, record_times_to_utc, sample_hr


class RecordHrTests(unittest.TestCase):
    def test_local_wall_time_and_input_not_modified(self):
        source = pd.DataFrame({"timestamp": pd.to_datetime(["2026-09-13 23:30", "2026-09-13 23:31"]),
                               "heart_rate": [60, 65], "bbi_latest": [500, 500]})
        original = source.copy(deep=True)
        obs, report = record_hr_to_observations(source, naive_timezone="Europe/Rome")
        self.assertEqual(obs.index[0], pd.Timestamp("2026-09-13 21:30", tz="UTC"))
        np.testing.assert_array_equal(obs.hr_bpm, [60, 65])
        self.assertEqual(report["value_column"], "heart_rate")
        pd.testing.assert_frame_equal(source, original)

    def test_already_indexed_and_renamed_table(self):
        source = pd.DataFrame({"hr_bpm": [70.]}, index=pd.DatetimeIndex(["2026-09-13 23:30"]))
        obs, _ = record_hr_to_observations(source, naive_timezone="Europe/Rome")
        self.assertEqual(obs.iloc[0].hr_bpm, 70)
        self.assertEqual(obs.index[0].hour, 21)

    def test_aware_times_are_not_shifted_twice(self):
        times = pd.date_range("2026-09-13 23:30", periods=2, freq="s", tz="Europe/Rome")
        local = record_times_to_utc(times, naive_timezone="Europe/Rome")
        utc = record_times_to_utc(times.tz_convert("UTC"), naive_timezone="Europe/Rome")
        pd.testing.assert_index_equal(local, utc)

    def test_dst_offset_is_not_fixed_at_two_hours(self):
        result = record_times_to_utc(["2026-01-01 23:30"], naive_timezone="Europe/Rome")
        self.assertEqual(result[0].hour, 22)

    def test_invalid_rows_and_duplicates_remain_barriers(self):
        source = pd.DataFrame({"timestamp": pd.to_datetime([
            "2026-01-01 00:00:02", "2026-01-01 00:00:00",
            "2026-01-01 00:00:01", "2026-01-01 00:00:01"]), "heart_rate": [80, 60, 70, 0]})
        obs, report = record_hr_to_observations(source)
        self.assertTrue(np.isnan(obs.iloc[1].hr_bpm))
        self.assertEqual(report["duplicate_timestamps_averaged"], 1)
        values, _, _ = sample_hr(obs, pd.to_datetime(["2026-01-01 00:00:00.5", "2026-01-01 00:00:01.5"]))
        self.assertTrue(np.isnan(values).all())

    def test_valid_duplicate_mean(self):
        source = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-01"] * 2), "heart_rate": [60, 80]})
        obs, _ = record_hr_to_observations(source)
        self.assertEqual(obs.iloc[0].hr_bpm, 70)

    def test_long_gaps_and_extrapolation_stay_missing(self):
        source = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-01", "2026-01-01 00:00:10"], format="mixed"),
                               "heart_rate": [60, 80]})
        obs, _ = record_hr_to_observations(source)
        values, _, _ = sample_hr(obs, pd.to_datetime(["2025-12-31 23:59:59", "2026-01-01 00:00:05"]))
        self.assertTrue(np.isnan(values).all())

    def test_record_hr_drives_known_burst_response_not_bbi_fields(self):
        times = pd.date_range("2026-09-13 23:30", periods=180, freq="s")
        rates = np.full(180, 60.)
        rates[61:70] = 90.
        source = pd.DataFrame({"timestamp": times, "heart_rate": rates, "bbi_latest": 2000})
        obs, _ = record_hr_to_observations(source, naive_timezone="Europe/Rome")
        bursts = pd.DataFrame({"start": record_times_to_utc([times[60]], "Europe/Rome"),
                               "end": record_times_to_utc([times[62]], "Europe/Rome"), "AUC": [1.]})
        result = analyze_burst_hr(bursts, obs, obs.index[0], obs.index[-1], isolation_s=10)
        event = result["events"].iloc[0]
        self.assertTrue(event.included)
        self.assertAlmostEqual(event.baseline_bpm, 60)
        self.assertAlmostEqual(event.hr_peak_increase_pct, 50)

    def test_empty_and_missing_hr(self):
        source = pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns]"), "heart_rate": pd.Series(dtype=float)})
        obs, report = record_hr_to_observations(source)
        self.assertTrue(obs.empty)
        self.assertEqual(report["input_rows"], 0)
        with self.assertRaises(ValueError):
            record_hr_to_observations(pd.DataFrame({"bbi_latest": [1000]}))

    def test_invalid_time_and_limits_raise(self):
        source = pd.DataFrame({"timestamp": [pd.NaT], "heart_rate": [60]})
        with self.assertRaises(ValueError):
            record_hr_to_observations(source)
        with self.assertRaises(ValueError):
            record_hr_to_observations(source, hr_limits_bpm=(200, 30))


if __name__ == "__main__":
    unittest.main()
