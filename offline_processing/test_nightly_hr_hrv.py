import unittest

import numpy as np
import pandas as pd

from nightly_hr_hrv import fixed_hrv_windows, aligned_profiles, minute_hr


class FixedWindowTests(unittest.TestCase):
    def fixture(self):
        start = pd.Timestamp("2026-01-01", tz="UTC")
        end = start + pd.Timedelta(minutes=10)
        intervals = pd.DataFrame({
            "callback_time_utc_approx": pd.date_range(start, periods=600, freq="s"),
            "sequence": np.arange(600), "bbi_clean_ms": np.tile([950., 1050.], 300),
            "segment_id": 0, "delivery_run_id": 0, "interpolated": False})
        segments = pd.DataFrame({"start": [start], "end": [end]})
        return intervals, segments, start, end

    def test_fixed_geometry_and_known_rmssd(self):
        one = fixed_hrv_windows(*self.fixture())
        five = fixed_hrv_windows(*self.fixture(), length_s=300)
        self.assertEqual(len(one), 10)
        self.assertEqual(len(five), 6)
        np.testing.assert_allclose(one.rmssd_ms, 100)
        np.testing.assert_allclose(five.rmssd_ms, 100)
        self.assertEqual(five.minute.tolist(), [2, 3, 4, 5, 6, 7])
        self.assertTrue(one.included.all())

    def test_movement_boundary_and_delivery_discontinuity(self):
        data, segments, start, end = self.fixture()
        segments = pd.DataFrame({"start": [start, start + pd.Timedelta(seconds=150)],
                                 "end": [start + pd.Timedelta(seconds=130), end]})
        data.loc[150:, 'segment_id'] = 1
        data.loc[150:, 'delivery_run_id'] = 1
        result = fixed_hrv_windows(data, segments, start, end)
        self.assertFalse(result.loc[2, 'included'])
        self.assertTrue(result.loc[3, 'included'])
        data.loc[200:, 'sequence'] += 1
        result = fixed_hrv_windows(data, segments, start, end)
        self.assertIn('sequence_discontinuity', result.loc[3, 'exclusion_reason'])
        self.assertTrue(np.isnan(result.loc[3, 'rmssd_ms']))

    def test_interpolation_threshold_and_empty_bbis(self):
        data, segments, start, end = self.fixture()
        data.loc[:3, 'interpolated'] = True
        result = fixed_hrv_windows(data, segments, start, end)
        self.assertIn('too_many_interpolated', result.loc[0, 'exclusion_reason'])
        self.assertTrue(result.loc[1, 'included'])
        result = fixed_hrv_windows(data.iloc[:0], segments, start, end)
        self.assertFalse(result.included.any())

    def test_equal_night_weight_and_missing_bins(self):
        frame = pd.DataFrame({'night': ['a', 'a', 'b', 'a'],
                              'spt_pct': [.1, .2, .3, 2.1], 'rmssd_ms': [10, 20, 60, 30]})
        individual, group = aligned_profiles(frame, 'rmssd_ms', 'percent')
        self.assertEqual(group.loc[0, 'mean'], 37.5)
        self.assertEqual(group.loc[0, 'n_nights'], 2)
        self.assertEqual(group.loc[1, 'n_nights'], 0)
        self.assertTrue(np.isnan(group.loc[1, 'mean']))

    def test_hr_gap_is_not_filled_and_partial_minute_omitted(self):
        start = pd.Timestamp('2026-01-01', tz='UTC')
        times = pd.date_range(start, periods=150, freq='s')
        hr = pd.DataFrame({'timestamp': times, 'heart_rate': 60.})
        hr.loc[70:90, 'heart_rate'] = np.nan
        result = minute_hr(hr, start, start + pd.Timedelta(seconds=150))
        self.assertEqual(len(result), 2)
        self.assertEqual(result.loc[0, 'hr_mean_bpm'], 60)
        self.assertTrue(np.isnan(result.loc[1, 'hr_mean_bpm']))


if __name__ == '__main__':
    unittest.main()
