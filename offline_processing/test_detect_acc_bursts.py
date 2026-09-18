import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import neurokit2 as nk

from detect_acc_bursts import (
    _burst_ranges, compute_envelope, detect_bursts, hl_envelopes_idx, prepare_acceleration,
)


def signal(values, fs=100, tz=None):
    return pd.Series(values, index=pd.date_range("2026-09-12", periods=len(values),
                                               freq=pd.to_timedelta(1 / fs, unit="s"), tz=tz))


class BurstTests(unittest.TestCase):
    def test_flat_signal_returns_typed_empty_table(self):
        bursts = detect_bursts(signal(np.ones(2000), tz="Europe/Rome"), 100, alfa=0.020)
        self.assertTrue(bursts.empty)
        self.assertEqual(list(bursts), ["start", "end", "duration", "peak-to-peak", "AUC"])
        self.assertEqual(str(bursts.start.dt.tz), "Europe/Rome")
        self.assertEqual(bursts["AUC"].dtype, float)

    def test_missing_or_invalid_threshold_rejected(self):
        for threshold in (None, -1, 0, np.nan, np.inf, "0.020"):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                detect_bursts(signal(np.ones(100)), 100, alfa=threshold)

    def test_short_inputs_and_nyquist_rejected(self):
        for count in (0, 1, 51):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "51 samples"):
                detect_bursts(signal(np.ones(count)), 100, alfa=0.020)
        for fs in (0, 20, np.nan):
            with self.subTest(fs=fs), self.assertRaises(ValueError):
                detect_bursts(signal(np.ones(100)), fs, alfa=0.020)

    def test_missing_and_nonfinite_samples_rejected(self):
        for bad in (np.nan, np.inf):
            acc = signal(np.ones(100))
            acc.iloc[20] = bad
            with self.assertRaisesRegex(ValueError, "non-finite"):
                detect_bursts(acc, 100, alfa=0.020)

    def test_bad_indices_and_wrong_sampling_rate_rejected(self):
        acc = signal(np.ones(100))
        cases = [acc.iloc[::-1], pd.concat([acc.iloc[:1], acc]),
                 acc.drop(acc.index[25]), acc.set_axis(acc.index.where(np.arange(100) != 0))]
        for case in cases:
            with self.assertRaises(ValueError):
                detect_bursts(case, 100, alfa=0.020)
        with self.assertRaises(TypeError):
            detect_bursts(pd.Series(np.ones(100)), 100, alfa=0.020)
        with self.assertRaisesRegex(ValueError, "sampling_rate"):
            detect_bursts(acc, 50, alfa=0.020)

    def test_constant_monotonic_and_empty_envelopes(self):
        for values in ([], [1], [1, 1, 1], np.arange(100)):
            acc = signal(values)
            for resample in (True, False):
                env = compute_envelope(acc, resample=resample)
                self.assertEqual(len(env), len(acc))
                self.assertTrue((env == 0).all())

    def test_extrema_validation(self):
        for size in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                hl_envelopes_idx([0, 1, 0], dmin=size)
        with self.assertRaises(ValueError):
            hl_envelopes_idx([[0, 1]])

    def test_envelopes_interpolate_independently(self):
        acc = signal(np.sin(np.arange(2345) / 13) * np.linspace(0.005, 0.05, 2345))
        low, high = hl_envelopes_idx(acc.values, dmin=10, dmax=10)
        positions = np.arange(len(acc))
        expected = np.maximum(np.interp(positions, high, acc.values[high]) -
                              np.interp(positions, low, acc.values[low]), 0)
        np.testing.assert_allclose(compute_envelope(acc), expected)
        np.testing.assert_allclose(compute_envelope(acc, resample=False), expected[high])

    def test_boundary_runs_and_exact_threshold(self):
        score = signal([0.04, 0.04, 0.02, 0.0, 0.03])
        end = score.index[-1] + pd.Timedelta(milliseconds=10)
        ranges = _burst_ranges(score, 0.02, score.index[0], end, 0)
        self.assertEqual(ranges, [[score.index[0], score.index[2]], [score.index[4], end]])

    def test_chained_merging_and_exact_five_second_gap(self):
        index = pd.date_range("2026-09-12", periods=20, freq="1s", tz="UTC")
        score = pd.Series([0.04, 0, 0.04, 0, 0.04] + [0] * 5 + [0.04] + [0] * 9, index=index)
        ranges = _burst_ranges(score, 0.02, index[0], index[-1], 5)
        self.assertEqual(ranges, [[index[0], index[5]], [index[10], index[11]]])

    def test_auc_uses_seconds_and_final_sample_interval(self):
        for fs in (50, 100):
            acc = signal(np.linspace(1, 1.1, fs * 10), fs=fs, tz="Europe/Rome")
            envelope = pd.Series(0.04, index=acc.index)
            with patch("detect_acc_bursts.nk.signal_filter", return_value=acc.values), \
                 patch("detect_acc_bursts.compute_envelope", return_value=envelope):
                bursts = detect_bursts(acc, fs, alfa=0.020)
            self.assertEqual(len(bursts), 1)
            self.assertAlmostEqual(bursts["AUC"].iloc[0], 0.4)
            self.assertEqual(bursts.duration.iloc[0], pd.Timedelta(seconds=10))
            self.assertAlmostEqual(bursts["peak-to-peak"].iloc[0], 0.1)
            self.assertEqual(str(bursts.start.dt.tz), "Europe/Rome")

    def test_auc_includes_merged_gap_in_seconds(self):
        acc = signal(np.ones(1000))
        env = signal(np.r_[np.zeros(100), np.full(100, 0.04), np.zeros(100),
                          np.full(100, 0.04), np.zeros(600)])
        with patch("detect_acc_bursts.nk.signal_filter", return_value=acc.values), \
             patch("detect_acc_bursts.compute_envelope", return_value=env):
            bursts = detect_bursts(acc, 100, alfa=0.020)
        self.assertEqual(len(bursts), 1)
        self.assertEqual(bursts.duration.iloc[0], pd.Timedelta(seconds=3))
        self.assertAlmostEqual(bursts["AUC"].iloc[0], 0.0798)

    def test_real_filter_and_burst_detection(self):
        time = np.arange(12000) / 100
        amplitude = np.where((time >= 40) & (time < 60), 0.04, 0.001)
        acc = signal(1 + amplitude * np.sin(2 * np.pi * 3 * time))
        bursts, diagnostics = detect_bursts(acc, 100, alfa=0.020, return_signals=True)
        expected = nk.signal_filter(acc.values, sampling_rate=100, lowcut=0.1,
                                    highcut=10, method="butterworth", order=8)
        np.testing.assert_allclose(diagnostics["filtered"], expected)
        self.assertEqual(len(bursts), 1)
        self.assertLess(bursts.start.iloc[0], acc.index[4500])
        self.assertGreater(bursts.end.iloc[0], acc.index[5500])
        self.assertTrue(np.isfinite(bursts[["peak-to-peak", "AUC"]]).all().all())

    def test_sparse_envelope_and_std_modes(self):
        time = np.arange(1234) / 100
        acc = signal(1 + (0.001 + time / 1000) * np.sin(2 * np.pi * 3 * time))
        for options in ({"resample_envelope": False, "alfa": 0.005},
                        {"envelope": False, "alfa": 2.0}):
            bursts, diagnostics = detect_bursts(acc, 100, return_signals=True, **options)
            self.assertGreater(len(bursts), 0)
            self.assertTrue(np.isfinite(bursts["AUC"]).all())
            self.assertLess(len(diagnostics["score"]), len(acc))

    def test_std_single_sample_edge_bin_does_not_make_auc_nan(self):
        time = np.arange(1201) / 100
        acc = signal(1 + (0.001 + time / 1000) * np.sin(2 * np.pi * 3 * time))
        bursts = detect_bursts(acc, 100, envelope=False, alfa=2.0)
        self.assertGreater(len(bursts), 0)
        self.assertTrue(np.isfinite(bursts["AUC"]).all())


class PreparationTests(unittest.TestCase):
    def table(self):
        return pd.DataFrame({"sample_time": pd.date_range("2026-09-12", periods=100, freq="10ms"),
                             "accel_x": 0.0, "accel_y": 0.0, "accel_z": 1000.0})

    def test_explicit_units_and_duplicate_sorting(self):
        table = self.table()
        table = pd.concat([table.iloc[::-1], table.iloc[[10]]], ignore_index=True)
        magnitude, quality = prepare_acceleration(table)
        np.testing.assert_allclose(magnitude, 1)
        self.assertTrue(magnitude.index.is_unique)
        self.assertTrue(magnitude.index.is_monotonic_increasing)
        self.assertEqual(quality["duplicate_timestamps_averaged"], 1)
        self.assertEqual(quality["regular_samples"], 100)
        table["accel_z"] /= 1000
        converted, _ = prepare_acceleration(table, input_unit="g")
        pd.testing.assert_series_equal(magnitude, converted)

    def test_jitter_regularized_without_extrapolation(self):
        table = self.table()
        table.loc[10:20, "sample_time"] += pd.Timedelta(milliseconds=2)
        result, _ = prepare_acceleration(table)
        self.assertEqual(result.index.freq, pd.Timedelta(milliseconds=10))
        self.assertLessEqual(result.index[-1], table.sample_time.max())

    def test_large_gap_rejected(self):
        table = self.table().drop(index=range(20, 60))
        with self.assertRaisesRegex(ValueError, "Split into continuous segments"):
            prepare_acceleration(table)

    def test_missing_samples_timestamps_and_bad_units_rejected(self):
        table = self.table()
        for column, value in (("sample_time", pd.NaT), ("accel_x", np.nan)):
            bad = table.copy()
            bad.loc[2, column] = value
            with self.assertRaises(ValueError):
                prepare_acceleration(bad)
        with self.assertRaises(ValueError):
            prepare_acceleration(table, input_unit="counts")


if __name__ == "__main__":
    unittest.main()
