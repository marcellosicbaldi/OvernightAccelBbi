"""Synthetic validation of lag direction, window geometry, QC and aggregation."""
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from polar_ccf import (CCFConfig, aggregate_curves, analyze_quiet_ccf, availability,
                       extract_beats, lowpass, normalized_ccf, plot_ccf, plot_extrema,
                       plot_representative, quiet_periods, representative_signals,
                       sensor_clock_index, union_duration_s)
from ppg_fiducials import def_bandpass, get_peak_onset


START = pd.Timestamp("2020-01-01 23:50", tz="UTC")


def bounds(pairs):
    out = pd.DataFrame([(START + pd.Timedelta(seconds=a), START + pd.Timedelta(seconds=b))
                        for a, b in pairs], columns=["start", "end"])
    for name in ("start", "end"):
        out[name] = pd.to_datetime(out[name], utc=True)
    return out


def synthetic_beats(duration=1300):
    t = np.arange(duration)
    times = START + pd.to_timedelta(t, unit="s")
    pp = 1 + .05 * np.sin(2 * np.pi * .07 * t)
    return pd.DataFrame({"pp_time": times + pd.Timedelta(seconds=.5),
                         "pp_start": times, "pp_end": times + pd.Timedelta(seconds=1),
                         "pwa_time": times + pd.Timedelta(seconds=1.2),
                         "pp_s": pp, "hr_bpm": 60 / pp,
                         "pwa": 2 + .1 * np.sin(2 * np.pi * .07 * (t - 3)), "valid": True})


class CCFTests(unittest.TestCase):
    def test_delay_has_positive_lag_and_reverse_has_negative(self):
        rng = np.random.default_rng(432)
        x = rng.normal(size=1200)
        y = np.r_[np.zeros(12), x[:-12]]
        lags, curve = normalized_ccf(x, y)
        self.assertEqual(lags[np.argmax(curve)], 3)
        reverse = normalized_ccf(y, x)[1]
        np.testing.assert_allclose(curve, reverse[::-1])
        self.assertEqual(lags[np.argmax(reverse)], -3)

    def test_normalization_demeans_and_is_scale_invariant(self):
        x = np.random.default_rng(1).normal(size=1200)
        lag, curve = normalized_ccf(x, x)
        self.assertAlmostEqual(curve[lag == 0][0], 1)
        self.assertLessEqual(abs(curve).max(), 1 + 1e-12)
        shifted = normalized_ccf(x * 1e-12 + 2e-12, x * 1e-12)[1]
        np.testing.assert_allclose(curve, shifted, atol=1e-12)
        np.testing.assert_allclose(normalized_ccf(x, -x)[1], -curve)

    def test_fixed_full_window_energy_normalization(self):
        x = np.arange(1200, dtype=float)
        lags, c = normalized_ccf(x, x)
        v = x - x.mean()
        self.assertAlmostEqual(c[lags == 25][0], np.dot(v[:-100], v[100:]) / np.dot(v, v))

    def test_bad_signals_rejected(self):
        for x, y in [(np.ones(1200), np.ones(1200)), ([1, np.nan], [1, 2]), ([1, 2], [1])]:
            with self.assertRaises(ValueError):
                normalized_ccf(x, y)

    def test_lowpass_attenuation_without_time_shift(self):
        t = np.arange(1200) / 4
        slow = np.sin(2 * np.pi * .05 * t)
        fast = np.sin(2 * np.pi * .5 * t)
        filtered = lowpass(slow + fast)
        middle = slice(160, -160)
        self.assertLess(np.std((filtered - slow)[middle]), .005)
        self.assertEqual(normalized_ccf(slow[middle], filtered[middle])[0][
            np.argmax(normalized_ccf(slow[middle], filtered[middle])[1])], 0)

    def test_configuration_validates_inconsistent_options(self):
        for options in [{"step_s": 301}, {"max_lag_s": 300}, {"fs": 0},
                        {"lowpass_hz": 2}, {"filter_order": 0}, {"pre_burst_s": -2},
                        {"max_bad_fraction": 1.1}, {"window_s": 300.1}]:
            with self.assertRaises(ValueError):
                CCFConfig(**options)


class QuietTests(unittest.TestCase):
    def test_asymmetric_guards_no_recording_edges(self):
        q = quiet_periods(bounds([(10, 20), (400, 410), (800, 810)]), bounds([(0, 1000)]))
        self.assertEqual(list(q.start), list(bounds([(25, 398), (415, 798)]).start))
        self.assertEqual(list(q.end), list(bounds([(25, 398), (415, 798)]).end))
        self.assertEqual(list(q.geometric_windows), [2, 2])

    def test_expanded_overlap_and_acquisition_gaps(self):
        q = quiet_periods(bounds([(0, 10), (14, 15), (800, 810)]),
                          bounds([(0, 300), (400, 1000)]))
        self.assertEqual(len(q), 2)
        self.assertEqual(q.start.iloc[0], START + pd.Timedelta(seconds=20))
        self.assertEqual(q.end.iloc[0], START + pd.Timedelta(seconds=300))
        self.assertFalse(q.long_enough.iloc[0])

    def test_fewer_than_two_bursts_cannot_define_interburst_period(self):
        for bursts in (bounds([]), bounds([(0, 10)])):
            self.assertTrue(quiet_periods(bursts, bounds([(0, 1000)])).empty)

    def test_insufficient_duration_is_retained_as_a_diagnostic(self):
        q = quiet_periods(bounds([(0, 5), (310, 320)]), bounds([(0, 400)]))
        self.assertFalse(q.long_enough.iloc[0])
        self.assertEqual(q.geometric_windows.sum(), 0)


class WindowTests(unittest.TestCase):
    def result(self, beats=None):
        periods = quiet_periods(bounds([(0, 5), (650, 660), (1290, 1300)]), bounds([(0, 1300)]))
        return analyze_quiet_ccf(synthetic_beats() if beats is None else beats, periods)

    def test_grid_step_and_windows_never_cross_quiet_boundaries(self):
        result = self.result()
        self.assertGreater(len(result["windows"]), 0)
        self.assertTrue(result["windows"].included.all())
        for quiet_id, windows in result["windows"].groupby("quiet_period_id"):
            q = result["periods"].loc[quiet_id]
            self.assertTrue((windows.start >= q.start).all())
            self.assertTrue((windows.end <= q.end).all())
            self.assertTrue(((windows.end - windows.start).dt.total_seconds() == 300).all())
            self.assertTrue((windows.start.diff().dropna().dt.total_seconds() == 60).all())
        self.assertEqual(result["curves"].groupby(["window_id", "signal", "mode"]).size().min(), 201)
        self.assertTrue((result["windows"].n_grid_samples == 1200).all())

    def test_invalid_beats_and_missing_support_fail_qc(self):
        beats = synthetic_beats()
        beats.loc[200:210, "valid"] = False
        result = self.result(beats)
        rejected = result["windows"].loc[~result["windows"].included]
        self.assertTrue(rejected.exclusion_reason.str.contains("flagged_beats").all())
        self.assertTrue(rejected.exclusion_reason.str.contains("interpolation_gap").any())
        self.assertTrue(rejected.hr_bpm_unfiltered_r_min.isna().all())
        self.assertFalse(result["curves"].window_id.isin(rejected.window_id).any())

    def test_missing_beat_rows_cannot_be_silently_bridged(self):
        result = self.result(synthetic_beats().drop(index=range(200, 211)))
        self.assertTrue(result["windows"].exclusion_reason.str.contains("interpolation_gap").any())

    def test_misaligned_hr_pp_is_rejected(self):
        beats = synthetic_beats()
        beats.loc[2, "hr_bpm"] += 1
        with self.assertRaisesRegex(ValueError, "same intervals"):
            self.result(beats)

    def test_equal_period_aggregation_does_not_weight_long_periods(self):
        curves = pd.DataFrame({"window_id": [0, 1, 2, 3], "quiet_period_id": [0, 0, 0, 1],
                               "signal": "hr_bpm", "mode": "unfiltered", "lag_s": 0,
                               "ccf": [1., 1., 1., -1.]})
        summary, _ = aggregate_curves(curves)
        a = summary.set_index("aggregation")
        self.assertAlmostEqual(a.loc["quiet_periods", "mean"], 0)
        self.assertAlmostEqual(a.loc["overlapping_windows_descriptive", "mean"], .5)
        self.assertEqual(a.loc["quiet_periods", "n"], 2)
        singleton, _ = aggregate_curves(curves.iloc[:1])
        self.assertTrue(singleton["sem"].isna().all())

    def test_union_does_not_count_overlap_twice(self):
        self.assertEqual(union_duration_s(bounds([(0, 300), (60, 360), (600, 900)])), 660)
        result = self.result()
        counts = availability(result)
        self.assertLess(counts.eligible_window_union_min, counts.eligible_window_sum_min_counts_overlap)

    def test_empty_results_and_figures(self):
        periods = quiet_periods(bounds([]), bounds([(0, 1000)]))
        result = analyze_quiet_ccf(synthetic_beats(), periods)
        self.assertEqual(availability(result).eligible_windows, 0)
        self.assertIsNone(representative_signals(synthetic_beats(), result))
        for fig in (plot_ccf(result), plot_extrema(result), plot_representative(None)):
            fig.canvas.draw()
            plt.close(fig)


class PulseTests(unittest.TestCase):
    def test_detector_bandpass_preserves_low_sleep_pulse_fundamental(self):
        t = np.arange(75 * 60) / 75
        x = np.sin(2 * np.pi * .8 * t)
        y = def_bandpass(x, 75, .45, 10)
        self.assertGreater(np.std(y[750:-750]) / np.std(x[750:-750]), .95)

    def test_extraction_uses_actual_midpoint_and_peak_times(self):
        fs = 55
        t = np.arange(fs * 90) / fs
        raw = pd.Series(-(2 + .2 * np.sin(2 * np.pi * .05 * t)) * np.sin(2 * np.pi * t),
                        index=START + pd.to_timedelta(t, unit="s"))
        beats, filtered, qc = extract_beats(raw)
        good = beats.loc[beats.valid]
        self.assertGreater(len(good), 50)
        self.assertAlmostEqual(good.pp_s.median(), 1, delta=.04)
        self.assertAlmostEqual(good.hr_bpm.median(), 60, delta=2)
        pd.testing.assert_series_equal(beats.pp_time, beats.pp_start + (beats.pp_end - beats.pp_start) / 2,
                                       check_names=False)
        self.assertTrue((beats.pwa_time > beats.foot_time).all())
        np.testing.assert_allclose(beats.pwa, filtered.loc[beats.pwa_time].to_numpy()
                                   - filtered.loc[beats.foot_time].to_numpy())
        self.assertEqual(qc["regularization_hz"], fs)

    def test_raw_gap_and_bad_polarity_fail(self):
        raw = pd.Series(np.sin(np.arange(1000)), index=pd.date_range(START, periods=1000, freq="20ms"))
        with self.assertRaises(ValueError):
            extract_beats(raw, polarity=0)
        raw.index = raw.index[:500].append(raw.index[500:] + pd.Timedelta(seconds=1))
        with self.assertRaisesRegex(ValueError, "gap"):
            extract_beats(raw)

    def test_fiducial_indices_are_paired_and_final_window_is_processed(self):
        fs = 55
        t = np.arange(55 * 35) / fs
        peaks, feet = get_peak_onset(np.sin(2 * np.pi * t), fs)
        self.assertEqual(len(peaks), len(feet))
        self.assertTrue((feet < peaks).all())
        self.assertTrue((peaks[:-1] < feet[1:]).all())
        self.assertGreater(peaks[-1], 33 * fs)
        self.assertGreater(feet[0], 0)  # sample position, not index into minima list

    def test_sensor_clock_labels_without_moving_samples(self):
        naive = pd.date_range("2020-01-01", periods=3, freq="1s")
        np.testing.assert_array_equal(sensor_clock_index(naive).asi8, naive.asi8)


if __name__ == "__main__":
    unittest.main()
