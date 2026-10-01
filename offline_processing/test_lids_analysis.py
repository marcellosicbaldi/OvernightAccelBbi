"""Synthetic checks of envelope units, missingness, fitting and internal time."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from lids_analysis import (LIDSConfig, acceleration_to_lids, analyze_nights,
                           fit_lids, internal_profile, save_results, minute_activity, activity_to_lids)
from detect_acc_bursts import prepare_acceleration, detect_bursts


class LIDSTests(unittest.TestCase):
    FS = 50

    def acceleration(self, minutes=70, baseline_mg=1000):
        seconds = np.arange(minutes * 60 * self.FS) / self.FS
        t = pd.date_range("2026-01-01 23:30", periods=len(seconds), freq="20ms")
        return pd.DataFrame({"sample_time": t, "accel_x": 0.0,
                             "accel_y": 0.0, "accel_z": baseline_mg + 10 * np.sin(2*np.pi*3*seconds)})

    def test_minute_means_match_burst_envelope_and_units(self):
        acc = self.acceleration(20)
        config = LIDSConfig(sampling_rate=self.FS)
        minutes, qc = minute_activity(acc, config)
        magnitude, _ = prepare_acceleration(acc, self.FS)
        _, diagnostic = detect_bursts(magnitude, self.FS, alfa=0.020, return_signals=True)
        expected = (diagnostic["score"] * 1000).resample("1min", origin=magnitude.index[0]).mean()
        np.testing.assert_allclose(minutes.envelope_mg, expected)
        self.assertEqual(qc["duration_min"], 20)
        acc[["accel_x", "accel_y", "accel_z"]] /= 1000
        converted, _ = minute_activity(acc, LIDSConfig(sampling_rate=self.FS, input_unit="g"))
        np.testing.assert_allclose(minutes.envelope_mg, converted.envelope_mg)

    def test_ten_minute_sum_before_nonlinear_transform(self):
        minutes = pd.DataFrame({"envelope_mg": np.arange(1., 26), "coverage": 1.},
                               index=pd.date_range("2026-01-01", periods=25, freq="1min"))
        bins = activity_to_lids(minutes)
        self.assertEqual(bins.activity_sum_mg.iloc[0], 55)
        self.assertEqual(bins.activity_sum_mg.iloc[1], 155)
        self.assertAlmostEqual(bins.lids_raw.iloc[0], 100/56)
        self.assertAlmostEqual(bins.lids.iloc[0], (100/56 + 100/156)/2)
        self.assertTrue(np.isnan(bins.lids_raw.iloc[-1]))
        minutes.iloc[5, 0] = np.nan
        bins = activity_to_lids(minutes)
        self.assertTrue(np.isnan(bins.lids_raw.iloc[0]))
        self.assertAlmostEqual(bins.lids.iloc[1], 100/156)

    def test_additive_gravity_offset_does_not_clip_activity(self):
        normal, _ = minute_activity(self.acceleration(10), LIDSConfig(sampling_rate=self.FS))
        shifted, _ = minute_activity(self.acceleration(10, baseline_mg=850), LIDSConfig(sampling_rate=self.FS))
        np.testing.assert_allclose(normal.envelope_mg, shifted.envelope_mg, rtol=1e-6, atol=1e-6)
        self.assertTrue((shifted.envelope_mg > 15).all())

    def test_gap_runs_are_filtered_independently(self):
        acc = self.acceleration(60)
        n = 20 * 60 * self.FS
        parts = [acc.iloc[:n], acc.iloc[2*n:]]
        combined, qc = minute_activity(pd.concat(parts), LIDSConfig(sampling_rate=self.FS))
        self.assertEqual(qc["acquisition_runs"], 2)
        self.assertTrue(combined.envelope_mg.iloc[20:40].isna().all())
        for part in parts:
            separate, _ = minute_activity(part, LIDSConfig(sampling_rate=self.FS))
            np.testing.assert_allclose(combined.loc[separate.index, "envelope_mg"], separate.envelope_mg)
        bins = activity_to_lids(combined)
        self.assertTrue(bins.lids.iloc[2:4].isna().all())

    def test_duplicates_do_not_inflate_coverage(self):
        acc = self.acceleration(20)
        duplicated = pd.concat([acc, acc.iloc[:100]], ignore_index=True)
        minutes, qc = minute_activity(duplicated, LIDSConfig(sampling_rate=self.FS))
        self.assertEqual(qc["duplicate_samples"], 100)
        np.testing.assert_allclose(minutes.coverage, 1)

    def test_nonfinite_xyz_reduces_coverage(self):
        acc = self.acceleration(20)
        acc.loc[:600, "accel_x"] = np.inf
        bins, _ = acceleration_to_lids(acc, LIDSConfig(sampling_rate=self.FS))
        self.assertTrue(np.isnan(bins.lids.iloc[0]))
        self.assertTrue(np.isfinite(bins.lids.iloc[1]))

    def test_short_runs_are_missing_not_inactivity(self):
        acc = self.acceleration(1).iloc[:40]
        minutes, qc = minute_activity(acc, LIDSConfig(sampling_rate=self.FS))
        self.assertEqual(qc["runs_too_short"], 1)
        self.assertTrue(minutes.envelope_mg.isna().all())

    def test_projection_matches_reference_and_recovers_period(self):
        t = np.arange(0, 720, 10)
        phase, amp = 0.6, 12
        y = 50 + amp * np.cos(2 * np.pi * t / 90 - phase)
        fit, curve = fit_lids(y, t)
        self.assertEqual(fit["period_min"], 90)
        self.assertAlmostEqual(fit["amplitude"], amp)
        self.assertAlmostEqual(fit["phase_rad"], phase)
        self.assertAlmostEqual(fit["mri"], 24)
        self.assertAlmostEqual(fit["r2"], 1)
        np.testing.assert_allclose(curve, y, atol=1e-12)
        # Explicit noninteger-cycle reference formula parity.
        t, y = t[:41], y[:41]
        fit, curve = fit_lids(y, t, LIDSConfig(period_min=90, period_max=90))
        angle = 2 * np.pi * t / 90
        expected = 2 * np.mean(y*np.cos(angle))*np.cos(angle) + 2 * np.mean(y*np.sin(angle))*np.sin(angle) + np.mean(y)
        np.testing.assert_allclose(curve, expected)

    def test_least_squares_recovers_incomplete_cycles_with_gaps(self):
        t = np.arange(0, 410, 10)
        y = 50 + 10 * np.cos(2 * np.pi * t / 95 + 0.4)
        y[4:7] = np.nan
        fit, _ = fit_lids(y, t, LIDSConfig(period_min=95, period_max=95, fit_method="least_squares"))
        self.assertAlmostEqual(fit["amplitude"], 10)
        self.assertAlmostEqual(fit["phase_rad"], -0.4)
        self.assertAlmostEqual(fit["r2"], 1)

    def test_constant_short_and_invalid_inputs(self):
        for y, t, status in [(np.ones(30), np.arange(30)*10, "constant_signal"),
                              (np.arange(3), np.arange(3)*10, "insufficient_data")]:
            fit, curve = fit_lids(y, t)
            self.assertEqual(fit["status"], status)
            self.assertTrue(np.isnan(curve).all())
            self.assertTrue(np.isnan(fit["period_min"]))
        with self.assertRaises(ValueError):
            LIDSConfig(period_min=10)
        with self.assertRaises(ValueError):
            LIDSConfig(sampling_rate=np.nan)
        with self.assertRaises(ValueError):
            acceleration_to_lids(self.acceleration(), LIDSConfig(sampling_rate=200))

    def test_internal_time_scaling_counts_and_equal_night_mean(self):
        # A: period 180 -> 20 external minutes = 10 internal minutes.
        # B ends sooner; its absence must not act like zero or carry forward.
        bouts = {"a": pd.DataFrame({"external_min": [0, 20, 40, 60], "lids": [20., 30, 40, 50]}),
                 "b": pd.DataFrame({"external_min": [0, 10], "lids": [60., 80]})}
        summary = pd.DataFrame({"status": ["ok", "ok"], "period_min": [180., 90]}, index=["a", "b"])
        normalized, profile, cycles = internal_profile(bouts, summary)
        self.assertEqual(profile.loc[10, "mean"], 55)
        self.assertEqual(profile.loc[20, "mean"], 40)
        self.assertEqual(profile.loc[20, "n_nights"], 1)
        self.assertTrue(np.isnan(profile.loc[20, "sem"]))
        self.assertEqual(profile.loc[40, "n_nights"], 0)
        self.assertEqual(len(profile), 36)
        self.assertEqual(profile.index[-1], 350)
        self.assertEqual(cycles.loc[1, "n_any"], 2)
        self.assertTrue(np.isnan(normalized.loc[20, "b"]))

    def test_interpolation_between_samples_but_not_across_gaps(self):
        bouts = {"a": pd.DataFrame({"external_min": [0, 10, 20, 30, 40],
                                    "lids": [20., 40, np.nan, 80, 100]})}
        summary = pd.DataFrame({"status": ["ok"], "period_min": [45.]}, index=["a"])
        normalized, _, _ = internal_profile(bouts, summary)
        self.assertEqual(normalized.loc[10, "a"], 30)
        self.assertTrue(normalized.loc[30:50, "a"].isna().all())
        self.assertEqual(normalized.loc[70, "a"], 90)

    def test_no_valid_fits_have_no_contributors(self):
        bouts = {"constant": pd.DataFrame({"external_min": [0, 10], "lids": [100., 100]})}
        summary = pd.DataFrame({"status": ["constant_signal"]}, index=["constant"])
        normalized, profile, cycles = internal_profile(bouts, summary)
        self.assertEqual(normalized.shape, (36, 0))
        self.assertTrue(profile["mean"].isna().all())
        self.assertTrue(profile.n_nights.eq(0).all())
        self.assertTrue(cycles.n_any.eq(0).all())

    def test_dst_aware_elapsed_time(self):
        acc = self.acceleration(70)
        acc["sample_time"] = pd.date_range("2026-10-25 00:30Z", periods=len(acc), freq="20ms")
        bins, qc = acceleration_to_lids(acc, LIDSConfig(sampling_rate=self.FS))
        np.testing.assert_allclose(bins.external_min, np.arange(7) * 10)
        self.assertEqual(qc["duration_min"], 70)

    def test_batch_reports_bad_file_and_saves_artifacts(self):
        import matplotlib
        matplotlib.use("Agg")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for night in ["good", "bad"]:
                (root / night).mkdir()
            acc = self.acceleration(180)
            seconds = np.arange(len(acc)) / self.FS
            amplitude = 10 + 8 * np.cos(2 * np.pi * seconds / (90*60))
            acc["accel_z"] = 1000 + amplitude * np.sin(2*np.pi*3*seconds)
            acc.to_parquet(root / "good" / "accelerometer_samples.parquet")
            pd.DataFrame({"wrong": [1]}).to_parquet(root / "bad" / "accelerometer_samples.parquet")
            result = analyze_nights(root, LIDSConfig(sampling_rate=self.FS))
            self.assertEqual(result["summary"].loc["bad", "status"], "input_error")
            self.assertEqual(result["summary"].loc["good", "status"], "ok")
            save_results(result, root / "output")
            self.assertTrue((root / "output" / "lids_average.png").exists())
            self.assertTrue((root / "output" / "good_lids.png").exists())


if __name__ == "__main__":
    unittest.main()
