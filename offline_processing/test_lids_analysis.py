"""Synthetic checks of ENMO units, missingness, fitting and internal time."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from lids_analysis import (LIDSConfig, acceleration_to_lids, analyze_nights,
                           fit_lids, internal_profile, save_results)


class LIDSTests(unittest.TestCase):
    def acceleration(self, minutes=70):
        t = pd.date_range("2026-01-01 23:30", periods=minutes * 60, freq="s")
        return pd.DataFrame({"sample_time": t, "accel_x": 0.0,
                             "accel_y": 0.0, "accel_z": 1009.0})

    def test_enmo_units_transform_and_night_boundary(self):
        acc = self.acceleration(25)
        bins, qc = acceleration_to_lids(acc, LIDSConfig(sampling_rate=1))
        np.testing.assert_allclose(bins.enmo_mg, 9)
        np.testing.assert_allclose(bins.lids[:2], 10)
        self.assertTrue(np.isnan(bins.lids.iloc[-1]))
        self.assertAlmostEqual(bins.coverage.iloc[-1], 0.5)
        self.assertEqual(qc["duration_min"], 25)
        acc[["accel_x", "accel_y", "accel_z"]] /= 1000
        converted, _ = acceleration_to_lids(acc, LIDSConfig(sampling_rate=1, input_unit="g"))
        np.testing.assert_allclose(bins.lids, converted.lids)

    def test_missing_data_not_inactivity_or_smoothed_across(self):
        acc = self.acceleration()
        acc.loc[acc.index >= 40 * 60, "accel_z"] = 1099
        acc = acc.drop(index=np.arange(20 * 60, 40 * 60))
        bins, _ = acceleration_to_lids(acc, LIDSConfig(sampling_rate=1))
        self.assertTrue(bins.lids.iloc[2:4].isna().all())
        np.testing.assert_allclose(bins.lids.iloc[:2], 10)
        np.testing.assert_allclose(bins.lids.iloc[4:], 1)

    def test_duplicates_do_not_inflate_coverage(self):
        acc = self.acceleration(20)
        duplicated = pd.concat([acc, acc.iloc[:100]], ignore_index=True)
        bins, qc = acceleration_to_lids(duplicated, LIDSConfig(sampling_rate=1))
        self.assertEqual(qc["duplicate_samples"], 100)
        np.testing.assert_allclose(bins.coverage, 1)

    def test_nonfinite_xyz_reduces_coverage(self):
        acc = self.acceleration(20)
        acc.loc[:100, "accel_x"] = np.inf
        bins, _ = acceleration_to_lids(acc, LIDSConfig(sampling_rate=1))
        self.assertTrue(np.isnan(bins.lids.iloc[0]))
        self.assertAlmostEqual(bins.lids.iloc[1], 10)

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
            acceleration_to_lids(self.acceleration(), LIDSConfig(sampling_rate=100))

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
        acc["sample_time"] = pd.date_range("2026-10-25 00:30Z", periods=len(acc), freq="s")
        bins, qc = acceleration_to_lids(acc, LIDSConfig(sampling_rate=1))
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
            t = np.arange(len(acc)) / 60
            desired_lids = 50 + 20 * np.cos(2 * np.pi * t / 90)
            acc["accel_z"] = 1000 + 100/desired_lids - 1
            acc.to_parquet(root / "good" / "accelerometer_samples.parquet")
            pd.DataFrame({"wrong": [1]}).to_parquet(root / "bad" / "accelerometer_samples.parquet")
            result = analyze_nights(root, LIDSConfig(sampling_rate=1))
            self.assertEqual(result["summary"].loc["bad", "status"], "input_error")
            self.assertEqual(result["summary"].loc["good", "status"], "ok")
            save_results(result, root / "output")
            self.assertTrue((root / "output" / "lids_average.png").exists())
            self.assertTrue((root / "output" / "good_lids.png").exists())


if __name__ == "__main__":
    unittest.main()
