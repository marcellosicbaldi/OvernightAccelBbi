"""Synthetic geometry, missing-data, persistence, and alignment regressions."""

import unittest

import numpy as np
import pandas as pd

from wrist_reorientation import (
    CHANGED, INTERMEDIATE, LITTLE, UNKNOWN, angle_degrees,
    classify_reorientation, measure_reorientation, prepare_wrist_axes,
)


class ReorientationTests(unittest.TestCase):
    def setUp(self):
        self.origin = pd.Timestamp("2026-01-01", tz="UTC")
        self.times = self.origin + pd.to_timedelta(np.arange(1000) / 10, unit="s")
        self.bursts = pd.DataFrame({"start": [self.origin + pd.Timedelta(seconds=30)],
                                    "end": [self.origin + pd.Timedelta(seconds=35)]}, index=[7])

    def axes(self, angle=60):
        xyz = np.tile([0., 0., 1.], (1000, 1))
        xyz[350:] = [np.sin(np.radians(angle)), 0, np.cos(np.radians(angle))]
        return pd.DataFrame(xyz, index=self.times, columns=["x_g", "y_g", "z_g"])

    def measure(self, axes, bursts=None, **kwargs):
        return measure_reorientation(axes, self.bursts if bursts is None else bursts,
                                     sampling_rate=10, **kwargs)

    def test_vector_angles_and_zero_vector(self):
        np.testing.assert_allclose(angle_degrees([[1, 0, 0], [-1, 0, 0], [0, 1, 0]], [1, 0, 0]), [0, 180, 90])
        self.assertTrue(np.isnan(angle_degrees([0, 0, 0], [1, 0, 0])))

    def test_known_stable_rotation_and_windows(self):
        row = self.measure(self.axes()).loc[7]
        self.assertEqual(row.reorientation_class, CHANGED)
        self.assertAlmostEqual(row.tilt_change_deg, 60)
        self.assertEqual(row.pre_start, self.origin + pd.Timedelta(seconds=18))
        self.assertEqual(row.hold_end, self.origin + pd.Timedelta(seconds=57))
        self.assertEqual(row.pre_coverage, 1)

    def test_thresholds_and_gray_zone(self):
        for angle, expected in [(0, LITTLE), (10, LITTLE), (20, INTERMEDIATE), (30, CHANGED), (90, CHANGED)]:
            with self.subTest(angle=angle):
                self.assertEqual(self.measure(self.axes(angle)).loc[7].reorientation_class, expected)

    def test_reversal_does_not_count_as_sustained(self):
        axes = self.axes()
        axes.iloc[470:] = [0., 0., 1.]
        row = self.measure(axes).loc[7]
        self.assertEqual(row.reorientation_class, UNKNOWN)
        self.assertIn("not_sustained", row.orientation_reason)

    def test_gap_is_not_interpolated(self):
        axes = self.axes().drop(self.times[200:211])
        row = self.measure(axes).loc[7]
        self.assertEqual(row.reorientation_class, UNKNOWN)
        self.assertGreater(row.pre_max_gap_s, 1)

    def test_recording_edge_is_unknown(self):
        bursts = self.bursts.copy()
        bursts.loc[7, "start"] = self.origin + pd.Timedelta(seconds=5)
        self.assertEqual(self.measure(self.axes(), bursts).loc[7].reorientation_class, UNKNOWN)

    def test_wrong_scale_fails_qc(self):
        row = self.measure(self.axes() * 1000).loc[7]
        self.assertEqual(row.reorientation_class, UNKNOWN)
        self.assertIn("gravity_scale", row.orientation_reason)

    def test_dynamic_window_fails_qc(self):
        axes = self.axes()
        axes.iloc[180:280:2, 0] = 0.3
        row = self.measure(axes).loc[7]
        self.assertEqual(row.reorientation_class, UNKNOWN)
        self.assertIn("dynamic_acceleration", row.orientation_reason)

    def test_neighboring_burst_excludes_window(self):
        bursts = pd.concat([self.bursts, pd.DataFrame({
            "start": [self.origin + pd.Timedelta(seconds=20)],
            "end": [self.origin + pd.Timedelta(seconds=21)]}, index=[9])])
        self.assertIn("pre:other_burst", self.measure(self.axes(), bursts).loc[7].orientation_reason)

    def test_missing_xyz_is_unknown(self):
        axes = self.axes()
        axes.iloc[180:280] = np.nan
        row = self.measure(axes).loc[7]
        self.assertEqual(row.reorientation_class, UNKNOWN)
        self.assertIn("no_valid_samples", row.orientation_reason)

    def test_preparation_units_sorting_duplicates_and_invalids(self):
        source = pd.DataFrame({"sample_time": [self.times[1], self.times[0], self.times[1]],
                               "accel_x": [0, 0, np.nan], "accel_y": [0, 0, 0], "accel_z": [1000, 1000, 1000]})
        result = prepare_wrist_axes(source)
        self.assertEqual(result.iloc[0].z_g, 1)
        self.assertTrue(result.iloc[1].isna().all())
        self.assertEqual(result.attrs["duplicate_timestamps_averaged"], 1)

    def test_sensitivity_does_not_remeasure(self):
        measured = self.measure(self.axes(35))
        changed = classify_reorientation(measured, change_angle_deg=45)
        self.assertEqual(changed.loc[7].reorientation_class, INTERMEDIATE)
        self.assertEqual(measured.loc[7].reorientation_class, CHANGED)

    def test_invalid_parameters(self):
        for kwargs in [{"window_s": 0}, {"min_coverage": 1.1}, {"buffer_s": -1},
                       {"little_angle_deg": 30, "change_angle_deg": 20}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.measure(self.axes(), **kwargs)

    def test_empty_bursts(self):
        result = self.measure(self.axes(), self.bursts.iloc[:0])
        self.assertTrue(result.empty)
        self.assertIn("reorientation_class", result)


if __name__ == "__main__":
    unittest.main()
