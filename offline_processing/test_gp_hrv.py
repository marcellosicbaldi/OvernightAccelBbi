"""GP geometry/cleaning and Garmin transport-boundary regressions."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from gp_hrv import build_variable_windows, clean_interval_run, interburst_hrv
from test_interburst_hrv import START, at, make_bursts, make_rows, supplied_pip


class GpHrvTests(unittest.TestCase):
    def compute(self, rows=None, bursts=None, end=900, **kwargs):
        if rows is None:
            rows = make_rows(int(end))
            for row in rows:
                row["bbi_ms"] = 1000
        return interburst_hrv(rows, make_bursts() if bursts is None else bursts,
                             START, at(end), **kwargs)

    def test_gp_variable_window_geometry_and_final_anchor(self):
        args = [pd.Timedelta(seconds=x) for x in (60, 300, 60)]
        self.assertEqual(build_variable_windows(START, at(59), *args), [])
        for duration in (60, 180, 300):
            self.assertEqual(build_variable_windows(START, at(duration), *args), [(START, at(duration))])
        windows = build_variable_windows(START, at(430), *args)
        self.assertEqual(windows, [(at(0), at(300)), (at(60), at(360)),
                                   (at(120), at(420)), (at(130), at(430))])

    def test_gp_burst_duration_and_one_second_post_margin(self):
        bursts = make_bursts((50, 51.99), (150, 152), (151, 160), (900, 910))
        before = bursts.copy(deep=True)
        result = self.compute(bursts=bursts)
        self.assertEqual(result["report"]["ignored_short_bursts"], 1)
        segments = result["segments"]
        self.assertEqual(list(segments.start), [START, at(161)])
        self.assertEqual(list(segments.end), [at(150), at(900)])
        windows = result["windows"]
        self.assertEqual(windows.window_length_s.iloc[0], 150)
        self.assertTrue(((windows.end <= at(150)) | (windows.start >= at(161))).all())
        self.assertTrue(windows.included.all())
        pd.testing.assert_frame_equal(bursts, before)

    def test_can_retain_short_bursts_for_stricter_segmentation(self):
        result = self.compute(bursts=make_bursts((100, 101)), min_burst_duration_s=0,
                              post_burst_guard_s=0)
        self.assertEqual(list(result["segments"].start), [START, at(101)])
        self.assertEqual(result["report"]["ignored_short_bursts"], 0)

    def test_constant_intervals_have_valid_zero_hrv_and_gp_pip(self):
        windows = self.compute(end=120)["windows"]
        self.assertEqual(len(windows), 1)
        window = windows.iloc[0]
        self.assertTrue(window.included)
        self.assertEqual(window.n_artifacts, 0)
        self.assertEqual(window.n_interpolated, 0)
        self.assertEqual(window.mean_hr_bpm, 60)
        self.assertEqual(window.rmssd_ms, 0)
        self.assertEqual(window.sdnn_ms, 0)
        self.assertAlmostEqual(window.pip, 118/120)

    def test_metrics_match_gp_formulas_on_cleaned_intervals(self):
        rows = make_rows(180)
        for i, row in enumerate(rows):
            row["bbi_ms"] = float(1000 + 50*np.sin(i*0.3))
        result = self.compute(rows=rows, end=180)
        ppi = result["intervals"].bbi_clean_ms.to_numpy()
        window = result["windows"].iloc[0]
        self.assertTrue(window.included)
        self.assertAlmostEqual(window.mean_hr_bpm, np.mean(60000/ppi))
        self.assertAlmostEqual(window.rmssd_ms, np.sqrt(np.mean(np.diff(ppi)**2)))
        self.assertAlmostEqual(window.sdnn_ms, np.std(ppi, ddof=1))
        self.assertAlmostEqual(window.pip, supplied_pip(ppi))

    def test_single_pass_classifier_flags_are_masked_and_interpolated(self):
        values = np.array([900, 1000, 1600, 1000, 1100], dtype=float)
        artifacts = dict(ectopic=[2], missed=[], extra=[], longshort=[])
        with patch("gp_hrv._find_artifacts", return_value=(artifacts, {})) as classifier:
            result = clean_interval_run(values)
        classifier.assert_called_once()
        np.testing.assert_array_equal(classifier.call_args.args[0], values/1000)
        np.testing.assert_array_equal(result["cleaned"], [900, 1000, 1000, 1000, 1100])
        self.assertEqual(list(np.flatnonzero(result["interpolated"])), [2])
        self.assertEqual(list(np.flatnonzero(result["artifact"])), [2])
        np.testing.assert_array_equal(values, [900, 1000, 1600, 1000, 1100])

    def test_classifier_matches_original_gp_on_injected_artifacts(self):
        # Golden classifications from the supplied GP kubios.py, single pass.
        values = 1000 + 50*np.sin(np.arange(600)*.3)
        indices = [100, 200, 300, 400, 500]
        values[indices] = [400, 1800, 250, 2100, 500]
        result = clean_interval_run(values)
        expected_flags = dict(ectopic=[], missed=[400], extra=[],
                              longshort=[100, 200, 300, 500])
        for name, expected in expected_flags.items():
            self.assertEqual(list(np.flatnonzero(result[name])), expected)
        expected = values.copy()
        for index in indices:
            expected[index] = (values[index-1] + values[index+1])/2
        np.testing.assert_allclose(result['cleaned'], expected)
        self.assertEqual(list(np.flatnonzero(result['interpolated'])), indices)
        self.assertEqual(list(np.flatnonzero(result['invalid'])), [300, 400])

    def test_gp_range_screening_and_endpoint_fill_are_visible(self):
        values = [np.nan, 1000, np.inf, 1000, 0, 1000, 2500]
        result = clean_interval_run(values)
        np.testing.assert_array_equal(result["cleaned"], np.full(7, 1000))
        self.assertEqual(list(np.flatnonzero(result["interpolated"])), [0, 2, 4, 6])
        self.assertTrue(np.isnan(clean_interval_run([np.nan, 0, np.inf])["cleaned"]).all())

    def test_cleaning_never_bridges_sequence_holes(self):
        rows = make_rows(120)
        for i, row in enumerate(rows):
            row["bbi_ms"] = 900 if i < 60 else 1100
        rows[59]["bbi_ms"] = None
        rows[61]["bbi_ms"] = None
        del rows[60]
        result = self.compute(rows=rows, end=120)
        clean = result["intervals"].set_index("sequence")
        self.assertEqual(clean.loc[60, "bbi_clean_ms"], 900)
        self.assertEqual(clean.loc[62, "bbi_clean_ms"], 1100)
        self.assertNotEqual(clean.loc[60, "delivery_run_id"], clean.loc[62, "delivery_run_id"])
        window = result["windows"].iloc[0]
        self.assertIn("missing_sequences", window.exclusion_reason)
        self.assertTrue(np.isnan(window.rmssd_ms))
        self.assertTrue(np.isnan(window.pip))

    def test_long_callback_gap_splits_cleaning_and_rejects_window(self):
        rows = make_rows(120)
        for row in rows:
            row["bbi_ms"] = 1000
        del rows[50:70]
        for i, row in enumerate(rows):
            row["sequence"] = i+1
        result = self.compute(rows=rows, end=120)
        self.assertEqual(result["intervals"].delivery_run_id.nunique(), 2)
        self.assertIn("callback_gap_or_edge_silence", result["windows"].exclusion_reason.iloc[0])

    def test_same_callback_intervals_are_individual_and_ordered(self):
        rows = make_rows(120)
        for i, row in enumerate(rows):
            row["bbi_ms"] = 1000
            row["callback_time_utc_approx"] = at(i//2*2).isoformat()
        result = self.compute(rows=rows, end=120)
        self.assertEqual(len(result["intervals"]), 120)
        self.assertEqual(result["windows"].n_intervals.iloc[0], 120)
        self.assertTrue(result["windows"].included.all())

    def test_window_end_is_exclusive_and_min_beats_is_not_min_pairs(self):
        rows = make_rows(61)
        for row in rows:
            row["bbi_ms"] = 1000
        result = self.compute(rows=rows, end=120, max_window_s=60)
        self.assertEqual(result["windows"].n_intervals.tolist(), [60, 1])
        self.assertTrue(result["windows"].included.iloc[0])
        self.assertFalse(result["windows"].included.iloc[1])
        sparse = rows[:30]
        for i, row in enumerate(sparse):
            row["callback_time_utc_approx"] = at(i*2).isoformat()
            row["bbi_ms"] = 2000
        window = self.compute(rows=sparse, end=60)["windows"].iloc[0]
        self.assertTrue(window.included)
        self.assertEqual(window.n_intervals, 30)

    def test_repaired_fraction_limit_and_raw_values_are_preserved(self):
        rows = make_rows(120)
        for row in rows:
            row["bbi_ms"] = 1000
        rows[50]["bbi_ms"] = None
        result = self.compute(rows=rows, end=120, max_interpolated_fraction=0)
        self.assertTrue(np.isnan(result["intervals"].bbi_ms.iloc[50]))
        self.assertEqual(result["intervals"].bbi_clean_ms.iloc[50], 1000)
        window = result["windows"].iloc[0]
        self.assertEqual(window.n_interpolated, 1)
        self.assertIn("too_many_interpolated_intervals", window.exclusion_reason)

    def test_missing_and_all_invalid_data_never_produce_metrics(self):
        for rows in ([], [{**row, "bbi_ms": None} for row in make_rows(120)]):
            windows = self.compute(rows=rows, end=120)["windows"]
            self.assertFalse(windows.included.any())
            self.assertTrue(windows.rmssd_ms.isna().all())
        result = self.compute(bursts=make_bursts((0, 900)))
        self.assertTrue(result["segments"].empty)
        self.assertTrue(result["windows"].empty)
        self.assertEqual(result["windows"].included.dtype, bool)
        self.assertTrue(self.compute(end=59)["windows"].empty)

    def test_short_dense_data_and_edge_silence_are_rejected(self):
        rows = make_rows(120)
        for row in rows:
            row["bbi_ms"] = 500
        window = self.compute(rows=rows, end=120)["windows"].iloc[0]
        self.assertIn("bbi_duration_inconsistent_with_window", window.exclusion_reason)
        window = self.compute(rows=make_rows(120)[10:], end=120)["windows"].iloc[0]
        self.assertIn("callback_gap_or_edge_silence", window.exclusion_reason)

    def test_invalid_configuration_and_timezone_bounds(self):
        for options in ({"step_s": 0}, {"min_window_s": 301}, {"step_s": 301},
                        {"post_burst_guard_s": -1}, {"min_burst_duration_s": -1},
                        {"min_beats": 2}, {"max_interpolated_fraction": 1.1},
                        {"duration_fraction_limits": (1.1, 0.9)}, {"bbi_limits_ms": (2000, 300)}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.compute(**options)
        with self.assertRaises(ValueError):
            interburst_hrv([], make_bursts(), START.tz_localize(None), at(120))
        bursts = make_bursts((100, 105))
        local = bursts.copy()
        for name in ("start", "end"):
            local[name] = local[name].dt.tz_convert("Europe/Rome")
        pd.testing.assert_frame_equal(self.compute(bursts=bursts)["windows"],
                                      self.compute(bursts=local)["windows"])


if __name__ == "__main__":
    unittest.main()
