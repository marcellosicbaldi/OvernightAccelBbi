"""Window geometry, known RMSSD values and missing-data barriers."""

import unittest

import numpy as np
import pandas as pd

from interburst_hrv import burst_free_segments, compute_HRF, interburst_rmssd, prepare_bbi_intervals

START = pd.Timestamp("2026-09-14 22:00", tz="UTC")


def at(seconds):
    return START + pd.Timedelta(seconds=seconds)


def make_bursts(*bounds):
    return pd.DataFrame([(at(a), at(b)) for a, b in bounds], columns=["start", "end"])


def make_rows(n=900):
    return [{"sequence": i + 1, "bbi_ms": 900 if i % 2 == 0 else 1100,
             "callback_time_utc_approx": at(i).isoformat()} for i in range(n)]


class InterburstHrvTests(unittest.TestCase):
    def compute(self, rows=None, bursts=None, end=900, **options):
        return interburst_rmssd(make_rows() if rows is None else rows,
                               make_bursts() if bursts is None else bursts,
                               START, at(end), **options)

    def test_known_rmssd_and_four_minute_overlap(self):
        result = self.compute()
        windows = result["windows"]
        self.assertEqual(len(windows), 11)
        self.assertTrue(windows.included.all())
        np.testing.assert_allclose(windows.rmssd_ms, 200)
        self.assertTrue((windows.n_adjacent_pairs == 299).all())
        self.assertTrue((windows.start.diff().dropna() == pd.Timedelta(seconds=60)).all())
        self.assertTrue(((windows.end - windows.start) == pd.Timedelta(seconds=300)).all())

    def test_matches_neurokit_on_complete_interval_sequence(self):
        import neurokit2 as nk
        expected = nk.hrv_time({"RRI": np.tile([900, 1100], 150)})["HRV_RMSSD"].iloc[0]
        self.assertAlmostEqual(self.compute()["windows"].rmssd_ms.iloc[0], expected)

    def test_windows_never_cross_bursts_and_grid_restarts_in_each_gap(self):
        bursts = make_bursts((360, 400))
        result = self.compute(bursts=bursts)
        windows = result["windows"]
        self.assertEqual(list((windows.start - START).dt.total_seconds()), [0, 60, 400, 460, 520, 580])
        self.assertTrue(((windows.end <= at(360)) | (windows.start >= at(400))).all())

    def test_overlapping_touching_and_outside_bursts_merge(self):
        segments = burst_free_segments(make_bursts((350, 400), (300, 350), (310, 390), (-20, 10), (890, 920)), START, at(900))
        self.assertEqual(list(segments.duration_s), [290, 490])

    def test_no_five_minute_gap_returns_typed_empty_table(self):
        result = self.compute(bursts=make_bursts((290, 620)))
        self.assertTrue(result["windows"].empty)
        self.assertEqual(result["windows"].included.dtype, bool)
        self.assertTrue(self.compute(bursts=make_bursts((0, 900)))["segments"].empty)

    def test_optional_guard(self):
        result = self.compute(bursts=make_bursts((360, 400)), guard_s=5)
        self.assertEqual(result["segments"].iloc[0].end, at(355))
        self.assertEqual(result["segments"].iloc[1].start, at(405))

    def test_same_callback_intervals_kept_in_order(self):
        rows = make_rows()
        for i, row in enumerate(rows):
            row["callback_time_utc_approx"] = at(i // 2 * 2).isoformat()
        result = self.compute(rows=rows)
        self.assertEqual(len(result["intervals"]), 900)
        self.assertTrue(result["windows"].included.all())
        np.testing.assert_allclose(result["windows"].rmssd_ms, 200)

    def test_invalid_interval_does_not_join_its_neighbors(self):
        rows = make_rows()
        rows[150]["bbi_ms"] = None
        window = self.compute(rows=rows)["windows"].iloc[0]
        self.assertFalse(window.included)
        self.assertTrue(np.isnan(window.rmssd_ms))
        self.assertEqual(window.n_adjacent_pairs, 297)
        self.assertEqual(window.rmssd_available_pairs_ms, 200)
        self.assertIn("invalid_bbi", window.exclusion_reason)

    def test_missing_sequence_does_not_join_its_neighbors(self):
        rows = make_rows()
        del rows[150]
        window = self.compute(rows=rows)["windows"].iloc[0]
        self.assertEqual(window.n_adjacent_pairs, 297)
        self.assertEqual(window.missing_sequences, 1)
        self.assertIn("missing_sequences", window.exclusion_reason)

    def test_long_gaps_and_edge_silence_are_excluded(self):
        rows = make_rows()
        del rows[100:120]
        # Even sequential app IDs do not excuse a silent callback gap.
        for i, row in enumerate(rows):
            row["sequence"] = i + 1
        self.assertIn("callback_gap_or_edge_silence", self.compute(rows=rows)["windows"].iloc[0].exclusion_reason)
        self.assertFalse(self.compute(rows=make_rows()[10:])["windows"].iloc[0].included)

    def test_interval_sum_flags_dense_callbacks_with_missing_beats(self):
        rows = make_rows()
        for row in rows:
            row["bbi_ms"] = 500
        window = self.compute(rows=rows)["windows"].iloc[0]
        self.assertEqual(window.bbi_duration_fraction, 0.5)
        self.assertIn("bbi_duration_inconsistent_with_window", window.exclusion_reason)

    def test_no_bbi_data_does_not_fall_back_to_record_hr(self):
        windows = self.compute(rows=[])["windows"]
        self.assertFalse(windows.included.any())
        self.assertTrue(windows.rmssd_ms.isna().all())
        with self.assertRaisesRegex(ValueError, "decoded numbered BBIs"):
            self.compute(rows=[{"timestamp": START, "hr_bpm": 60}])

    def test_window_end_is_exclusive(self):
        rows = make_rows()
        rows[300]["bbi_ms"] = None
        windows = self.compute(rows=rows)["windows"]
        self.assertTrue(windows.iloc[0].included)
        self.assertFalse(windows.iloc[1].included)

    def test_duplicate_sequence_and_nonmonotonic_time_rejected(self):
        rows = make_rows(4)
        with self.assertRaisesRegex(ValueError, "unique positive integers"):
            prepare_bbi_intervals(rows + [rows[0]])
        rows[2]["callback_time_utc_approx"] = at(0).isoformat()
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            prepare_bbi_intervals(rows)

    def test_aware_times_and_input_preservation(self):
        rows = make_rows()
        bursts = make_bursts((360, 400))
        before = bursts.copy(deep=True)
        result = self.compute(rows=rows, bursts=bursts)
        local = bursts.copy()
        for column in ["start", "end"]:
            local[column] = local[column].dt.tz_convert("Europe/Rome")
        pd.testing.assert_frame_equal(result["windows"], self.compute(rows=rows, bursts=local)["windows"])
        pd.testing.assert_frame_equal(bursts, before)

    def test_invalid_parameters(self):
        for options in [{"step_s": 0}, {"window_s": 20, "step_s": 60}, {"guard_s": -1},
                        {"min_pairs": 0}, {"bbi_limits_ms": (2000, 300)}]:
            with self.assertRaises(ValueError):
                self.compute(**options)

    def test_mixed_whole_and_fractional_second_arrival_times(self):
        rows = make_rows(4)
        rows[1]["callback_time_utc_approx"] = at(1.005).isoformat()
        intervals = prepare_bbi_intervals(rows)
        self.assertEqual(intervals.callback_time_utc_approx.iloc[1], at(1.005))

    def test_nonfinite_values_are_rejected_without_runtime_warnings(self):
        rows = make_rows()
        rows[100]["bbi_ms"] = float("inf")
        rows[101]["bbi_ms"] = float("inf")
        window = self.compute(rows=rows)["windows"].iloc[0]
        self.assertEqual(window.n_invalid_intervals, 2)
        self.assertFalse(window.included)

    def test_zero_rmssd_is_not_missing(self):
        rows = make_rows()
        for row in rows:
            row["bbi_ms"] = 1000
        windows = self.compute(rows=rows)["windows"]
        self.assertTrue(windows.included.all())
        np.testing.assert_array_equal(windows.rmssd_ms, 0)

    def test_pip_matches_supplied_function_for_every_complete_window(self):
        rows = make_rows()
        rng = np.random.default_rng(29)
        for row, value in zip(rows, rng.integers(950, 1051, len(rows))):
            row["bbi_ms"] = int(value)
        windows = self.compute(rows=rows)["windows"]
        self.assertTrue(windows.included.all())
        for window in windows.itertuples():
            a = int((window.start - START).total_seconds())
            values = [row["bbi_ms"] for row in rows[a:a + 300]]
            expected = supplied_pip(values)
            self.assertEqual(window.pip, expected)
            self.assertEqual(window.pip_pct, expected * 100)
            self.assertEqual(window.n_valid_triplets, 298)

    def test_pip_diagnostic_counts_never_cross_missing_invalid_or_silent_data(self):
        for defect in ("invalid", "sequence", "time"):
            rows = make_rows()
            if defect == "invalid":
                rows[150]["bbi_ms"] = None
            else:
                del rows[150:160 if defect == "time" else 151]
                if defect == "time":
                    for i, row in enumerate(rows):
                        row["sequence"] = i + 1
            window = self.compute(rows=rows)["windows"].iloc[0]
            self.assertFalse(window.included)
            self.assertTrue(np.isnan(window.pip))
            self.assertTrue(np.isnan(window.pip_pct))
            expected_triplets = 286 if defect == "time" else 295
            self.assertEqual(window.n_valid_triplets, expected_triplets)
            self.assertEqual(window.n_inflection_points, expected_triplets)

    def test_pip_keeps_same_callback_intervals_and_window_boundaries(self):
        rows = make_rows()
        for i, row in enumerate(rows):
            row["callback_time_utc_approx"] = at(i // 2 * 2).isoformat()
        rows[300]["bbi_ms"] = None
        windows = self.compute(rows=rows)["windows"]
        self.assertAlmostEqual(windows.iloc[0].pip, 298 / 300)
        self.assertTrue(np.isnan(windows.iloc[1].pip))

    def test_pip_unavailable_for_empty_or_too_short_windows(self):
        self.assertTrue(self.compute(rows=[])["windows"].pip.isna().all())
        empty = self.compute(bursts=make_bursts((0, 900)))["windows"]
        self.assertTrue({"pip", "pip_pct", "n_valid_triplets"}.issubset(empty.columns))
        short = self.compute(rows=make_rows(2), end=2, window_s=2, step_s=2,
                             min_pairs=1)["windows"].iloc[0]
        self.assertTrue(short.included)
        self.assertTrue(np.isnan(short.pip))
        self.assertEqual(short.n_valid_triplets, 0)


def supplied_pip(ppi):
    """Independent scalar reference from the user's snippet."""
    n_inflection = np.zeros(len(ppi))
    diff_ppi = np.diff(np.asarray(ppi, dtype=float))
    eps = 1e-10
    for i in range(1, len(diff_ppi)):
        if diff_ppi[i] * diff_ppi[i - 1] <= eps:
            n_inflection[i] = 1
    return np.sum(n_inflection) / len(ppi)


class PipFormulaTests(unittest.TestCase):
    def test_known_values_denominator_and_zero_difference_rule(self):
        for intervals, expected in [([900, 1000, 1100, 1200], 0.0),
                                    ([900, 1100, 900, 1100], 2 / 4),
                                    ([1000, 1000, 1000, 1000], 2 / 4),
                                    ([900, 1000, 1000, 1100], 2 / 4)]:
            self.assertEqual(compute_HRF(intervals), expected)
            self.assertEqual(compute_HRF(intervals), supplied_pip(intervals))

    def test_epsilon_and_unsigned_input_match_user_formula(self):
        for intervals in [[1000, 1000.000001, 1000.000002],
                          np.array([1100, 900, 1100, 900], dtype=np.uint16)]:
            self.assertEqual(compute_HRF(intervals), supplied_pip(intervals))

    def test_insufficient_or_invalid_input(self):
        for intervals in [[], [1000], [1000, 1100]]:
            self.assertTrue(np.isnan(compute_HRF(intervals)))
        for intervals in [[1000, np.nan, 900], [1000, np.inf, 900], [1000, 0, 900],
                          [1000, -1, 900], [[900, 1100, 1000]]]:
            with self.assertRaises(ValueError):
                compute_HRF(intervals)


if __name__ == "__main__":
    unittest.main()
