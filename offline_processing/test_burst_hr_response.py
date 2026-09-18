import unittest

import numpy as np
import pandas as pd

from burst_hr_response import (
    BASELINE, RELATIVE_SECONDS, analyze_burst_hr, assign_auc_tertiles, bbi_to_hr, sample_hr,
)


ORIGIN = pd.Timestamp("2026-09-13", tz="UTC")


def at(seconds):
    return ORIGIN + pd.to_timedelta(seconds, unit="s")


def rows(intervals, seconds=None, sequences=None):
    seconds = np.arange(len(intervals)) if seconds is None else seconds
    sequences = np.arange(1, len(intervals) + 1) if sequences is None else sequences
    return [{"bbi_ms": bbi, "callback_time_utc_approx": at(t).isoformat(), "sequence": int(seq)}
            for bbi, t, seq in zip(intervals, seconds, sequences)]


def observations(seconds, hr=60.0):
    index = at(np.asarray(seconds))
    return pd.DataFrame({"hr_bpm": hr, "break_before": False}, index=index)


def bursts(starts=(60,), ends=None, auc=None):
    ends = np.asarray(starts) + 1 if ends is None else ends
    auc = np.arange(1, len(starts) + 1) if auc is None else auc
    return pd.DataFrame({"start": at(starts), "end": at(ends), "AUC": auc})


class BbiPreparationTests(unittest.TestCase):
    def test_all_equal_intervals_preserved_and_batch_hr_averaged(self):
        obs, report = bbi_to_hr(rows([1000, 1000, 500], [1, 1, 1]))
        self.assertEqual(report["received_bbi"], 3)
        self.assertEqual(obs.n_bbi.iloc[0], 3)
        self.assertEqual(obs.hr_bpm.iloc[0], 80)
        self.assertEqual(obs.index[0], at(1))

    def test_bad_bbi_invalidates_whole_callback(self):
        obs, report = bbi_to_hr(rows([1000, 1, None, 5000], [0, 0, 1, 2]))
        self.assertEqual(report["invalid_or_out_of_range_bbi"], 3)
        self.assertTrue(obs.hr_bpm.isna().all())

    def test_sequence_holes_block_interpolation(self):
        obs, _ = bbi_to_hr(rows([1000, 1000, 1000], sequences=[1, 3, 4]))
        hr, _, _ = sample_hr(obs, at([0.5, 1, 1.5]))
        self.assertTrue(np.isnan(hr[0]))
        np.testing.assert_allclose(hr[1:], 60)

    def test_hole_inside_batch_invalidates_it(self):
        obs, _ = bbi_to_hr(rows([1000, 1000, 1000], [0, 1, 1], [1, 2, 4]))
        self.assertTrue(np.isnan(obs.hr_bpm.iloc[-1]))

    def test_bad_order_duplicates_and_times_rejected(self):
        for seq in ([1, 1], [2, 1], [0, 1]):
            with self.assertRaises(ValueError):
                bbi_to_hr(rows([1000, 1000], sequences=seq))
        with self.assertRaises(ValueError):
            bbi_to_hr(rows([1000, 1000], seconds=[2, 1]))
        with self.assertRaises(ValueError):
            bbi_to_hr(rows([1000]), bbi_limits_ms=(2000, 300))

    def test_empty_bbi_is_valid_missing_data(self):
        obs, report = bbi_to_hr([])
        hr, _, _ = sample_hr(obs, at([0, 1]))
        self.assertTrue(np.isnan(hr).all())
        self.assertEqual(report["received_bbi"], 0)


class InterpolationTests(unittest.TestCase):
    def test_linear_interpolation_and_no_extrapolation(self):
        obs = observations([0, 2], [60, 80])
        hr, gaps, repaired = sample_hr(obs, at([-1, 0, 1, 2, 3]))
        np.testing.assert_allclose(hr, [np.nan, 60, 70, 80, np.nan], equal_nan=True)
        self.assertEqual(gaps[2], 2)
        self.assertFalse(repaired.any())

    def test_long_and_exact_limit_gaps(self):
        obs = observations([0, 3, 7])
        hr, _, _ = sample_hr(obs, at([1, 3, 5, 7]), max_gap_s=3)
        np.testing.assert_allclose(hr, [60, 60, np.nan, 60], equal_nan=True)

    def test_invalid_callback_not_bridged(self):
        obs = observations([0, 1, 2], [60, np.nan, 60])
        hr, _, _ = sample_hr(obs, at([0.5, 1, 1.5]))
        self.assertTrue(np.isnan(hr).all())

    def test_short_annotated_artifact_cubic_repair(self):
        obs = observations(np.arange(20))
        obs.loc[at(np.arange(5, 10)), "hr_bpm"] = 200
        hr, _, repaired = sample_hr(obs, at(np.arange(20)), artifacts=[(at(5), at(10))])
        np.testing.assert_allclose(hr, 60)
        self.assertEqual(repaired.sum(), 5)

    def test_ten_second_artifact_not_repaired(self):
        obs = observations(np.arange(30))
        hr, _, repaired = sample_hr(obs, at(np.arange(5, 15)), artifacts=[(at(5), at(15))])
        self.assertTrue(np.isnan(hr).all())
        self.assertFalse(repaired.any())

    def test_unsupported_spline_and_dropout_not_repaired(self):
        obs = observations([0, 1, 10, 11])
        hr, _, repaired = sample_hr(obs, at([5]), artifacts=[(at(2), at(9))])
        self.assertTrue(np.isnan(hr).all())
        self.assertFalse(repaired.any())
        obs = observations(np.arange(20))
        obs.loc[at(7), "break_before"] = True
        hr, _, repaired = sample_hr(obs, at([7]), artifacts=[(at(5), at(10))])
        self.assertTrue(np.isnan(hr).all())
        self.assertFalse(repaired.any())

    def test_overlapping_and_reversed_annotations_rejected(self):
        for annotations in ([(at(5), at(4))], [(at(1), at(3)), (at(2), at(4))]):
            with self.assertRaises(ValueError):
                sample_hr(observations(np.arange(20)), at([1]), artifacts=annotations)


class EventAnalysisTests(unittest.TestCase):
    def analyze(self, events=None, obs=None, **kwargs):
        return analyze_burst_hr(bursts() if events is None else events,
                                observations(np.arange(301)) if obs is None else obs,
                                at(0), at(300), **kwargs)

    def test_current_epoch_baseline_and_known_response(self):
        obs = observations(np.arange(301))
        obs.loc[at(65), "hr_bpm"] = 72
        result = self.analyze(obs=obs)
        event = result["events"].iloc[0]
        self.assertEqual(list(RELATIVE_SECONDS), list(range(-19, 55)))
        self.assertEqual(BASELINE.sum(), 14)
        self.assertTrue(event.included)
        self.assertEqual(event.baseline_bpm, 60)
        self.assertAlmostEqual(event.hr_peak_increase_pct, 20)
        self.assertEqual(event.hr_peak_latency_s, 5)
        self.assertAlmostEqual(result["epochs_pct"].iloc[0, np.where(RELATIVE_SECONDS == 5)[0][0]], 20)

    def test_baseline_right_endpoint_excluded(self):
        obs = observations(np.arange(301))
        obs.loc[at(55), "hr_bpm"] = 120  # -5 s must not enter the fourteen-sample baseline.
        result = self.analyze(obs=obs)
        self.assertEqual(result["events"].baseline_bpm.iloc[0], 60)

    def test_decline_not_reported_as_positive_peak(self):
        obs = observations(np.arange(301))
        obs.loc[at(np.arange(61, 115)), "hr_bpm"] = 54
        event = self.analyze(obs=obs)["events"].iloc[0]
        self.assertAlmostEqual(event.hr_post_max_pct, -10)
        self.assertEqual(event.hr_peak_increase_pct, 0)
        self.assertTrue(np.isnan(event.hr_peak_latency_s))

    def test_isolation_uses_offset_and_exact_thirty_seconds(self):
        events = bursts([60, 91, 120], [61, 92, 121])
        result = self.analyze(events=events)["events"]
        self.assertTrue(result.isolated.iloc[0])
        self.assertFalse(result.isolated.iloc[1])
        self.assertFalse(result.isolated.iloc[2])
        self.assertTrue(result.other_movement_in_epoch.iloc[0])

    def test_long_and_nested_movements_are_not_isolated(self):
        events = bursts([60, 70, 110], [100, 71, 111])
        result = self.analyze(events=events)["events"]
        self.assertFalse(result.isolated.any())
        self.assertEqual(result.gap_before_s.iloc[-1], 10)

    def test_optional_full_epoch_exclusion_is_explicit(self):
        events = bursts([60, 95])
        result = self.analyze(events=events)
        self.assertEqual(result["events"].included.sum(), 2)
        strict = self.analyze(events=events, exclude_late_overlap=True)
        self.assertEqual(strict["events"].included.sum(), 1)
        self.assertIn("other_movement_in_hr_epoch", strict["events"].exclusion_reason.iloc[0])

    def test_missing_hr_is_excluded_not_zero_response(self):
        obs = observations(np.r_[np.arange(60), np.arange(70, 301)])
        result = self.analyze(obs=obs)
        event = result["events"].iloc[0]
        self.assertFalse(event.included)
        self.assertTrue(np.isnan(event.hr_peak_increase_pct))
        self.assertIn("insufficient_bbi", event.exclusion_reason)
        self.assertEqual(result["summary"].included_bursts.sum(), 0)

    def test_full_epoch_guard_includes_extended_window_and_endpoint(self):
        for second_onset in (110, 114):  # +50 and +54 s are inside the current epoch.
            events = bursts([60, second_onset])
            result = self.analyze(events=events, exclude_late_overlap=True)
            self.assertTrue(result["events"].other_movement_in_epoch.iloc[0])
            self.assertFalse(result["events"].included.iloc[0])
            self.assertIn("other_movement_in_hr_epoch", result["events"].exclusion_reason.iloc[0])
        outside = self.analyze(events=bursts([60, 115]), exclude_late_overlap=True)
        self.assertFalse(outside["events"].other_movement_in_epoch.iloc[0])
        self.assertTrue(outside["events"].included.iloc[0])

    def test_nonisolated_numerics_kept_but_not_summarized(self):
        result = self.analyze(events=bursts([60, 65]))
        self.assertTrue(result["events"].hr_complete.all())
        self.assertFalse(result["events"].included.any())
        self.assertTrue(result["events"].hr_peak_increase_pct.notna().all())

    def test_recording_edges_and_timezone_handling(self):
        events = bursts([5, 60])
        events.start = events.start.dt.tz_convert("Europe/Rome")
        events.end = events.end.dt.tz_convert("Europe/Rome")
        result = self.analyze(events=events)
        self.assertIn("recording_edge", result["events"].exclusion_reason.iloc[0])
        self.assertTrue(result["events"].included.iloc[1])
        self.assertEqual(str(result["events"].start.dt.tz), "UTC")

    def test_tertiles_use_all_candidates_not_hr_subset(self):
        events = bursts([60, 120, 180], auc=[1, 2, 100])
        obs = observations(np.r_[np.arange(110), np.arange(160, 301)])
        result = self.analyze(events=events, obs=obs)
        self.assertEqual(list(result["events"].auc_tertile), ["Low", "Medium", "High"])
        self.assertFalse(result["events"].included.iloc[1])
        self.assertEqual(result["summary"].total_bursts.tolist(), [1, 1, 1])
        self.assertTrue(result["curves"].query("auc_tertile == 'Low'").sem_pct.isna().all())

    def test_tertile_ties_empty_and_invalid_auc(self):
        groups, cutoffs = assign_auc_tertiles([2, 2, 2])
        self.assertEqual(list(groups), ["Low"] * 3)
        self.assertEqual(cutoffs, (2, 2))
        self.assertEqual(len(assign_auc_tertiles([])[0]), 0)
        for values in ([np.nan], [-1]):
            with self.assertRaises(ValueError):
                assign_auc_tertiles(values)

    def test_empty_events_and_no_bbi_remain_inspectable(self):
        empty = self.analyze(events=bursts([]))
        self.assertEqual(len(empty["events"]), 0)
        self.assertIn("hr_post_min_pct", empty["events"])
        self.assertEqual(empty["summary"].total_bursts.sum(), 0)
        obs, _ = bbi_to_hr([])
        result = self.analyze(obs=obs)
        self.assertEqual(result["events"].hr_coverage_pct.iloc[0], 0)
        self.assertFalse(result["events"].included.iloc[0])


if __name__ == "__main__":
    unittest.main()
