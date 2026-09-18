"""Check ID-safe joins, outcome-blind matching, and empty-group behavior."""

import unittest

import numpy as np
import pandas as pd

from burst_hr_response import analyze_burst_hr, bbi_to_hr
from reorientation_hr import join_hr_orientation, match_by_auc, summarize_groups, summarize_pairs
from wrist_reorientation import CHANGED, LITTLE, UNKNOWN, measure_reorientation


def events_fixture():
    start = pd.date_range("2026-01-01", periods=4, freq="2min", tz="UTC")
    return pd.DataFrame({
        "start": start, "end": start + pd.Timedelta(seconds=5),
        "AUC": [1., 1.2, .9, 1.1], "auc_tertile": ["Low"] * 4,
        "reorientation_class": [CHANGED, CHANGED, LITTLE, LITTLE],
        "tilt_change_deg": [60., 45., 3., 4.], "orientation_usable": True,
        "orientation_reason": "", "included": True, "exclusion_reason": "",
        "comparison_included": True, "hr_peak_increase_pct": [10., 20., 5., 7.],
        "hr_post_mean_pct": [4., 8., 2., 3.],
    }, index=pd.Index([10, 11, 20, 21], name="burst_id"))


class ComparisonTests(unittest.TestCase):
    def test_matching_maximizes_pairs_before_closeness(self):
        pairs = match_by_auc(events_fixture(), max_auc_ratio=1.15)
        self.assertEqual(set(zip(pairs.changed_id, pairs.little_id)), {(10, 20), (11, 21)})
        self.assertFalse(pairs.changed_id.duplicated().any())
        self.assertFalse(pairs.little_id.duplicated().any())
        self.assertTrue((pairs.auc_ratio <= 1.15).all())

    def test_matching_never_crosses_tertiles(self):
        events = events_fixture()
        events.loc[[20, 21], "auc_tertile"] = "Medium"
        self.assertTrue(match_by_auc(events).empty)

    def test_matching_is_outcome_blind(self):
        events = events_fixture()
        original = match_by_auc(events)
        events["hr_peak_increase_pct"] = [100, -20, 40, 0]
        updated = match_by_auc(events)
        pd.testing.assert_frame_equal(original[["changed_id", "little_id"]], updated[["changed_id", "little_id"]])

    def test_invalid_or_ineligible_auc_not_matched(self):
        events = events_fixture()
        events.loc[10, "AUC"] = 0
        events.loc[11, "comparison_included"] = False
        self.assertTrue(match_by_auc(events).empty)

    def test_exact_auc_and_invalid_caliper(self):
        events = events_fixture()
        self.assertTrue(match_by_auc(events, 1).empty)
        events.loc[20, "AUC"] = 1
        self.assertEqual(len(match_by_auc(events, 1)), 1)
        with self.assertRaises(ValueError):
            match_by_auc(events, .9)

    def test_join_checks_ids_and_timestamps(self):
        events = events_fixture()
        orientation = events[["reorientation_class", "orientation_usable", "orientation_reason", "tilt_change_deg"]].copy()
        orientation["burst_start"], orientation["burst_end"] = events.start, events.end
        hr = events.drop(columns=list(orientation.columns.intersection(events.columns)) + ["comparison_included"])
        joined = join_hr_orientation(hr, orientation.iloc[::-1])
        self.assertTrue(joined.comparison_included.all())
        bad = orientation.copy()
        bad.loc[10, "burst_start"] += pd.Timedelta(seconds=1)
        with self.assertRaisesRegex(ValueError, "timestamps"):
            join_hr_orientation(hr, bad)
        with self.assertRaisesRegex(ValueError, "same burst IDs"):
            join_hr_orientation(hr, orientation.iloc[:3])
        orientation.loc[10, "reorientation_class"] = UNKNOWN
        orientation.loc[10, "orientation_usable"] = False
        self.assertFalse(join_hr_orientation(hr, orientation).loc[10].comparison_included)

    def test_group_and_paired_effects(self):
        events = events_fixture()
        epochs = pd.DataFrame([[0, 10], [0, 20], [0, 5], [0, 7]], index=events.index, columns=[-1, 1])
        summary, curves = summarize_groups(events, epochs)
        self.assertEqual(summary.comparison_n.sum(), 4)
        self.assertTrue(curves.loc[curves.auc_tertile == "High", "mean_pct"].isna().all())
        pairs = match_by_auc(events, 1.15)
        paired, _ = summarize_pairs(pairs, epochs)
        self.assertEqual(paired.iloc[0].peak_difference_mean_pp, 9)
        self.assertEqual(paired.iloc[0].pairs_n, 2)

    def test_missing_hr_epochs_raise(self):
        events = events_fixture()
        epochs = pd.DataFrame(0., index=events.index, columns=[-1, 1])
        epochs.loc[10, 1] = np.nan
        with self.assertRaises(ValueError):
            summarize_groups(events, epochs)

    def test_empty_events_and_pairs(self):
        events = events_fixture().iloc[:0]
        epochs = pd.DataFrame(index=events.index, columns=[-1, 1], dtype=float)
        summary, _ = summarize_groups(events, epochs)
        self.assertEqual(summary.comparison_n.sum(), 0)
        pairs = match_by_auc(events)
        paired, curves = summarize_pairs(pairs, epochs)
        self.assertEqual(paired.pairs_n.sum(), 0)
        self.assertTrue(curves.mean_pct.isna().all())

    def test_empty_pipeline_preserves_join_contract(self):
        origin = pd.Timestamp("2026-01-01", tz="UTC")
        bursts = events_fixture().iloc[:0][["start", "end", "AUC"]]
        obs, _ = bbi_to_hr([])
        hr = analyze_burst_hr(bursts, obs, origin, origin + pd.Timedelta(hours=1))
        axes = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"), columns=["x_g", "y_g", "z_g"], dtype=float)
        orientation = measure_reorientation(axes, hr["events"])
        events = join_hr_orientation(hr["events"], orientation)
        self.assertTrue(events.empty)
        self.assertTrue(match_by_auc(events).empty)


if __name__ == "__main__":
    unittest.main()
