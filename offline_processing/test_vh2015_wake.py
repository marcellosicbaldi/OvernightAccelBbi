"""Synthetic VH2015 and selected-window regressions; no private recordings."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis_windows import (
    detect_bursts_in_intervals, interval_ids, select_observations, validate_intervals,
)
from burst_hr_response import analyze_burst_hr, sample_hr
from detect_acc_bursts import detect_bursts, prepare_acceleration
from gp_hrv import interburst_hrv
from vh2015_wake import (
    classify_epochs, compute_anglez, detect_vh2015, select_analysis_intervals, state_episodes,
    vh2015_sleep_wake_from_anglez,
)
from test_burst_hr_response import at, bursts, observations, rows


def acceleration(seconds=630, fs=100, active=True, tz="Europe/Rome"):
    t = np.arange(round(seconds * fs)) / fs
    angle = np.zeros(len(t))
    if active:
        angle[(t >= 300) & (t < 330)] = 40 * np.sin(t[(t >= 300) & (t < 330)] / 2)
    xyz = np.column_stack([np.cos(np.radians(angle)), np.zeros(len(t)), np.sin(np.radians(angle))])
    return pd.DataFrame({"sample_time": pd.date_range("2026-01-01", periods=len(t),
                         freq=pd.Timedelta(seconds=1/fs), tz=tz),
                         "accel_x": xyz[:, 0] * 1000, "accel_y": xyz[:, 1] * 1000,
                         "accel_z": xyz[:, 2] * 1000})


def windows(*pairs):
    return pd.DataFrame({"start": at([p[0] for p in pairs]), "end": at([p[1] for p in pairs])})


class Vh2015Tests(unittest.TestCase):
    def test_ported_angle_matches_reference_golden_with_resize_fix(self):
        # Values computed independently with the supplied sleep_menghini module.
        t = np.arange(3007) / 100
        xyz = np.column_stack([np.cos(t/7), np.sin(t/4)*0.2, np.sin(t/7)])
        np.testing.assert_allclose(compute_anglez(xyz, 100),
                                   [25.08, 59.2979, 75.2327, 36.5505, -3.795, -8.3208])

    def test_exact_five_minute_and_five_degree_rules_and_edges(self):
        self.assertTrue(vh2015_sleep_wake_from_anglez(np.zeros(60)).all())
        self.assertFalse(vh2015_sleep_wake_from_anglez(np.zeros(59)).any())
        self.assertTrue(vh2015_sleep_wake_from_anglez(np.tile([0, 5], 30)).all())
        self.assertFalse(vh2015_sleep_wake_from_anglez(np.tile([0, 5.001], 30)).any())
        angle = np.r_[np.zeros(60), np.tile([30, -30], 6), np.zeros(60)]
        np.testing.assert_array_equal(vh2015_sleep_wake_from_anglez(angle),
                                      np.r_[np.ones(60), np.zeros(12), np.ones(60)])

    def test_minimum_wake_none_zero_exact_and_suppression(self):
        angle = np.r_[np.zeros(60), np.tile([30, -30], 6), np.zeros(60)]
        for minimum in (None, 0, 60):
            epochs = classify_epochs(angle, at(0), min_wake_duration_s=minimum)
            self.assertEqual(state_episodes(epochs).duration_s.tolist(), [60])
            self.assertFalse(epochs.wake_suppressed.any())
        epochs = classify_epochs(angle, at(0), min_wake_duration_s=60.01)
        self.assertTrue(state_episodes(epochs).empty)
        self.assertEqual(state_episodes(epochs, "raw_wake").duration_s.tolist(), [60])
        self.assertEqual(epochs.wake_suppressed.sum(), 12)
        self.assertTrue(epochs.sleep.all())

    def test_edge_wake_bouts_also_use_minimum(self):
        epochs = classify_epochs(np.r_[30, -30, np.zeros(60), 30], at(0), min_wake_duration_s=10)
        self.assertEqual(state_episodes(epochs).duration_s.tolist(), [10])
        self.assertTrue(epochs.wake_suppressed.iloc[-1])

    def test_raw_crop_time_units_duplicates_and_input_preservation(self):
        raw = acceleration(seconds=630.07)
        before = raw.copy(deep=True)
        result = detect_vh2015(raw)
        self.assertEqual(len(result["epochs"]), 126)
        self.assertAlmostEqual(result["report"]["unclassified_tail_s"], 0.07)
        self.assertEqual(result["recording_start"], raw.sample_time.iloc[0])
        self.assertEqual(str(result["epochs"].start.dt.tz), "Europe/Rome")
        self.assertGreater(len(result["wake_episodes"]), 0)
        scaled = raw.copy()
        scaled[["accel_x", "accel_y", "accel_z"]] /= 1000
        duplicate = pd.concat([scaled, scaled.iloc[[100]]]).sample(frac=1, random_state=42)
        second = detect_vh2015(duplicate)
        pd.testing.assert_frame_equal(result["epochs"], second["epochs"])
        self.assertEqual(second["report"]["duplicate_timestamps_averaged"], 1)
        pd.testing.assert_frame_equal(raw, before)

    def test_all_modes_partition_complete_epochs_and_whole_keeps_tail(self):
        result = detect_vh2015(acceleration(seconds=630.07))
        selected = {mode: select_analysis_intervals(result, mode)
                    for mode in ("whole_spt", "sleep_only", "wake_only")}
        self.assertEqual(len(selected["whole_spt"]), 1)
        self.assertEqual(selected["whole_spt"].end.iloc[0], result["recording_end"])
        epochs = result["epochs"]
        sleep_ids = interval_ids(epochs.start, selected["sleep_only"])
        wake_ids = interval_ids(epochs.start, selected["wake_only"])
        np.testing.assert_array_equal(sleep_ids >= 0, epochs.sleep)
        np.testing.assert_array_equal(wake_ids >= 0, epochs.wake)
        self.assertTrue(((sleep_ids >= 0) ^ (wake_ids >= 0)).all())
        self.assertEqual(interval_ids([epochs.end.iloc[-1]], selected["wake_only"])[0], -1)
        with self.assertRaises(ValueError):
            select_analysis_intervals(result, "unknown")

    def test_short_all_sleep_all_wake_and_invalid_inputs(self):
        short = detect_vh2015(acceleration(4, active=False))
        self.assertTrue(short["epochs"].empty)
        self.assertTrue(select_analysis_intervals(short, "wake_only").empty)
        self.assertEqual(len(select_analysis_intervals(short)), 1)
        self.assertTrue(detect_vh2015(acceleration(10, active=False))["epochs"].wake.all())
        asleep = detect_vh2015(acceleration(300, active=False))
        self.assertTrue(select_analysis_intervals(asleep, "wake_only").empty)
        self.assertEqual(len(select_analysis_intervals(asleep, "sleep_only")), 1)
        raw = acceleration(10)
        for bad in (np.nan, np.inf):
            invalid = raw.copy()
            invalid.loc[0, "accel_z"] = bad
            with self.assertRaises(ValueError):
                detect_vh2015(invalid)
        with self.assertRaisesRegex(ValueError, "gap"):
            detect_vh2015(raw.drop(index=range(100, 150)))
        for minimum in (-1, np.nan, np.inf, "30", True):
            with self.assertRaises(ValueError):
                classify_epochs([0], at(0), min_wake_duration_s=minimum)
        with self.assertRaises(ValueError):
            vh2015_sleep_wake_from_anglez([0, np.nan])


class SelectedIntervalTests(unittest.TestCase):
    def test_interval_edges_empty_and_invalid_overlap(self):
        selected = validate_intervals(windows((0, 10), (10, 20), (21, 30)), at(0), at(30))
        np.testing.assert_array_equal(interval_ids(at([-1, 0, 10, 20, 21, 30]), selected),
                                      [-1, 0, 1, -1, 2, -1])
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_intervals(windows((0, 20), (10, 30)), at(0), at(30))
        self.assertTrue(validate_intervals(windows(), at(0), at(30)).empty)

    def test_bursts_filtered_independently_across_gap_shorter_than_merge(self):
        raw = acceleration(20, active=False, tz="UTC")
        start = raw.sample_time.iloc[0]
        selected = pd.DataFrame({"start": [start, start + pd.Timedelta(seconds=8)],
                                 "end": [start + pd.Timedelta(seconds=5), start + pd.Timedelta(seconds=13)]})
        # Force a continuous above-threshold score in each allowed interval.
        with patch("detect_acc_bursts.compute_envelope", side_effect=lambda x, **kw: pd.Series(0.05, index=x.index)):
            detected, signals, qc = detect_bursts_in_intervals(raw, selected)
        self.assertEqual(len(detected), 2)
        self.assertEqual(detected.analysis_interval_id.tolist(), [0, 1])
        self.assertEqual(detected.duration.dt.total_seconds().tolist(), [5, 5])
        self.assertEqual(len(signals), 2)
        self.assertTrue((qc.status == "ok").all())
        for signal, interval in zip(signals, selected.itertuples()):
            self.assertTrue((signal["magnitude"].index >= interval.start).all())
            self.assertTrue((signal["magnitude"].index < interval.end).all())

    def test_whole_night_matches_existing_burst_pipeline(self):
        raw = acceleration(20, active=False)
        raw.accel_x += 100 * np.sin(np.arange(len(raw)) / 7)
        magnitude, _ = prepare_acceleration(raw)
        expected = detect_bursts(magnitude, 100, alfa=0.04)
        actual, _, _ = detect_bursts_in_intervals(raw, None)
        pd.testing.assert_frame_equal(actual.drop(columns="analysis_interval_id").rename_axis(None), expected)

    def test_empty_and_filter_padding_selection(self):
        raw = acceleration(10, active=False, tz="UTC")
        start = raw.sample_time.iloc[0]
        selected = pd.DataFrame({"start": [start], "end": [start + pd.Timedelta(seconds=0.5)]})
        detected, signals, qc = detect_bursts_in_intervals(raw, selected)
        self.assertTrue(detected.empty)
        self.assertFalse(signals)
        self.assertEqual(qc.status.tolist(), ["too_short_for_filter"])
        detected, signals, _ = detect_bursts_in_intervals(raw, selected.iloc[:0])
        self.assertTrue(detected.empty)
        self.assertIn("analysis_interval_id", detected)

    def test_selected_hr_barrier_even_with_subsecond_exclusion(self):
        selected = windows((0, 5.2), (5.8, 20))
        obs = select_observations(observations(np.arange(20)), selected)
        values, _, _ = sample_hr(obs, at([5, 5.5, 6]))
        np.testing.assert_allclose(values, [60, np.nan, 60], equal_nan=True)

    def test_hr_epochs_and_repair_cannot_cross_selected_boundary(self):
        # A 1-second excluded gap is shorter than the interpolation limit.
        selected = windows((0, 65), (66, 150))
        obs = observations(np.arange(151))
        result = analyze_burst_hr(bursts([40, 100]), obs, at(0), at(150),
                                  analysis_intervals=selected, artifacts=[(at(63), at(68))], isolation_s=0)
        first = result["events"].iloc[0]
        self.assertFalse(first.included)
        self.assertIn("analysis_interval_edge", first.exclusion_reason)
        self.assertTrue(result["epochs_bpm"].iloc[0].loc[25:].isna().all())
        self.assertTrue(result["epochs_bpm"].iloc[0].loc[23:24].isna().all())
        self.assertEqual(first.cubic_repaired_samples, 0)
        self.assertTrue(result["events"].included.iloc[1])

    def test_hr_tertiles_global_selection_and_interval_isolation(self):
        selected = windows((0, 80), (100, 200), (250, 400))
        result = analyze_burst_hr(bursts([40, 90, 140, 290], auc=[1, 1000, 2, 3]),
                                  observations(np.arange(401)), at(0), at(400),
                                  analysis_intervals=selected)
        events = result["events"]
        self.assertEqual(events.AUC.tolist(), [1, 2, 3])
        self.assertEqual(events.auc_tertile.tolist(), ["Low", "Medium", "High"])
        self.assertTrue(events.gap_before_s.isna().all())
        self.assertTrue(events.gap_after_s.isna().all())
        self.assertFalse(events.included.iloc[0])  # response extends past 80 s
        self.assertTrue(events.included.iloc[1:].all())

    def test_interpolation_anchors_must_be_in_same_interval(self):
        selected = windows((0, 60.5), (61, 200))
        result = analyze_burst_hr(bursts([40.25]), observations(np.arange(201)), at(0), at(200),
                                  analysis_intervals=selected, isolation_s=0)
        # +20=60.25 is inside, but its right anchor at 61 is excluded from interval 0.
        self.assertTrue(np.isnan(result["epochs_bpm"].iloc[0][20]))

    def test_empty_hr_selection_remains_typed_and_summarizable(self):
        result = analyze_burst_hr(bursts([40]), observations(np.arange(201)), at(0), at(200),
                                  analysis_intervals=windows())
        self.assertTrue(result["events"].empty)
        self.assertEqual(result["summary"].total_bursts.sum(), 0)

    def test_hrv_cleaning_and_windows_never_span_excluded_periods(self):
        selected = windows((0, 120), (121, 241))
        result = interburst_hrv(rows(np.full(241, 1000)), bursts([]), at(0), at(241),
                                analysis_intervals=selected)
        self.assertEqual(result["segments"].analysis_interval_id.tolist(), [0, 1])
        self.assertEqual(result["windows"].window_length_s.tolist(), [120, 120])
        self.assertTrue(result["windows"].included.all())
        self.assertFalse((result["intervals"].callback_time_utc_approx == at(120)).any())
        self.assertEqual(result["intervals"].delivery_run_id.nunique(), 2)
        empty = interburst_hrv([], bursts([]), at(0), at(241), analysis_intervals=windows())
        self.assertTrue(empty["windows"].empty)
        self.assertTrue(empty["intervals"].empty)

    def test_hrv_post_burst_guard_stops_at_selection_edge(self):
        result = interburst_hrv(rows(np.full(241, 1000)), bursts([115], ends=[120]),
                                at(0), at(241), analysis_intervals=windows((0, 120), (120.5, 241)))
        self.assertEqual(result["blocked_bursts"].end.iloc[0], at(120))
        self.assertEqual(result["segments"].start.iloc[-1], at(120.5))


if __name__ == "__main__":
    unittest.main()
