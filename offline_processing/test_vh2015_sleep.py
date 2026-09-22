"""Synthetic VH2015 and selected-window regressions; no personal recordings."""

import unittest

import numpy as np
import pandas as pd

from analysis_windows import (classify_bursts, first_overlapping_interval, interval_ids,
                              select_observations, validate_intervals)
from burst_hr_response import analyze_burst_hr, sample_hr
from detect_acc_bursts import detect_bursts_in_intervals, select_bursts_by_state
from gp_hrv import interburst_hrv
from vh2015_sleep import (apply_minimum_wake, compute_anglez, detect_sleep_wake,
                          select_analysis_intervals, vh2015_sleep_wake_from_anglez)


ORIGIN = pd.Timestamp("2026-01-01", tz="UTC")


def bounds(pairs):
    return pd.DataFrame([(ORIGIN + pd.Timedelta(seconds=a), ORIGIN + pd.Timedelta(seconds=b))
                         for a, b in pairs], columns=["start", "end"])


def acceleration(seconds, rate=100, angle=None):
    t = np.arange(round(seconds * rate)) / rate
    degrees = np.full(len(t), 30.0) if angle is None else angle(t)
    radians = np.deg2rad(degrees)
    return pd.DataFrame({"sample_time": ORIGIN + pd.to_timedelta(t, unit="s"),
                         "accel_x": 1000 * np.cos(radians), "accel_y": 0.0,
                         "accel_z": 1000 * np.sin(radians)})


class VH2015Tests(unittest.TestCase):
    def test_five_minutes_including_edges_and_strict_angle_threshold(self):
        self.assertFalse(vh2015_sleep_wake_from_anglez(np.zeros(59)).any())
        self.assertTrue(vh2015_sleep_wake_from_anglez(np.zeros(60)).all())
        self.assertTrue(vh2015_sleep_wake_from_anglez(np.tile([0, 5], 30)).all())
        self.assertFalse(vh2015_sleep_wake_from_anglez(np.tile([0, 5.001], 30)).any())
        values = np.r_[np.zeros(60), np.tile([10, 20], 6), np.zeros(60)]
        expected = np.r_[np.ones(60), np.zeros(12), np.ones(60)]
        np.testing.assert_array_equal(vh2015_sleep_wake_from_anglez(values), expected)

    def test_raw_xyz_sleep_wake_and_scale_invariance(self):
        raw = acceleration(900, angle=lambda t: np.where((t >= 360) & (t < 480),
                            np.where((t // 10) % 2 == 0, -30, 30), 30))
        original = raw.copy(deep=True)
        result = detect_sleep_wake(raw)
        epochs = result["epochs"]
        self.assertEqual(len(epochs), 180)
        self.assertTrue(epochs.iloc[20].sleep)
        self.assertTrue(epochs.iloc[80].wake)
        self.assertTrue(epochs.iloc[-20].sleep)
        self.assertEqual(result["coverage"].start.iloc[0], ORIGIN)
        self.assertEqual(result["coverage"].end.iloc[-1], ORIGIN + pd.Timedelta(seconds=900))
        scaled = raw.copy()
        scaled[["accel_x", "accel_y", "accel_z"]] /= 1000
        pd.testing.assert_frame_equal(epochs, detect_sleep_wake(scaled)["epochs"])
        pd.testing.assert_frame_equal(raw, original)
        xyz = raw[["accel_x", "accel_y", "accel_z"]].to_numpy()
        np.testing.assert_array_equal(epochs.anglez, compute_anglez(xyz, 100))

    def test_acquisition_gaps_cannot_create_a_five_minute_sleep_run(self):
        first = acceleration(180)
        second = first.copy()
        second.sample_time += pd.Timedelta(seconds=190)
        result = detect_sleep_wake(pd.concat([first, second], ignore_index=True))
        self.assertEqual(len(result["coverage"]), 2)
        self.assertFalse(result["epochs"].sleep.any())
        self.assertEqual(len(select_analysis_intervals(result, "wake_only")), 2)
        self.assertTrue(select_analysis_intervals(result, "sleep_only").empty)

    def test_partial_epoch_and_short_input_are_unclassified_not_wake(self):
        result = detect_sleep_wake(acceleration(302.5))
        self.assertEqual(result["report"]["unclassified_seconds"], 2.5)
        self.assertEqual(select_analysis_intervals(result, "whole").end.iloc[0],
                         ORIGIN + pd.Timedelta(seconds=302.5))
        self.assertEqual(select_analysis_intervals(result, "sleep_only").end.iloc[0],
                         ORIGIN + pd.Timedelta(seconds=300))
        short = detect_sleep_wake(acceleration(2))
        self.assertTrue(short["epochs"].empty)
        self.assertTrue(select_analysis_intervals(short, "wake_only").empty)
        self.assertEqual(len(select_analysis_intervals(short)), 1)
        # At 100 Hz the odd centered median needs 51 downsampled points, so an
        # isolated run of exactly 5 s cannot provide a defined reference angle.
        self.assertTrue(detect_sleep_wake(acceleration(5))["epochs"].empty)

    def test_jitter_duplicates_timezone_and_invalid_data(self):
        raw = acceleration(310)
        raw.sample_time += pd.to_timedelta(np.tile([0, 0.001], len(raw) // 2), unit="s")
        duplicated = pd.concat([raw.iloc[::-1], raw.iloc[:1]], ignore_index=True)
        scored = detect_sleep_wake(duplicated)
        self.assertEqual(scored["report"]["duplicate_timestamps_averaged"], 1)
        self.assertTrue(scored["epochs"].sleep.all())
        local = raw.copy()
        local.sample_time = local.sample_time.dt.tz_convert("Europe/Rome").dt.tz_localize(None)
        pd.testing.assert_frame_equal(scored["epochs"], detect_sleep_wake(local, naive_timezone="Europe/Rome")["epochs"])
        raw.loc[10, "accel_z"] = np.nan
        with self.assertRaises(ValueError):
            detect_sleep_wake(raw)

    def test_minimum_wake_none_short_and_exact_cutoff(self):
        epochs = bounds([(i * 5, (i + 1) * 5) for i in range(10)])
        epochs["wake_raw"] = [True, False, True, True, False, True, False, False, True, True]
        epochs["source_segment_id"] = [0] * 9 + [1]
        untouched, episodes = apply_minimum_wake(epochs)
        np.testing.assert_array_equal(untouched.wake, epochs.wake_raw)
        self.assertTrue(episodes.counted_wake.all())
        filtered, episodes = apply_minimum_wake(epochs, 10)
        np.testing.assert_array_equal(filtered.wake, [False, False, True, True, False, False, False, False, False, False])
        self.assertEqual(episodes.counted_wake.sum(), 1)
        self.assertTrue((filtered.sleep == ~filtered.wake).all())
        for minimum in [-1, np.inf, np.nan]:
            with self.assertRaises(ValueError):
                apply_minimum_wake(epochs, minimum)


class SelectedWindowTests(unittest.TestCase):
    def test_bounds_half_open_and_hr_break_even_for_short_exclusion(self):
        intervals = bounds([(0, 3), (4, 10)])
        times = ORIGIN + pd.to_timedelta([0, 2, 3, 4, 10], unit="s")
        np.testing.assert_array_equal(interval_ids(times, intervals), [0, 0, -1, 1, -1])
        obs = pd.DataFrame({"hr_bpm": 60., "break_before": False}, index=times)
        selected = select_observations(obs, intervals)
        hr, _, _ = sample_hr(selected, times, max_gap_s=20)
        self.assertTrue(np.isnan(hr[2]))
        self.assertTrue(np.isnan(hr[4]))
        with self.assertRaises(ValueError):
            validate_intervals(bounds([(0, 3), (2, 4)]))

    def test_bursts_do_not_merge_or_filter_across_excluded_gap(self):
        raw = acceleration(30)
        t = np.arange(len(raw)) / 100
        scale = 1 + 0.1 * np.sin(2 * np.pi * 2 * t)
        raw[["accel_x", "accel_y", "accel_z"]] *= scale[:, None]
        intervals = bounds([(0, 12), (14, 30)])
        result = detect_bursts_in_intervals(raw, intervals)
        bursts = result["bursts"]
        self.assertEqual(len(bursts), 2)  # 2-second exclusion is shorter than merge_gap_s.
        np.testing.assert_array_equal(bursts.analysis_interval_id, [0, 1])
        self.assertLessEqual(bursts.end.iloc[0], intervals.end.iloc[0])
        self.assertGreaterEqual(bursts.start.iloc[1], intervals.start.iloc[1])
        mutated = raw.copy()
        mutated.loc[(t >= 12) & (t < 14), "accel_x"] = 1e9
        pd.testing.assert_frame_equal(bursts, detect_bursts_in_intervals(mutated, intervals)["bursts"])

    def test_hr_boundary_epochs_and_artifacts_never_use_another_interval(self):
        intervals = bounds([(0, 200), (201, 400)])
        bursts = bounds([(60, 62), (180, 182), (202, 204), (300, 302)])
        bursts["AUC"] = [1., 2., 3., 4.]
        times = pd.date_range(ORIGIN, periods=401, freq="s")
        observations = pd.DataFrame({"hr_bpm": np.where(np.arange(401) >= 201, 120., 60.),
                                     "break_before": False}, index=times)
        result = analyze_burst_hr(bursts, observations, ORIGIN, times[-1],
                                  max_gap_s=1000, analysis_intervals=intervals,
                                  artifacts=[(times[198], times[203])])
        events, epochs = result["events"], result["epochs_bpm"]
        self.assertTrue(events.loc[0, "included"])
        self.assertTrue(events.loc[3, "included"])
        self.assertFalse(events.loc[1, "included"])
        self.assertFalse(events.loc[2, "included"])
        self.assertIn("analysis_interval_edge", events.loc[1, "exclusion_reason"])
        self.assertTrue(epochs.loc[1, 20:49].isna().all())
        self.assertTrue(epochs.loc[2, -20:-2].isna().all())
        self.assertTrue(np.isnan(epochs.loc[1, 18]))  # No spline anchors across boundary.
        self.assertEqual(events.loc[2, "gap_before_s"], 20.)
        np.testing.assert_allclose(result["report"]["auc_tertile_cutoffs_g_s"], [2., 3.])
        crossing = bounds([(190, 210)]).assign(AUC=1.)
        boundary = analyze_burst_hr(crossing, observations, ORIGIN, times[-1], analysis_intervals=intervals)
        self.assertFalse(boundary["events"].included.any())
        self.assertIn("burst_crosses_analysis_interval", boundary["events"].exclusion_reason.iloc[0])
        self.assertTrue(boundary["epochs_bpm"].loc[0, 10:49].isna().all())
        with self.assertRaises(ValueError):
            analyze_burst_hr(bounds([(200, 201)]).assign(AUC=1.), observations,
                             ORIGIN, times[-1], analysis_intervals=intervals)

    def test_empty_selection_and_short_interval_are_explicit(self):
        empty = bounds([])
        detected = detect_bursts_in_intervals(acceleration(2), empty)
        self.assertTrue(detected["bursts"].empty)
        observations = pd.DataFrame({"hr_bpm": [60.], "break_before": [False]}, index=[ORIGIN])
        result = analyze_burst_hr(detected["bursts"], observations, ORIGIN,
                                  ORIGIN + pd.Timedelta(seconds=2), analysis_intervals=empty)
        self.assertTrue(result["events"].empty)
        self.assertEqual(result["report"]["total_bursts"], 0)
        short = detect_bursts_in_intervals(acceleration(0.3), bounds([(0, 0.3)]))
        self.assertTrue(short["intervals"].empty)
        self.assertEqual(short["quality"].status.iloc[0], "too_short_for_filter")

    def test_hrv_windows_and_cleaning_respect_selected_intervals(self):
        intervals = bounds([(0, 120), (125, 245)])
        rows = [{"sequence": i + 1, "bbi_ms": 1000.,
                 "callback_time_utc_approx": (ORIGIN + pd.Timedelta(seconds=i)).isoformat()}
                for i in range(245)]
        result = interburst_hrv(rows, bounds([]), ORIGIN, ORIGIN + pd.Timedelta(seconds=245),
                                analysis_intervals=intervals)
        self.assertEqual(len(result["segments"]), 2)
        self.assertEqual(len(result["windows"]), 2)
        self.assertTrue(result["windows"].included.all())
        self.assertEqual(result["intervals"].delivery_run_id.nunique(), 2)
        self.assertEqual(len(result["intervals"]), 240)
        empty = interburst_hrv(rows, bounds([]), ORIGIN, ORIGIN + pd.Timedelta(seconds=245),
                               analysis_intervals=bounds([]))
        self.assertTrue(empty["windows"].empty)
        self.assertTrue(empty["intervals"].empty)


class WholeBurstWakeAssignmentTests(unittest.TestCase):
    def test_one_millisecond_anywhere_is_wake_but_touching_is_sleep(self):
        sleep = bounds([(0, 10), (20, 30)])
        wake = bounds([(10, 20)])
        bursts = bounds([(5, 10), (5, 10.001), (19.999, 25), (20, 25),
                         (5, 25), (12, 13), (29, 31)])
        classified = classify_bursts(bursts, sleep, wake)
        self.assertEqual(classified.sleep_wake.tolist(),
                         ["sleep", "wake", "wake", "sleep", "wake", "wake", "unclassified"])
        pd.testing.assert_frame_equal(classified[["start", "end"]], bursts)

    def test_nanosecond_overlap_is_not_rounded_away(self):
        sleep, wake = bounds([(0, 10)]), bounds([(10, 20)])
        event = bounds([(5, 10)])
        event.loc[0, "end"] += pd.Timedelta(nanoseconds=1)
        self.assertEqual(classify_bursts(event, sleep, wake).sleep_wake.iloc[0], "wake")

    def test_burst_spanning_state_boundaries_is_never_redetected_as_sleep(self):
        raw = acceleration(30)
        t = np.arange(len(raw)) / 100
        raw[["accel_x", "accel_y", "accel_z"]] *= (1 + 0.1 * np.sin(2 * np.pi * 2 * t))[:, None]
        full = detect_bursts_in_intervals(raw, bounds([(0, 30)]))
        sleep, wake = bounds([(0, 12), (14, 30)]), bounds([(12, 14)])
        sleeping = select_bursts_by_state(full, sleep, wake, "sleep_only")
        waking = select_bursts_by_state(full, sleep, wake, "wake_only")
        self.assertTrue(sleeping["bursts"].empty)
        self.assertEqual(len(waking["bursts"]), 1)
        pd.testing.assert_frame_equal(waking["bursts"][["start", "end", "AUC", "duration"]],
                                      full["bursts"][["start", "end", "AUC", "duration"]])
        self.assertEqual(len(sleeping["signals"]), 2)
        for part in sleeping["signals"]:
            self.assertFalse(((part["score"].index >= wake.start.iloc[0])
                              & (part["score"].index < wake.end.iloc[0])).any())
        self.assertEqual(len(waking["intervals"]), 1)

    def test_minimum_wake_setting_applies_before_burst_labels(self):
        epochs = bounds([(0, 5), (5, 10), (10, 15)])
        epochs["wake_raw"] = [False, True, False]
        epochs["source_segment_id"] = 0
        event = bounds([(4, 6)])
        for minimum, expected in [(None, "wake"), (5, "wake"), (10, "sleep")]:
            filtered, _ = apply_minimum_wake(epochs, minimum)
            result = {"epochs": filtered, "coverage": bounds([(0, 15)])}
            scored = classify_bursts(event, select_analysis_intervals(result, "sleep_only"),
                                      select_analysis_intervals(result, "wake_only"))
            self.assertEqual(scored.sleep_wake.iloc[0], expected)

    def test_wake_burst_onset_outside_wake_is_retained_without_sleep_hr(self):
        intervals = bounds([(100, 200), (205, 300)])
        events = bounds([(99.999, 105), (190, 220)]).assign(AUC=[1., 2.])
        times = pd.date_range(ORIGIN, periods=301, freq="s")
        observations = pd.DataFrame({"hr_bpm": 60., "break_before": False}, index=times)
        result = analyze_burst_hr(events, observations, ORIGIN, times[-1], analysis_intervals=intervals)
        self.assertEqual(len(result["events"]), 2)
        self.assertFalse(result["events"].included.any())
        self.assertTrue(result["events"].exclusion_reason.str.contains("burst_crosses_analysis_interval").all())
        self.assertTrue(result["epochs_bpm"].loc[0, -20:0].isna().all())
        self.assertTrue(result["epochs_bpm"].loc[1, 10:49].isna().all())
        self.assertEqual(first_overlapping_interval(events.start, events.end, intervals).tolist(), [0, 0])

    def test_unclassified_and_empty_selections(self):
        full = detect_bursts_in_intervals(acceleration(30), bounds([(0, 30)]))
        for mode in ("whole", "sleep_only", "wake_only"):
            selected = select_bursts_by_state(full, bounds([]), bounds([]), mode)
            self.assertTrue(selected["bursts"].empty)
            if mode != "whole":
                self.assertTrue(selected["intervals"].empty)

    def test_excluded_wake_tails_still_block_sleep_hr_isolation_and_late_overlap(self):
        times = pd.date_range(ORIGIN, periods=301, freq="s")
        observations = pd.DataFrame({"hr_bpm": 60., "break_before": False}, index=times)
        # Sleep starts at 50, but a wake-labelled movement continues until 70.
        # A second wake-labelled burst starts at 140, preceding the wake at 160.
        movement = bounds([(40, 70), (80, 82), (100, 102), (140, 170)]).assign(AUC=[1., 2., 3., 4.])
        for selected_id, reason in [(1, "not_isolated_30s"), (2, "other_movement_in_hr_epoch")]:
            result = analyze_burst_hr(movement.iloc[[selected_id]], observations, ORIGIN, times[-1],
                                      analysis_intervals=bounds([(50, 160)]), movement_bursts=movement,
                                      exclude_late_overlap=True)
            self.assertFalse(result["events"].included.any())
            self.assertIn(reason, result["events"].exclusion_reason.iloc[0])
            self.assertEqual(result["report"]["total_bursts"], 1)
        # The same tails must prevent quiet HRV windows in the sleep interval.
        hrv = interburst_hrv([], movement, ORIGIN, times[-1], analysis_intervals=bounds([(50, 160)]))
        self.assertEqual(hrv["segments"].start.iloc[0], ORIGIN + pd.Timedelta(seconds=71))
        self.assertEqual(hrv["segments"].end.iloc[-1], ORIGIN + pd.Timedelta(seconds=140))


if __name__ == "__main__":
    unittest.main()
