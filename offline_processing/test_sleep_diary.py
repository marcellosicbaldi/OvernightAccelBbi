"""Regression tests for lightsoff-date matching and midnight/DST boundaries."""

from datetime import time
import unittest

import pandas as pd

from sleep_diary import select_sleep_interval


class SleepDiaryTests(unittest.TestCase):
    def diary(self, day="2026-09-14", lights=time(23, 42), wake=time(7, 12)):
        return pd.DataFrame({"Date": [pd.Timestamp(day)], "lightsoff": [lights], "wakeup": [wake]})

    def select(self, diary, start="2026-09-14 23:00", end="2026-09-15 08:00"):
        return select_sleep_interval(diary, pd.Timestamp(start, tz="Europe/Rome"),
                                     pd.Timestamp(end, tz="Europe/Rome"))

    def test_midnight_rollover_and_no_input_mutation(self):
        diary = self.diary()
        before = diary.copy(deep=True)
        selected, skipped = self.select(diary)
        self.assertEqual(str(selected.lightsoff_at), "2026-09-14 23:42:00+02:00")
        self.assertEqual(str(selected.wakeup_at), "2026-09-15 07:12:00+02:00")
        self.assertTrue(selected.full_diary_interval_recorded)
        self.assertTrue(skipped.empty)
        pd.testing.assert_frame_equal(diary, before)

    def test_after_midnight_lightsoff_uses_entered_date(self):
        selected, _ = self.select(self.diary(lights=time(0, 11)),
                                  "2026-09-13 23:00", "2026-09-14 08:00")
        self.assertEqual(selected.lightsoff_at.day, 14)
        self.assertEqual(selected.wakeup_at.day, 14)

    def test_current_recording_matches_overlap_not_row_order_or_date_alone(self):
        diary = pd.concat([self.diary(lights=time(0, 11)), self.diary()], ignore_index=True)
        selected, _ = self.select(diary)
        self.assertEqual(selected.excel_row, 3)
        self.assertEqual(selected.lightsoff_at.hour, 23)

    def test_partial_recording_starting_next_day(self):
        selected, _ = self.select(self.diary(), "2026-09-15 01:00", "2026-09-15 06:00")
        self.assertFalse(selected.full_diary_interval_recorded)
        self.assertEqual(selected.overlap_seconds, 5 * 3600)
        self.assertEqual(selected.recorded_start.hour, 1)

    def test_dst_calendar_rollover_spring_and_autumn(self):
        for day, next_day, duration in [("2026-03-28", "2026-03-29", 7),
                                        ("2026-10-24", "2026-10-25", 9)]:
            selected, _ = self.select(self.diary(day, time(23), time(7)),
                                      day + " 22:00", next_day + " 08:00")
            self.assertEqual(selected.wakeup_at.hour, 7)
            self.assertEqual((selected.wakeup_at - selected.lightsoff_at).total_seconds(), duration * 3600)

    def test_excel_fraction_and_string_times(self):
        selected, _ = self.select(self.diary(lights=23.5 / 24, wake="7:12"))
        self.assertEqual(selected.lightsoff_at.minute, 30)
        self.assertEqual(selected.wakeup_at.hour, 7)

    def test_incomplete_rows_reported_without_changing_workbook(self):
        diary = pd.concat([self.diary(), self.diary(wake=None)], ignore_index=True)
        selected, skipped = self.select(diary)
        self.assertEqual(selected.excel_row, 2)
        self.assertEqual(skipped.iloc[0].excel_row, 3)

    def test_no_match_and_ambiguous_match_raise(self):
        with self.assertRaisesRegex(ValueError, "No complete diary interval"):
            self.select(self.diary("2026-09-12"))
        with self.assertRaisesRegex(ValueError, "Multiple diary intervals"):
            self.select(pd.concat([self.diary(), self.diary()]))

    def test_touching_interval_does_not_match(self):
        with self.assertRaisesRegex(ValueError, "No complete diary interval"):
            self.select(self.diary(), "2026-09-15 07:12", "2026-09-15 08:00")

    def test_invalid_and_ambiguous_clock_times_raise(self):
        for diary in [self.diary(lights="25:00"), self.diary(wake=time(23, 42)),
                      self.diary("2026-10-25", time(2, 30), time(7))]:
            with self.assertRaisesRegex(ValueError, "Diary Excel row 2"):
                self.select(diary)

    def test_naive_recording_bounds_rejected(self):
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            select_sleep_interval(self.diary(), "2026-09-14", "2026-09-15")


if __name__ == "__main__":
    unittest.main()
