"""Aggregation checks independent of private recordings."""
import unittest

import numpy as np
import pandas as pd

from across_nights import summarize
from burst_hr_response import RELATIVE_SECONDS


class AcrossNightsTests(unittest.TestCase):
    def fixture(self):
        # Unequal event counts ensure an event-weighted curve would be wrong.
        index = pd.MultiIndex.from_tuples(
            [("a", 0), ("b", 0), ("b", 1), ("b", 2), ("b", 3), ("b", 4)],
            names=["night", "burst_id"])
        events = pd.DataFrame({"AUC": [1, 1, 1, 4, 8, 100],
                               "peak-to-peak": [1, 2, 3, 4, 5, 6],
                               "duration_s": [2] * 6,
                               "included": [True] * 5 + [False],
                               "hr_peak_increase_pct": [10, 20, 30, 40, 50, 90],
                               "hr_peak_latency_s": [1] * 6}, index=index)
        epochs = pd.DataFrame(np.array([10, 20, 30, 40, 50, 90])[:, None]
                              * np.ones((6, len(RELATIVE_SECONDS))),
                              index=index, columns=RELATIVE_SECONDS)
        nights = pd.DataFrame({"spt_s": [100, 200, 300]},
                              index=pd.Index(["a", "b", "empty"], name="night"))
        return events, epochs, nights

    def test_common_cutoffs_include_hr_excluded_bursts(self):
        events, epochs, nights = self.fixture()
        result = summarize(events, epochs, nights)
        np.testing.assert_allclose(result["cutoffs"], np.quantile(events.AUC, [1/3, 2/3]))
        self.assertEqual(result["events"].auc_tertile.iloc[:3].tolist(), ["Low"] * 3)
        self.assertEqual(len(result["burst_pairs"]), 5)

    def test_equal_night_curve_and_spt_denominator(self):
        result = summarize(*self.fixture())
        low = result["grand_curves"].query("group == 'Low'")
        np.testing.assert_allclose(low['mean'], (10 + (20 + 30) / 2) / 2)
        self.assertTrue((low['count'] == 2).all())
        all_rows = result["nightly"].query("group == 'All'").set_index("night")
        self.assertEqual(all_rows.loc['b', 'burst_spt_pct'], 5)
        self.assertEqual(all_rows.loc['empty', 'burst_spt_pct'], 0)
        self.assertTrue(np.isnan(all_rows.loc['empty', 'auc_mean_g_s']))
        paired = result["night_pairs"].query("night == 'b' and size_group == 'Small'").iloc[0]
        self.assertEqual(paired.hr_peak_increase_pct, 25)
        self.assertEqual(paired['peak-to-peak'], 2.5)

    def test_no_eligible_hr_and_constant_correlations(self):
        events, epochs, nights = self.fixture()
        events['included'] = False
        result = summarize(events, epochs, nights)
        self.assertTrue(result['grand_curves'].empty)
        self.assertTrue(result['correlations'].pearson_r.isna().all())
        self.assertEqual(result['nightly'].hr_n.sum(), 0)


if __name__ == '__main__':
    unittest.main()
