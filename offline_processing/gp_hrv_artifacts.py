"""GP_pipeline's interval-based Kubios artifact classifier.

The two functions below are copied without algorithm changes from the user's
GP_pipeline/wrist/heart_rate_variability/kubios.py. They classify interval values
in seconds; no peak-position correction, insertion, or deletion is performed.
See docs/HRV_PIPELINE.md for provenance and THIRD_PARTY_NOTICES.md for the
NeuroKit MIT license. Call through gp_hrv, which checks input continuity.
"""

import numpy as np
import pandas as pd


def _find_artifacts(
    env_diff,
    c1=0.13,
    c2=0.17,
    alpha=5.2,
    window_width=91,
    medfilt_order=11,
    sampling_rate=1000,
):
    # Compute period series (make sure it has same numer of elements as peaks);
    # peaks are in samples, convert to seconds.
    # rr = np.ediff1d(peaks, to_begin=0) / sampling_rate
    # For subsequent analysis it is important that the first element has
    # a value in a realistic range (e.g., for median filtering).
    # rr[0] = np.mean(rr[1:])
    rr = env_diff
    # Artifact identification #################################################
    ###########################################################################

    # Compute dRRs: time series of differences of consecutive periods (dRRs).
    drrs = np.ediff1d(rr, to_begin=0)
    drrs[0] = np.mean(drrs[1:])
    # Normalize by threshold.
    th1 = _compute_threshold(drrs, alpha, window_width)
    # ignore division by 0 warning
    old_setting = np.seterr(divide="ignore", invalid="ignore")
    drrs = drrs.astype(float)
    th1 = th1.astype(float)
    drrs /= th1
    # return old setting
    np.seterr(**old_setting)

    # Cast dRRs to subspace s12.
    # Pad drrs with one element.
    padding = 2
    drrs_pad = np.pad(drrs, padding, "reflect")

    s12 = np.zeros(drrs.size)
    for d in np.arange(padding, padding + drrs.size):
        if drrs_pad[d] > 0:
            s12[d - padding] = np.max([drrs_pad[d - 1], drrs_pad[d + 1]])
        elif drrs_pad[d] < 0:
            s12[d - padding] = np.min([drrs_pad[d - 1], drrs_pad[d + 1]])
    # Cast dRRs to subspace s22.
    s22 = np.zeros(drrs.size)
    for d in np.arange(padding, padding + drrs.size):
        if drrs_pad[d] >= 0:
            s22[d - padding] = np.min([drrs_pad[d + 1], drrs_pad[d + 2]])
        elif drrs_pad[d] < 0:
            s22[d - padding] = np.max([drrs_pad[d + 1], drrs_pad[d + 2]])
    # Compute mRRs: time series of deviation of RRs from median.
    df = pd.DataFrame({"signal": rr})
    medrr = df.rolling(medfilt_order, center=True, min_periods=1).median().signal.values
    mrrs = rr - medrr
    mrrs[mrrs < 0] = mrrs[mrrs < 0] * 2
    # Normalize by threshold.
    th2 = _compute_threshold(mrrs, alpha, window_width)
    old_setting = np.seterr(divide="ignore", invalid="ignore")
    mrrs /= th2
    np.seterr(**old_setting)

    # Artifact classification #################################################
    ###########################################################################

    # Artifact classes.
    extra_idcs = []
    missed_idcs = []
    ectopic_idcs = []
    longshort_idcs = []

    i = 0
    while i < rr.size - 2:  # The flow control is implemented based on Figure 1
        if np.abs(drrs[i]) <= 1:  # Figure 1
            i += 1
            continue
        eq1 = np.logical_and(
            drrs[i] > 1, s12[i] < (-c1 * drrs[i] - c2)
        )  # pylint: disable=E1111
        eq2 = np.logical_and(
            drrs[i] < -1, s12[i] > (-c1 * drrs[i] + c2)
        )  # pylint: disable=E1111

        if np.any([eq1, eq2]):
            # If any of the two equations is true.
            ectopic_idcs.append(i)
            i += 1
            continue
        # If none of the two equations is true.
        if ~np.any([np.abs(drrs[i]) > 1, np.abs(mrrs[i]) > 3]):  # Figure 1
            i += 1
            continue
        longshort_candidates = [i]
        # Check if the following beat also needs to be evaluated.
        if np.abs(drrs[i + 1]) < np.abs(drrs[i + 2]):
            longshort_candidates.append(i + 1)
        for j in longshort_candidates:
            # Long beat.
            eq3 = np.logical_and(drrs[j] > 1, s22[j] < -1)  # pylint: disable=E1111
            # Long or short.
            eq4 = np.abs(mrrs[j]) > 3  # Figure 1
            # Short beat.
            eq5 = np.logical_and(drrs[j] < -1, s22[j] > 1)  # pylint: disable=E1111

            if ~np.any([eq3, eq4, eq5]):
                # If none of the three equations is true: normal beat.
                i += 1
                continue
            # If any of the three equations is true: check for missing or extra
            # peaks.

            # Missing.
            eq6 = np.abs(rr[j] / 2 - medrr[j]) < th2[j]  # Figure 1
            # Extra.
            eq7 = np.abs(rr[j] + rr[j + 1] - medrr[j]) < th2[j]  # Figure 1

            # Check if extra.
            if np.all([eq5, eq7]):
                extra_idcs.append(j)
                i += 1
                continue
            # Check if missing.
            if np.all([eq3, eq6]):
                missed_idcs.append(j)
                i += 1
                continue
            # If neither classified as extra or missing, classify as "long or
            # short".
            longshort_idcs.append(j)
            i += 1
    # Prepare output
    artifacts = {
        "ectopic": ectopic_idcs,
        "missed": missed_idcs,
        "extra": extra_idcs,
        "longshort": longshort_idcs,
    }

    subspaces = {
        "rr": rr,
        "drrs": drrs,
        "mrrs": mrrs,
        "s12": s12,
        "s22": s22,
        "c1": c1,
        "c2": c2,
    }

    return artifacts, subspaces


def _compute_threshold(signal, alpha, window_width):
    df = pd.DataFrame({"signal": np.abs(signal)})
    q1 = (
        df.rolling(window_width, center=True, min_periods=1)
        .quantile(0.25)
        .signal.values
    )
    q3 = (
        df.rolling(window_width, center=True, min_periods=1)
        .quantile(0.75)
        .signal.values
    )
    th = alpha * ((q3 - q1) / 2)

    return th
