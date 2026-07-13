"""
Additions to oos_evaluation.py: bootstrap confidence intervals for the
per-year macro-F1 of the actual models (not the baselines).

Resamples rows *within each year* with replacement, recomputes macro-F1
each time, and takes the 2.5th/97.5th percentiles as a 95% CI. This tells
you whether a dip below the baseline line is a real effect or could just
be sampling noise given how many rows that year has.

Usage: call `bootstrap_year_f1(y_true, y_pred, years, model_name)` once
per model, passing the same arrays you already compute macro_f1 from.
It returns a DataFrame with columns: model, year, macro_f1, ci_lo, ci_hi.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def bootstrap_year_f1(y_true, y_pred, years, model_name,
                      n_bootstrap=200, ci=0.95, seed=0):
    """Per-year macro-F1 with a bootstrap confidence interval.

    y_true, y_pred, years: 1-D arrays/Series of equal length, one entry
        per prediction (row-aligned).
    model_name: label to store in the output 'model' column, e.g.
        "random_forest" or "sequence_cnn1d" -- matches what's already
        used in the `rows` list in stage_6_oos.
    n_bootstrap: number of resamples per year. 200 is a reasonable
        default; increase if the CI looks unstable, at the cost of
        runtime (each resample refits nothing, just recomputes F1 on
        an array, so this is cheap even at a few thousand).
    ci: confidence level, e.g. 0.95 for a 95% interval.

    Returns a DataFrame: model, year, macro_f1, ci_lo, ci_hi
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    years = np.asarray(years)
    rng = np.random.default_rng(seed)
    alpha = (1 - ci) / 2

    out_rows = []
    for yr in sorted(set(years)):
        m = years == yr
        yt, yp = y_true[m], y_pred[m]
        n = len(yt)
        point = f1_score(yt, yp, average="macro", zero_division=0)

        boot_scores = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_scores[i] = f1_score(yt[idx], yp[idx], average="macro",
                                      zero_division=0)
        lo, hi = np.quantile(boot_scores, [alpha, 1 - alpha])

        out_rows.append({"model": model_name, "year": int(yr),
                         "macro_f1": point, "ci_lo": lo, "ci_hi": hi})

    return pd.DataFrame(out_rows)