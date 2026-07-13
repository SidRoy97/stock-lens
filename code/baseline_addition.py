"""
Additions to oos_evaluation.py: per-year chance baselines for macro-F1.

Two baselines, computed per year from the TRUE label distribution only
(no dependence on model predictions):

1. majority_baseline  - always predict the most frequent true class that year.
   This is the floor: the minimum a "dumb" classifier could score.

2. stratified_random_baseline - predict by sampling from the true label
   distribution each year (i.e. random guessing that respects class balance).
   This is the real chance line to compare models against, replacing the
   flat 0.333 dashed line (which assumes uniform 3-class balance that your
   data doesn't have -- Neutral is ~53%, not ~33%).

Drop `compute_year_baselines` into oos_evaluation.py and call it inside
stage_6_oos() right before building `res`, then concat the baseline rows
into `res` so they show up in the same CSV and plot.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def compute_year_baselines(df, label_col="true_label", year_col="year",
                            n_bootstrap=200, seed=0):
    """Per-year majority-class and stratified-random macro-F1 baselines.

    Returns a DataFrame with columns: model, year, macro_f1
    (model in {"baseline_majority", "baseline_stratified_random"}),
    matching the schema of the `rows` list already built in stage_6_oos.

    The stratified-random baseline is averaged over n_bootstrap draws per
    year rather than computed once, since a single random draw is noisy --
    averaging gives you a stable expected-chance value to compare against.
    """
    rng = np.random.default_rng(seed)
    out_rows = []

    for yr, grp in df.groupby(year_col):
        y_true = grp[label_col].values
        classes, counts = np.unique(y_true, return_counts=True)
        probs = counts / counts.sum()

        # majority-class baseline: predict the modal class every time
        majority_class = classes[np.argmax(counts)]
        y_pred_majority = np.full_like(y_true, majority_class)
        maj_f1 = f1_score(y_true, y_pred_majority, average="macro",
                          zero_division=0)
        out_rows.append({"model": "baseline_majority", "year": int(yr),
                         "macro_f1": maj_f1})

        # stratified-random baseline: sample predictions ~ true class
        # distribution, averaged over multiple draws for stability
        boot_f1s = []
        for _ in range(n_bootstrap):
            y_pred_rand = rng.choice(classes, size=len(y_true), p=probs)
            boot_f1s.append(f1_score(y_true, y_pred_rand, average="macro",
                                     zero_division=0))
        out_rows.append({"model": "baseline_stratified_random",
                         "year": int(yr), "macro_f1": float(np.mean(boot_f1s))})

    return pd.DataFrame(out_rows)