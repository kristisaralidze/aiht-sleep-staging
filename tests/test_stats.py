import numpy as np
import pandas as pd
import pytest

from src import stats


def _per_subject(base, other):
    subjs = [f"s{i}" for i in range(len(base))]
    return pd.concat([
        pd.DataFrame({"run": "base", "subject": subjs, "kappa": base}),
        pd.DataFrame({"run": "other", "subject": subjs, "kappa": other})])


def test_counts_subjects_improved_worse_and_tied():
    row = stats.paired_comparison(_per_subject([0.1, 0.5, 0.3, 0.2], [0.2, 0.4, 0.3, 0.4]),
                                  "base", "other")
    assert (row["n_improved"], row["n_worse"], row["n_tied"]) == (2, 1, 1)
    assert row["n_subjects"] == 4


def test_differences_are_model_minus_baseline():
    row = stats.paired_comparison(_per_subject([0.1, 0.2, 0.3], [0.2, 0.3, 0.4]), "base", "other")
    assert row["mean_diff"] == pytest.approx(0.1)
    assert row["median_diff"] == pytest.approx(0.1)
    assert row["mean_baseline"] == pytest.approx(0.2)
    assert row["mean_model"] == pytest.approx(0.3)


def test_p_value_matches_exact_signed_rank_when_all_subjects_improve():
    base = np.linspace(0.1, 0.5, 8)
    other = base + np.linspace(0.01, 0.08, 8)
    row = stats.paired_comparison(_per_subject(base, other), "base", "other")
    assert row["p_value"] == pytest.approx(2 / 2 ** 8)


def test_pairs_are_matched_by_subject_not_by_row_order():
    frame = _per_subject([0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5])
    shuffled = frame.sample(frac=1.0, random_state=0)
    row = stats.paired_comparison(shuffled, "base", "other")
    assert row["n_improved"] == 4


def test_rejects_runs_with_different_subjects():
    frame = _per_subject([0.1, 0.2], [0.2, 0.3]).iloc[:3]
    with pytest.raises(ValueError, match="same subjects"):
        stats.paired_comparison(frame, "base", "other")
