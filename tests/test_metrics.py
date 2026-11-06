import numpy as np
import pandas as pd
import pytest

from src import metrics


def test_perfect_predictions_score_one():
    y = np.array([0, 1, 2, 1])
    scores = metrics.score(y, y)
    assert scores["accuracy"] == 1.0
    assert scores["kappa"] == 1.0
    assert scores["macro_f1"] == 1.0


def test_per_class_f1_matches_hand_calculation():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    scores = metrics.score(y_true, y_pred)
    assert scores["f1_wake"] == pytest.approx(0.5)
    assert scores["f1_nrem"] == pytest.approx(0.8)
    assert scores["f1_rem"] == pytest.approx(2 / 3)


def test_rem_f1_is_undefined_for_subject_without_rem():
    scores = metrics.score(np.array([0, 1, 1]), np.array([0, 1, 2]))
    assert np.isnan(scores["f1_rem"])


def test_macro_f1_averages_only_classes_present():
    scores = metrics.score(np.array([0, 1, 1]), np.array([0, 1, 1]))
    assert scores["macro_f1"] == 1.0


def _per_subject():
    return pd.DataFrame({
        "subject": ["a", "b", "c"], "n_epochs": [10, 10, 10], "accuracy": [0.5, 0.7, 0.9],
        "kappa": [0.2, 0.4, 0.6], "macro_f1": [0.5, 0.5, 0.5], "f1_wake": [0.5, 0.5, 0.5],
        "f1_nrem": [0.5, 0.5, 0.5], "f1_rem": [0.4, np.nan, 0.6]})


def test_summary_reports_subject_mean_and_sd():
    summary = metrics.summarise_subjects(_per_subject())
    assert summary["kappa_subj_mean"] == pytest.approx(0.4)
    assert summary["kappa_subj_sd"] == pytest.approx(0.2)


def test_summary_averages_rem_f1_only_over_subjects_with_rem():
    summary = metrics.summarise_subjects(_per_subject())
    assert summary["f1_rem_subj_mean"] == pytest.approx(0.5)
    assert summary["n_subjects_rem"] == 2
    assert summary["n_subjects"] == 3


def test_confusion_counts_and_row_normalisation():
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 2])
    counts, rates = metrics.confusion(y_true, y_pred)
    np.testing.assert_array_equal(counts, [[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    np.testing.assert_allclose(rates[0], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(rates.sum(axis=1), 1.0)
