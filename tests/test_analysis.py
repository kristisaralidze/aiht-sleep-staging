import numpy as np
import pandas as pd
import pytest

from src.analysis import disagreement_by_class, gap_fraction, transition_mask


def test_transition_mask_marks_two_epochs_each_side():
    labels = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    times = np.arange(8) * 30.0
    assert transition_mask(labels, times, 30).tolist() == [0, 0, 1, 1, 1, 1, 0, 0]


def test_transition_mask_ignores_changes_across_breaks():
    labels = np.array([1, 1, 1, 2, 2, 2])
    times = np.array([0, 30, 60, 600, 630, 660], dtype=float)
    assert not transition_mask(labels, times, 30).any()


def test_transition_mask_clips_at_segment_edges():
    labels = np.array([0, 1, 1, 1])
    times = np.arange(4) * 30.0
    assert transition_mask(labels, times, 30).tolist() == [1, 1, 1, 0]


def test_gap_fraction_counts_windows_touching_a_gap():
    samples = np.concatenate([np.arange(0, 300), np.arange(400, 1000)]).astype(float)
    epochs = np.arange(0, 990, 30.0)
    fraction = gap_fraction(epochs, samples, 30, half_width_s=0, max_gap_s=5)
    touching = ((epochs + 30 > 299) & (epochs < 400)).mean()
    assert fraction == pytest.approx(touching)


def test_gap_fraction_is_zero_without_gaps():
    samples = np.arange(0, 1000, 0.5)
    assert gap_fraction(np.arange(0, 900, 30.0), samples, 30, 285, 1.0) == 0.0


def test_disagreement_by_class_splits_who_is_right():
    times = np.arange(4) * 30.0
    a = pd.DataFrame({"epoch_time": times, "y_true": [0, 0, 2, 2], "y_pred": [0, 1, 2, 2]})
    b = pd.DataFrame({"epoch_time": times, "y_true": [0, 0, 2, 2], "y_pred": [1, 1, 1, 2]})
    table = disagreement_by_class(a, b, "a", "b").set_index("true_class")
    assert table.loc[0, "n_disagree"] == 1 and table.loc[0, "a_right"] == 1
    assert table.loc[2, "n_disagree"] == 1 and table.loc[2, "a_right"] == 1
    assert table["both_wrong"].sum() == 0


def test_disagreement_requires_aligned_epochs():
    a = pd.DataFrame({"epoch_time": [0.0, 30.0], "y_true": [0, 1], "y_pred": [0, 1]})
    b = a.assign(epoch_time=[0.0, 60.0])
    with pytest.raises(ValueError):
        disagreement_by_class(a, b, "a", "b")
