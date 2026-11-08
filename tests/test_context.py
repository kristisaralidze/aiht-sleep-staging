from dataclasses import replace

import numpy as np

from src import context, loso, models
from tests.conftest import make_subject

BASE = {"C": 1.0}


def _ramp(n=6, drop=()):
    subj = make_subject("s", n=n, drop=drop)
    features = np.column_stack([np.arange(subj.labels.size, dtype=float),
                                np.arange(subj.labels.size) * 10.0,
                                np.full(subj.labels.size, 7.0)])
    return replace(subj, features=features)


def test_k_zero_returns_the_original_features():
    subj = _ramp()
    np.testing.assert_array_equal(context.window_subject(subj, 0).features, subj.features)


def test_window_adds_neighbours_for_non_time_columns_only():
    windowed = context.window_subject(_ramp(), 1)
    assert windowed.features.shape == (6, 2 * 3 + 1)
    assert windowed.feature_names == ("signal_t-1", "signal_t+0", "signal_t+1", "noise_t-1",
                                      "noise_t+0", "noise_t+1", "clock")


def test_window_values_come_from_neighbouring_epochs():
    windowed = context.window_subject(_ramp(), 1)
    np.testing.assert_array_equal(windowed.features[2, :3], [1.0, 2.0, 3.0])


def test_window_repeats_edge_epoch_at_night_boundaries():
    windowed = context.window_subject(_ramp(), 2)
    np.testing.assert_array_equal(windowed.features[0, :5], [0.0, 0.0, 0.0, 1.0, 2.0])
    np.testing.assert_array_equal(windowed.features[5, :5], [3.0, 4.0, 5.0, 5.0, 5.0])


def test_window_never_crosses_a_sequence_break():
    subj = _ramp(n=8, drop=(3, 4))  # kept rows 0,1,2 | 3,4,5
    windowed = context.window_subject(subj, 1)
    np.testing.assert_array_equal(windowed.features[2, :3], [1.0, 2.0, 2.0])
    np.testing.assert_array_equal(windowed.features[3, :3], [3.0, 3.0, 4.0])


def test_windowing_does_not_mutate_the_subject():
    subj = _ramp()
    before = subj.features.copy()
    context.window_subject(subj, 2)
    np.testing.assert_array_equal(subj.features, before)


def test_window_model_applies_the_same_window_to_train_and_test(subjs):
    model = context.build_windowed(models.build_logistic, {**BASE, "k": 2}, 0).fit(subjs[1:])
    assert model.predict_proba(subjs[0]).shape == (subjs[0].labels.size, 3)


def test_windowed_model_ignores_held_out_labels(subjs):
    def build(params, seed):
        return context.build_windowed(models.build_logistic, params, seed)

    spec = loso.ModelSpec(name="w", build=build, grid=[{**BASE, "k": 0}, {**BASE, "k": 2}])
    train, test = subjs[1:], subjs[0]
    first = loso.run_fold(spec, train, test, seed=1, n_inner=2)
    second = loso.run_fold(spec, train, replace(test, labels=test.labels[::-1].copy()),
                           seed=1, n_inner=2)
    assert first.params == second.params
    np.testing.assert_array_equal(first.proba, second.proba)


def test_interior_mask_drops_epochs_near_night_edges():
    times = np.arange(10) * 30.0
    mask = context.interior_mask(times, 30, 2)
    np.testing.assert_array_equal(mask, [0, 0, 1, 1, 1, 1, 1, 1, 0, 0])


def test_interior_mask_drops_epochs_near_sequence_breaks():
    times = np.array([0, 30, 60, 90, 120, 900, 930, 960, 990, 1020]) * 1.0
    mask = context.interior_mask(times, 30, 2)
    np.testing.assert_array_equal(mask, [0, 0, 1, 0, 0, 0, 0, 1, 0, 0])


def test_interior_mask_is_empty_for_a_segment_shorter_than_the_window():
    assert not context.interior_mask(np.arange(5) * 30.0, 30, 3).any()


def test_interior_mask_keeps_everything_for_k_zero():
    assert context.interior_mask(np.array([0.0, 30.0, 300.0]), 30, 0).all()
