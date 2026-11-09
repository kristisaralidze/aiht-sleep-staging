from dataclasses import replace
from functools import partial

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src import gru, loso
from tests.conftest import make_subject

PARAMS = {"hidden": 8, "layers": 1, "dropout": 0.0, "input_dropout": 0.0, "lr": 0.01,
          "weight_decay": 0.0, "max_epochs": 30, "patience": 10, "batch_size": 2,
          "val_fraction": 0.25}


def _fit(subjs, **changes):
    return gru.build_gru({**PARAMS, **changes}, 0).fit(subjs)


def test_segments_split_at_breaks():
    subj = make_subject("s", n=10, drop=(4, 5))
    scaler = StandardScaler().fit(subj.features)
    assert [len(y) for _, y in gru.segment_tensors(subj, scaler)] == [4, 4]


def test_predictions_have_one_probability_row_per_epoch(subjs):
    model = _fit(subjs[1:])
    proba = model.predict_proba(subjs[0])
    assert proba.shape == (subjs[0].labels.size, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_gru_learns_the_synthetic_task(subjs):
    model = _fit(subjs[1:], max_epochs=60)
    assert (model.predict(subjs[0]) == subjs[0].labels).mean() > 0.8


def test_same_seed_gives_same_predictions(subjs):
    first = _fit(subjs[1:]).predict_proba(subjs[0])
    second = _fit(subjs[1:]).predict_proba(subjs[0])
    np.testing.assert_array_equal(first, second)


def test_segments_after_a_break_cannot_influence_earlier_ones(subjs):
    model = _fit(subjs[1:])
    test = make_subject("t", n=60, seed=7, drop=(30, 31))
    changed = test.features.copy()
    changed[30:] += 5.0
    before = model.predict_proba(test)
    after = model.predict_proba(replace(test, features=changed))
    np.testing.assert_allclose(after[:30], before[:30], atol=1e-6)
    assert not np.allclose(after[30:], before[30:])


def test_padding_does_not_change_predictions(subjs):
    model = _fit(subjs[1:])
    night = make_subject("t", n=80, seed=8, drop=(20, 21))   # segments of 20 and 58 epochs
    together = model.predict_proba(night)
    first = replace(night, features=night.features[:20], labels=night.labels[:20],
                    epoch_times=night.epoch_times[:20])
    np.testing.assert_allclose(model.predict_proba(first), together[:20], atol=1e-5)


def test_stop_epoch_is_logged(subjs):
    model = _fit(subjs[1:])
    assert 1 <= model.learned_params["stop_epoch"] <= PARAMS["max_epochs"]


def test_rejects_training_data_missing_a_class(subjs):
    only_two = [replace(s, labels=np.minimum(s.labels, 1)) for s in subjs]
    with pytest.raises(ValueError, match="class"):
        _fit(only_two)


def test_test_subject_never_enters_training_validation_or_class_weights(subjs, monkeypatch):
    fits, splits, weight_labels = [], [], []
    real_split, real_weights = gru.split_for_early_stopping, gru.balanced_weights

    def spy_split(fit_subjs, fraction, seed):
        fit_part, val_part = real_split(fit_subjs, fraction, seed)
        splits.append((fit_part, val_part))
        return fit_part, val_part

    def spy_weights(y):
        weight_labels.append(np.asarray(y).copy())
        return real_weights(y)

    class Spy(gru.GruModel):
        def fit(self, fit_subjs):
            fits.append([s.subject_id for s in fit_subjs])
            return super().fit(fit_subjs)

    monkeypatch.setattr(gru, "split_for_early_stopping", spy_split)
    monkeypatch.setattr(gru, "balanced_weights", spy_weights)
    spec = loso.ModelSpec(name="g", build=Spy, grid=[{**PARAMS, "max_epochs": 2},
                                                     {**PARAMS, "max_epochs": 3}])
    train, test = subjs[1:], subjs[0]
    loso.run_fold(spec, train, test, seed=0, n_inner=2)
    assert len(fits) == 2 * 2 + 1
    assert all(test.subject_id not in ids for ids in fits)
    for (fit_part, val_part), y in zip(splits, weight_labels):
        assert test.subject_id not in {s.subject_id for s in fit_part + val_part}
        np.testing.assert_array_equal(y, np.concatenate([s.labels for s in fit_part]))


def test_scrambled_test_labels_do_not_change_tuning_or_predictions(subjs):
    spec = loso.ModelSpec(name="g", build=gru.build_gru,
                          grid=[{**PARAMS, "hidden": 4}, {**PARAMS, "hidden": 8}])
    train, test = subjs[1:], subjs[0]
    first = loso.run_fold(spec, train, test, seed=1, n_inner=2)
    second = loso.run_fold(spec, train, replace(test, labels=test.labels[::-1].copy()),
                           seed=1, n_inner=2)
    assert first.params == second.params
    np.testing.assert_array_equal(first.proba, second.proba)


def _search(candidates):
    return gru.build_gru_search(candidates, {}, 0)


def test_search_uses_one_fixed_split_of_training_subjects(subjs, monkeypatch):
    splits = []
    real_split = gru.split_for_early_stopping

    def spy_split(fit_subjs, fraction, seed):
        parts = real_split(fit_subjs, fraction, seed)
        splits.append(parts)
        return parts

    monkeypatch.setattr(gru, "split_for_early_stopping", spy_split)
    _search([PARAMS, {**PARAMS, "hidden": 4}]).fit(subjs[1:])
    assert len(splits) == 1
    fit_part, val_part = splits[0]
    assert len(val_part) == 1 and len(fit_part) == 4


def test_search_keeps_the_candidate_with_the_best_validation_score(subjs):
    model = _search([{**PARAMS, "max_epochs": 1, "hidden": 2}, {**PARAMS, "max_epochs": 40}])
    model.fit(subjs[1:])
    learned = model.learned_params
    assert learned["hidden"] == 8
    assert 0 <= learned["val_macro_f1"] <= 1
    assert 1 <= learned["stop_epoch"] <= 40


def test_search_never_sees_the_test_subject(subjs, monkeypatch):
    seen, weight_labels = [], []
    real_fit_split, real_weights = gru.GruModel.fit_split, gru.balanced_weights

    def spy_fit_split(self, fit_part, val_part):
        seen.append({s.subject_id for s in fit_part} | {s.subject_id for s in val_part})
        weight_labels.append(np.concatenate([s.labels for s in fit_part]))
        return real_fit_split(self, fit_part, val_part)

    def spy_weights(y):
        np.testing.assert_array_equal(y, weight_labels[-1])
        return real_weights(y)

    monkeypatch.setattr(gru.GruModel, "fit_split", spy_fit_split)
    monkeypatch.setattr(gru, "balanced_weights", spy_weights)
    spec = loso.ModelSpec(name="g", build=partial(gru.build_gru_search,
                                                  [PARAMS, {**PARAMS, "hidden": 4}]), grid=[{}])
    train, test = subjs[1:], subjs[0]
    fold = loso.run_fold(spec, train, test, seed=0, n_inner=2)
    assert len(seen) == 2
    assert all(test.subject_id not in ids for ids in seen)
    assert {"hidden", "dropout", "stop_epoch"} <= set(fold.params)


def test_search_ignores_held_out_labels(subjs):
    spec = loso.ModelSpec(name="g", build=partial(gru.build_gru_search,
                                                  [PARAMS, {**PARAMS, "hidden": 4}]), grid=[{}])
    train, test = subjs[1:], subjs[0]
    first = loso.run_fold(spec, train, test, seed=1, n_inner=2)
    second = loso.run_fold(spec, train, replace(test, labels=test.labels[::-1].copy()),
                           seed=1, n_inner=2)
    assert first.params == second.params
    np.testing.assert_array_equal(first.proba, second.proba)
