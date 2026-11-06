"""LOSO behaviour and the leakage guarantees."""
from dataclasses import replace

import numpy as np
import pytest

from src import loso, models
from tests.conftest import make_subject


class SpyModel:
    """Records every subject seen by fit and predict."""

    log = []

    def __init__(self, params, seed):
        self.params = params

    def fit(self, subjs):
        SpyModel.log.append(("fit", tuple(s.subject_id for s in subjs)))
        return self

    def predict_proba(self, subj):
        SpyModel.log.append(("predict", subj.subject_id))
        return np.tile([0.1, 0.8, 0.1], (subj.labels.size, 1))

    def predict(self, subj):
        return self.predict_proba(subj).argmax(axis=1)


def spy_spec(grid=({"a": 1}, {"a": 2})):
    return loso.ModelSpec(name="spy", build=SpyModel, grid=list(grid))


@pytest.fixture(autouse=True)
def clear_log():
    SpyModel.log = []


def test_inner_folds_partition_subjects_and_never_overlap(subjs):
    ids = [s.subject_id for s in subjs]
    folds = loso.inner_folds(subjs, n_splits=3, seed=0)
    assert len(folds) == 3
    held_out = []
    for train, val in folds:
        train_ids = {s.subject_id for s in train}
        val_ids = {s.subject_id for s in val}
        assert not train_ids & val_ids
        assert train_ids | val_ids == set(ids)
        held_out.extend(val_ids)
    assert sorted(held_out) == sorted(ids)


def test_inner_folds_cap_splits_at_subject_count(subjs):
    assert len(loso.inner_folds(subjs[:3], n_splits=5, seed=0)) == 3


def test_test_subject_never_reaches_tuning_or_fit(subjs):
    test, train = subjs[0], subjs[1:]
    loso.run_fold(spy_spec(), train, test, seed=0, n_inner=3)
    fits = [ids for kind, ids in SpyModel.log if kind == "fit"]
    assert len(fits) == 2 * 3 + 1
    assert all(test.subject_id not in ids for ids in fits)


def test_test_subject_is_predicted_only_after_the_final_fit(subjs):
    test, train = subjs[0], subjs[1:]
    loso.run_fold(spy_spec(), train, test, seed=0, n_inner=3)
    kinds = [entry[0] for entry in SpyModel.log]
    final_fit = len(kinds) - 1 - kinds[::-1].index("fit")
    predicted_at = [i for i, entry in enumerate(SpyModel.log)
                    if entry == ("predict", test.subject_id)]
    assert predicted_at
    assert min(predicted_at) > final_fit


def test_run_fold_rejects_test_subject_in_training_set(subjs):
    with pytest.raises(ValueError, match="s0"):
        loso.run_fold(spy_spec(), subjs, subjs[0], seed=0, n_inner=3)


def test_run_loso_scores_every_subject_exactly_once(subjs):
    result = loso.run_loso(spy_spec(({"a": 1},)), subjs, seed=0, n_inner=3, n_jobs=1)
    assert sorted(fold.subject_id for fold in result.folds) == sorted(
        s.subject_id for s in subjs)


def test_single_candidate_skips_tuning(subjs):
    loso.run_fold(spy_spec(({"a": 1},)), subjs[1:], subjs[0], seed=0, n_inner=3)
    assert len([entry for entry in SpyModel.log if entry[0] == "fit"]) == 1


def test_tuning_picks_the_better_candidate(subjs):
    class Picky(SpyModel):
        def predict_proba(self, subj):
            onehot = np.eye(3)[subj.labels] if self.params["a"] == 2 else None
            return onehot if onehot is not None else np.tile([1.0, 0, 0], (subj.labels.size, 1))

    spec = loso.ModelSpec(name="p", build=Picky, grid=[{"a": 1}, {"a": 2}])
    tuned = loso.tune(spec, subjs, fixed={}, seed=0, n_inner=3)
    assert tuned.params == {"a": 2}
    assert tuned.score == pytest.approx(1.0)


def test_fold_params_are_merged_per_test_subject(subjs):
    seen = []

    class Remember(SpyModel):
        def __init__(self, params, seed):
            super().__init__(params, seed)
            seen.append(params)

    spec = loso.ModelSpec(name="r", build=Remember, grid=[{"k": 1}],
                          fold_params={"s0": {"c": 9}})
    loso.run_fold(spec, subjs[1:], subjs[0], seed=0, n_inner=3)
    assert seen[-1] == {"c": 9, "k": 1}


def test_run_loso_records_time_and_params(subjs):
    result = loso.run_loso(spy_spec(({"a": 1},)), subjs, seed=0, n_inner=3, n_jobs=1)
    assert result.wall_seconds >= 0
    assert result.fold_seconds_total >= 0
    assert all(fold.params == {"a": 1} for fold in result.folds)


def test_best_inner_score_averages_folds(subjs):
    fold = loso.FoldResult("s", np.zeros(1), np.zeros(1, dtype=int), np.zeros(1, dtype=int),
                           np.zeros((1, 3)), {}, 0.4, 1.0)
    other = replace(fold, inner_score=0.6)
    assert loso.LosoResult("x", [fold, other], 1.0, 2).mean_inner_score == pytest.approx(0.5)


def test_scrambling_held_out_labels_cannot_change_tuning_or_predictions():
    train = [make_subject(f"t{i}", seed=i) for i in range(4)]
    test = make_subject("held", seed=9)
    spec = loso.ModelSpec(name="m", build=models.build_logistic, grid=[{"C": 0.01}, {"C": 1.0}])
    first = loso.run_fold(spec, train, test, seed=0, n_inner=2)
    scrambled = replace(test, labels=np.roll(test.labels, 7))
    second = loso.run_fold(spec, train, scrambled, seed=0, n_inner=2)
    assert first.params == second.params
    np.testing.assert_array_equal(first.proba, second.proba)


def test_fold_seed_does_not_depend_on_which_subjects_are_present(subjs):
    seeds = []

    class Seeds(SpyModel):
        def __init__(self, params, seed):
            super().__init__(params, seed)
            seeds.append(seed)

    spec = loso.ModelSpec(name="s", build=Seeds, grid=[{"a": 1}])
    loso.run_loso(spec, subjs, seed=0, n_inner=2, n_jobs=1)
    all_present = dict(zip((s.subject_id for s in subjs), seeds))
    seeds.clear()
    loso.run_loso(spec, subjs[2:], seed=0, n_inner=2, n_jobs=1)
    one_excluded = dict(zip((s.subject_id for s in subjs[2:]), seeds))
    assert all(one_excluded[key] == all_present[key] for key in one_excluded)
    assert len(set(all_present.values())) == len(subjs)


def test_fold_seed_stays_in_valid_range():
    assert 0 <= loso.fold_seed(2 ** 31 - 1, "s5") < loso.MAX_SEED


def test_tuning_fails_clearly_when_no_candidate_scores():
    class Nothing(SpyModel):
        def predict(self, subj):
            return np.zeros(subj.labels.size, dtype=int)

    empty = [replace(make_subject("a", seed=0), labels=np.full(120, 5)),
             replace(make_subject("b", seed=1), labels=np.full(120, 5))]
    spec = loso.ModelSpec(name="n", build=Nothing, grid=[{"a": 1}, {"a": 2}])
    with pytest.raises(ValueError, match="No valid candidate"):
        loso.tune(spec, empty, fixed={}, seed=0, n_inner=2)
