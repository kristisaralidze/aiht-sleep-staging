from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from src import hmm, loso, models
from tests.conftest import make_subject

BASE = {"C": 1.0}


def _subject(labels, times=None):
    labels = np.array(labels)
    times = np.arange(labels.size) * 30.0 if times is None else np.array(times, dtype=float)
    base = make_subject("s", n=labels.size + 10)
    return replace(base, labels=labels, epoch_times=times, features=np.zeros((labels.size, 3)))


def test_transitions_count_only_adjacent_epochs():
    subj = _subject([0, 0, 1, 2], times=[0, 30, 60, 300])
    params = hmm.fit_transitions([subj], pseudocount=0.0)
    np.testing.assert_allclose(params.transition[0], [0.5, 0.5, 0.0])
    assert params.transition[1, 2] == 0.0 or np.isnan(params.transition[1, 2])


def test_pseudocount_keeps_all_transitions_possible():
    params = hmm.fit_transitions([_subject([0, 0, 0, 1])], pseudocount=1.0)
    assert (params.transition > 0).all()
    np.testing.assert_allclose(params.transition.sum(axis=1), 1.0)


def test_start_uses_first_epochs_and_resume_uses_class_frequency():
    params = hmm.fit_transitions([_subject([0, 1, 1, 1]), _subject([0, 2, 1, 1])], pseudocount=0.0)
    np.testing.assert_allclose(params.start, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(params.resume, [2 / 8, 5 / 8, 1 / 8])


def _brute_force(log_start, log_transition, log_emission):
    best, best_path = -np.inf, None
    n = log_emission.shape[0]
    for path in product(range(3), repeat=n):
        value = log_start[path[0]] + log_emission[0, path[0]]
        for t in range(1, n):
            value += log_transition[path[t - 1], path[t]] + log_emission[t, path[t]]
        if value > best:
            best, best_path = value, path
    return np.array(best_path)


def test_viterbi_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(20):
        log_start = np.log(rng.dirichlet(np.ones(3)))
        log_transition = np.log(rng.dirichlet(np.ones(3), size=3))
        log_emission = np.log(rng.dirichlet(np.ones(3), size=6))
        np.testing.assert_array_equal(hmm.viterbi(log_start, log_transition, log_emission),
                                      _brute_force(log_start, log_transition, log_emission))


def test_posterior_matches_enumeration():
    rng = np.random.default_rng(1)
    log_start = np.log(rng.dirichlet(np.ones(3)))
    log_transition = np.log(rng.dirichlet(np.ones(3), size=3))
    log_emission = np.log(rng.dirichlet(np.ones(3), size=4))
    expected = np.zeros((4, 3))
    for path in product(range(3), repeat=4):
        value = log_start[path[0]] + log_emission[0, path[0]]
        for t in range(1, 4):
            value += log_transition[path[t - 1], path[t]] + log_emission[t, path[t]]
        for t in range(4):
            expected[t, path[t]] += np.exp(value)
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(hmm.posterior(log_start, log_transition, log_emission), expected)


def _sticky():
    transition = np.full((3, 3), 0.005)
    np.fill_diagonal(transition, 0.99)
    return hmm.HmmParams(start=np.full(3, 1 / 3), resume=np.full(3, 1 / 3), transition=transition)


def test_smoothing_removes_an_isolated_blip():
    proba = np.tile([0.1, 0.8, 0.1], (9, 1))
    proba[4] = [0.1, 0.35, 0.55]
    path, _ = hmm.smooth(_sticky(), proba, _subject([1] * 9), emission_weight=1.0)
    np.testing.assert_array_equal(path, np.ones(9))


def test_smoothing_does_not_carry_state_across_a_break():
    wake, rem = [0.9, 0.05, 0.05], [0.05, 0.05, 0.9]
    proba = np.array([wake] * 4 + [rem] * 4)
    times = [0, 30, 60, 90, 3000, 3030, 3060, 3090]
    path, _ = hmm.smooth(_sticky(), proba, _subject([0] * 8, times), emission_weight=1.0)
    np.testing.assert_array_equal(path, [0] * 4 + [2] * 4)


def test_low_emission_weight_smooths_more():
    proba = np.tile([0.1, 0.8, 0.1], (9, 1))
    proba[3:6] = [0.8, 0.1, 0.1]
    subj = _subject([1] * 9)
    strong, _ = hmm.smooth(_sticky(), proba, subj, emission_weight=4.0)
    weak, _ = hmm.smooth(_sticky(), proba, subj, emission_weight=0.25)
    assert (strong == 0).sum() == 3
    assert (weak == 0).sum() == 0


def _spec():
    def build(params, seed):
        return hmm.HmmSmoothed(models.build_logistic, params, seed, emission_weights=(0.5, 1.0),
                               pseudocount=1.0, n_inner=2)

    return loso.ModelSpec(name="h", build=build, grid=[BASE])


def test_hmm_params_ignore_the_held_out_subject(subjs):
    train = subjs[1:]
    model = hmm.HmmSmoothed(models.build_logistic, BASE, 0, emission_weights=(1.0,),
                            pseudocount=1.0, n_inner=2).fit(train)
    expected = hmm.fit_transitions(train, pseudocount=1.0)
    np.testing.assert_array_equal(model.params.transition, expected.transition)
    with_test = hmm.fit_transitions(subjs, pseudocount=1.0)
    assert not np.array_equal(model.params.transition, with_test.transition)


def test_hmm_model_ignores_held_out_labels(subjs):
    train, test = subjs[1:], subjs[0]
    first = loso.run_fold(_spec(), train, test, seed=1, n_inner=2)
    second = loso.run_fold(_spec(), train, replace(test, labels=test.labels[::-1].copy()),
                           seed=1, n_inner=2)
    np.testing.assert_array_equal(first.y_pred, second.y_pred)
    np.testing.assert_array_equal(first.proba, second.proba)


def test_hmm_tunes_emission_weight_without_the_held_out_subject(subjs):
    seen = []

    class Spy:
        def __init__(self, params, seed):
            self.inner = models.build_logistic(params, seed)

        def fit(self, fit_subjects):
            seen.append({s.subject_id for s in fit_subjects})
            self.inner.fit(fit_subjects)
            return self

        def predict_proba(self, subj):
            return self.inner.predict_proba(subj)

    model = hmm.HmmSmoothed(Spy, BASE, 0, emission_weights=(0.5, 1.0), pseudocount=1.0, n_inner=2)
    model.fit(subjs[1:])
    assert model.emission_weight in (0.5, 1.0)
    assert len(seen) == 3
    assert all("s0" not in ids for ids in seen)


def test_predict_proba_rows_sum_to_one(subjs):
    model = hmm.HmmSmoothed(models.build_logistic, BASE, 0, emission_weights=(1.0,),
                            pseudocount=1.0, n_inner=2).fit(subjs[1:])
    np.testing.assert_allclose(model.predict_proba(subjs[0]).sum(axis=1), 1.0)
    assert model.predict(subjs[0]).shape == subjs[0].labels.shape


def test_rejects_invalid_probability_shape():
    with pytest.raises(ValueError):
        hmm.smooth(_sticky(), np.ones((3, 2)), _subject([0, 1, 2]), emission_weight=1.0)
