from dataclasses import replace

import numpy as np
import pytest

from src import dataset, loso, models
from tests.conftest import make_subject

MLP_PARAMS = {"hidden": (8,), "weight_decay": 0.0, "lr": 0.01, "max_epochs": 40, "patience": 10,
              "batch_size": 64, "val_fraction": 0.25}
LOGISTIC = {"C": 1.0}
FOREST = {"n_estimators": 20, "max_depth": 4, "min_samples_leaf": 2, "max_features": 1.0}


@pytest.mark.parametrize("build,params", [(models.build_logistic, LOGISTIC),
                                          (models.build_forest, FOREST),
                                          (models.build_mlp, MLP_PARAMS)])
def test_models_learn_and_return_three_probabilities(subjs, build, params):
    model = build(params, 0).fit(subjs[1:])
    proba = model.predict_proba(subjs[0])
    assert proba.shape == (subjs[0].labels.size, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (model.predict(subjs[0]) == subjs[0].labels).mean() > 0.8


@pytest.mark.parametrize("build,params", [(models.build_logistic, LOGISTIC),
                                          (models.build_forest, FOREST),
                                          (models.build_mlp, MLP_PARAMS)])
def test_same_seed_gives_same_predictions(subjs, build, params):
    first = build(params, 3).fit(subjs[1:]).predict_proba(subjs[0])
    second = build(params, 3).fit(subjs[1:]).predict_proba(subjs[0])
    np.testing.assert_array_equal(first, second)


def test_scaler_is_fit_on_training_subjects_only(subjs):
    train = subjs[1:]
    model = models.build_logistic(LOGISTIC, 0).fit(train)
    X, _ = dataset.stack(train)
    np.testing.assert_allclose(model.scaler_mean(), X.mean(axis=0))
    assert not np.allclose(model.scaler_mean(), dataset.stack(subjs)[0].mean(axis=0))


def test_mlp_scaler_uses_only_the_fit_part_of_training(subjs):
    train = subjs[1:]
    fit_part, _ = models.split_for_early_stopping(train, MLP_PARAMS["val_fraction"], 0)
    model = models.build_mlp(MLP_PARAMS, 0).fit(train)
    np.testing.assert_allclose(model.scaler_mean(), dataset.stack(fit_part)[0].mean(axis=0))
    assert not np.allclose(model.scaler_mean(), dataset.stack(train)[0].mean(axis=0))


def test_mlp_class_weights_use_only_the_fit_part_of_training(subjs, monkeypatch):
    train = subjs[1:]
    seen = []
    real = models.balanced_weights

    def spy(labels):
        seen.append(labels)
        return real(labels)

    monkeypatch.setattr(models, "balanced_weights", spy)
    models.build_mlp(MLP_PARAMS, 0).fit(train)
    fit_part, _ = models.split_for_early_stopping(train, MLP_PARAMS["val_fraction"], 0)
    np.testing.assert_array_equal(seen[0], dataset.stack(fit_part)[1])


def test_mlp_early_stopping_holds_out_whole_subjects(subjs):
    fit_part, val_part = models.split_for_early_stopping(subjs, 0.34, seed=0)
    assert not {s.subject_id for s in fit_part} & {s.subject_id for s in val_part}
    assert len(val_part) == 2
    assert len(fit_part) == 4


def test_balanced_weights_favour_rare_classes():
    labels = np.array([0] * 10 + [1] * 80 + [2] * 10)
    weights = models.balanced_weights(labels)
    assert weights[0] == weights[2] > weights[1]
    np.testing.assert_allclose((weights[labels]).sum(), labels.size)


def test_class_weights_raise_rare_class_recall():
    rng = np.random.default_rng(0)
    subjs = []
    for index in range(4):
        base = make_subject(f"s{index}", n=300, seed=index)
        rare = np.where(base.labels == 0, rng.random(base.labels.size) < 0.15, True)
        features = base.features.copy()
        features[:, 0] = base.labels * 0.5 + rng.normal(scale=0.8, size=base.labels.size)
        subjs.append(replace(base, features=features[rare], labels=base.labels[rare],
                             epoch_times=base.epoch_times[rare]))
    model = models.build_logistic(LOGISTIC, 0).fit(subjs[1:])
    predicted = model.predict(subjs[0])
    assert (predicted == 0).mean() > (subjs[0].labels == 0).mean()


def test_model_rejects_training_data_missing_a_class(subjs):
    only_two = [replace(s, labels=np.minimum(s.labels, 1)) for s in subjs]
    with pytest.raises(ValueError, match="class"):
        models.build_logistic(LOGISTIC, 0).fit(only_two)


@pytest.mark.parametrize("build,params", [(models.build_logistic, LOGISTIC),
                                          (models.build_forest, FOREST),
                                          (models.build_mlp, MLP_PARAMS)])
def test_scrambled_test_labels_do_not_change_predictions(subjs, build, params):
    spec = loso.ModelSpec(name="m", build=build, grid=[params])
    train, test = subjs[1:], subjs[0]
    first = loso.run_fold(spec, train, test, seed=1, n_inner=2)
    second = loso.run_fold(spec, train, replace(test, labels=test.labels[::-1].copy()),
                           seed=1, n_inner=2)
    np.testing.assert_array_equal(first.proba, second.proba)
