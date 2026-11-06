"""Leave-one-subject-out evaluation with tuning nested inside each training fold.

Models see whole subjects, so the same loop serves tabular models and the GRU:

    model = spec.build(params, seed).fit(training_subjects)
    model.predict(subject); model.predict_proba(subject)

The left-out subject is passed to `run_fold` separately and is only ever given
to the final model's predict calls. Scaling, class weights, early stopping and
HMM estimation all live inside `fit`, so they cannot see it either.
"""
from __future__ import annotations

import time
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Protocol, Sequence, Tuple

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from src import metrics
from src.dataset import SubjectData

Params = Dict[str, Any]
MAX_SEED = 2 ** 31


class SubjectModel(Protocol):
    def fit(self, subjs: Sequence[SubjectData]) -> "SubjectModel": ...

    def predict_proba(self, subj: SubjectData) -> np.ndarray: ...

    def predict(self, subj: SubjectData) -> np.ndarray: ...


@dataclass(frozen=True)
class ModelSpec:
    """How to build a model, what to tune, and any per-fold fixed params."""

    name: str
    build: Callable[[Params, int], SubjectModel]
    grid: Sequence[Params]
    fold_params: Mapping[str, Params] = field(default_factory=dict)


@dataclass(frozen=True)
class TuneResult:
    params: Params
    score: float


@dataclass(frozen=True)
class FoldResult:
    subject_id: str
    epoch_times: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    proba: np.ndarray
    params: Params
    inner_score: float
    seconds: float


@dataclass(frozen=True)
class LosoResult:
    name: str
    folds: List[FoldResult]
    wall_seconds: float
    n_jobs: int

    @property
    def mean_inner_score(self) -> float:
        return float(np.mean([fold.inner_score for fold in self.folds]))

    @property
    def fold_seconds_total(self) -> float:
        return float(sum(fold.seconds for fold in self.folds))


def inner_folds(subjs: Sequence[SubjectData], n_splits: int,
                seed: int) -> List[Tuple[List[SubjectData], List[SubjectData]]]:
    """Split subjects, never epochs, into (train, validation) pairs."""
    splitter = KFold(n_splits=min(n_splits, len(subjs)), shuffle=True, random_state=seed)
    return [([subjs[i] for i in train], [subjs[i] for i in val])
            for train, val in splitter.split(np.arange(len(subjs)))]


def tune(spec: ModelSpec, subjs: Sequence[SubjectData], fixed: Params, seed: int,
         n_inner: int) -> TuneResult:
    """Pick the grid candidate with the best pooled inner macro F1 (first wins ties)."""
    folds = inner_folds(subjs, n_inner, seed)
    best = TuneResult(params={}, score=-np.inf)
    for candidate in spec.grid:
        params = {**fixed, **candidate}
        true_parts, pred_parts = [], []
        for train, val in folds:
            model = spec.build(params, seed).fit(train)
            for subj in val:
                true_parts.append(subj.labels)
                pred_parts.append(model.predict(subj))
        score = metrics.macro_f1(np.concatenate(true_parts), np.concatenate(pred_parts))
        if score > best.score:
            best = TuneResult(params=params, score=score)
    if not np.isfinite(best.score):
        raise ValueError(f"No valid candidate while tuning {spec.name}")
    return best


def run_fold(spec: ModelSpec, train: Sequence[SubjectData], test: SubjectData, seed: int,
             n_inner: int) -> FoldResult:
    """Tune on `train` only, refit on all of `train`, then predict `test`."""
    if test.subject_id in {subj.subject_id for subj in train}:
        raise ValueError(f"Test subject {test.subject_id} is in the training set")
    started = time.perf_counter()
    fixed = dict(spec.fold_params.get(test.subject_id, {}))
    if len(spec.grid) > 1:
        tuned = tune(spec, train, fixed, seed, n_inner)
    else:
        tuned = TuneResult(params={**fixed, **spec.grid[0]}, score=float("nan"))
    model = spec.build(tuned.params, seed).fit(train)
    y_pred = model.predict(test)
    proba = model.predict_proba(test)
    return FoldResult(
        subject_id=test.subject_id,
        epoch_times=test.epoch_times,
        y_true=test.labels,
        y_pred=y_pred,
        proba=proba,
        params={**tuned.params, **getattr(model, "learned_params", {})},
        inner_score=tuned.score,
        seconds=time.perf_counter() - started,
    )


def fold_seed(seed: int, subject_id: str) -> int:
    """Seed a fold from the subject id, so excluding others leaves it unchanged."""
    return (seed + zlib.crc32(subject_id.encode())) % MAX_SEED


def _worker(spec: ModelSpec, train: List[SubjectData], test: SubjectData, seed: int,
            n_inner: int) -> FoldResult:
    torch.set_num_threads(1)
    return run_fold(spec, train, test, seed, n_inner)


def run_loso(spec: ModelSpec, subjs: Sequence[SubjectData], seed: int, n_inner: int,
             n_jobs: int) -> LosoResult:
    """Run one fold per subject, seeded by `fold_seed`."""
    started = time.perf_counter()
    jobs = (delayed(_worker)(spec, [s for s in subjs if s is not test], test,
                             fold_seed(seed, test.subject_id), n_inner)
            for test in subjs)
    folds = Parallel(n_jobs=n_jobs)(jobs)
    return LosoResult(name=spec.name, folds=list(folds),
                      wall_seconds=time.perf_counter() - started, n_jobs=n_jobs)
