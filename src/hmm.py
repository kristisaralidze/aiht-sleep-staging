"""Context baseline: HMM smoothing of a per-epoch model's probabilities.

Hidden states are the three classes. Transitions come from the true label
sequences of training subjects, counting only time-adjacent epochs. Decoding
restarts at every sequence break, so no state is carried across missing data:
a night starts from the first-epoch distribution, and a segment after a break
from the overall class frequencies.

Base models use balanced class weights, so their output approximates
p(class | x) under a uniform prior, which is proportional to the likelihood
p(x | class). It is used directly as the emission, and the real class prior enters
through the transitions. Neighbouring probabilities are correlated, so an emission
weight (tuned on inner folds, never on the left-out subject) sets how much the
per-epoch evidence counts against the transitions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp

from src import metrics
from src.dataset import SubjectData, segment_bounds
from src.loso import inner_folds

N_STATES = len(metrics.CLASSES)
MIN_PROBABILITY = 1e-12


@dataclass(frozen=True)
class HmmParams:
    start: np.ndarray
    resume: np.ndarray
    transition: np.ndarray


def _normalise(counts: np.ndarray) -> np.ndarray:
    return counts / np.maximum(counts.sum(axis=-1, keepdims=True), MIN_PROBABILITY)


def fit_transitions(subjs: Sequence[SubjectData], pseudocount: float) -> HmmParams:
    """Estimate start, resume and transition probabilities from true labels."""
    transitions = np.full((N_STATES, N_STATES), pseudocount, dtype=float)
    start = np.full(N_STATES, pseudocount, dtype=float)
    frequency = np.full(N_STATES, pseudocount, dtype=float)
    for subj in subjs:
        start[subj.labels[0]] += 1
        np.add.at(frequency, subj.labels, 1)
        for first, stop in segment_bounds(subj):
            run = subj.labels[first:stop]
            np.add.at(transitions, (run[:-1], run[1:]), 1)
    return HmmParams(start=_normalise(start), resume=_normalise(frequency),
                     transition=_normalise(transitions))


def viterbi(log_start: np.ndarray, log_transition: np.ndarray,
            log_emission: np.ndarray) -> np.ndarray:
    """Return the most likely state path."""
    n_steps = log_emission.shape[0]
    best = log_start + log_emission[0]
    came_from = np.zeros((n_steps, N_STATES), dtype=int)
    for step in range(1, n_steps):
        candidates = best[:, None] + log_transition
        came_from[step] = candidates.argmax(axis=0)
        best = candidates.max(axis=0) + log_emission[step]
    path = np.empty(n_steps, dtype=int)
    path[-1] = best.argmax()
    for step in range(n_steps - 1, 0, -1):
        path[step - 1] = came_from[step, path[step]]
    return path


def posterior(log_start: np.ndarray, log_transition: np.ndarray,
              log_emission: np.ndarray) -> np.ndarray:
    """Return p(state at t | whole segment) by forward-backward."""
    n_steps = log_emission.shape[0]
    forward = np.empty((n_steps, N_STATES))
    backward = np.zeros((n_steps, N_STATES))
    forward[0] = log_start + log_emission[0]
    for step in range(1, n_steps):
        forward[step] = (logsumexp(forward[step - 1][:, None] + log_transition, axis=0)
                         + log_emission[step])
    for step in range(n_steps - 2, -1, -1):
        backward[step] = logsumexp(
            log_transition + (log_emission[step + 1] + backward[step + 1])[None, :], axis=1)
    joint = forward + backward
    return np.exp(joint - logsumexp(joint, axis=1, keepdims=True))


def smooth(params: HmmParams, proba: np.ndarray, subj: SubjectData,
           emission_weight: float) -> Tuple[np.ndarray, np.ndarray]:
    """Decode one night segment by segment; return the Viterbi path and posteriors."""
    if proba.shape != (subj.labels.size, N_STATES):
        raise ValueError(f"Expected probabilities of shape {(subj.labels.size, N_STATES)}, "
                         f"got {proba.shape}")
    log_emission = emission_weight * np.log(np.maximum(proba, MIN_PROBABILITY))
    log_transition = np.log(np.maximum(params.transition, MIN_PROBABILITY))
    path = np.empty(proba.shape[0], dtype=int)
    smoothed = np.empty_like(proba, dtype=float)
    for first, stop in segment_bounds(subj):
        initial = params.start if first == 0 else params.resume
        log_start = np.log(np.maximum(initial, MIN_PROBABILITY))
        path[first:stop] = viterbi(log_start, log_transition, log_emission[first:stop])
        smoothed[first:stop] = posterior(log_start, log_transition, log_emission[first:stop])
    return path, smoothed


class HmmSmoothed:
    """A base model followed by HMM smoothing, all estimated from `fit` subjects."""

    def __init__(self, base_build: Callable[[Dict[str, Any], int], Any], params: Dict[str, Any],
                 seed: int, emission_weights: Sequence[float], pseudocount: float,
                 n_inner: int) -> None:
        self.base_build = base_build
        self.base_params = params
        self.seed = seed
        self.emission_weights = tuple(emission_weights)
        self.pseudocount = pseudocount
        self.n_inner = n_inner
        self.emission_weight = self.emission_weights[0]
        self.base = None
        self.params = HmmParams(np.empty(0), np.empty(0), np.empty((0, 0)))

    def _tune_emission_weight(self, subjs: Sequence[SubjectData]) -> float:
        """Score each weight on inner-fold predictions from models that never saw them."""
        truth = []
        decoded = {weight: [] for weight in self.emission_weights}
        for train, val in inner_folds(subjs, self.n_inner, self.seed):
            base = self.base_build(self.base_params, self.seed).fit(train)
            hmm_params = fit_transitions(train, self.pseudocount)
            for subj in val:
                proba = base.predict_proba(subj)
                truth.append(subj.labels)
                for weight in self.emission_weights:
                    decoded[weight].append(smooth(hmm_params, proba, subj, weight)[0])
        y = np.concatenate(truth)
        scores = [metrics.macro_f1(y, np.concatenate(decoded[weight]))
                  for weight in self.emission_weights]
        return self.emission_weights[int(np.argmax(scores))]

    def fit(self, subjs: Sequence[SubjectData]) -> "HmmSmoothed":
        if len(self.emission_weights) > 1:
            self.emission_weight = self._tune_emission_weight(subjs)
        self.base = self.base_build(self.base_params, self.seed).fit(subjs)
        self.params = fit_transitions(subjs, self.pseudocount)
        return self

    def _smooth(self, subj: SubjectData) -> Tuple[np.ndarray, np.ndarray]:
        return smooth(self.params, self.base.predict_proba(subj), subj,
                      self.emission_weight)

    def predict(self, subj: SubjectData) -> np.ndarray:
        return self._smooth(subj)[0]

    def predict_proba(self, subj: SubjectData) -> np.ndarray:
        return self._smooth(subj)[1]

    @property
    def learned_params(self) -> Dict[str, float]:
        return {"emission_weight": self.emission_weight}
