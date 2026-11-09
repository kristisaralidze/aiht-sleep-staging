"""Context baseline: give a per-epoch model the features of +/-k neighbouring epochs."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, Sequence

import numpy as np

from src.dataset import SubjectData, bounds_from_times, segment_bounds

TIME_GROUP = "time"


def _neighbour_indices(subj: SubjectData, k: int) -> np.ndarray:
    """Row indices of each epoch's neighbours, repeating the edge of its segment."""
    n_epochs = subj.labels.size
    first = np.empty(n_epochs, dtype=int)
    last = np.empty(n_epochs, dtype=int)
    for start, stop in segment_bounds(subj):
        first[start:stop] = start
        last[start:stop] = stop - 1
    wanted = np.arange(n_epochs)[:, None] + np.arange(-k, k + 1)[None, :]
    return np.clip(wanted, first[:, None], last[:, None])


def window_subject(subj: SubjectData, k: int) -> SubjectData:
    """Copy with signal features spanning +/-k epochs within each segment."""
    if k == 0:
        return subj
    indices = _neighbour_indices(subj, k)
    blocks, names, groups = [], [], []
    for column, (name, group) in enumerate(zip(subj.feature_names, subj.feature_groups)):
        if group == TIME_GROUP:
            continue
        blocks.append(subj.features[:, column][indices])
        names += [f"{name}_t{offset:+d}" for offset in range(-k, k + 1)]
        groups += [group] * (2 * k + 1)
    for column, (name, group) in enumerate(zip(subj.feature_names, subj.feature_groups)):
        if group == TIME_GROUP:
            blocks.append(subj.features[:, [column]])
            names.append(name)
            groups.append(group)
    return replace(subj, features=np.hstack(blocks), feature_names=tuple(names),
                   feature_groups=tuple(groups))


def interior_mask(epoch_times: np.ndarray, epoch_s: int, k: int) -> np.ndarray:
    """True for epochs whose +/-k window stays inside their segment."""
    mask = np.zeros(epoch_times.size, dtype=bool)
    for start, stop in bounds_from_times(epoch_times, epoch_s):
        mask[start + k:stop - k] = True
    return mask


class WindowedModel:
    """Wrap a per-epoch model so train and test get the same window."""

    def __init__(self, base: Any, k: int) -> None:
        self.base = base
        self.k = k

    def fit(self, subjs: Sequence[SubjectData]) -> "WindowedModel":
        self.base.fit([window_subject(subj, self.k) for subj in subjs])
        return self

    def predict_proba(self, subj: SubjectData) -> np.ndarray:
        return self.base.predict_proba(window_subject(subj, self.k))

    def predict(self, subj: SubjectData) -> np.ndarray:
        return self.base.predict(window_subject(subj, self.k))


def build_windowed(base_build: Callable[[Dict[str, Any], int], Any], params: Dict[str, Any],
                   seed: int) -> WindowedModel:
    """Build a windowed model; `k` is read from params, the rest go to the base model."""
    base_params = {key: value for key, value in params.items() if key != "k"}
    return WindowedModel(base_build(base_params, seed), params["k"])
