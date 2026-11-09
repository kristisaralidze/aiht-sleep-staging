"""Bidirectional GRU over whole nights, split into segments at sequence breaks.
GruSearch tunes on one fixed 24/6 split of the training subjects.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from src.dataset import SubjectData, segment_bounds, stack
from src.metrics import CLASSES, macro_f1
from src.models import balanced_weights, check_all_classes, split_for_early_stopping

PAD_LABEL = -100
Segment = Tuple[torch.Tensor, torch.Tensor]


class GruNet(nn.Module):
    def __init__(self, n_inputs: int, hidden: int, layers: int, dropout: float,
                 input_dropout: float) -> None:
        super().__init__()
        self.input_dropout = nn.Dropout(input_dropout)
        self.gru = nn.GRU(n_inputs, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.output_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, len(CLASSES))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(self.input_dropout(x), lengths, batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return self.head(self.output_dropout(out))


def segment_tensors(subj: SubjectData, scaler: StandardScaler) -> List[Segment]:
    """Scaled features and labels of each run of time-adjacent epochs."""
    X = scaler.transform(subj.features)
    return [(torch.tensor(X[start:stop], dtype=torch.float32),
             torch.tensor(subj.labels[start:stop], dtype=torch.long))
            for start, stop in segment_bounds(subj)]


def pad_batch(segments: Sequence[Segment]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad segments to a batch; padded labels are PAD_LABEL."""
    lengths = torch.tensor([len(y) for _, y in segments], dtype=torch.long)
    x = pad_sequence([x for x, _ in segments], batch_first=True)
    y = pad_sequence([y for _, y in segments], batch_first=True, padding_value=PAD_LABEL)
    return x, y, lengths


class GruModel:
    """Whole-night BiGRU with class-weighted loss, stopped early on held-out subjects."""

    def __init__(self, params: Dict[str, Any], seed: int) -> None:
        self.params = params
        self.seed = seed
        self.scaler = StandardScaler()
        self.network = nn.Module()
        self.stop_epoch = 0

    def _segments(self, subjs: Sequence[SubjectData]) -> List[Segment]:
        return [segment for subj in subjs for segment in segment_tensors(subj, self.scaler)]

    def _train_one_epoch(self, segments: List[Segment], loss_fn: nn.Module,
                         optimiser: torch.optim.Optimizer, generator: torch.Generator) -> None:
        self.network.train()
        order = torch.randperm(len(segments), generator=generator).tolist()
        size = self.params["batch_size"]
        for first in range(0, len(order), size):
            x, y, lengths = pad_batch([segments[i] for i in order[first:first + size]])
            optimiser.zero_grad()
            logits = self.network(x, lengths)
            loss_fn(logits.reshape(-1, len(CLASSES)), y.reshape(-1)).backward()
            optimiser.step()

    def _loss(self, segments: List[Segment], loss_fn: nn.Module) -> float:
        self.network.eval()
        x, y, lengths = pad_batch(segments)
        with torch.no_grad():
            logits = self.network(x, lengths)
        return loss_fn(logits.reshape(-1, len(CLASSES)), y.reshape(-1)).item()

    def fit(self, subjs: Sequence[SubjectData]) -> "GruModel":
        fit_part, val_part = split_for_early_stopping(subjs, self.params["val_fraction"],
                                                      self.seed)
        return self.fit_split(fit_part, val_part)

    def fit_split(self, fit_part: Sequence[SubjectData],
                  val_part: Sequence[SubjectData]) -> "GruModel":
        """Train on `fit_part` and stop early on `val_part`."""
        params = self.params
        X, y = stack(fit_part)
        check_all_classes(y)
        self.scaler.fit(X)
        train_segments, val_segments = self._segments(fit_part), self._segments(val_part)

        torch.manual_seed(self.seed)
        generator = torch.Generator().manual_seed(self.seed)
        self.network = GruNet(X.shape[1], params["hidden"], params["layers"], params["dropout"],
                              params["input_dropout"])
        weights = torch.tensor(balanced_weights(y), dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=PAD_LABEL)
        optimiser = torch.optim.Adam(self.network.parameters(), lr=params["lr"],
                                     weight_decay=params["weight_decay"])

        best_loss, best_state, stale = float("inf"), None, 0
        for epoch in range(1, params["max_epochs"] + 1):
            self._train_one_epoch(train_segments, loss_fn, optimiser, generator)
            val_loss = self._loss(val_segments, loss_fn)
            if val_loss < best_loss:
                best_loss, stale, self.stop_epoch = val_loss, 0, epoch
                best_state = {k: v.clone() for k, v in self.network.state_dict().items()}
            else:
                stale += 1
                if stale >= params["patience"]:
                    break
        if best_state is None:
            raise ValueError("GRU validation loss was never finite")
        self.network.load_state_dict(best_state)
        return self

    def predict_proba(self, subj: SubjectData) -> np.ndarray:
        segments = segment_tensors(subj, self.scaler)
        x, _, lengths = pad_batch(segments)
        self.network.eval()
        with torch.no_grad():
            proba = torch.softmax(self.network(x, lengths).double(), dim=2).numpy()
        return np.concatenate([proba[i, :length] for i, length in enumerate(lengths.tolist())])

    def predict(self, subj: SubjectData) -> np.ndarray:
        return self.predict_proba(subj).argmax(axis=1)

    @property
    def learned_params(self) -> Dict[str, int]:
        return {"stop_epoch": self.stop_epoch}


def build_gru(params: Dict[str, Any], seed: int) -> GruModel:
    return GruModel(params, seed)


class GruSearch:
    """Pick GRU settings on one fixed split of the training subjects."""

    def __init__(self, candidates: Sequence[Dict[str, Any]], seed: int) -> None:
        self.candidates = list(candidates)
        self.seed = seed
        self.model = GruModel({}, seed)
        self.chosen = {}

    def fit(self, subjs: Sequence[SubjectData]) -> "GruSearch":
        fraction = self.candidates[0]["val_fraction"]
        fit_part, val_part = split_for_early_stopping(subjs, fraction, self.seed)
        y_val = np.concatenate([subj.labels for subj in val_part])
        best_score = -np.inf
        for params in self.candidates:
            model = GruModel(params, self.seed).fit_split(fit_part, val_part)
            score = macro_f1(y_val, np.concatenate([model.predict(subj) for subj in val_part]))
            if score > best_score:
                best_score, self.model = score, model
                self.chosen = {"hidden": params["hidden"], "dropout": params["dropout"],
                               "val_macro_f1": score}
        return self

    def predict_proba(self, subj: SubjectData) -> np.ndarray:
        return self.model.predict_proba(subj)

    def predict(self, subj: SubjectData) -> np.ndarray:
        return self.model.predict(subj)

    @property
    def learned_params(self) -> Dict[str, Any]:
        return {**self.chosen, "stop_epoch": self.model.stop_epoch}


def build_gru_search(candidates: Sequence[Dict[str, Any]], params: Dict[str, Any],
                     seed: int) -> GruSearch:
    """Build a search; `params` comes from the LOSO spec and is unused."""
    return GruSearch(candidates, seed)
