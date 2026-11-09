"""Per-epoch classifiers. Every model scales features and weights classes in `fit`.
The MLP is a small PyTorch network because MLPClassifier has no class weights.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch import nn

from src.dataset import SubjectData, stack
from src.metrics import CLASSES

Params = Dict[str, Any]


def check_all_classes(y: np.ndarray) -> None:
    missing = sorted(set(CLASSES) - set(np.unique(y).tolist()))
    if missing:
        raise ValueError(f"Training data is missing class codes {missing}")


def balanced_weights(y: np.ndarray) -> np.ndarray:
    """Weight per class, inversely proportional to its frequency."""
    return compute_class_weight("balanced", classes=np.array(CLASSES), y=y)


class TabularModel:
    """A scikit-learn classifier behind a scaler fit on training subjects only."""

    def __init__(self, classifier: Any) -> None:
        self.scaler = StandardScaler()
        self.classifier = classifier

    def fit(self, subjs: Sequence[SubjectData]) -> "TabularModel":
        X, y = stack(subjs)
        check_all_classes(y)
        self.classifier.fit(self.scaler.fit_transform(X), y)
        return self

    def predict_proba(self, subj: SubjectData) -> np.ndarray:
        return self.classifier.predict_proba(self.scaler.transform(subj.features))

    def predict(self, subj: SubjectData) -> np.ndarray:
        return self.predict_proba(subj).argmax(axis=1)

    def scaler_mean(self) -> np.ndarray:
        return self.scaler.mean_


def split_for_early_stopping(subjs: Sequence[SubjectData], val_fraction: float,
                             seed: int) -> Tuple[List[SubjectData], List[SubjectData]]:
    """Hold out whole training subjects to decide when to stop."""
    order = np.random.default_rng(seed).permutation(len(subjs))
    n_val = max(1, int(round(len(subjs) * val_fraction)))
    return ([subjs[i] for i in order[n_val:]], [subjs[i] for i in order[:n_val]])


def _network(n_inputs: int, hidden: Sequence[int]) -> nn.Sequential:
    layers = []
    width = n_inputs
    for size in hidden:
        layers += [nn.Linear(width, size), nn.ReLU()]
        width = size
    return nn.Sequential(*layers, nn.Linear(width, len(CLASSES)))


class TorchMlp:
    """MLP with class-weighted loss, stopped early on held-out training subjects."""

    def __init__(self, params: Params, seed: int) -> None:
        self.params = params
        self.seed = seed
        self.scaler = StandardScaler()
        self.network = nn.Sequential()

    def _tensors(self, subjs: Sequence[SubjectData]) -> Tuple[torch.Tensor, torch.Tensor]:
        X, y = stack(subjs)
        scaled = self.scaler.transform(X)
        return torch.tensor(scaled, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def _train_one_epoch(self, inputs: torch.Tensor, targets: torch.Tensor, loss_fn: nn.Module,
                         optimiser: torch.optim.Optimizer, generator: torch.Generator) -> None:
        order = torch.randperm(inputs.shape[0], generator=generator)
        for batch in order.split(self.params["batch_size"]):
            optimiser.zero_grad()
            loss_fn(self.network(inputs[batch]), targets[batch]).backward()
            optimiser.step()

    def fit(self, subjs: Sequence[SubjectData]) -> "TorchMlp":
        params = self.params
        fit_part, val_part = split_for_early_stopping(subjs, params["val_fraction"], self.seed)
        X, y = stack(fit_part)
        check_all_classes(y)
        self.scaler.fit(X)
        inputs, targets = self._tensors(fit_part)
        val_inputs, val_targets = self._tensors(val_part)

        torch.manual_seed(self.seed)
        generator = torch.Generator().manual_seed(self.seed)
        self.network = _network(inputs.shape[1], tuple(params["hidden"]))
        weights = torch.tensor(balanced_weights(y), dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        optimiser = torch.optim.AdamW(self.network.parameters(), lr=params["lr"],
                                      weight_decay=params["weight_decay"])

        best_loss, best_state, stale = float("inf"), None, 0
        for _ in range(params["max_epochs"]):
            self.network.train()
            self._train_one_epoch(inputs, targets, loss_fn, optimiser, generator)
            self.network.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.network(val_inputs), val_targets).item()
            if val_loss < best_loss:
                best_loss, stale = val_loss, 0
                best_state = {k: v.clone() for k, v in self.network.state_dict().items()}
            else:
                stale += 1
                if stale >= params["patience"]:
                    break
        if best_state is None:
            raise ValueError("MLP validation loss was never finite")
        self.network.load_state_dict(best_state)
        return self

    def predict_proba(self, subj: SubjectData) -> np.ndarray:
        scaled = self.scaler.transform(subj.features)
        self.network.eval()
        with torch.no_grad():
            logits = self.network(torch.tensor(scaled, dtype=torch.float32))
        return torch.softmax(logits.double(), dim=1).numpy()

    def predict(self, subj: SubjectData) -> np.ndarray:
        return self.predict_proba(subj).argmax(axis=1)

    def scaler_mean(self) -> np.ndarray:
        return self.scaler.mean_


def build_logistic(params: Params, seed: int) -> TabularModel:
    return TabularModel(LogisticRegression(C=params["C"], class_weight="balanced", max_iter=1000,
                                           random_state=seed))


def build_forest(params: Params, seed: int) -> TabularModel:
    return TabularModel(RandomForestClassifier(class_weight="balanced", random_state=seed,
                                               n_jobs=1, **params))


def build_mlp(params: Params, seed: int) -> TorchMlp:
    return TorchMlp(params, seed)


BUILDERS = {"logistic": build_logistic, "forest": build_forest, "mlp": build_mlp}
