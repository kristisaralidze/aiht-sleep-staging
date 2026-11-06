"""Small synthetic subjects shared by the LOSO, context and leakage tests."""
import numpy as np
import pytest

from src.dataset import SubjectData

NAMES = ("signal", "noise", "clock")
GROUPS = ("motion", "heart_rate", "time")


def make_subject(subject_id, n=120, seed=0, drop=()):
    """A night where `signal` separates the classes, with long same-class runs."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(rng.choice(3, size=n // 10 + 1, p=[0.2, 0.5, 0.3]), 10)[:n]
    labels[:3] = (0, 1, 2)
    signal = labels + rng.normal(scale=0.4, size=n)
    features = np.column_stack([signal, rng.normal(size=n), np.linspace(-1, 1, n)])
    times = np.arange(n) * 30.0
    keep = np.setdiff1d(np.arange(n), np.array(drop, dtype=int))
    return SubjectData(subject_id=subject_id, features=features[keep], labels=labels[keep],
                       epoch_times=times[keep], feature_names=NAMES, feature_groups=GROUPS,
                       epoch_s=30)


@pytest.fixture
def subjs():
    return [make_subject(f"s{i}", seed=i) for i in range(6)]
