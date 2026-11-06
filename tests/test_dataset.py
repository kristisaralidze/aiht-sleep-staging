import numpy as np
import pytest

from src import dataset
from tests.conftest import make_subject


def _save(path, subject_id, n=4):
    np.savez(path / f"{subject_id}.npz", subject_id=np.array(subject_id),
             features=np.arange(n * 3, dtype=float).reshape(n, 3), labels=np.zeros(n, dtype=int),
             epoch_times=np.arange(n) * 30.0, feature_names=np.array(["a", "b", "c"]),
             feature_groups=np.array(["motion", "heart_rate", "time"]))


def test_load_subjects_selects_feature_columns_in_requested_order(tmp_path):
    _save(tmp_path, "1")
    loaded = dataset.load_subjects(tmp_path, ("c", "a"), epoch_s=30)
    assert loaded[0].feature_names == ("c", "a")
    assert loaded[0].feature_groups == ("time", "motion")
    np.testing.assert_array_equal(loaded[0].features[0], [2.0, 0.0])


def test_load_subjects_excludes_requested_ids(tmp_path):
    _save(tmp_path, "1")
    _save(tmp_path, "2")
    loaded = dataset.load_subjects(tmp_path, ("a",), epoch_s=30, exclude=("1",))
    assert [s.subject_id for s in loaded] == ["2"]


def test_load_subjects_rejects_unknown_exclusion(tmp_path):
    _save(tmp_path, "1")
    with pytest.raises(ValueError, match="999"):
        dataset.load_subjects(tmp_path, ("a",), epoch_s=30, exclude=("999",))


def test_load_subjects_rejects_unknown_feature(tmp_path):
    _save(tmp_path, "1")
    with pytest.raises(ValueError, match="nope"):
        dataset.load_subjects(tmp_path, ("nope",), epoch_s=30)


def test_load_subjects_raises_on_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        dataset.load_subjects(tmp_path, ("a",), epoch_s=30)


def test_segment_bounds_split_at_breaks():
    subj = make_subject("s", n=10, drop=(4, 5))
    assert dataset.segment_bounds(subj) == [(0, 4), (4, 8)]


def test_segment_bounds_single_segment_without_breaks():
    assert dataset.segment_bounds(make_subject("s", n=10)) == [(0, 10)]


def test_stack_concatenates_subjects():
    first, second = make_subject("a", n=10), make_subject("b", n=20)
    X, y = dataset.stack([first, second])
    assert X.shape == (30, 3)
    assert y.shape == (30,)
