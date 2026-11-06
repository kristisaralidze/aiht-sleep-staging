import numpy as np
import pytest

from src import raw_io


def _write_subject(raw_dir, subj, modalities=("motion", "heart_rate", "labels")):
    names = {
        "motion": ("motion", f"{subj}_acceleration.txt", "0.0 0.1 0.2 0.3\n0.5 0.1 0.2 0.3\n"),
        "heart_rate": ("heart_rate", f"{subj}_heartrate.txt", "0.0,60\n5.0,61\n"),
        "labels": ("labels", f"{subj}_labeled_sleep.txt", "0 0\n30 2\n"),
    }
    for key in modalities:
        folder, filename, text = names[key]
        (raw_dir / folder).mkdir(exist_ok=True)
        (raw_dir / folder / filename).write_text(text)


def test_clean_samples_sorts_by_time():
    samples = np.array([[2.0, 20.0], [0.0, 0.0], [1.0, 10.0]])
    np.testing.assert_array_equal(raw_io.clean_samples(samples)[:, 0], [0.0, 1.0, 2.0])


def test_clean_samples_keeps_one_row_per_timestamp():
    samples = np.array([[1.0, 10.0], [1.0, 99.0], [2.0, 20.0]])
    cleaned = raw_io.clean_samples(samples)
    assert cleaned.shape == (2, 2)
    assert cleaned[0, 1] == 10.0


def test_clean_samples_drops_non_finite_rows():
    samples = np.array([[0.0, np.nan], [1.0, 1.0], [np.inf, 2.0]])
    np.testing.assert_array_equal(raw_io.clean_samples(samples), [[1.0, 1.0]])


def test_clean_samples_does_not_mutate_input():
    samples = np.array([[2.0, 20.0], [0.0, 0.0]])
    original = samples.copy()
    raw_io.clean_samples(samples)
    np.testing.assert_array_equal(samples, original)


def test_list_subjects_returns_sorted_ids(tmp_path):
    _write_subject(tmp_path, "200")
    _write_subject(tmp_path, "100")
    assert raw_io.list_subjects(tmp_path) == ["100", "200"]


def test_list_subjects_raises_when_a_modality_is_missing(tmp_path):
    _write_subject(tmp_path, "100")
    _write_subject(tmp_path, "200", modalities=("motion", "labels"))
    (tmp_path / "heart_rate").mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="200"):
        raw_io.list_subjects(tmp_path)


def test_list_subjects_raises_on_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        raw_io.list_subjects(tmp_path)


def test_loaders_parse_each_file_format_including_first_row(tmp_path):
    _write_subject(tmp_path, "100")
    assert raw_io.load_motion(tmp_path, "100").shape == (2, 4)
    np.testing.assert_array_equal(raw_io.load_heart_rate(tmp_path, "100"), [[0, 60], [5, 61]])
    np.testing.assert_array_equal(raw_io.load_labels(tmp_path, "100"), [[0, 0], [30, 2]])
