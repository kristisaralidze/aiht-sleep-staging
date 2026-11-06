from pathlib import Path

import numpy as np
import pytest

from src import epochs, preprocess, raw_io
from src.utils import load_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "preprocess.yaml"


def _write_night(raw_dir, subj="100", n_epochs=60, gap_epochs=(20, 24)):
    rng = np.random.default_rng(2)
    for folder in ("motion", "heart_rate", "labels"):
        (raw_dir / folder).mkdir(exist_ok=True)
    seconds = n_epochs * 30
    motion_t = np.round(np.arange(-120, seconds + 120, 0.02), 2)
    keep = (motion_t < gap_epochs[0] * 30) | (motion_t >= gap_epochs[1] * 30)
    motion = np.column_stack([motion_t, rng.normal(size=(motion_t.size, 3)) * 0.05])[keep]
    # Reverse a block so the pipeline has to sort it.
    motion[1000:1010] = motion[1000:1010][::-1]
    hr_t = np.arange(-120, seconds + 120, 5.0)
    heart = np.column_stack([hr_t, 60 + rng.normal(size=hr_t.size)])
    stages = np.tile([0, 1, 2, 3, 4, 5], n_epochs // 6)
    stages[3] = -1
    labels = np.column_stack([np.arange(n_epochs) * 30, stages])
    np.savetxt(raw_dir / "motion" / f"{subj}_acceleration.txt", motion, fmt="%.4f")
    np.savetxt(raw_dir / "heart_rate" / f"{subj}_heartrate.txt", heart, fmt="%.3f", delimiter=",")
    np.savetxt(raw_dir / "labels" / f"{subj}_labeled_sleep.txt", labels, fmt="%d")


@pytest.fixture
def record(tmp_path):
    _write_night(tmp_path)
    return preprocess.process_subject(tmp_path, "100", load_config(CONFIG))


def test_process_subject_drops_unscored_and_gap_epochs(record):
    assert record.features.shape[0] == 60 - 1 - 4
    assert record.labels.shape == record.epoch_times.shape
    assert not np.isin(record.epoch_times, np.arange(20, 24) * 30.0).any()
    assert 90.0 not in record.epoch_times


def test_process_subject_features_are_finite(record):
    assert np.isfinite(record.features).all()


def test_process_subject_rejects_non_finite_features(tmp_path, monkeypatch):
    _write_night(tmp_path)

    def broken(*args, **kwargs):
        return np.array([[np.nan]]), ["x"], ["motion"]

    monkeypatch.setattr(preprocess, "build_features", broken)
    with pytest.raises(ValueError, match="non-finite"):
        preprocess.process_subject(tmp_path, "100", load_config(CONFIG))


def test_labels_stay_aligned_with_epoch_times(record):
    index = int(np.where(record.epoch_times == 150.0)[0][0])
    assert record.labels[index] == epochs.REM


def test_save_and_load_round_trip(record, tmp_path):
    path = preprocess.save_record(record, tmp_path)
    assert path.name == "100.npz"
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["features"], record.features)
        np.testing.assert_array_equal(saved["labels"], record.labels)
        np.testing.assert_array_equal(saved["epoch_times"], record.epoch_times)
        assert list(saved["feature_names"]) == list(record.feature_names)
        assert list(saved["feature_groups"]) == list(record.feature_groups)
        assert str(saved["subject_id"]) == "100"


def test_summary_row_counts_every_labelled_epoch(record):
    row = preprocess.summary_row(record)
    assert row["n_labelled"] == 60
    assert row["n_unscored"] + row["n_no_signal"] + row["n_epochs"] == 60
    assert row["n_wake"] + row["n_nrem"] + row["n_rem"] == row["n_epochs"]
    assert row["n_sequence_breaks"] == 2


def test_summary_table_appends_pooled_row(record):
    table = preprocess.summary_table([preprocess.summary_row(record)] * 2)
    pooled = table.iloc[-1]
    assert pooled["subject"] == "ALL"
    assert pooled["n_epochs"] == 2 * table.iloc[0]["n_epochs"]
    assert pooled[["pct_wake", "pct_nrem", "pct_rem"]].sum() == pytest.approx(100.0, abs=0.1)


def test_process_subject_is_deterministic(tmp_path):
    _write_night(tmp_path)
    config = load_config(CONFIG)
    first = preprocess.process_subject(tmp_path, "100", config)
    second = preprocess.process_subject(tmp_path, "100", config)
    np.testing.assert_array_equal(first.features, second.features)


def test_raw_io_module_is_used_for_sorting(tmp_path):
    _write_night(tmp_path)
    assert np.all(np.diff(raw_io.load_motion(tmp_path, "100")[:, 0]) > 0)
