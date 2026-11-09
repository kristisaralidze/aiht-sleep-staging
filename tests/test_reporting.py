import json
from dataclasses import replace

import numpy as np
import pandas as pd

from src import loso, reporting


def _result(name="run"):
    folds = []
    for index, subj in enumerate(("a", "b")):
        y_true = np.array([0, 1, 2, 1])
        y_pred = np.array([0, 1, 1, 1]) if index else y_true
        folds.append(loso.FoldResult(subj, np.arange(4) * 30.0, y_true, y_pred,
                                     np.eye(3)[y_pred], {"C": 1.0}, 0.5, 2.0))
    return loso.LosoResult(name, folds, wall_seconds=3.0, n_jobs=2)


def test_predictions_written_per_subject_with_probabilities(tmp_path):
    reporting.save_predictions(_result(), tmp_path)
    frame = pd.read_csv(tmp_path / "run_a.csv")
    assert list(frame.columns) == ["subject", "epoch_time", "y_true", "y_pred", "p_wake", "p_nrem",
                                   "p_rem"]
    assert len(frame) == 4


def test_metrics_row_contains_pooled_and_subject_statistics():
    row, per_subject = reporting.evaluate(_result())
    assert row["run"] == "run"
    assert 0 < row["kappa_pooled"] < 1
    assert "kappa_subj_mean" in row and "kappa_subj_sd" in row
    assert list(per_subject["subject"]) == ["a", "b"]


def test_evaluate_can_restrict_to_a_subject_subset():
    row, per_subject = reporting.evaluate(_result(), only_subjects=("a",))
    assert row["accuracy_pooled"] == 1.0
    assert list(per_subject["subject"]) == ["a"]


def test_inner_score_is_averaged_over_the_selected_subjects_only():
    base = _result()
    folds = [replace(base.folds[0], inner_score=0.2), replace(base.folds[1], inner_score=0.8)]
    row, _ = reporting.evaluate(replace(base, folds=folds), only_subjects=("a",))
    assert row["mean_inner_macro_f1"] == 0.2


def test_upsert_replaces_matching_rows_and_keeps_the_others(tmp_path):
    path = tmp_path / "table.csv"
    pd.DataFrame({"run": ["a", "b"], "score": [1, 2]}).to_csv(path, index=False)
    new = pd.DataFrame({"run": ["b"], "score": [9]})
    merged = reporting.upsert_rows(path, new, keys=["run"])
    assert merged.set_index("run")["score"].to_dict() == {"a": 1, "b": 9}
    assert pd.read_csv(path).equals(merged)


def test_upsert_matches_on_all_keys(tmp_path):
    path = tmp_path / "table.csv"
    old = pd.DataFrame({"excluded": ["x", "y"], "model": ["m", "m"], "score": [1, 2]})
    old.to_csv(path, index=False)
    new = pd.DataFrame({"excluded": ["x"], "model": ["m"], "score": [9]})
    merged = reporting.upsert_rows(path, new, keys=["excluded", "model"])
    assert sorted(merged["score"]) == [2, 9]


def test_upsert_creates_a_missing_table(tmp_path):
    path = tmp_path / "table.csv"
    new = pd.DataFrame({"run": ["a"], "score": [1]})
    assert len(reporting.upsert_rows(path, new, keys=["run"])) == 1
    assert path.exists()


def test_confusion_figure_written_as_png(tmp_path):
    reporting.save_confusion_figure(_result(), tmp_path)
    assert (tmp_path / "confusion_run.png").stat().st_size > 0


def test_timing_log_replaces_rerun_of_the_same_run(tmp_path):
    path = tmp_path / "timing.csv"
    reporting.log_timing(_result(), path, script="02")
    reporting.log_timing(_result(), path, script="02")
    reporting.log_timing(_result("other"), path, script="03")
    frame = pd.read_csv(path)
    assert sorted(frame["run"]) == ["other", "run"]
    assert frame.loc[frame["run"] == "run", "fold_seconds_total"].iloc[0] == 4.0


def test_fold_log_round_trips_params(tmp_path):
    reporting.save_fold_log(_result(), tmp_path)
    assert reporting.load_fold_params("run", tmp_path) == {"a": {"C": 1.0}, "b": {"C": 1.0}}
    assert json.loads((tmp_path / "folds_run.json").read_text())["mean_inner_score"] == 0.5
