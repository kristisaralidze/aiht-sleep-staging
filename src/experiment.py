"""Glue shared by the experiment scripts: paths, specs, and running one model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import ParameterGrid

from src import loso, models, reporting
from src.dataset import SubjectData, load_subjects


@dataclass(frozen=True)
class Experiment:
    config: Dict[str, Any]
    paths: Dict[str, Path]
    script: str

    def subjs(self, feature_set: str, exclude: Sequence[str] = ()) -> List[SubjectData]:
        return load_subjects(self.paths["processed_dir"],
                             self.config["feature_sets"][feature_set],
                             self.config["epoch_s"], exclude)

    def run(self, spec: loso.ModelSpec, subjs: Sequence[SubjectData]
            ) -> Tuple[loso.LosoResult, Dict[str, Any], pd.DataFrame]:
        """Run LOSO for one spec and save all of its outputs."""
        settings = self.config["loso"]
        result = loso.run_loso(spec, subjs, self.config["seed"], settings["n_inner"],
                               settings["n_jobs"])
        row, per_subject = reporting.evaluate(result)
        reporting.save_predictions(result, self.paths["predictions_dir"])
        reporting.save_confusion_figure(result, self.paths["figures_dir"])
        reporting.save_fold_log(result, self.paths["logs_dir"])
        reporting.log_timing(result, self.paths["logs_dir"] / "timing.csv", self.script)
        print(f"{spec.name:32s} folds={len(result.folds):2d} "
              f"macro_f1={row['macro_f1_pooled']:.3f} kappa={row['kappa_subj_mean']:.3f} "
              f"wall={result.wall_seconds:.0f}s", flush=True)
        return result, row, per_subject


def resolve_paths(root: Path, config: Dict[str, Any]) -> Dict[str, Path]:
    return {name: root / value for name, value in config["paths"].items()}


def model_grid(model_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand a model's grid, merging its fixed parameters into each candidate."""
    return [{**model_config["fixed"], **candidate}
            for candidate in ParameterGrid(model_config["grid"])]


def baseline_spec(name: str, model: str, config: Dict[str, Any]) -> loso.ModelSpec:
    return loso.ModelSpec(name=name, build=models.BUILDERS[model],
                          grid=model_grid(config["models"][model]))
