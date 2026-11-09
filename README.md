CMT paper ID: XXXX

# Sleep stage classification from Apple Watch motion and heart rate

AIHT 2026 course project, Team 21 (Aalto University).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/00_download_data.sh      # Walch et al. 2019, PhysioNet sleep-accel 1.0.0
```

## Run order

`bash scripts/run_all.sh` runs every step below from an empty `results/` folder (about 10 h on 8 cores). Steps can also be run one at a time, in this order; script 06 goes last because it reads the outputs of 03b, 05 and 07.

| Step | Command | Produces |
|---|---|---|
| 1 | `python scripts/01_preprocess.py` | `data/processed/*.npz` (30 s epochs, features, labels), `results/tables/class_balance.csv` |
| 1b | `python scripts/01b_validate_vs_walch.py` | `results/tables/validate_vs_walch.csv` (feature parity with the reference code) |
| 1c | `python scripts/01c_motion_rate_check.py` | `results/tables/motion_rate_by_subject.csv`, `motion_rate_thinning.csv` |
| 2 | `python scripts/02_baselines.py` | `results/tables/baselines.csv`, `baselines_per_subject.csv`, `baseline_selection.csv`, `sensitivity.csv` |
| 3 | `python scripts/03_context_baselines.py` | `results/tables/context.csv`, `context_per_subject.csv` |
| 3b | `python scripts/03b_window_k_scores.py` | `results/tables/window_k_scores.csv` (inner-CV macro F1 for every k) |
| 4 | `python scripts/04_gru.py` | GRU rows in `context.csv`, `context_per_subject.csv` and `sensitivity.csv` |
| 5 | `python scripts/05_ablations.py` | `results/tables/ablations.csv`, `ablations_per_subject.csv` |
| 7 | `python scripts/07_stats.py` | `results/tables/stats.csv` (paired per-subject kappa tests) |
| 8 | `python scripts/08_edge_check.py` | `results/tables/edge_check.csv` (window gain without edge epochs) |
| 9 | `python scripts/09_subject_analysis.py` | `results/tables/subject_analysis_*.csv` |
| 6 | `python scripts/06_figures.py` | `results/tables/main_results.csv`, `results/figures/fig1-6_*.pdf/png` |

Every LOSO run also writes `results/predictions/<run>_<subject>.csv`, `results/figures/confusion_<run>.png`, `results/logs/folds_<run>.json` (tuned parameters per fold) and a row in `results/logs/timing.csv`.

## Tables and figures

| Output | Content | Script |
|---|---|---|
| `results/tables/main_results.csv` | Final comparison of the main models | 06 (from 02, 03, 04, 07) |
| `results/tables/baselines.csv` | Per-epoch baselines, all feature sets | 02 |
| `results/tables/context.csv` | Window, HMM and GRU context models | 03, 04 |
| `results/tables/ablations.csv` | Motion / HR / motion + HR / all features | 05 |
| `results/tables/sensitivity.csv` | Runs without 7749105 and without the 10 Hz subjects | 02, 04 |
| `results/tables/stats.csv` | Wilcoxon signed-rank tests on per-subject kappa | 07 |
| `results/tables/edge_check.csv` | Window gain on interior epochs only | 08 |
| `results/tables/subject_analysis_*.csv` | Kappa vs subject covariates, transitions, model disagreement | 09 |
| `results/figures/fig1_main_comparison` | Macro F1 and kappa with per-subject spread | 06 |
| `results/figures/fig2_confusion` | Row-normalised confusion matrices: RF, RF + window, BiGRU | 06 |
| `results/figures/fig3_hypnograms` | Scored vs predicted hypnograms, best / median / worst subject | 06 |
| `results/figures/fig4_ablations` | Feature ablations | 06 (from 05) |
| `results/figures/fig5_window_k` | Inner-CV macro F1 vs window size k | 06 (from 03b) |
| `results/figures/fig6_paired_kappa` | Per-subject kappa, RF to RF + window to BiGRU | 06 |

All experiments use leave-one-subject-out evaluation; hyperparameters are tuned inside each training fold only. The forests and other baselines tune with 5-fold inner CV over training subjects. The GRU tunes on a single fixed inner split (24 train / 6 validation subjects), which picks both its setting and its stopping epoch; 5-fold inner CV would take about 5 times longer. The ablation GRUs skip the grid and use hidden 64 and dropout 0.5, the setting gru_all chose most often; the same split then only decides when to stop (see `configs/ablations.yaml`). Logs of the full run are in `results/logs/`. Exact package versions are in `results/logs/env.txt`, compute per run in `results/logs/timing.csv`. Run the tests with `python -m pytest`.

Final numbers per run are in the last log that ran it: `02_baselines.log` for logistic regression and MLP, `02_baselines_forest_grid.log` for the forests (the forest rows in `02_baselines.log` are from an earlier, smaller grid), `03_context_baselines.log` for window_forest_local and hmm_forest_all, `03_context_window_all_k60.log` for window_forest_all, `04_gru.log` for the GRU runs and `05_ablations.log` for the ablations. The tables in `results/tables/` always hold the final numbers.

## Contribution statement

- Kristine Saralidze: code: preprocessing, models, evaluation, figures
- Erin Kvist: report
- Theoni Constantinou: report
- AI tools: see submission README
