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

| Step | Command | Produces |
|---|---|---|
| 1 | `python scripts/01_preprocess.py` | `data/processed/*.npz` (30 s epochs, features, labels), `results/tables/class_balance.csv` |
| 1b | `python scripts/01b_validate_vs_walch.py` | `results/tables/validate_vs_walch.csv` (feature parity with the reference code) |
| 1c | `python scripts/01c_motion_rate_check.py` | `results/tables/motion_rate_by_subject.csv`, `motion_rate_thinning.csv` |
| 2 | `python scripts/02_baselines.py` | Table 1: `results/tables/baselines.csv`, `baseline_selection.csv`, `sensitivity.csv` |
| 3 | `python scripts/03_context_baselines.py` | Table 2: `results/tables/context.csv` |
| 4 | `python scripts/04_gru.py` | Table 2 (GRU rows), `results/predictions/gru_*.csv` |
| 5 | `python scripts/05_ablations.py` | Table 3: `results/tables/ablations.csv` |
| 6 | `python scripts/06_figures.py` | Fig. 1-3: `results/figures/` |
| 7 | `python scripts/07_stats.py` | `results/tables/stats.csv` (paired per-subject kappa tests) |
| 8 | `python scripts/08_edge_check.py` | `results/tables/edge_check.csv` (window gain without edge epochs) |

All experiments use leave-one-subject-out evaluation; hyperparameters are tuned inside each training fold only. Logs of the full run are in `results/logs/`. Exact package versions are in `results/logs/env.txt`, compute per run in `results/logs/timing.csv`. Run the tests with `python -m pytest`.

## Contribution statement

- Kristine Saralidze: code: preprocessing, models, evaluation, figures
- Erin Kvist: report
- Theoni Constantinou: report
- AI tools: TODO: which tools, what they did, roughly what share of the work
