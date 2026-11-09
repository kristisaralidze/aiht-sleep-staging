#!/usr/bin/env python
"""Paper figures and the final comparison table, from saved tables and predictions.

Outputs: results/tables/main_results.csv, results/figures/fig1-6_*.pdf/png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import metrics  # noqa: E402
from src.analysis import load_predictions  # noqa: E402
from src.dataset import bounds_from_times  # noqa: E402

TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
PREDICTIONS = ROOT / "results" / "predictions"
EPOCH_S = 30
COLUMN_IN = 4.8
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"

# run: (label, colour, marker). Order is the order in every figure and table.
MODELS = {
    "logistic_all": ("LR", "#8c8c8c", "v"),
    "mlp_all": ("MLP", "#4d4d4d", "^"),
    "forest_all": ("RF", "#0072B2", "o"),
    "hmm_forest_all": ("RF + HMM", "#CC79A7", "D"),
    "window_forest_all": ("RF + window", "#D55E00", "s"),
    "gru_all": ("BiGRU", "#009E73", "P"),
}
ABLATION_MODELS = {"forest": "forest_all", "window_forest": "window_forest_all",
                   "gru": "gru_all"}
FEATURE_SETS = {"motion": "Motion", "hr": "HR", "motion_hr": "Motion + HR",
                "all": "All (+ time)"}
STAGES = ("Wake", "NREM", "REM")
STAGE_LEVEL = {0: 2, 2: 1, 1: 0}    # hypnogram rows: Wake on top, then REM, then NREM

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(figure, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        figure.savefig(FIGURES / f"{name}.{extension}", dpi=300)
    plt.close(figure)
    print(f"{name}.pdf/png")


def read(name):
    return pd.read_csv(TABLES / name, dtype={"subject": str})


def summary_rows():
    return pd.concat([read("baselines.csv"), read("context.csv")]).drop_duplicates("run")


def per_subject_rows():
    return pd.concat([read("baselines_per_subject.csv"), read("context_per_subject.csv")])


def main_results():
    """One row per main model, with the paired test against RF where it exists."""
    summary = summary_rows().set_index("run").loc[list(MODELS)].reset_index()
    stats = read("stats.csv")
    p_vs_rf = stats[stats["baseline"] == "forest_all"].set_index("model")["p_value"]
    table = pd.DataFrame({
        "model": [MODELS[run][0] for run in summary["run"]],
        "run": summary["run"],
        "macro_f1": summary["macro_f1_pooled"],
        "f1_wake": summary["f1_wake_pooled"],
        "f1_nrem": summary["f1_nrem_pooled"],
        "f1_rem": summary["f1_rem_pooled"],
        "kappa_mean": summary["kappa_subj_mean"],
        "kappa_sd": summary["kappa_subj_sd"],
        "kappa_pooled": summary["kappa_pooled"],
        "accuracy": summary["accuracy_pooled"],
        "inner_macro_f1": summary["mean_inner_macro_f1"],
        "p_kappa_vs_rf": summary["run"].map(p_vs_rf),
    })
    table.to_csv(TABLES / "main_results.csv", index=False)
    print("main_results.csv")
    return table


def strip_panel(axis, values_by_run, centre_by_run, ylabel, centre_label):
    rng = np.random.default_rng(0)
    for position, run in enumerate(MODELS):
        label, colour, marker = MODELS[run]
        values = values_by_run[run]
        jitter = rng.uniform(-0.18, 0.18, values.size)
        axis.scatter(position + jitter, values, s=7, color=colour, alpha=0.45, linewidths=0,
                     marker=marker)
        axis.errorbar(position + 0.32, values.mean(), yerr=values.std(ddof=1), color=colour,
                      fmt="none", capsize=0, linewidth=1.2)
        axis.scatter(position + 0.32, centre_by_run[run], s=22, color=colour, marker=marker,
                     edgecolors="white", linewidths=0.6, zorder=3)
    axis.set_xticks(range(len(MODELS)), [MODELS[run][0] for run in MODELS], rotation=45,
                    ha="right", rotation_mode="anchor")
    axis.set_ylabel(ylabel)
    axis.yaxis.grid(True, color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.set_title(centre_label, color=MUTED, fontsize=6.5, loc="left")


def fig1_main(per_subject, summary):
    figure, axes = plt.subplots(1, 2, figsize=(COLUMN_IN, 2.3), constrained_layout=True)
    summary = summary.set_index("run")
    by_run = {run: per_subject[per_subject["run"] == run] for run in MODELS}
    strip_panel(axes[0], {run: frame["macro_f1"].to_numpy() for run, frame in by_run.items()},
                summary["macro_f1_pooled"], "Macro F1", "marker: pooled; bar: subject mean +/- SD")
    strip_panel(axes[1], {run: frame["kappa"].to_numpy() for run, frame in by_run.items()},
                {run: frame["kappa"].mean() for run, frame in by_run.items()},
                "Cohen's kappa", "marker and bar: subject mean +/- SD")
    save(figure, "fig1_main_comparison")


def fig2_confusion(predictions):
    runs = ("forest_all", "window_forest_all", "gru_all")
    figure, axes = plt.subplots(1, 3, figsize=(COLUMN_IN, 1.8), constrained_layout=True)
    for axis, run in zip(axes, runs):
        frames = predictions[run].values()
        y_true = np.concatenate([frame["y_true"].to_numpy() for frame in frames])
        y_pred = np.concatenate([frame["y_pred"].to_numpy() for frame in frames])
        _, rates = metrics.confusion(y_true, y_pred)
        axis.imshow(rates, cmap="Blues", vmin=0, vmax=1)
        for row in range(3):
            for column in range(3):
                axis.text(column, row, f"{rates[row, column]:.0%}", ha="center", va="center",
                          fontsize=7, color="white" if rates[row, column] > 0.55 else INK)
        axis.set_xticks(range(3), STAGES)
        axis.set_yticks(range(3), STAGES if axis is axes[0] else [])
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_title(MODELS[run][0])
        axis.set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    save(figure, "fig2_confusion")


def with_breaks(times, values):
    """Insert NaN at sequence breaks so step lines are not drawn across gaps."""
    hours, levels = [], []
    for start, stop in bounds_from_times(times, EPOCH_S):
        hours += list(times[start:stop] / 3600.0) + [np.nan]
        levels += list(values[start:stop]) + [np.nan]
    return np.array(hours), np.array(levels, dtype=float)


def fig3_hypnograms(predictions, per_subject):
    shown = ("window_forest_all", "gru_all")
    kappa = (per_subject[per_subject["run"].isin(shown)]
             .groupby("subject")["kappa"].mean().sort_values())
    picks = {"Best": kappa.index[-1], "Median": kappa.index[len(kappa) // 2],
             "Worst": kappa.index[0]}
    lookup = per_subject.set_index(["run", "subject"])["kappa"]
    rows = [("Scored (PSG)", None)] + [(MODELS[run][0], run) for run in shown]
    figure = plt.figure(figsize=(COLUMN_IN, 5.4))
    blocks = figure.subfigures(len(picks), 1, hspace=0.02)
    level = np.vectorize(STAGE_LEVEL.get)
    end_hours = max(predictions[shown[0]][subj]["epoch_time"].max() / 3600.0
                    for subj in picks.values()) + 0.1
    for block, (title, subj) in zip(blocks, picks.items()):
        frame = predictions[shown[0]][subj]
        times = frame["epoch_time"].to_numpy()
        axes = block.subplots(len(rows), 1, sharex=True, gridspec_kw={"hspace": 0.15})
        for offset, ((label, run), axis) in enumerate(zip(rows, axes)):
            values = frame["y_true"] if run is None else predictions[run][subj]["y_pred"]
            hours, levels = with_breaks(times, level(values.to_numpy()))
            colour = INK if run is None else MODELS[run][1]
            axis.step(hours, levels, where="post", color=colour, linewidth=0.8)
            axis.set_ylim(-0.4, 2.4)
            axis.set_xlim(0, end_hours)
            axis.set_yticks([0, 1, 2], ["N", "R", "W"])
            axis.tick_params(axis="y", length=0, pad=1)
            axis.yaxis.grid(True, color=GRID, linewidth=0.5)
            axis.text(1.0, 0.5, label, transform=axis.transAxes, ha="left", va="center",
                      fontsize=6.5, color=colour)
            if offset == 0:
                k_text = ", ".join(f"{MODELS[r][0]} {lookup[(r, subj)]:.2f}" for r in shown)
                axis.set_title(f"{title}: subject {subj} (kappa {k_text})", loc="left",
                               fontsize=7)
        axes[-1].set_xlabel("Hours since start of PSG recording" if block is blocks[-1] else "")
    save(figure, "fig3_hypnograms")


def fig4_ablations():
    if not (TABLES / "ablations.csv").exists():
        print("fig4 skipped: run scripts/05_ablations.py first")
        return
    table = read("ablations.csv").set_index(["model", "feature_set"])
    per_subject = pd.concat([per_subject_rows(), read("ablations_per_subject.csv")])
    figure, axes = plt.subplots(1, 2, figsize=(COLUMN_IN, 2.1), constrained_layout=True)
    x = np.arange(len(FEATURE_SETS))
    for shift, (model, reference) in zip((-0.12, 0.0, 0.12), ABLATION_MODELS.items()):
        label, colour, marker = MODELS[reference]
        runs = [table.loc[(model, fs), "run"] for fs in FEATURE_SETS]
        f1 = [table.loc[(model, fs), "macro_f1_pooled"] for fs in FEATURE_SETS]
        kappas = [per_subject[per_subject["run"] == run]["kappa"] for run in runs]
        mean = np.array([k.mean() for k in kappas])
        se = np.array([k.std(ddof=1) / np.sqrt(k.size) for k in kappas])
        axes[0].plot(x + shift, f1, color=colour, marker=marker, markersize=4, linewidth=1.0,
                     label=label)
        axes[1].errorbar(x + shift, mean, yerr=se, color=colour, marker=marker, markersize=4,
                         linewidth=1.0, capsize=0, label=label)
    for axis, ylabel in zip(axes, ("Macro F1 (pooled)", "Kappa (mean +/- SE)")):
        axis.set_xticks(x, list(FEATURE_SETS.values()), rotation=20, ha="right")
        axis.set_ylabel(ylabel)
        axis.yaxis.grid(True, color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper left")
    save(figure, "fig4_ablations")


def fig5_window_k():
    path = TABLES / "window_k_scores.csv"
    if not path.exists():
        print("fig5 skipped: run scripts/03b_window_k_scores.py first")
        return
    scores = read("window_k_scores.csv")
    wide = scores.pivot(index="subject", columns="k", values="inner_macro_f1")
    chosen = scores[scores["chosen"]]["k"].value_counts()
    colour = MODELS["window_forest_all"][1]
    figure, axis = plt.subplots(figsize=(COLUMN_IN * 0.6, 2.0), constrained_layout=True)
    for _, row in wide.iterrows():
        axis.plot(wide.columns, row.to_numpy(), color=MUTED, alpha=0.25, linewidth=0.6)
    axis.plot(wide.columns, wide.mean().to_numpy(), color=colour, marker="s", markersize=4,
              linewidth=1.4, label="mean over outer folds")
    for k in wide.columns:
        axis.annotate(f"{chosen.get(k, 0)}", (k, wide[k].mean()), textcoords="offset points",
                      xytext=(0, 6), ha="center", fontsize=6.5, color=colour)
    axis.set_xticks(list(wide.columns))
    axis.set_xlabel("Window half-width k (epochs)")
    axis.set_ylabel("Inner-CV macro F1")
    axis.yaxis.grid(True, color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.text(0.0, 1.02, "grey: each outer fold\nnumbers: folds choosing k",
              transform=axis.transAxes, color=MUTED, fontsize=6.5)
    save(figure, "fig5_window_k")


def fig6_paired(per_subject):
    runs = ("forest_all", "window_forest_all", "gru_all")
    shown = per_subject[per_subject["run"].isin(runs)]
    wide = shown.pivot(index="subject", columns="run", values="kappa")[list(runs)]
    figure, axis = plt.subplots(figsize=(COLUMN_IN * 0.6, 2.4), constrained_layout=True)
    x = np.arange(len(runs))
    for _, row in wide.iterrows():
        axis.plot(x, row.to_numpy(), color=MUTED, alpha=0.35, linewidth=0.6)
    for position, run in enumerate(runs):
        label, colour, marker = MODELS[run]
        axis.scatter(np.full(len(wide), position), wide[run], s=9, color=colour, marker=marker,
                     linewidths=0, zorder=3)
    axis.plot(x, wide.mean().to_numpy(), color=INK, linewidth=1.4, marker="o", markersize=4,
              zorder=4, label="mean")
    axis.set_xticks(x, [MODELS[run][0] for run in runs])
    axis.set_xlim(-0.3, len(runs) - 0.7)
    axis.set_ylabel("Cohen's kappa (per subject)")
    axis.yaxis.grid(True, color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="lower right")
    save(figure, "fig6_paired_kappa")


def main():
    summary = summary_rows()
    per_subject = per_subject_rows()
    predictions = {run: load_predictions(run, PREDICTIONS)
                   for run in ("forest_all", "window_forest_all", "gru_all")}
    main_results()
    fig1_main(per_subject, summary)
    fig2_confusion(predictions)
    fig3_hypnograms(predictions, per_subject)
    fig4_ablations()
    fig5_window_k()
    fig6_paired(per_subject)


if __name__ == "__main__":
    main()
