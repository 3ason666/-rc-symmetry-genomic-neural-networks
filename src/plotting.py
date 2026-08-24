from __future__ import annotations

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "results" / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve

sns.set_theme(style="whitegrid")


def _save(fig, directory: Path, stem: str):
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(directory / f"{stem}.png", dpi=160); fig.savefig(directory / f"{stem}.pdf")
    plt.close(fig)


def create_summary_plots(sample_df: pd.DataFrame, run_df: pd.DataFrame, figure_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    perf = run_df.melt(id_vars=["model_type", "seed"], value_vars=["test_accuracy", "test_auroc", "test_auprc", "test_f1"], var_name="metric", value_name="value")
    sns.barplot(perf, x="model_type", y="value", hue="metric", ax=ax, errorbar="sd")
    ax.set(title=f"Test classification performance (runs n={len(run_df)})", ylim=(0, 1), xlabel="Model", ylabel="Metric value")
    _save(fig, figure_dir, "01_test_classification_performance")

    fig, ax = plt.subplots(figsize=(7, 4)); sns.histplot(sample_df, x="prediction_difference", hue="model_type", element="step", stat="density", common_norm=False, ax=ax)
    ax.set(title=f"Forward vs RC prediction difference (samples n={len(sample_df)})", xlabel="|p(x) - p(RC(x))|")
    _save(fig, figure_dir, "02_prediction_difference_distribution")

    fig, ax = plt.subplots(figsize=(6, 4)); sns.barplot(run_df, x="model_type", y="symmetry_flip_rate", errorbar="sd", ax=ax)
    ax.set(title=f"Symmetry flip rate at threshold 0.5 (runs n={len(run_df)})", ylabel="Fraction with differing class")
    _save(fig, figure_dir, "03_symmetry_flip_rate")

    fig, ax = plt.subplots(figsize=(7, 4)); sns.boxplot(sample_df, x="model_type", y="attribution_pearson_absolute", ax=ax)
    ax.set(title=f"Absolute attribution RC consistency (ISM samples n={len(sample_df)})", ylabel="Pearson correlation")
    _save(fig, figure_dir, "04_attribution_consistency_distribution")

    fig, ax = plt.subplots(figsize=(7, 5)); sns.scatterplot(sample_df, x="prediction_consistency", y="attribution_pearson_absolute", hue="model_type", alpha=.7, ax=ax)
    ax.set(title=f"Prediction vs attribution consistency (n={len(sample_df)}; association, not causation)")
    _save(fig, figure_dir, "05_prediction_vs_attribution_consistency")

    motif_cols = ["motif_top8_overlap_absolute", "motif_mass_fraction_absolute", "position_auprc_absolute"]
    motif = sample_df.melt(id_vars=["model_type"], value_vars=motif_cols, var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(8, 4)); sns.barplot(motif, x="model_type", y="value", hue="metric", errorbar="sd", ax=ax)
    ax.set(title=f"Ground-truth motif localization (positive ISM samples n={len(sample_df)})", ylabel="Score")
    _save(fig, figure_dir, "06_motif_localization")

    fig, ax = plt.subplots(figsize=(8, 4)); sns.pointplot(perf, x="model_type", y="value", hue="metric", errorbar="sd", dodge=.3, ax=ax)
    ax.set(title=f"Across-seed mean ± SD (seeds n={run_df.seed.nunique()})", ylim=(0, 1))
    _save(fig, figure_dir, "07_across_seed_error_bars")


def create_roc_plot(prediction_df: pd.DataFrame, figure_dir: Path):
    """Plot mean test ROC across seeds for forward-sequence predictions."""
    grid = np.linspace(0.0, 1.0, 501)
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = dict(zip(sorted(prediction_df.model_type.unique()), sns.color_palette("deep")))
    for model_type, model_group in prediction_df.groupby("model_type"):
        interpolated, aucs = [], []
        for seed, seed_group in model_group.groupby("seed"):
            fpr, tpr, _ = roc_curve(seed_group.label, seed_group.p_forward)
            curve = np.interp(grid, fpr, tpr); curve[0] = 0.0; curve[-1] = 1.0
            interpolated.append(curve); aucs.append(auc(fpr, tpr))
        curves = np.stack(interpolated)
        mean, std = curves.mean(axis=0), curves.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(grid)
        color = palette[model_type]
        ax.plot(grid, mean, color=color, lw=2,
                label=f"{model_type}: AUROC {np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}")
        ax.fill_between(grid, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), color=color, alpha=.16)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Random classifier")
    seeds = prediction_df.seed.nunique(); per_seed = prediction_df.groupby(["model_type", "seed"]).size().iloc[0]
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01),
           title=f"Full experiment test ROC — forward sequences\n{per_seed} test samples per seed; {seeds} seeds; band = ±1 SD")
    ax.legend(loc="lower right")
    _save(fig, figure_dir, "09_test_roc_curves")


def create_integrated_dashboard(sample_df: pd.DataFrame, run_df: pd.DataFrame,
                                prediction_df: pd.DataFrame, output_path: Path):
    """Create one self-contained scientific overview of the full experiment."""
    colors = dict(zip(["CNN-Raw", "CNN-Aug", "CNN-Pair"], sns.color_palette("deep", 3)))
    fig = plt.figure(figsize=(18, 15))
    grid_spec = fig.add_gridspec(3, 3, hspace=.42, wspace=.32)

    ax = fig.add_subplot(grid_spec[0, 0])
    performance = run_df.melt(id_vars=["model_type", "seed"],
                              value_vars=["test_accuracy", "test_auroc", "test_auprc", "test_f1"],
                              var_name="metric", value_name="value")
    sns.barplot(performance, x="model_type", y="value", hue="metric", errorbar="sd", ax=ax)
    ax.set(title="A. Test classification (mean ± SD)", xlabel="", ylabel="Score", ylim=(.98, 1.002))
    ax.tick_params(axis="x", rotation=15); ax.legend(fontsize=8, loc="lower right")

    ax = fig.add_subplot(grid_spec[0, 1])
    sns.barplot(run_df, x="model_type", y="prediction_mean_absolute_difference",
                hue="model_type", palette=colors, legend=False, errorbar="sd", ax=ax)
    sns.stripplot(run_df, x="model_type", y="prediction_mean_absolute_difference", color="black", size=5, ax=ax)
    ax.set(title="B. RC prediction difference (lower is better)", xlabel="", ylabel="Mean |p(x)-p(RC(x))|")
    ax.tick_params(axis="x", rotation=15)

    ax = fig.add_subplot(grid_spec[0, 2])
    sns.barplot(run_df, x="model_type", y="symmetry_flip_rate", hue="model_type",
                palette=colors, legend=False, errorbar="sd", ax=ax)
    ax.set(title="C. Symmetry flip rate at threshold 0.5", xlabel="", ylabel="Fraction of test samples")
    ax.tick_params(axis="x", rotation=15)

    ax = fig.add_subplot(grid_spec[1, 0])
    consistency = sample_df.melt(id_vars=["model_type"],
        value_vars=["attribution_pearson_absolute", "attribution_spearman_absolute", "attribution_cosine_absolute"],
        var_name="metric", value_name="value")
    consistency["metric"] = consistency.metric.str.replace("attribution_", "", regex=False).str.replace("_absolute", "", regex=False)
    sns.barplot(consistency, x="model_type", y="value", hue="metric", errorbar="sd", ax=ax)
    ax.set(title="D. Aligned RC attribution consistency", xlabel="", ylabel="Similarity", ylim=(.55, 1.01))
    ax.tick_params(axis="x", rotation=15); ax.legend(fontsize=8)

    ax = fig.add_subplot(grid_spec[1, 1])
    motif = sample_df.melt(id_vars=["model_type"],
        value_vars=["motif_top8_overlap_absolute", "motif_mass_fraction_absolute", "position_auprc_absolute"],
        var_name="metric", value_name="value")
    motif["metric"] = motif.metric.map({"motif_top8_overlap_absolute": "top-8 overlap",
                                        "motif_mass_fraction_absolute": "mass fraction",
                                        "position_auprc_absolute": "position AUPRC"})
    sns.barplot(motif, x="model_type", y="value", hue="metric", errorbar="sd", ax=ax)
    ax.set(title="E. Ground-truth motif localization", xlabel="", ylabel="Score", ylim=(.85, 1.01))
    ax.tick_params(axis="x", rotation=15); ax.legend(fontsize=8, loc="lower right")

    ax = fig.add_subplot(grid_spec[1, 2])
    sns.scatterplot(sample_df, x="prediction_consistency", y="attribution_pearson_absolute",
                    hue="model_type", palette=colors, alpha=.38, s=20, ax=ax)
    ax.set(title="F. Prediction vs attribution consistency", xlabel="Prediction consistency",
           ylabel="Attribution Pearson")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(grid_spec[2, 0])
    grid = np.linspace(0, 1, 501)
    for model_type, group in prediction_df.groupby("model_type"):
        curves, aucs = [], []
        for _, seed_group in group.groupby("seed"):
            fpr, tpr, _ = roc_curve(seed_group.label, seed_group.p_forward)
            curves.append(np.interp(grid, fpr, tpr)); aucs.append(auc(fpr, tpr))
        ax.plot(grid, np.mean(curves, axis=0), lw=2, color=colors[model_type],
                label=f"{model_type} AUC={np.mean(aucs):.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set(title="G. Test ROC (forward sequences)", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(fontsize=8, loc="lower right")

    ax = fig.add_subplot(grid_spec[2, 1]); ax.axis("off")
    ax.text(0, 1, "H. Study design", va="top", fontsize=14, weight="bold")
    ax.text(0, .85, "Synthetic DNA: 8,000 train / 1,000 validation / 1,000 test\n"
                    "Sequence length: 100 bp; planted motif: TGATTTAT\n"
                    "Models: Raw, random RC augmentation, paired RC training\n"
                    "Seeds: 42, 123, 2026 (9 trained checkpoints)\n"
                    "ISM: same 200 positive test samples per run\n"
                    "Leakage audit: 0 duplicate sequences; 0 cross-split RC pairs",
            va="top", fontsize=11, linespacing=1.55)

    ax = fig.add_subplot(grid_spec[2, 2]); ax.axis("off")
    ax.text(0, 1, "I. Integrated interpretation", va="top", fontsize=14, weight="bold")
    ax.text(0, .85, "• Classification is at ceiling (AUROC = 1.0).\n"
                    "• Aug and Pair reduce forward-vs-RC probability differences.\n"
                    "• Pearson/cosine and top-8 attribution agreement are near 1.\n"
                    "• Spearman is lower (≈0.65–0.72): background ranks vary.\n"
                    "• Motif localization is also near ceiling (AUPRC ≈0.997).\n"
                    "• Prediction–attribution association is weak within this\n"
                    "  restricted ceiling range; correlation is not causation.\n"
                    "• Phase 1 therefore validates the pipeline but does not\n"
                    "  provide a strong counterexample for H1 or H3.",
            va="top", fontsize=11, linespacing=1.45)

    fig.suptitle("Phase 1 Full Experiment — Integrated Results Dashboard", fontsize=22, weight="bold", y=.995)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_representative_samples(sample_df: pd.DataFrame, arrays: dict, figure_dir: Path):
    candidates = [
        ("prediction consistent / attribution consistent", sample_df.assign(_score=sample_df.prediction_consistency.rank(pct=True) + sample_df.attribution_pearson_absolute.rank(pct=True)).sort_values("_score", ascending=False)),
        ("prediction consistent / attribution inconsistent", sample_df[sample_df.prediction_consistency >= sample_df.prediction_consistency.median()].sort_values("attribution_pearson_absolute")),
        ("prediction inconsistent", sample_df.sort_values("prediction_difference", ascending=False)),
        ("attribution consistent / motif localization weak", sample_df.assign(_score=sample_df.attribution_pearson_absolute.rank(pct=True) - sample_df.position_auprc_absolute.rank(pct=True)).sort_values("_score", ascending=False)),
        ("motif localization strong", sample_df.sort_values("position_auprc_absolute", ascending=False)),
        ("near-constant attribution", sample_df.sort_values("attribution_std_absolute")),
    ]
    rows, used = [], set()
    for category, ordered in candidates:
        for index, row in ordered.iterrows():
            key = (row.model_type, int(row.seed), row.sample_id)
            if key not in used:
                used.add(key); rows.append((category, index, row)); break
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, max(3, 3.0 * len(rows))), squeeze=False)
    for ax, (category, _, row) in zip(axes[:, 0], rows):
        key = (row.model_type, int(row.seed), row.sample_id)
        forward, aligned = arrays[key]
        x = np.arange(len(forward)); ax.plot(x, forward, label="Forward |ISM|", lw=1.2); ax.plot(x, aligned, label="Aligned RC |ISM|", lw=1.2)
        ax.axvspan(int(row.motif_start), int(row.motif_end) - 1, color="gold", alpha=.25, label=f"True motif ({row.motif_orientation})")
        ax.set_title(f"{category} | {row.model_type}, seed {int(row.seed)}, {row.sample_id}; p={row.p_forward:.3f}, pRC={row.p_rc:.3f}, r={row.attribution_pearson_absolute:.3f}\noriginal sequence: {row.sequence}", fontsize=8)
        ax.set_ylabel("Attribution"); ax.legend(loc="upper right", fontsize=7, ncol=3)
    axes[-1, 0].set_xlabel("Original-sequence position (0-based)")
    _save(fig, figure_dir, "08_representative_attributions")
