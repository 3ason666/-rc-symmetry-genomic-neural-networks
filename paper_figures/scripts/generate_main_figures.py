from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


PROJECT = ROOT
RESULTS = PROJECT / "results"
P2D = ROOT / "paper_figures" / "data" / "phase2d_rcps_confirmatory"
P2E = ROOT / "paper_figures" / "data" / "phase2e_biological_correctness"
P2F = ROOT / "paper_figures" / "data" / "phase2f_one_time_test"
P3C = ROOT / "paper_figures" / "data" / "phase3c_one_time_test"
P3E = ROOT / "paper_figures" / "data" / "phase3e_near_duplicate_sensitivity"
FIG_ROOT = ROOT / "paper_figures"
MAIN = FIG_ROOT / "main"
DATA = FIG_ROOT / "data"
CAPTIONS = FIG_ROOT / "captions"
VALIDATION = FIG_ROOT / "validation"

MODEL_ORDER = ["CNN-Raw", "CNN-Aug", "CNN-RCPS"]
COLORS = {
    "CNN-Raw": "#D55E00",
    "CNN-Aug": "#0072B2",
    "CNN-RCPS": "#009E73",
}
FOLD_MARKERS = {"fold_a": "o", "fold_b": "s", "fold_c": "D"}
SEED_OFFSET = {42: -0.11, 123: 0.0, 2026: 0.11}
NUCLEOTIDES = ["A", "C", "G", "T"]
NEUTRAL = "#3C4858"
LIGHT = "#EEF2F6"
GRID = "#D8DEE7"
SUCCESS = "#2A9D8F"
AMBER = "#E9C46A"


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required frozen source missing: {path}")
    return path


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(require(path), **kwargs)


def read_json(path: Path):
    return json.loads(require(path).read_text(encoding="utf-8"))


def read_threshold(path: Path) -> float:
    text = require(path).read_text(encoding="utf-8")
    match = re.search(r"prediction_consistent_max_abs_difference:\s*([0-9.eE+-]+)", text)
    if not match:
        raise ValueError(f"Threshold not found in {path}")
    return float(match.group(1))


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.edgecolor": NEUTRAL,
            "axes.labelcolor": "#1B2632",
            "xtick.color": "#344050",
            "ytick.color": "#344050",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.08,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def finish_figure(fig: plt.Figure, number: int, title: str) -> None:
    out = MAIN / f"figure{number}"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"figure{number}.svg", bbox_inches="tight")
    fig.savefig(out / f"figure{number}.pdf", bbox_inches="tight")
    fig.savefig(out / f"figure{number}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def rounded_box(
    ax,
    xy,
    width,
    height,
    text,
    facecolor=LIGHT,
    edgecolor=NEUTRAL,
    fontsize=8.5,
    textcolor="#17212B",
    linewidth=1.0,
):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=textcolor,
        linespacing=1.25,
    )
    return patch


def arrow(ax, start, end, color=NEUTRAL, lw=1.2, mutation_scale=11):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        color=color,
        linewidth=lw,
        mutation_scale=mutation_scale,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)
    return patch


def strip_summary(ax, frame: pd.DataFrame, value: str, ylabel: str, y_lim=None, zero=False):
    xmap = {model: i for i, model in enumerate(MODEL_ORDER)}
    for _, row in frame.iterrows():
        model = row["model_type"]
        x = xmap[model] + SEED_OFFSET[int(row["seed"])]
        marker = FOLD_MARKERS[str(row["fold_id"])]
        ax.scatter(
            x,
            row[value],
            s=31,
            marker=marker,
            facecolor=COLORS[model],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.9,
            zorder=3,
        )
    for i, model in enumerate(MODEL_ORDER):
        vals = frame.loc[frame["model_type"] == model, value].astype(float)
        ax.scatter(i, vals.mean(), marker="D", s=50, facecolor="white", edgecolor="#111111", linewidth=1.25, zorder=5)
    ax.set_xticks(range(3), MODEL_ORDER)
    ax.set_ylabel(ylabel)
    if y_lim:
        ax.set_ylim(*y_lim)
    if zero:
        ax.axhline(0, color=GRID, lw=0.9, zorder=0)
    ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.text(
        0.01,
        0.98,
        "points: fold x seed; diamond: mean",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="#596574",
    )


def build_figure1() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.7, 7.3))
    fig.subplots_adjust(top=0.96, bottom=0.06, left=0.07, right=0.98, wspace=0.20, hspace=0.20)

    ax = axes[0, 0]
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    panel_label(ax, "A")
    ax.set_title("Paired orientation evaluation", loc="left", fontweight="bold")
    rounded_box(ax, (0.02, 0.66), 0.18, 0.17, "S\n5'-ACGT...-3'", facecolor="#EAF4FB", edgecolor=COLORS["CNN-Aug"])
    rounded_box(ax, (0.02, 0.25), 0.18, 0.17, "RC(S)\n5'-...ACGT-3'", facecolor="#EAF4FB", edgecolor=COLORS["CNN-Aug"])
    rounded_box(ax, (0.35, 0.43), 0.20, 0.22, "same trained\nmodel", facecolor="#F5F0FA", edgecolor="#7B5EA7")
    rounded_box(ax, (0.69, 0.68), 0.27, 0.14, "Prediction\np(S), p(RC(S))", facecolor="#FFF4E6", edgecolor="#C77C16")
    rounded_box(ax, (0.69, 0.24), 0.27, 0.18, "Attribution\nA(S), align[A(RC(S))]", facecolor="#EAF7F1", edgecolor=SUCCESS)
    for y in (0.745, 0.335):
        arrow(ax, (0.20, y), (0.35, 0.54))
    arrow(ax, (0.55, 0.56), (0.69, 0.75))
    arrow(ax, (0.55, 0.49), (0.69, 0.33))
    ax.text(0.825, 0.59, "compare separately", ha="center", va="center", fontsize=8, color=NEUTRAL)

    ax = axes[0, 1]
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    panel_label(ax, "B")
    ax.set_title("Three independent evidence layers", loc="left", fontweight="bold")
    boxes = [
        (0.68, "Prediction consistency", "|p(S) - p(RC(S))| and label flips", "#EAF4FB", "#0072B2"),
        (0.41, "Attribution consistency", "RC-aligned Exact ISM agreement", "#F5F0FA", "#7B5EA7"),
        (0.14, "Biological validity", "Motif localization and in silico perturbation", "#EAF7F1", "#2A9D8F"),
    ]
    for y, head, sub, fc, ec in boxes:
        rounded_box(ax, (0.08, y), 0.84, 0.17, f"{head}\n{sub}", facecolor=fc, edgecolor=ec, fontsize=9)

    ax = axes[1, 0]
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    panel_label(ax, "C")
    ax.set_title("Reverse-complement strategies", loc="left", fontweight="bold")
    strategy = [
        ("CNN-Raw", "Single orientation\nNo RC constraint"),
        ("CNN-Aug", "Random S / RC(S)\nAugmented training"),
        ("CNN-RCPS", "Shared parameters\nExact RC symmetry"),
    ]
    for i, (model, desc) in enumerate(strategy):
        x = 0.03 + i * 0.325
        rounded_box(ax, (x, 0.25), 0.29, 0.47, f"{model}\n\n{desc}", facecolor=mpl.colors.to_rgba(COLORS[model], 0.11), edgecolor=COLORS[model], fontsize=9, linewidth=1.4)
    ax.text(0.5, 0.1, "Raw / augmentation / architectural enforcement", ha="center", fontsize=8, color=NEUTRAL)

    ax = axes[1, 1]
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    panel_label(ax, "D")
    ax.set_title("Evidence progression and parallel external settings", loc="left", fontweight="bold")
    rounded_box(ax, (0.35, 0.82), 0.30, 0.09, "Synthetic tasks", facecolor="#F7F8FA", fontsize=7.7)
    rounded_box(ax, (0.35, 0.66), 0.30, 0.09, "K562 confirmation", facecolor="#F7F8FA", fontsize=7.7)
    rounded_box(ax, (0.35, 0.50), 0.30, 0.09, "Frozen K562 test", facecolor="#FFF3DE", edgecolor="#B87900", fontsize=7.7)
    arrow(ax, (0.50, 0.82), (0.50, 0.75), lw=1.0)
    arrow(ax, (0.50, 0.66), (0.50, 0.59), lw=1.0)

    rounded_box(
        ax,
        (0.03, 0.29),
        0.42,
        0.13,
        "Fetal erythroid GATA1\nsame TF / different context",
        facecolor="#EAF7F1",
        edgecolor=SUCCESS,
        fontsize=7.2,
    )
    rounded_box(
        ax,
        (0.55, 0.29),
        0.42,
        0.13,
        "GM12878 CTCF\ndifferent TF / different context",
        facecolor="#F5F0FA",
        edgecolor="#7B5EA7",
        fontsize=7.2,
    )
    arrow(ax, (0.50, 0.50), (0.24, 0.42), lw=1.0)
    arrow(ax, (0.50, 0.50), (0.76, 0.42), lw=1.0)
    rounded_box(ax, (0.27, 0.225), 0.46, 0.045, "Primary external replication (Phase 3C)", facecolor="#EDF5F3", edgecolor=SUCCESS, fontsize=6.5)
    arrow(ax, (0.24, 0.29), (0.40, 0.27), lw=0.8)
    arrow(ax, (0.76, 0.29), (0.60, 0.27), lw=0.8)

    rounded_box(ax, (0.27, 0.13), 0.46, 0.06, "Cross-stage overlap audit", facecolor="#F1F3F6", edgecolor=NEUTRAL, fontsize=6.9)
    rounded_box(ax, (0.27, 0.04), 0.46, 0.06, "Near-duplicate sensitivity analysis (Phase 3E)", facecolor="#FFF3DE", edgecolor="#B87900", fontsize=6.6)
    arrow(ax, (0.50, 0.225), (0.50, 0.19), lw=0.8)
    arrow(ax, (0.50, 0.13), (0.50, 0.10), lw=0.8)

    finish_figure(
        fig,
        1,
        "Experimental framework for separating prediction consistency, attribution consistency, and biological validity",
    )


def build_figure2(train: pd.DataFrame, dataset_counts: pd.Series, test_chromosomes: list[str], h2: pd.Series) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.7))
    fig.subplots_adjust(top=0.96, bottom=0.08, left=0.08, right=0.98, wspace=0.27, hspace=0.31)

    ax = axes[0, 0]
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    panel_label(ax, "A")
    ax.set_title("K562 GATA1 dataset design", loc="left", fontweight="bold")
    total = int(dataset_counts.sum())
    rounded_box(ax, (0.05, 0.68), 0.90, 0.18, f"One matched K562 GATA1 dataset\n256 bp | total N = {total:,}", facecolor="#EFF5FA", edgecolor="#4C6A85", fontsize=9.5)
    development_n = int(dataset_counts["train"] + dataset_counts["validation"])
    rounded_box(
        ax,
        (0.05, 0.24),
        0.57,
        0.29,
        f"Development-side partitions | N = {development_n:,}\n"
        f"P2D chromosome-CV source: train N = {int(dataset_counts['train']):,}\n"
        f"Reserved validation: N = {int(dataset_counts['validation']):,}",
        facecolor="#F7F8FA",
        edgecolor=NEUTRAL,
        fontsize=7.7,
    )
    rounded_box(
        ax,
        (0.67, 0.24),
        0.28,
        0.29,
        f"Frozen held-out test\nN = {int(dataset_counts['test']):,}\n"
        + ", ".join(test_chromosomes),
        facecolor="#FFF3DE",
        edgecolor="#B87900",
        fontsize=7.7,
    )
    arrow(ax, (0.50, 0.68), (0.34, 0.53), lw=0.9)
    arrow(ax, (0.50, 0.68), (0.81, 0.53), lw=0.9)
    ax.text(0.50, 0.10, "Predefined partitions within the same matched dataset", ha="center", fontsize=7.5, color=NEUTRAL)

    ax = axes[0, 1]
    panel_label(ax, "B")
    ax.set_title("Confirmatory discrimination", loc="left", fontweight="bold")
    strip_summary(ax, train, "holdout_auroc", "AUROC", y_lim=(0.825, 0.895))

    ax = axes[1, 0]
    panel_label(ax, "C")
    ax.set_title("Prediction asymmetry", loc="left", fontweight="bold")
    strip_summary(ax, train, "prediction_mean_absolute_difference", "Mean Δp = |p(S) - p(RC(S))|", y_lim=(-0.004, 0.14), zero=True)
    ax.text(
        0.60,
        0.97,
        f"Aug - Raw = {h2['estimate']:.4f}\n95% CI [{h2['ci95_low']:.4f}, {h2['ci95_high']:.4f}]",
        transform=ax.transAxes,
        fontsize=7.5,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GRID),
    )

    ax = axes[1, 1]
    panel_label(ax, "D")
    ax.set_title("Classification flip rate", loc="left", fontweight="bold")
    strip_summary(ax, train, "symmetry_flip_rate", "Fraction crossing decision threshold", y_lim=(-0.004, 0.16), zero=True)

    finish_figure(fig, 2, "Reverse-complement augmentation improves prediction consistency without enforcing exact symmetry")


def select_figure3b(threshold: float):
    attr_path = P2F / "test_attribution_results.csv"
    pred_path = P2F / "test_prediction_results.csv"
    attrs = read_csv(attr_path)
    preds = read_csv(pred_path)
    raw = attrs.loc[attrs["model_type"] == "CNN-Raw"].copy()
    keys = ["fold_id", "model_type", "seed", "sample_id"]
    pred_cols = keys + ["p_forward", "p_rc", "prediction_difference"]
    joined = raw.merge(preds[pred_cols], on=keys, how="left", suffixes=("_attr", "_pred"), validate="one_to_one")
    if joined[["p_forward", "p_rc", "prediction_difference_pred"]].isna().any().any():
        raise ValueError("Figure 3 prediction join has missing rows")
    max_delta_error = np.max(np.abs(joined["prediction_difference_attr"] - joined["prediction_difference_pred"]))
    if max_delta_error > 1e-12:
        raise ValueError(f"Figure 3 delta_p source disagreement: {max_delta_error}")
    joined["prediction_difference"] = joined["prediction_difference_attr"]
    eligible = joined.loc[joined["prediction_difference"] <= threshold].copy()
    if eligible.empty:
        raise ValueError("No CNN-Raw prediction-consistent Exact-ISM observations")
    median_l1 = float(eligible["absolute_normalized_l1"].median())
    eligible["distance_to_median"] = np.abs(eligible["absolute_normalized_l1"] - median_l1)
    eligible = eligible.sort_values(
        ["distance_to_median", "sample_id", "fold_id", "seed"],
        kind="mergesort",
    )
    selected = eligible.iloc[0].copy()
    slug = {"CNN-Raw": "cnn_raw", "CNN-Aug": "cnn_aug", "CNN-RCPS": "cnn_rcps"}[selected["model_type"]]
    run = f"{selected['fold_id']}__{slug}__seed_{int(selected['seed'])}"
    npz_path = require(P2F / "runs" / run / "test_exact_ism_attributions.npz")
    archive = np.load(npz_path, allow_pickle=False)
    sample_ids = archive["sample_ids"].astype(str)
    matches = np.flatnonzero(sample_ids == str(selected["sample_id"]))
    if len(matches) != 1:
        raise ValueError(f"Selected sample appears {len(matches)} times in {npz_path}")
    idx = int(matches[0])
    forward = np.asarray(archive["forward_matrix"][idx], dtype=float)
    aligned = np.asarray(archive["aligned_rc_matrix"][idx], dtype=float)
    if forward.shape != aligned.shape or forward.shape[1] != 4:
        raise ValueError(f"Unexpected Exact-ISM matrix shapes: {forward.shape}, {aligned.shape}")
    return raw, eligible, selected, median_l1, npz_path, forward, aligned


def build_figure3(
    raw_attr: pd.DataFrame,
    eligible: pd.DataFrame,
    selected: pd.Series,
    median_l1: float,
    npz_path: Path,
    forward: np.ndarray,
    aligned: np.ndarray,
    attr_runs: pd.DataFrame,
    h1b: pd.Series,
    threshold: float,
) -> None:
    fig = plt.figure(figsize=(11.2, 9.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], wspace=0.28, hspace=0.32, top=0.96, bottom=0.07)

    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "A")
    ax.set_title("Frozen Exact-ISM cohort: CNN-Raw", loc="left", fontweight="bold")
    fold_colors = {"fold_a": "#4C78A8", "fold_b": "#7A5195", "fold_c": "#59A14F"}
    for fold, group in raw_attr.groupby("fold_id", sort=True):
        ax.scatter(
            group["prediction_difference"],
            group["absolute_normalized_l1"],
            s=14,
            alpha=0.35,
            color=fold_colors.get(fold, NEUTRAL),
            linewidth=0,
            label=fold.replace("fold_", "fold ").upper(),
        )
    ax.axvline(threshold, color="#B51F1F", linestyle="--", linewidth=1.2, label=f"Δp = {threshold:.2f}")
    ax.set_xlabel("Prediction asymmetry Δp")
    ax.set_ylabel("Exact ISM normalized L1")
    ax.grid(color=GRID, lw=0.6, alpha=0.75)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.text(
        0.02,
        0.02,
        f"{len(raw_attr):,} sample-run observations; {raw_attr['sample_id'].nunique():,} unique sequences\n(80 sequences x 9 fold-seed runs); not all 3,496 test sequences",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color=NEUTRAL,
        bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.25", alpha=0.92),
    )

    heat_gs = gs[0, 1].subgridspec(2, 1, hspace=0.18)
    limit = float(max(np.max(np.abs(forward)), np.max(np.abs(aligned))))
    if limit == 0:
        limit = 1e-12
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    heat_axes = []
    for i, (matrix, name) in enumerate([(forward, "ISM on S"), (aligned, "RC-aligned ISM on RC(S)")]):
        hax = fig.add_subplot(heat_gs[i, 0])
        heat_axes.append(hax)
        image = hax.imshow(matrix.T, aspect="auto", interpolation="nearest", cmap="RdBu_r", norm=norm, origin="upper")
        hax.set_yticks(range(4), NUCLEOTIDES)
        hax.set_ylabel("Base")
        hax.set_title(name, loc="left", fontsize=8.5, fontweight="bold")
        hax.set_xlim(-0.5, forward.shape[0] - 0.5)
        if i == 1:
            hax.set_xlabel("Sequence position (bp)")
        else:
            hax.set_xticklabels([])
        hax.spines["top"].set_visible(True)
        hax.spines["right"].set_visible(True)
    panel_label(heat_axes[0], "B")
    cbar = fig.colorbar(image, ax=heat_axes, orientation="vertical", fraction=0.035, pad=0.02)
    cbar.set_label("Exact ISM logit change", fontsize=8)
    heat_axes[0].text(
        0.99,
        1.13,
        f"Δp = {selected['prediction_difference']:.4f}  |  Exact ISM L1 = {selected['absolute_normalized_l1']:.4f}",
        transform=heat_axes[0].transAxes,
        fontsize=7.1,
        color=NEUTRAL,
        va="bottom",
        ha="right",
    )

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "C")
    ax.set_title("Confirmatory Exact ISM asymmetry", loc="left", fontweight="bold")
    exact = attr_runs.loc[attr_runs["method"] == "exact_ism"].copy()
    strip_summary(ax, exact, "mean_absolute_normalized_l1", "Mean Exact ISM normalized L1", y_lim=(-0.012, 0.42), zero=True)
    ax.text(0.98, 0.06, "RCPS near numerical zero\n(linear scale)", transform=ax.transAxes, ha="right", fontsize=7, color=NEUTRAL)

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "D")
    ax.set_title("Near-identical predictions, asymmetric explanations", loc="left", fontweight="bold")
    estimate = float(h1b["estimate"])
    lo = float(h1b["ci95_low"])
    hi = float(h1b["ci95_high"])
    ax.axvline(0, color=GRID, lw=1)
    ax.errorbar(
        estimate,
        0,
        xerr=[[estimate - lo], [hi - estimate]],
        fmt="o",
        color=COLORS["CNN-Raw"],
        ecolor=COLORS["CNN-Raw"],
        capsize=4,
        markersize=7,
        linewidth=1.6,
    )
    ax.set_xlim(0, max(0.42, hi * 1.2))
    ax.set_ylim(-0.8, 0.8)
    ax.set_yticks([0], ["CNN-Raw\nΔp <= 0.01"])
    ax.set_xlabel("Exact ISM normalized L1")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.text(
        0.03,
        0.90,
        f"Estimate {estimate:.4f}\n95% CI [{lo:.4f}, {hi:.4f}]\neligible n = {int(h1b['n'])}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    finish_figure(fig, 3, "Prediction consistency does not imply attribution consistency")


def build_figure4(localization: pd.DataFrame, disruption_boot: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.8))
    fig.subplots_adjust(top=0.96, bottom=0.08, left=0.08, right=0.98, wspace=0.29, hspace=0.31)

    ax = axes[0, 0]
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    panel_label(ax, "A")
    ax.set_title("Biological feature diagnostics", loc="left", fontweight="bold")
    ax.plot([0.08, 0.92], [0.72, 0.72], color=NEUTRAL, lw=5, solid_capstyle="round")
    ax.add_patch(Rectangle((0.40, 0.66), 0.18, 0.12, facecolor="#E76F51", edgecolor="none"))
    ax.text(0.49, 0.84, "GATA1 motif", ha="center", fontsize=8, fontweight="bold")
    x = np.linspace(0.08, 0.92, 40)
    curve = 0.44 + 0.22 * np.exp(-((x - 0.49) / 0.09) ** 2)
    ax.plot(x, curve, color="#7B5EA7", lw=2)
    ax.fill_between(x, 0.42, curve, color="#7B5EA7", alpha=0.16)
    ax.text(0.08, 0.36, "Localization: attribution mass within motif", fontsize=8, color=NEUTRAL)
    rounded_box(ax, (0.08, 0.08), 0.36, 0.16, "Disrupt motif\nmeasure logit drop", facecolor="#FCE8E3", edgecolor="#C84C34")
    rounded_box(ax, (0.56, 0.08), 0.36, 0.16, "Disrupt matched flank\nnegative control", facecolor="#EFF2F5", edgecolor="#64748B")
    arrow(ax, (0.49, 0.66), (0.26, 0.24), color="#C84C34")
    arrow(ax, (0.70, 0.69), (0.74, 0.24), color="#64748B")

    ax = axes[0, 1]
    panel_label(ax, "B")
    ax.set_title("Motif attribution mass", loc="left", fontweight="bold")
    rng = np.random.default_rng(20260824)
    positions = np.arange(3)
    data = [localization.loc[localization["model_type"] == m, "motif_mass_fraction"].astype(float).to_numpy() for m in MODEL_ORDER]
    violin = ax.violinplot(data, positions=positions, widths=0.72, showmeans=False, showextrema=False, showmedians=False)
    for body, model in zip(violin["bodies"], MODEL_ORDER):
        body.set_facecolor(COLORS[model])
        body.set_edgecolor(COLORS[model])
        body.set_alpha(0.18)
    for i, (model, vals) in enumerate(zip(MODEL_ORDER, data)):
        jitter = rng.uniform(-0.16, 0.16, len(vals))
        ax.scatter(i + jitter, vals, s=6, color=COLORS[model], alpha=0.16, linewidth=0)
        q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
        ax.plot([i, i], [q1, q3], color="#1D2733", lw=4, solid_capstyle="butt")
        ax.scatter(i, med, s=18, color="white", edgecolor="#1D2733", zorder=5)
        ax.scatter(i, vals.mean(), s=40, marker="D", color=COLORS[model], edgecolor="white", linewidth=0.6, zorder=5)
    ax.set_xticks(positions, MODEL_ORDER)
    ax.set_ylabel("Fraction of absolute attribution mass in motif")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.text(0.02, 0.98, "strong PWM hits; points are eligible sample-run observations", transform=ax.transAxes, va="top", fontsize=7, color=NEUTRAL)

    ax = axes[1, 0]
    panel_label(ax, "C")
    ax.set_title("Motif disruption exceeds flank disruption", loc="left", fontweight="bold")
    rows = disruption_boot.loc[disruption_boot["endpoint"] == "motif_minus_flank_logit_drop_vs_zero"].copy()
    rows["model_type"] = rows["contrast"]
    rows["order"] = rows["model_type"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    rows = rows.sort_values("order")
    y = np.arange(len(rows))[::-1]
    ax.axvline(0, color=NEUTRAL, lw=1, linestyle="--")
    for yi, (_, row) in zip(y, rows.iterrows()):
        est, lo, hi = float(row["estimate"]), float(row["ci95_low"]), float(row["ci95_high"])
        model = row["model_type"]
        ax.errorbar(est, yi, xerr=[[est - lo], [hi - est]], fmt="o", color=COLORS[model], capsize=3, markersize=6, lw=1.5)
    ax.set_yticks(y, rows["model_type"])
    ax.set_xlabel("Motif - flank logit drop (estimate and 95% CI)")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.text(0.02, 0.03, "Paired, cluster-aware bootstrap", transform=ax.transAxes, fontsize=7, color=NEUTRAL)

    ax = axes[1, 1]
    panel_label(ax, "D")
    ax.set_title("Independent evaluation axes", loc="left", fontweight="bold")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    arrow(ax, (0.18, 0.18), (0.91, 0.18), color=COLORS["CNN-Aug"], lw=1.4)
    arrow(ax, (0.18, 0.18), (0.18, 0.89), color=SUCCESS, lw=1.4)
    ax.text(0.58, 0.08, "RC consistency", ha="center", fontweight="bold", color=COLORS["CNN-Aug"])
    ax.text(0.06, 0.56, "Biological\nfeature validity", ha="center", va="center", rotation=90, fontweight="bold", color=SUCCESS)
    rounded_box(ax, (0.39, 0.50), 0.42, 0.20, "Measure both axes\nindependently", facecolor="#F6F7F9", edgecolor=NEUTRAL, fontsize=9)

    finish_figure(fig, 4, "Reverse-complement symmetry and biological feature validity require separate evaluation")


def decision_map(path: Path) -> dict[str, dict]:
    return {row["hypothesis"]: row for row in read_json(path)}


def build_figure5(
    k_decisions: dict,
    g_decisions: dict,
    c_decisions: dict,
    primary_h2: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.9, 8.1))
    fig.subplots_adjust(top=0.96, bottom=0.08, left=0.08, right=0.98, wspace=0.30, hspace=0.43)

    ax = axes[0, 0]
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    panel_label(ax, "A")
    ax.set_title("Parallel external settings and robustness", loc="left", fontweight="bold")
    rounded_box(ax, (0.02, 0.47), 0.22, 0.18, "K562 GATA1\nfrozen test", facecolor="#EFF5FA", edgecolor="#4C6A85", fontsize=7.7)
    rounded_box(ax, (0.34, 0.70), 0.30, 0.16, "Fetal erythroid GATA1\nsame TF / different context", facecolor="#EAF7F1", edgecolor=SUCCESS, fontsize=6.9)
    rounded_box(ax, (0.34, 0.30), 0.30, 0.16, "GM12878 CTCF\ndifferent TF / different context", facecolor="#F5F0FA", edgecolor="#7B5EA7", fontsize=6.9)
    arrow(ax, (0.24, 0.56), (0.34, 0.78), lw=1.0)
    arrow(ax, (0.24, 0.56), (0.34, 0.38), lw=1.0)
    rounded_box(ax, (0.73, 0.47), 0.24, 0.18, "External replication\nPhase 3C primary", facecolor="#EDF5F3", edgecolor=SUCCESS, fontsize=7.1)
    arrow(ax, (0.64, 0.78), (0.73, 0.60), lw=0.9)
    arrow(ax, (0.64, 0.38), (0.73, 0.52), lw=0.9)
    rounded_box(ax, (0.42, 0.08), 0.25, 0.13, "Cross-stage\noverlap audit", facecolor="#F1F3F6", edgecolor=NEUTRAL, fontsize=7.0)
    rounded_box(ax, (0.74, 0.08), 0.23, 0.13, "Near-duplicate\nsensitivity", facecolor="#FFF3DE", edgecolor="#B87900", fontsize=7.0)
    arrow(ax, (0.85, 0.47), (0.55, 0.21), lw=0.9)
    arrow(ax, (0.67, 0.145), (0.74, 0.145), lw=0.9)
    ax.text(0.855, 0.02, "Phase 3E", ha="center", fontsize=6.7, color="#8B6508")

    ax = axes[0, 1]
    panel_label(ax, "B")
    ax.set_title("Hypothesis decision matrix", loc="left", fontweight="bold")
    hypotheses = ["H1a", "H1b", "H2", "H3a", "H3b"]
    columns = ["K562", "Fetal GATA1", "CTCF"]
    maps = [k_decisions, g_decisions, c_decisions]
    matrix = np.zeros((len(hypotheses), len(columns)))
    labels = np.empty(matrix.shape, dtype=object)
    for i, hyp in enumerate(hypotheses):
        for j, mapping in enumerate(maps):
            status = mapping[hyp]["status"]
            if status == "not_estimable":
                matrix[i, j] = 0
                labels[i, j] = "NE"
            else:
                matrix[i, j] = 1
                labels[i, j] = "✓"
    c_h2 = comparison.loc[(comparison["task_id"] == "p3_ctcf_gm12878") & (comparison["hypothesis"] == "H2")].iloc[0]
    if c_h2["primary_status"] == "reproduced" and c_h2["sensitivity_status"] != "reproduced":
        matrix[2, 2] = 2
        labels[2, 2] = "✓*"
    cmap = mpl.colors.ListedColormap(["#D9DEE5", "#B8E0D2", "#F7DFA0"])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(3), columns)
    ax.set_yticks(range(5), hypotheses)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
    for i in range(5):
        for j in range(3):
            ax.text(j, i, labels[i, j], ha="center", va="center", fontsize=11, fontweight="bold", color="#22303D")
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.text(0, -0.10, "✓ supported in primary analysis", transform=ax.transAxes, fontsize=6.5, color=NEUTRAL)
    ax.text(0, -0.16, "NE = NOT ESTIMABLE (eligible pool=76)", transform=ax.transAxes, fontsize=6.5, color=NEUTRAL)
    ax.text(0, -0.22, "* primary supported; sensitivity 95% CI crosses zero", transform=ax.transAxes, fontsize=6.5, color="#8B6508")

    ax = axes[1, 0]
    panel_label(ax, "C")
    ax.set_title("H2 primary effects", loc="left", fontweight="bold")
    labels_c = ["K562 GATA1", "Fetal GATA1", "GM12878 CTCF"]
    y = np.arange(3)[::-1]
    ax.axvline(0, color=NEUTRAL, linestyle="--", lw=1)
    for yi, label in zip(y, labels_c):
        row = primary_h2.loc[primary_h2["dataset_label"] == label].iloc[0]
        est, lo, hi = row[["estimate", "ci95_low", "ci95_high"]].astype(float)
        ax.errorbar(est, yi, xerr=[[est - lo], [hi - est]], fmt="o", color="#355C7D", ecolor="#355C7D", capsize=3, lw=1.5, markersize=6)
    ax.set_yticks(y, labels_c)
    ax.set_xlabel("CNN-Aug - CNN-Raw mean prediction asymmetry")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.text(0.02, 0.03, "Negative effects favor augmentation", transform=ax.transAxes, fontsize=7, color=NEUTRAL)

    ax = axes[1, 1]
    panel_label(ax, "D")
    ax.set_title("Primary vs near-duplicate sensitivity", loc="left", fontweight="bold")
    tasks = [("p3_gata1_fetal", "Fetal GATA1"), ("p3_ctcf_gm12878", "GM12878 CTCF")]
    base_y = {"p3_gata1_fetal": 1, "p3_ctcf_gm12878": 0}
    ax.axvline(0, color=NEUTRAL, linestyle="--", lw=1)
    for task, label in tasks:
        row = comparison.loc[(comparison["task_id"] == task) & (comparison["hypothesis"] == "H2")].iloc[0]
        for offset, prefix, color, marker, legend_label in [
            (0.10, "primary", "#355C7D", "o", "Phase 3C primary"),
            (-0.10, "sensitivity", "#E08B2C", "s", "Phase 3E sensitivity"),
        ]:
            est = float(row[f"{prefix}_estimate"])
            lo = float(row[f"{prefix}_ci95_low"])
            hi = float(row[f"{prefix}_ci95_high"])
            ax.errorbar(
                est,
                base_y[task] + offset,
                xerr=[[est - lo], [hi - est]],
                fmt=marker,
                color=color,
                ecolor=color,
                capsize=3,
                lw=1.5,
                markersize=5.5,
                label=legend_label if task == tasks[0][0] else None,
            )
    ax.set_yticks([1, 0], ["Fetal GATA1", "GM12878 CTCF"])
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel("CNN-Aug - CNN-Raw mean prediction asymmetry")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.legend(frameon=False, loc="lower left")
    ax.text(
        0.98,
        0.06,
        "CTCF sensitivity CI crosses zero",
        transform=ax.transAxes,
        ha="right",
        fontsize=7.5,
        color="#8B6508",
        fontweight="bold",
    )

    finish_figure(fig, 5, "External replication and robustness of reverse-complement consistency findings")


def provenance_rows(selected_npz: Path) -> list[dict]:
    fields = [
        "figure",
        "panel",
        "source_file",
        "dataset",
        "split",
        "model",
        "metric",
        "filter",
        "aggregation",
        "statistical_method",
    ]

    def row(fig, panel, sources, dataset, split, model, metric, filt, aggregation, method):
        values = [fig, panel, sources, dataset, split, model, metric, filt, aggregation, method]
        return dict(zip(fields, values))

    return [
        row("Figure 1", "A", "conceptual schematic", "study framework", "not applicable", "same trained model", "prediction and RC-aligned attribution", "none", "none", "none"),
        row("Figure 1", "B", "conceptual schematic", "study framework", "not applicable", "not applicable", "three evidence layers", "none", "none", "none"),
        row("Figure 1", "C", "conceptual schematic", "study framework", "not applicable", "CNN-Raw|CNN-Aug|CNN-RCPS", "RC handling strategy", "none", "none", "none"),
        row("Figure 1", "D", "conceptual schematic", "study framework", "not applicable", "not applicable", "evidence progression", "none", "none", "none"),
        row("Figure 2", "A", f"{rel(PROJECT / 'data/phase2/processed/phase2_matched_dataset.csv')}|{rel(P2F / 'resolved_config.yaml')}", "K562 GATA1", "train|validation|frozen test", "not applicable", "sequence length and split counts", "none", "count by split", "descriptive"),
        row("Figure 2", "B", rel(P2D / "training_run_summary.csv"), "K562 GATA1", "Phase 2D chromosome holdout", "CNN-Raw|CNN-Aug|CNN-RCPS", "AUROC", "all 3 folds x 3 seeds", "individual runs plus arithmetic mean", "descriptive"),
        row("Figure 2", "C", f"{rel(P2D / 'training_run_summary.csv')}|{rel(P2D / 'hierarchical_paired_bootstrap.csv')}", "K562 GATA1", "Phase 2D chromosome holdout", "CNN-Raw|CNN-Aug|CNN-RCPS", "mean absolute prediction difference", "all 3 folds x 3 seeds", "individual runs plus arithmetic mean; Aug-Raw contrast", "paired hierarchical bootstrap for contrast"),
        row("Figure 2", "D", rel(P2D / "training_run_summary.csv"), "K562 GATA1", "Phase 2D chromosome holdout", "CNN-Raw|CNN-Aug|CNN-RCPS", "classification flip rate", "all 3 folds x 3 seeds", "individual runs plus arithmetic mean", "descriptive"),
        row("Figure 3", "A", rel(P2F / "test_attribution_results.csv"), "K562 GATA1", "Phase 2F frozen held-out Exact-ISM cohort", "CNN-Raw", "prediction_difference vs absolute_normalized_l1", "model_type=CNN-Raw", "720 sample-run observations (80 unique sequences x 9 fold-seed runs)", "descriptive scatter"),
        row("Figure 3", "B", f"{rel(P2F / 'test_attribution_results.csv')}|{rel(P2F / 'test_prediction_results.csv')}|{rel(selected_npz)}", "K562 GATA1", "Phase 2F frozen held-out Exact-ISM cohort", "CNN-Raw", "forward_matrix and aligned_rc_matrix", "prediction_difference<=0.01; closest to eligible L1 median; ties by sample_id, fold_id, seed", "single deterministic observation; shared color normalization", "prespecified deterministic selection"),
        row("Figure 3", "C", rel(P2D / "attribution_run_summary.csv"), "K562 GATA1", "Phase 2D chromosome holdout", "CNN-Raw|CNN-Aug|CNN-RCPS", "mean Exact ISM absolute_normalized_l1", "method=exact_ism; all 3 folds x 3 seeds", "individual runs plus arithmetic mean", "descriptive"),
        row("Figure 3", "D", rel(P2F / "test_confirmatory_bootstrap.csv"), "K562 GATA1", "Phase 2F frozen held-out Exact-ISM cohort", "CNN-Raw", "absolute_normalized_l1", "H1b; prediction_difference<=0.01", "estimate and 95% CI", "frozen hierarchical bootstrap"),
        row("Figure 4", "A", "conceptual schematic", "K562 GATA1", "Phase 2E", "not applicable", "motif localization and in silico disruption", "none", "none", "conceptual"),
        row("Figure 4", "B", rel(P2E / "motif_localization_results.csv"), "K562 GATA1", "Phase 2E confirmatory", "CNN-Raw|CNN-Aug|CNN-RCPS", "motif_mass_fraction", "strong_hit_080=True", "eligible sample-run distribution; median, IQR, and arithmetic mean", "descriptive"),
        row("Figure 4", "C", rel(P2E / "confirmatory_bootstrap_results.csv"), "K562 GATA1", "Phase 2E confirmatory", "CNN-Raw|CNN-Aug|CNN-RCPS", "motif_minus_flank_logit_drop_vs_zero", "confirmatory endpoint", "estimate and 95% CI", "paired cluster-aware bootstrap"),
        row("Figure 4", "D", "conceptual schematic", "study interpretation", "not applicable", "not applicable", "RC consistency vs biological validity", "none", "none", "conceptual"),
        row("Figure 5", "A", "conceptual schematic", "K562 GATA1|Fetal GATA1|GM12878 CTCF", "Phase 2F|Phase 3C|Phase 3E", "not applicable", "replication sequence", "none", "none", "conceptual"),
        row("Figure 5", "B", f"{rel(P2F / 'test_hypothesis_decisions.json')}|{rel(P3C / 'p3_gata1_fetal/test_hypothesis_decisions.json')}|{rel(P3C / 'p3_ctcf_gm12878/test_hypothesis_decisions.json')}|{rel(P3E / 'report_tables/hypothesis_comparison.csv')}", "K562 GATA1|Fetal GATA1|GM12878 CTCF", "Phase 2F frozen test|Phase 3C primary|Phase 3E sensitivity qualifier", "CNN-Raw|CNN-Aug|CNN-RCPS", "H1a/H1b/H2/H3a/H3b decisions", "exclude H3c exploratory", "frozen decision status", "frozen decision rules"),
        row("Figure 5", "C", f"{rel(P2F / 'test_confirmatory_bootstrap.csv')}|{rel(P3C / 'p3_gata1_fetal/test_confirmatory_bootstrap.csv')}|{rel(P3C / 'p3_ctcf_gm12878/test_confirmatory_bootstrap.csv')}", "K562 GATA1|Fetal GATA1|GM12878 CTCF", "Phase 2F frozen test|Phase 3C primary", "CNN-Aug minus CNN-Raw", "H2 prediction asymmetry contrast", "hypothesis=H2", "estimate and 95% CI", "task-specific frozen bootstrap"),
        row("Figure 5", "D", rel(P3E / "report_tables/hypothesis_comparison.csv"), "Fetal GATA1|GM12878 CTCF", "Phase 3C primary vs Phase 3E near-duplicate sensitivity", "CNN-Aug minus CNN-Raw", "H2 prediction asymmetry contrast", "hypothesis=H2", "primary and sensitivity estimates with 95% CI", "task-specific frozen bootstrap; paired presentation only"),
    ]


def write_captions(selection: pd.Series, eligible_n: int, median_l1: float) -> None:
    captions = {
        1: (
            "Figure 1. Experimental framework for separating prediction consistency, attribution consistency, and biological validity. "
            "(A) Each sequence and its reverse complement are evaluated by the same trained model; predictions and RC-aligned attributions are compared separately. "
            "(B) Prediction consistency, attribution consistency, and biological validity constitute distinct evidence layers. "
            "(C) CNN-Raw, CNN-Aug, and CNN-RCPS represent no RC constraint, stochastic RC augmentation, and architectural RC parameter sharing, respectively. "
            "(D) Evidence progresses from synthetic tasks through K562 confirmation and frozen testing, then branches into parallel fetal erythroid GATA1 and GM12878 CTCF external settings before cross-stage overlap audit and near-duplicate sensitivity analysis. Phase 3C primary external replication remains distinct from Phase 3E sensitivity analysis. This schematic contains no simulated quantitative data."
        ),
        2: (
            "Figure 2. Reverse-complement augmentation improves prediction consistency without enforcing exact symmetry. "
            "(A) The single 256-bp matched K562 GATA1 dataset (N=23,310) was partitioned into a train split used as the Phase 2D chromosome-CV source (N=16,318), a reserved validation split (N=3,496), and a frozen held-out Phase 2F test split (N=3,496; chr3, chr8, chr14, and chr18). "
            "(B-D) Phase 2D fold-by-seed observations for AUROC, mean prediction asymmetry, and classification flip rate; diamonds denote arithmetic means. "
            "The paired hierarchical bootstrap contrast in panel C quantifies CNN-Aug minus CNN-Raw prediction asymmetry. CNN-RCPS achieves numerical prediction symmetry."
        ),
        3: (
            "Figure 3. Prediction consistency does not imply attribution consistency. "
            "(A) Per-observation prediction asymmetry and Exact ISM normalized L1 for CNN-Raw in the frozen Phase 2F Exact-ISM cohort (720 sample-run observations: 80 unique sequences across nine fold-seed runs), not all 3,496 test sequences. "
            f"The vertical line marks the frozen prediction-consistency threshold. (B) Two 256-by-4 Exact ISM maps for the deterministically selected observation. Among CNN-Raw observations with Δp<=0.01 (eligible n={eligible_n}), the observation with Exact ISM L1 closest to the eligible median ({median_l1:.4f}) was selected, with ties ordered by sample ID, fold, and seed: sample {selection['sample_id']}, {selection['fold_id']}, seed {int(selection['seed'])}, p(S)={float(selection['p_forward']):.8f}, p(RC(S))={float(selection['p_rc']):.8f}, Δp={float(selection['prediction_difference']):.4f}, and Exact ISM L1={float(selection['absolute_normalized_l1']):.4f}. "
            "Both heatmaps use identical position axes, A/C/G/T channel order, and color normalization. "
            "(C) Phase 2D Exact ISM normalized L1 across fold-by-seed runs on a linear scale; CNN-RCPS values are near numerical zero. "
            "(D) Frozen Phase 2F CNN-Raw H1b estimate and 95% CI among prediction-consistent observations. Near-identical predictions can coexist with substantial attribution asymmetry."
        ),
        4: (
            "Figure 4. Reverse-complement symmetry and biological feature validity require separate evaluation. "
            "(A) Schematic of GATA1 motif localization and matched in silico motif-versus-flank disruption. "
            "(B) Phase 2E motif attribution mass among strong PWM hits; distributions show eligible sample-run observations, with median, interquartile range, and mean. "
            "(C) Motif-minus-flank logit-drop estimates and 95% confidence intervals from paired cluster-aware bootstrap analyses. "
            "(D) RC consistency and biological feature validity are independent evaluation axes that must be measured separately. Perturbations are computational, not wet-lab experiments."
        ),
        5: (
            "Figure 5. External replication and robustness of reverse-complement consistency findings. "
            "(A) The frozen K562 result is evaluated in two parallel external settings: fetal erythroid GATA1 (same TF, different context) and GM12878 CTCF (different TF, different context). Primary Phase 3C external replication is followed by cross-stage overlap audit and a distinct Phase 3E near-duplicate sensitivity analysis. "
            "(B) Frozen hypothesis decision matrix. Fetal GATA1 H1b is NOT ESTIMABLE (NE; eligible pool=76). CTCF H2 is supported in the Phase 3C primary analysis but sensitivity-qualified. "
            "(C) Task-specific H2 effects (CNN-Aug minus CNN-Raw prediction asymmetry) and 95% CIs for frozen K562, fetal GATA1, and CTCF primary analyses. "
            "(D) Phase 3C primary and Phase 3E near-duplicate sensitivity estimates are shown separately; the CTCF sensitivity CI crosses zero."
        ),
    }
    for number, text in captions.items():
        (CAPTIONS / f"figure{number}_caption.txt").write_text(text + "\n", encoding="utf-8")


def write_summary(
    train: pd.DataFrame,
    h2_p2d: pd.Series,
    raw_attr: pd.DataFrame,
    eligible: pd.DataFrame,
    selected: pd.Series,
    median_l1: float,
    npz_path: Path,
    p2f_h1b: pd.Series,
    comparison: pd.DataFrame,
) -> None:
    perf = train.groupby("model_type", sort=False)[["holdout_auroc", "prediction_mean_absolute_difference", "symmetry_flip_rate"]].mean().reindex(MODEL_ORDER)
    c_h2 = comparison.loc[(comparison["task_id"] == "p3_ctcf_gm12878") & (comparison["hypothesis"] == "H2")].iloc[0]
    lines = [
        "# Main Figure results summary",
        "",
        "All quantitative values below were read or aggregated from frozen source files at figure-generation time; prompt reference values were not used as plotting data.",
        "",
        "## Figure 2",
        "",
    ]
    for model, row in perf.iterrows():
        lines.append(
            f"- {model}: mean AUROC={row['holdout_auroc']:.6f}; mean prediction asymmetry={row['prediction_mean_absolute_difference']:.6f}; mean flip rate={row['symmetry_flip_rate']:.6f}."
        )
    lines += [
        f"- P2D CNN-Aug minus CNN-Raw prediction asymmetry={float(h2_p2d['estimate']):.6f}, 95% CI [{float(h2_p2d['ci95_low']):.6f}, {float(h2_p2d['ci95_high']):.6f}].",
        "",
        "## Figure 3",
        "",
        f"- Figure 3A rows={len(raw_attr)}; unique sample IDs={raw_attr['sample_id'].nunique()}.",
        f"- Figure 3B eligible subset rows={len(eligible)}; eligible L1 median={median_l1:.12g}.",
        f"- Selected sample_id={selected['sample_id']}; fold={selected['fold_id']}; seed={int(selected['seed'])}; p(S)={float(selected['p_forward']):.12g}; p(RC(S))={float(selected['p_rc']):.12g}; delta_p={float(selected['prediction_difference']):.12g}; Exact ISM L1={float(selected['absolute_normalized_l1']):.12g}.",
        f"- Selected NPZ source: `{rel(npz_path)}`.",
        f"- P2F CNN-Raw prediction-consistent Exact ISM L1={float(p2f_h1b['estimate']):.12g}, 95% CI [{float(p2f_h1b['ci95_low']):.12g}, {float(p2f_h1b['ci95_high']):.12g}].",
        "",
        "## Figure 5",
        "",
        "- Fetal GATA1 H1b is displayed as NOT ESTIMABLE (NE), not failed.",
        f"- CTCF H2 sensitivity effect={float(c_h2['sensitivity_estimate']):.12g}, 95% CI [{float(c_h2['sensitivity_ci95_low']):.12g}, {float(c_h2['sensitivity_ci95_high']):.12g}]; the interval crosses zero.",
        "- Phase 3C primary external replication and Phase 3E sensitivity are kept separate.",
        "",
        "## Discrepancy check",
        "",
        "- No new discrepancy detected while assembling Main Figures 1-5.",
    ]
    (VALIDATION / "figure_results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_style()
    for directory in [DATA, CAPTIONS, VALIDATION, *(MAIN / f"figure{i}" for i in range(1, 6))]:
        directory.mkdir(parents=True, exist_ok=True)

    train = read_csv(P2D / "training_run_summary.csv")
    attr_runs = read_csv(P2D / "attribution_run_summary.csv")
    p2d_boot = read_csv(P2D / "hierarchical_paired_bootstrap.csv")
    h2_p2d = p2d_boot.loc[
        (p2d_boot["family"] == "prediction_primary")
        & (p2d_boot["contrast"] == "CNN-Aug_minus_CNN-Raw")
        & (p2d_boot["endpoint"] == "prediction_difference")
    ].iloc[0]

    dataset = read_csv(PROJECT / "data" / "phase2" / "processed" / "phase2_matched_dataset.csv", usecols=["split", "chromosome"])
    dataset_counts = dataset.groupby("split").size()
    test_chromosomes = sorted(dataset.loc[dataset["split"] == "test", "chromosome"].unique(), key=lambda x: int(x.replace("chr", "")))
    if int(dataset_counts.sum()) != 23310 or int(dataset_counts["test"]) != 3496:
        raise ValueError("K562 dataset counts conflict with frozen validated design")

    threshold = read_threshold(P2F / "resolved_config.yaml")
    raw_attr, eligible, selected, median_l1, npz_path, forward, aligned = select_figure3b(threshold)
    p2f_boot = read_csv(P2F / "test_confirmatory_bootstrap.csv")
    p2f_h1b = p2f_boot.loc[p2f_boot["hypothesis"] == "H1b"].iloc[0]

    localization = read_csv(P2E / "motif_localization_results.csv")
    strong = localization.loc[localization["strong_hit_080"].astype(str).str.lower() == "true"].copy()
    if strong.groupby("model_type").size().to_dict() != {"CNN-Aug": 324, "CNN-RCPS": 324, "CNN-Raw": 324}:
        raise ValueError("Phase 2E strong-hit counts disagree with frozen summary")
    disruption_boot = read_csv(P2E / "confirmatory_bootstrap_results.csv")

    k_decisions = decision_map(P2F / "test_hypothesis_decisions.json")
    g_decisions = decision_map(P3C / "p3_gata1_fetal" / "test_hypothesis_decisions.json")
    c_decisions = decision_map(P3C / "p3_ctcf_gm12878" / "test_hypothesis_decisions.json")
    if g_decisions["H1b"]["status"] != "not_estimable" or "76" not in g_decisions["H1b"]["evidence"]:
        raise ValueError("Fetal GATA1 H1b frozen status changed")

    h2_rows = []
    for label, path in [
        ("K562 GATA1", P2F / "test_confirmatory_bootstrap.csv"),
        ("Fetal GATA1", P3C / "p3_gata1_fetal" / "test_confirmatory_bootstrap.csv"),
        ("GM12878 CTCF", P3C / "p3_ctcf_gm12878" / "test_confirmatory_bootstrap.csv"),
    ]:
        frame = read_csv(path)
        h2 = frame.loc[frame["hypothesis"] == "H2"].iloc[0].copy()
        h2["dataset_label"] = label
        h2_rows.append(h2)
    primary_h2 = pd.DataFrame(h2_rows)
    comparison = read_csv(P3E / "report_tables" / "hypothesis_comparison.csv")
    c_sens = comparison.loc[(comparison["task_id"] == "p3_ctcf_gm12878") & (comparison["hypothesis"] == "H2")].iloc[0]
    if not (float(c_sens["sensitivity_ci95_low"]) < 0 < float(c_sens["sensitivity_ci95_high"])):
        raise ValueError("CTCF H2 sensitivity interval no longer crosses zero")

    build_figure1()
    build_figure2(train, dataset_counts, test_chromosomes, h2_p2d)
    build_figure3(raw_attr, eligible, selected, median_l1, npz_path, forward, aligned, attr_runs, p2f_h1b, threshold)
    build_figure4(strong, disruption_boot)
    build_figure5(k_decisions, g_decisions, c_decisions, primary_h2, comparison)

    provenance = pd.DataFrame(provenance_rows(npz_path))
    revision_notes = {
        ("Figure 1", "D"): "parallel external-replication branches; overlap audit and sensitivity separated",
        ("Figure 2", "A"): "single-dataset partition structure clarified from frozen split/config",
        ("Figure 2", "C"): "contrast annotation moved to an empty plotting region",
        ("Figure 3", "B"): "full frozen sample metadata moved from panel to caption",
        ("Figure 3", "C"): "linear-scale RCPS near-numerical-zero label clarified",
        ("Figure 3", "D"): "interpretive conclusion moved from panel to caption",
        ("Figure 4", "D"): "orthogonal-axis schematic text reduced",
        ("Figure 5", "A"): "external datasets redrawn as parallel settings",
    }
    provenance["revision_note"] = [
        revision_notes.get(
            (row.figure, row.panel),
            "figure-level title removed; typography, panel labels, and margins standardized",
        )
        for row in provenance.itertuples(index=False)
    ]
    provenance["quantitative_source_change"] = "none"
    provenance.to_csv(VALIDATION / "figure_data_provenance.csv", index=False, encoding="utf-8")
    write_captions(selected, len(eligible), median_l1)
    write_summary(train, h2_p2d, raw_attr, eligible, selected, median_l1, npz_path, p2f_h1b, comparison)

    manifest = {
        "figures": {f"figure{i}": [rel(MAIN / f"figure{i}" / f"figure{i}.{ext}") for ext in ("svg", "pdf", "png")] for i in range(1, 6)},
        "figure3a_rows": int(len(raw_attr)),
        "figure3a_unique_samples": int(raw_attr["sample_id"].nunique()),
        "figure3b": {
            "eligible_n": int(len(eligible)),
            "median_l1": median_l1,
            "sample_id": str(selected["sample_id"]),
            "fold_id": str(selected["fold_id"]),
            "seed": int(selected["seed"]),
            "p_forward": float(selected["p_forward"]),
            "p_rc": float(selected["p_rc"]),
            "prediction_difference": float(selected["prediction_difference"]),
            "absolute_normalized_l1": float(selected["absolute_normalized_l1"]),
            "npz_path": rel(npz_path),
        },
        "fetal_h1b_display": "NE / NOT ESTIMABLE",
        "ctcf_h2_sensitivity_ci_crosses_zero": True,
        "new_discrepancy": False,
    }
    (VALIDATION / "main_figure_build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
