from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / ".matplotlib_phase15"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from scipy.stats import wilcoxon
from sklearn.metrics import roc_curve

from src.data import save_dataset
from src.dna_utils import align_rc_full_attribution, align_rc_position_attribution, reverse_complement
from src.interpret import run_ism_for_sequences
from src.metrics import (attribution_consistency, interval_localization,
                         motif_localization, prediction_consistency_metrics,
                         safe_similarity)
from src.near_motif import annotate_near_motifs
from src.phase15_data import generate_phase15_data
from src.phase15_statistics import build_full_statistics, benjamini_hochberg
from src.training import predict_sequences, train_model

sns.set_theme(style="whitegrid")


def _json_dump(value, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=True), encoding="utf-8")


def _region_metrics(scores: np.ndarray, start, end, top_k: int) -> dict:
    if pd.isna(start) or pd.isna(end):
        return {"top8_overlap": np.nan, "mass_fraction": np.nan,
                "vs_background": np.nan, "position_auprc": np.nan}
    return motif_localization(scores, int(start), int(end), top_k)


def _interval_metrics(scores: np.ndarray, encoded_intervals, top_k: int) -> dict:
    if pd.isna(encoded_intervals):
        return _region_metrics(scores, np.nan, np.nan, top_k)
    intervals = json.loads(encoded_intervals) if isinstance(encoded_intervals, str) else encoded_intervals
    return interval_localization(scores, intervals, top_k)


def _paired_wilcoxon(a: pd.Series, b: pd.Series, alternative: str) -> float:
    left, right = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 2 or np.allclose(left[mask], right[mask]):
        return np.nan
    return float(wilcoxon(left[mask], right[mask], alternative=alternative).pvalue)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Small dependency-free Markdown table formatter."""
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append("NA" if not np.isfinite(value) else f"{value:.6g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _create_figures(sample_df: pd.DataFrame, run_df: pd.DataFrame,
                    prediction_df: pd.DataFrame, arrays: dict, output_dir: Path,
                    stage: str) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    order = list(dict.fromkeys(run_df.model_type.tolist()))
    colors = sns.color_palette("colorblind", len(order))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    performance = run_df.melt(
        id_vars="model_type", value_vars=["validation_auroc", "test_auroc", "test_auprc"],
        var_name="metric", value_name="value"
    )
    sns.barplot(performance, x="model_type", y="value", hue="metric", order=order, ax=axes[0, 0])
    axes[0, 0].set(title="Classification: validation vs shifted test", ylim=(0, 1), xlabel="")
    axes[0, 0].tick_params(axis="x", rotation=25)

    sns.boxplot(sample_df, x="model_type", y="prediction_difference", order=order, ax=axes[0, 1])
    axes[0, 1].set(title="Prediction inconsistency", ylabel="|p(S)-p(RC(S))|", xlabel="")
    axes[0, 1].tick_params(axis="x", rotation=25)

    sns.boxplot(sample_df, x="model_type", y="attribution_pearson_absolute", order=order, ax=axes[0, 2])
    axes[0, 2].axhline(.9, color="grey", linestyle="--", linewidth=1)
    axes[0, 2].set(title="Aligned ISM consistency", ylabel="Pearson (absolute ISM)", xlabel="", ylim=(-.1, 1.05))
    axes[0, 2].tick_params(axis="x", rotation=25)

    sns.barplot(sample_df, x="model_type", y="causal_mass_fraction", order=order, errorbar="sd", ax=axes[1, 0])
    axes[1, 0].set(title="Attribution mass in causal motif", ylabel="Mass fraction", xlabel="")
    axes[1, 0].tick_params(axis="x", rotation=25)

    region_columns = ["causal_mass_fraction", "shortcut_mass_fraction"]
    if "near_motif_mass_fraction" in sample_df:
        region_columns.insert(1, "near_motif_mass_fraction")
    region = sample_df.melt(
        id_vars="model_type", value_vars=region_columns,
        var_name="region", value_name="mass_fraction"
    ).dropna()
    sns.barplot(region, x="model_type", y="mass_fraction", hue="region", order=order, errorbar="sd", ax=axes[1, 1])
    axes[1, 1].set(title="Causal evidence vs shortcut evidence", ylabel="Mass fraction", xlabel="")
    axes[1, 1].tick_params(axis="x", rotation=25)

    consistency = sample_df.assign(
        pred_consistent=sample_df.prediction_difference <= sample_df.prediction_consistent_epsilon,
        attr_asymmetric=sample_df.attribution_pearson_absolute < sample_df.attribution_asymmetric_pearson,
    )
    counts = consistency.groupby("model_type").apply(
        lambda group: float((group.pred_consistent & group.attr_asymmetric).mean()),
        include_groups=False,
    ).reindex(order)
    axes[1, 2].bar(order, counts.values, color=colors)
    axes[1, 2].set(title="H1b/H3a joint event", ylabel="Fraction: prediction-consistent but attribution-asymmetric", ylim=(0, 1))
    axes[1, 2].tick_params(axis="x", rotation=25)

    fig.suptitle(f"Phase 1.5 {stage} dashboard", fontsize=16)
    fig.tight_layout()
    fig.savefig(figure_dir / "phase15_dashboard.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for color, model_type in zip(colors, order):
        group = prediction_df[prediction_df.model_type == model_type]
        fpr, tpr, _ = roc_curve(group.clean_label, group.p_forward)
        auc_value = run_df[run_df.model_type == model_type].test_auroc.mean()
        ax.plot(fpr, tpr, label=f"{model_type} (mean AUC={auc_value:.3f})", color=color)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Shifted test ROC")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figure_dir / "phase15_test_roc.png", dpi=180); plt.close(fig)

    representative_seed = int(sorted(sample_df.seed.unique())[0])
    raw = sample_df[(sample_df.model_type == "CNN-Raw") & (sample_df.seed == representative_seed)]
    if len(raw):
        sample_id = raw.sort_values("prediction_difference", ascending=False).iloc[0].sample_id
        rows = sample_df[(sample_df.sample_id == sample_id) &
                         (sample_df.seed == representative_seed)].set_index("model_type")
        fig, axes = plt.subplots(len(order), 1, figsize=(14, 2.3 * len(order)), sharex=True)
        for ax, model_type, color in zip(np.atleast_1d(axes), order, colors):
            forward, aligned = arrays[(model_type, representative_seed, sample_id)]
            x = np.arange(len(forward))
            ax.plot(x, forward, color=color, label="ISM on S")
            ax.plot(x, aligned, color="black", linestyle="--", alpha=.7, label="aligned ISM on RC(S)")
            row = rows.loc[model_type]
            for interval_index, (start, end) in enumerate(json.loads(row.causal_intervals)):
                ax.axvspan(start, end - 1, color="limegreen", alpha=.18,
                           label="causal motif" if interval_index == 0 else None)
            for interval_index, (start, end) in enumerate(json.loads(row.decoy_intervals)):
                ax.axvspan(start, end - 1, color="orange", alpha=.15,
                           label="decoy" if interval_index == 0 else None)
            for interval_index, (start, end) in enumerate(json.loads(row.shortcut_intervals)):
                ax.axvspan(start, end - 1, color="red", alpha=.14,
                           label="shortcut" if interval_index == 0 else None)
            near_intervals = json.loads(getattr(row, "incidental_near_motif_intervals", "[]"))
            for interval_index, (start, end) in enumerate(near_intervals):
                ax.axvspan(start, end - 1, color="mediumpurple", alpha=.16,
                           label="incidental near-motif" if interval_index == 0 else None)
            absent = []
            if pd.isna(row.decoy_start): absent.append("decoy absent")
            if pd.isna(row.shortcut_start): absent.append("shortcut absent")
            if absent:
                ax.text(.995, .06, "; ".join(absent), transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=7, color="dimgray")
            ax.set_ylabel(model_type); ax.legend(loc="upper right", ncol=4, fontsize=7)
        axes[-1].set_xlabel("Sequence position")
        fig.suptitle(f"Representative sample {sample_id}, seed {representative_seed}: RC-aligned absolute ISM")
        fig.tight_layout(); fig.savefig(figure_dir / "phase15_representative_ism.png", dpi=180); plt.close(fig)


def _write_report(config: dict, sample_df: pd.DataFrame, run_df: pd.DataFrame,
                  tests: pd.DataFrame, output_dir: Path, elapsed: float) -> None:
    eps = float(config["interpretation"]["prediction_consistent_epsilon"])
    attr_cut = float(config["interpretation"]["attribution_asymmetric_pearson"])
    stage = str(config.get("experiment_stage", "pilot"))
    run_summary = run_df.groupby("model_type").agg(
        validation_auroc=("validation_auroc", "mean"),
        test_auroc=("test_auroc", "mean"),
    ).reset_index()
    summary_spec = dict(
        pred_MAD=("prediction_difference", "mean"),
        pred_max=("prediction_difference", "max"),
        attr_Pearson=("attribution_pearson_absolute", "mean"),
        matrix_error_max=("attribution_matrix_max_abs_error", "max"),
        causal_mass=("causal_mass_fraction", "mean"),
        shortcut_mass=("shortcut_mass_fraction", "mean"),
        H1b_count=("prediction_consistent_but_attr_asymmetric", "sum"),
    )
    if "near_motif_mass_fraction" in sample_df:
        summary_spec["near_motif_mass"] = ("near_motif_mass_fraction", "mean")
    summary = sample_df.groupby("model_type").agg(**summary_spec).reset_index().merge(run_summary, on="model_type")
    table = _markdown_table(summary)
    strict = summary[summary.model_type.isin(["CNN-PostHoc", "CNN-RCPS"])]
    strict_lines = [
        f"- {row.model_type}: max prediction difference={row.pred_max:.3g}; "
        f"max aligned full-ISM error={row.matrix_error_max:.3g}."
        for row in strict.itertuples()
    ]
    report = f"""# Phase 1.5 {stage} report

Regime: `{config['data']['regime']}`  
Experiment stage: `{stage}`; seeds: {config['training']['seeds']}.  
Prediction-consistent threshold: |p(S)-p(RC(S))| ≤ {eps}.  
Attribution-asymmetric threshold: aligned absolute-ISM Pearson < {attr_cut}.

## Main results

{table}

## Exact-symmetry checks

{chr(10).join(strict_lines)}

## Hypothesis-directed reading

- H1a: read `pred_MAD`, `pred_max`, symmetry flips, and the paired comparisons against CNN-Raw.
- H1b/H3a: `H1b_count` counts samples whose predictions pass the consistency threshold while their attribution maps fail the attribution threshold.
- H2: compare CNN-Aug/CNN-Pair against CNN-Raw using the paired prediction-difference tests.
- H3b: RCPS should be at floating-point tolerance for both prediction and aligned full ISM.
- H4: compare causal attribution mass with shortcut mass and compare validation AUROC with shifted-test AUROC. High symmetry alone is not biological correctness.

## Paired tests

{_markdown_table(tests)}

Elapsed time: {elapsed:.1f} seconds.
"""
    (output_dir / "PHASE15_REPORT.md").write_text(report, encoding="utf-8")


def run(config_path: Path, output_dir: Path | None = None) -> Path:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("device") != "cpu":
        raise ValueError("Phase 1.5 currently requires device: cpu")
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    stage = str(config.get("experiment_stage", "pilot"))
    output_dir = output_dir or ROOT / "results" / config["project_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "resolved_config.yaml")

    data, report = generate_phase15_data(config["data"])
    near_cfg = config.get("interpretation", {}).get("near_motif_control", {})
    if near_cfg.get("enabled", False):
        motifs = near_cfg.get("motifs") or [config["data"]["causal_motif"]]
        data, near_report = annotate_near_motifs(
            data, motifs, int(near_cfg.get("max_hamming", 2))
        )
        report["incidental_near_motif_control"] = near_report
    save_dataset(data, output_dir / "phase15_dataset.csv")
    _json_dump({"data_config": config["data"], "leakage_report": report}, output_dir / "data_generation_report.json")
    test = data[data.split == "test"].reset_index(drop=True)
    positives = test[test.clean_label == 1]
    n_ism = min(int(config["interpretation"]["positive_samples"]), len(positives))
    explained = positives.sample(n=n_ism, random_state=int(config["interpretation"]["sample_seed"])).sort_values("sample_id").reset_index(drop=True)
    explained[["sample_id"]].to_csv(output_dir / "ism_sample_ids.csv", index=False)

    sample_rows: list[dict] = []
    prediction_rows: list[dict] = []
    run_rows: list[dict] = []
    arrays: dict = {}
    eps = float(config["interpretation"]["prediction_consistent_epsilon"])
    attr_cut = float(config["interpretation"]["attribution_asymmetric_pearson"])
    top_k = int(config["interpretation"]["top_k"])

    for seed in config["training"]["seeds"]:
        for model_type in config["training"]["model_types"]:
            print(f"Training {model_type}, seed={seed}", flush=True)
            run_dir = output_dir / "runs" / f"{model_type.lower().replace('-', '_')}_seed_{seed}"
            model, metadata = train_model(model_type, int(seed), config, data, run_dir)
            sequences = test.sequence.tolist()
            rc_sequences = [reverse_complement(sequence) for sequence in sequences]
            pf = predict_sequences(model, sequences, int(config["training"]["batch_size"]))
            pr = predict_sequences(model, rc_sequences, int(config["training"]["batch_size"]))
            pred_summary, pred_each = prediction_consistency_metrics(pf, pr, float(config["training"]["threshold"]))
            for index, row in test.iterrows():
                prediction_rows.append({
                    "model_type": model_type, "seed": seed, "sample_id": row.sample_id,
                    "label": row.label, "clean_label": row.clean_label,
                    "has_shortcut": int(pd.notna(row.shortcut_start)),
                    "p_forward": pf[index], "p_rc": pr[index],
                    **{key: value[index] for key, value in pred_each.items()},
                })

            exp_sequences = explained.sequence.tolist()
            exp_rc = [reverse_complement(sequence) for sequence in exp_sequences]
            ism_batch = int(config["interpretation"]["batch_size"])
            difference = config["interpretation"]["difference"]
            fm, fs, fa, f_time = run_ism_for_sequences(model, exp_sequences, ism_batch, difference, f"{model_type} S")
            rm, rs, ra, r_time = run_ism_for_sequences(model, exp_rc, ism_batch, difference, f"{model_type} RC")
            am = align_rc_full_attribution(rm)
            ass = align_rc_position_attribution(rs)
            aa = align_rc_position_attribution(ra)
            np.savez_compressed(
                run_dir / "phase15_ism_attributions.npz",
                sample_ids=explained.sample_id.astype(str).to_numpy(dtype=str),
                forward_matrix=fm, aligned_rc_matrix=am,
                forward_signed=fs, aligned_rc_signed=ass,
                forward_absolute=fa, aligned_rc_absolute=aa,
            )
            test_index = {sample_id: index for index, sample_id in enumerate(test.sample_id)}
            for local_index, row in explained.iterrows():
                absolute_consistency, _ = attribution_consistency(fa[local_index], aa[local_index], top_k)
                signed_consistency, _ = attribution_consistency(fs[local_index], ass[local_index], top_k)
                full_pearson, _ = safe_similarity(fm[local_index], am[local_index], "pearson")
                causal = _interval_metrics(fa[local_index], row.causal_intervals, top_k)
                decoy = _interval_metrics(fa[local_index], row.decoy_intervals, top_k)
                shortcut_metrics = _interval_metrics(fa[local_index], row.shortcut_intervals, top_k)
                near_intervals = getattr(row, "incidental_near_motif_intervals", "[]")
                near_metrics = _interval_metrics(fa[local_index], near_intervals, top_k)
                ix = test_index[row.sample_id]
                pred_diff = float(pred_each["prediction_difference"][ix])
                attr_pearson = float(absolute_consistency["pearson"])
                sample_rows.append({
                    "model_type": model_type, "seed": seed, "sample_id": row.sample_id,
                    "sequence": row.sequence, "label": row.label, "clean_label": row.clean_label,
                    "causal_start": row.causal_start, "causal_end": row.causal_end,
                    "causal_intervals": row.causal_intervals,
                    "causal_orientation": row.causal_orientation, "causal_mutations": row.causal_mutations,
                    "decoy_start": row.decoy_start, "decoy_end": row.decoy_end,
                    "decoy_intervals": row.decoy_intervals,
                    "shortcut_start": row.shortcut_start, "shortcut_end": row.shortcut_end,
                    "shortcut_intervals": row.shortcut_intervals,
                    "incidental_near_motif_intervals": near_intervals,
                    "p_forward": pf[ix], "p_rc": pr[ix],
                    **{key: value[ix] for key, value in pred_each.items()},
                    "prediction_consistent_epsilon": eps,
                    "attribution_asymmetric_pearson": attr_cut,
                    "attribution_pearson_absolute": attr_pearson,
                    "attribution_spearman_absolute": absolute_consistency["spearman"],
                    "attribution_cosine_absolute": absolute_consistency["cosine"],
                    "attribution_top8_overlap": absolute_consistency["top8_overlap"],
                    "attribution_pearson_signed": signed_consistency["pearson"],
                    "attribution_full_matrix_pearson": full_pearson,
                    "attribution_matrix_max_abs_error": float(np.max(np.abs(fm[local_index] - am[local_index]))),
                    "prediction_consistent_but_attr_asymmetric": int(pred_diff <= eps and attr_pearson < attr_cut),
                    **{f"causal_{key}": value for key, value in causal.items()},
                    **{f"decoy_{key}": value for key, value in decoy.items()},
                    **{f"shortcut_{key}": value for key, value in shortcut_metrics.items()},
                    **{f"near_motif_{key}": value for key, value in near_metrics.items()},
                })
                arrays[(model_type, int(seed), row.sample_id)] = (fa[local_index], aa[local_index])
            run_rows.append({
                "model_type": model_type, "seed": seed, "best_epoch": metadata["best_epoch"],
                "parameter_count": metadata["parameter_count"], "training_seconds": metadata["training_seconds"],
                "ism_seconds": f_time + r_time,
                **{f"validation_{key}": value for key, value in metadata["validation_metrics"].items()},
                **{f"test_{key}": value for key, value in metadata["test_metrics"].items()},
                **pred_summary,
            })

    sample_df = pd.DataFrame(sample_rows)
    prediction_df = pd.DataFrame(prediction_rows)
    run_df = pd.DataFrame(run_rows)
    sample_df.to_csv(output_dir / "phase15_per_sample_results.csv", index=False)
    prediction_df.to_csv(output_dir / "phase15_all_test_prediction_pairs.csv", index=False)
    run_df.to_csv(output_dir / "phase15_run_summary.csv", index=False)

    pivot_pred = prediction_df.pivot(index=["seed", "sample_id"], columns="model_type", values="prediction_difference")
    pivot_attr = sample_df.pivot(index=["seed", "sample_id"], columns="model_type", values="attribution_pearson_absolute")
    test_rows = []
    for model_type in [item for item in config["training"]["model_types"] if item != "CNN-Raw"]:
        test_rows.append({
            "comparison": f"{model_type} vs CNN-Raw prediction difference",
            "alternative": "model lower than raw",
            "paired_wilcoxon_p": _paired_wilcoxon(pivot_pred[model_type], pivot_pred["CNN-Raw"], "less"),
        })
        test_rows.append({
            "comparison": f"{model_type} vs CNN-Raw attribution Pearson",
            "alternative": "model greater than raw",
            "paired_wilcoxon_p": _paired_wilcoxon(pivot_attr[model_type], pivot_attr["CNN-Raw"], "greater"),
        })
    tests = pd.DataFrame(test_rows)
    tests["fdr_bh"] = benjamini_hochberg(tests["paired_wilcoxon_p"])
    tests.to_csv(output_dir / "phase15_paired_tests.csv", index=False)
    if stage == "full":
        stats_cfg = config.get("statistics", {})
        run_crossseed, sample_seed, sample_crossseed, bootstrap = build_full_statistics(
            prediction_df, sample_df, run_df, list(config["training"]["model_types"]),
            int(stats_cfg.get("bootstrap_replicates", 2000)),
            int(stats_cfg.get("bootstrap_seed", 5150)),
        )
        run_crossseed.to_csv(output_dir / "phase15_crossseed_run_summary.csv", index=False)
        sample_seed.to_csv(output_dir / "phase15_ism_summary_by_seed.csv", index=False)
        sample_crossseed.to_csv(output_dir / "phase15_crossseed_ism_summary.csv", index=False)
        bootstrap.to_csv(output_dir / "phase15_hierarchical_bootstrap.csv", index=False)
    _create_figures(sample_df, run_df, prediction_df, arrays, output_dir, stage)
    elapsed = time.perf_counter() - started
    _write_report(config, sample_df, run_df, tests, output_dir, elapsed)
    _json_dump({"elapsed_seconds": elapsed, "stage": stage,
                "seeds": config["training"]["seeds"]}, output_dir / "phase15_runtime.json")
    print(f"Completed Phase 1.5 {stage} in {elapsed:.1f}s: {output_dir.resolve()}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
