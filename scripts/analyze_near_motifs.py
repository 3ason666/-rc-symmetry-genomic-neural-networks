from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / ".matplotlib_near_motif"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from src.near_motif import annotate_near_motifs, parse_intervals


sns.set_theme(style="whitegrid")


def _mask(length: int, intervals) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for start, end in intervals:
        mask[int(start):int(end)] = True
    return mask


def _mass_fraction(scores: np.ndarray, intervals) -> float:
    values = np.abs(np.asarray(scores, float))
    denom = values.sum()
    if denom <= 0 or not intervals:
        return 0.0
    return float(values[_mask(len(values), intervals)].sum() / denom)


def _density_enrichment(scores: np.ndarray, target_intervals, excluded_intervals) -> float:
    values = np.abs(np.asarray(scores, float))
    target = _mask(len(values), target_intervals)
    excluded = _mask(len(values), excluded_intervals)
    background = ~excluded
    if not target.any() or not background.any():
        return np.nan
    background_mean = float(values[background].mean())
    return float(values[target].mean() / background_mean) if background_mean > 0 else np.nan


def _load_npz(results_dir: Path, model_type: str, seed: int) -> dict:
    run_name = f"{model_type.lower().replace('-', '_')}_seed_{seed}"
    path = results_dir / "runs" / run_name / "phase15_ism_attributions.npz"
    with np.load(path) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _prevalence_table(dataset: pd.DataFrame, motifs: list[str], thresholds: list[int]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        annotated, _ = annotate_near_motifs(dataset, motifs, threshold)
        for (split, label), group in annotated.groupby(["split", "clean_label"]):
            present = group.incidental_near_motif_count.gt(0)
            rows.append({
                "max_hamming": threshold,
                "split": split,
                "clean_label": int(label),
                "label_name": "positive" if int(label) else "negative",
                "n_samples": int(len(group)),
                "n_present": int(present.sum()),
                "fraction_present": float(present.mean()),
                "mean_hit_windows": float(group.incidental_near_motif_count.mean()),
                "mean_merged_bases": float(group.incidental_near_motif_base_count.mean()),
            })
    return pd.DataFrame(rows)


def _attribution_table(results_dir: Path, annotated: pd.DataFrame) -> pd.DataFrame:
    sample_df = pd.read_csv(results_dir / "phase15_per_sample_results.csv")
    annotations = annotated.set_index("sample_id")
    cache: dict[tuple[str, int], dict] = {}
    index_cache: dict[tuple[str, int], dict[str, int]] = {}
    rows = []
    for row in sample_df.itertuples(index=False):
        key = (str(row.model_type), int(row.seed))
        if key not in cache:
            cache[key] = _load_npz(results_dir, *key)
            index_cache[key] = {
                str(sample_id): index
                for index, sample_id in enumerate(cache[key]["sample_ids"].astype(str))
            }
        scores = cache[key]["forward_absolute"][index_cache[key][str(row.sample_id)]]
        annotation = annotations.loc[row.sample_id]
        near = parse_intervals(annotation.incidental_near_motif_intervals)
        causal = parse_intervals(row.causal_intervals)
        decoy = parse_intervals(row.decoy_intervals)
        shortcut = parse_intervals(row.shortcut_intervals)
        near_mass = _mass_fraction(scores, near)
        causal_mass = _mass_fraction(scores, causal)
        decoy_mass = _mass_fraction(scores, decoy)
        shortcut_mass = _mass_fraction(scores, shortcut)
        annotated_union = causal + decoy + shortcut + near
        residual = max(0.0, 1.0 - causal_mass - decoy_mass - shortcut_mass - near_mass)
        rows.append({
            "model_type": row.model_type,
            "seed": int(row.seed),
            "sample_id": row.sample_id,
            "near_motif_present": int(bool(near)),
            "near_motif_hit_windows": int(annotation.incidental_near_motif_count),
            "near_motif_merged_bases": int(annotation.incidental_near_motif_base_count),
            "causal_mass_fraction": causal_mass,
            "decoy_mass_fraction": decoy_mass,
            "shortcut_mass_fraction": shortcut_mass,
            "near_motif_mass_fraction": near_mass,
            "ordinary_background_mass_fraction": residual,
            "near_motif_density_enrichment": _density_enrichment(scores, near, annotated_union),
            "prediction_difference": float(row.prediction_difference),
            "attribution_pearson_absolute": float(row.attribution_pearson_absolute),
            "incidental_near_motif_intervals": annotation.incidental_near_motif_intervals,
            "incidental_near_motif_hits": annotation.incidental_near_motif_hits,
        })
    return pd.DataFrame(rows)


def _summary_table(attribution: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_type, group in attribution.groupby("model_type", sort=False):
        conditional = group[group.near_motif_present == 1]
        rows.append({
            "model_type": model_type,
            "n_seeds": int(group.seed.nunique()),
            "n_ism_rows": int(len(group)),
            "near_motif_prevalence": float(group.near_motif_present.mean()),
            "causal_mass_mean": float(group.causal_mass_fraction.mean()),
            "decoy_mass_mean": float(group.decoy_mass_fraction.mean()),
            "near_motif_mass_mean_all": float(group.near_motif_mass_fraction.mean()),
            "near_motif_mass_mean_when_present": float(conditional.near_motif_mass_fraction.mean()),
            "near_motif_mass_sd_when_present": float(conditional.near_motif_mass_fraction.std()),
            "near_motif_density_enrichment_when_present": float(
                conditional.near_motif_density_enrichment.mean()
            ),
            "ordinary_background_mass_mean": float(group.ordinary_background_mass_fraction.mean()),
        })
    return pd.DataFrame(rows)


def _plot_representative(results_dir: Path, annotated: pd.DataFrame,
                         attribution: pd.DataFrame, output_dir: Path,
                         max_hamming: int) -> tuple[str, list[dict]]:
    representative_seed = int(attribution.seed.min())
    raw = attribution[(attribution.model_type == "CNN-Raw") &
                      (attribution.seed == representative_seed)]
    sample_id = str(raw.sort_values("prediction_difference", ascending=False).iloc[0].sample_id)
    sample = annotated.set_index("sample_id").loc[sample_id]
    causal = parse_intervals(sample.causal_intervals)
    decoy = parse_intervals(sample.decoy_intervals)
    shortcut = parse_intervals(sample.shortcut_intervals)
    near = parse_intervals(sample.incidental_near_motif_intervals)
    hits = json.loads(sample.incidental_near_motif_hits)
    order = list(dict.fromkeys(attribution.model_type.tolist()))
    colors = sns.color_palette("colorblind", len(order))
    fig, axes = plt.subplots(len(order), 1, figsize=(14, 2.35 * len(order)), sharex=True)
    for ax, model_type, color in zip(np.atleast_1d(axes), order, colors):
        arrays = _load_npz(results_dir, model_type, representative_seed)
        index = {str(value): i for i, value in enumerate(arrays["sample_ids"].astype(str))}[sample_id]
        forward = arrays["forward_absolute"][index]
        aligned = arrays["aligned_rc_absolute"][index]
        x = np.arange(len(forward))
        ax.plot(x, forward, color=color, label="ISM on S")
        ax.plot(x, aligned, color="black", linestyle="--", alpha=.7,
                label="aligned ISM on RC(S)")
        for i, (start, end) in enumerate(causal):
            ax.axvspan(start, end - 1, color="limegreen", alpha=.18,
                       label="causal motif" if i == 0 else None)
        for i, (start, end) in enumerate(decoy):
            ax.axvspan(start, end - 1, color="orange", alpha=.15,
                       label="decoy" if i == 0 else None)
        for i, (start, end) in enumerate(shortcut):
            ax.axvspan(start, end - 1, color="red", alpha=.14,
                       label="shortcut" if i == 0 else None)
        for i, (start, end) in enumerate(near):
            ax.axvspan(start, end - 1, color="mediumpurple", alpha=.18,
                       label=f"incidental near-motif (d<={max_hamming})" if i == 0 else None)
        ax.set_ylabel(model_type)
        ax.legend(loc="upper right", ncol=5, fontsize=7)
    axes[-1].set_xlabel("Sequence position (0-based)")
    fig.suptitle(
        f"Representative sample {sample_id}, seed {representative_seed}: near-motif control"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "representative_ism_with_near_motif.png", dpi=180)
    plt.close(fig)
    return sample_id, hits


def _plot_summary(prevalence: pd.DataFrame, attribution: pd.DataFrame,
                  output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    test_prev = prevalence[prevalence.split == "test"].copy()
    sns.barplot(test_prev, x="label_name", y="fraction_present", hue="max_hamming",
                ax=axes[0])
    axes[0].set(
        title="Incidental near-motif prevalence in test set",
        xlabel="Clean label", ylabel="Fraction of sequences with >=1 hit", ylim=(0, 1),
    )
    mass = attribution.melt(
        id_vars="model_type",
        value_vars=["causal_mass_fraction", "near_motif_mass_fraction", "decoy_mass_fraction"],
        var_name="region", value_name="absolute_ism_mass_fraction",
    )
    mass["region"] = mass["region"].map({
        "causal_mass_fraction": "causal motif",
        "near_motif_mass_fraction": "incidental near-motif",
        "decoy_mass_fraction": "inserted decoy",
    })
    sns.barplot(mass, x="model_type", y="absolute_ism_mass_fraction", hue="region",
                errorbar="sd", ax=axes[1])
    axes[1].set(
        title="Attribution mass by annotated region",
        xlabel="", ylabel="Mean absolute-ISM mass fraction", ylim=(0, 1),
    )
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / "near_motif_control_summary.png", dpi=180)
    plt.close(fig)


def run(results_dir: Path, output_dir: Path | None = None) -> Path:
    results_dir = results_dir.resolve()
    output_dir = (output_dir or results_dir / "near_motif_control").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((results_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    control = config.get("interpretation", {}).get("near_motif_control", {})
    motifs = [str(value).upper() for value in (control.get("motifs") or [config["data"]["causal_motif"]])]
    max_hamming = int(control.get("max_hamming", 2))
    thresholds = [int(value) for value in control.get("sensitivity_thresholds", [1, max_hamming])]
    thresholds = sorted(set(thresholds + [max_hamming]))
    dataset = pd.read_csv(results_dir / "phase15_dataset.csv")
    annotated, scan_report = annotate_near_motifs(dataset, motifs, max_hamming)
    annotated.to_csv(output_dir / "phase15_dataset_near_motif_annotated.csv", index=False)
    prevalence = _prevalence_table(dataset, motifs, thresholds)
    prevalence.to_csv(output_dir / "near_motif_prevalence.csv", index=False)
    attribution = _attribution_table(results_dir, annotated)
    attribution.to_csv(output_dir / "near_motif_ism_results.csv", index=False)
    summary = _summary_table(attribution)
    summary.to_csv(output_dir / "near_motif_ism_summary.csv", index=False)
    sample_id, hits = _plot_representative(
        results_dir, annotated, attribution, output_dir, max_hamming
    )
    _plot_summary(prevalence, attribution, output_dir)
    report = {
        "analysis_status": "post_hoc_exploratory_control",
        "results_dir": str(results_dir),
        "motifs": motifs,
        "primary_diagnostic_max_hamming": max_hamming,
        "sensitivity_thresholds": thresholds,
        "exclude_annotated_regions": True,
        "scan_report": scan_report,
        "representative_sample_id": sample_id,
        "representative_sample_hits": hits,
    }
    (output_dir / "near_motif_control_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "NEAR_MOTIF_CONTROL.md").write_text(
        "# Phase 1.5 incidental near-motif control\n\n"
        "Status: post hoc exploratory diagnostic; not a frozen H1-H4 endpoint.\n\n"
        f"Motif(s): {motifs}; both strands; primary Hamming threshold <= {max_hamming}; "
        f"sensitivity thresholds {thresholds}. Annotated causal, decoy, and shortcut "
        "intervals were excluded before scanning.\n\n"
        f"Representative sample: `{sample_id}`.\n",
        encoding="utf-8",
    )
    print(f"Completed near-motif control: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
