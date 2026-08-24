from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_curve

from scripts.validate_phase2_manifest import validate_manifest
from src.dna_utils import align_rc_full_attribution, align_rc_position_attribution, reverse_complement
from src.interpret import run_ism_for_sequences
from src.metrics import attribution_consistency, prediction_consistency_metrics, safe_similarity
from src.training import predict_sequences, train_model_validation_only


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(value, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=True), encoding="utf-8")


def load_modeling_view(config: dict) -> tuple[pd.DataFrame, dict]:
    dataset_path = ROOT / config["dataset"]["path"]
    observed_sha = sha256_file(dataset_path)
    expected_sha = str(config["dataset"]["sha256"])
    if observed_sha != expected_sha:
        raise ValueError(f"Phase 2 dataset SHA-256 mismatch: {observed_sha} != {expected_sha}")

    full = pd.read_csv(dataset_path)
    required = {"sample_id", "sequence", "label", "split", "pair_id", "chromosome"}
    missing = required - set(full.columns)
    if missing:
        raise ValueError(f"Phase 2 dataset is missing columns: {sorted(missing)}")
    allowed = set(config["dataset"]["allowed_training_splits"])
    if allowed != {"train", "validation"}:
        raise ValueError("The Phase 2A pilot must allow exactly train and validation")
    modeling = full[full["split"].isin(allowed)].copy()
    sealed_rows = int((full["split"] == "test").sum())
    del full

    if set(modeling["split"].unique()) != allowed:
        raise ValueError("Both train and validation must be present")
    expected_length = int(config["dataset"]["sequence_length"])
    if not modeling["sequence"].str.len().eq(expected_length).all():
        raise ValueError("Unexpected sequence length in train/validation")
    if modeling["sequence"].str.contains(r"[^ACGT]", regex=True).any():
        raise ValueError("Non-ACGT sequence in train/validation")
    if set(modeling["label"].unique()) != {0, 1}:
        raise ValueError("Both labels must be present in the modeling view")

    counts = (
        modeling.groupby(["split", "label"], observed=True).size().rename("n").reset_index()
    )
    contract = {
        "dataset_path": config["dataset"]["path"],
        "dataset_sha256": observed_sha,
        "allowed_modeling_splits": sorted(allowed),
        "modeling_rows": int(len(modeling)),
        "modeling_counts": counts.to_dict(orient="records"),
        "sealed_test_rows_excluded_before_training": sealed_rows,
        "test_policy": config["dataset"]["test_policy"],
        "test_predictions_generated": False,
        "test_metrics_generated": False,
    }
    return modeling, contract


def select_ism_samples(
    validation: pd.DataFrame,
    prediction_pairs: pd.DataFrame,
    positive_n: int,
    negative_n: int,
    sample_seed: int,
) -> pd.DataFrame:
    positive = validation[validation["label"] == 1]
    if len(positive) < positive_n:
        raise ValueError("Not enough validation positives for frozen ISM sample count")
    selected_positive = positive.sample(n=positive_n, random_state=sample_seed).copy()
    selected_positive["selection_reason"] = "deterministic_random_positive"
    selected_positive["selection_score"] = np.nan

    scores = (
        prediction_pairs[prediction_pairs["label"] == 0]
        .assign(mean_orientation_probability=lambda frame: 0.5 * (frame.p_forward + frame.p_rc))
        .groupby("sample_id", observed=True)["mean_orientation_probability"]
        .mean()
        .rename("selection_score")
        .reset_index()
    )
    negative = validation[validation["label"] == 0].merge(scores, on="sample_id", how="inner")
    negative = negative.sort_values(
        ["selection_score", "sample_id"], ascending=[False, True], kind="mergesort"
    )
    if len(negative) < negative_n:
        raise ValueError("Not enough scored validation negatives for frozen ISM sample count")
    selected_negative = negative.head(negative_n).copy()
    selected_negative["selection_reason"] = "hard_negative_high_mean_model_probability"
    selected = pd.concat([selected_positive, selected_negative], ignore_index=True)
    return selected.sort_values(["label", "sample_id"], ascending=[False, True]).reset_index(drop=True)


def attribution_metrics(forward, aligned, top_k: int) -> dict:
    consistency, issues = attribution_consistency(forward, aligned, top_k)
    difference = np.asarray(forward, float) - np.asarray(aligned, float)
    l1_denom = np.abs(forward).sum() + np.abs(aligned).sum()
    l2_denom = np.linalg.norm(forward) + np.linalg.norm(aligned)
    return {
        "pearson": consistency["pearson"],
        "spearman": consistency["spearman"],
        "cosine": consistency["cosine"],
        "top8_overlap": consistency["top8_overlap"],
        "normalized_l1": float(np.abs(difference).sum() / l1_denom) if l1_denom else np.nan,
        "normalized_l2": float(np.linalg.norm(difference) / l2_denom) if l2_denom else np.nan,
        "maximum_aligned_error": float(np.max(np.abs(difference))),
        "issues": ";".join(issues),
    }


def create_figures(predictions: pd.DataFrame, attributions: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    models = list(dict.fromkeys(predictions["model_type"]))
    colors = {"CNN-Raw": "#2563EB", "CNN-Aug": "#D97706", "CNN-Pair": "#059669"}

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    for model_type in models:
        group = predictions[predictions.model_type == model_type]
        fpr, tpr, _ = roc_curve(group.label, group.p_forward)
        auc = np.trapezoid(tpr, fpr)
        ax.plot(fpr, tpr, lw=2, color=colors.get(model_type), label=f"{model_type} (AUROC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#6B7280", linestyle="--", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Phase 2A validation ROC (test sealed)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "validation_roc.png", dpi=180)
    fig.savefig(output_dir / "validation_roc.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    values = [predictions.loc[predictions.model_type == model, "prediction_difference"] for model in models]
    boxes = ax.boxplot(values, tick_labels=models, showfliers=False, patch_artist=True)
    for patch, model in zip(boxes["boxes"], models):
        patch.set_facecolor(colors.get(model, "#9CA3AF"))
        patch.set_alpha(0.55)
    ax.axhline(0.01, color="#991B1B", linestyle="--", linewidth=1, label="frozen consistency threshold")
    ax.set(ylabel="|p(S) - p(RC(S))|", title="Validation prediction asymmetry")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "validation_prediction_asymmetry.png", dpi=180)
    fig.savefig(output_dir / "validation_prediction_asymmetry.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    values = [attributions.loc[attributions.model_type == model, "absolute_pearson"].dropna() for model in models]
    boxes = ax.boxplot(values, tick_labels=models, showfliers=False, patch_artist=True)
    for patch, model in zip(boxes["boxes"], models):
        patch.set_facecolor(colors.get(model, "#9CA3AF"))
        patch.set_alpha(0.55)
    ax.axhline(0.90, color="#991B1B", linestyle="--", linewidth=1, label="frozen warning threshold")
    ax.set(ylabel="RC-aligned absolute ISM Pearson", title="Validation attribution symmetry")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "validation_attribution_symmetry.png", dpi=180)
    fig.savefig(output_dir / "validation_attribution_symmetry.pdf")
    plt.close(fig)


def run(config_path: Path, output_dir: Path | None = None) -> Path:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("device") != "cpu":
        raise ValueError("Phase 2A pilot requires device: cpu")
    if config["dataset"].get("test_policy") != "sealed_no_model_access":
        raise ValueError("Phase 2A test policy must remain sealed_no_model_access")

    manifest = ROOT / "configs" / "phase2_dataset_manifest.yaml"
    manifest_errors = validate_manifest(manifest)
    if manifest_errors:
        raise ValueError("Frozen Phase 2 manifest is invalid: " + "; ".join(manifest_errors))

    torch.set_num_threads(max(1, int(config.get("cpu_threads", 1))))
    output_dir = output_dir or ROOT / "results" / config["project_name"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty pilot directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "resolved_config.yaml")
    modeling, contract = load_modeling_view(config)
    write_json(contract, output_dir / "sealed_test_policy.json")
    validation = modeling[modeling.split == "validation"].reset_index(drop=True)

    print(
        f"Python={platform.python_version()} | PyTorch={torch.__version__} | device=cpu | "
        f"threads={torch.get_num_threads()}"
    )
    print(
        f"Phase2A train={int((modeling.split == 'train').sum())} | "
        f"validation={len(validation)} | sealed_test_rows={contract['sealed_test_rows_excluded_before_training']}"
    )

    models = {}
    run_rows = []
    prediction_frames = []
    for seed in config["training"]["seeds"]:
        for model_type in config["training"]["model_types"]:
            print(f"Training {model_type}, seed={seed}, selection=validation_loss, test=sealed", flush=True)
            run_dir = output_dir / "runs" / f"{model_type.lower().replace('-', '_')}_seed_{seed}"
            model, metadata = train_model_validation_only(
                model_type, int(seed), config, modeling, run_dir
            )
            models[(model_type, int(seed))] = model
            pf = predict_sequences(model, validation.sequence.tolist(), int(config["training"]["batch_size"]))
            rc_sequences = [reverse_complement(sequence) for sequence in validation.sequence]
            pr = predict_sequences(model, rc_sequences, int(config["training"]["batch_size"]))
            consistency_summary, per_sample = prediction_consistency_metrics(
                pf, pr, float(config["training"]["threshold"])
            )
            prediction_frame = validation[
                ["sample_id", "pair_id", "label", "chromosome", "gc_fraction", "source_signal"]
            ].copy()
            prediction_frame.insert(0, "seed", int(seed))
            prediction_frame.insert(0, "model_type", model_type)
            prediction_frame["p_forward"] = pf
            prediction_frame["p_rc"] = pr
            for key, values in per_sample.items():
                prediction_frame[key] = values
            prediction_frames.append(prediction_frame)
            run_rows.append(
                {
                    "model_type": model_type,
                    "seed": int(seed),
                    "best_epoch": metadata["best_epoch"],
                    "parameter_count": metadata["parameter_count"],
                    "training_seconds": metadata["training_seconds"],
                    **{f"validation_{key}": value for key, value in metadata["validation_metrics"].items()},
                    **consistency_summary,
                    "test_evaluated": False,
                }
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output_dir / "validation_prediction_pairs.csv", index=False, encoding="utf-8")
    selected = select_ism_samples(
        validation,
        predictions,
        int(config["interpretation"]["validation_positive_samples"]),
        int(config["interpretation"]["validation_hard_negative_samples"]),
        int(config["interpretation"]["sample_seed"]),
    )
    selected[
        ["sample_id", "label", "chromosome", "selection_reason", "selection_score"]
    ].to_csv(output_dir / "ism_validation_sample_ids.csv", index=False, encoding="utf-8")

    attribute_rows = []
    top_k = int(config["interpretation"]["top_k"])
    prediction_limit = float(config["interpretation"]["prediction_consistent_max_abs_difference"])
    pearson_limit = float(config["interpretation"]["attribution_pearson_warning_below"])
    top8_limit = float(config["interpretation"]["attribution_top8_warning_below"])
    sequences = selected.sequence.tolist()
    rc_sequences = [reverse_complement(sequence) for sequence in sequences]
    for (model_type, seed), model in models.items():
        print(f"Exact ISM {model_type}, seed={seed}, validation n={len(selected)}", flush=True)
        run_dir = output_dir / "runs" / f"{model_type.lower().replace('-', '_')}_seed_{seed}"
        f_matrix, f_signed, f_absolute, f_seconds = run_ism_for_sequences(
            model,
            sequences,
            int(config["interpretation"]["batch_size"]),
            config["interpretation"]["difference"],
            f"{model_type} forward ISM",
        )
        r_matrix, r_signed, r_absolute, r_seconds = run_ism_for_sequences(
            model,
            rc_sequences,
            int(config["interpretation"]["batch_size"]),
            config["interpretation"]["difference"],
            f"{model_type} RC ISM",
        )
        aligned_matrix = align_rc_full_attribution(r_matrix)
        aligned_signed = align_rc_position_attribution(r_signed)
        aligned_absolute = align_rc_position_attribution(r_absolute)
        np.savez_compressed(
            run_dir / "validation_ism_attributions.npz",
            sample_ids=selected.sample_id.astype(str).to_numpy(dtype=str),
            forward_matrix=f_matrix,
            aligned_rc_matrix=aligned_matrix,
            forward_signed=f_signed,
            aligned_rc_signed=aligned_signed,
            forward_absolute=f_absolute,
            aligned_rc_absolute=aligned_absolute,
        )
        prediction_index = predictions[
            (predictions.model_type == model_type) & (predictions.seed == seed)
        ].set_index("sample_id")
        for idx, row in selected.iterrows():
            signed = attribution_metrics(f_signed[idx], aligned_signed[idx], top_k)
            absolute = attribution_metrics(f_absolute[idx], aligned_absolute[idx], top_k)
            full_pearson, full_issue = safe_similarity(f_matrix[idx], aligned_matrix[idx], "pearson")
            pred = prediction_index.loc[row.sample_id]
            prediction_consistent = float(pred.prediction_difference) <= prediction_limit
            warning = (
                (np.isfinite(absolute["pearson"]) and absolute["pearson"] < pearson_limit)
                or absolute["top8_overlap"] < top8_limit
            )
            attribute_rows.append(
                {
                    "model_type": model_type,
                    "seed": seed,
                    "sample_id": row.sample_id,
                    "label": int(row.label),
                    "selection_reason": row.selection_reason,
                    "p_forward": float(pred.p_forward),
                    "p_rc": float(pred.p_rc),
                    "prediction_difference": float(pred.prediction_difference),
                    "prediction_consistent": bool(prediction_consistent),
                    "attribution_warning": bool(warning),
                    **{f"signed_{key}": value for key, value in signed.items()},
                    **{f"absolute_{key}": value for key, value in absolute.items()},
                    "full_matrix_pearson": full_pearson,
                    "full_matrix_issue": full_issue or "",
                    "ism_seconds_forward": f_seconds,
                    "ism_seconds_rc": r_seconds,
                }
            )

    attributions = pd.DataFrame(attribute_rows)
    attributions.to_csv(output_dir / "validation_attribution_results.csv", index=False, encoding="utf-8")
    summaries = []
    for (model_type, seed), group in attributions.groupby(["model_type", "seed"], observed=True):
        consistent = group[group.prediction_consistent]
        summaries.append(
            {
                "model_type": model_type,
                "seed": int(seed),
                "n_explained": int(len(group)),
                "n_prediction_consistent": int(len(consistent)),
                "mean_absolute_pearson": float(group.absolute_pearson.mean()),
                "median_absolute_pearson": float(group.absolute_pearson.median()),
                "mean_absolute_top8_overlap": float(group.absolute_top8_overlap.mean()),
                "mean_absolute_normalized_l1": float(group.absolute_normalized_l1.mean()),
                "attribution_warning_rate": float(group.attribution_warning.mean()),
                "warning_rate_within_prediction_consistent": (
                    float(consistent.attribution_warning.mean()) if len(consistent) else np.nan
                ),
                "ism_seconds": float(group.ism_seconds_forward.iloc[0] + group.ism_seconds_rc.iloc[0]),
            }
        )
    attribution_summary = pd.DataFrame(summaries)
    attribution_summary.to_csv(output_dir / "validation_attribution_summary.csv", index=False, encoding="utf-8")

    run_summary = pd.DataFrame(run_rows).merge(
        attribution_summary, on=["model_type", "seed"], how="left", validate="one_to_one"
    )
    run_summary.to_csv(output_dir / "run_summary.csv", index=False, encoding="utf-8")
    create_figures(predictions, attributions, output_dir / "figures")

    elapsed = time.perf_counter() - started
    completion = {
        "phase": "Phase 2A CNN pilot",
        "status": "completed_validation_only",
        "elapsed_seconds": elapsed,
        "models": config["training"]["model_types"],
        "seeds": config["training"]["seeds"],
        "train_rows": int((modeling.split == "train").sum()),
        "validation_rows": int(len(validation)),
        "test_policy": "sealed_no_model_access",
        "test_predictions_generated": False,
        "test_metrics_generated": False,
        "ism_validation_samples": int(len(selected)),
        "thresholds_frozen_before_results": {
            "prediction_consistent_max_abs_difference": prediction_limit,
            "attribution_pearson_warning_below": pearson_limit,
            "attribution_top8_warning_below": top8_limit,
        },
    }
    write_json(completion, output_dir / "pilot_completion.json")
    print(f"Completed Phase 2A validation-only pilot in {elapsed:.2f}s: {output_dir}", flush=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Run the frozen Phase 2A validation-only CNN pilot")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase2a_pilot.yaml")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
