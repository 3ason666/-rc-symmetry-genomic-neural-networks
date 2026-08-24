from __future__ import annotations

import argparse
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
from torch.nn import functional as F

from scripts.run_phase2a import attribution_metrics, load_modeling_view, write_json
from scripts.validate_phase2_manifest import validate_manifest
from src.dna_utils import align_rc_full_attribution, align_rc_position_attribution, batch_one_hot_encode, reverse_complement
from src.interpret import run_ism_for_sequences
from src.metrics import classification_metrics, prediction_consistency_metrics
from src.models import (
    PostHocConjoined,
    build_model,
    parameter_count,
    reverse_complement_feature_pairs,
    reverse_complement_one_hot,
)
from src.training import DEVICE, train_model_validation_only


@torch.inference_mode()
def predict_logits(model, sequences, batch_size):
    model.eval()
    chunks = []
    for start in range(0, len(sequences), batch_size):
        inputs = batch_one_hot_encode(sequences[start : start + batch_size]).to(DEVICE)
        chunks.append(model(inputs).cpu().numpy())
    return np.concatenate(chunks) if chunks else np.array([], dtype=float)


def metrics_from_logits(labels, logits, threshold):
    labels_array = np.asarray(labels, dtype=np.float32)
    logits_array = np.asarray(logits, dtype=np.float32)
    loss = F.binary_cross_entropy_with_logits(
        torch.from_numpy(logits_array), torch.from_numpy(labels_array), reduction="mean"
    ).item()
    probabilities = torch.sigmoid(torch.from_numpy(logits_array)).numpy()
    return classification_metrics(labels_array, probabilities, loss, threshold), probabilities


def build_posthoc_from_phase2a(config):
    checkpoint_path = ROOT / config["phase2a_sources"]["conjoined_backbone_checkpoint"]
    saved = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    if saved.get("model_type") != "CNN-Aug" or saved.get("architecture") != "standard":
        raise ValueError("Post-hoc backbone must be the frozen Phase 2A CNN-Aug checkpoint")
    backbone = build_model(saved["model_config"], architecture="standard").to(DEVICE)
    backbone.load_state_dict(saved["model_state"])
    return PostHocConjoined(backbone).to(DEVICE).eval(), saved


def save_posthoc_checkpoint(model, source, path):
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": source["model_config"],
            "architecture": "posthoc_conjoined",
            "model_type": "CNN-PostHoc",
            "seed": source["seed"],
            "best_epoch": source["best_epoch"],
            "selection_split": "validation",
            "test_evaluated": False,
            "source_model_type": source["model_type"],
        },
        path,
    )


def load_p2b_checkpoint(path):
    saved = torch.load(path, map_location=DEVICE, weights_only=True)
    architecture = saved["architecture"]
    if architecture == "rcps":
        model = build_model(saved["model_config"], architecture="rcps").to(DEVICE)
    elif architecture == "posthoc_conjoined":
        model = PostHocConjoined(build_model(saved["model_config"], architecture="standard")).to(DEVICE)
    else:
        raise ValueError(f"Unsupported P2B checkpoint architecture: {architecture}")
    model.load_state_dict(saved["model_state"])
    return model.eval(), saved


@torch.inference_mode()
def feature_equivariance_max_error(model, sequences, batch_size):
    if not hasattr(model, "features"):
        return np.nan
    maximum = 0.0
    for start in range(0, len(sequences), batch_size):
        inputs = batch_one_hot_encode(sequences[start : start + batch_size]).to(DEVICE)
        observed = model.features(reverse_complement_one_hot(inputs))
        expected = reverse_complement_feature_pairs(model.features(inputs))
        maximum = max(maximum, float(torch.max(torch.abs(observed - expected)).item()))
    return maximum


def create_figures(comparison, audit, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    order = ["CNN-Raw", "CNN-Aug", "CNN-Pair", "CNN-PostHoc", "CNN-RCPS"]
    colors = ["#2563EB", "#D97706", "#059669", "#7C3AED", "#DB2777"]
    table = comparison.set_index("model_type").loc[order]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bars = ax.bar(order, table.validation_auroc, color=colors, alpha=0.8)
    ax.set_ylim(0.75, 0.91)
    ax.set_ylabel("Validation AUROC")
    ax.set_title("Phase 2A/2B validation classification performance (test sealed)")
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, table.validation_auroc):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "validation_auroc_all_models.png", dpi=180)
    fig.savefig(output_dir / "validation_auroc_all_models.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8))
    axes[0].bar(order, table.prediction_mean_absolute_difference, color=colors, alpha=0.8)
    axes[0].axhline(0.01, color="#991B1B", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Mean |p(S)-p(RC(S))|")
    axes[0].set_title("Prediction asymmetry")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(order, table.mean_absolute_normalized_l1, color=colors, alpha=0.8)
    axes[1].set_ylabel("Mean absolute ISM normalized L1")
    axes[1].set_title("Attribution asymmetry")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_and_attribution_asymmetry.png", dpi=180)
    fig.savefig(output_dir / "prediction_and_attribution_asymmetry.pdf")
    plt.close(fig)

    strict = audit.set_index("model_type").loc[["CNN-PostHoc", "CNN-RCPS"]]
    labels = ["Prediction max error", "ISM matrix max error", "Feature equivariance", "Reload error"]
    columns = [
        "prediction_max_abs_probability_error",
        "ism_full_matrix_max_abs_error",
        "feature_equivariance_max_abs_error",
        "checkpoint_reload_max_abs_probability_error",
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    for offset, (model, color) in zip((-0.18, 0.18), (("CNN-PostHoc", "#7C3AED"), ("CNN-RCPS", "#DB2777"))):
        values = strict.loc[model, columns].astype(float).to_numpy()
        values = np.maximum(values, 1e-16)
        ax.bar(x + offset, values, width=0.34, label=model, color=color, alpha=0.8)
    ax.axhline(1e-6, color="#7A5A00", linestyle="--", linewidth=1, label="prediction/feature/reload tolerance")
    ax.axhline(1e-5, color="#991B1B", linestyle=":", linewidth=1.2, label="ISM tolerance")
    ax.set_yscale("log")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Maximum absolute error (log scale)")
    ax.set_title("Phase 2B strict RC invariance gates")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "strict_invariance_gates.png", dpi=180)
    fig.savefig(output_dir / "strict_invariance_gates.pdf")
    plt.close(fig)


def run(config_path, output_dir=None):
    started = time.perf_counter()
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if config.get("device") != "cpu":
        raise ValueError("P2B requires device: cpu")
    if config["dataset"].get("test_policy") != "sealed_no_model_access":
        raise ValueError("P2B test policy must remain sealed_no_model_access")
    manifest_errors = validate_manifest(ROOT / "configs" / "phase2_dataset_manifest.yaml")
    if manifest_errors:
        raise ValueError("Frozen Phase 2 manifest is invalid: " + "; ".join(manifest_errors))
    torch.set_num_threads(max(1, int(config.get("cpu_threads", 1))))
    output_dir = Path(output_dir) if output_dir else ROOT / "results" / config["project_name"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty P2B directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "resolved_config.yaml")

    modeling, contract = load_modeling_view(config)
    contract.update({
        "phase": "Phase 2B",
        "test_predictions_generated": False,
        "test_metrics_generated": False,
    })
    write_json(contract, output_dir / "sealed_test_policy.json")
    validation = modeling[modeling.split == "validation"].reset_index(drop=True)
    sequences = validation.sequence.tolist()
    rc_sequences = [reverse_complement(sequence) for sequence in sequences]
    labels = validation.label.to_numpy()
    batch_size = int(config["training"]["batch_size"])
    threshold = float(config["training"]["threshold"])

    shared_ids = pd.read_csv(ROOT / config["phase2a_sources"]["shared_ism_sample_ids"])
    if len(shared_ids) != 100 or set(shared_ids.label) != {0, 1}:
        raise ValueError("P2B requires the frozen 100-sample Phase 2A ISM set")
    selected = shared_ids.merge(validation, on=["sample_id", "label", "chromosome"], how="left", validate="one_to_one")
    if selected.sequence.isna().any():
        raise ValueError("A frozen P2A ISM sample is absent from validation")
    selected.to_csv(output_dir / "shared_ism_validation_samples.csv", index=False, encoding="utf-8")

    print(
        f"Python={platform.python_version()} | PyTorch={torch.__version__} | threads={torch.get_num_threads()} | "
        f"train={int((modeling.split == 'train').sum())} | validation={len(validation)} | test=sealed",
        flush=True,
    )

    trained_models = {}
    metadata_by_model = {}
    posthoc_dir = output_dir / "runs" / "cnn_posthoc_seed_42"
    posthoc_dir.mkdir(parents=True, exist_ok=True)
    posthoc, source_checkpoint = build_posthoc_from_phase2a(config)
    save_posthoc_checkpoint(posthoc, source_checkpoint, posthoc_dir / "best_checkpoint.pt")
    trained_models["CNN-PostHoc"] = posthoc
    metadata_by_model["CNN-PostHoc"] = {
        "model_type": "CNN-PostHoc",
        "seed": 42,
        "best_epoch": int(source_checkpoint["best_epoch"]),
        "training_seconds": 0.0,
        "parameter_count": parameter_count(posthoc),
        "source": "frozen Phase 2A CNN-Aug checkpoint",
        "test_evaluated": False,
    }
    print("Constructed CNN-PostHoc from frozen CNN-Aug checkpoint", flush=True)

    print("Training CNN-RCPS, seed=42, selection=validation_loss, test=sealed", flush=True)
    rcps_dir = output_dir / "runs" / "cnn_rcps_seed_42"
    rcps, rcps_metadata = train_model_validation_only("CNN-RCPS", 42, config, modeling, rcps_dir)
    trained_models["CNN-RCPS"] = rcps
    metadata_by_model["CNN-RCPS"] = rcps_metadata

    gate_cfg = config["invariance_gates"]
    audit_rows = []
    run_rows = []
    attribution_rows = []
    for model_type in ("CNN-PostHoc", "CNN-RCPS"):
        print(f"Auditing {model_type}: prediction, reload, feature and exact ISM", flush=True)
        run_dir = output_dir / "runs" / f"{model_type.lower().replace('-', '_')}_seed_42"
        checkpoint_path = run_dir / "best_checkpoint.pt"
        before = trained_models[model_type].eval()
        before_logits = predict_logits(before, sequences, batch_size)
        reloaded, saved = load_p2b_checkpoint(checkpoint_path)
        after_logits = predict_logits(reloaded, sequences, batch_size)
        rc_logits = predict_logits(reloaded, rc_sequences, batch_size)
        before_prob = torch.sigmoid(torch.from_numpy(before_logits)).numpy()
        after_prob = torch.sigmoid(torch.from_numpy(after_logits)).numpy()
        rc_prob = torch.sigmoid(torch.from_numpy(rc_logits)).numpy()
        reload_probability_error = float(np.max(np.abs(before_prob - after_prob)))
        reload_logit_error = float(np.max(np.abs(before_logits - after_logits)))
        prediction_probability_error = float(np.max(np.abs(after_prob - rc_prob)))
        prediction_logit_error = float(np.max(np.abs(after_logits - rc_logits)))
        classification, _ = metrics_from_logits(labels, after_logits, threshold)
        consistency_summary, consistency_each = prediction_consistency_metrics(after_prob, rc_prob, threshold)
        feature_error = (
            feature_equivariance_max_error(reloaded, sequences[:512], batch_size)
            if model_type == "CNN-RCPS"
            else np.nan
        )
        sample_tensor = batch_one_hot_encode(sequences[:32])
        double_rc_exact = bool(torch.equal(reverse_complement_one_hot(reverse_complement_one_hot(sample_tensor)), sample_tensor))

        ism_sequences = selected.sequence.tolist()
        ism_rc_sequences = [reverse_complement(sequence) for sequence in ism_sequences]
        f_matrix, f_signed, f_absolute, f_seconds = run_ism_for_sequences(
            reloaded,
            ism_sequences,
            int(config["interpretation"]["batch_size"]),
            config["interpretation"]["difference"],
            f"{model_type} forward ISM",
        )
        r_matrix, r_signed, r_absolute, r_seconds = run_ism_for_sequences(
            reloaded,
            ism_rc_sequences,
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
        matrix_max_error = float(np.max(np.abs(f_matrix - aligned_matrix)))
        absolute_max_error = float(np.max(np.abs(f_absolute - aligned_absolute)))
        signed_max_error = float(np.max(np.abs(f_signed - aligned_signed)))
        for index, sample in selected.iterrows():
            metrics = attribution_metrics(f_absolute[index], aligned_absolute[index], int(config["interpretation"]["top_k"]))
            attribution_rows.append({
                "model_type": model_type,
                "seed": 42,
                "sample_id": sample.sample_id,
                "label": int(sample.label),
                **{f"absolute_{key}": value for key, value in metrics.items()},
            })
        model_attribute = pd.DataFrame([row for row in attribution_rows if row["model_type"] == model_type])
        audit = {
            "model_type": model_type,
            "checkpoint_architecture": saved["architecture"],
            "prediction_max_abs_probability_error": prediction_probability_error,
            "prediction_max_abs_logit_error": prediction_logit_error,
            "checkpoint_reload_max_abs_probability_error": reload_probability_error,
            "checkpoint_reload_max_abs_logit_error": reload_logit_error,
            "feature_equivariance_max_abs_error": feature_error,
            "double_rc_input_exact": double_rc_exact,
            "ism_full_matrix_max_abs_error": matrix_max_error,
            "ism_absolute_position_max_abs_error": absolute_max_error,
            "ism_signed_position_max_abs_error": signed_max_error,
            "prediction_gate_passed": prediction_probability_error <= float(gate_cfg["prediction_max_abs_tolerance"]),
            "reload_gate_passed": reload_probability_error <= float(gate_cfg["checkpoint_reload_max_abs_tolerance"]),
            "feature_gate_passed": (
                True if model_type == "CNN-PostHoc" else feature_error <= float(gate_cfg["feature_equivariance_max_abs_tolerance"])
            ),
            "ism_gate_passed": matrix_max_error <= float(gate_cfg["exact_ism_aligned_max_abs_tolerance"]),
        }
        audit["all_gates_passed"] = bool(
            audit["prediction_gate_passed"]
            and audit["reload_gate_passed"]
            and audit["feature_gate_passed"]
            and audit["ism_gate_passed"]
            and double_rc_exact
        )
        audit_rows.append(audit)
        metadata = metadata_by_model[model_type]
        metadata.update({
            "validation_metrics": classification,
            "selection_split": "validation",
            "test_evaluated": False,
            "checkpoint_reload_audited": True,
        })
        write_json(metadata, run_dir / "run_metadata.json")
        run_rows.append({
            "model_type": model_type,
            "seed": 42,
            "best_epoch": metadata["best_epoch"],
            "parameter_count": metadata["parameter_count"],
            "training_seconds": metadata["training_seconds"],
            **{f"validation_{key}": value for key, value in classification.items()},
            **consistency_summary,
            "prediction_max_abs_difference": prediction_probability_error,
            "mean_absolute_pearson": float(model_attribute.absolute_pearson.mean()),
            "median_absolute_pearson": float(model_attribute.absolute_pearson.median()),
            "mean_absolute_top8_overlap": float(model_attribute.absolute_top8_overlap.mean()),
            "mean_absolute_normalized_l1": float(model_attribute.absolute_normalized_l1.mean()),
            "mean_absolute_normalized_l2": float(model_attribute.absolute_normalized_l2.mean()),
            "ism_seconds": f_seconds + r_seconds,
            "test_evaluated": False,
        })

    audit_frame = pd.DataFrame(audit_rows)
    audit_frame.to_csv(output_dir / "strict_invariance_audit.csv", index=False, encoding="utf-8")
    pd.DataFrame(attribution_rows).to_csv(
        output_dir / "validation_attribution_results.csv", index=False, encoding="utf-8"
    )
    p2b_summary = pd.DataFrame(run_rows)
    p2b_summary.to_csv(output_dir / "run_summary.csv", index=False, encoding="utf-8")

    p2a = pd.read_csv(ROOT / config["phase2a_sources"]["results_dir"] / "run_summary.csv")
    common = [
        "model_type",
        "seed",
        "best_epoch",
        "parameter_count",
        "training_seconds",
        "validation_loss",
        "validation_accuracy",
        "validation_f1",
        "validation_auroc",
        "validation_auprc",
        "prediction_mean_absolute_difference",
        "prediction_median_absolute_difference",
        "prediction_p95_absolute_difference",
        "prediction_pearson",
        "prediction_spearman",
        "symmetry_flip_rate",
        "mean_absolute_pearson",
        "mean_absolute_top8_overlap",
        "mean_absolute_normalized_l1",
        "ism_seconds",
        "test_evaluated",
    ]
    comparison = pd.concat([p2a[common], p2b_summary[common]], ignore_index=True)
    comparison.to_csv(output_dir / "phase2a_phase2b_model_comparison.csv", index=False, encoding="utf-8")
    create_figures(comparison, audit_frame, output_dir / "figures")

    all_passed = bool(audit_frame.all_gates_passed.all())
    gate = {
        "phase": "Phase 2B strict RC models",
        "analysis_split": "validation",
        "test_seal_intact": True,
        "all_strict_invariance_gates_passed": all_passed,
        "per_model": audit_rows,
        "h3a_validation_observation": {
            "cnn_aug_mean_attribution_normalized_l1": float(p2a.set_index("model_type").loc["CNN-Aug", "mean_absolute_normalized_l1"]),
            "cnn_rcps_mean_attribution_normalized_l1": float(p2b_summary.set_index("model_type").loc["CNN-RCPS", "mean_absolute_normalized_l1"]),
            "scope": "single_seed_validation_pilot_not_prediction_performance_matched",
        },
        "h3b_validation_gate": "passed" if all_passed else "failed",
        "decision": {
            "proceed_to_phase2c_transformer": all_passed,
            "unseal_test_now": False,
            "start_phase2_full_now": False,
        },
    }
    write_json(gate, output_dir / "phase2b_gate_assessment.json")
    completion = {
        "status": "completed_validation_only" if all_passed else "completed_with_failed_gate",
        "elapsed_seconds": time.perf_counter() - started,
        "test_predictions_generated": False,
        "test_metrics_generated": False,
        "shared_ism_samples": int(len(selected)),
        "all_strict_invariance_gates_passed": all_passed,
    }
    write_json(completion, output_dir / "pilot_completion.json")
    print(audit_frame.to_string(index=False), flush=True)
    print(f"Completed Phase 2B in {completion['elapsed_seconds']:.2f}s: {output_dir}", flush=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Run Phase 2B strict RC model pilot with test sealed")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase2b_pilot.yaml")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
