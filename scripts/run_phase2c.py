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

from scripts.run_phase2a import attribution_metrics, load_modeling_view, write_json
from scripts.validate_phase2_manifest import validate_manifest
from src.dna_utils import (
    align_rc_full_attribution,
    align_rc_position_attribution,
    reverse_complement,
)
from src.interpret import run_captum_for_sequences, run_ism_for_sequences
from src.metrics import prediction_consistency_metrics, safe_similarity
from src.models import build_model
from src.training import DEVICE, predict_sequences, train_model_validation_only


MODEL_ORDER = [
    "Transformer-Raw",
    "Transformer-Aug-None",
    "Transformer-Aug-Absolute",
    "Transformer-Aug-Relative",
]
ARCHITECTURE_BY_MODEL = {
    "Transformer-Raw": "transformer_absolute",
    "Transformer-Aug-None": "transformer_none",
    "Transformer-Aug-Absolute": "transformer_absolute",
    "Transformer-Aug-Relative": "transformer_relative",
}
DISPLAY = {
    "Transformer-Raw": "TF-Raw-Abs",
    "Transformer-Aug-None": "TF-Aug-None",
    "Transformer-Aug-Absolute": "TF-Aug-Abs",
    "Transformer-Aug-Relative": "TF-Aug-Rel",
}
COLORS = {
    "CNN-Aug": "#D97706",
    "Transformer-Raw": "#2563EB",
    "Transformer-Aug-None": "#059669",
    "Transformer-Aug-Absolute": "#7C3AED",
    "Transformer-Aug-Relative": "#DB2777",
}


def run_dir_name(model_type: str, seed: int) -> str:
    return f"{model_type.lower().replace('-', '_')}_seed_{seed}"


def load_checkpoint(path: Path):
    saved = torch.load(path, map_location=DEVICE, weights_only=True)
    model = build_model(saved["model_config"], architecture=saved["architecture"]).to(DEVICE)
    model.load_state_dict(saved["model_state"])
    return model.eval(), saved


def paired_bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    replicates: int,
    seed: int,
) -> dict:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or left.size == 0:
        raise ValueError("Paired bootstrap requires equal non-empty vectors")
    difference = left - right
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 1000):
        count = min(1000, replicates - start)
        indices = rng.integers(0, difference.size, size=(count, difference.size))
        estimates[start : start + count] = difference[indices].mean(axis=1)
    return {
        "n_pairs": int(difference.size),
        "estimate": float(difference.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
    }


def create_figures(run_summary, attribution_summary, h3c, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    table = run_summary.set_index("model_type").loc[MODEL_ORDER]
    labels = [DISPLAY[model] for model in MODEL_ORDER]
    colors = [COLORS[model] for model in MODEL_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    bars = axes[0].bar(labels, table.validation_auroc, color=colors, alpha=0.82)
    axes[0].axhline(0.80, color="#7A5A00", linestyle="--", linewidth=1, label="pilot floor")
    axes[0].set_ylim(0.74, 0.91)
    axes[0].set_ylabel("Validation AUROC")
    axes[0].set_title("Classification performance (test sealed)")
    axes[0].tick_params(axis="x", rotation=24)
    axes[0].grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, table.validation_auroc):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.003, f"{value:.3f}", ha="center", fontsize=8)
    axes[1].bar(labels, table.prediction_mean_absolute_difference, color=colors, alpha=0.82)
    axes[1].axhline(0.01, color="#991B1B", linestyle="--", linewidth=1, label="consistency threshold")
    axes[1].set_ylabel("Mean |p(S)-p(RC(S))|")
    axes[1].set_title("Prediction asymmetry")
    axes[1].tick_params(axis="x", rotation=24)
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "transformer_validation_performance_consistency.png", dpi=180)
    fig.savefig(output_dir / "transformer_validation_performance_consistency.pdf")
    plt.close(fig)

    ism = attribution_summary[attribution_summary.method == "exact_ism"].set_index("model_type")
    cnn_value = float(h3c["cnn_aug_mean_normalized_l1"])
    plot_labels = ["CNN-Aug"] + labels
    plot_values = [cnn_value] + [float(ism.loc[model, "mean_absolute_normalized_l1"]) for model in MODEL_ORDER]
    plot_colors = [COLORS["CNN-Aug"]] + colors
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    bars = ax.bar(plot_labels, plot_values, color=plot_colors, alpha=0.82)
    ax.set_ylabel("Mean RC-aligned absolute ISM normalized L1")
    ax.set_title("H3c primary endpoint: attribution asymmetry")
    ax.tick_params(axis="x", rotation=24)
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, plot_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.004, f"{value:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "h3c_exact_ism_comparison.png", dpi=180)
    fig.savefig(output_dir / "h3c_exact_ism_comparison.pdf")
    plt.close(fig)

    methods = ["exact_ism", "integrated_gradients", "deeplift", "gradient_shap"]
    method_labels = ["ISM", "IG", "DeepLIFT", "GradientSHAP"]
    x = np.arange(len(MODEL_ORDER))
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    width = 0.19
    for index, (method, method_label) in enumerate(zip(methods, method_labels)):
        grouped = attribution_summary[attribution_summary.method == method].set_index("model_type")
        values = [float(grouped.loc[model, "mean_absolute_normalized_l1"]) for model in MODEL_ORDER]
        ax.bar(x + (index - 1.5) * width, values, width=width, label=method_label, alpha=0.82)
    ax.set_xticks(x, labels, rotation=22, ha="right")
    ax.set_ylabel("Mean RC-aligned absolute normalized L1")
    ax.set_title("Attribution-method sensitivity analysis")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "attribution_method_sensitivity.png", dpi=180)
    fig.savefig(output_dir / "attribution_method_sensitivity.pdf")
    plt.close(fig)


def run(config_path: Path, output_dir: Path | None = None) -> Path:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("device") != "cpu":
        raise ValueError("Phase 2C pilot requires device: cpu")
    if config["dataset"].get("test_policy") != "sealed_no_model_access":
        raise ValueError("Phase 2C test policy must remain sealed_no_model_access")
    if config["training"]["model_types"] != MODEL_ORDER:
        raise ValueError("Phase 2C frozen model order or membership changed")
    manifest_errors = validate_manifest(ROOT / "configs" / "phase2_dataset_manifest.yaml")
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
    sequences = validation.sequence.tolist()
    rc_sequences = [reverse_complement(sequence) for sequence in sequences]
    labels = validation.label.to_numpy()

    shared_path = ROOT / config["phase2_sources"]["shared_ism_sample_ids"]
    shared = pd.read_csv(shared_path)
    if len(shared) != 100 or set(shared.label) != {0, 1}:
        raise ValueError("P2C requires the frozen 100-sample Phase 2A ISM set")
    selected = shared.merge(
        validation,
        on=["sample_id", "label", "chromosome"],
        how="left",
        validate="one_to_one",
    )
    if selected.sequence.isna().any():
        raise ValueError("A frozen Phase 2A ISM sample is absent from validation")
    selected.to_csv(output_dir / "shared_attribution_validation_samples.csv", index=False, encoding="utf-8")

    print(
        f"Python={platform.python_version()} | PyTorch={torch.__version__} | Captum=0.8.0 | "
        f"threads={torch.get_num_threads()} | train={int((modeling.split == 'train').sum())} | "
        f"validation={len(validation)} | test=sealed",
        flush=True,
    )

    models = {}
    run_rows = []
    prediction_rows = []
    batch_size = int(config["training"]["batch_size"])
    threshold = float(config["training"]["threshold"])
    for seed in config["training"]["seeds"]:
        for model_type in MODEL_ORDER:
            print(f"Training {model_type}, seed={seed}, selection=validation_loss, test=sealed", flush=True)
            run_dir = output_dir / "runs" / run_dir_name(model_type, int(seed))
            model, metadata = train_model_validation_only(model_type, int(seed), config, modeling, run_dir)
            models[(model_type, int(seed))] = model.eval()
            pf = predict_sequences(model, sequences, batch_size)
            pr = predict_sequences(model, rc_sequences, batch_size)
            consistency, each = prediction_consistency_metrics(pf, pr, threshold)
            reloaded, saved = load_checkpoint(run_dir / "best_checkpoint.pt")
            reload_pf = predict_sequences(reloaded, sequences, batch_size)
            reload_error = float(np.max(np.abs(pf - reload_pf)))
            if saved["architecture"] != ARCHITECTURE_BY_MODEL[model_type]:
                raise ValueError("Checkpoint architecture does not match frozen P2C design")
            run_rows.append(
                {
                    "model_type": model_type,
                    "position_encoding": saved["architecture"].removeprefix("transformer_"),
                    "seed": int(seed),
                    "best_epoch": metadata["best_epoch"],
                    "parameter_count": metadata["parameter_count"],
                    "training_seconds": metadata["training_seconds"],
                    **{f"validation_{key}": value for key, value in metadata["validation_metrics"].items()},
                    **consistency,
                    "checkpoint_reload_max_abs_probability_error": reload_error,
                    "test_evaluated": False,
                }
            )
            for index, row in validation.iterrows():
                prediction_rows.append(
                    {
                        "model_type": model_type,
                        "seed": int(seed),
                        "sample_id": row.sample_id,
                        "label": int(row.label),
                        "p_forward": float(pf[index]),
                        "p_rc": float(pr[index]),
                        "prediction_difference": float(each["prediction_difference"][index]),
                        "prediction_flip": int(each["prediction_flip"][index]),
                    }
                )
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(output_dir / "validation_prediction_pairs.csv", index=False, encoding="utf-8")

    interpretation = config["interpretation"]
    selected_sequences = selected.sequence.tolist()
    selected_rc = [reverse_complement(sequence) for sequence in selected_sequences]
    top_k = int(interpretation["top_k"])
    pred_limit = float(interpretation["prediction_consistent_max_abs_difference"])
    pearson_limit = float(interpretation["attribution_pearson_warning_below"])
    top8_limit = float(interpretation["attribution_top8_warning_below"])
    attribution_rows = []

    for (model_type, seed), model in models.items():
        run_dir = output_dir / "runs" / run_dir_name(model_type, seed)
        prediction_index = predictions[
            (predictions.model_type == model_type) & (predictions.seed == seed)
        ].set_index("sample_id")
        for method in interpretation["methods"]:
            print(f"Attribution {model_type}: {method}, validation n={len(selected)}", flush=True)
            if method == "exact_ism":
                f_matrix, f_signed, f_absolute, f_seconds = run_ism_for_sequences(
                    model,
                    selected_sequences,
                    int(interpretation["ism_batch_size"]),
                    interpretation["difference"],
                    f"{DISPLAY[model_type]} forward ISM",
                )
                r_matrix, r_signed, r_absolute, r_seconds = run_ism_for_sequences(
                    model,
                    selected_rc,
                    int(interpretation["ism_batch_size"]),
                    interpretation["difference"],
                    f"{DISPLAY[model_type]} RC ISM",
                )
            else:
                kwargs = {
                    "batch_size": int(interpretation["gradient_batch_size"]),
                    "n_steps": int(interpretation["integrated_gradients_steps"]),
                    "n_samples": int(interpretation["gradient_shap_samples"]),
                    "random_seed": int(interpretation["random_seed"]),
                }
                f_matrix, f_signed, f_absolute, f_seconds = run_captum_for_sequences(
                    model, selected_sequences, method=method, **kwargs
                )
                r_matrix, r_signed, r_absolute, r_seconds = run_captum_for_sequences(
                    model, selected_rc, method=method, **kwargs
                )
            aligned_matrix = align_rc_full_attribution(r_matrix)
            aligned_signed = align_rc_position_attribution(r_signed)
            aligned_absolute = align_rc_position_attribution(r_absolute)
            np.savez_compressed(
                run_dir / f"validation_{method}_attributions.npz",
                sample_ids=selected.sample_id.astype(str).to_numpy(dtype=str),
                forward_matrix=f_matrix,
                aligned_rc_matrix=aligned_matrix,
                forward_signed=f_signed,
                aligned_rc_signed=aligned_signed,
                forward_absolute=f_absolute,
                aligned_rc_absolute=aligned_absolute,
            )
            for index, row in selected.iterrows():
                absolute = attribution_metrics(f_absolute[index], aligned_absolute[index], top_k)
                signed = attribution_metrics(f_signed[index], aligned_signed[index], top_k)
                full_pearson, full_issue = safe_similarity(f_matrix[index], aligned_matrix[index], "pearson")
                pred = prediction_index.loc[row.sample_id]
                warning = (
                    (np.isfinite(absolute["pearson"]) and absolute["pearson"] < pearson_limit)
                    or absolute["top8_overlap"] < top8_limit
                )
                attribution_rows.append(
                    {
                        "model_type": model_type,
                        "seed": seed,
                        "method": method,
                        "sample_id": row.sample_id,
                        "label": int(row.label),
                        "selection_reason": row.selection_reason,
                        "prediction_difference": float(pred.prediction_difference),
                        "prediction_consistent": bool(float(pred.prediction_difference) <= pred_limit),
                        "attribution_warning": bool(warning),
                        **{f"absolute_{key}": value for key, value in absolute.items()},
                        **{f"signed_{key}": value for key, value in signed.items()},
                        "full_matrix_pearson": full_pearson,
                        "full_matrix_issue": full_issue or "",
                        "attribution_seconds_forward": f_seconds,
                        "attribution_seconds_rc": r_seconds,
                    }
                )

    attributions = pd.DataFrame(attribution_rows)
    attributions.to_csv(output_dir / "validation_attribution_results.csv", index=False, encoding="utf-8")
    summary_rows = []
    for (model_type, seed, method), group in attributions.groupby(
        ["model_type", "seed", "method"], observed=True
    ):
        consistent = group[group.prediction_consistent]
        summary_rows.append(
            {
                "model_type": model_type,
                "seed": int(seed),
                "method": method,
                "n_explained": int(len(group)),
                "n_prediction_consistent": int(len(consistent)),
                "mean_absolute_pearson": float(group.absolute_pearson.mean()),
                "median_absolute_pearson": float(group.absolute_pearson.median()),
                "mean_absolute_top8_overlap": float(group.absolute_top8_overlap.mean()),
                "mean_absolute_normalized_l1": float(group.absolute_normalized_l1.mean()),
                "mean_absolute_normalized_l2": float(group.absolute_normalized_l2.mean()),
                "attribution_warning_rate": float(group.attribution_warning.mean()),
                "warning_rate_within_prediction_consistent": (
                    float(consistent.attribution_warning.mean()) if len(consistent) else np.nan
                ),
                "attribution_seconds": float(
                    group.attribution_seconds_forward.iloc[0]
                    + group.attribution_seconds_rc.iloc[0]
                ),
            }
        )
    attribution_summary = pd.DataFrame(summary_rows)
    attribution_summary.to_csv(output_dir / "validation_attribution_summary.csv", index=False, encoding="utf-8")

    run_summary = pd.DataFrame(run_rows)
    ism_summary = attribution_summary[attribution_summary.method == "exact_ism"].drop(columns="method")
    run_summary = run_summary.merge(ism_summary, on=["model_type", "seed"], how="left", validate="one_to_one")
    run_summary.to_csv(output_dir / "run_summary.csv", index=False, encoding="utf-8")

    p2a_dir = ROOT / config["phase2_sources"]["phase2a_results_dir"]
    cnn_aug_summary = pd.read_csv(p2a_dir / "run_summary.csv").set_index("model_type").loc["CNN-Aug"]
    cnn_aug_attr = pd.read_csv(p2a_dir / "validation_attribution_results.csv")
    cnn_aug_attr = cnn_aug_attr[cnn_aug_attr.model_type == "CNN-Aug"][["sample_id", "absolute_normalized_l1"]]
    primary = attributions[
        (attributions.model_type == "Transformer-Aug-Absolute")
        & (attributions.method == "exact_ism")
    ][["sample_id", "absolute_normalized_l1"]]
    paired = primary.merge(cnn_aug_attr, on="sample_id", suffixes=("_transformer", "_cnn"), validate="one_to_one")
    h3c_cfg = config["h3c"]
    primary_bootstrap = paired_bootstrap_difference(
        paired.absolute_normalized_l1_transformer.to_numpy(),
        paired.absolute_normalized_l1_cnn.to_numpy(),
        int(h3c_cfg["paired_bootstrap_replicates"]),
        int(h3c_cfg["paired_bootstrap_seed"]),
    )
    contrast_rows = []
    exact = attributions[attributions.method == "exact_ism"]
    for left_model in ("Transformer-Aug-Absolute", "Transformer-Aug-Relative"):
        left = exact[exact.model_type == left_model][["sample_id", "absolute_normalized_l1"]]
        right = exact[exact.model_type == "Transformer-Aug-None"][["sample_id", "absolute_normalized_l1"]]
        pair = left.merge(right, on="sample_id", suffixes=("_left", "_right"), validate="one_to_one")
        contrast_rows.append(
            {
                "contrast": f"{left_model}_minus_Transformer-Aug-None",
                **paired_bootstrap_difference(
                    pair.absolute_normalized_l1_left.to_numpy(),
                    pair.absolute_normalized_l1_right.to_numpy(),
                    int(h3c_cfg["paired_bootstrap_replicates"]),
                    int(h3c_cfg["paired_bootstrap_seed"]),
                ),
            }
        )
    contrasts = pd.DataFrame(
        [
            {
                "contrast": "Transformer-Aug-Absolute_minus_CNN-Aug",
                **primary_bootstrap,
            },
            *contrast_rows,
        ]
    )
    contrasts.to_csv(output_dir / "h3c_paired_bootstrap.csv", index=False, encoding="utf-8")

    transformer_abs_auroc = float(
        run_summary.set_index("model_type").loc["Transformer-Aug-Absolute", "validation_auroc"]
    )
    cnn_aug_auroc = float(cnn_aug_summary.validation_auroc)
    performance_difference = transformer_abs_auroc - cnn_aug_auroc
    performance_matched = abs(performance_difference) <= float(
        h3c_cfg["performance_match_max_abs_auroc_difference"]
    )
    if performance_matched and primary_bootstrap["ci95_low"] > 0:
        h3c_status = "supported_in_validation_pilot"
    elif performance_matched and primary_bootstrap["ci95_high"] < 0:
        h3c_status = "opposite_direction_in_validation_pilot"
    else:
        h3c_status = "inconclusive_in_validation_pilot"
    h3c = {
        "hypothesis": "H3c",
        "scope": "single_seed_validation_pilot",
        "primary_endpoint": h3c_cfg["primary_endpoint"],
        "transformer_aug_absolute_validation_auroc": transformer_abs_auroc,
        "cnn_aug_validation_auroc": cnn_aug_auroc,
        "validation_auroc_difference": performance_difference,
        "performance_matched_within_frozen_margin": performance_matched,
        "cnn_aug_mean_normalized_l1": float(cnn_aug_summary.mean_absolute_normalized_l1),
        "transformer_aug_absolute_mean_normalized_l1": float(
            attribution_summary.set_index(["model_type", "method"]).loc[
                ("Transformer-Aug-Absolute", "exact_ism"), "mean_absolute_normalized_l1"
            ]
        ),
        "paired_difference_transformer_minus_cnn": primary_bootstrap,
        "status": h3c_status,
        "causal_claim_allowed": False,
    }
    write_json(h3c, output_dir / "h3c_assessment.json")

    method_rows = []
    for model_type in MODEL_ORDER:
        pivot = attributions[attributions.model_type == model_type].pivot(
            index="sample_id", columns="method", values="absolute_normalized_l1"
        )
        for method in ("integrated_gradients", "deeplift", "gradient_shap"):
            pearson, issue = safe_similarity(pivot.exact_ism, pivot[method], "pearson")
            spearman, rank_issue = safe_similarity(pivot.exact_ism, pivot[method], "spearman")
            method_rows.append(
                {
                    "model_type": model_type,
                    "comparison": f"exact_ism_vs_{method}",
                    "pearson": pearson,
                    "spearman": spearman,
                    "issues": ";".join(item for item in (issue, rank_issue) if item),
                }
            )
    pd.DataFrame(method_rows).to_csv(
        output_dir / "attribution_method_concordance.csv", index=False, encoding="utf-8"
    )

    expected_rows = len(MODEL_ORDER) * len(interpretation["methods"]) * len(selected)
    reload_pass = bool((run_summary.checkpoint_reload_max_abs_probability_error <= 1e-6).all())
    performance_floor_pass = bool(
        (run_summary.validation_auroc >= float(h3c_cfg["minimum_validation_auroc"])).all()
    )
    attribution_complete = len(attributions) == expected_rows and not attributions.absolute_normalized_l1.isna().any()
    all_gates = reload_pass and performance_floor_pass and attribution_complete
    gate = {
        "phase": "Phase 2C Transformer position-encoding pilot",
        "analysis_split": "validation",
        "test_seal_intact": True,
        "checkpoint_reload_gate_passed": reload_pass,
        "validation_auroc_floor_gate_passed": performance_floor_pass,
        "all_attribution_methods_complete": attribution_complete,
        "expected_attribution_rows": expected_rows,
        "observed_attribution_rows": int(len(attributions)),
        "all_phase2c_completion_gates_passed": all_gates,
        "h3c_status": h3c_status,
        "decision": {
            "proceed_to_phase2d_biological_validation": all_gates,
            "unseal_test_now": False,
            "start_phase2_full_now": False,
        },
    }
    write_json(gate, output_dir / "phase2c_gate_assessment.json")
    create_figures(run_summary, attribution_summary, h3c, output_dir / "figures")

    completion = {
        "status": "completed_validation_only" if all_gates else "completed_with_failed_gate",
        "elapsed_seconds": time.perf_counter() - started,
        "models": MODEL_ORDER,
        "methods": interpretation["methods"],
        "shared_attribution_samples": int(len(selected)),
        "test_predictions_generated": False,
        "test_metrics_generated": False,
        "all_phase2c_completion_gates_passed": all_gates,
    }
    write_json(completion, output_dir / "pilot_completion.json")
    print(run_summary.to_string(index=False), flush=True)
    print(json.dumps(h3c, indent=2), flush=True)
    print(f"Completed Phase 2C in {completion['elapsed_seconds']:.2f}s: {output_dir}", flush=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Run frozen Phase 2C Transformer pilot")
    parser.add_argument("--config", default="configs/phase2c_pilot.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run(
        ROOT / args.config,
        ROOT / args.output_dir if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
