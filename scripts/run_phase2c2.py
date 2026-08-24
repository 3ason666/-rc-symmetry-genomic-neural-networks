from __future__ import annotations

import argparse
import copy
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
from src.dna_utils import align_rc_full_attribution, align_rc_position_attribution, reverse_complement
from src.interpret import run_captum_for_sequences, run_ism_for_sequences
from src.metrics import prediction_consistency_metrics, safe_similarity
from src.models import build_model
from src.training import DEVICE, predict_sequences, train_model_validation_only


MODEL_ORDER = [
    "CNN-Aug",
    "Transformer-Raw",
    "Transformer-Aug-None",
    "Transformer-Aug-Absolute",
    "Transformer-Aug-Relative",
]
ARCHITECTURE_BY_MODEL = {
    "CNN-Aug": "standard",
    "Transformer-Raw": "transformer_absolute",
    "Transformer-Aug-None": "transformer_none",
    "Transformer-Aug-Absolute": "transformer_absolute",
    "Transformer-Aug-Relative": "transformer_relative",
}
DISPLAY = {
    "CNN-Aug": "CNN-Aug",
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


def config_for_model(config: dict, model_type: str) -> dict:
    current = copy.deepcopy(config)
    if model_type == "CNN-Aug":
        current["model"] = copy.deepcopy(config["cnn_model"])
        current["training"]["learning_rate"] = float(config["training"]["cnn_learning_rate"])
    else:
        current["model"] = copy.deepcopy(config["transformer_model"])
        current["training"]["learning_rate"] = float(
            config["training"]["transformer_learning_rate"]
        )
    return current


def load_checkpoint(path: Path):
    saved = torch.load(path, map_location=DEVICE, weights_only=True)
    model = build_model(saved["model_config"], architecture=saved["architecture"]).to(DEVICE)
    model.load_state_dict(saved["model_state"])
    return model.eval(), saved


def hierarchical_paired_bootstrap(
    differences: pd.DataFrame,
    replicates: int,
    seed: int,
) -> dict:
    required = {"seed", "sample_id", "difference"}
    if required - set(differences.columns):
        raise ValueError("Hierarchical bootstrap input is missing required columns")
    groups = {
        int(seed_value): group.difference.to_numpy(dtype=float)
        for seed_value, group in differences.groupby("seed", sort=True)
    }
    if not groups or any(values.size == 0 for values in groups.values()):
        raise ValueError("Hierarchical bootstrap requires non-empty seed groups")
    seed_values = np.array(sorted(groups), dtype=int)
    seed_means = np.array([groups[value].mean() for value in seed_values], dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled_seeds = rng.choice(seed_values, size=len(seed_values), replace=True)
        sampled_means = []
        for sampled_seed in sampled_seeds:
            values = groups[int(sampled_seed)]
            sampled = rng.choice(values, size=values.size, replace=True)
            sampled_means.append(float(sampled.mean()))
        estimates[replicate] = float(np.mean(sampled_means))
    return {
        "n_seeds": int(len(seed_values)),
        "n_seed_sample_pairs": int(len(differences)),
        "estimate": float(seed_means.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
    }


def create_figures(run_summary: pd.DataFrame, attr_summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [DISPLAY[model] for model in MODEL_ORDER]
    x = np.arange(len(MODEL_ORDER))

    fig, ax = plt.subplots(figsize=(9.4, 5.2), constrained_layout=True)
    for model_index, model in enumerate(MODEL_ORDER):
        values = run_summary.loc[run_summary.model_type.eq(model), "validation_auroc"].to_numpy()
        ax.scatter(np.full(values.size, model_index), values, color=COLORS[model], s=42, zorder=3)
        ax.hlines(values.mean(), model_index - 0.24, model_index + 0.24, color="black", linewidth=2)
    ax.axhline(0.80, color="#7A5A00", linestyle="--", linewidth=1, label="Relative floor")
    ax.set_xticks(x, labels, rotation=22, ha="right")
    ax.set_ylabel("Validation AUROC")
    ax.set_title("P2C.2 multi-seed classification performance (test sealed)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "validation_auroc_by_seed.png", dpi=180)
    fig.savefig(output_dir / "validation_auroc_by_seed.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 5.2), constrained_layout=True)
    for model_index, model in enumerate(MODEL_ORDER):
        values = run_summary.loc[
            run_summary.model_type.eq(model), "prediction_mean_absolute_difference"
        ].to_numpy()
        ax.scatter(np.full(values.size, model_index), values, color=COLORS[model], s=42, zorder=3)
        ax.hlines(values.mean(), model_index - 0.24, model_index + 0.24, color="black", linewidth=2)
    ax.axhline(0.01, color="#991B1B", linestyle="--", linewidth=1, label="Consistency threshold")
    ax.set_xticks(x, labels, rotation=22, ha="right")
    ax.set_ylabel("Mean |p(S)-p(RC(S))|")
    ax.set_title("P2C.2 prediction asymmetry by seed")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "prediction_asymmetry_by_seed.png", dpi=180)
    fig.savefig(output_dir / "prediction_asymmetry_by_seed.pdf")
    plt.close(fig)

    exact = attr_summary[attr_summary.method.eq("exact_ism")]
    fig, ax = plt.subplots(figsize=(9.4, 5.2), constrained_layout=True)
    for model_index, model in enumerate(MODEL_ORDER):
        values = exact.loc[
            exact.model_type.eq(model), "mean_absolute_normalized_l1"
        ].to_numpy()
        ax.scatter(np.full(values.size, model_index), values, color=COLORS[model], s=42, zorder=3)
        ax.hlines(values.mean(), model_index - 0.24, model_index + 0.24, color="black", linewidth=2)
    ax.set_xticks(x, labels, rotation=22, ha="right")
    ax.set_ylabel("Mean RC-aligned exact ISM normalized L1")
    ax.set_title("P2C.2 attribution asymmetry by seed")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output_dir / "exact_ism_asymmetry_by_seed.png", dpi=180)
    fig.savefig(output_dir / "exact_ism_asymmetry_by_seed.pdf")
    plt.close(fig)


def run(config_path: Path, output_dir: Path | None = None) -> Path:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("device") != "cpu":
        raise ValueError("P2C.2 is frozen to the current PC CPU")
    if config["dataset"].get("test_policy") != "sealed_no_model_access":
        raise ValueError("P2C.2 requires test sealing")
    if config["training"]["model_types"] != MODEL_ORDER:
        raise ValueError("P2C.2 frozen model order changed")
    if config["training"]["seeds"] != [42, 123, 2026]:
        raise ValueError("P2C.2 frozen seeds changed")
    if not config["h3c"].get("none_is_non_gating"):
        raise ValueError("P2C.2 protocol amendment for None is missing")
    manifest_errors = validate_manifest(ROOT / "configs" / "phase2_dataset_manifest.yaml")
    if manifest_errors:
        raise ValueError("Frozen Phase 2 manifest is invalid: " + "; ".join(manifest_errors))

    torch.set_num_threads(max(1, int(config.get("cpu_threads", 1))))
    output_dir = output_dir or ROOT / "results" / config["project_name"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite P2C.2 directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "resolved_config.yaml")
    modeling, contract = load_modeling_view(config)
    contract.update(
        {
            "protocol_revision": config["protocol_revision"],
            "validation_role": "one_time_multiseed_confirmation",
            "test_predictions_generated": False,
            "test_metrics_generated": False,
        }
    )
    write_json(contract, output_dir / "sealed_test_policy.json")
    write_json(config["amendment"], output_dir / "protocol_amendment.json")

    validation = modeling[modeling.split.eq("validation")].reset_index(drop=True)
    sequences = validation.sequence.tolist()
    rc_sequences = [reverse_complement(sequence) for sequence in sequences]
    shared = pd.read_csv(ROOT / config["phase2_sources"]["shared_ism_sample_ids"])
    if len(shared) != 100 or set(shared.label) != {0, 1}:
        raise ValueError("P2C.2 requires the frozen 100-sample attribution set")
    selected = shared.merge(
        validation,
        on=["sample_id", "label", "chromosome"],
        how="left",
        validate="one_to_one",
    )
    if selected.sequence.isna().any():
        raise ValueError("A frozen attribution sample is absent from validation")
    selected.to_csv(output_dir / "shared_attribution_validation_samples.csv", index=False, encoding="utf-8")

    print(
        f"Python={platform.python_version()} | PyTorch={torch.__version__} | device={DEVICE} | "
        f"threads={torch.get_num_threads()} | train={int(modeling.split.eq('train').sum())} | "
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
            print(f"Training {model_type}, seed={seed}, test=sealed", flush=True)
            current = config_for_model(config, model_type)
            run_dir = output_dir / "runs" / run_dir_name(model_type, int(seed))
            model, metadata = train_model_validation_only(
                model_type, int(seed), current, modeling, run_dir
            )
            models[(model_type, int(seed))] = model.eval()
            pf = predict_sequences(model, sequences, batch_size)
            pr = predict_sequences(model, rc_sequences, batch_size)
            consistency, each = prediction_consistency_metrics(pf, pr, threshold)
            reloaded, saved = load_checkpoint(run_dir / "best_checkpoint.pt")
            reload_pf = predict_sequences(reloaded, sequences, batch_size)
            reload_error = float(np.max(np.abs(pf - reload_pf)))
            if saved["architecture"] != ARCHITECTURE_BY_MODEL[model_type]:
                raise ValueError("Checkpoint architecture differs from P2C.2 protocol")
            run_rows.append(
                {
                    "model_type": model_type,
                    "seed": int(seed),
                    "architecture": saved["architecture"],
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
    run_summary = pd.DataFrame(run_rows)
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
            predictions.model_type.eq(model_type) & predictions.seed.eq(seed)
        ].set_index("sample_id")
        for method in interpretation["methods"]:
            print(f"Attribution {model_type}, seed={seed}: {method}, n={len(selected)}", flush=True)
            if method == "exact_ism":
                f_matrix, f_signed, f_absolute, f_seconds = run_ism_for_sequences(
                    model,
                    selected_sequences,
                    int(interpretation["ism_batch_size"]),
                    interpretation["difference"],
                    f"{DISPLAY[model_type]} seed {seed} forward ISM",
                )
                r_matrix, r_signed, r_absolute, r_seconds = run_ism_for_sequences(
                    model,
                    selected_rc,
                    int(interpretation["ism_batch_size"]),
                    interpretation["difference"],
                    f"{DISPLAY[model_type]} seed {seed} RC ISM",
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
                full_pearson, full_issue = safe_similarity(
                    f_matrix[index], aligned_matrix[index], "pearson"
                )
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
    attribution_summary.to_csv(
        output_dir / "validation_attribution_summary.csv", index=False, encoding="utf-8"
    )
    exact_summary = attribution_summary[attribution_summary.method.eq("exact_ism")].drop(
        columns="method"
    )
    run_summary = run_summary.merge(
        exact_summary, on=["model_type", "seed"], how="left", validate="one_to_one"
    )
    run_summary.to_csv(output_dir / "run_summary.csv", index=False, encoding="utf-8")

    exact = attributions[attributions.method.eq("exact_ism")]
    left = exact[exact.model_type.eq("Transformer-Aug-Absolute")][
        ["seed", "sample_id", "absolute_normalized_l1"]
    ]
    right = exact[exact.model_type.eq("CNN-Aug")][
        ["seed", "sample_id", "absolute_normalized_l1"]
    ]
    paired = left.merge(
        right,
        on=["seed", "sample_id"],
        suffixes=("_transformer", "_cnn"),
        validate="one_to_one",
    )
    paired["difference"] = (
        paired.absolute_normalized_l1_transformer - paired.absolute_normalized_l1_cnn
    )
    paired.to_csv(output_dir / "h3c_paired_differences.csv", index=False, encoding="utf-8")
    h3c_cfg = config["h3c"]
    hierarchical = hierarchical_paired_bootstrap(
        paired[["seed", "sample_id", "difference"]],
        int(h3c_cfg["hierarchical_bootstrap_replicates"]),
        int(h3c_cfg["hierarchical_bootstrap_seed"]),
    )
    per_seed = (
        paired.groupby("seed").difference.agg(["count", "mean", "std"]).reset_index()
    )
    per_seed.to_csv(output_dir / "h3c_per_seed_effects.csv", index=False, encoding="utf-8")

    auroc = run_summary.pivot(index="seed", columns="model_type", values="validation_auroc")
    performance_difference = auroc["Transformer-Aug-Absolute"] - auroc["CNN-Aug"]
    performance_match = performance_difference.abs() <= float(
        h3c_cfg["performance_match_max_abs_auroc_difference"]
    )
    relative_floor = auroc["Transformer-Aug-Relative"] >= float(
        h3c_cfg["relative_minimum_validation_auroc_every_seed"]
    )
    performance_table = pd.DataFrame(
        {
            "seed": auroc.index.astype(int),
            "cnn_aug_auroc": auroc["CNN-Aug"].to_numpy(),
            "transformer_aug_absolute_auroc": auroc["Transformer-Aug-Absolute"].to_numpy(),
            "absolute_minus_cnn_auroc": performance_difference.to_numpy(),
            "absolute_performance_matched": performance_match.to_numpy(),
            "transformer_aug_relative_auroc": auroc["Transformer-Aug-Relative"].to_numpy(),
            "relative_floor_passed": relative_floor.to_numpy(),
        }
    )
    performance_table.to_csv(
        output_dir / "h3c_performance_matching_by_seed.csv", index=False, encoding="utf-8"
    )
    if performance_match.all() and hierarchical["ci95_low"] > 0:
        h3c_status = "supported_in_multiseed_validation"
    elif performance_match.all() and hierarchical["ci95_high"] < 0:
        h3c_status = "opposite_direction_in_multiseed_validation"
    else:
        h3c_status = "inconclusive_in_multiseed_validation"
    h3c = {
        "hypothesis": "H3c",
        "scope": "three_seed_validation_confirmation",
        "primary_endpoint": h3c_cfg["primary_endpoint"],
        "seeds": [42, 123, 2026],
        "performance_match_for_every_seed": bool(performance_match.all()),
        "relative_floor_passed_for_every_seed": bool(relative_floor.all()),
        "none_role": "non_gating_structural_negative_control",
        "hierarchical_paired_difference_transformer_minus_cnn": hierarchical,
        "status": h3c_status,
        "test_seal_intact": True,
        "causal_claim_allowed": bool(performance_match.all()),
    }
    write_json(h3c, output_dir / "h3c_multiseed_assessment.json")

    method_rows = []
    for (model_type, seed), group in attributions.groupby(["model_type", "seed"], observed=True):
        pivot = group.pivot(index="sample_id", columns="method", values="absolute_normalized_l1")
        for method in ("integrated_gradients", "deeplift", "gradient_shap"):
            pearson, issue = safe_similarity(pivot.exact_ism, pivot[method], "pearson")
            spearman, rank_issue = safe_similarity(pivot.exact_ism, pivot[method], "spearman")
            method_rows.append(
                {
                    "model_type": model_type,
                    "seed": int(seed),
                    "comparison": f"exact_ism_vs_{method}",
                    "pearson": pearson,
                    "spearman": spearman,
                    "issues": ";".join(item for item in (issue, rank_issue) if item),
                }
            )
    pd.DataFrame(method_rows).to_csv(
        output_dir / "attribution_method_concordance.csv", index=False, encoding="utf-8"
    )

    expected_rows = len(MODEL_ORDER) * len(config["training"]["seeds"]) * len(
        interpretation["methods"]
    ) * len(selected)
    reload_pass = bool((run_summary.checkpoint_reload_max_abs_probability_error <= 1e-6).all())
    attribution_complete = bool(
        len(attributions) == expected_rows
        and not attributions.absolute_normalized_l1.isna().any()
    )
    all_gates = bool(
        reload_pass
        and attribution_complete
        and performance_match.all()
        and relative_floor.all()
    )
    gate = {
        "phase": "Phase 2C.2 c4 multi-seed validation confirmation",
        "analysis_split": "validation",
        "test_seal_intact": True,
        "checkpoint_reload_gate_passed": reload_pass,
        "all_attribution_methods_complete": attribution_complete,
        "expected_attribution_rows": int(expected_rows),
        "observed_attribution_rows": int(len(attributions)),
        "absolute_performance_match_every_seed": bool(performance_match.all()),
        "relative_auroc_floor_every_seed": bool(relative_floor.all()),
        "none_non_gating": True,
        "all_phase2c2_confirmation_gates_passed": all_gates,
        "h3c_status": h3c_status,
        "decision": {
            "proceed_to_phase2d_biological_validation": all_gates,
            "unseal_test_now": False,
            "start_phase2_full_now": False,
        },
    }
    write_json(gate, output_dir / "phase2c2_gate_assessment.json")
    create_figures(run_summary, attribution_summary, output_dir / "figures")
    completion = {
        "status": "completed_validation_only" if all_gates else "completed_with_failed_gate",
        "elapsed_seconds": time.perf_counter() - started,
        "models": MODEL_ORDER,
        "seeds": config["training"]["seeds"],
        "methods": interpretation["methods"],
        "shared_attribution_samples_per_model_seed": int(len(selected)),
        "test_predictions_generated": False,
        "test_metrics_generated": False,
        "all_phase2c2_confirmation_gates_passed": all_gates,
    }
    write_json(completion, output_dir / "phase2c2_completion.json")
    print(run_summary.to_string(index=False), flush=True)
    print(json.dumps(h3c, indent=2), flush=True)
    print(f"Completed P2C.2 in {completion['elapsed_seconds']:.2f}s: {output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen P2C.2 c4 multi-seed validation")
    parser.add_argument("--config", default="configs/phase2c2_c4_multiseed.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run(ROOT / args.config, ROOT / args.output_dir if args.output_dir else None)


if __name__ == "__main__":
    main()
