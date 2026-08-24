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
from scripts.run_phase2c3 import build_cv_views
from scripts.validate_phase2_manifest import validate_manifest
from src.dna_utils import (
    align_rc_full_attribution,
    align_rc_position_attribution,
    batch_one_hot_encode,
    reverse_complement,
)
from src.interpret import run_captum_for_sequences, run_ism_for_sequences
from src.metrics import prediction_consistency_metrics, safe_similarity
from src.models import (
    build_model,
    reverse_complement_feature_pairs,
    reverse_complement_one_hot,
)
from src.training import DEVICE, predict_sequences, train_model_validation_only


MODEL_ORDER = ["CNN-Raw", "CNN-Aug", "CNN-RCPS"]
MODEL_COLORS = {"CNN-Raw": "#2563EB", "CNN-Aug": "#D97706", "CNN-RCPS": "#059669"}
CONTRASTS = [
    ("CNN-Aug", "CNN-Raw", "CNN-Aug_minus_CNN-Raw"),
    ("CNN-RCPS", "CNN-Aug", "CNN-RCPS_minus_CNN-Aug"),
]


def run_dir_name(fold_id: str, model_type: str, seed: int) -> str:
    model = model_type.lower().replace("-", "_")
    return f"{fold_id}__{model}__seed_{seed}"


def load_checkpoint(path: Path):
    saved = torch.load(path, map_location=DEVICE, weights_only=True)
    model = build_model(saved["model_config"], architecture=saved["architecture"]).to(DEVICE)
    model.load_state_dict(saved["model_state"])
    return model.eval(), saved


@torch.inference_mode()
def feature_equivariance_max_error(model, sequences: list[str], batch_size: int) -> float:
    maximum = 0.0
    for start in range(0, len(sequences), batch_size):
        inputs = batch_one_hot_encode(sequences[start : start + batch_size]).to(DEVICE)
        observed = model.features(reverse_complement_one_hot(inputs))
        expected = reverse_complement_feature_pairs(model.features(inputs))
        maximum = max(maximum, float(torch.max(torch.abs(observed - expected)).item()))
    return maximum


def validate_frozen_config(config: dict) -> None:
    if config.get("protocol_revision") != "phase2d_rcps_confirmatory_v1":
        raise ValueError("P2D protocol revision changed")
    if config.get("device") != "cpu":
        raise ValueError("P2D v1 is frozen to CPU")
    if config["dataset"].get("source_split") != "train":
        raise ValueError("P2D may use only the original train split")
    if config["dataset"].get("forbidden_model_splits") != ["validation", "test"]:
        raise ValueError("P2D forbidden split list changed")
    if config["dataset"].get("test_policy") != "sealed_no_model_access":
        raise ValueError("P2D requires a sealed test set")
    if config["training"].get("model_types") != MODEL_ORDER:
        raise ValueError("P2D frozen model order changed")
    if config["training"].get("seeds") != [42, 123, 2026]:
        raise ValueError("P2D frozen seeds changed")
    if config["interpretation"].get("methods") != [
        "exact_ism",
        "integrated_gradients",
        "deeplift",
        "gradient_shap",
    ]:
        raise ValueError("P2D frozen attribution methods changed")
    decision = config["decision"]
    if decision.get("access_original_validation") or decision.get("unseal_test_now"):
        raise ValueError("P2D config attempts to access a forbidden split")
    if decision.get("allow_new_transformer_tuning"):
        raise ValueError("P2D may not restart Transformer tuning")


def freeze_attribution_samples(
    views: dict[str, pd.DataFrame], config: dict, output_dir: Path
) -> pd.DataFrame:
    path = output_dir / "frozen_attribution_samples.csv"
    n = int(config["interpretation"]["samples_per_label_per_fold"])
    base_seed = int(config["interpretation"]["sample_seed"])
    rows = []
    for fold_index, (fold_id, view) in enumerate(views.items()):
        holdout = view[view.split.eq("validation")].copy()
        for label in (0, 1):
            pool = holdout[holdout.label.eq(label)]
            if len(pool) < n:
                raise ValueError(f"{fold_id} has fewer than {n} label={label} samples")
            selected = pool.sample(n=n, random_state=base_seed + 1009 * fold_index + label)
            selected = selected.sort_values("sample_id", kind="mergesort").copy()
            selected["fold_id"] = fold_id
            selected["selection_reason"] = f"frozen_random_label_{label}"
            rows.append(selected)
    frozen = pd.concat(rows, ignore_index=True)
    frozen = frozen.sort_values(["fold_id", "label", "sample_id"], kind="mergesort")
    columns = [
        "fold_id",
        "sample_id",
        "pair_id",
        "canonical_key",
        "chromosome",
        "label",
        "sequence",
        "selection_reason",
    ]
    frozen = frozen[columns].reset_index(drop=True)
    if path.exists():
        existing = pd.read_csv(path)
        pd.testing.assert_frame_equal(existing, frozen, check_dtype=False)
    else:
        frozen.to_csv(path, index=False, encoding="utf-8")
    return frozen


def hierarchical_paired_bootstrap(
    differences: pd.DataFrame, replicates: int, seed: int
) -> dict:
    required = {"fold_id", "seed", "sample_id", "difference"}
    if required - set(differences.columns):
        raise ValueError("P2D bootstrap input lacks paired hierarchy columns")
    groups = {
        (str(fold_id), int(seed_value)): group.difference.to_numpy(dtype=float)
        for (fold_id, seed_value), group in differences.groupby(
            ["fold_id", "seed"], sort=True, observed=True
        )
    }
    folds = sorted({key[0] for key in groups})
    seeds_by_fold = {
        fold_id: np.array(sorted(key[1] for key in groups if key[0] == fold_id), dtype=int)
        for fold_id in folds
    }
    if not folds or any(len(values) == 0 for values in groups.values()):
        raise ValueError("P2D bootstrap requires non-empty fold-seed groups")
    cell_means = np.array([values.mean() for values in groups.values()], dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    fold_array = np.array(folds, dtype=object)
    for replicate in range(replicates):
        sampled_folds = rng.choice(fold_array, size=len(folds), replace=True)
        fold_means = []
        for fold_id in sampled_folds:
            seed_values = seeds_by_fold[str(fold_id)]
            sampled_seeds = rng.choice(seed_values, size=len(seed_values), replace=True)
            seed_means = []
            for seed_value in sampled_seeds:
                values = groups[(str(fold_id), int(seed_value))]
                seed_means.append(float(rng.choice(values, size=len(values), replace=True).mean()))
            fold_means.append(float(np.mean(seed_means)))
        estimates[replicate] = float(np.mean(fold_means))
    return {
        "n_folds": int(len(folds)),
        "n_fold_seed_cells": int(len(groups)),
        "n_paired_rows": int(len(differences)),
        "estimate": float(cell_means.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
    }


def summit_metrics(values: np.ndarray, start: int, end: int, center: int) -> dict:
    scores = np.abs(np.asarray(values, dtype=float))
    total = float(scores.sum())
    coordinates = np.arange(scores.size, dtype=float)
    return {
        "summit_central_mass_fraction": (
            float(scores[start:end].sum() / total) if total > 0 else np.nan
        ),
        "summit_weighted_distance": (
            float(np.sum(scores * np.abs(coordinates - center)) / total)
            if total > 0
            else np.nan
        ),
    }


def aggregate_training_outputs(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, predictions = [], []
    for path in sorted((output_dir / "runs").glob("*/run_summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    for path in sorted((output_dir / "runs").glob("*/holdout_prediction_pairs.csv")):
        predictions.append(pd.read_csv(path))
    summary = pd.DataFrame(summaries)
    pairs = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    if len(summary):
        summary.sort_values(["fold_id", "seed", "model_type"]).to_csv(
            output_dir / "training_run_summary.csv", index=False, encoding="utf-8"
        )
    if len(pairs):
        pairs.sort_values(["fold_id", "seed", "model_type", "sample_id"]).to_csv(
            output_dir / "holdout_prediction_pairs.csv", index=False, encoding="utf-8"
        )
    return summary, pairs


def train_stage(
    views: dict[str, pd.DataFrame], config: dict, output_dir: Path, max_runs: int | None
) -> int:
    completed_now = 0
    batch_size = int(config["training"]["batch_size"])
    threshold = float(config["training"]["threshold"])
    gate = config["invariance_gates"]
    for fold_id, view in views.items():
        holdout = view[view.split.eq("validation")].reset_index(drop=True)
        sequences = holdout.sequence.tolist()
        rc_sequences = [reverse_complement(sequence) for sequence in sequences]
        for seed in config["training"]["seeds"]:
            for model_type in MODEL_ORDER:
                run_dir = output_dir / "runs" / run_dir_name(fold_id, model_type, int(seed))
                summary_path = run_dir / "run_summary.json"
                if summary_path.exists() and (run_dir / "holdout_prediction_pairs.csv").exists():
                    continue
                if max_runs is not None and completed_now >= max_runs:
                    aggregate_training_outputs(output_dir)
                    return completed_now
                print(f"P2D train {fold_id} {model_type} seed={seed} test=sealed", flush=True)
                model, metadata = train_model_validation_only(
                    model_type, int(seed), config, view, run_dir
                )
                forward = predict_sequences(model, sequences, batch_size)
                reverse = predict_sequences(model, rc_sequences, batch_size)
                consistency, each = prediction_consistency_metrics(forward, reverse, threshold)
                restored, saved = load_checkpoint(run_dir / "best_checkpoint.pt")
                restored_forward = predict_sequences(restored, sequences, batch_size)
                reload_error = float(np.max(np.abs(forward - restored_forward)))
                rcps_prediction_error = (
                    float(np.max(np.abs(restored_forward - reverse)))
                    if model_type == "CNN-RCPS"
                    else np.nan
                )
                feature_error = (
                    feature_equivariance_max_error(restored, sequences, batch_size)
                    if model_type == "CNN-RCPS"
                    else np.nan
                )
                sample_tensor = batch_one_hot_encode(sequences[: min(32, len(sequences))])
                double_rc_exact = bool(
                    torch.equal(
                        reverse_complement_one_hot(reverse_complement_one_hot(sample_tensor)),
                        sample_tensor,
                    )
                )
                prediction_rows = []
                for index, row in holdout.iterrows():
                    prediction_rows.append(
                        {
                            "fold_id": fold_id,
                            "model_type": model_type,
                            "seed": int(seed),
                            "sample_id": row.sample_id,
                            "pair_id": row.pair_id,
                            "canonical_key": row.canonical_key,
                            "chromosome": row.chromosome,
                            "label": int(row.label),
                            "p_forward": float(restored_forward[index]),
                            "p_rc": float(reverse[index]),
                            "prediction_difference": float(each["prediction_difference"][index]),
                            "prediction_flip": int(each["prediction_flip"][index]),
                        }
                    )
                pd.DataFrame(prediction_rows).to_csv(
                    run_dir / "holdout_prediction_pairs.csv", index=False, encoding="utf-8"
                )
                summary = {
                    "fold_id": fold_id,
                    "model_type": model_type,
                    "seed": int(seed),
                    "architecture": saved["architecture"],
                    "best_epoch": int(metadata["best_epoch"]),
                    "parameter_count": int(metadata["parameter_count"]),
                    "training_seconds": float(metadata["training_seconds"]),
                    **{
                        f"holdout_{key}": value
                        for key, value in metadata["validation_metrics"].items()
                    },
                    **consistency,
                    "checkpoint_reload_max_abs_probability_error": reload_error,
                    "rcps_prediction_max_abs_probability_error": rcps_prediction_error,
                    "rcps_feature_equivariance_max_abs_error": feature_error,
                    "double_rc_input_exact": double_rc_exact,
                    "reload_gate_passed": reload_error
                    <= float(gate["checkpoint_reload_max_abs_tolerance"]),
                    "rcps_prediction_gate_passed": (
                        True
                        if model_type != "CNN-RCPS"
                        else rcps_prediction_error
                        <= float(gate["prediction_max_abs_tolerance"])
                    ),
                    "rcps_feature_gate_passed": (
                        True
                        if model_type != "CNN-RCPS"
                        else feature_error
                        <= float(gate["feature_equivariance_max_abs_tolerance"])
                    ),
                    "original_validation_rows_evaluated": 0,
                    "test_rows_evaluated": 0,
                }
                write_json(summary, summary_path)
                completed_now += 1
                aggregate_training_outputs(output_dir)
    return completed_now


def aggregate_attribution_outputs(output_dir: Path) -> pd.DataFrame:
    parts = [
        pd.read_csv(path)
        for path in sorted((output_dir / "runs").glob("*/attribution_results.csv"))
    ]
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(frame):
        frame.sort_values(
            ["fold_id", "seed", "model_type", "method", "sample_id"]
        ).to_csv(output_dir / "attribution_results.csv", index=False, encoding="utf-8")
    return frame


def attribution_stage(
    frozen: pd.DataFrame, config: dict, output_dir: Path, max_runs: int | None
) -> int:
    completed_now = 0
    interpretation = config["interpretation"]
    bio = config["biological_validity"]
    for fold_id in [fold["fold_id"] for fold in config["cross_validation"]["folds"]]:
        selected = frozen[frozen.fold_id.eq(fold_id)].reset_index(drop=True)
        sequences = selected.sequence.tolist()
        rc_sequences = [reverse_complement(sequence) for sequence in sequences]
        for seed in config["training"]["seeds"]:
            for model_type in MODEL_ORDER:
                run_dir = output_dir / "runs" / run_dir_name(fold_id, model_type, int(seed))
                complete_path = run_dir / "attribution_complete.json"
                if complete_path.exists() and (run_dir / "attribution_results.csv").exists():
                    continue
                if max_runs is not None and completed_now >= max_runs:
                    aggregate_attribution_outputs(output_dir)
                    return completed_now
                if not (run_dir / "run_summary.json").exists():
                    raise RuntimeError(f"Training is incomplete for {run_dir.name}")
                model, _ = load_checkpoint(run_dir / "best_checkpoint.pt")
                prediction_index = pd.read_csv(
                    run_dir / "holdout_prediction_pairs.csv"
                ).set_index("sample_id")
                rows = []
                exact_matrix_max_error = np.nan
                for method in interpretation["methods"]:
                    print(
                        f"P2D attribution {fold_id} {model_type} seed={seed} "
                        f"method={method} n={len(selected)}",
                        flush=True,
                    )
                    if method == "exact_ism":
                        f_matrix, f_signed, f_absolute, f_seconds = run_ism_for_sequences(
                            model,
                            sequences,
                            int(interpretation["ism_batch_size"]),
                            interpretation["difference"],
                            f"{run_dir.name} forward ISM",
                        )
                        r_matrix, r_signed, r_absolute, r_seconds = run_ism_for_sequences(
                            model,
                            rc_sequences,
                            int(interpretation["ism_batch_size"]),
                            interpretation["difference"],
                            f"{run_dir.name} RC ISM",
                        )
                    else:
                        kwargs = {
                            "batch_size": int(interpretation["gradient_batch_size"]),
                            "n_steps": int(interpretation["integrated_gradients_steps"]),
                            "n_samples": int(interpretation["gradient_shap_samples"]),
                            "random_seed": int(interpretation["random_seed"]),
                        }
                        f_matrix, f_signed, f_absolute, f_seconds = run_captum_for_sequences(
                            model, sequences, method=method, **kwargs
                        )
                        r_matrix, r_signed, r_absolute, r_seconds = run_captum_for_sequences(
                            model, rc_sequences, method=method, **kwargs
                        )
                    aligned_matrix = align_rc_full_attribution(r_matrix)
                    aligned_signed = align_rc_position_attribution(r_signed)
                    aligned_absolute = align_rc_position_attribution(r_absolute)
                    np.savez_compressed(
                        run_dir / f"{method}_attributions.npz",
                        sample_ids=selected.sample_id.astype(str).to_numpy(dtype=str),
                        forward_matrix=f_matrix,
                        aligned_rc_matrix=aligned_matrix,
                        forward_signed=f_signed,
                        aligned_rc_signed=aligned_signed,
                        forward_absolute=f_absolute,
                        aligned_rc_absolute=aligned_absolute,
                    )
                    if method == "exact_ism":
                        exact_matrix_max_error = float(
                            np.max(np.abs(f_matrix - aligned_matrix))
                        )
                    for index, sample in selected.iterrows():
                        absolute = attribution_metrics(
                            f_absolute[index], aligned_absolute[index], int(interpretation["top_k"])
                        )
                        signed = attribution_metrics(
                            f_signed[index], aligned_signed[index], int(interpretation["top_k"])
                        )
                        full_pearson, full_issue = safe_similarity(
                            f_matrix[index], aligned_matrix[index], "pearson"
                        )
                        pred = prediction_index.loc[sample.sample_id]
                        summit = {
                            "summit_central_mass_fraction_forward": np.nan,
                            "summit_weighted_distance_forward": np.nan,
                            "summit_central_mass_fraction_aligned_rc": np.nan,
                            "summit_weighted_distance_aligned_rc": np.nan,
                        }
                        if int(sample.label) == 1:
                            f_summit = summit_metrics(
                                f_absolute[index],
                                int(bio["central_window_start"]),
                                int(bio["central_window_end"]),
                                int(bio["summit_index_zero_based"]),
                            )
                            r_summit = summit_metrics(
                                aligned_absolute[index],
                                int(bio["central_window_start"]),
                                int(bio["central_window_end"]),
                                int(bio["summit_index_zero_based"]),
                            )
                            summit = {
                                f"{key}_forward": value for key, value in f_summit.items()
                            }
                            summit.update(
                                {f"{key}_aligned_rc": value for key, value in r_summit.items()}
                            )
                        rows.append(
                            {
                                "fold_id": fold_id,
                                "model_type": model_type,
                                "seed": int(seed),
                                "method": method,
                                "sample_id": sample.sample_id,
                                "chromosome": sample.chromosome,
                                "label": int(sample.label),
                                "prediction_difference": float(pred.prediction_difference),
                                "prediction_consistent": bool(
                                    float(pred.prediction_difference)
                                    <= float(
                                        interpretation[
                                            "prediction_consistent_max_abs_difference"
                                        ]
                                    )
                                ),
                                **{f"absolute_{key}": value for key, value in absolute.items()},
                                **{f"signed_{key}": value for key, value in signed.items()},
                                "full_matrix_pearson": full_pearson,
                                "full_matrix_issue": full_issue or "",
                                "attribution_seconds_forward": float(f_seconds),
                                "attribution_seconds_rc": float(r_seconds),
                                **summit,
                            }
                        )
                pd.DataFrame(rows).to_csv(
                    run_dir / "attribution_results.csv", index=False, encoding="utf-8"
                )
                gate_passed = bool(
                    model_type != "CNN-RCPS"
                    or exact_matrix_max_error
                    <= float(config["invariance_gates"]["exact_ism_aligned_max_abs_tolerance"])
                )
                write_json(
                    {
                        "fold_id": fold_id,
                        "model_type": model_type,
                        "seed": int(seed),
                        "methods": interpretation["methods"],
                        "samples": int(len(selected)),
                        "exact_ism_aligned_matrix_max_abs_error": exact_matrix_max_error,
                        "rcps_exact_ism_gate_passed": gate_passed,
                    },
                    complete_path,
                )
                completed_now += 1
                aggregate_attribution_outputs(output_dir)
    return completed_now


def paired_contrasts(
    frame: pd.DataFrame,
    endpoint: str,
    key_columns: list[str],
    config: dict,
    family: str,
) -> tuple[pd.DataFrame, list[dict]]:
    pair_rows, summaries = [], []
    for later, earlier, name in CONTRASTS:
        left = frame[frame.model_type.eq(later)][key_columns + [endpoint]]
        right = frame[frame.model_type.eq(earlier)][key_columns + [endpoint]]
        paired = left.merge(
            right,
            on=key_columns,
            suffixes=("_later", "_earlier"),
            validate="one_to_one",
        )
        paired["contrast"] = name
        paired["endpoint"] = endpoint
        paired["difference"] = paired[f"{endpoint}_later"] - paired[f"{endpoint}_earlier"]
        pair_rows.append(paired)
        summaries.append(
            {
                "family": family,
                "contrast": name,
                "endpoint": endpoint,
                **hierarchical_paired_bootstrap(
                    paired[["fold_id", "seed", "sample_id", "difference"]],
                    int(config["statistics"]["hierarchical_bootstrap_replicates"]),
                    int(config["statistics"]["hierarchical_bootstrap_seed"])
                    + len(summaries),
                ),
            }
        )
    return pd.concat(pair_rows, ignore_index=True), summaries


def create_figures(run_summary: pd.DataFrame, attr_summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric, title, ylabel, filename in (
        (
            "prediction_mean_absolute_difference",
            "P2D prediction asymmetry across fold–seed runs",
            "Mean |p(S)-p(RC(S))|",
            "prediction_asymmetry.png",
        ),
        (
            "holdout_auroc",
            "P2D classification performance across fold–seed runs",
            "Holdout AUROC",
            "holdout_auroc.png",
        ),
    ):
        fig, ax = plt.subplots(figsize=(7.6, 5.0), constrained_layout=True)
        for index, model_type in enumerate(MODEL_ORDER):
            values = run_summary.loc[run_summary.model_type.eq(model_type), metric].to_numpy()
            ax.scatter(np.full(len(values), index), values, s=38, color=MODEL_COLORS[model_type])
            ax.hlines(values.mean(), index - 0.24, index + 0.24, color="black", linewidth=2)
        ax.set_xticks(range(len(MODEL_ORDER)), MODEL_ORDER)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        fig.savefig(output_dir / filename, dpi=180)
        fig.savefig(output_dir / filename.replace(".png", ".pdf"))
        plt.close(fig)
    exact = attr_summary[attr_summary.method.eq("exact_ism")]
    fig, ax = plt.subplots(figsize=(7.6, 5.0), constrained_layout=True)
    for index, model_type in enumerate(MODEL_ORDER):
        values = exact.loc[
            exact.model_type.eq(model_type), "mean_absolute_normalized_l1"
        ].to_numpy()
        ax.scatter(np.full(len(values), index), values, s=38, color=MODEL_COLORS[model_type])
        ax.hlines(values.mean(), index - 0.24, index + 0.24, color="black", linewidth=2)
    ax.set_xticks(range(len(MODEL_ORDER)), MODEL_ORDER)
    ax.set_ylabel("Mean RC-aligned Exact ISM normalized L1")
    ax.set_title("P2D attribution asymmetry across fold–seed runs")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output_dir / "exact_ism_asymmetry.png", dpi=180)
    fig.savefig(output_dir / "exact_ism_asymmetry.pdf")
    plt.close(fig)


def finalize_stage(config: dict, output_dir: Path) -> None:
    run_summary, predictions = aggregate_training_outputs(output_dir)
    attributions = aggregate_attribution_outputs(output_dir)
    expected_runs = len(MODEL_ORDER) * len(config["training"]["seeds"]) * len(
        config["cross_validation"]["folds"]
    )
    if len(run_summary) != expected_runs:
        raise RuntimeError(f"P2D training incomplete: {len(run_summary)}/{expected_runs}")
    expected_attr = (
        expected_runs
        * len(config["interpretation"]["methods"])
        * 2
        * int(config["interpretation"]["samples_per_label_per_fold"])
    )
    if len(attributions) != expected_attr:
        raise RuntimeError(f"P2D attribution incomplete: {len(attributions)}/{expected_attr}")

    summary_rows = []
    for (fold_id, seed, model_type, method), group in attributions.groupby(
        ["fold_id", "seed", "model_type", "method"], observed=True
    ):
        summary_rows.append(
            {
                "fold_id": fold_id,
                "seed": int(seed),
                "model_type": model_type,
                "method": method,
                "n": int(len(group)),
                "mean_absolute_normalized_l1": float(group.absolute_normalized_l1.mean()),
                "median_absolute_normalized_l1": float(group.absolute_normalized_l1.median()),
                "mean_absolute_normalized_l2": float(group.absolute_normalized_l2.mean()),
                "mean_absolute_pearson": float(group.absolute_pearson.mean()),
                "mean_absolute_top8_overlap": float(group.absolute_top8_overlap.mean()),
            }
        )
    attr_summary = pd.DataFrame(summary_rows)
    attr_summary.to_csv(output_dir / "attribution_run_summary.csv", index=False, encoding="utf-8")

    prediction_pairs, prediction_bootstrap = paired_contrasts(
        predictions,
        "prediction_difference",
        ["fold_id", "seed", "sample_id"],
        config,
        "prediction_primary",
    )
    prediction_pairs.to_csv(
        output_dir / "prediction_primary_paired_differences.csv", index=False, encoding="utf-8"
    )
    all_attr_pairs, all_bootstrap = [], []
    for method in config["interpretation"]["methods"]:
        current = attributions[attributions.method.eq(method)]
        pairs, summaries = paired_contrasts(
            current,
            "absolute_normalized_l1",
            ["fold_id", "seed", "sample_id"],
            config,
            "attribution_primary" if method == "exact_ism" else "attribution_secondary",
        )
        pairs["method"] = method
        all_attr_pairs.append(pairs)
        for item in summaries:
            item["method"] = method
        all_bootstrap.extend(summaries)
    pd.concat(all_attr_pairs, ignore_index=True).to_csv(
        output_dir / "attribution_paired_differences.csv", index=False, encoding="utf-8"
    )
    bootstrap = pd.DataFrame(prediction_bootstrap + all_bootstrap)
    bootstrap.to_csv(output_dir / "hierarchical_paired_bootstrap.csv", index=False, encoding="utf-8")

    hypothesis_rows = []
    hypothesis_specs = [
        (
            "H1a",
            "CNN-Raw prediction asymmetry",
            predictions[predictions.model_type.eq("CNN-Raw")],
            "prediction_difference",
        ),
        (
            "H1b",
            "CNN-Raw Exact ISM asymmetry among prediction-consistent samples",
            attributions[
                attributions.model_type.eq("CNN-Raw")
                & attributions.method.eq("exact_ism")
                & attributions.prediction_consistent
            ],
            "absolute_normalized_l1",
        ),
        (
            "H3a",
            "CNN-Aug residual Exact ISM asymmetry",
            attributions[
                attributions.model_type.eq("CNN-Aug")
                & attributions.method.eq("exact_ism")
            ],
            "absolute_normalized_l1",
        ),
        (
            "H3a",
            "CNN-Aug residual Exact ISM asymmetry among prediction-consistent samples",
            attributions[
                attributions.model_type.eq("CNN-Aug")
                & attributions.method.eq("exact_ism")
                & attributions.prediction_consistent
            ],
            "absolute_normalized_l1",
        ),
        (
            "H3b",
            "CNN-RCPS Exact ISM numerical residual",
            attributions[
                attributions.model_type.eq("CNN-RCPS")
                & attributions.method.eq("exact_ism")
            ],
            "absolute_normalized_l1",
        ),
    ]
    for index, (hypothesis, analysis, source, endpoint) in enumerate(hypothesis_specs):
        current = source[["fold_id", "seed", "sample_id", endpoint]].rename(
            columns={endpoint: "difference"}
        )
        hypothesis_rows.append(
            {
                "hypothesis": hypothesis,
                "analysis": analysis,
                "endpoint": endpoint,
                **hierarchical_paired_bootstrap(
                    current,
                    int(config["statistics"]["hierarchical_bootstrap_replicates"]),
                    int(config["statistics"]["hierarchical_bootstrap_seed"]) + 100 + index,
                ),
            }
        )
    pd.DataFrame(hypothesis_rows).to_csv(
        output_dir / "hypothesis_endpoint_bootstrap.csv", index=False, encoding="utf-8"
    )

    concordance_rows = []
    for (fold_id, seed, model_type), group in attributions.groupby(
        ["fold_id", "seed", "model_type"], observed=True
    ):
        pivot = group.pivot(index="sample_id", columns="method", values="absolute_normalized_l1")
        for method in ("integrated_gradients", "deeplift", "gradient_shap"):
            pearson, issue1 = safe_similarity(pivot.exact_ism, pivot[method], "pearson")
            spearman, issue2 = safe_similarity(pivot.exact_ism, pivot[method], "spearman")
            concordance_rows.append(
                {
                    "fold_id": fold_id,
                    "seed": int(seed),
                    "model_type": model_type,
                    "comparison": f"exact_ism_vs_{method}",
                    "pearson": pearson,
                    "spearman": spearman,
                    "issues": ";".join(x for x in (issue1, issue2) if x),
                }
            )
    pd.DataFrame(concordance_rows).to_csv(
        output_dir / "attribution_method_concordance.csv", index=False, encoding="utf-8"
    )

    attribution_audits = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_dir / "runs").glob("*/attribution_complete.json"))
    ]
    audit = pd.DataFrame(attribution_audits)
    audit.to_csv(output_dir / "exact_ism_invariance_audit.csv", index=False, encoding="utf-8")
    gates = {
        "phase": "Phase 2D confirmatory RC symmetry",
        "expected_training_runs": expected_runs,
        "observed_training_runs": int(len(run_summary)),
        "expected_attribution_rows": expected_attr,
        "observed_attribution_rows": int(len(attributions)),
        "checkpoint_reload_gate_passed": bool(run_summary.reload_gate_passed.all()),
        "rcps_prediction_gate_passed": bool(run_summary.rcps_prediction_gate_passed.all()),
        "rcps_feature_gate_passed": bool(run_summary.rcps_feature_gate_passed.all()),
        "rcps_exact_ism_gate_passed": bool(audit.rcps_exact_ism_gate_passed.all()),
        "double_rc_gate_passed": bool(run_summary.double_rc_input_exact.all()),
        "original_validation_rows_evaluated": int(
            run_summary.original_validation_rows_evaluated.sum()
        ),
        "test_rows_evaluated": int(run_summary.test_rows_evaluated.sum()),
        "test_seal_intact": bool(run_summary.test_rows_evaluated.sum() == 0),
    }
    gates["all_scientific_interpretation_gates_passed"] = bool(
        gates["checkpoint_reload_gate_passed"]
        and gates["rcps_prediction_gate_passed"]
        and gates["rcps_feature_gate_passed"]
        and gates["rcps_exact_ism_gate_passed"]
        and gates["double_rc_gate_passed"]
        and gates["original_validation_rows_evaluated"] == 0
        and gates["test_rows_evaluated"] == 0
    )
    write_json(gates, output_dir / "phase2d_gate_assessment.json")
    create_figures(run_summary, attr_summary, output_dir / "figures")
    write_json(
        {
            "status": "completed",
            "protocol_revision": config["protocol_revision"],
            "models": MODEL_ORDER,
            "seeds": config["training"]["seeds"],
            "folds": [fold["fold_id"] for fold in config["cross_validation"]["folds"]],
            "test_seal_intact": True,
            "all_scientific_interpretation_gates_passed": gates[
                "all_scientific_interpretation_gates_passed"
            ],
        },
        output_dir / "phase2d_completion.json",
    )


def prepare(config_path: Path, output_dir: Path) -> tuple[dict, dict[str, pd.DataFrame], pd.DataFrame]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_frozen_config(config)
    errors = validate_manifest(ROOT / "configs" / "phase2_dataset_manifest.yaml")
    if errors:
        raise ValueError("Frozen Phase 2 manifest is invalid: " + "; ".join(errors))
    torch.set_num_threads(max(1, int(config.get("cpu_threads", 1))))
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = output_dir / "resolved_config.yaml"
    if resolved.exists() and resolved.read_bytes() != config_path.read_bytes():
        raise ValueError("Existing P2D output uses a different resolved config")
    if not resolved.exists():
        shutil.copy2(config_path, resolved)
        shutil.copy2(ROOT / config["protocol_path"], output_dir / "frozen_protocol.md")
    modeling, contract = load_modeling_view(config)
    views, fold_manifest = build_cv_views(modeling, config)
    contract.update(
        {
            "phase": "Phase 2D",
            "protocol_revision": config["protocol_revision"],
            "source_split": "train",
            "original_validation_rows_accessed_by_model": 0,
            "test_rows_accessed_by_model": 0,
            "test_predictions_generated": False,
            "test_metrics_generated": False,
        }
    )
    write_json(contract, output_dir / "sealed_test_policy.json")
    write_json(fold_manifest, output_dir / "cross_validation_manifest.json")
    frozen = freeze_attribution_samples(views, config, output_dir)
    write_json(
        {
            "status": "prepared",
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "device": str(DEVICE),
            "cpu_threads": int(torch.get_num_threads()),
            "training_runs_expected": len(MODEL_ORDER)
            * len(config["training"]["seeds"])
            * len(views),
            "frozen_attribution_samples": int(len(frozen)),
            "original_validation_rows_accessed_by_model": 0,
            "test_rows_accessed_by_model": 0,
        },
        output_dir / "phase2d_preparation.json",
    )
    return config, views, frozen


def run(
    config_path: Path,
    output_dir: Path | None = None,
    stage: str = "full",
    max_runs: int | None = None,
) -> Path:
    started = time.perf_counter()
    output_dir = output_dir or ROOT / "results" / "phase2d_rcps_confirmatory"
    config, views, frozen = prepare(config_path, output_dir)
    if stage in {"train", "full"}:
        train_stage(views, config, output_dir, max_runs)
    if stage in {"attribution", "full"}:
        attribution_stage(frozen, config, output_dir, max_runs)
    if stage in {"finalize", "full"}:
        finalize_stage(config, output_dir)
    print(
        f"P2D stage={stage} elapsed={time.perf_counter() - started:.2f}s output={output_dir}",
        flush=True,
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Phase 2D RC confirmatory study")
    parser.add_argument("--config", default="configs/phase2d_rcps_confirmatory.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--stage", choices=["prepare", "train", "attribution", "finalize", "full"], default="full"
    )
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()
    run(
        ROOT / args.config,
        ROOT / args.output_dir if args.output_dir else None,
        stage=args.stage,
        max_runs=args.max_runs,
    )


if __name__ == "__main__":
    main()
