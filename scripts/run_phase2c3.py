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
from torch import nn
from torch.utils.data import DataLoader

from scripts.run_phase2a import load_modeling_view, write_json
from scripts.validate_phase2_manifest import validate_manifest
from src.data import SequenceDataset
from src.dna_utils import reverse_complement
from src.metrics import prediction_consistency_metrics
from src.models import build_model, parameter_count
from src.training import DEVICE, evaluate, predict_sequences, set_seed, train_model_validation_only


REMEDY_MODEL_TYPE = "Transformer-Aug-Absolute"
REMEDY_ARCHITECTURE = "transformer_absolute"


def build_cv_views(modeling: pd.DataFrame, config: dict) -> tuple[dict[str, pd.DataFrame], dict]:
    source = modeling[modeling.split.eq("train")].copy()
    expected_source_rows = int(config["dataset"]["expected_source_rows"])
    if len(source) != expected_source_rows:
        raise ValueError(
            f"Unexpected Phase 2 train size: {len(source)} != {expected_source_rows}"
        )
    observed = set(source.chromosome.astype(str).unique())
    folds = config["cross_validation"]["folds"]
    holdout_lists = [list(map(str, fold["holdout_chromosomes"])) for fold in folds]
    flattened = [chromosome for values in holdout_lists for chromosome in values]
    if len(flattened) != len(set(flattened)):
        raise ValueError("A chromosome appears in more than one P2C.3 holdout fold")
    if set(flattened) != observed:
        raise ValueError(
            f"P2C.3 folds do not cover train chromosomes exactly: "
            f"missing={sorted(observed - set(flattened))}, extra={sorted(set(flattened) - observed)}"
        )

    views: dict[str, pd.DataFrame] = {}
    manifest_rows = []
    for fold in folds:
        fold_id = str(fold["fold_id"])
        holdout = set(map(str, fold["holdout_chromosomes"]))
        current = source.copy()
        current["original_split"] = current["split"]
        current["split"] = current.chromosome.astype(str).map(
            lambda chromosome: "validation" if chromosome in holdout else "train"
        )
        expected = int(fold["expected_holdout_rows"])
        observed_holdout = int(current.split.eq("validation").sum())
        if observed_holdout != expected:
            raise ValueError(f"{fold_id} holdout rows {observed_holdout} != expected {expected}")
        if current.groupby("pair_id").split.nunique().max() != 1:
            raise ValueError(f"A matched pair crosses {fold_id}")
        if current.groupby("canonical_key").split.nunique().max() != 1:
            raise ValueError(f"A canonical RC group crosses {fold_id}")
        if set(current.original_split.unique()) != {"train"}:
            raise ValueError("Original validation/test rows entered a P2C.3 model view")
        views[fold_id] = current.drop(columns="original_split").reset_index(drop=True)
        manifest_rows.append(
            {
                "fold_id": fold_id,
                "inner_train_rows": int(current.split.eq("train").sum()),
                "inner_validation_rows": observed_holdout,
                "inner_train_chromosomes": sorted(
                    current.loc[current.split.eq("train"), "chromosome"].unique()
                ),
                "inner_validation_chromosomes": sorted(holdout),
                "pair_leaks": 0,
                "canonical_rc_leaks": 0,
            }
        )
    manifest = {
        "source_split": "train",
        "source_rows": int(len(source)),
        "source_chromosomes": sorted(observed),
        "folds": manifest_rows,
        "original_validation_rows_accessed_by_model": 0,
        "sealed_test_rows_accessed_by_model": 0,
    }
    return views, manifest


def cnn_config(config: dict) -> dict:
    current = copy.deepcopy(config)
    current["model"] = copy.deepcopy(config["cnn_model"])
    current["training"].update(
        {
            "learning_rate": float(config["training"]["cnn"]["learning_rate"]),
            "max_epochs": int(config["training"]["cnn"]["max_epochs"]),
            "patience": int(config["training"]["cnn"]["patience"]),
        }
    )
    return current


def load_checkpoint(path: Path):
    saved = torch.load(path, map_location=DEVICE, weights_only=True)
    model = build_model(saved["model_config"], architecture=saved["architecture"]).to(DEVICE)
    model.load_state_dict(saved["model_state"])
    return model.eval(), saved


def train_absolute_remedy(
    remedy: dict,
    seed: int,
    config: dict,
    data: pd.DataFrame,
    run_dir: Path,
):
    if set(data.split.astype(str).unique()) != {"train", "validation"}:
        raise ValueError("P2C.3 remedy training requires an inner train/validation view")
    set_seed(seed)
    started = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = config["training"]
    train_ds = SequenceDataset(
        data[data.split.eq("train")], "augment", float(train_cfg["augmentation_probability"])
    )
    valid_ds = SequenceDataset(data[data.split.eq("validation")], "raw")
    generator = torch.Generator().manual_seed(seed)
    loader_args = {
        "batch_size": int(train_cfg["batch_size"]),
        "num_workers": int(train_cfg.get("num_workers", 0)),
    }
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_args)
    valid_loader = DataLoader(valid_ds, shuffle=False, **loader_args)
    model = build_model(config["transformer_model"], architecture=REMEDY_ARCHITECTURE).to(DEVICE)
    optimizer_name = str(remedy["optimizer"])
    optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[optimizer_name]
    optimizer = optimizer_class(
        model.parameters(),
        lr=float(remedy["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler_name = str(remedy["scheduler"])
    if scheduler_name == "none":
        scheduler = None
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(remedy["max_epochs"]),
            eta_min=float(remedy["cosine_eta_min"]),
        )
    else:
        raise ValueError(f"Unknown P2C.3 scheduler: {scheduler_name}")
    criterion = nn.BCEWithLogitsLoss()
    best_loss, best_epoch, stale, history = np.inf, 0, 0, []
    checkpoint = run_dir / "best_checkpoint.pt"
    for epoch in range(1, int(remedy["max_epochs"]) + 1):
        model.train()
        train_loss, count = 0.0, 0
        for inputs, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs.to(DEVICE))
            loss = criterion(logits, targets.to(DEVICE))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(targets)
            count += len(targets)
        val_metrics = evaluate(model, valid_loader, float(train_cfg["threshold"]))
        history.append(
            {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": train_loss / count,
                **{f"inner_validation_{key}": value for key, value in val_metrics.items()},
            }
        )
        if val_metrics["loss"] < best_loss - 1e-8:
            best_loss, best_epoch, stale = val_metrics["loss"], epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": config["transformer_model"],
                    "architecture": REMEDY_ARCHITECTURE,
                    "model_type": REMEDY_MODEL_TYPE,
                    "remedy_id": remedy["remedy_id"],
                    "seed": seed,
                    "best_epoch": epoch,
                    "selection_split": "inner_validation",
                    "original_validation_evaluated": False,
                    "test_evaluated": False,
                },
                checkpoint,
            )
        else:
            stale += 1
        if scheduler is not None:
            scheduler.step()
        if stale >= int(remedy["patience"]):
            break

    reloaded, saved = load_checkpoint(checkpoint)
    inner_metrics = evaluate(reloaded, valid_loader, float(train_cfg["threshold"]))
    elapsed = time.perf_counter() - started
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False, encoding="utf-8")
    metadata = {
        "model_type": REMEDY_MODEL_TYPE,
        "remedy_id": remedy["remedy_id"],
        "architecture": REMEDY_ARCHITECTURE,
        "seed": seed,
        "best_epoch": int(best_epoch),
        "parameter_count": int(parameter_count(reloaded)),
        "training_seconds": float(elapsed),
        "optimizer": optimizer_name,
        "scheduler": scheduler_name,
        "learning_rate": float(remedy["learning_rate"]),
        "max_epochs": int(remedy["max_epochs"]),
        "patience": int(remedy["patience"]),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "device": str(DEVICE),
        "selection_split": "inner_validation",
        "original_validation_evaluated": False,
        "test_evaluated": False,
        "inner_validation_metrics": inner_metrics,
    }
    write_json(metadata, run_dir / "run_metadata.json")
    return reloaded, saved, metadata


def cluster_bootstrap(values: np.ndarray, replicates: int, seed: int) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("Bootstrap requires at least one paired fold-seed effect")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for index in range(replicates):
        estimates[index] = rng.choice(values, size=values.size, replace=True).mean()
    return {
        "n_fold_seed_pairs": int(values.size),
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
    }


def rank_remedies(rows: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for remedy_id, group in rows.groupby("remedy_id", sort=False):
        summaries.append(
            {
                "remedy_id": remedy_id,
                "fold_seed_pairs": int(len(group)),
                "fold_seed_matches": int(group.joint_gate_passed.sum()),
                "worst_absolute_auroc_gap": float(group.absolute_auroc_gap.max()),
                "minimum_transformer_auroc": float(group.transformer_auroc.min()),
                "mean_transformer_auroc": float(group.transformer_auroc.mean()),
                "mean_absolute_minus_cnn_auroc": float(group.absolute_minus_cnn_auroc.mean()),
                "mean_prediction_asymmetry": float(group.prediction_mean_absolute_difference.mean()),
                "summed_training_seconds": float(group.training_seconds.sum()),
            }
        )
    summary = pd.DataFrame(summaries)
    return summary.sort_values(
        [
            "fold_seed_matches",
            "worst_absolute_auroc_gap",
            "minimum_transformer_auroc",
            "mean_transformer_auroc",
            "mean_prediction_asymmetry",
            "summed_training_seconds",
        ],
        ascending=[False, True, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def create_figures(rows: pd.DataFrame, ranking: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    remedies = ranking.remedy_id.tolist()
    colors = ["#2563EB", "#7C3AED", "#059669"]
    x = np.arange(len(remedies))

    fig, ax = plt.subplots(figsize=(9.4, 5.2), constrained_layout=True)
    for index, remedy in enumerate(remedies):
        values = rows.loc[rows.remedy_id.eq(remedy), "absolute_minus_cnn_auroc"].to_numpy()
        jitter = np.linspace(-0.08, 0.08, values.size)
        ax.scatter(np.full(values.size, index) + jitter, values, color=colors[index], s=36)
        ax.hlines(values.mean(), index - 0.25, index + 0.25, color="black", linewidth=2)
    ax.axhline(0.03, color="#991B1B", linestyle="--", linewidth=1)
    ax.axhline(-0.03, color="#991B1B", linestyle="--", linewidth=1, label="Frozen ±0.03 gap")
    ax.axhline(0.0, color="#4B5563", linewidth=0.8)
    ax.set_xticks(x, remedies, rotation=18, ha="right")
    ax.set_ylabel("Transformer Absolute AUROC - CNN AUROC")
    ax.set_title("P2C.3 train-only paired performance gaps")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "paired_auroc_gap_by_remedy.png", dpi=180)
    fig.savefig(output_dir / "paired_auroc_gap_by_remedy.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 5.2), constrained_layout=True)
    for index, remedy in enumerate(remedies):
        values = rows.loc[rows.remedy_id.eq(remedy), "transformer_auroc"].to_numpy()
        jitter = np.linspace(-0.08, 0.08, values.size)
        ax.scatter(np.full(values.size, index) + jitter, values, color=colors[index], s=36)
        ax.hlines(values.mean(), index - 0.25, index + 0.25, color="black", linewidth=2)
    ax.axhline(0.80, color="#7A5A00", linestyle="--", linewidth=1, label="Frozen AUROC floor")
    ax.set_xticks(x, remedies, rotation=18, ha="right")
    ax.set_ylabel("Inner-validation AUROC")
    ax.set_title("P2C.3 train-only Transformer performance")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "transformer_auroc_by_remedy.png", dpi=180)
    fig.savefig(output_dir / "transformer_auroc_by_remedy.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 5.2), constrained_layout=True)
    for index, remedy in enumerate(remedies):
        values = rows.loc[
            rows.remedy_id.eq(remedy), "prediction_mean_absolute_difference"
        ].to_numpy()
        jitter = np.linspace(-0.08, 0.08, values.size)
        ax.scatter(np.full(values.size, index) + jitter, values, color=colors[index], s=36)
        ax.hlines(values.mean(), index - 0.25, index + 0.25, color="black", linewidth=2)
    ax.axhline(0.01, color="#991B1B", linestyle="--", linewidth=1, label="Consistency threshold")
    ax.set_xticks(x, remedies, rotation=18, ha="right")
    ax.set_ylabel("Mean |p(S)-p(RC(S))|")
    ax.set_title("P2C.3 train-only prediction asymmetry")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "prediction_asymmetry_by_remedy.png", dpi=180)
    fig.savefig(output_dir / "prediction_asymmetry_by_remedy.pdf")
    plt.close(fig)


def run(config_path: Path, output_dir: Path | None = None) -> Path:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("device") != "cpu":
        raise ValueError("P2C.3 is frozen to the current PC CPU")
    if config["dataset"].get("source_split") != "train":
        raise ValueError("P2C.3 may only use the frozen train split")
    if config["dataset"].get("forbidden_model_splits") != ["validation", "test"]:
        raise ValueError("P2C.3 forbidden split declaration changed")
    if config["decision"].get("access_original_validation") is not False:
        raise ValueError("P2C.3 must not access original validation for modeling")
    if config["decision"].get("unseal_test_now") is not False:
        raise ValueError("P2C.3 must keep test sealed")
    if config["training"].get("seeds") != [42, 123, 2026]:
        raise ValueError("P2C.3 frozen seeds changed")
    remedy_ids = [str(remedy["remedy_id"]) for remedy in config["remedies"]]
    if remedy_ids != ["r0_c4_baseline", "r1_c4_high_lr_long", "r2_c4_adamw_cosine"]:
        raise ValueError("P2C.3 frozen remedy order changed")
    manifest_errors = validate_manifest(ROOT / "configs" / "phase2_dataset_manifest.yaml")
    if manifest_errors:
        raise ValueError("Frozen Phase 2 manifest is invalid: " + "; ".join(manifest_errors))

    torch.set_num_threads(max(1, int(config.get("cpu_threads", 1))))
    output_dir = output_dir or ROOT / "results" / config["project_name"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite P2C.3 directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "resolved_config.yaml")

    modeling, contract = load_modeling_view(config)
    views, fold_manifest = build_cv_views(modeling, config)
    contract.update(
        {
            "protocol_revision": config["protocol_revision"],
            "scope": config["scope"],
            "model_source_split": "train",
            "original_validation_rows_excluded_before_model_training": int(
                modeling.split.eq("validation").sum()
            ),
            "original_validation_predictions_generated": False,
            "original_validation_metrics_generated": False,
            "test_predictions_generated": False,
            "test_metrics_generated": False,
        }
    )
    write_json(contract, output_dir / "sealed_data_policy.json")
    write_json(fold_manifest, output_dir / "cross_validation_manifest.json")
    split_rows = []
    for fold_id, view in views.items():
        current = view[["sample_id", "pair_id", "canonical_key", "label", "chromosome", "split"]].copy()
        current.insert(0, "fold_id", fold_id)
        split_rows.append(current)
    pd.concat(split_rows, ignore_index=True).to_csv(
        output_dir / "inner_cv_assignments.csv", index=False, encoding="utf-8"
    )

    print(
        f"Python={platform.python_version()} | PyTorch={torch.__version__} | device={DEVICE} | "
        f"threads={torch.get_num_threads()} | source=train-only | original_validation=excluded | "
        f"test=sealed",
        flush=True,
    )

    comparator_rows = []
    remedy_rows = []
    prediction_rows = []
    batch_size = int(config["training"]["batch_size"])
    threshold = float(config["training"]["threshold"])
    gap_limit = float(config["selection"]["performance_match_max_abs_auroc_difference"])
    floor = float(config["selection"]["transformer_minimum_auroc"])

    for fold_id, inner in views.items():
        holdout = inner[inner.split.eq("validation")].reset_index(drop=True)
        sequences = holdout.sequence.tolist()
        rc_sequences = [reverse_complement(sequence) for sequence in sequences]
        for seed in config["training"]["seeds"]:
            seed = int(seed)
            print(f"P2C.3 {fold_id} | CNN-Aug | seed={seed}", flush=True)
            cnn_dir = output_dir / "runs" / fold_id / f"seed_{seed}" / "cnn_aug"
            cnn, cnn_meta = train_model_validation_only(
                "CNN-Aug", seed, cnn_config(config), inner, cnn_dir
            )
            cnn_pf = predict_sequences(cnn, sequences, batch_size)
            cnn_pr = predict_sequences(cnn, rc_sequences, batch_size)
            cnn_consistency, cnn_each = prediction_consistency_metrics(cnn_pf, cnn_pr, threshold)
            cnn_reload, _ = load_checkpoint(cnn_dir / "best_checkpoint.pt")
            cnn_reload_error = float(
                np.max(np.abs(cnn_pf - predict_sequences(cnn_reload, sequences, batch_size)))
            )
            cnn_metrics = cnn_meta["validation_metrics"]
            comparator_rows.append(
                {
                    "fold_id": fold_id,
                    "seed": seed,
                    "best_epoch": int(cnn_meta["best_epoch"]),
                    "parameter_count": int(cnn_meta["parameter_count"]),
                    "training_seconds": float(cnn_meta["training_seconds"]),
                    "cnn_auroc": float(cnn_metrics["auroc"]),
                    "cnn_auprc": float(cnn_metrics["auprc"]),
                    "prediction_mean_absolute_difference": float(
                        cnn_consistency["prediction_mean_absolute_difference"]
                    ),
                    "checkpoint_reload_max_abs_probability_error": cnn_reload_error,
                }
            )
            for row_index, row in holdout.iterrows():
                prediction_rows.append(
                    {
                        "fold_id": fold_id,
                        "seed": seed,
                        "model_id": "CNN-Aug",
                        "sample_id": row.sample_id,
                        "label": int(row.label),
                        "p_s": float(cnn_pf[row_index]),
                        "p_rc": float(cnn_pr[row_index]),
                        "absolute_difference": float(
                            cnn_each["prediction_difference"][row_index]
                        ),
                    }
                )

            for remedy in config["remedies"]:
                remedy_id = str(remedy["remedy_id"])
                print(f"P2C.3 {fold_id} | {remedy_id} | seed={seed}", flush=True)
                run_dir = output_dir / "runs" / fold_id / f"seed_{seed}" / remedy_id
                model, _, metadata = train_absolute_remedy(remedy, seed, config, inner, run_dir)
                pf = predict_sequences(model, sequences, batch_size)
                pr = predict_sequences(model, rc_sequences, batch_size)
                consistency, each = prediction_consistency_metrics(pf, pr, threshold)
                reloaded, saved = load_checkpoint(run_dir / "best_checkpoint.pt")
                reload_error = float(
                    np.max(np.abs(pf - predict_sequences(reloaded, sequences, batch_size)))
                )
                if saved["architecture"] != REMEDY_ARCHITECTURE:
                    raise ValueError("P2C.3 checkpoint architecture changed")
                metrics = metadata["inner_validation_metrics"]
                delta = float(metrics["auroc"] - cnn_metrics["auroc"])
                absolute_gap = abs(delta)
                match = absolute_gap <= gap_limit
                floor_pass = float(metrics["auroc"]) >= floor
                remedy_rows.append(
                    {
                        "fold_id": fold_id,
                        "seed": seed,
                        "remedy_id": remedy_id,
                        "optimizer": metadata["optimizer"],
                        "scheduler": metadata["scheduler"],
                        "learning_rate": metadata["learning_rate"],
                        "max_epochs": metadata["max_epochs"],
                        "patience": metadata["patience"],
                        "best_epoch": metadata["best_epoch"],
                        "parameter_count": metadata["parameter_count"],
                        "training_seconds": metadata["training_seconds"],
                        "cnn_auroc": float(cnn_metrics["auroc"]),
                        "transformer_loss": float(metrics["loss"]),
                        "transformer_accuracy": float(metrics["accuracy"]),
                        "transformer_f1": float(metrics["f1"]),
                        "transformer_auroc": float(metrics["auroc"]),
                        "transformer_auprc": float(metrics["auprc"]),
                        "absolute_minus_cnn_auroc": delta,
                        "absolute_auroc_gap": absolute_gap,
                        "performance_match_passed": bool(match),
                        "transformer_floor_passed": bool(floor_pass),
                        "joint_gate_passed": bool(match and floor_pass),
                        "prediction_mean_absolute_difference": float(
                            consistency["prediction_mean_absolute_difference"]
                        ),
                        "prediction_median_absolute_difference": float(
                            consistency["prediction_median_absolute_difference"]
                        ),
                        "prediction_p95_absolute_difference": float(
                            consistency["prediction_p95_absolute_difference"]
                        ),
                        "prediction_pearson": float(consistency["prediction_pearson"]),
                        "prediction_spearman": float(consistency["prediction_spearman"]),
                        "symmetry_flip_rate": float(consistency["symmetry_flip_rate"]),
                        "checkpoint_reload_max_abs_probability_error": reload_error,
                        "original_validation_evaluated": False,
                        "test_evaluated": False,
                    }
                )
                for row_index, row in holdout.iterrows():
                    prediction_rows.append(
                        {
                            "fold_id": fold_id,
                            "seed": seed,
                            "model_id": remedy_id,
                            "sample_id": row.sample_id,
                            "label": int(row.label),
                            "p_s": float(pf[row_index]),
                            "p_rc": float(pr[row_index]),
                            "absolute_difference": float(
                                each["prediction_difference"][row_index]
                            ),
                        }
                    )

    comparator = pd.DataFrame(comparator_rows)
    results = pd.DataFrame(remedy_rows)
    predictions = pd.DataFrame(prediction_rows)
    comparator.to_csv(output_dir / "cnn_comparator_summary.csv", index=False, encoding="utf-8")
    results.to_csv(output_dir / "remedy_fold_seed_results.csv", index=False, encoding="utf-8")
    predictions.to_csv(output_dir / "inner_validation_prediction_pairs.csv", index=False, encoding="utf-8")

    ranking = rank_remedies(results)
    ranking.to_csv(output_dir / "remedy_ranking.csv", index=False, encoding="utf-8")
    bootstrap = {}
    for index, remedy_id in enumerate(ranking.remedy_id):
        values = results.loc[
            results.remedy_id.eq(remedy_id), "absolute_minus_cnn_auroc"
        ].to_numpy()
        bootstrap[remedy_id] = cluster_bootstrap(
            values,
            int(config["statistics"]["paired_bootstrap_replicates"]),
            int(config["statistics"]["paired_bootstrap_seed"]) + index,
        )
    write_json(bootstrap, output_dir / "paired_gap_bootstrap.json")

    top = ranking.iloc[0]
    required_pairs = len(config["cross_validation"]["folds"]) * len(config["training"]["seeds"])
    strict_pass = bool(
        int(top.fold_seed_matches) == required_pairs
        and bool(config["selection"]["select_only_if_all_nine_fold_seed_pairs_pass"])
    )
    selected_id = str(top.remedy_id) if strict_pass else None
    reload_pass = bool(
        (comparator.checkpoint_reload_max_abs_probability_error <= 1e-6).all()
        and (results.checkpoint_reload_max_abs_probability_error <= 1e-6).all()
    )
    decision = {
        "status": "train_only_remediation_screen_complete",
        "ranked_first_remedy": str(top.remedy_id),
        "strict_remedy_selected": selected_id,
        "required_fold_seed_pairs": int(required_pairs),
        "ranked_first_matches": int(top.fold_seed_matches),
        "strict_train_only_gate_passed": strict_pass,
        "selection_rules_applied_in_frozen_order": config["selection"][
            "ordered_ranking_rules"
        ],
        "paired_gap_bootstrap_for_ranked_first": bootstrap[str(top.remedy_id)],
        "checkpoint_reload_gate_passed": reload_pass,
        "original_validation_used_for_modeling": False,
        "test_seal_intact": True,
        "proceed_to_phase2d": False,
        "independent_confirmation_required": True,
        "next_action": (
            "freeze_selected_remedy_and_obtain_new_untouched_confirmation_resource"
            if strict_pass
            else "no_remedy_met_strict_train_only_gate_revise_model_class_before_new_confirmation"
        ),
    }
    write_json(decision, output_dir / "remediation_decision.json")
    gate = {
        "phase": "Phase 2C.3 train-only remediation selection",
        "source_split": "train",
        "original_validation_excluded": True,
        "test_seal_intact": True,
        "expected_cnn_runs": int(required_pairs),
        "observed_cnn_runs": int(len(comparator)),
        "expected_transformer_runs": int(required_pairs * len(config["remedies"])),
        "observed_transformer_runs": int(len(results)),
        "checkpoint_reload_gate_passed": reload_pass,
        "strict_train_only_gate_passed": strict_pass,
        "independent_confirmation_still_required": True,
        "proceed_to_phase2d": False,
        "unseal_test_now": False,
    }
    write_json(gate, output_dir / "phase2c3_gate_assessment.json")
    create_figures(results, ranking, output_dir / "figures")
    completion = {
        "status": "completed_train_only_remediation",
        "elapsed_seconds": time.perf_counter() - started,
        "folds": len(config["cross_validation"]["folds"]),
        "seeds": config["training"]["seeds"],
        "remedies": remedy_ids,
        "cnn_training_runs": int(len(comparator)),
        "transformer_training_runs": int(len(results)),
        "ranked_first_remedy": str(top.remedy_id),
        "strict_remedy_selected": selected_id,
        "original_validation_predictions_generated": False,
        "original_validation_metrics_generated": False,
        "test_predictions_generated": False,
        "test_metrics_generated": False,
    }
    write_json(completion, output_dir / "phase2c3_completion.json")
    print(ranking.to_string(index=False), flush=True)
    print(json.dumps(decision, indent=2), flush=True)
    print(f"Completed P2C.3 in {completion['elapsed_seconds']:.2f}s: {output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run train-only Phase 2C.3 remediation selection")
    parser.add_argument("--config", default="configs/phase2c3_train_only_remediation.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run(ROOT / args.config, ROOT / args.output_dir if args.output_dir else None)


if __name__ == "__main__":
    main()
