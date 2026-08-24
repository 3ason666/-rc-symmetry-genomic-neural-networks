from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_phase2a import load_modeling_view, write_json
from src.training import train_model_validation_only


def build_inner_calibration_view(modeling: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    source = modeling[modeling.split.eq("train")].copy()
    if len(source) == 0:
        raise ValueError("Frozen Phase 2 train split is empty")
    holdout = set(map(str, config["calibration_split"]["holdout_chromosomes"]))
    observed = set(source.chromosome.astype(str).unique())
    if not holdout or not holdout.issubset(observed):
        raise ValueError(f"Unknown inner holdout chromosome: {sorted(holdout - observed)}")
    source["original_split"] = source["split"]
    source["split"] = source.chromosome.astype(str).map(
        lambda chromosome: "validation" if chromosome in holdout else "train"
    )
    expected_train = int(config["calibration_split"]["expected_inner_train_rows"])
    expected_validation = int(config["calibration_split"]["expected_inner_validation_rows"])
    counts = source.split.value_counts().to_dict()
    if counts != {"train": expected_train, "validation": expected_validation}:
        raise ValueError(f"Unexpected inner split counts: {counts}")
    if source.groupby("pair_id").split.nunique().max() != 1:
        raise ValueError("A matched pair crosses the inner calibration boundary")
    if source.groupby("canonical_key").split.nunique().max() != 1:
        raise ValueError("A canonical RC group crosses the inner calibration boundary")
    if set(source.original_split.unique()) != {"train"}:
        raise ValueError("Frozen validation/test rows entered P2C.1 calibration")
    manifest = {
        "source_split": "train",
        "inner_train_rows": expected_train,
        "inner_validation_rows": expected_validation,
        "inner_train_chromosomes": sorted(source.loc[source.split.eq("train"), "chromosome"].unique()),
        "inner_validation_chromosomes": sorted(holdout),
        "pair_leaks": 0,
        "canonical_rc_leaks": 0,
        "frozen_validation_rows_accessed_by_model": 0,
        "sealed_test_rows_accessed_by_model": 0,
    }
    return source.drop(columns="original_split"), manifest


def candidate_config(base: dict, candidate: dict) -> dict:
    merged = {
        "model": {
            **base["model"],
            "d_model": int(candidate["d_model"]),
            "n_heads": int(candidate["n_heads"]),
            "num_layers": int(candidate["num_layers"]),
            "feedforward_dim": int(candidate["feedforward_dim"]),
            "dropout": float(candidate["dropout"]),
        },
        "training": {
            **base["training"],
            "seeds": [int(base["training"]["calibration_seed"])],
            "learning_rate": float(candidate["learning_rate"]),
        },
    }
    return merged


def summarize_candidates(rows: pd.DataFrame, floor: float) -> pd.DataFrame:
    summaries = []
    for candidate_id, group in rows.groupby("candidate_id", sort=False):
        summaries.append(
            {
                "candidate_id": candidate_id,
                "variants_at_or_above_floor": int((group.inner_validation_auroc >= floor).sum()),
                "minimum_variant_auroc": float(group.inner_validation_auroc.min()),
                "mean_variant_auroc": float(group.inner_validation_auroc.mean()),
                "mean_variant_loss": float(group.inner_validation_loss.mean()),
                "summed_training_seconds": float(group.training_seconds.sum()),
            }
        )
    summary = pd.DataFrame(summaries)
    return summary.sort_values(
        [
            "variants_at_or_above_floor",
            "minimum_variant_auroc",
            "mean_variant_auroc",
            "mean_variant_loss",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def run(config_path: Path, output_dir: Path | None = None) -> Path:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("device") != "cpu":
        raise ValueError("P2C.1 calibration is frozen to CPU")
    if config["dataset"].get("test_policy") != "sealed_no_model_access":
        raise ValueError("P2C.1 requires a sealed test set")
    if config["decision"].get("access_frozen_validation_during_calibration") is not False:
        raise ValueError("P2C.1 must not tune on the frozen validation split")
    output_dir = output_dir or ROOT / "results" / config["project_name"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite calibration directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "resolved_config.yaml")
    torch.set_num_threads(max(1, int(config.get("cpu_threads", 1))))

    modeling, contract = load_modeling_view(config)
    inner, split_manifest = build_inner_calibration_view(modeling, config)
    contract.update(
        {
            "calibration_scope": "frozen_phase2_train_split_only",
            "frozen_validation_rows_excluded_before_model_training": int(
                modeling.split.eq("validation").sum()
            ),
            "test_predictions_generated": False,
            "test_metrics_generated": False,
        }
    )
    write_json(contract, output_dir / "sealed_data_policy.json")
    write_json(split_manifest, output_dir / "inner_split_manifest.json")
    inner[["sample_id", "pair_id", "canonical_key", "label", "split", "chromosome"]].to_csv(
        output_dir / "inner_calibration_split.csv", index=False, encoding="utf-8"
    )

    seed = int(config["training"]["calibration_seed"])
    model_types = list(config["training"]["model_types"])
    rows = []
    for candidate in config["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        current = candidate_config(config, candidate)
        for model_type in model_types:
            print(f"P2C.1 {candidate_id} | {model_type} | seed={seed}", flush=True)
            run_dir = output_dir / "runs" / candidate_id / model_type.lower().replace("-", "_")
            _, metadata = train_model_validation_only(model_type, seed, current, inner, run_dir)
            metrics = metadata["validation_metrics"]
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "model_type": model_type,
                    "calibration_seed": seed,
                    "d_model": int(candidate["d_model"]),
                    "n_heads": int(candidate["n_heads"]),
                    "num_layers": int(candidate["num_layers"]),
                    "feedforward_dim": int(candidate["feedforward_dim"]),
                    "dropout": float(candidate["dropout"]),
                    "learning_rate": float(candidate["learning_rate"]),
                    "best_epoch": int(metadata["best_epoch"]),
                    "parameter_count": int(metadata["parameter_count"]),
                    "training_seconds": float(metadata["training_seconds"]),
                    **{f"inner_validation_{key}": value for key, value in metrics.items()},
                }
            )
    run_summary = pd.DataFrame(rows)
    run_summary.to_csv(output_dir / "calibration_run_summary.csv", index=False, encoding="utf-8")
    floor = float(config["selection"]["minimum_inner_validation_auroc"])
    ranking = summarize_candidates(run_summary, floor)
    ranking.to_csv(output_dir / "candidate_ranking.csv", index=False, encoding="utf-8")
    selected_id = str(ranking.iloc[0].candidate_id)
    selected_candidate = next(
        candidate for candidate in config["candidates"] if candidate["candidate_id"] == selected_id
    )
    selected_rows = run_summary[run_summary.candidate_id.eq(selected_id)]
    decision = {
        "status": "calibration_complete",
        "selected_candidate_id": selected_id,
        "selected_candidate": selected_candidate,
        "selection_rules_applied_in_frozen_order": config["selection"]["ordered_rules"],
        "variants_at_or_above_floor": int(ranking.iloc[0].variants_at_or_above_floor),
        "all_variants_at_or_above_floor": bool(
            ranking.iloc[0].variants_at_or_above_floor == len(model_types)
        ),
        "minimum_variant_auroc": float(ranking.iloc[0].minimum_variant_auroc),
        "mean_variant_auroc": float(ranking.iloc[0].mean_variant_auroc),
        "inner_validation_results": selected_rows[
            ["model_type", "inner_validation_loss", "inner_validation_auroc", "inner_validation_auprc"]
        ].to_dict(orient="records"),
        "next_action": "freeze_selected_candidate_then_run_confirmation_seeds_on_original_validation",
        "frozen_validation_used_for_tuning": False,
        "test_seal_intact": True,
        "unseal_test_now": False,
    }
    write_json(decision, output_dir / "calibration_decision.json")
    completion = {
        "status": "completed_train_only_calibration",
        "elapsed_seconds": time.perf_counter() - started,
        "candidates": len(config["candidates"]),
        "models_per_candidate": len(model_types),
        "training_runs": int(len(run_summary)),
        "selected_candidate_id": selected_id,
        "frozen_validation_predictions_generated": False,
        "test_predictions_generated": False,
        "test_metrics_generated": False,
    }
    write_json(completion, output_dir / "calibration_completion.json")
    print(ranking.to_string(index=False), flush=True)
    print(json.dumps(decision, indent=2), flush=True)
    print(f"Completed P2C.1 calibration in {completion['elapsed_seconds']:.2f}s", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run train-only Phase 2C.1 CPU calibration")
    parser.add_argument("--config", default="configs/phase2c1_cpu_calibration.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run(ROOT / args.config, ROOT / args.output_dir if args.output_dir else None)


if __name__ == "__main__":
    main()
