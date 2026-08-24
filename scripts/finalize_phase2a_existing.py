from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from scripts.run_phase2a import create_figures, write_json


def finalize(results_dir: Path) -> None:
    required = [
        "resolved_config.yaml",
        "sealed_test_policy.json",
        "validation_prediction_pairs.csv",
        "validation_attribution_results.csv",
        "validation_attribution_summary.csv",
        "run_summary.csv",
        "ism_validation_sample_ids.csv",
    ]
    missing = [name for name in required if not (results_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot finalize incomplete Phase 2A directory: {missing}")

    config = yaml.safe_load((results_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    contract = json.loads((results_dir / "sealed_test_policy.json").read_text(encoding="utf-8"))
    if contract.get("test_predictions_generated") is not False or contract.get("test_metrics_generated") is not False:
        raise ValueError("Test seal is not intact")
    predictions = pd.read_csv(results_dir / "validation_prediction_pairs.csv")
    attributions = pd.read_csv(results_dir / "validation_attribution_results.csv")
    run_summary = pd.read_csv(results_dir / "run_summary.csv")
    selected = pd.read_csv(results_dir / "ism_validation_sample_ids.csv")
    if "test_evaluated" not in run_summary or run_summary["test_evaluated"].astype(str).str.lower().ne("false").any():
        raise ValueError("Run summary does not prove that test evaluation stayed disabled")

    create_figures(predictions, attributions, results_dir / "figures")
    completion = {
        "phase": "Phase 2A CNN pilot",
        "status": "completed_validation_only",
        "recovered_after_plotting_only_failure": True,
        "models": config["training"]["model_types"],
        "seeds": config["training"]["seeds"],
        "train_rows": next(
            int(row["n"])
            for row in contract["modeling_counts"]
            if row["split"] == "train" and int(row["label"]) == 0
        ) * 2,
        "validation_rows": next(
            int(row["n"])
            for row in contract["modeling_counts"]
            if row["split"] == "validation" and int(row["label"]) == 0
        ) * 2,
        "test_policy": "sealed_no_model_access",
        "test_predictions_generated": False,
        "test_metrics_generated": False,
        "ism_validation_samples": int(len(selected)),
        "thresholds_frozen_before_results": {
            "prediction_consistent_max_abs_difference": float(
                config["interpretation"]["prediction_consistent_max_abs_difference"]
            ),
            "attribution_pearson_warning_below": float(
                config["interpretation"]["attribution_pearson_warning_below"]
            ),
            "attribution_top8_warning_below": float(
                config["interpretation"]["attribution_top8_warning_below"]
            ),
        },
        "summed_training_seconds": float(run_summary["training_seconds"].sum()),
        "summed_ism_seconds": float(run_summary["ism_seconds"].sum()),
    }
    write_json(completion, results_dir / "pilot_completion.json")
    print(f"Finalized existing Phase 2A results: {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Finalize a completed Phase 2A run after a plotting-only failure")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "phase2a_cnn_pilot",
    )
    args = parser.parse_args()
    finalize(args.results_dir)


if __name__ == "__main__":
    main()
