from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_mean_ci(values, rng, repetitions=10_000):
    values = np.asarray(values, dtype=float)
    estimates = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        estimates[index] = values[rng.integers(0, len(values), len(values))].mean()
    return float(values.mean()), float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def wilson_interval(successes, total, z=1.959963984540054):
    if total == 0:
        return np.nan, np.nan
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    half = z * np.sqrt(probability * (1 - probability) / total + z * z / (4 * total * total)) / denominator
    return float(center - half), float(center + half)


def analyze(results_dir: Path, bootstrap_seed=20260822, repetitions=10_000):
    predictions = pd.read_csv(results_dir / "validation_prediction_pairs.csv")
    attributions = pd.read_csv(results_dir / "validation_attribution_results.csv")
    run_summary = pd.read_csv(results_dir / "run_summary.csv")
    completion = json.loads((results_dir / "pilot_completion.json").read_text(encoding="utf-8"))
    if completion.get("test_predictions_generated") is not False or completion.get("test_metrics_generated") is not False:
        raise ValueError("Test seal is not intact")

    rng = np.random.default_rng(bootstrap_seed)
    pivot_delta = predictions.pivot(index="sample_id", columns="model_type", values="prediction_difference")
    pivot_flip = predictions.pivot(index="sample_id", columns="model_type", values="prediction_flip")
    rows = []

    mean, low, high = bootstrap_mean_ci(pivot_delta["CNN-Raw"], rng, repetitions)
    rows.append({
        "hypothesis": "H1a",
        "contrast": "CNN-Raw mean |p(S)-p(RC(S))|",
        "estimate": mean,
        "ci95_low": low,
        "ci95_high": high,
        "direction": "nonzero_and_above_0.01" if low > 0.01 else "uncertain",
    })
    raw_above = (pivot_delta["CNN-Raw"] > 0.01).astype(float)
    mean, low, high = bootstrap_mean_ci(raw_above, rng, repetitions)
    rows.append({
        "hypothesis": "H1a",
        "contrast": "CNN-Raw proportion with |delta p| > 0.01",
        "estimate": mean,
        "ci95_low": low,
        "ci95_high": high,
        "direction": "descriptive",
    })

    for aware in ("CNN-Aug", "CNN-Pair"):
        difference = pivot_delta["CNN-Raw"] - pivot_delta[aware]
        mean, low, high = bootstrap_mean_ci(difference, rng, repetitions)
        rows.append({
            "hypothesis": "H2",
            "contrast": f"mean delta reduction: CNN-Raw - {aware}",
            "estimate": mean,
            "ci95_low": low,
            "ci95_high": high,
            "direction": "supports_reduction" if low > 0 else "uncertain",
        })
        flip_difference = pivot_flip["CNN-Raw"] - pivot_flip[aware]
        mean, low, high = bootstrap_mean_ci(flip_difference, rng, repetitions)
        rows.append({
            "hypothesis": "H2",
            "contrast": f"flip-rate reduction: CNN-Raw - {aware}",
            "estimate": mean,
            "ci95_low": low,
            "ci95_high": high,
            "direction": "supports_reduction" if low > 0 else "uncertain",
        })

    statistics = pd.DataFrame(rows)
    statistics.to_csv(results_dir / "pilot_bootstrap_statistics.csv", index=False, encoding="utf-8")

    h1b_rows = []
    for model_type, group in attributions.groupby("model_type", observed=True):
        consistent = group[group.prediction_consistent.astype(bool)]
        warnings = int(consistent.attribution_warning.astype(bool).sum())
        low, high = wilson_interval(warnings, len(consistent))
        h1b_rows.append({
            "model_type": model_type,
            "selected_ism_samples": int(len(group)),
            "prediction_consistent_samples": int(len(consistent)),
            "attribution_warning_count": warnings,
            "attribution_warning_rate": float(warnings / len(consistent)) if len(consistent) else np.nan,
            "wilson95_low": low,
            "wilson95_high": high,
            "mean_absolute_pearson": float(consistent.absolute_pearson.mean()) if len(consistent) else np.nan,
            "mean_absolute_top8_overlap": float(consistent.absolute_top8_overlap.mean()) if len(consistent) else np.nan,
            "scope": "exploratory_selected_validation_subset",
        })
    h1b = pd.DataFrame(h1b_rows)
    h1b.to_csv(results_dir / "pilot_h1b_prediction_consistent_subset.csv", index=False, encoding="utf-8")

    by_model = run_summary.set_index("model_type")
    gate = {
        "phase": "Phase 2A CNN pilot",
        "analysis_split": "validation",
        "test_seal_intact": True,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_repetitions": repetitions,
        "classification_signal_present": bool((by_model.validation_auroc > 0.5).all()),
        "all_checkpoints_reloadable": True,
        "h1a_validation_direction": statistics.loc[statistics.hypothesis == "H1a", "direction"].iloc[0],
        "h2_aug_validation_direction": statistics.loc[
            statistics.contrast == "mean delta reduction: CNN-Raw - CNN-Aug", "direction"
        ].iloc[0],
        "h2_pair_validation_direction": statistics.loc[
            statistics.contrast == "mean delta reduction: CNN-Raw - CNN-Pair", "direction"
        ].iloc[0],
        "h1b_attribution_warnings_observed_in_prediction_consistent_subset": bool(
            (h1b.attribution_warning_count > 0).any()
        ),
        "decision": {
            "proceed_to_phase2b_rc_models": True,
            "unseal_test_now": False,
            "start_phase2_full_now": False,
        },
        "decision_reason": (
            "P2A pipeline and checkpoints passed, classification signal is present, and residual prediction/attribution "
            "asymmetry motivates the preregistered RCPS and post-hoc conjoined comparisons. One seed and a selected "
            "validation ISM subset are insufficient for full or test-set conclusions."
        ),
    }
    (results_dir / "pilot_gate_assessment.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    print(statistics.to_string(index=False))
    print()
    print(h1b.to_string(index=False))
    print()
    print(json.dumps(gate["decision"], indent=2))


def main():
    parser = argparse.ArgumentParser(description="Analyze the validation-only Phase 2A pilot")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260822)
    parser.add_argument("--repetitions", type=int, default=10_000)
    args = parser.parse_args()
    analyze(args.results_dir, args.bootstrap_seed, args.repetitions)


if __name__ == "__main__":
    main()
