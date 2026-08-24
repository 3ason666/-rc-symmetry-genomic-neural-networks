from __future__ import annotations

import numpy as np
import pandas as pd


def benjamini_hochberg(p_values) -> np.ndarray:
    """Benjamini-Hochberg adjusted p values, preserving missing entries."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def _hierarchical_scalar_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    by_seed = {
        int(seed): group[value_column].dropna().to_numpy(float)
        for seed, group in frame.groupby("seed")
    }
    seeds = np.asarray(list(by_seed))
    estimates = np.empty(n_bootstrap, dtype=float)
    for bootstrap_index in range(n_bootstrap):
        sampled_seed_means = []
        for seed in rng.choice(seeds, size=len(seeds), replace=True):
            values = by_seed[int(seed)]
            sampled_seed_means.append(float(rng.choice(values, size=len(values), replace=True).mean()))
        estimates[bootstrap_index] = float(np.mean(sampled_seed_means))
    return estimates


def hierarchical_paired_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    model_type: str,
    reference_type: str = "CNN-Raw",
    n_bootstrap: int = 2000,
    random_seed: int = 5150,
) -> dict:
    """Equal-seed-weight paired bootstrap of model minus reference."""
    pivot = frame.pivot(index=["seed", "sample_id"], columns="model_type", values=value_column)
    paired = pivot[[model_type, reference_type]].dropna().reset_index()
    paired["difference"] = paired[model_type] - paired[reference_type]
    rng = np.random.default_rng(random_seed)
    draws = _hierarchical_scalar_bootstrap(paired, "difference", n_bootstrap, rng)
    observed = float(paired.groupby("seed")["difference"].mean().mean())
    return {
        "model_type": model_type,
        "reference_type": reference_type,
        "metric": value_column,
        "effect_model_minus_reference": observed,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "bootstrap_p_two_sided": float(min(1.0, 2 * min(np.mean(draws <= 0), np.mean(draws >= 0)))),
        "n_seeds": int(paired.seed.nunique()),
        "n_pairs": int(len(paired)),
        "n_bootstrap": int(n_bootstrap),
    }


def scalar_bootstrap_summary(
    frame: pd.DataFrame,
    value_column: str,
    model_type: str,
    n_bootstrap: int = 2000,
    random_seed: int = 5150,
) -> dict:
    subset = frame[frame.model_type == model_type]
    rng = np.random.default_rng(random_seed)
    draws = _hierarchical_scalar_bootstrap(subset, value_column, n_bootstrap, rng)
    observed = float(subset.groupby("seed")[value_column].mean().mean())
    return {
        "model_type": model_type,
        "metric": value_column,
        "estimate": observed,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "n_seeds": int(subset.seed.nunique()),
        "n_samples": int(len(subset)),
        "n_bootstrap": int(n_bootstrap),
    }


def build_full_statistics(
    prediction_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    run_df: pd.DataFrame,
    model_types: list[str],
    n_bootstrap: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_metrics = [
        "validation_auroc", "test_auroc", "test_auprc", "test_accuracy", "test_f1",
        "prediction_mean_absolute_difference", "prediction_p95_absolute_difference",
        "symmetry_flip_rate",
    ]
    run_crossseed_rows = []
    for model_type, group in run_df.groupby("model_type"):
        row = {"model_type": model_type, "n_seeds": int(group.seed.nunique())}
        for metric in run_metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        run_crossseed_rows.append(row)
    run_crossseed = pd.DataFrame(run_crossseed_rows)

    sample_seed = sample_df.groupby(["model_type", "seed"]).agg(
        n_ism=("sample_id", "size"),
        attribution_pearson_absolute=("attribution_pearson_absolute", "mean"),
        attribution_spearman_absolute=("attribution_spearman_absolute", "mean"),
        causal_mass_fraction=("causal_mass_fraction", "mean"),
        shortcut_mass_fraction=("shortcut_mass_fraction", "mean"),
        h1b_h3a_fraction=("prediction_consistent_but_attr_asymmetric", "mean"),
        matrix_max_abs_error=("attribution_matrix_max_abs_error", "max"),
    ).reset_index()
    sample_crossseed = sample_seed.groupby("model_type").agg(
        n_seeds=("seed", "nunique"),
        attribution_pearson_mean=("attribution_pearson_absolute", "mean"),
        attribution_pearson_sd=("attribution_pearson_absolute", "std"),
        causal_mass_mean=("causal_mass_fraction", "mean"),
        causal_mass_sd=("causal_mass_fraction", "std"),
        shortcut_mass_mean=("shortcut_mass_fraction", "mean"),
        shortcut_mass_sd=("shortcut_mass_fraction", "std"),
        h1b_h3a_fraction_mean=("h1b_h3a_fraction", "mean"),
        h1b_h3a_fraction_sd=("h1b_h3a_fraction", "std"),
        matrix_max_abs_error=("matrix_max_abs_error", "max"),
    ).reset_index()

    comparisons = []
    for offset, model_type in enumerate(model_types):
        if model_type == "CNN-Raw":
            continue
        comparisons.append(hierarchical_paired_bootstrap(
            prediction_df, "prediction_difference", model_type,
            n_bootstrap=n_bootstrap, random_seed=bootstrap_seed + offset,
        ))
        comparisons.append(hierarchical_paired_bootstrap(
            sample_df, "attribution_pearson_absolute", model_type,
            n_bootstrap=n_bootstrap, random_seed=bootstrap_seed + 100 + offset,
        ))
        comparisons.append(hierarchical_paired_bootstrap(
            sample_df, "causal_mass_fraction", model_type,
            n_bootstrap=n_bootstrap, random_seed=bootstrap_seed + 200 + offset,
        ))
    comparison_df = pd.DataFrame(comparisons)
    comparison_df["fdr_bh"] = benjamini_hochberg(comparison_df["bootstrap_p_two_sided"])
    return run_crossseed, sample_seed, sample_crossseed, comparison_df
