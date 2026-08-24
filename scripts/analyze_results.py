from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from pandas.api.types import is_numeric_dtype
from src.metrics import describe_columns, safe_similarity


def analyze(results_dir: Path):
    samples = pd.read_csv(results_dir / "per_sample_results.csv")
    value_cols = [c for c in samples.columns
                  if c.startswith(("prediction_", "attribution_", "motif_", "position_"))
                  and c != "prediction_flip" and is_numeric_dtype(samples[c])]
    describe_columns(samples, ["model_type", "seed"], value_cols).to_csv(results_dir / "summary_by_model_seed.csv", index=False, encoding="utf-8")
    describe_columns(samples, ["model_type"], value_cols).to_csv(results_dir / "summary_by_model.csv", index=False, encoding="utf-8")
    seed_means = samples.groupby(["model_type", "seed"], as_index=False)[value_cols].mean()
    describe_columns(seed_means, ["model_type"], value_cols).to_csv(
        results_dir / "summary_across_seeds.csv", index=False, encoding="utf-8"
    )
    relation_rows = []
    for keys, group in samples.groupby(["model_type", "seed"]):
        for method in ("pearson", "spearman"):
            value, issue = safe_similarity(group.prediction_consistency, group.attribution_pearson_absolute, method)
            relation_rows.append({"model_type": keys[0], "seed": keys[1], "method": method, "correlation": value, "issue": issue})
    for model_type, group in samples.groupby("model_type"):
        for method in ("pearson", "spearman"):
            value, issue = safe_similarity(group.prediction_consistency, group.attribution_pearson_absolute, method)
            relation_rows.append({"model_type": model_type, "seed": "all", "method": method,
                                  "correlation": value, "issue": issue})
    pd.DataFrame(relation_rows).to_csv(results_dir / "prediction_attribution_relationship.csv", index=False, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--results-dir", type=Path, required=True)
    analyze(parser.parse_args().results_dir)


if __name__ == "__main__":
    main()
