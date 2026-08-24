from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "results" / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results" / "phase3d_cross_phase_overlap_audit"
PRIMARY = ROOT / "results" / "phase3c_one_time_test"
SENSITIVITY = ROOT / "results" / "phase3e_near_duplicate_sensitivity" / "evaluation"
OUTPUT = ROOT / "results" / "phase3e_near_duplicate_sensitivity" / "report_tables"
TASKS = ("p3_gata1_fetal", "p3_ctcf_gm12878")


def decision_lookup(root: Path, task: str) -> dict[str, dict]:
    rows = json.loads((root / task / "test_hypothesis_decisions.json").read_text(encoding="utf-8"))
    return {row["hypothesis"]: row for row in rows}


def hypothesis_comparison() -> pd.DataFrame:
    rows = []
    for task in TASKS:
        primary_boot = pd.read_csv(PRIMARY / task / "test_confirmatory_bootstrap.csv").set_index("hypothesis")
        sensitivity_boot = pd.read_csv(SENSITIVITY / task / "test_confirmatory_bootstrap.csv").set_index("hypothesis")
        primary_decisions = decision_lookup(PRIMARY, task)
        sensitivity_decisions = decision_lookup(SENSITIVITY, task)
        for hypothesis in ("H1a", "H1b", "H2", "H3a", "H3b", "H4"):
            primary_row = primary_boot.loc[hypothesis] if hypothesis in primary_boot.index else None
            sensitivity_row = sensitivity_boot.loc[hypothesis] if hypothesis in sensitivity_boot.index else None
            rows.append({
                "task_id": task,
                "hypothesis": hypothesis,
                "primary_status": primary_decisions[hypothesis]["status"],
                "sensitivity_status": sensitivity_decisions[hypothesis]["status"],
                "status_stable": primary_decisions[hypothesis]["status"] == sensitivity_decisions[hypothesis]["status"],
                "primary_n": int(primary_row["n"]) if primary_row is not None and pd.notna(primary_row["n"]) else 0,
                "primary_estimate": float(primary_row["estimate"]) if primary_row is not None and pd.notna(primary_row["estimate"]) else np.nan,
                "primary_ci95_low": float(primary_row["ci95_low"]) if primary_row is not None and pd.notna(primary_row["ci95_low"]) else np.nan,
                "primary_ci95_high": float(primary_row["ci95_high"]) if primary_row is not None and pd.notna(primary_row["ci95_high"]) else np.nan,
                "sensitivity_n": int(sensitivity_row["n"]) if sensitivity_row is not None and pd.notna(sensitivity_row["n"]) else 0,
                "sensitivity_estimate": float(sensitivity_row["estimate"]) if sensitivity_row is not None and pd.notna(sensitivity_row["estimate"]) else np.nan,
                "sensitivity_ci95_low": float(sensitivity_row["ci95_low"]) if sensitivity_row is not None and pd.notna(sensitivity_row["ci95_low"]) else np.nan,
                "sensitivity_ci95_high": float(sensitivity_row["ci95_high"]) if sensitivity_row is not None and pd.notna(sensitivity_row["ci95_high"]) else np.nan,
                "primary_evidence": primary_decisions[hypothesis]["evidence"],
                "sensitivity_evidence": sensitivity_decisions[hypothesis]["evidence"],
            })
    return pd.DataFrame(rows)


def ensemble_comparison() -> pd.DataFrame:
    rows = []
    for task in TASKS:
        for analysis, root in (("primary", PRIMARY), ("near_duplicate_removed", SENSITIVITY)):
            frame = pd.read_csv(root / task / "test_prediction_ensemble_summary.csv")
            frame.insert(0, "analysis", analysis)
            frame.insert(0, "task_id", task)
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def overlap_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = json.loads((AUDIT / "audit_summary.json").read_text(encoding="utf-8"))
    rows = []
    for task, task_summary in summary["tasks"].items():
        union = task_summary["scope_summaries"]["union"]
        for method in ("coordinate", "exact", "high_similarity"):
            item = union[method]
            rows.append({
                "task_id": task,
                "method": method,
                "p3_train_rows": union["p3_train_rows"],
                "unique_p3_rows": item["unique_p3_rows"],
                "fraction_of_p3_train": item["fraction_of_p3_train"],
                "p2f_test_rows": task_summary["p2f_test_rows"],
                "unique_p2f_rows": item["unique_p2f_test_rows"],
                "fraction_of_p2f_test": item["fraction_of_p2f_test"],
                "triggered": task_summary["triggers"]["coordinate_stratification" if method == "coordinate" else ("exact_overlap_rerun" if method == "exact" else "high_similarity_rerun")],
            })
    overview = pd.DataFrame(rows)

    pair_rows = []
    for method, filename in (
        ("coordinate", "coordinate_overlap_pairs.csv.gz"),
        ("exact", "exact_sequence_overlap_pairs.csv.gz"),
        ("high_similarity", "high_similarity_pairs.csv.gz"),
    ):
        pairs = pd.read_csv(AUDIT / filename)
        for task, group in pairs.groupby("task_id", observed=True):
            pair_rows.append({
                "task_id": task,
                "method": method,
                "pair_count": int(len(group)),
                "label_concordant_pairs": int((group.p3_label.astype(int) == group.p2f_label.astype(int)).sum()),
                "label_concordance_fraction": float((group.p3_label.astype(int) == group.p2f_label.astype(int)).mean()),
            })
    return overview, pd.DataFrame(pair_rows)


def effect_figure(comparison: pd.DataFrame) -> None:
    plot = comparison[comparison.hypothesis.isin(["H1a", "H2", "H3a"])].copy()
    labels = {"H1a": "H1a Raw prediction asymmetry", "H2": "H2 Aug - Raw asymmetry", "H3a": "H3a Aug attribution asymmetry"}
    task_labels = {"p3_gata1_fetal": "Fetal GATA1 (primary)", "p3_ctcf_gm12878": "GM12878 CTCF (supportive)"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharex=False)
    colors = {"primary": "#2E74B5", "sensitivity": "#C96B2C"}
    for ax, task in zip(axes, TASKS):
        sub = plot[plot.task_id.eq(task)].set_index("hypothesis").loc[["H1a", "H2", "H3a"]]
        y = np.arange(3)
        for analysis, offset, marker in (("primary", -0.11, "o"), ("sensitivity", 0.11, "s")):
            estimates = sub[f"{analysis}_estimate"].to_numpy(float)
            low = sub[f"{analysis}_ci95_low"].to_numpy(float)
            high = sub[f"{analysis}_ci95_high"].to_numpy(float)
            ax.errorbar(estimates, y + offset, xerr=np.vstack([estimates - low, high - estimates]), fmt=marker,
                        color=colors[analysis], capsize=3, linewidth=1.5, markersize=5,
                        label="Primary P3C" if analysis == "primary" else "Near-duplicate removed")
        ax.axvline(0, color="#666666", linewidth=0.8, linestyle="--")
        ax.set_yticks(y, [labels[h] for h in ["H1a", "H2", "H3a"]])
        ax.set_title(task_labels[task], fontsize=11, weight="bold")
        ax.set_xlabel("Estimate with 95% CI")
        ax.grid(axis="x", color="#D9DEE5", linewidth=0.6)
        ax.invert_yaxis()
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Primary versus cross-phase near-duplicate sensitivity", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(OUTPUT / "phase3c_vs_phase3e_effects.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    comparison = hypothesis_comparison()
    comparison.to_csv(OUTPUT / "hypothesis_comparison.csv", index=False)
    ensemble_comparison().to_csv(OUTPUT / "ensemble_comparison.csv", index=False)
    overview, concordance = overlap_tables()
    overview.to_csv(OUTPUT / "overlap_overview.csv", index=False)
    concordance.to_csv(OUTPUT / "overlap_label_concordance.csv", index=False)
    effect_figure(comparison)
    print(json.dumps({"status": "completed", "rows": {"hypothesis": len(comparison), "overlap": len(overview), "concordance": len(concordance)}}))


if __name__ == "__main__":
    main()
