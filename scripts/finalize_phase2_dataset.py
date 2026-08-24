from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2_matching import (
    assign_quantile,
    choose_chromosome_splits,
    cross_split_near_duplicates,
    match_exact_strata,
    maximum_homopolymer,
    nonoverlapping_priority_selection,
    quantile_edges,
    shannon_entropy,
)
from src.phase2_data import IntervalIndex, parse_bed3


PROCESSED = PROJECT_ROOT / "data" / "phase2" / "processed"
METADATA = PROJECT_ROOT / "metadata" / "phase2"
PROTOCOLS = PROJECT_ROOT / "protocols"
POSITIVE_PATH = PROCESSED / "positive_candidates.csv.gz"
NEGATIVE_PATH = PROCESSED / "negative_candidates.csv.gz"
DATASET_PATH = PROCESSED / "phase2_matched_dataset.csv"
PAIR_PATH = PROCESSED / "phase2_matching_pairs.csv"
LOW_COMPLEXITY_SENSITIVITY_PATH = PROCESSED / "phase2_low_complexity_filtered_sensitivity.csv"
CONSERVATIVE_DATASET_PATH = PROCESSED / "phase2_conservative_sensitivity_dataset.csv"
CONSERVATIVE_PAIR_PATH = PROCESSED / "phase2_conservative_sensitivity_pairs.csv"
AUDIT_PATH = METADATA / "final_dataset_audit.json"
REFERENCE_AUDIT_PATH = METADATA / "reference_equivalence_audit.json"
SPLIT_PATH = PROTOCOLS / "phase2_frozen_chromosome_split.txt"
GATA_PEAK_PATH = PROJECT_ROOT / "data" / "phase2" / "raw" / "ENCFF148JKK.bed.gz"

MATCHING_SEED = 20260821
ATAC_QUANTILE_BINS = 10
LOW_COMPLEXITY_ENTROPY_THRESHOLD = 1.2
LOW_COMPLEXITY_HOMOPOLYMER_THRESHOLD = 12


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_records(frame: pd.DataFrame) -> list[dict]:
    return frame.to_dict(orient="records")


def assign_accessibility_quantiles(
    positives: list[dict], negatives: list[dict]
) -> tuple[list[dict], list[dict], dict[str, list[float]]]:
    by_chromosome: dict[str, list[float]] = defaultdict(list)
    for row in negatives:
        by_chromosome[str(row["chromosome"])].append(float(row["source_signal"]))
    edge_map = {
        chromosome: quantile_edges(values, bins=ATAC_QUANTILE_BINS)
        for chromosome, values in by_chromosome.items()
    }
    for row in positives:
        row["atac_quantile"] = assign_quantile(
            float(row["atac_overlap_signal"]), edge_map[str(row["chromosome"])]
        )
    for row in negatives:
        row["atac_quantile"] = assign_quantile(
            float(row["source_signal"]), edge_map[str(row["chromosome"])]
        )
    return positives, negatives, {
        chromosome: [float(value) for value in edges]
        for chromosome, edges in edge_map.items()
    }


def build_matched_rows(pairs: list[tuple[dict, dict]]) -> tuple[list[dict], list[dict]]:
    dataset_rows: list[dict] = []
    pair_rows: list[dict] = []
    ordered = sorted(
        pairs,
        key=lambda pair: (
            int(str(pair[0]["chromosome"]).removeprefix("chr")),
            int(pair[0]["start"]),
            str(pair[0]["sample_id"]),
        ),
    )
    for pair_index, (positive, negative) in enumerate(ordered, start=1):
        pair_id = f"pair_{pair_index:06d}"
        pair_rows.append(
            {
                "pair_id": pair_id,
                "chromosome": positive["chromosome"],
                "gc_bin_0.02": int(positive["gc_bin_0.02"]),
                "atac_quantile": int(positive["atac_quantile"]),
                "positive_gc_bin_0.02": int(positive["gc_bin_0.02"]),
                "negative_gc_bin_0.02": int(negative["gc_bin_0.02"]),
                "positive_atac_quantile": int(positive["atac_quantile"]),
                "negative_atac_quantile": int(negative["atac_quantile"]),
                "positive_sample_id": positive["sample_id"],
                "negative_sample_id": negative["sample_id"],
                "positive_gc": positive["gc_fraction"],
                "negative_gc": negative["gc_fraction"],
                "positive_atac_signal": positive["atac_overlap_signal"],
                "negative_atac_signal": negative["source_signal"],
            }
        )
        for label, source in ((1, positive), (0, negative)):
            dataset_rows.append(
                {
                    "sample_id": source["sample_id"],
                    "pair_id": pair_id,
                    "sequence": source["sequence"],
                    "canonical_key": source["canonical_key"],
                    "label": label,
                    "chromosome": source["chromosome"],
                    "start": int(source["start"]),
                    "end": int(source["end"]),
                    "summit": int(source["summit"]),
                    "source_accession": source["source_accession"],
                    "source_signal": source["source_signal"],
                    "gc_fraction": source["gc_fraction"],
                    "gc_bin_0.02": int(source["gc_bin_0.02"]),
                    "atac_quantile": int(source["atac_quantile"]),
                }
            )
    return dataset_rows, pair_rows


def leakage_audit(rows: list[dict], split_map: dict[str, str]) -> dict[str, object]:
    for row in rows:
        row["split"] = split_map[str(row["chromosome"])]

    sample_ids = [str(row["sample_id"]) for row in rows]
    same_peak_duplicates = len(sample_ids) - len(set(sample_ids))

    canonical_groups: dict[str, list[dict]] = defaultdict(list)
    sequence_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        canonical_groups[str(row["canonical_key"])].append(row)
        sequence_groups[str(row["sequence"])].append(row)
    cross_split_canonical = [
        key for key, group in canonical_groups.items() if len({row["split"] for row in group}) > 1
    ]
    cross_split_exact = [
        key for key, group in sequence_groups.items() if len({row["split"] for row in group}) > 1
    ]
    label_conflicts = [
        key for key, group in canonical_groups.items() if len({int(row["label"]) for row in group}) > 1
    ]

    near_duplicates = cross_split_near_duplicates(rows, max_hamming_distance=2)
    low_complexity = []
    for row in rows:
        sequence = str(row["sequence"])
        entropy = shannon_entropy(sequence)
        homopolymer = maximum_homopolymer(sequence)
        if entropy < LOW_COMPLEXITY_ENTROPY_THRESHOLD or homopolymer >= LOW_COMPLEXITY_HOMOPOLYMER_THRESHOLD:
            low_complexity.append(
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "label": int(row["label"]),
                    "entropy": entropy,
                    "maximum_homopolymer": homopolymer,
                }
            )

    blockers = {
        "same_peak_duplicates": same_peak_duplicates,
        "cross_split_exact_duplicates": len(cross_split_exact),
        "cross_split_exact_or_rc_duplicates": len(cross_split_canonical),
        "cross_split_hamming_distance_le_2": len(near_duplicates),
        "opposite_label_canonical_conflicts": len(label_conflicts),
    }
    return {
        "blockers": blockers,
        "passed": all(value == 0 for value in blockers.values()),
        "cross_split_exact_examples": cross_split_exact[:20],
        "cross_split_canonical_examples": cross_split_canonical[:20],
        "cross_split_near_duplicate_examples": near_duplicates[:20],
        "opposite_label_conflict_examples": label_conflicts[:20],
        "low_complexity": {
            "definition": {
                "shannon_entropy_below": LOW_COMPLEXITY_ENTROPY_THRESHOLD,
                "maximum_homopolymer_at_least": LOW_COMPLEXITY_HOMOPOLYMER_THRESHOLD,
            },
            "count": len(low_complexity),
            "examples": low_complexity[:20],
        },
    }


def matching_balance(pair_rows: list[dict]) -> dict[str, object]:
    positive_gc = np.asarray([float(row["positive_gc"]) for row in pair_rows])
    negative_gc = np.asarray([float(row["negative_gc"]) for row in pair_rows])
    positive_atac = np.log1p([float(row["positive_atac_signal"]) for row in pair_rows])
    negative_atac = np.log1p([float(row["negative_atac_signal"]) for row in pair_rows])

    def standardized_mean_difference(left: np.ndarray, right: np.ndarray) -> float:
        pooled = np.sqrt((left.var(ddof=1) + right.var(ddof=1)) / 2)
        return float((left.mean() - right.mean()) / pooled) if pooled else 0.0

    gc_ks = ks_2samp(positive_gc, negative_gc)
    atac_ks = ks_2samp(positive_atac, negative_atac)
    gc_smd = standardized_mean_difference(positive_gc, negative_gc)
    atac_smd = standardized_mean_difference(positive_atac, negative_atac)
    exact_stratum_mismatches = sum(
        int(row["positive_gc_bin_0.02"]) != int(row["negative_gc_bin_0.02"])
        or int(row["positive_atac_quantile"]) != int(row["negative_atac_quantile"])
        for row in pair_rows
    )
    return {
        "passed": abs(gc_smd) < 0.1 and abs(atac_smd) < 0.1 and exact_stratum_mismatches == 0,
        "threshold_absolute_smd": 0.1,
        "exact_stratum_mismatches": exact_stratum_mismatches,
        "gc": {
            "mean_absolute_pair_difference": float(np.abs(positive_gc - negative_gc).mean()),
            "maximum_absolute_pair_difference": float(np.abs(positive_gc - negative_gc).max()),
            "standardized_mean_difference": gc_smd,
            "ks_statistic": float(gc_ks.statistic),
            "ks_pvalue": float(gc_ks.pvalue),
        },
        "log1p_atac_signal": {
            "mean_absolute_pair_difference": float(np.abs(positive_atac - negative_atac).mean()),
            "standardized_mean_difference": atac_smd,
            "ks_statistic": float(atac_ks.statistic),
            "ks_pvalue": float(atac_ks.pvalue),
        },
    }
def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def counts_by_split(rows: list[dict]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        result[split] = {
            "total": len(selected),
            "positive": sum(int(row["label"]) == 1 for row in selected),
            "negative": sum(int(row["label"]) == 0 for row in selected),
        }
    return result


def main() -> int:
    if not POSITIVE_PATH.exists() or not NEGATIVE_PATH.exists():
        raise FileNotFoundError("candidate tables are missing; run build_phase2_candidates_ensembl first")
    positive_frame = pd.read_csv(POSITIVE_PATH)
    negative_frame = pd.read_csv(NEGATIVE_PATH)
    positives = as_records(
        positive_frame[(positive_frame["peak_set"] == "optimal") & (positive_frame["eligible_primary"] == 1)]
    )
    negatives = as_records(negative_frame)

    # Recheck against every GATA1 BED interval, including the three records
    # whose summit offsets are invalid and therefore absent from positive windows.
    all_gata_intervals = parse_bed3(GATA_PEAK_PATH)
    gata_exclusion = {
        chromosome: IntervalIndex((max(0, start - 128), end + 128) for start, end in intervals)
        for chromosome, intervals in all_gata_intervals.items()
    }
    empty_interval_index = IntervalIndex()
    negatives_before_all_peak_check = len(negatives)
    negatives = [
        row
        for row in negatives
        if not gata_exclusion.get(str(row["chromosome"]), empty_interval_index).overlaps(
            int(row["start"]), int(row["end"])
        )
    ]
    additional_negative_exclusions = negatives_before_all_peak_check - len(negatives)

    positives, overlap_rejected = nonoverlapping_priority_selection(positives)
    positives, negatives, quantile_edge_map = assign_accessibility_quantiles(positives, negatives)
    pairs, unmatched = match_exact_strata(positives, negatives, seed=MATCHING_SEED)
    dataset_rows, pair_rows = build_matched_rows(pairs)

    pair_counts = Counter(str(positive["chromosome"]) for positive, _ in pairs)
    splits = choose_chromosome_splits(dict(pair_counts))
    split_map = {
        chromosome: split
        for split, chromosomes in splits.items()
        for chromosome in chromosomes
    }
    leakage = leakage_audit(dataset_rows, split_map)
    balance = matching_balance(pair_rows)
    dataset_rows.sort(
        key=lambda row: (
            {"train": 0, "validation": 1, "test": 2}[row["split"]],
            int(str(row["chromosome"]).removeprefix("chr")),
            int(row["label"]),
            int(row["start"]),
            str(row["sample_id"]),
        )
    )

    dataset_fields = [
        "sample_id", "pair_id", "sequence", "canonical_key", "label", "split",
        "chromosome", "start", "end", "summit", "source_accession", "source_signal",
        "gc_fraction", "gc_bin_0.02", "atac_quantile",
    ]
    pair_fields = [
        "pair_id", "chromosome", "gc_bin_0.02", "atac_quantile",
        "positive_gc_bin_0.02", "negative_gc_bin_0.02",
        "positive_atac_quantile", "negative_atac_quantile",
        "positive_sample_id", "negative_sample_id", "positive_gc", "negative_gc",
        "positive_atac_signal", "negative_atac_signal",
    ]
    write_csv(DATASET_PATH, dataset_rows, dataset_fields)
    write_csv(PAIR_PATH, pair_rows, pair_fields)

    split_counts = counts_by_split(dataset_rows)

    low_complexity_pair_ids = {
        str(row["pair_id"])
        for row in dataset_rows
        if shannon_entropy(str(row["sequence"])) < LOW_COMPLEXITY_ENTROPY_THRESHOLD
        or maximum_homopolymer(str(row["sequence"])) >= LOW_COMPLEXITY_HOMOPOLYMER_THRESHOLD
    }
    low_complexity_sensitivity_rows = [
        row for row in dataset_rows if str(row["pair_id"]) not in low_complexity_pair_ids
    ]
    write_csv(LOW_COMPLEXITY_SENSITIVITY_PATH, low_complexity_sensitivity_rows, dataset_fields)

    conservative_positives = as_records(
        positive_frame[
            (positive_frame["peak_set"] == "conservative")
            & (positive_frame["eligible_primary"] == 1)
        ]
    )
    conservative_positives, conservative_overlap_rejected = nonoverlapping_priority_selection(
        conservative_positives
    )
    conservative_positives, conservative_negatives, _ = assign_accessibility_quantiles(
        conservative_positives, [dict(row) for row in negatives]
    )
    conservative_pairs, conservative_unmatched = match_exact_strata(
        conservative_positives, conservative_negatives, seed=MATCHING_SEED + 1
    )
    conservative_rows, conservative_pair_rows = build_matched_rows(conservative_pairs)
    conservative_leakage = leakage_audit(conservative_rows, split_map)
    conservative_balance = matching_balance(conservative_pair_rows)
    conservative_rows.sort(
        key=lambda row: (
            {"train": 0, "validation": 1, "test": 2}[row["split"]],
            int(str(row["chromosome"]).removeprefix("chr")),
            int(row["label"]),
            int(row["start"]),
            str(row["sample_id"]),
        )
    )
    write_csv(CONSERVATIVE_DATASET_PATH, conservative_rows, dataset_fields)
    write_csv(CONSERVATIVE_PAIR_PATH, conservative_pair_rows, pair_fields)

    reference_audit = json.loads(REFERENCE_AUDIT_PATH.read_text(encoding="utf-8"))
    all_gates_passed = (
        bool(reference_audit.get("passed"))
        and bool(balance["passed"])
        and bool(leakage["passed"])
        and bool(conservative_balance["passed"])
        and bool(conservative_leakage["passed"])
    )

    audit = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matching": {
            "seed": MATCHING_SEED,
            "gc_bin_width": 0.02,
            "atac_quantile_bins": ATAC_QUANTILE_BINS,
            "positive_candidates_before_overlap_filter": int(
                ((positive_frame["peak_set"] == "optimal") & (positive_frame["eligible_primary"] == 1)).sum()
            ),
            "positive_overlap_rejected": len(overlap_rejected),
            "positive_after_overlap_filter": len(positives),
            "matched_pairs": len(pairs),
            "unmatched_positives": len(unmatched),
            "additional_negatives_excluded_by_all_GATA1_intervals": additional_negative_exclusions,
            "quantile_edges_by_chromosome": quantile_edge_map,
        },
        "chromosome_pair_counts": dict(sorted(pair_counts.items(), key=lambda item: int(item[0].removeprefix("chr")))),
        "splits": splits,
        "split_counts": split_counts,
        "matching_balance": balance,
        "leakage_audit": leakage,
        "low_complexity_sensitivity": {
            "policy": "Remove the complete matched pair if either member triggers the frozen low-complexity rule.",
            "excluded_pairs": len(low_complexity_pair_ids),
            "retained_pairs": len(low_complexity_sensitivity_rows) // 2,
            "split_counts": counts_by_split(low_complexity_sensitivity_rows),
            "dataset_sha256": file_sha256(LOW_COMPLEXITY_SENSITIVITY_PATH),
        },
        "conservative_peak_sensitivity": {
            "matching_seed": MATCHING_SEED + 1,
            "positive_candidates_before_overlap_filter": int(
                ((positive_frame["peak_set"] == "conservative") & (positive_frame["eligible_primary"] == 1)).sum()
            ),
            "positive_overlap_rejected": len(conservative_overlap_rejected),
            "matched_pairs": len(conservative_pairs),
            "unmatched_positives": len(conservative_unmatched),
            "split_counts": counts_by_split(conservative_rows),
            "matching_balance": conservative_balance,
            "leakage_audit": conservative_leakage,
            "dataset_sha256": file_sha256(CONSERVATIVE_DATASET_PATH),
            "pair_table_sha256": file_sha256(CONSERVATIVE_PAIR_PATH),
        },
        "dataset_sha256": file_sha256(DATASET_PATH),
        "pair_table_sha256": file_sha256(PAIR_PATH),
        "reference_equivalence": {
            "passed": bool(reference_audit.get("passed")),
            "audit_path": str(REFERENCE_AUDIT_PATH.relative_to(PROJECT_ROOT)),
            "audit_sha256": file_sha256(REFERENCE_AUDIT_PATH),
        },
        "remaining_gate": None if all_gates_passed else "One or more data gates failed; inspect this audit.",
        "formal_training_allowed": all_gates_passed,
    }
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    split_lines = [
        "PHASE 2 FROZEN CHROMOSOME SPLIT",
        "Version: 1.0 candidate freeze",
        f"Generated UTC: {audit['generated_at_utc']}",
        f"Matching seed: {MATCHING_SEED}",
        f"Dataset SHA-256: {audit['dataset_sha256']}",
        "",
    ]
    for split in ("train", "validation", "test"):
        counts = split_counts[split]
        split_lines.extend(
            [
                f"{split.upper()}: {', '.join(splits[split])}",
                f"  total={counts['total']} positive={counts['positive']} negative={counts['negative']}",
            ]
        )
    split_lines.extend(
        [
            "",
            f"Leakage gate passed: {leakage['passed']}",
            f"Formal training allowed: {all_gates_passed}",
        ]
    )
    SPLIT_PATH.write_text("\n".join(split_lines) + "\n", encoding="utf-8")
    print(f"matched_pairs={len(pairs)}")
    print(f"splits={splits}")
    print(f"leakage_passed={str(leakage['passed']).lower()}")
    print(f"dataset={DATASET_PATH}")
    print(f"audit={AUDIT_PATH}")
    return 0 if all_gates_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
