from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2_matching import maximum_homopolymer, shannon_entropy


PROCESSED = PROJECT_ROOT / "data" / "phase3" / "processed"
RESULTS = PROJECT_ROOT / "results" / "phase3a_external_selection"
CONFIG = PROJECT_ROOT / "configs" / "phase3a_external_replication.yaml"
PROTOCOL = PROJECT_ROOT / "protocols" / "phase3a_external_replication_selection_protocol.md"
CONSTRUCTION = RESULTS / "dataset_construction_qc.json"
NEAR_AUDIT = RESULTS / "near_duplicate_audit.json"
OUTPUT = RESULTS / "phase3a_final_gate.json"
TRAINING_MANIFEST = PROJECT_ROOT / "metadata" / "phase3b_training_manifest.json"
TASKS = ("p3_gata1_fetal", "p3_ctcf_gm12878")
LOW_COMPLEXITY_ENTROPY = 1.2
LOW_COMPLEXITY_HOMOPOLYMER = 12


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_gzip_csv(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_deterministic_gzip_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as binary:
        with gzip.GzipFile(filename="", mode="wb", fileobj=binary, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    temporary.replace(path)


def balance(rows: list[dict]) -> dict:
    positive = [row for row in rows if int(row["label"]) == 1]
    negative = [row for row in rows if int(row["label"]) == 0]
    positive_gc = np.asarray([float(row["gc_fraction"]) for row in positive])
    negative_gc = np.asarray([float(row["gc_fraction"]) for row in negative])
    positive_access = np.log2(1 + np.asarray([float(row["accessibility_signal"]) for row in positive]))
    negative_access = np.log2(1 + np.asarray([float(row["accessibility_signal"]) for row in negative]))
    gc = ks_2samp(positive_gc, negative_gc)
    access = ks_2samp(positive_access, negative_access)
    return {
        "positive": len(positive),
        "negative": len(negative),
        "gc_ks_d": float(gc.statistic),
        "accessibility_log2p1_ks_d": float(access.statistic),
        "passed": bool(
            len(positive) == len(negative)
            and gc.statistic <= 0.10
            and access.statistic <= 0.10
        ),
    }


def partition_audits(rows: list[dict]) -> dict:
    result = {}
    for field in ("fixed_partition", "fold1_partition", "fold2_partition", "fold3_partition"):
        result[field] = {}
        for value in sorted({row[field] for row in rows}):
            selected = [row for row in rows if row[field] == value]
            result[field][value] = balance(selected)
    return result


def low_complexity_audit(rows: list[dict]) -> dict:
    counts: Counter[str] = Counter()
    examples = []
    for row in rows:
        sequence = row["sequence"]
        entropy = shannon_entropy(sequence)
        homopolymer = maximum_homopolymer(sequence)
        trigger = entropy < LOW_COMPLEXITY_ENTROPY or homopolymer >= LOW_COMPLEXITY_HOMOPOLYMER
        counts["total"] += 1
        counts["triggered"] += int(trigger)
        counts[f'label_{row["label"]}_total'] += 1
        counts[f'label_{row["label"]}_triggered'] += int(trigger)
        if trigger and len(examples) < 20:
            examples.append(
                {
                    "sample_id": row["sample_id"],
                    "label": int(row["label"]),
                    "entropy": entropy,
                    "maximum_homopolymer": homopolymer,
                }
            )
    return {
        "definition": {
            "shannon_entropy_below": LOW_COMPLEXITY_ENTROPY,
            "maximum_homopolymer_at_least": LOW_COMPLEXITY_HOMOPOLYMER,
        },
        "action": "report_only_no_silent_deletion",
        "count": counts["triggered"],
        "fraction": counts["triggered"] / counts["total"] if counts["total"] else 0.0,
        "by_label": {
            "positive": {
                "count": counts["label_1_triggered"],
                "total": counts["label_1_total"],
            },
            "negative": {
                "count": counts["label_0_triggered"],
                "total": counts["label_0_total"],
            },
        },
        "examples": examples,
    }


def main() -> int:
    construction = json.loads(CONSTRUCTION.read_text(encoding="utf-8"))
    near = json.loads(NEAR_AUDIT.read_text(encoding="utf-8"))
    task_results = {}
    manifest_tasks = {}
    all_passed = construction["status"] == "passed_pending_near_duplicate_audit" and near["status"] == "passed"

    for task_id in TASKS:
        dataset_path = PROCESSED / f"{task_id}_matched_dataset.csv.gz"
        pair_path = PROCESSED / f"{task_id}_matching_pairs.csv.gz"
        rows = load_gzip_csv(dataset_path)
        pairs = load_gzip_csv(pair_path)
        by_pair: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_pair[row["pair_id"]].append(row)
        pair_integrity = (
            len(by_pair) == len(pairs)
            and all(len(group) == 2 and {int(row["label"]) for row in group} == {0, 1} for group in by_pair.values())
        )
        canonical_unique = len({row["canonical_key"] for row in rows}) == len(rows)
        sample_ids_unique = len({row["sample_id"] for row in rows}) == len(rows)
        partition_qc = partition_audits(rows)
        every_partition_balanced = all(
            report["passed"]
            for field in partition_qc.values()
            for report in field.values()
        )
        fixed_test = [row for row in rows if row["fixed_partition"] == "test"]
        development = [row for row in rows if row["fixed_partition"] == "development"]
        test_positive_counts = Counter(
            row["chromosome"] for row in fixed_test if int(row["label"]) == 1
        )
        test_per_class = sum(test_positive_counts.values())
        maximum_test_chromosome_fraction = max(test_positive_counts.values()) / test_per_class
        fold_minimums = all(
            partition_qc[f"fold{fold}_partition"]["train"]["positive"] >= 1000
            and partition_qc[f"fold{fold}_partition"]["validation"]["positive"] >= 200
            for fold in (1, 2, 3)
        )
        final_match_rate = len(pairs) / construction["tasks"][task_id]["retained_positive_candidates"]
        gates = {
            "pair_integrity": pair_integrity,
            "sample_ids_unique": sample_ids_unique,
            "exact_and_rc_canonical_keys_unique": canonical_unique,
            "every_partition_gc_and_accessibility_ks_d_le_0_10": every_partition_balanced,
            "final_matched_rate_ge_0_90": final_match_rate >= 0.90,
            "fixed_test_at_least_three_chromosomes": len(test_positive_counts) >= 3,
            "fixed_test_maximum_single_chromosome_fraction_le_0_50": maximum_test_chromosome_fraction <= 0.50,
            "fixed_test_minimum_200_per_class": test_per_class >= 200,
            "all_folds_minimum_train_1000_validation_200_per_class": fold_minimums,
            "near_duplicate_primary_audit_passed": near["status"] == "passed",
        }
        task_passed = all(gates.values())
        all_passed &= task_passed

        development_path = PROCESSED / f"{task_id}_development.csv.gz"
        sealed_test_path = PROCESSED / f"{task_id}_sealed_test.csv.gz"
        write_deterministic_gzip_csv(development_path, development)
        write_deterministic_gzip_csv(sealed_test_path, fixed_test)
        task_results[task_id] = {
            "status": "passed" if task_passed else "failed",
            "final_pairs": len(pairs),
            "final_matched_rate": final_match_rate,
            "removed_near_duplicate_pairs": near["tasks"][task_id]["removed_pairs"],
            "partition_qc": partition_qc,
            "fixed_test_positive_counts_by_chromosome": dict(test_positive_counts),
            "fixed_test_maximum_single_chromosome_fraction": maximum_test_chromosome_fraction,
            "low_complexity": low_complexity_audit(rows),
            "gates": gates,
            "full_locked_dataset_sha256": sha256(dataset_path),
            "pair_table_sha256": sha256(pair_path),
        }
        manifest_tasks[task_id] = {
            "development_path": str(development_path.relative_to(PROJECT_ROOT)),
            "development_sha256": sha256(development_path),
            "development_rows": len(development),
            "sealed_test_path": str(sealed_test_path.relative_to(PROJECT_ROOT)),
            "sealed_test_sha256": sha256(sealed_test_path),
            "sealed_test_rows": len(fixed_test),
            "test_access_allowed": False,
            "fixed_test_chromosomes": sorted(test_positive_counts, key=lambda value: int(value[3:])),
            "fold_partitions": construction["tasks"][task_id]["splits"]["development_folds"],
        }

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all_passed else "failed",
        "formal_training_allowed": bool(all_passed),
        "test_unsealing_allowed": False,
        "deferred_non_data_gate": {
            "H1b_prediction_consistent_test_subgroup_minimum_100": "assess_only_after_frozen_models_and_one_time_test_unseal"
        },
        "inputs": {
            "construction_qc_sha256": sha256(CONSTRUCTION),
            "near_duplicate_audit_sha256": sha256(NEAR_AUDIT),
            "config_sha256": sha256(CONFIG),
            "protocol_sha256": sha256(PROTOCOL),
        },
        "tasks": task_results,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_for_p3b_training" if all_passed else "blocked",
        "formal_training_allowed": bool(all_passed),
        "test_unsealing_allowed": False,
        "models": ["CNN-Raw", "CNN-Aug", "CNN-RCPS"],
        "seeds": [42, 123, 2026],
        "folds": [1, 2, 3],
        "expected_checkpoints_per_task": 27,
        "test_forbidden_uses": [
            "early_stopping", "checkpoint_selection", "epoch_selection",
            "threshold_selection", "hyperparameter_selection", "code_debugging",
        ],
        "gate_path": str(OUTPUT.relative_to(PROJECT_ROOT)),
        "gate_sha256": sha256(OUTPUT),
        "config_sha256": sha256(CONFIG),
        "protocol_sha256": sha256(PROTOCOL),
        "tasks": manifest_tasks,
    }
    TRAINING_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"status={payload['status']} training_allowed={payload['formal_training_allowed']}")
    print(f"gate={OUTPUT}")
    print(f"manifest={TRAINING_MANIFEST}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
