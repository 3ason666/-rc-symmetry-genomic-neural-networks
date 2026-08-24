from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "metadata" / "phase3a_candidate_registry.json"
RAW_DIR = ROOT / "data" / "phase3" / "raw"
OUTPUT_PATH = ROOT / "results" / "phase3a_external_selection" / "bed_qc.json"
PRIMARY_CHROMOSOMES = {f"chr{index}" for index in range(1, 23)}


def inspect_bed(path: Path, narrow_peak: bool) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": path.relative_to(ROOT).as_posix(),
            "missing": True,
            "passed_structural_qc": False,
        }
    line_count = 0
    invalid_coordinate_count = 0
    invalid_summit_count = 0
    short_row_count = 0
    primary_chromosome_count = 0
    chromosome_counts: Counter[str] = Counter()
    minimum_columns = 10 if narrow_peak else 3
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip() or raw_line.startswith(("#", "track", "browser")):
                continue
            line_count += 1
            fields = raw_line.rstrip("\n").split("\t")
            if len(fields) < minimum_columns:
                short_row_count += 1
                continue
            chromosome = fields[0]
            chromosome_counts[chromosome] += 1
            primary_chromosome_count += int(chromosome in PRIMARY_CHROMOSOMES)
            try:
                start, end = int(fields[1]), int(fields[2])
                if start < 0 or end <= start:
                    invalid_coordinate_count += 1
                if narrow_peak:
                    summit = int(fields[9])
                    if summit < 0 or summit >= end - start:
                        invalid_summit_count += 1
            except ValueError:
                invalid_coordinate_count += 1
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "missing": False,
        "line_count": line_count,
        "minimum_required_columns": minimum_columns,
        "short_row_count": short_row_count,
        "invalid_coordinate_count": invalid_coordinate_count,
        "invalid_summit_count": invalid_summit_count if narrow_peak else None,
        "primary_chromosome_rows": primary_chromosome_count,
        "nonprimary_chromosome_rows": line_count - primary_chromosome_count,
        "chromosome_counts": dict(sorted(chromosome_counts.items())),
        "passed_structural_qc": line_count > 0 and short_row_count == 0 and invalid_coordinate_count == 0,
    }


def main() -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    rows = []
    for task in registry["confirmatory_tasks"]:
        for item in task["files"]:
            path = RAW_DIR / f"{item['accession']}.bed.gz"
            result = inspect_bed(path, narrow_peak=item["format"] == "bed narrowPeak")
            result.update({"task_id": task["task_id"], "accession": item["accession"], "role": item["role"]})
            rows.append(result)
    missing_count = sum(bool(row.get("missing")) for row in rows)
    if missing_count:
        status = "incomplete_pending_download"
    else:
        status = "passed" if all(row["passed_structural_qc"] for row in rows) else "failed"
    report = {
        "status": status,
        "file_count": len(rows),
        "missing_file_count": missing_count,
        "rows": rows,
        "training_allowed": False,
        "next_gate": "coordinate_filter_sequence_retrieval_matching_balance_and_leakage_audit",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
