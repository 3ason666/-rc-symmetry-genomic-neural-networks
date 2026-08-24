from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "phase2" / "raw"
OUTPUT = PROJECT_ROOT / "metadata" / "phase2" / "download_qc.json"

EXPECTED = {
    "ENCFF148JKK.bed.gz": {
        "role": "positive_peaks",
        "md5": "d5183b4c65853c9dea2b299fd17f2562",
        "bytes": 262784,
        "rows": 14684,
        "min_columns": 10,
        "invalid_summits": 3,
    },
    "ENCFF875JHB.bed.gz": {
        "role": "positive_peaks_conservative_sensitivity",
        "md5": "8e03c2d46eded620a080b6b726df9c6e",
        "bytes": 197526,
        "rows": 10936,
        "min_columns": 10,
        "invalid_summits": 3,
    },
    "ENCFF333TAT.bed.gz": {
        "role": "accessible_negative_pool",
        "md5": "0f7a6c13e23c2e3fc8716153a89ed481",
        "bytes": 7067871,
        "rows": 269800,
        "min_columns": 10,
        "invalid_summits": 0,
    },
    "ENCFF769AUF.bed.gz": {
        "role": "ctcf_positive_control",
        "md5": "7d086cac19c5311a77b7e21e3d931435",
        "bytes": 919491,
        "rows": 51759,
        "min_columns": 10,
        "invalid_summits": 28,
    },
    "ENCFF356LFX.bed.gz": {
        "role": "blacklist",
        "md5": "393688b4f06c9ce26165d47433dd8c37",
        "bytes": 8211,
        "rows": 910,
        "min_columns": 3,
        "invalid_summits": None,
    },
}


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_bed(path: Path, expected: dict[str, object]) -> dict[str, object]:
    chromosome_counts: Counter[str] = Counter()
    rows = 0
    invalid_rows = 0
    invalid_summit_rows = 0
    minimum_columns: int | None = None
    maximum_columns = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.rstrip().split("\t")
            rows += 1
            minimum_columns = len(fields) if minimum_columns is None else min(minimum_columns, len(fields))
            maximum_columns = max(maximum_columns, len(fields))
            try:
                start, end = int(fields[1]), int(fields[2])
                valid = len(fields) >= int(expected["min_columns"]) and start >= 0 and end > start
            except (IndexError, ValueError):
                valid = False
            if valid:
                chromosome_counts[fields[0]] += 1
                if expected["invalid_summits"] is not None:
                    summit_offset = int(fields[9])
                    if summit_offset < 0 or start + summit_offset >= end:
                        invalid_summit_rows += 1
            else:
                invalid_rows += 1

    observed_md5 = md5_file(path)
    observed_bytes = path.stat().st_size
    checks = {
        "md5_match": observed_md5 == expected["md5"],
        "size_match": observed_bytes == expected["bytes"],
        "row_count_match": rows == expected["rows"],
        "all_coordinates_valid": invalid_rows == 0,
        "minimum_columns_met": (minimum_columns or 0) >= int(expected["min_columns"]),
    }
    if expected["invalid_summits"] is not None:
        checks["invalid_summit_count_match"] = invalid_summit_rows == expected["invalid_summits"]
    return {
        "role": expected["role"],
        "path": str(path.relative_to(PROJECT_ROOT)),
        "observed_md5": observed_md5,
        "observed_bytes": observed_bytes,
        "observed_rows": rows,
        "invalid_rows": invalid_rows,
        "invalid_summit_rows": invalid_summit_rows,
        "invalid_summit_policy": "exclude_from_summit_centered_window_extraction",
        "minimum_columns": minimum_columns,
        "maximum_columns": maximum_columns,
        "chromosome_counts": dict(sorted(chromosome_counts.items())),
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    results: dict[str, object] = {}
    missing: list[str] = []
    for name, expected in EXPECTED.items():
        path = RAW_DIR / name
        if not path.exists():
            missing.append(name)
            continue
        results[name] = inspect_bed(path, expected)

    overall_passed = not missing and all(item["passed"] for item in results.values())
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_passed": overall_passed,
        "missing_files": missing,
        "files": results,
        "note": "BED download/QC gate only; reference FASTA, sequence extraction, matching, split and leakage gates remain pending.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"output={OUTPUT}")
    print(f"overall_passed={str(overall_passed).lower()}")
    for name, result in results.items():
        print(f"{name}: rows={result['observed_rows']} passed={str(result['passed']).lower()}")
    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
