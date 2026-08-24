from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "phase2" / "raw"
SOURCE = RAW / "GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz"
TARGET = RAW / "GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
INDEX = TARGET.with_suffix(TARGET.suffix + ".fai")
QC = PROJECT_ROOT / "results" / "phase3a_external_selection" / "reference_qc.json"
EXPECTED_GZIP_MD5 = "a08035b6a6e31780e96a34008ff21bd6"
EXPECTED_GZIP_BYTES = 872_949_833


def digest(path: Path, algorithm: str) -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def decompress_and_index() -> list[dict[str, int | str]]:
    temporary_fasta = TARGET.with_suffix(TARGET.suffix + ".part")
    temporary_index = INDEX.with_suffix(INDEX.suffix + ".part")
    records: list[dict[str, int | str]] = []
    offset = 0
    current: dict[str, int | str] | None = None

    with gzip.open(SOURCE, "rb") as source, temporary_fasta.open("wb") as target:
        for line in source:
            if line.startswith(b">"):
                if current is not None:
                    records.append(current)
                name = line[1:].split(None, 1)[0].decode("ascii")
                current = {
                    "name": name,
                    "length": 0,
                    "offset": offset + len(line),
                    "line_bases": 0,
                    "line_width": 0,
                }
            else:
                if current is None:
                    raise ValueError("FASTA sequence appeared before the first header")
                bases = line.rstrip(b"\r\n")
                if not bases:
                    continue
                if int(current["line_bases"]) == 0:
                    current["line_bases"] = len(bases)
                    current["line_width"] = len(line)
                current["length"] = int(current["length"]) + len(bases)
            target.write(line)
            offset += len(line)
        if current is not None:
            records.append(current)

    if not records:
        raise ValueError("reference FASTA contains no records")
    with temporary_index.open("w", encoding="ascii", newline="\n") as handle:
        for row in records:
            handle.write(
                f'{row["name"]}\t{row["length"]}\t{row["offset"]}\t'
                f'{row["line_bases"]}\t{row["line_width"]}\n'
            )
    temporary_fasta.replace(TARGET)
    temporary_index.replace(INDEX)
    return records


def read_index() -> list[dict[str, int | str]]:
    records = []
    with INDEX.open("r", encoding="ascii") as handle:
        for line in handle:
            name, length, offset, line_bases, line_width = line.rstrip().split("\t")
            records.append(
                {
                    "name": name,
                    "length": int(length),
                    "offset": int(offset),
                    "line_bases": int(line_bases),
                    "line_width": int(line_width),
                }
            )
    return records


def main() -> int:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    source_size = SOURCE.stat().st_size
    source_md5 = digest(SOURCE, "md5")
    if source_size != EXPECTED_GZIP_BYTES or source_md5 != EXPECTED_GZIP_MD5:
        raise ValueError(
            f"reference gzip mismatch: bytes={source_size}, md5={source_md5}"
        )

    reused = TARGET.exists() and INDEX.exists()
    records = read_index() if reused else decompress_and_index()
    primary = {str(row["name"]): int(row["length"]) for row in records}
    missing_primary = [f"chr{i}" for i in range(1, 23) if f"chr{i}" not in primary]
    invalid_index = [
        str(row["name"])
        for row in records
        if int(row["length"]) <= 0
        or int(row["offset"]) < 0
        or int(row["line_bases"]) <= 0
        or int(row["line_width"]) < int(row["line_bases"])
    ]
    passed = not missing_primary and not invalid_index
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "source": str(SOURCE.relative_to(PROJECT_ROOT)),
        "source_bytes": source_size,
        "source_md5": source_md5,
        "fasta": str(TARGET.relative_to(PROJECT_ROOT)),
        "fasta_bytes": TARGET.stat().st_size,
        "fasta_sha256": digest(TARGET, "sha256"),
        "index": str(INDEX.relative_to(PROJECT_ROOT)),
        "index_sha256": digest(INDEX, "sha256"),
        "record_count": len(records),
        "primary_chromosome_lengths": {
            f"chr{i}": primary.get(f"chr{i}") for i in range(1, 23)
        },
        "missing_primary_chromosomes": missing_primary,
        "invalid_index_records": invalid_index,
        "reused_existing_fasta_and_index": reused,
        "formal_training_allowed": False,
    }
    QC.parent.mkdir(parents=True, exist_ok=True)
    QC.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"status={payload['status']}")
    print(f"records={len(records)} fasta_bytes={payload['fasta_bytes']}")
    print(f"qc={QC}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
