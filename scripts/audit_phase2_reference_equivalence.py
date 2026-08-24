from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

METADATA = PROJECT_ROOT / "metadata" / "phase2"
POSITIVE_PATH = PROJECT_ROOT / "data" / "phase2" / "processed" / "positive_candidates.csv.gz"
OUTPUT = METADATA / "reference_equivalence_audit.json"
ENSEMBL_INFO_URL = "https://rest.ensembl.org/info/assembly/homo_sapiens"
REPORT_URLS = {
    "GCA_000001405.15": (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/"
        "GCA_000001405.15_GRCh38/GCA_000001405.15_GRCh38_assembly_report.txt"
    ),
    "GCA_000001405.29": (
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/"
        "GCA_000001405.29_GRCh38.p14/GCA_000001405.29_GRCh38.p14_assembly_report.txt"
    ),
}


def fetch_text(url: str, *, accept: str = "text/plain", retries: int = 6) -> str:
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"Accept": accept, "User-Agent": "RC-Attribution-Phase2/0.2"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read().decode("utf-8")
        except OSError:
            if attempt == retries:
                raise
            time.sleep(min(30, 2**attempt))
    raise RuntimeError("unreachable")


def parse_primary_chromosomes(text: str) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 10 or fields[1] != "assembled-molecule":
            continue
        chromosome = fields[2]
        if chromosome not in {str(value) for value in range(1, 23)}:
            continue
        records[f"chr{chromosome}"] = {
            "genbank_accession": fields[4],
            "refseq_accession": fields[6],
            "length": int(fields[8]),
            "ucsc_name": fields[9],
        }
    return records


def ncbi_sequence(accession: str, start: int, end: int) -> str:
    parameters = urllib.parse.urlencode(
        {
            "db": "nuccore",
            "id": accession,
            "seq_start": start + 1,
            "seq_stop": end,
            "rettype": "fasta",
            "retmode": "text",
        }
    )
    text = fetch_text(
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{parameters}"
    )
    sequence = "".join(line.strip() for line in text.splitlines() if not line.startswith(">"))
    return sequence.upper()


def main() -> int:
    if not POSITIVE_PATH.exists():
        raise FileNotFoundError("positive candidate table is not available yet")
    METADATA.mkdir(parents=True, exist_ok=True)

    reports: dict[str, str] = {}
    parsed: dict[str, dict[str, dict[str, object]]] = {}
    report_sha256: dict[str, str] = {}
    for accession, url in REPORT_URLS.items():
        text = fetch_text(url)
        reports[accession] = text
        parsed[accession] = parse_primary_chromosomes(text)
        report_sha256[accession] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        snapshot = METADATA / f"{accession}_assembly_report.txt"
        snapshot.write_text(text, encoding="utf-8")

    frozen = parsed["GCA_000001405.15"]
    current = parsed["GCA_000001405.29"]
    component_identity = frozen == current and len(frozen) == 22

    ensembl_info = json.loads(fetch_text(ENSEMBL_INFO_URL, accept="application/json"))
    ensembl_lengths = {
        f"chr{item['name']}": int(item["length"])
        for item in ensembl_info["top_level_region"]
        if item.get("coord_system") == "chromosome" and item.get("name") in {str(v) for v in range(1, 23)}
    }
    length_identity = all(ensembl_lengths.get(chromosome) == record["length"] for chromosome, record in current.items())

    frame = pd.read_csv(POSITIVE_PATH)
    frame = frame[(frame["peak_set"] == "optimal") & (frame["eligible_primary"] == 1)]
    spot_checks = []
    for chromosome in sorted(frozen, key=lambda value: int(value.removeprefix("chr"))):
        row = frame[frame["chromosome"] == chromosome].sort_values(["start", "sample_id"]).iloc[0]
        observed = ncbi_sequence(
            str(frozen[chromosome]["refseq_accession"]), int(row["start"]), int(row["end"])
        )
        expected = str(row["sequence"]).upper()
        spot_checks.append(
            {
                "chromosome": chromosome,
                "sample_id": row["sample_id"],
                "start": int(row["start"]),
                "end": int(row["end"]),
                "refseq_accession": frozen[chromosome]["refseq_accession"],
                "sequence_match": observed == expected,
                "length": len(observed),
            }
        )
        time.sleep(0.36)

    passed = (
        component_identity
        and ensembl_info.get("assembly_accession") == "GCA_000001405.29"
        and ensembl_info.get("default_coord_system_version") == "GRCh38"
        and length_identity
        and all(check["sequence_match"] and check["length"] == 256 for check in spot_checks)
    )
    audit = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "method": (
            "Compare chr1-chr22 component accession versions and lengths between the frozen "
            "GCA_000001405.15 assembly and Ensembl's GCA_000001405.29 assembly, then compare "
            "one deterministic 256-bp candidate per chromosome against the corresponding "
            "NCBI RefSeq chromosome accession."
        ),
        "ncbi_assembly_report_urls": REPORT_URLS,
        "ncbi_assembly_report_sha256": report_sha256,
        "primary_component_identity": component_identity,
        "primary_components": frozen,
        "ensembl_assembly": {
            "assembly_name": ensembl_info.get("assembly_name"),
            "assembly_accession": ensembl_info.get("assembly_accession"),
            "default_coord_system_version": ensembl_info.get("default_coord_system_version"),
            "primary_chromosome_lengths_match": length_identity,
        },
        "coordinate_and_sequence_spot_checks": spot_checks,
        "conclusion": (
            "The Ensembl-retrieved chr1-chr22 windows are reference-equivalent for this study."
            if passed
            else "Reference equivalence was not established; formal training remains blocked."
        ),
    }
    OUTPUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"output={OUTPUT}")
    print(f"passed={str(passed).lower()}")
    print(f"spot_checks={len(spot_checks)}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
