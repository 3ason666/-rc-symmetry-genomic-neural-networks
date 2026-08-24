from __future__ import annotations

import csv
import gzip
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2_data import (
    DNA,
    PRIMARY_CHROMOSOMES,
    IntervalIndex,
    maximum_overlapping_signals,
    parse_bed3,
    parse_narrowpeak,
    sequence_features,
    window_from_summit,
)


RAW = PROJECT_ROOT / "data" / "phase2" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "phase2" / "processed"
CACHE = PROJECT_ROOT / "data" / "phase2" / "sequence_cache" / "ensembl_grch38"
QC_PATH = PROJECT_ROOT / "metadata" / "phase2" / "candidate_build_qc.json"

ENSEMBL_URL = "https://rest.ensembl.org/sequence/region/human"
BATCH_SIZE = 50
WORKERS = 64
RETRIES = 8
WINDOW_BP = 256


@dataclass(frozen=True)
class Candidate:
    sample_id: str
    label: int
    peak_set: str
    source_accession: str
    chromosome: str
    start: int
    end: int
    peak_start: int
    peak_end: int
    summit: int
    source_score: float
    source_signal: float
    atac_overlap_signal: float | None
    eligible_primary: bool

    @property
    def query(self) -> str:
        chromosome = self.chromosome.removeprefix("chr")
        return f"{chromosome}:{self.start + 1}..{self.end}:1"


def batched(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def count_nonempty_rows(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip() and not line.startswith(("#", "track", "browser")))


def coordinate_candidates() -> tuple[list[Candidate], dict[str, object]]:
    positive_sources = [
        ("optimal", "ENCFF148JKK", RAW / "ENCFF148JKK.bed.gz"),
        ("conservative", "ENCFF875JHB", RAW / "ENCFF875JHB.bed.gz"),
    ]
    optimal = parse_narrowpeak(
        positive_sources[0][2], positive_sources[0][1], invalid_summit_policy="skip"
    )
    all_optimal_intervals = parse_bed3(positive_sources[0][2])
    atac = parse_narrowpeak(RAW / "ENCFF333TAT.bed.gz", "ENCFF333TAT")
    blacklist = parse_bed3(RAW / "ENCFF356LFX.bed.gz")
    gata_exclusion = {
        chromosome: IntervalIndex((max(0, start - 128), end + 128) for start, end in intervals)
        for chromosome, intervals in all_optimal_intervals.items()
    }

    candidates: list[Candidate] = []
    qc: dict[str, object] = {"coordinate_filters": {}, "atac_overlap_by_peak_set": {}}
    for peak_set, accession, path in positive_sources:
        peaks_by_chromosome = parse_narrowpeak(path, accession, invalid_summit_policy="skip")
        total_rows = count_nonempty_rows(path)
        parsed_rows = sum(len(values) for values in peaks_by_chromosome.values())
        counters: Counter[str] = Counter(
            input_rows=total_rows,
            invalid_summit=total_rows - parsed_rows,
        )
        accessible = 0
        retained = 0
        for chromosome in PRIMARY_CHROMOSOMES:
            blacklist_index = IntervalIndex(blacklist.get(chromosome, []))
            coordinate_rows = []
            retained_peaks = []
            for peak in peaks_by_chromosome.get(chromosome, []):
                start, end = window_from_summit(peak.summit, WINDOW_BP)
                if start < 0:
                    counters["window_before_chromosome"] += 1
                    continue
                if blacklist_index.overlaps(start, end):
                    counters["blacklist_overlap"] += 1
                    continue
                coordinate_rows.append((peak.sample_id, start, end))
                retained_peaks.append((peak, start, end))
            signals = maximum_overlapping_signals(atac.get(chromosome, []), coordinate_rows)
            for peak, start, end in retained_peaks:
                atac_signal = signals[peak.sample_id]
                is_accessible = atac_signal is not None
                accessible += int(is_accessible)
                retained += 1
                candidates.append(
                    Candidate(
                        sample_id=peak.sample_id,
                        label=1,
                        peak_set=peak_set,
                        source_accession=accession,
                        chromosome=chromosome,
                        start=start,
                        end=end,
                        peak_start=peak.start,
                        peak_end=peak.end,
                        summit=peak.summit,
                        source_score=peak.score,
                        source_signal=peak.signal,
                        atac_overlap_signal=atac_signal,
                        eligible_primary=is_accessible,
                    )
                )
        counters["non_primary_chromosome"] = parsed_rows - sum(
            len(peaks_by_chromosome.get(chromosome, [])) for chromosome in PRIMARY_CHROMOSOMES
        )
        counters["coordinate_retained"] = retained
        qc["coordinate_filters"][accession] = dict(counters)
        qc["atac_overlap_by_peak_set"][peak_set] = {
            "retained_coordinate_windows": retained,
            "atac_overlapping_windows": accessible,
            "fraction": accessible / retained if retained else None,
        }

    negative_path = RAW / "ENCFF333TAT.bed.gz"
    total_rows = count_nonempty_rows(negative_path)
    counters = Counter(input_rows=total_rows)
    retained = 0
    for chromosome in PRIMARY_CHROMOSOMES:
        blacklist_index = IntervalIndex(blacklist.get(chromosome, []))
        gata_index = gata_exclusion.get(chromosome, IntervalIndex())
        for peak in atac.get(chromosome, []):
            start, end = window_from_summit(peak.summit, WINDOW_BP)
            if start < 0:
                counters["window_before_chromosome"] += 1
                continue
            if blacklist_index.overlaps(start, end):
                counters["blacklist_overlap"] += 1
                continue
            if gata_index.overlaps(start, end):
                counters["gata1_or_buffer_overlap"] += 1
                continue
            retained += 1
            candidates.append(
                Candidate(
                    sample_id=peak.sample_id,
                    label=0,
                    peak_set="atac_negative_pool",
                    source_accession="ENCFF333TAT",
                    chromosome=chromosome,
                    start=start,
                    end=end,
                    peak_start=peak.start,
                    peak_end=peak.end,
                    summit=peak.summit,
                    source_score=peak.score,
                    source_signal=peak.signal,
                    atac_overlap_signal=peak.signal,
                    eligible_primary=True,
                )
            )
    primary_atac_rows = sum(len(atac.get(chromosome, [])) for chromosome in PRIMARY_CHROMOSOMES)
    counters["non_primary_chromosome"] = total_rows - primary_atac_rows
    counters["coordinate_retained"] = retained
    qc["coordinate_filters"]["ENCFF333TAT"] = dict(counters)
    candidates.sort(
        key=lambda row: (
            int(row.chromosome.removeprefix("chr")),
            row.start,
            row.end,
            row.label,
            row.peak_set,
            row.sample_id,
        )
    )
    return candidates, qc


def fetch_batch(index: int, regions: list[str]) -> Path:
    cache_path = CACHE / f"batch_{index:05d}.json.gz"
    if cache_path.exists():
        with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
            cached = json.load(handle)
        if sorted(cached) == sorted(regions) and all(len(value) == WINDOW_BP for value in cached.values()):
            return cache_path

    payload = json.dumps({"regions": regions}).encode("utf-8")
    for attempt in range(1, RETRIES + 1):
        try:
            request = urllib.request.Request(
                ENSEMBL_URL,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "RC-Attribution-Phase2/0.2",
                },
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                decoded = json.load(response)
            observed = {item["query"]: item["seq"].upper() for item in decoded}
            if sorted(observed) != sorted(regions):
                raise ValueError("Ensembl response queries do not match the request")
            if any(len(sequence) != WINDOW_BP for sequence in observed.values()):
                raise ValueError("Ensembl returned a non-256-bp sequence")
            temporary = cache_path.with_suffix(".json.gz.part")
            with gzip.open(temporary, "wt", encoding="utf-8") as handle:
                json.dump(observed, handle, sort_keys=True)
            temporary.replace(cache_path)
            return cache_path
        except (OSError, ValueError, urllib.error.HTTPError) as error:
            if attempt == RETRIES:
                raise
            retry_after = 0
            if isinstance(error, urllib.error.HTTPError):
                retry_after = int(error.headers.get("Retry-After", "0") or 0)
            time.sleep(max(retry_after, min(30, 2**attempt)))
    raise RuntimeError("unreachable")


def retrieve_sequences(candidates: list[Candidate]) -> dict[str, str]:
    unique_regions = sorted(
        {candidate.query for candidate in candidates},
        key=lambda query: (
            int(query.split(":", 1)[0]),
            int(query.split(":", 1)[1].split("..", 1)[0]),
        ),
    )
    batches = batched(unique_regions, BATCH_SIZE)
    CACHE.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    completed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(fetch_batch, index, regions): index
            for index, regions in enumerate(batches)
        }
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 25 == 0 or completed == len(batches):
                elapsed = time.monotonic() - started
                print(
                    f"sequence_batches={completed}/{len(batches)} "
                    f"elapsed_minutes={elapsed / 60:.1f}",
                    flush=True,
                )

    sequences: dict[str, str] = {}
    for index, regions in enumerate(batches):
        with gzip.open(CACHE / f"batch_{index:05d}.json.gz", "rt", encoding="utf-8") as handle:
            observed = json.load(handle)
        if sorted(observed) != sorted(regions):
            raise ValueError(f"cached batch {index} no longer matches its frozen regions")
        sequences.update(observed)
    return sequences


def write_candidates(
    candidates: list[Candidate],
    sequences: dict[str, str],
    qc: dict[str, object],
) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "label",
        "peak_set",
        "source_accession",
        "chromosome",
        "start",
        "end",
        "peak_start",
        "peak_end",
        "summit",
        "source_score",
        "source_signal",
        "atac_overlap_signal",
        "gc_fraction",
        "gc_bin_0.02",
        "sequence",
        "canonical_key",
        "eligible_primary",
    ]
    outputs = {
        "positive": PROCESSED / "positive_candidates.csv.gz",
        "negative": PROCESSED / "negative_candidates.csv.gz",
    }
    handles = {
        name: gzip.open(path, "wt", encoding="utf-8", newline="")
        for name, path in outputs.items()
    }
    writers = {name: csv.DictWriter(handle, fieldnames=fields) for name, handle in handles.items()}
    for writer in writers.values():
        writer.writeheader()

    sequence_qc: dict[str, Counter[str]] = {
        "optimal": Counter(),
        "conservative": Counter(),
        "atac_negative_pool": Counter(),
    }
    try:
        for candidate in candidates:
            sequence = sequences[candidate.query]
            counter = sequence_qc[candidate.peak_set]
            counter["retrieved"] += 1
            invalid = set(sequence).difference(DNA)
            if invalid:
                counter["contains_non_acgt"] += 1
                continue
            gc_fraction, gc_bin, canonical_key = sequence_features(sequence)
            counter["retained"] += 1
            if candidate.eligible_primary:
                counter["eligible_primary"] += 1
            row = {
                **candidate.__dict__,
                "atac_overlap_signal": ""
                if candidate.atac_overlap_signal is None
                else candidate.atac_overlap_signal,
                "gc_fraction": f"{gc_fraction:.8f}",
                "gc_bin_0.02": gc_bin,
                "sequence": sequence,
                "canonical_key": canonical_key,
                "eligible_primary": int(candidate.eligible_primary),
            }
            writers["positive" if candidate.label == 1 else "negative"].writerow(row)
    finally:
        for handle in handles.values():
            handle.close()

    qc.update(
        {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "sequence_source": {
                "provider": "Ensembl REST",
                "endpoint": ENSEMBL_URL,
                "assembly_name_at_retrieval": "GRCh38.p14",
                "assembly_accession_at_retrieval": "GCA_000001405.29",
                "default_coordinate_system_version": "GRCh38",
                "window_bp": WINDOW_BP,
                "status": "provisional_until_ENCODE_FASTA_cross_validation",
            },
            "sequence_filters": {name: dict(counter) for name, counter in sequence_qc.items()},
            "outputs": {name: str(path.relative_to(PROJECT_ROOT)) for name, path in outputs.items()},
            "formal_training_allowed": False,
        }
    )
    QC_PATH.parent.mkdir(parents=True, exist_ok=True)
    QC_PATH.write_text(json.dumps(qc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"qc={QC_PATH}")


def main() -> int:
    candidates, qc = coordinate_candidates()
    print(f"coordinate_candidates={len(candidates)}", flush=True)
    sequences = retrieve_sequences(candidates)
    print(f"unique_sequences_retrieved={len(sequences)}", flush=True)
    write_candidates(candidates, sequences, qc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
