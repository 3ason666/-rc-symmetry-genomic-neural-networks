from __future__ import annotations

import bisect
import csv
import gzip
import hashlib
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2_data import (
    DNA,
    PRIMARY_CHROMOSOMES,
    IntervalIndex,
    parse_bed3,
    parse_narrowpeak,
    sequence_features,
    window_from_summit,
)
from src.phase2_matching import assign_quantile, quantile_edges, stable_rank


RAW = PROJECT_ROOT / "data" / "phase3" / "raw"
REFERENCE = PROJECT_ROOT / "data" / "phase2" / "raw" / "GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta"
BLACKLIST = PROJECT_ROOT / "data" / "phase2" / "raw" / "ENCFF356LFX.bed.gz"
PROCESSED = PROJECT_ROOT / "data" / "phase3" / "processed"
RESULTS = PROJECT_ROOT / "results" / "phase3a_external_selection"
WINDOW_BP = 256
BUFFER_BP = 128
MATCHING_SEED = 20260823
GC_BIN_WIDTH = 0.02
ACCESSIBILITY_BINS = 10
ACCESSIBILITY_LOG2P1_CALIPER = 1.0


TASKS = {
    "p3_gata1_fetal": {
        "primary": "ENCFF802VPT",
        "conservative": "ENCFF305RPQ",
        "accessibility": ("ENCFF253NOJ", "ENCFF404PZM"),
        "accessibility_kind": "dnase_hotspot",
    },
    "p3_ctcf_gm12878": {
        "primary": "ENCFF138REW",
        "conservative": "ENCFF796WRU",
        "accessibility": ("ENCFF687HBT",),
        "accessibility_kind": "narrowPeak",
    },
}


@dataclass(frozen=True)
class AccessiblePeak:
    accession: str
    row_number: int
    chromosome: str
    start: int
    end: int
    summit: int
    signal: float

    @property
    def sample_id(self) -> str:
        return f"{self.accession}:{self.row_number}"


class FastaIndex:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = path.open("rb")
        self.records: dict[str, tuple[int, int, int, int]] = {}
        with Path(str(path) + ".fai").open("r", encoding="ascii") as handle:
            for line in handle:
                name, length, offset, line_bases, line_width = line.rstrip().split("\t")
                self.records[name] = (
                    int(length), int(offset), int(line_bases), int(line_width)
                )

    def close(self) -> None:
        self.handle.close()

    def fetch(self, chromosome: str, start: int, end: int) -> str:
        length, offset, line_bases, line_width = self.records[chromosome]
        if start < 0 or end > length or end <= start:
            raise ValueError(f"out-of-bounds FASTA query: {chromosome}:{start}-{end}")
        byte_start = offset + (start // line_bases) * line_width + (start % line_bases)
        final_base = end - 1
        byte_end = offset + (final_base // line_bases) * line_width + (final_base % line_bases) + 1
        self.handle.seek(byte_start)
        sequence = self.handle.read(byte_end - byte_start).replace(b"\n", b"").replace(b"\r", b"")
        sequence = sequence[: end - start].decode("ascii").upper()
        if len(sequence) != end - start:
            raise ValueError(f"short FASTA query: {chromosome}:{start}-{end}")
        return sequence


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def parse_accessibility(accessions: tuple[str, ...], kind: str) -> dict[str, list[AccessiblePeak]]:
    result: dict[str, list[AccessiblePeak]] = defaultdict(list)
    for accession in accessions:
        path = RAW / f"{accession}.bed.gz"
        if kind == "narrowPeak":
            parsed = parse_narrowpeak(path, accession, invalid_summit_policy="skip")
            for chromosome, peaks in parsed.items():
                for peak in peaks:
                    result[chromosome].append(
                        AccessiblePeak(
                            accession,
                            peak.row_number,
                            chromosome,
                            peak.start,
                            peak.end,
                            peak.summit,
                            peak.signal,
                        )
                    )
        else:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for row_number, line in enumerate(handle, start=1):
                    if not line.strip() or line.startswith(("#", "track", "browser")):
                        continue
                    fields = line.rstrip().split("\t")
                    if len(fields) < 8:
                        raise ValueError(f"{path.name}:{row_number}: expected 8-column Hotspot BED")
                    chromosome = fields[0]
                    start, end, summit = int(fields[1]), int(fields[2]), int(fields[6])
                    if start < 0 or end <= start or summit < start or summit >= end:
                        continue
                    result[chromosome].append(
                        AccessiblePeak(
                            accession,
                            row_number,
                            chromosome,
                            start,
                            end,
                            summit,
                            float(fields[4]),
                        )
                    )
    for peaks in result.values():
        peaks.sort(key=lambda row: (row.start, row.end, row.row_number, row.accession))
    return dict(result)


def max_overlap_signals(
    peaks: list[AccessiblePeak], queries: list[tuple[str, int, int]]
) -> dict[str, float | None]:
    ordered_queries = sorted(queries, key=lambda row: (row[1], row[2], row[0]))
    active: list[AccessiblePeak] = []
    peak_index = 0
    result: dict[str, float | None] = {}
    for query_id, start, end in ordered_queries:
        active = [peak for peak in active if peak.end > start]
        while peak_index < len(peaks) and peaks[peak_index].start < end:
            if peaks[peak_index].end > start:
                active.append(peaks[peak_index])
            peak_index += 1
        values = [peak.signal for peak in active if peak.start < end and peak.end > start]
        result[query_id] = max(values) if values else None
    return result


def keep_nonoverlapping_by_signal(rows: list[dict]) -> tuple[list[dict], int]:
    occupied: dict[str, list[tuple[int, int]]] = defaultdict(list)
    kept: list[dict] = []
    rejected = 0
    for row in sorted(
        rows,
        key=lambda item: (
            -float(item["source_signal"]),
            str(item["chromosome"]),
            int(item["start"]),
            str(item["sample_id"]),
        ),
    ):
        chromosome = str(row["chromosome"])
        start, end = int(row["start"]), int(row["end"])
        starts = [prior[0] for prior in occupied[chromosome]]
        index = bisect.bisect_left(starts, start)
        neighbors = occupied[chromosome][max(0, index - 1) : index + 1]
        if any(left < end and start < right for left, right in neighbors):
            rejected += 1
            continue
        occupied[chromosome].insert(index, (start, end))
        kept.append(row)
    kept.sort(key=lambda row: (int(str(row["chromosome"])[3:]), int(row["start"])))
    return kept, rejected


def candidate_rows(task_id: str, spec: dict, fasta: FastaIndex) -> tuple[list[dict], list[dict], dict]:
    primary = parse_narrowpeak(RAW / f'{spec["primary"]}.bed.gz', spec["primary"], invalid_summit_policy="skip")
    all_target_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for accession in (spec["primary"], spec["conservative"]):
        for chromosome, intervals in parse_bed3(RAW / f"{accession}.bed.gz").items():
            all_target_intervals[chromosome].extend(intervals)
    exclusions = {
        chromosome: IntervalIndex((max(0, start - BUFFER_BP), end + BUFFER_BP) for start, end in intervals)
        for chromosome, intervals in all_target_intervals.items()
    }
    blacklist = parse_bed3(BLACKLIST)
    accessibility = parse_accessibility(spec["accessibility"], spec["accessibility_kind"])
    chrom_lengths = {name: values[0] for name, values in fasta.records.items()}
    qc: dict[str, object] = {"positive_filters": Counter(), "negative_filters": Counter()}

    coordinate_positives: list[dict] = []
    for chromosome in PRIMARY_CHROMOSOMES:
        blacklist_index = IntervalIndex(blacklist.get(chromosome, []))
        queries: list[tuple[str, int, int]] = []
        staged: list[tuple[object, int, int]] = []
        for peak in primary.get(chromosome, []):
            qc["positive_filters"]["parsed_primary"] += 1
            start, end = window_from_summit(peak.summit, WINDOW_BP)
            if start < 0 or end > chrom_lengths[chromosome]:
                qc["positive_filters"]["out_of_bounds"] += 1
                continue
            if blacklist_index.overlaps(start, end):
                qc["positive_filters"]["blacklist_overlap"] += 1
                continue
            queries.append((peak.sample_id, start, end))
            staged.append((peak, start, end))
        signals = max_overlap_signals(accessibility.get(chromosome, []), queries)
        for peak, start, end in staged:
            access_signal = signals[peak.sample_id]
            if access_signal is None:
                qc["positive_filters"]["no_accessibility_overlap"] += 1
                continue
            sequence = fasta.fetch(chromosome, start, end)
            if set(sequence).difference(DNA):
                qc["positive_filters"]["non_acgt"] += 1
                continue
            gc, gc_bin, canonical = sequence_features(sequence)
            coordinate_positives.append(
                {
                    "task_id": task_id,
                    "sample_id": peak.sample_id,
                    "label": 1,
                    "chromosome": chromosome,
                    "start": start,
                    "end": end,
                    "summit": peak.summit,
                    "source_accession": peak.accession,
                    "source_signal": peak.signal,
                    "accessibility_signal": float(access_signal),
                    "gc_fraction": gc,
                    "gc_bin_0.02": gc_bin,
                    "sequence": sequence,
                    "canonical_key": canonical,
                }
            )
    positives, overlap_rejected = keep_nonoverlapping_by_signal(coordinate_positives)
    qc["positive_filters"]["overlapping_positive_window"] = overlap_rejected
    qc["positive_filters"]["retained"] = len(positives)

    negatives: list[dict] = []
    for chromosome in PRIMARY_CHROMOSOMES:
        blacklist_index = IntervalIndex(blacklist.get(chromosome, []))
        target_index = exclusions.get(chromosome, IntervalIndex())
        for peak in accessibility.get(chromosome, []):
            qc["negative_filters"]["parsed_accessibility"] += 1
            start, end = window_from_summit(peak.summit, WINDOW_BP)
            if start < 0 or end > chrom_lengths[chromosome]:
                qc["negative_filters"]["out_of_bounds"] += 1
                continue
            if blacklist_index.overlaps(start, end):
                qc["negative_filters"]["blacklist_overlap"] += 1
                continue
            if target_index.overlaps(start, end):
                qc["negative_filters"]["target_or_128bp_buffer_overlap"] += 1
                continue
            sequence = fasta.fetch(chromosome, start, end)
            if set(sequence).difference(DNA):
                qc["negative_filters"]["non_acgt"] += 1
                continue
            gc, gc_bin, canonical = sequence_features(sequence)
            negatives.append(
                {
                    "task_id": task_id,
                    "sample_id": peak.sample_id,
                    "label": 0,
                    "chromosome": chromosome,
                    "start": start,
                    "end": end,
                    "summit": peak.summit,
                    "source_accession": peak.accession,
                    "source_signal": peak.signal,
                    "accessibility_signal": peak.signal,
                    "gc_fraction": gc,
                    "gc_bin_0.02": gc_bin,
                    "sequence": sequence,
                    "canonical_key": canonical,
                }
            )
    qc["negative_filters"]["retained"] = len(negatives)
    qc["positive_filters"] = dict(qc["positive_filters"])
    qc["negative_filters"] = dict(qc["negative_filters"])
    return positives, negatives, qc


def assign_accessibility_deciles(positives: list[dict], negatives: list[dict]) -> dict[str, list[float]]:
    by_chromosome: dict[str, list[float]] = defaultdict(list)
    for row in negatives:
        by_chromosome[str(row["chromosome"])].append(float(row["accessibility_signal"]))
    edges = {
        chromosome: quantile_edges(values, ACCESSIBILITY_BINS)
        for chromosome, values in by_chromosome.items()
    }
    for row in itertools.chain(positives, negatives):
        row["accessibility_decile"] = assign_quantile(
            float(row["accessibility_signal"]), edges[str(row["chromosome"])]
        )
    return {chromosome: values.tolist() for chromosome, values in edges.items()}


class FenwickTree:
    def __init__(self, size: int) -> None:
        self.size = size
        self.tree = [0] * (size + 1)
        for index in range(1, size + 1):
            self.tree[index] += 1
            parent = index + (index & -index)
            if parent <= size:
                self.tree[parent] += self.tree[index]

    def add(self, zero_based_index: int, delta: int) -> None:
        index = zero_based_index + 1
        while index <= self.size:
            self.tree[index] += delta
            index += index & -index

    def prefix(self, exclusive_end: int) -> int:
        total = 0
        index = exclusive_end
        while index > 0:
            total += self.tree[index]
            index -= index & -index
        return total

    def kth(self, rank: int) -> int:
        if rank < 1 or rank > self.prefix(self.size):
            raise IndexError(rank)
        index = 0
        bit = 1 << (self.size.bit_length() - 1)
        while bit:
            next_index = index + bit
            if next_index <= self.size and self.tree[next_index] < rank:
                index = next_index
                rank -= self.tree[next_index]
            bit >>= 1
        return index


def nearest_without_replacement(query_rows: list[dict], pool_rows: list[dict]) -> tuple[list[tuple[dict, dict]], list[dict]]:
    pool = sorted(
        pool_rows,
        key=lambda row: (float(row["accessibility_signal"]), str(row["sample_id"])),
    )
    values = [float(row["accessibility_signal"]) for row in pool]
    available = FenwickTree(len(pool))
    pairs: list[tuple[dict, dict]] = []
    unmatched: list[dict] = []
    for query in sorted(
        query_rows,
        key=lambda row: (float(row["accessibility_signal"]), str(row["sample_id"])),
    ):
        remaining = available.prefix(len(pool))
        if remaining == 0:
            unmatched.append(query)
            continue
        insertion = bisect.bisect_left(values, float(query["accessibility_signal"]))
        before = available.prefix(insertion)
        candidate_indices: list[int] = []
        if before:
            candidate_indices.append(available.kth(before))
        if before < remaining:
            candidate_indices.append(available.kth(before + 1))
        chosen = min(
            candidate_indices,
            key=lambda index: (
                abs(values[index] - float(query["accessibility_signal"])),
                str(pool[index]["sample_id"]),
            ),
        )
        available.add(chosen, -1)
        pairs.append((query, pool[chosen]))
    return pairs, unmatched


def remove_canonical_duplicates(positives: list[dict], negatives: list[dict]) -> tuple[list[dict], list[dict], dict[str, int]]:
    positive_by_key: dict[str, dict] = {}
    for row in sorted(
        positives,
        key=lambda item: (-float(item["source_signal"]), str(item["sample_id"])),
    ):
        positive_by_key.setdefault(str(row["canonical_key"]), row)
    positive_keys = set(positive_by_key)
    negative_by_key: dict[str, dict] = {}
    label_conflict_removed = 0
    for row in sorted(negatives, key=lambda item: str(item["sample_id"])):
        key = str(row["canonical_key"])
        if key in positive_keys:
            label_conflict_removed += 1
            continue
        negative_by_key.setdefault(key, row)
    return (
        list(positive_by_key.values()),
        list(negative_by_key.values()),
        {
            "positive_duplicate_rows_removed": len(positives) - len(positive_by_key),
            "negative_duplicate_rows_removed": len(negatives) - label_conflict_removed - len(negative_by_key),
            "negative_label_conflict_rows_removed": label_conflict_removed,
        },
    )


def match_rows(positives: list[dict], negatives: list[dict], seed: int) -> tuple[list[tuple[dict, dict]], list[dict]]:
    positive_groups: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    negative_groups: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in positives:
        positive_groups[(str(row["chromosome"]), int(row["gc_bin_0.02"]), int(row["accessibility_decile"]))].append(row)
    for row in negatives:
        negative_groups[(str(row["chromosome"]), int(row["gc_bin_0.02"]), int(row["accessibility_decile"]))].append(row)
    pairs: list[tuple[dict, dict]] = []
    unmatched: list[dict] = []
    for key in sorted(positive_groups):
        pos = positive_groups[key]
        neg = negative_groups.get(key, [])
        if not neg:
            unmatched.extend(pos)
            continue
        if len(neg) >= len(pos):
            group_pairs, group_unmatched = nearest_without_replacement(pos, neg)
            pairs.extend(group_pairs)
            unmatched.extend(group_unmatched)
        else:
            reverse_pairs, _ = nearest_without_replacement(neg, pos)
            used_positive_ids = {str(positive["sample_id"]) for _, positive in reverse_pairs}
            pairs.extend((positive, negative) for negative, positive in reverse_pairs)
            unmatched.extend(row for row in pos if str(row["sample_id"]) not in used_positive_ids)
    return pairs, unmatched


def choose_subset(counts: dict[str, int], target: float, minimum_chromosomes: int, max_fraction: float | None = None, excluded: set[str] | None = None) -> tuple[str, ...]:
    excluded = excluded or set()
    chromosomes = [chrom for chrom in PRIMARY_CHROMOSOMES if chrom in counts and chrom not in excluded]
    best: tuple[float, int, tuple[str, ...]] | None = None
    for size in range(minimum_chromosomes, min(8, len(chromosomes)) + 1):
        for subset in itertools.combinations(chromosomes, size):
            total = sum(counts[chrom] for chrom in subset)
            if max_fraction is not None and max(counts[chrom] for chrom in subset) / total > max_fraction:
                continue
            candidate = (abs(total - target), size, subset)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError("no chromosome subset satisfies the frozen constraints")
    return best[2]


def choose_balanced_subset(
    pair_rows: list[dict],
    available: list[str],
    target: float,
    minimum_chromosomes: int,
    complement_scope: set[str],
    max_fraction: float | None = None,
    forbidden_subsets: set[tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    forbidden_subsets = forbidden_subsets or set()
    counts = Counter(str(row["chromosome"]) for row in pair_rows)
    candidates: list[tuple[float, int, tuple[str, ...]]] = []
    for size in range(minimum_chromosomes, min(6, len(available)) + 1):
        for subset in itertools.combinations(available, size):
            total = sum(counts[chromosome] for chromosome in subset)
            if total < 200:
                continue
            if max_fraction is not None and max(counts[chromosome] for chromosome in subset) / total > max_fraction:
                continue
            candidates.append((abs(total - target), size, subset))
    candidates.sort()
    for _, _, subset in candidates:
        if subset in forbidden_subsets:
            continue
        selected = set(subset)
        complement = complement_scope.difference(selected)
        selected_rows = [row for row in pair_rows if str(row["chromosome"]) in selected]
        complement_rows = [row for row in pair_rows if str(row["chromosome"]) in complement]
        if not complement_rows:
            continue
        selected_balance = balance(selected_rows)
        complement_balance = balance(complement_rows)
        if (
            selected_balance["gc_passed"]
            and selected_balance["accessibility_passed"]
            and complement_balance["gc_passed"]
            and complement_balance["accessibility_passed"]
        ):
            return subset
    raise ValueError("no chromosome subset satisfies the frozen count and balance constraints")


def choose_splits(pair_rows: list[dict]) -> dict[str, object]:
    pair_counts = dict(Counter(str(row["chromosome"]) for row in pair_rows))
    total = sum(pair_counts.values())
    all_chromosomes = [chromosome for chromosome in PRIMARY_CHROMOSOMES if chromosome in pair_counts]
    test = choose_balanced_subset(
        pair_rows,
        all_chromosomes,
        total * 0.15,
        3,
        set(all_chromosomes),
        0.50,
    )
    validation_sets: list[tuple[str, ...]] = []
    development = tuple(chrom for chrom in PRIMARY_CHROMOSOMES if chrom in pair_counts and chrom not in test)
    for _ in range(3):
        validation = choose_balanced_subset(
            pair_rows,
            list(development),
            total * 0.15,
            1,
            set(development),
            forbidden_subsets=set(validation_sets),
        )
        validation_sets.append(validation)
    return {
        "fixed_test": test,
        "development_folds": [
            {
                "fold": index + 1,
                "validation": validation,
                "train": tuple(chrom for chrom in development if chrom not in validation),
            }
            for index, validation in enumerate(validation_sets)
        ],
    }


def build_output_rows(pairs: list[tuple[dict, dict]]) -> tuple[list[dict], list[dict]]:
    samples: list[dict] = []
    pair_rows: list[dict] = []
    ordered = sorted(pairs, key=lambda pair: (int(str(pair[0]["chromosome"])[3:]), int(pair[0]["start"]), str(pair[0]["sample_id"])))
    for index, (positive, negative) in enumerate(ordered, start=1):
        pair_id = f"pair_{index:06d}"
        pair_rows.append(
            {
                "pair_id": pair_id,
                "chromosome": positive["chromosome"],
                "positive_sample_id": positive["sample_id"],
                "negative_sample_id": negative["sample_id"],
                "gc_bin_0.02": positive["gc_bin_0.02"],
                "accessibility_decile": positive["accessibility_decile"],
                "positive_gc": positive["gc_fraction"],
                "negative_gc": negative["gc_fraction"],
                "positive_accessibility": positive["accessibility_signal"],
                "negative_accessibility": negative["accessibility_signal"],
            }
        )
        for source in (positive, negative):
            row = dict(source)
            row["pair_id"] = pair_id
            samples.append(row)
    return samples, pair_rows


def balance(pair_rows: list[dict]) -> dict:
    positive_gc = np.asarray([float(row["positive_gc"]) for row in pair_rows])
    negative_gc = np.asarray([float(row["negative_gc"]) for row in pair_rows])
    positive_access = np.log2(1 + np.asarray([float(row["positive_accessibility"]) for row in pair_rows]))
    negative_access = np.log2(1 + np.asarray([float(row["negative_accessibility"]) for row in pair_rows]))
    gc = ks_2samp(positive_gc, negative_gc)
    access = ks_2samp(positive_access, negative_access)
    return {
        "gc_ks_d": float(gc.statistic),
        "gc_ks_pvalue": float(gc.pvalue),
        "accessibility_log2p1_ks_d": float(access.statistic),
        "accessibility_log2p1_ks_pvalue": float(access.pvalue),
        "gc_passed": bool(gc.statistic <= 0.10),
        "accessibility_passed": bool(access.statistic <= 0.10),
    }


def write_csv_gz(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> int:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    fasta = FastaIndex(REFERENCE)
    overall: dict[str, object] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matching_seed": MATCHING_SEED,
        "tasks": {},
        "formal_training_allowed": False,
    }
    all_passed = True
    try:
        for task_index, (task_id, spec) in enumerate(TASKS.items()):
            print(f"task={task_id} stage=candidate_generation", flush=True)
            positives, negatives, candidate_qc = candidate_rows(task_id, spec, fasta)
            print(f"task={task_id} positives={len(positives)} negatives={len(negatives)}", flush=True)
            positives, negatives, dedup_qc = remove_canonical_duplicates(positives, negatives)
            edges = assign_accessibility_deciles(positives, negatives)
            pairs, unmatched = match_rows(positives, negatives, MATCHING_SEED + task_index)
            within_caliper = []
            caliper_rejected = []
            for positive, negative in pairs:
                distance = abs(
                    math.log2(1.0 + float(positive["accessibility_signal"]))
                    - math.log2(1.0 + float(negative["accessibility_signal"]))
                )
                if distance <= ACCESSIBILITY_LOG2P1_CALIPER:
                    within_caliper.append((positive, negative))
                else:
                    caliper_rejected.append(positive)
            pairs = within_caliper
            unmatched.extend(caliper_rejected)
            samples, pair_rows = build_output_rows(pairs)
            pair_counts = dict(Counter(str(row["chromosome"]) for row in pair_rows))
            splits = choose_splits(pair_rows)
            fixed_test = set(splits["fixed_test"])
            for row in samples:
                row["fixed_partition"] = "test" if row["chromosome"] in fixed_test else "development"
                for fold in splits["development_folds"]:
                    row[f'fold{fold["fold"]}_partition'] = (
                        "test" if row["chromosome"] in fixed_test
                        else "validation" if row["chromosome"] in set(fold["validation"])
                        else "train"
                    )
            match_rate = len(pairs) / len(positives) if positives else 0.0
            stats = balance(pair_rows)
            test_counts = {chrom: pair_counts[chrom] for chrom in splits["fixed_test"]}
            max_test_fraction = max(test_counts.values()) / sum(test_counts.values())
            fold_counts = []
            for fold in splits["development_folds"]:
                validation_count = sum(pair_counts[chrom] for chrom in fold["validation"])
                train_count = sum(pair_counts[chrom] for chrom in fold["train"])
                fold_counts.append({"fold": fold["fold"], "train_per_class": train_count, "validation_per_class": validation_count})
            exact_canonical_duplicates = len(samples) - len({str(row["canonical_key"]) for row in samples})
            label_conflicts = sum(
                1
                for labels in defaultdict(set).values()
                if len(labels) > 1
            )
            gates = {
                "minimum_retained_positive": len(positives) >= 1500,
                "primary_matching_rate": match_rate >= 0.90,
                "gc_balance": stats["gc_passed"],
                "accessibility_balance": stats["accessibility_passed"],
                "test_minimum_three_chromosomes": len(splits["fixed_test"]) >= 3,
                "test_maximum_single_chromosome_fraction": max_test_fraction <= 0.50,
                "test_minimum_per_class": sum(test_counts.values()) >= 200,
                "all_folds_train_minimum_per_class": all(row["train_per_class"] >= 1000 for row in fold_counts),
                "all_folds_validation_minimum_per_class": all(row["validation_per_class"] >= 200 for row in fold_counts),
                "exact_or_rc_duplicates_absent": exact_canonical_duplicates == 0,
            }
            task_passed = all(gates.values())
            all_passed &= task_passed
            sample_fields = [
                "task_id", "sample_id", "pair_id", "label", "chromosome", "start", "end", "summit",
                "source_accession", "source_signal", "accessibility_signal", "gc_fraction", "gc_bin_0.02",
                "accessibility_decile", "sequence", "canonical_key", "fixed_partition",
                "fold1_partition", "fold2_partition", "fold3_partition",
            ]
            pair_fields = [
                "pair_id", "chromosome", "positive_sample_id", "negative_sample_id", "gc_bin_0.02",
                "accessibility_decile", "positive_gc", "negative_gc", "positive_accessibility", "negative_accessibility",
            ]
            dataset_path = PROCESSED / f"{task_id}_matched_dataset.csv.gz"
            pairs_path = PROCESSED / f"{task_id}_matching_pairs.csv.gz"
            write_csv_gz(dataset_path, samples, sample_fields)
            write_csv_gz(pairs_path, pair_rows, pair_fields)
            overall["tasks"][task_id] = {
                "status": "passed_pending_near_duplicate_audit" if task_passed else "failed",
                "candidate_qc": candidate_qc,
                "canonical_deduplication": dedup_qc,
                "retained_positive_candidates": len(positives),
                "retained_negative_candidates": len(negatives),
                "matched_pairs": len(pairs),
                "unmatched_positives": len(unmatched),
                "matched_rate": match_rate,
                "accessibility_log2p1_caliper": ACCESSIBILITY_LOG2P1_CALIPER,
                "caliper_rejected_positives": len(caliper_rejected),
                "balance": stats,
                "pair_counts_by_chromosome": dict(sorted(pair_counts.items(), key=lambda item: int(item[0][3:]))),
                "splits": splits,
                "fixed_test_counts_by_chromosome": test_counts,
                "fixed_test_maximum_single_chromosome_fraction": max_test_fraction,
                "fold_counts": fold_counts,
                "exact_or_rc_duplicate_rows": exact_canonical_duplicates,
                "label_conflicts_placeholder": label_conflicts,
                "gates": gates,
                "accessibility_decile_edges_by_chromosome": edges,
                "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)),
                "dataset_sha256": file_sha256(dataset_path),
                "pair_path": str(pairs_path.relative_to(PROJECT_ROOT)),
                "pair_sha256": file_sha256(pairs_path),
            }
            print(f"task={task_id} matched={len(pairs)} rate={match_rate:.4f} gates={task_passed}", flush=True)
    finally:
        fasta.close()
    overall["status"] = "passed_pending_near_duplicate_audit" if all_passed else "failed"
    audit_path = RESULTS / "dataset_construction_qc.json"
    audit_path.write_text(json.dumps(overall, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"status={overall['status']} qc={audit_path}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
