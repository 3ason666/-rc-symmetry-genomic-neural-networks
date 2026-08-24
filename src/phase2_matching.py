from __future__ import annotations

import hashlib
import itertools
import math
from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np


def stable_rank(identifier: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).hexdigest()


def quantile_edges(values: Iterable[float], bins: int = 10) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("cannot define quantiles from an empty collection")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    return np.quantile(array, np.linspace(0.0, 1.0, bins + 1), method="linear")


def assign_quantile(value: float, edges: np.ndarray) -> int:
    return int(np.searchsorted(edges[1:-1], float(value), side="right"))


def nonoverlapping_priority_selection(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Greedily keep high-signal fixed-window records without interval overlap."""
    kept: list[dict] = []
    rejected: list[dict] = []
    occupied: dict[str, list[tuple[int, int]]] = defaultdict(list)
    ordered = sorted(
        records,
        key=lambda row: (
            -float(row["source_signal"]),
            -float(row["source_score"]),
            str(row["chromosome"]),
            int(row["start"]),
            str(row["sample_id"]),
        ),
    )
    for record in ordered:
        chromosome = str(record["chromosome"])
        start, end = int(record["start"]), int(record["end"])
        overlaps = any(prior_start < end and start < prior_end for prior_start, prior_end in occupied[chromosome])
        if overlaps:
            rejected.append(record)
        else:
            kept.append(record)
            occupied[chromosome].append((start, end))
    kept.sort(key=lambda row: (int(str(row["chromosome"]).removeprefix("chr")), int(row["start"])))
    return kept, rejected


def match_exact_strata(
    positives: list[dict],
    negatives: list[dict],
    *,
    seed: int,
) -> tuple[list[tuple[dict, dict]], list[dict]]:
    negative_groups: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in negatives:
        key = (str(row["chromosome"]), int(row["gc_bin_0.02"]), int(row["atac_quantile"]))
        negative_groups[key].append(row)
    for group in negative_groups.values():
        group.sort(key=lambda row: (stable_rank(str(row["sample_id"]), seed), str(row["sample_id"])))

    positive_groups: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in positives:
        key = (str(row["chromosome"]), int(row["gc_bin_0.02"]), int(row["atac_quantile"]))
        positive_groups[key].append(row)

    pairs: list[tuple[dict, dict]] = []
    unmatched: list[dict] = []
    for key in sorted(positive_groups):
        positive_group = sorted(positive_groups[key], key=lambda row: str(row["sample_id"]))
        negative_group = negative_groups.get(key, [])
        matched_count = min(len(positive_group), len(negative_group))
        pairs.extend(zip(positive_group[:matched_count], negative_group[:matched_count]))
        unmatched.extend(positive_group[matched_count:])
    return pairs, unmatched


def _candidate_subsets(counts: Mapping[str, int], target: float) -> list[tuple[float, int, tuple[str, ...]]]:
    chromosomes = sorted(counts, key=lambda value: int(value.removeprefix("chr")))
    upper = max(counts.values()) + target
    candidates: list[tuple[float, int, tuple[str, ...]]] = []

    def visit(index: int, selected: tuple[str, ...], total: int) -> None:
        if total <= upper:
            candidates.append((abs(total - target), total, selected))
        else:
            return
        for next_index in range(index, len(chromosomes)):
            chromosome = chromosomes[next_index]
            visit(next_index + 1, selected + (chromosome,), total + int(counts[chromosome]))

    visit(0, (), 0)
    candidates.sort(key=lambda item: (item[0], len(item[2]), item[2]))
    return candidates


def choose_chromosome_splits(
    counts: Mapping[str, int],
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, tuple[str, ...]]:
    if not counts or any(value <= 0 for value in counts.values()):
        raise ValueError("all eligible chromosome counts must be positive")
    total = sum(counts.values())
    validation_target = total * validation_fraction
    test_target = total * test_fraction
    validation_candidates = _candidate_subsets(counts, validation_target)
    test_candidates = _candidate_subsets(counts, test_target)

    best: tuple[float, int, tuple[str, ...], tuple[str, ...]] | None = None
    for validation_error, _, validation in validation_candidates:
        if not validation:
            continue
        if best is not None and validation_error > best[0]:
            break
        validation_set = set(validation)
        for test_error, _, test in test_candidates:
            combined_error = validation_error + test_error
            if best is not None and combined_error > best[0]:
                break
            if not test or validation_set.intersection(test):
                continue
            candidate = (combined_error, len(validation) + len(test), validation, test)
            if best is None or candidate < best:
                best = candidate
            break
    if best is None:
        raise ValueError("could not assign disjoint validation and test chromosomes")
    _, _, validation, test = best
    held_out = set(validation).union(test)
    train = tuple(
        chromosome
        for chromosome in sorted(counts, key=lambda value: int(value.removeprefix("chr")))
        if chromosome not in held_out
    )
    return {"train": train, "validation": validation, "test": test}


def shannon_entropy(sequence: str) -> float:
    counts = {base: sequence.count(base) for base in "ACGT"}
    length = len(sequence)
    return -sum((count / length) * math.log2(count / length) for count in counts.values() if count)


def maximum_homopolymer(sequence: str) -> int:
    return max((sum(1 for _ in group) for _, group in itertools.groupby(sequence)), default=0)


def cross_split_near_duplicates(
    records: list[dict],
    max_hamming_distance: int = 2,
) -> list[tuple[str, str, int]]:
    if max_hamming_distance != 2:
        raise ValueError("current exact block index is defined for Hamming distance <= 2")
    if not records:
        return []
    length = len(str(records[0]["canonical_key"]))
    cut1, cut2 = length // 3, 2 * length // 3
    buckets: dict[tuple[int, str], list[dict]] = defaultdict(list)
    leaks: list[tuple[str, str, int]] = []
    compared: set[tuple[str, str]] = set()
    for record in records:
        sequence = str(record["canonical_key"])
        blocks = (sequence[:cut1], sequence[cut1:cut2], sequence[cut2:])
        for block_index, block in enumerate(blocks):
            for prior in buckets[(block_index, block)]:
                if prior["split"] == record["split"]:
                    continue
                pair = tuple(sorted((str(prior["sample_id"]), str(record["sample_id"]))))
                if pair in compared:
                    continue
                compared.add(pair)
                prior_sequence = str(prior["canonical_key"])
                distance = sum(left != right for left, right in zip(prior_sequence, sequence))
                if distance <= max_hamming_distance:
                    leaks.append((pair[0], pair[1], distance))
            buckets[(block_index, block)].append(record)
    return leaks
