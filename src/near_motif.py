from __future__ import annotations

import json
from collections.abc import Iterable, Sequence

import pandas as pd

from src.dna_utils import reverse_complement


def parse_intervals(value) -> list[list[int]]:
    """Normalize JSON-encoded or in-memory half-open intervals."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    intervals = json.loads(value) if isinstance(value, str) else value
    return [[int(start), int(end)] for start, end in intervals]


def merge_intervals(intervals: Iterable[Sequence[int]]) -> list[list[int]]:
    """Return the union of overlapping or touching half-open intervals."""
    ordered = sorted((int(start), int(end)) for start, end in intervals)
    merged: list[list[int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def _overlaps(interval: Sequence[int], excluded: Sequence[Sequence[int]]) -> bool:
    start, end = int(interval[0]), int(interval[1])
    return any(start < int(other_end) and int(other_start) < end
               for other_start, other_end in excluded)


def hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("Hamming distance requires equal-length strings")
    return sum(a != b for a, b in zip(left.upper(), right.upper()))


def scan_incidental_near_motifs(
    sequence: str,
    motifs: Sequence[str],
    max_hamming: int,
    excluded_intervals: Sequence[Sequence[int]] = (),
) -> tuple[list[dict], list[list[int]]]:
    """Find motif-like background windows outside annotated regions.

    Each genomic window is retained once, using its closest forward/RC target.
    The returned hit list preserves individual windows; the interval list is the
    merged union used for attribution-mass calculations and plot shading.
    """
    sequence = str(sequence).upper()
    if max_hamming < 0:
        raise ValueError("max_hamming must be non-negative")
    excluded = [[int(start), int(end)] for start, end in excluded_intervals]
    best_by_interval: dict[tuple[int, int], dict] = {}
    for motif_index, motif_value in enumerate(motifs):
        motif = str(motif_value).upper()
        targets = [("forward", motif)]
        rc = reverse_complement(motif)
        if rc != motif:
            targets.append(("reverse_complement", rc))
        for start in range(len(sequence) - len(motif) + 1):
            end = start + len(motif)
            if _overlaps((start, end), excluded):
                continue
            window = sequence[start:end]
            candidates = [
                (hamming_distance(window, target), orientation, target)
                for orientation, target in targets
            ]
            distance, orientation, target = min(candidates, key=lambda item: (item[0], item[1]))
            if distance > max_hamming:
                continue
            hit = {
                "start": start,
                "end": end,
                "instance": window,
                "motif": motif,
                "motif_index": motif_index,
                "orientation": orientation,
                "target": target,
                "hamming_distance": int(distance),
            }
            key = (start, end)
            previous = best_by_interval.get(key)
            if previous is None or (hit["hamming_distance"], hit["motif_index"]) < (
                previous["hamming_distance"], previous["motif_index"]
            ):
                best_by_interval[key] = hit
    hits = sorted(best_by_interval.values(), key=lambda item: (item["start"], item["end"]))
    return hits, merge_intervals((hit["start"], hit["end"]) for hit in hits)


def annotate_near_motifs(
    frame: pd.DataFrame,
    motifs: Sequence[str],
    max_hamming: int,
    excluded_columns: Sequence[str] = (
        "causal_intervals", "decoy_intervals", "shortcut_intervals"
    ),
) -> tuple[pd.DataFrame, dict]:
    """Add reproducible incidental-near-motif annotations to a dataset."""
    annotated = frame.copy()
    hit_json: list[str] = []
    interval_json: list[str] = []
    counts: list[int] = []
    base_counts: list[int] = []
    for row in annotated.itertuples(index=False):
        excluded: list[list[int]] = []
        for column in excluded_columns:
            if hasattr(row, column):
                excluded.extend(parse_intervals(getattr(row, column)))
        hits, intervals = scan_incidental_near_motifs(
            row.sequence, motifs, max_hamming, excluded
        )
        hit_json.append(json.dumps(hits, separators=(",", ":")))
        interval_json.append(json.dumps(intervals, separators=(",", ":")))
        counts.append(len(hits))
        base_counts.append(sum(end - start for start, end in intervals))
    annotated["incidental_near_motif_hits"] = hit_json
    annotated["incidental_near_motif_intervals"] = interval_json
    annotated["incidental_near_motif_count"] = counts
    annotated["incidental_near_motif_base_count"] = base_counts
    report = {
        "motifs": [str(motif).upper() for motif in motifs],
        "max_hamming": int(max_hamming),
        "excluded_columns": list(excluded_columns),
        "samples_with_incidental_near_motif": int(sum(count > 0 for count in counts)),
        "sample_fraction_with_incidental_near_motif": float(sum(count > 0 for count in counts) / len(counts)),
        "total_hit_windows": int(sum(counts)),
        "total_merged_bases": int(sum(base_counts)),
    }
    return annotated, report
