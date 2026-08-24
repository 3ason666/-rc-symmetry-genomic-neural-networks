from __future__ import annotations

import gzip
import heapq
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


PRIMARY_CHROMOSOMES = tuple(f"chr{i}" for i in range(1, 23))
DNA = frozenset("ACGT")
_COMPLEMENT = str.maketrans("ACGT", "TGCA")


@dataclass(frozen=True)
class Peak:
    accession: str
    row_number: int
    chromosome: str
    start: int
    end: int
    score: float
    signal: float
    summit: int

    @property
    def sample_id(self) -> str:
        return f"{self.accession}:{self.row_number}"


def parse_narrowpeak(
    path: Path,
    accession: str,
    *,
    invalid_summit_policy: str = "error",
) -> dict[str, list[Peak]]:
    if invalid_summit_policy not in {"error", "skip"}:
        raise ValueError("invalid_summit_policy must be 'error' or 'skip'")
    by_chromosome: dict[str, list[Peak]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) < 10:
                raise ValueError(f"{path.name}:{row_number}: expected narrowPeak with 10 columns")
            start, end = int(fields[1]), int(fields[2])
            summit_offset = int(fields[9])
            if start < 0 or end <= start:
                raise ValueError(f"{path.name}:{row_number}: invalid coordinates")
            if summit_offset < 0 or start + summit_offset >= end:
                if invalid_summit_policy == "skip":
                    continue
                raise ValueError(f"{path.name}:{row_number}: invalid summit")
            peak = Peak(
                accession=accession,
                row_number=row_number,
                chromosome=fields[0],
                start=start,
                end=end,
                score=float(fields[4]),
                signal=float(fields[6]),
                summit=start + summit_offset,
            )
            by_chromosome.setdefault(peak.chromosome, []).append(peak)
    for peaks in by_chromosome.values():
        peaks.sort(key=lambda item: (item.start, item.end, item.row_number))
    return by_chromosome


def parse_bed3(path: Path) -> dict[str, list[tuple[int, int]]]:
    by_chromosome: dict[str, list[tuple[int, int]]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) < 3:
                raise ValueError(f"{path.name}:{row_number}: expected at least 3 BED columns")
            start, end = int(fields[1]), int(fields[2])
            if start < 0 or end <= start:
                raise ValueError(f"{path.name}:{row_number}: invalid coordinates")
            by_chromosome.setdefault(fields[0], []).append((start, end))
    return {chromosome: merge_intervals(intervals) for chromosome, intervals in by_chromosome.items()}


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted(intervals)
    merged: list[list[int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


class IntervalIndex:
    def __init__(self, intervals: Iterable[tuple[int, int]] = ()) -> None:
        self.intervals = merge_intervals(intervals)
        self.starts = [start for start, _ in self.intervals]

    def overlaps(self, start: int, end: int) -> bool:
        if end <= start:
            raise ValueError("end must be greater than start")
        insertion = bisect_left(self.starts, end)
        return insertion > 0 and self.intervals[insertion - 1][1] > start


def window_from_summit(summit: int, length: int = 256) -> tuple[int, int]:
    if length <= 0:
        raise ValueError("length must be positive")
    start = summit - length // 2
    return start, start + length


def sequence_features(sequence: str) -> tuple[float, int, str]:
    sequence = sequence.upper()
    invalid = set(sequence).difference(DNA)
    if invalid:
        raise ValueError(f"invalid DNA characters: {sorted(invalid)}")
    gc_fraction = (sequence.count("G") + sequence.count("C")) / len(sequence)
    gc_bin = min(49, int(gc_fraction / 0.02))
    reverse_complement = sequence.translate(_COMPLEMENT)[::-1]
    canonical_key = min(sequence, reverse_complement)
    return gc_fraction, gc_bin, canonical_key


def maximum_overlapping_signal(peaks: list[Peak], start: int, end: int) -> float | None:
    maximum: float | None = None
    for peak in peaks:
        if peak.start >= end:
            break
        if peak.end > start:
            maximum = peak.signal if maximum is None else max(maximum, peak.signal)
    return maximum


def maximum_overlapping_signals(
    peaks: list[Peak],
    intervals: Iterable[tuple[str, int, int]],
) -> dict[str, float | None]:
    """Return maximum peak signal for sorted half-open query intervals.

    Query intervals may be supplied unsorted; fixed-width summit windows are
    sorted internally so their ends are monotonic and a sweep-line heap can be
    used instead of repeatedly scanning all ATAC peaks.
    """
    ordered_queries = sorted(intervals, key=lambda item: (item[1], item[2], item[0]))
    ordered_peaks = sorted(peaks, key=lambda item: (item.start, item.end, item.row_number))
    active: list[tuple[float, int, int]] = []
    peak_index = 0
    result: dict[str, float | None] = {}
    for query_id, start, end in ordered_queries:
        while peak_index < len(ordered_peaks) and ordered_peaks[peak_index].start < end:
            peak = ordered_peaks[peak_index]
            heapq.heappush(active, (-peak.signal, peak.end, peak.row_number))
            peak_index += 1
        while active and active[0][1] <= start:
            heapq.heappop(active)
        result[query_id] = -active[0][0] if active else None
    return result


def fasta_records(path: Path) -> Iterator[tuple[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    name: str | None = None
    chunks: list[str] = []
    with opener(path, "rt", encoding="ascii") as handle:
        for line in handle:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks).upper()
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
    if name is not None:
        yield name, "".join(chunks).upper()
