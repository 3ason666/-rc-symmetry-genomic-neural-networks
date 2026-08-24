from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from rapidfuzz.distance import Levenshtein


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "phase3" / "processed"
RESULTS = PROJECT_ROOT / "results" / "phase3a_external_selection"
CONSTRUCTION_QC = RESULTS / "dataset_construction_qc.json"
TASKS = ("p3_gata1_fetal", "p3_ctcf_gm12878")
KMER = 12
SKETCH_SIZE = 96
MAX_HASH_DOCUMENT_FREQUENCY = 500
QGRAM = 6
MINIMUM_SHARED_QGRAMS = 45
BLOOM_QGRAM = 8
BLOOM_BITS = 8192
MINIMUM_SHARED_BLOOM_QGRAMS = 20
LONG_BLOOM_QGRAM = 10
LONG_BLOOM_BITS = 32768
MINIMUM_SHARED_LONG_BLOOM_QGRAMS = 8
MINIMUM_CONSISTENT_SEED_SUPPORT = 4
COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def hash64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("ascii"), digest_size=8).digest(), "big")


def sketch(sequence: str) -> tuple[int, ...]:
    hashes = set()
    for index in range(len(sequence) - KMER + 1):
        kmer = sequence[index : index + KMER]
        canonical = min(kmer, reverse_complement(kmer))
        hashes.add(hash64(canonical))
    return tuple(sorted(hashes)[:SKETCH_SIZE])


def qgram_mask(sequence: str) -> int:
    encoding = {"A": 0, "C": 1, "G": 2, "T": 3}
    mask = 0
    for index in range(len(sequence) - QGRAM + 1):
        kmer = sequence[index : index + QGRAM]
        rc_kmer = reverse_complement(kmer)
        canonical = min(kmer, rc_kmer)
        code = 0
        for base in canonical:
            code = (code << 2) | encoding[base]
        mask |= 1 << code
    return mask


def bloom_qgram_mask(sequence: str) -> int:
    encoding = {"A": 0, "C": 1, "G": 2, "T": 3}
    mask = 0
    shift = 64 - (BLOOM_BITS.bit_length() - 1)
    for index in range(len(sequence) - BLOOM_QGRAM + 1):
        kmer = sequence[index : index + BLOOM_QGRAM]
        canonical = min(kmer, reverse_complement(kmer))
        code = 0
        for base in canonical:
            code = (code << 2) | encoding[base]
        bucket = ((code * 11_400_714_819_323_198_485) & ((1 << 64) - 1)) >> shift
        mask |= 1 << bucket
    return mask


def long_bloom_qgram_mask(sequence: str) -> int:
    encoding = {"A": 0, "C": 1, "G": 2, "T": 3}
    mask = 0
    shift = 64 - (LONG_BLOOM_BITS.bit_length() - 1)
    for index in range(len(sequence) - LONG_BLOOM_QGRAM + 1):
        kmer = sequence[index : index + LONG_BLOOM_QGRAM]
        canonical = min(kmer, reverse_complement(kmer))
        code = 0
        for base in canonical:
            code = (code << 2) | encoding[base]
        bucket = ((code * 11_400_714_819_323_198_485) & ((1 << 64) - 1)) >> shift
        mask |= 1 << bucket
    return mask


def levenshtein_myers(pattern: str, text: str) -> int:
    if not pattern:
        return len(text)
    if len(pattern) > len(text):
        pattern, text = text, pattern
    masks = {base: 0 for base in "ACGT"}
    for index, base in enumerate(pattern):
        masks[base] |= 1 << index
    positive = ~0
    negative = 0
    score = len(pattern)
    last = 1 << (len(pattern) - 1)
    for base in text:
        equal = masks.get(base, 0)
        xv = equal | negative
        xh = (((equal & positive) + positive) ^ positive) | equal
        ph = negative | ~(xh | positive)
        mh = positive & xh
        if ph & last:
            score += 1
        elif mh & last:
            score -= 1
        ph = (ph << 1) | 1
        mh <<= 1
        positive = mh | ~(xv | ph)
        negative = ph & xv
    return score


def plausible_offsets(sequence_a: str, sequence_b: str, maximum_shift: int) -> list[int]:
    positions: dict[str, list[int]] = defaultdict(list)
    for index in range(len(sequence_b) - KMER + 1):
        positions[sequence_b[index : index + KMER]].append(index)
    counts: Counter[int] = Counter()
    for left_index in range(len(sequence_a) - KMER + 1):
        kmer = sequence_a[left_index : left_index + KMER]
        for right_index in positions.get(kmer, ()):
            shift = left_index - right_index
            if abs(shift) <= maximum_shift:
                counts[shift] += 1
    selected = [shift for shift, count in counts.items() if count >= 2]
    selected.sort(key=lambda shift: (-counts[shift], abs(shift), shift))
    if 0 not in selected:
        selected.append(0)
    return selected[:8]


def maximum_consistent_seed_support(sequence_a: str, sequence_b: str, maximum_shift: int) -> int:
    maximum = 0
    for oriented_b in (sequence_b, reverse_complement(sequence_b)):
        positions: dict[str, list[int]] = defaultdict(list)
        for index in range(len(oriented_b) - KMER + 1):
            positions[oriented_b[index : index + KMER]].append(index)
        counts: Counter[int] = Counter()
        for left_index in range(len(sequence_a) - KMER + 1):
            for right_index in positions.get(sequence_a[left_index : left_index + KMER], ()):
                shift = left_index - right_index
                if abs(shift) <= maximum_shift:
                    counts[shift] += 1
        maximum = max(maximum, max(counts.values(), default=0))
    return maximum


def best_alignment(sequence_a: str, sequence_b: str, coverage_min: float) -> tuple[float, float, str, int]:
    minimum_overlap = int(len(sequence_a) * coverage_min + 0.999999)
    maximum_shift = len(sequence_a) - minimum_overlap
    best = (-1.0, 0.0, "forward", 0)
    for orientation, oriented_b in (("forward", sequence_b), ("reverse_complement", reverse_complement(sequence_b))):
        for shift in plausible_offsets(sequence_a, oriented_b, maximum_shift):
            if shift >= 0:
                left = sequence_a[shift:]
                right = oriented_b[: len(left)]
            else:
                right = oriented_b[-shift:]
                left = sequence_a[: len(right)]
            overlap = min(len(left), len(right))
            if overlap < minimum_overlap:
                continue
            distance = Levenshtein.distance(left, right)
            identity = 1.0 - distance / max(len(left), len(right))
            coverage = min(len(left), len(right)) / max(len(sequence_a), len(sequence_b))
            candidate = (identity, coverage, orientation, shift)
            if candidate[:2] > best[:2]:
                best = candidate
    return best


def crosses_any_partition(left: dict, right: dict) -> bool:
    return any(
        left[field] != right[field]
        for field in ("fixed_partition", "fold1_partition", "fold2_partition", "fold3_partition")
    )


def load_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    fields = list(rows[0])
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def candidate_pairs(rows: list[dict]) -> tuple[set[tuple[int, int]], dict]:
    sketches = [sketch(row["sequence"]) for row in rows]
    frequencies = Counter(value for values in sketches for value in values)
    usable = {
        value for value, frequency in frequencies.items()
        if frequency <= MAX_HASH_DOCUMENT_FREQUENCY
    }
    inverted: dict[int, list[int]] = defaultdict(list)
    one_hit: set[int] = set()
    candidates: set[tuple[int, int]] = set()
    size = len(rows)
    collision_events = 0
    for current, values in enumerate(sketches):
        for value in values:
            if value not in usable:
                continue
            for prior in inverted[value]:
                if not crosses_any_partition(rows[prior], rows[current]):
                    continue
                collision_events += 1
                encoded = prior * size + current
                if encoded in one_hit:
                    candidates.add((prior, current))
                else:
                    one_hit.add(encoded)
            inverted[value].append(current)
    return candidates, {
        "kmer": KMER,
        "sketch_size": SKETCH_SIZE,
        "maximum_hash_document_frequency": MAX_HASH_DOCUMENT_FREQUENCY,
        "unique_hashes": len(frequencies),
        "excluded_common_hashes": sum(frequency > MAX_HASH_DOCUMENT_FREQUENCY for frequency in frequencies.values()),
        "collision_events": collision_events,
        "pairs_with_at_least_one_shared_hash": len(one_hit),
        "pairs_with_at_least_two_shared_hashes": len(candidates),
    }


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def partition_priority(row: dict) -> tuple[int, str, str]:
    if row["fixed_partition"] == "test":
        priority = 3
    elif "validation" in (row["fold1_partition"], row["fold2_partition"], row["fold3_partition"]):
        priority = 2
    else:
        priority = 1
    return priority, row["chromosome"], row["sample_id"]


def audit_task(task_id: str) -> dict:
    dataset_path = PROCESSED / f"{task_id}_matched_dataset.csv.gz"
    pair_path = PROCESSED / f"{task_id}_matching_pairs.csv.gz"
    rows = load_rows(dataset_path)
    candidates, sketch_qc = candidate_pairs(rows)
    print(f"task={task_id} near_duplicate_candidates={len(candidates)}", flush=True)
    qgram_masks = [qgram_mask(row["sequence"]) for row in rows]
    bloom_masks = [bloom_qgram_mask(row["sequence"]) for row in rows]
    long_bloom_masks = [long_bloom_qgram_mask(row["sequence"]) for row in rows]
    candidates_after_qgram = [
        (left, right)
        for left, right in candidates
        if (qgram_masks[left] & qgram_masks[right]).bit_count() >= MINIMUM_SHARED_QGRAMS
        and (bloom_masks[left] & bloom_masks[right]).bit_count() >= MINIMUM_SHARED_BLOOM_QGRAMS
        and (long_bloom_masks[left] & long_bloom_masks[right]).bit_count() >= MINIMUM_SHARED_LONG_BLOOM_QGRAMS
    ]
    candidates_after_position = [
        (left, right)
        for left, right in candidates_after_qgram
        if maximum_consistent_seed_support(
            rows[left]["sequence"], rows[right]["sequence"], 51
        ) >= MINIMUM_CONSISTENT_SEED_SUPPORT
    ]
    sketch_qc["qgram_size"] = QGRAM
    sketch_qc["minimum_shared_qgrams"] = MINIMUM_SHARED_QGRAMS
    sketch_qc["bloom_qgram_size"] = BLOOM_QGRAM
    sketch_qc["bloom_bits"] = BLOOM_BITS
    sketch_qc["minimum_shared_bloom_qgrams"] = MINIMUM_SHARED_BLOOM_QGRAMS
    sketch_qc["long_bloom_qgram_size"] = LONG_BLOOM_QGRAM
    sketch_qc["long_bloom_bits"] = LONG_BLOOM_BITS
    sketch_qc["minimum_shared_long_bloom_qgrams"] = MINIMUM_SHARED_LONG_BLOOM_QGRAMS
    sketch_qc["pairs_after_qgram_filter"] = len(candidates_after_qgram)
    sketch_qc["minimum_consistent_12mer_seed_support"] = MINIMUM_CONSISTENT_SEED_SUPPORT
    sketch_qc["pairs_after_position_filter"] = len(candidates_after_position)
    print(
        f"task={task_id} after_qgram_filter={len(candidates_after_qgram)} "
        f"after_position_filter={len(candidates_after_position)}",
        flush=True,
    )
    primary_edges: list[dict] = []
    sensitivity_edges: list[dict] = []
    for number, (left, right) in enumerate(sorted(candidates_after_position), start=1):
        identity, coverage, orientation, shift = best_alignment(
            rows[left]["sequence"], rows[right]["sequence"], 0.80
        )
        edge = {
            "left_sample_id": rows[left]["sample_id"],
            "right_sample_id": rows[right]["sample_id"],
            "left_chromosome": rows[left]["chromosome"],
            "right_chromosome": rows[right]["chromosome"],
            "identity": identity,
            "bidirectional_coverage": coverage,
            "orientation": orientation,
            "shift": shift,
            "left_index": left,
            "right_index": right,
        }
        if identity >= 0.80 and coverage >= 0.80:
            sensitivity_edges.append(edge)
        if identity >= 0.90 and coverage >= 0.90:
            primary_edges.append(edge)
        if number % 10000 == 0:
            print(f"task={task_id} verified={number}/{len(candidates_after_position)}", flush=True)

    union = UnionFind(len(rows))
    for edge in primary_edges:
        union.union(int(edge["left_index"]), int(edge["right_index"]))
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        groups[union.find(index)].append(index)
    removal_pair_ids: set[str] = set()
    leaking_clusters = []
    for members in groups.values():
        if len(members) < 2:
            continue
        signatures = {
            tuple(rows[index][field] for field in ("fixed_partition", "fold1_partition", "fold2_partition", "fold3_partition"))
            for index in members
        }
        if len(signatures) < 2:
            continue
        keep = max(members, key=lambda index: partition_priority(rows[index]))
        remove = [index for index in members if index != keep]
        removal_pair_ids.update(rows[index]["pair_id"] for index in remove)
        leaking_clusters.append(
            {
                "kept_sample_id": rows[keep]["sample_id"],
                "removed_sample_ids": [rows[index]["sample_id"] for index in remove],
                "removed_pair_ids": sorted({rows[index]["pair_id"] for index in remove}),
            }
        )

    before_rows = len(rows)
    if removal_pair_ids:
        rows = [row for row in rows if row["pair_id"] not in removal_pair_ids]
        pairs = load_rows(pair_path)
        pairs = [row for row in pairs if row["pair_id"] not in removal_pair_ids]
        write_rows(dataset_path, rows)
        write_rows(pair_path, pairs)

    edge_path = RESULTS / f"{task_id}_near_duplicate_edges.json"
    edge_path.write_text(
        json.dumps(
            {
                "primary_90_90_edges": primary_edges,
                "sensitivity_80_80_edges": sensitivity_edges,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return {
        "sketch_qc": sketch_qc,
        "primary_90_90_cross_partition_edges": len(primary_edges),
        "sensitivity_80_80_cross_partition_edges": len(sensitivity_edges),
        "primary_leaking_clusters": len(leaking_clusters),
        "removed_pairs": len(removal_pair_ids),
        "rows_before": before_rows,
        "rows_after": len(rows),
        "remaining_pairs": len(rows) // 2,
        "cluster_actions": leaking_clusters[:100],
        "edges_path": str(edge_path.relative_to(PROJECT_ROOT)),
        "edges_sha256": sha256(edge_path),
        "dataset_sha256_after_audit": sha256(dataset_path),
        "pair_sha256_after_audit": sha256(pair_path),
    }


def synthetic_recall(identity: float, replicates: int, seed: int) -> dict:
    rng = random.Random(seed)
    detected_one = 0
    detected_two = 0
    qgram_detected = 0
    bloom_detected = 0
    long_bloom_detected = 0
    position_seed_detected = 0
    verified = 0
    shared_counts = []
    for _ in range(replicates):
        original = "".join(rng.choice("ACGT") for _ in range(256))
        mutated = list(original)
        mutation_count = math.floor((1.0 - identity) * len(mutated))
        for index in rng.sample(range(len(mutated)), mutation_count):
            mutated[index] = rng.choice([base for base in "ACGT" if base != mutated[index]])
        mutated_sequence = "".join(mutated)
        if rng.random() < 0.5:
            mutated_sequence = reverse_complement(mutated_sequence)
        shared = len(set(sketch(original)).intersection(sketch(mutated_sequence)))
        shared_counts.append(shared)
        detected_one += int(shared >= 1)
        detected_two += int(shared >= 2)
        qgram_detected += int(
            (qgram_mask(original) & qgram_mask(mutated_sequence)).bit_count()
            >= MINIMUM_SHARED_QGRAMS
        )
        bloom_detected += int(
            (bloom_qgram_mask(original) & bloom_qgram_mask(mutated_sequence)).bit_count()
            >= MINIMUM_SHARED_BLOOM_QGRAMS
        )
        long_bloom_detected += int(
            (long_bloom_qgram_mask(original) & long_bloom_qgram_mask(mutated_sequence)).bit_count()
            >= MINIMUM_SHARED_LONG_BLOOM_QGRAMS
        )
        position_seed_detected += int(
            maximum_consistent_seed_support(original, mutated_sequence, 51)
            >= MINIMUM_CONSISTENT_SEED_SUPPORT
        )
        observed_identity, coverage, _, _ = best_alignment(original, mutated_sequence, 0.80)
        verified += int(observed_identity >= identity - 1e-9 and coverage >= 0.80)
    return {
        "target_identity": identity,
        "replicates": replicates,
        "candidate_recall_at_least_one_shared_hash": detected_one / replicates,
        "candidate_recall_at_least_two_shared_hashes": detected_two / replicates,
        "qgram_filter_recall": qgram_detected / replicates,
        "bloom_qgram_filter_recall": bloom_detected / replicates,
        "long_bloom_qgram_filter_recall": long_bloom_detected / replicates,
        "position_seed_filter_recall": position_seed_detected / replicates,
        "alignment_verification_recall": verified / replicates,
        "minimum_shared_hashes": min(shared_counts),
        "median_shared_hashes": sorted(shared_counts)[replicates // 2],
    }


def main() -> int:
    construction = json.loads(CONSTRUCTION_QC.read_text(encoding="utf-8"))
    validation = {
        "primary_90_percent": synthetic_recall(0.90, 250, 20260823),
        "sensitivity_80_percent": synthetic_recall(0.80, 250, 20260824),
    }
    tasks = {}
    for task_id in TASKS:
        tasks[task_id] = audit_task(task_id)
    primary_recall = validation["primary_90_percent"]["candidate_recall_at_least_two_shared_hashes"]
    sensitivity_recall = validation["sensitivity_80_percent"]["candidate_recall_at_least_two_shared_hashes"]
    passed = (
        construction.get("status") == "passed_pending_near_duplicate_audit"
        and primary_recall >= 0.99
        and sensitivity_recall >= 0.90
        and validation["primary_90_percent"]["qgram_filter_recall"] >= 0.99
        and validation["primary_90_percent"]["bloom_qgram_filter_recall"] >= 0.99
        and validation["sensitivity_80_percent"]["qgram_filter_recall"] >= 0.95
        and validation["sensitivity_80_percent"]["bloom_qgram_filter_recall"] >= 0.95
        and validation["primary_90_percent"]["long_bloom_qgram_filter_recall"] >= 0.99
        and validation["sensitivity_80_percent"]["long_bloom_qgram_filter_recall"] >= 0.95
        and validation["primary_90_percent"]["position_seed_filter_recall"] >= 0.99
        and validation["sensitivity_80_percent"]["position_seed_filter_recall"] >= 0.95
        and validation["primary_90_percent"]["alignment_verification_recall"] >= 0.99
        and validation["sensitivity_80_percent"]["alignment_verification_recall"] >= 0.95
    )
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_version": "phase3_near_duplicate_v1",
        "reverse_complement_aware": True,
        "primary_definition": {"identity_min": 0.90, "bidirectional_coverage_min": 0.90},
        "sensitivity_definition": {"identity_min": 0.80, "bidirectional_coverage_min": 0.80},
        "synthetic_validation": validation,
        "tasks": tasks,
        "status": "passed" if passed else "failed",
        "formal_training_allowed": False,
        "next_gate": "post_removal_balance_counts_and_manifest_freeze" if passed else "review_near_duplicate_implementation",
    }
    output = RESULTS / "near_duplicate_audit.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"status={payload['status']} qc={output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
