from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from rapidfuzz.distance import Levenshtein


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "phase3d_cross_phase_overlap_audit.yaml"
PROTOCOL_PATH = PROJECT_ROOT / "protocols" / "phase3d_cross_phase_overlap_audit_protocol.md"
RUNNER_PATH = Path(__file__).resolve()
COMPLEMENT = str.maketrans("ACGT", "TGCA")
MASK64 = (1 << 64) - 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def atomic_csv_gz(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def canonical_sequence(sequence: str) -> str:
    sequence = sequence.upper()
    rc = reverse_complement(sequence)
    return min(sequence, rc)


def canonical_sha256(sequence: str) -> str:
    return hashlib.sha256(canonical_sequence(sequence).encode("ascii")).hexdigest()


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def canonical_kmer_codes(sequence: str, kmer_size: int):
    encoding = {"A": 0, "C": 1, "G": 2, "T": 3}
    mask = (1 << (2 * kmer_size)) - 1
    forward = 0
    reverse = 0
    high_shift = 2 * (kmer_size - 1)
    for index, base in enumerate(sequence):
        code = encoding[base]
        forward = ((forward << 2) | code) & mask
        reverse = (reverse >> 2) | ((3 - code) << high_shift)
        if index >= kmer_size - 1:
            yield min(forward, reverse)


def minhash_sketch(sequence: str, kmer_size: int, seed: int, sketch_size: int) -> tuple[int, ...]:
    hashed = {splitmix64(code ^ seed) for code in canonical_kmer_codes(sequence, kmer_size)}
    return tuple(sorted(hashed)[:sketch_size])


def best_rc_alignment(sequence_a: str, sequence_b: str, coverage_min: float) -> tuple[float, float, str, int]:
    minimum_overlap = int(max(len(sequence_a), len(sequence_b)) * coverage_min + 0.999999)
    maximum_shift = max(len(sequence_a), len(sequence_b)) - minimum_overlap
    best = (-1.0, 0.0, "forward", 0)
    for orientation, oriented_b in (("forward", sequence_b), ("reverse_complement", reverse_complement(sequence_b))):
        for shift in range(-maximum_shift, maximum_shift + 1):
            if shift >= 0:
                left = sequence_a[shift:]
                right = oriented_b[: len(left)]
            else:
                right = oriented_b[-shift:]
                left = sequence_a[: len(right)]
            overlap = min(len(left), len(right))
            if overlap < minimum_overlap:
                continue
            left = left[:overlap]
            right = right[:overlap]
            distance = Levenshtein.distance(left, right)
            identity = 1.0 - distance / max(len(left), len(right))
            coverage = overlap / max(len(sequence_a), len(sequence_b))
            candidate = (identity, coverage, orientation, shift)
            if candidate[:2] > best[:2]:
                best = candidate
    return best


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_config(config: dict) -> None:
    if config["protocol_revision"] != "phase3d_cross_phase_overlap_audit_v1":
        raise ValueError("unexpected protocol revision")
    if config["scope"]["p2f_access_mode"] != "isolated_read_only":
        raise ValueError("P2F access must remain isolated read-only")
    if config["scope"]["p2f_predictions_or_attributions_allowed"]:
        raise ValueError("P2F predictions and attributions are forbidden")
    if config["high_similarity"]["kmer_size"] != 15:
        raise ValueError("frozen k-mer size changed")
    if config["high_similarity"]["confirmation_identity_min"] != 0.90:
        raise ValueError("frozen identity threshold changed")
    if config["high_similarity"]["confirmation_bidirectional_coverage_min"] != 0.90:
        raise ValueError("frozen coverage threshold changed")
    if config["coordinate_overlap"]["trigger_fraction_of_p3_train"] != 0.05:
        raise ValueError("coordinate trigger changed")
    if config["exact_sequence_overlap"]["trigger_fraction_of_p3_train"] != 0.01:
        raise ValueError("exact trigger changed")
    if config["reporting"]["persist_raw_sequences"]:
        raise ValueError("raw sequences may not be persisted")


def validate_completion(path: Path, expected_sha256: str, protocol_prefix: str) -> dict:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"completion hash mismatch: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") != "completed" or int(payload.get("test_execution_count", 0)) != 1:
        raise ValueError(f"primary result not completed exactly once: {path}")
    if not str(payload.get("protocol_revision", "")).startswith(protocol_prefix):
        raise ValueError(f"unexpected completion protocol: {path}")
    return payload


def preflight(config: dict) -> dict:
    validate_config(config)
    checks = []

    def check_file(label: str, relative_path: str, expected: str):
        path = PROJECT_ROOT / relative_path
        observed = sha256_file(path)
        checks.append({"resource": label, "path": relative_path, "expected_sha256": expected, "observed_sha256": observed, "passed": observed == expected})
        if observed != expected:
            raise ValueError(f"hash mismatch: {label}")

    inputs = config["inputs"]
    check_file("p2f_dataset", inputs["p2f_dataset_path"], inputs["p2f_dataset_sha256"])
    check_file("p2f_completion", inputs["p2f_completion_path"], inputs["p2f_completion_sha256"])
    check_file("p3_completion", inputs["p3_completion_path"], inputs["p3_completion_sha256"])
    for task_id, spec in inputs["p3_tasks"].items():
        check_file(f"{task_id}_development", spec["development_path"], spec["development_sha256"])

    validate_completion(PROJECT_ROOT / inputs["p2f_completion_path"], inputs["p2f_completion_sha256"], "phase2f_")
    validate_completion(PROJECT_ROOT / inputs["p3_completion_path"], inputs["p3_completion_sha256"], "phase3c_")
    return {
        "protocol_revision": config["protocol_revision"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_rows_loaded": False,
        "checks": checks,
        "all_passed": all(check["passed"] for check in checks),
        "config_sha256": sha256_file(CONFIG_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "runner_sha256": sha256_file(RUNNER_PATH),
    }


def load_p2f_test(config: dict) -> list[dict]:
    path = PROJECT_ROOT / config["inputs"]["p2f_dataset_path"]
    required = {"sample_id", "label", "split", "chromosome", "start", "end", "sequence"}
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("P2F dataset columns changed")
        for row in reader:
            if row["split"] != config["inputs"]["p2f_test_split"]:
                continue
            sequence = row["sequence"].upper()
            rows.append({
                "sample_id": row["sample_id"],
                "label": int(row["label"]),
                "chromosome": row["chromosome"],
                "start": int(row["start"]),
                "end": int(row["end"]),
                "sequence": sequence,
                "canonical_sha256": canonical_sha256(sequence),
            })
    if len(rows) != int(config["inputs"]["p2f_expected_test_rows"]):
        raise ValueError("P2F test row count changed")
    return rows


def load_p3_development(task_id: str, config: dict) -> list[dict]:
    spec = config["inputs"]["p3_tasks"][task_id]
    path = PROJECT_ROOT / spec["development_path"]
    rows = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sequence = row["sequence"].upper()
            fold_train = {fold: row[f"fold{fold}_partition"] == "train" for fold in (1, 2, 3)}
            rows.append({
                "task_id": task_id,
                "sample_id": row["sample_id"],
                "label": int(row["label"]),
                "chromosome": row["chromosome"],
                "start": int(row["start"]),
                "end": int(row["end"]),
                "sequence": sequence,
                "canonical_sha256": canonical_sha256(sequence),
                "fold_train": fold_train,
                "union_train": any(fold_train.values()),
            })
    if len(rows) != int(spec["expected_development_rows"]):
        raise ValueError(f"P3 development row count changed: {task_id}")
    return [row for row in rows if row["union_train"]]


def interval_overlap_pairs(p3_rows: list[dict], p2_rows: list[dict]) -> list[tuple[int, int, int]]:
    p3_by_chromosome = defaultdict(list)
    p2_by_chromosome = defaultdict(list)
    for index, row in enumerate(p3_rows):
        p3_by_chromosome[row["chromosome"]].append((row["start"], row["end"], index))
    for index, row in enumerate(p2_rows):
        p2_by_chromosome[row["chromosome"]].append((row["start"], row["end"], index))
    pairs = []
    for chromosome in sorted(set(p3_by_chromosome) & set(p2_by_chromosome)):
        left_rows = sorted(p3_by_chromosome[chromosome])
        right_rows = sorted(p2_by_chromosome[chromosome])
        active = []
        right_cursor = 0
        for left_start, left_end, left_index in left_rows:
            while right_cursor < len(right_rows) and right_rows[right_cursor][0] < left_end:
                active.append(right_rows[right_cursor])
                right_cursor += 1
            active = [entry for entry in active if entry[1] > left_start]
            for right_start, right_end, right_index in active:
                overlap_bp = min(left_end, right_end) - max(left_start, right_start)
                if overlap_bp > 0:
                    pairs.append((left_index, right_index, overlap_bp))
    return pairs


def exact_overlap_pairs(p3_rows: list[dict], p2_rows: list[dict]) -> list[tuple[int, int]]:
    p2_by_hash = defaultdict(list)
    for index, row in enumerate(p2_rows):
        p2_by_hash[row["canonical_sha256"]].append(index)
    pairs = []
    for p3_index, row in enumerate(p3_rows):
        for p2_index in p2_by_hash.get(row["canonical_sha256"], ()): 
            pairs.append((p3_index, p2_index))
    return pairs


def high_similarity_pairs(p3_rows: list[dict], p2_rows: list[dict], config: dict) -> tuple[list[dict], dict]:
    spec = config["high_similarity"]
    kmer_size = int(spec["kmer_size"])
    seed = int(spec["minhash_seed"])
    sketch_size = int(spec["sketch_size"])
    minimum_shared = int(spec["minimum_shared_sketch_hashes"])
    maximum_frequency = int(spec["maximum_p2f_hash_document_frequency"])
    identity_min = float(spec["confirmation_identity_min"])
    coverage_min = float(spec["confirmation_bidirectional_coverage_min"])

    p2_sketches = [minhash_sketch(row["sequence"], kmer_size, seed, sketch_size) for row in p2_rows]
    frequencies = Counter(value for sketch in p2_sketches for value in sketch)
    usable = {value for value, frequency in frequencies.items() if frequency <= maximum_frequency}
    inverted = defaultdict(list)
    for p2_index, sketch in enumerate(p2_sketches):
        for value in sketch:
            if value in usable:
                inverted[value].append(p2_index)

    candidate_pairs = []
    for p3_index, row in enumerate(p3_rows):
        sketch = minhash_sketch(row["sequence"], kmer_size, seed, sketch_size)
        counts = Counter()
        for value in sketch:
            for p2_index in inverted.get(value, ()):
                counts[p2_index] += 1
        for p2_index, shared in counts.items():
            if shared >= minimum_shared:
                candidate_pairs.append((p3_index, p2_index, shared, len(sketch), len(p2_sketches[p2_index])))

    confirmed = []
    for p3_index, p2_index, shared, p3_size, p2_size in candidate_pairs:
        identity, coverage, orientation, shift = best_rc_alignment(
            p3_rows[p3_index]["sequence"], p2_rows[p2_index]["sequence"], coverage_min
        )
        if identity >= identity_min and coverage >= coverage_min:
            confirmed.append({
                "p3_index": p3_index,
                "p2_index": p2_index,
                "shared_sketch_hashes": shared,
                "estimated_jaccard": shared / max(1, p3_size + p2_size - shared),
                "identity": identity,
                "coverage": coverage,
                "orientation": orientation,
                "shift": shift,
            })
    return confirmed, {
        "p2f_unique_sketch_hashes": len(frequencies),
        "p2f_excluded_common_hashes": sum(frequency > maximum_frequency for frequency in frequencies.values()),
        "candidate_pair_count": len(candidate_pairs),
        "confirmed_pair_count": len(confirmed),
    }


def unique_indices(pairs, position: int) -> set[int]:
    return {pair[position] if not isinstance(pair, dict) else pair["p3_index" if position == 0 else "p2_index"] for pair in pairs}


def fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def fold_summary(p3_rows: list[dict], p2_count: int, coordinate_pairs, exact_pairs, similarity_pairs) -> dict:
    result = {}
    for scope in ("union", "fold1", "fold2", "fold3"):
        eligible = set(range(len(p3_rows))) if scope == "union" else {
            index for index, row in enumerate(p3_rows) if row["fold_train"][int(scope[-1])]
        }
        denominator = len(eligible)
        row = {"p3_train_rows": denominator}
        for label, pairs in (("coordinate", coordinate_pairs), ("exact", exact_pairs), ("high_similarity", similarity_pairs)):
            p3_indices = unique_indices(pairs, 0) & eligible
            p2_indices = {
                (pair[1] if not isinstance(pair, dict) else pair["p2_index"])
                for pair in pairs
                if (pair[0] if not isinstance(pair, dict) else pair["p3_index"]) in eligible
            }
            row[label] = {
                "unique_p3_rows": len(p3_indices),
                "fraction_of_p3_train": fraction(len(p3_indices), denominator),
                "unique_p2f_test_rows": len(p2_indices),
                "fraction_of_p2f_test": fraction(len(p2_indices), p2_count),
            }
        result[scope] = row
    return result


def public_pair_row(task_id: str, p3: dict, p2: dict) -> dict:
    return {
        "task_id": task_id,
        "p3_sample_id": p3["sample_id"],
        "p3_label": p3["label"],
        "p3_chromosome": p3["chromosome"],
        "p3_start": p3["start"],
        "p3_end": p3["end"],
        "p2f_sample_id": p2["sample_id"],
        "p2f_label": p2["label"],
        "p2f_chromosome": p2["chromosome"],
        "p2f_start": p2["start"],
        "p2f_end": p2["end"],
    }


def run_audit(config: dict, output_dir: Path) -> dict:
    completion_path = output_dir / "completion.json"
    if completion_path.exists():
        raise RuntimeError("Phase 3D audit is already completed and locked")

    input_audit = preflight(config)
    atomic_json(output_dir / "input_audit.json", input_audit)
    p2_rows = load_p2f_test(config)
    input_audit["test_rows_loaded"] = True
    input_audit["observed_p2f_test_rows"] = len(p2_rows)
    atomic_json(output_dir / "input_audit.json", input_audit)

    task_summaries = {}
    all_coordinate_rows = []
    all_exact_rows = []
    all_similarity_rows = []
    for task_id in config["inputs"]["p3_tasks"]:
        p3_rows = load_p3_development(task_id, config)
        coordinate_pairs = interval_overlap_pairs(p3_rows, p2_rows)
        exact_pairs = exact_overlap_pairs(p3_rows, p2_rows)
        similarity_pairs, sketch_qc = high_similarity_pairs(p3_rows, p2_rows, config)
        scopes = fold_summary(p3_rows, len(p2_rows), coordinate_pairs, exact_pairs, similarity_pairs)

        for p3_index, p2_index, overlap_bp in coordinate_pairs:
            row = public_pair_row(task_id, p3_rows[p3_index], p2_rows[p2_index])
            row["overlap_bp"] = overlap_bp
            all_coordinate_rows.append(row)
        for p3_index, p2_index in exact_pairs:
            row = public_pair_row(task_id, p3_rows[p3_index], p2_rows[p2_index])
            row["canonical_sequence_sha256"] = p3_rows[p3_index]["canonical_sha256"]
            all_exact_rows.append(row)
        for pair in similarity_pairs:
            row = public_pair_row(task_id, p3_rows[pair["p3_index"]], p2_rows[pair["p2_index"]])
            row.update({key: pair[key] for key in ("shared_sketch_hashes", "estimated_jaccard", "identity", "coverage", "orientation", "shift")})
            all_similarity_rows.append(row)

        coordinate_trigger = max(scope["coordinate"]["fraction_of_p3_train"] for scope in scopes.values()) > float(config["coordinate_overlap"]["trigger_fraction_of_p3_train"])
        exact_trigger = max(scope["exact"]["fraction_of_p3_train"] for scope in scopes.values()) > float(config["exact_sequence_overlap"]["trigger_fraction_of_p3_train"])
        similarity_trigger = any(scope["high_similarity"]["unique_p3_rows"] > 0 for scope in scopes.values())
        task_summaries[task_id] = {
            "p3_union_train_rows": len(p3_rows),
            "p2f_test_rows": len(p2_rows),
            "scope_summaries": scopes,
            "pair_counts": {
                "coordinate": len(coordinate_pairs),
                "exact": len(exact_pairs),
                "high_similarity": len(similarity_pairs),
            },
            "minhash_screening_qc": sketch_qc,
            "triggers": {
                "coordinate_stratification": coordinate_trigger,
                "exact_overlap_rerun": exact_trigger,
                "high_similarity_rerun": similarity_trigger,
            },
        }

    coordinate_fields = ["task_id", "p3_sample_id", "p3_label", "p3_chromosome", "p3_start", "p3_end", "p2f_sample_id", "p2f_label", "p2f_chromosome", "p2f_start", "p2f_end", "overlap_bp"]
    exact_fields = coordinate_fields[:-1] + ["canonical_sequence_sha256"]
    similarity_fields = coordinate_fields[:-1] + ["shared_sketch_hashes", "estimated_jaccard", "identity", "coverage", "orientation", "shift"]
    atomic_csv_gz(output_dir / "coordinate_overlap_pairs.csv.gz", all_coordinate_rows, coordinate_fields)
    atomic_csv_gz(output_dir / "exact_sequence_overlap_pairs.csv.gz", all_exact_rows, exact_fields)
    atomic_csv_gz(output_dir / "high_similarity_pairs.csv.gz", all_similarity_rows, similarity_fields)

    overall = {
        "protocol_revision": config["protocol_revision"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "p2f_access_mode": "isolated_read_only",
        "p2f_predictions_or_attributions_accessed": False,
        "raw_sequences_persisted": False,
        "tasks": task_summaries,
        "overall_triggers": {
            "coordinate_stratification": any(item["triggers"]["coordinate_stratification"] for item in task_summaries.values()),
            "exact_overlap_rerun": any(item["triggers"]["exact_overlap_rerun"] for item in task_summaries.values()),
            "high_similarity_rerun": any(item["triggers"]["high_similarity_rerun"] for item in task_summaries.values()),
        },
        "primary_p3_results_replaced": False,
    }
    atomic_json(output_dir / "audit_summary.json", overall)
    completion = {
        "status": "completed_locked",
        "protocol_revision": config["protocol_revision"],
        "completed_at_utc": overall["completed_at_utc"],
        "audit_execution_count": 1,
        "p2f_test_rows_loaded": len(p2_rows),
        "predictions_or_attributions_accessed": False,
        "raw_sequences_persisted": False,
        "audit_summary_sha256": sha256_file(output_dir / "audit_summary.json"),
    }
    atomic_json(completion_path, completion)
    return overall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight_only == args.execute:
        raise SystemExit("choose exactly one of --preflight-only or --execute")
    config = load_config()
    output_dir = PROJECT_ROOT / config["reporting"]["output_dir"]
    if args.preflight_only:
        payload = preflight(config)
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(output_dir / "preflight.json", payload)
        print(json.dumps({"status": "preflight_passed", "checks": len(payload["checks"]), "test_rows_loaded": False}))
    else:
        payload = run_audit(config, output_dir)
        print(json.dumps({"status": "completed_locked", "overall_triggers": payload["overall_triggers"]}))


if __name__ == "__main__":
    main()

