from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .data import validate_splits
from .dna_utils import reverse_complement


def _gc_background(length: int, gc_fraction: float, rng: np.random.Generator) -> str:
    probabilities = [(1.0 - gc_fraction) / 2, gc_fraction / 2,
                     gc_fraction / 2, (1.0 - gc_fraction) / 2]
    return "".join(rng.choice(list("ACGT"), size=length, p=probabilities).tolist())


def _mutate_motif(motif: str, mutation_count: int, rng: np.random.Generator) -> str:
    bases = list(motif)
    if mutation_count <= 0:
        return motif
    for position in rng.choice(len(bases), size=min(mutation_count, len(bases)), replace=False):
        choices = [base for base in "ACGT" if base != bases[position]]
        bases[position] = str(rng.choice(choices))
    return "".join(bases)


def _sample_pwm_motif(consensus: str, match_probability: float,
                      rng: np.random.Generator) -> str:
    """Sample from a simple consensus-centered PWM with equal alternate mass."""
    sampled = []
    for consensus_base in consensus:
        probabilities = np.full(4, (1.0 - match_probability) / 3.0)
        probabilities["ACGT".index(consensus_base)] = match_probability
        sampled.append(str(rng.choice(list("ACGT"), p=probabilities)))
    return "".join(sampled)


def _overlaps(start: int, length: int, occupied: list[tuple[int, int]]) -> bool:
    end = start + length
    return any(start < other_end and end > other_start for other_start, other_end in occupied)


def _choose_start(
    sequence_length: int,
    item_length: int,
    occupied: list[tuple[int, int]],
    rng: np.random.Generator,
    low: int = 0,
    high: int | None = None,
) -> int:
    high = sequence_length - item_length + 1 if high is None else min(high, sequence_length - item_length + 1)
    candidates = [start for start in range(low, high) if not _overlaps(start, item_length, occupied)]
    if not candidates:
        raise RuntimeError("Could not find a non-overlapping insertion site")
    return int(rng.choice(candidates))


def _insert(sequence: str, item: str, start: int) -> str:
    return sequence[:start] + item + sequence[start + len(item):]


def _split_value(mapping: dict, split: str, label: int, default: float) -> float:
    split_values = mapping.get(split, {})
    return float(split_values.get("positive" if label else "negative", default))


def generate_phase15_data(config: dict) -> tuple[pd.DataFrame, dict]:
    """Generate difficult synthetic data with explicit causal and nuisance regions.

    The clean label is determined only by whether a causal motif instance was
    planted. Decoys, GC composition, and the optional edge shortcut are nuisance
    variables whose correlations can change by split.
    """
    if config.get("regime") == "motif_grammar":
        return generate_motif_grammar_data(config)
    length = int(config["sequence_length"])
    causal_motif = str(config["causal_motif"]).upper()
    decoys = [str(item).upper() for item in config.get("decoy_motifs", [])]
    rng = np.random.default_rng(int(config["data_seed"]))
    mutation_probabilities = np.asarray(
        config.get("causal_mutation_probabilities", [1.0]), dtype=float
    )
    mutation_probabilities /= mutation_probabilities.sum()
    gc_mapping = config.get("gc_by_split", {})
    label_noise = config.get("label_noise_by_split", {})
    decoy_probability = config.get("decoy_probability_by_label", {"positive": 0.5, "negative": 1.0})
    shortcut = config.get("shortcut", {})
    count_probabilities = np.asarray(
        config.get("causal_count_probabilities_positive", [0.0, 1.0]), dtype=float
    )
    count_probabilities /= count_probabilities.sum()
    pwm_matches = np.asarray(config.get("pwm_match_probabilities", []), dtype=float)
    pwm_weights = np.asarray(config.get("pwm_strength_probabilities", []), dtype=float)
    if len(pwm_matches):
        if len(pwm_weights) != len(pwm_matches):
            raise ValueError("pwm_strength_probabilities must match pwm_match_probabilities")
        pwm_weights /= pwm_weights.sum()

    records: list[dict] = []
    seen_canonical: set[str] = set()
    duplicate_attempts = 0
    rejected_accidental_causal = 0
    for split, size_key in (("train", "train_size"), ("validation", "validation_size"), ("test", "test_size")):
        size = int(config[size_key])
        clean_labels = np.array([1] * (size // 2) + [0] * (size - size // 2))
        rng.shuffle(clean_labels)
        for local_index, clean_label in enumerate(clean_labels):
            while True:
                gc_fraction = _split_value(gc_mapping, split, int(clean_label), 0.5)
                sequence = _gc_background(length, gc_fraction, rng)
                occupied: list[tuple[int, int]] = []
                causal_start = causal_end = None
                causal_orientation = None
                causal_instance = None
                mutation_count = None
                decoy_start = decoy_end = None
                decoy_instance = None
                shortcut_start = shortcut_end = None
                shortcut_instance = None
                causal_intervals: list[list[int]] = []
                causal_instances: list[str] = []
                causal_orientations: list[str] = []
                causal_mutation_counts: list[int] = []
                causal_pwm_matches: list[float | None] = []
                decoy_intervals: list[list[int]] = []
                decoy_instances: list[str] = []

                shortcut_rate = 0.0
                if shortcut.get("enabled", False):
                    shortcut_rate = _split_value(
                        shortcut.get("probability_by_split", {}), split, int(clean_label), 0.0
                    )
                if shortcut_rate and rng.random() < shortcut_rate:
                    shortcut_instance = str(shortcut["pattern"]).upper()
                    edge_width = int(shortcut.get("edge_width", 24))
                    shortcut_start = _choose_start(
                        length, len(shortcut_instance), occupied, rng,
                        low=0, high=max(1, edge_width - len(shortcut_instance) + 1),
                    )
                    shortcut_end = shortcut_start + len(shortcut_instance)
                    occupied.append((shortcut_start, shortcut_end))
                    sequence = _insert(sequence, shortcut_instance, shortcut_start)

                if clean_label:
                    causal_count = int(rng.choice(len(count_probabilities), p=count_probabilities))
                    if causal_count < 1:
                        raise ValueError("Positive samples require at least one causal motif")
                    for _ in range(causal_count):
                        if len(pwm_matches):
                            pwm_match = float(rng.choice(pwm_matches, p=pwm_weights))
                            current_instance = _sample_pwm_motif(causal_motif, pwm_match, rng)
                            current_mutations = sum(a != b for a, b in zip(current_instance, causal_motif))
                        else:
                            pwm_match = None
                            current_mutations = int(rng.choice(len(mutation_probabilities), p=mutation_probabilities))
                            current_instance = _mutate_motif(causal_motif, current_mutations, rng)
                        current_orientation = "forward" if rng.random() < 0.5 else "reverse_complement"
                        inserted = current_instance if current_orientation == "forward" else reverse_complement(current_instance)
                        current_start = _choose_start(length, len(inserted), occupied, rng)
                        current_end = current_start + len(inserted)
                        occupied.append((current_start, current_end))
                        sequence = _insert(sequence, inserted, current_start)
                        causal_intervals.append([current_start, current_end])
                        causal_instances.append(current_instance)
                        causal_orientations.append(current_orientation)
                        causal_mutation_counts.append(current_mutations)
                        causal_pwm_matches.append(pwm_match)
                    causal_start, causal_end = causal_intervals[0]
                    causal_instance = causal_instances[0]
                    causal_orientation = causal_orientations[0]
                    mutation_count = causal_mutation_counts[0]

                if "decoy_count_probabilities_by_label" in config:
                    raw_counts = config["decoy_count_probabilities_by_label"]["positive" if clean_label else "negative"]
                    decoy_count_weights = np.asarray(raw_counts, dtype=float)
                    decoy_count_weights /= decoy_count_weights.sum()
                    decoy_count = int(rng.choice(len(decoy_count_weights), p=decoy_count_weights))
                else:
                    decoy_rate = float(decoy_probability.get("positive" if clean_label else "negative", 0.0))
                    decoy_count = int(bool(decoys) and rng.random() < decoy_rate)
                for _ in range(decoy_count):
                    current_decoy = str(rng.choice(decoys))
                    if rng.random() < 0.5:
                        current_decoy = reverse_complement(current_decoy)
                    current_start = _choose_start(length, len(current_decoy), occupied, rng)
                    current_end = current_start + len(current_decoy)
                    occupied.append((current_start, current_end))
                    sequence = _insert(sequence, current_decoy, current_start)
                    decoy_intervals.append([current_start, current_end])
                    decoy_instances.append(current_decoy)
                if decoy_intervals:
                    decoy_start, decoy_end = decoy_intervals[0]
                    decoy_instance = decoy_instances[0]

                if not clean_label and (
                    causal_motif in sequence or reverse_complement(causal_motif) in sequence
                ):
                    rejected_accidental_causal += 1
                    continue
                canonical = min(sequence, reverse_complement(sequence))
                if canonical in seen_canonical:
                    duplicate_attempts += 1
                    continue
                seen_canonical.add(canonical)
                break

            observed_label = int(clean_label)
            noise_rate = float(label_noise.get(split, 0.0))
            label_flipped = bool(rng.random() < noise_rate)
            if label_flipped:
                observed_label = 1 - observed_label
            records.append({
                "sample_id": f"{split}_{local_index:06d}",
                "sequence": sequence,
                "label": observed_label,
                "clean_label": int(clean_label),
                "label_flipped": label_flipped,
                "split": split,
                "background_gc_target": gc_fraction,
                "causal_start": causal_start,
                "causal_end": causal_end,
                "causal_orientation": causal_orientation,
                "causal_instance": causal_instance,
                "causal_mutations": mutation_count,
                "causal_count": len(causal_intervals),
                "causal_instances": json.dumps(causal_instances),
                "causal_orientations": json.dumps(causal_orientations),
                "causal_mutation_counts": json.dumps(causal_mutation_counts),
                "causal_pwm_match_probabilities": json.dumps(causal_pwm_matches),
                "decoy_start": decoy_start,
                "decoy_end": decoy_end,
                "decoy_instance": decoy_instance,
                "decoy_count": len(decoy_intervals),
                "decoy_instances": json.dumps(decoy_instances),
                "shortcut_start": shortcut_start,
                "shortcut_end": shortcut_end,
                "shortcut_instance": shortcut_instance,
                "causal_intervals": json.dumps(causal_intervals),
                "decoy_intervals": json.dumps(decoy_intervals),
                "shortcut_intervals": json.dumps([] if shortcut_start is None else [[shortcut_start, shortcut_end]]),
            })

    frame = pd.DataFrame(records)
    report = validate_splits(frame)
    report.update({
        "generation_duplicate_attempts_rejected": duplicate_attempts,
        "accidental_exact_causal_negatives_rejected": rejected_accidental_causal,
        "observed_label_flips": int(frame["label_flipped"].sum()),
        "clean_class_counts": {
            f"{split}/label_{label}": int(count)
            for (split, label), count in frame.groupby(["split", "clean_label"]).size().items()
        },
        "shortcut_counts": {
            f"{split}/label_{label}": int(count)
            for (split, label), count in frame.assign(has_shortcut=frame["shortcut_start"].notna())
            .groupby(["split", "clean_label"])["has_shortcut"].sum().items()
        },
    })
    return frame, report


def generate_motif_grammar_data(config: dict) -> tuple[pd.DataFrame, dict]:
    """Generate an RC-invariant two-motif spacing/orientation grammar task."""
    length = int(config["sequence_length"])
    motif_a = str(config["motif_a"]).upper()
    motif_b = str(config["motif_b"]).upper()
    positive_gap = tuple(int(value) for value in config.get("positive_gap", [10, 24]))
    negative_gap = tuple(int(value) for value in config.get("negative_gap", [40, 64]))
    match_probabilities = np.asarray(config.get("pwm_match_probabilities", [1.0]), float)
    strength_weights = np.asarray(config.get("pwm_strength_probabilities", [1.0]), float)
    strength_weights /= strength_weights.sum()
    label_noise = config.get("label_noise_by_split", {})
    rng = np.random.default_rng(int(config["data_seed"]))
    seen_canonical: set[str] = set()
    records: list[dict] = []
    duplicate_attempts = 0

    for split, size_key in (("train", "train_size"), ("validation", "validation_size"), ("test", "test_size")):
        size = int(config[size_key])
        labels = np.array([1] * (size // 2) + [0] * (size - size // 2))
        rng.shuffle(labels)
        for local_index, clean_label in enumerate(labels):
            while True:
                sequence = _gc_background(length, float(config.get("background_gc", 0.5)), rng)
                if clean_label:
                    gap = int(rng.integers(positive_gap[0], positive_gap[1] + 1))
                    valid_orientation = True
                else:
                    use_wrong_gap = bool(rng.random() < 0.5)
                    gap = int(rng.integers(*(negative_gap[0], negative_gap[1] + 1))) if use_wrong_gap else int(rng.integers(positive_gap[0], positive_gap[1] + 1))
                    valid_orientation = use_wrong_gap

                match_a = float(rng.choice(match_probabilities, p=strength_weights))
                match_b = float(rng.choice(match_probabilities, p=strength_weights))
                instance_a = _sample_pwm_motif(motif_a, match_a, rng)
                instance_b = _sample_pwm_motif(motif_b, match_b, rng)
                rc_representation = bool(rng.random() < 0.5)
                if not rc_representation:
                    left_name, left_instance = "A", instance_a
                    right_name, right_instance = "B", reverse_complement(instance_b) if valid_orientation else instance_b
                else:
                    left_name, left_instance = "B", instance_b
                    right_name, right_instance = "A", reverse_complement(instance_a) if valid_orientation else instance_a
                total_width = len(left_instance) + gap + len(right_instance)
                left_start = int(rng.integers(0, length - total_width + 1))
                left_end = left_start + len(left_instance)
                right_start = left_end + gap
                right_end = right_start + len(right_instance)
                sequence = _insert(sequence, left_instance, left_start)
                sequence = _insert(sequence, right_instance, right_start)
                canonical = min(sequence, reverse_complement(sequence))
                if canonical in seen_canonical:
                    duplicate_attempts += 1
                    continue
                seen_canonical.add(canonical)
                break

            observed_label = int(clean_label)
            label_flipped = bool(rng.random() < float(label_noise.get(split, 0.0)))
            if label_flipped:
                observed_label = 1 - observed_label
            pair_intervals = [[left_start, left_end], [right_start, right_end]]
            causal_intervals = pair_intervals if clean_label else []
            decoy_intervals = [] if clean_label else pair_intervals
            records.append({
                "sample_id": f"{split}_{local_index:06d}", "sequence": sequence,
                "label": observed_label, "clean_label": int(clean_label),
                "label_flipped": label_flipped, "split": split,
                "background_gc_target": float(config.get("background_gc", 0.5)),
                "causal_start": left_start if clean_label else None,
                "causal_end": left_end if clean_label else None,
                "causal_orientation": "convergent_pair" if clean_label else None,
                "causal_instance": f"{left_instance}|{right_instance}" if clean_label else None,
                "causal_mutations": (sum(a != b for a, b in zip(instance_a, motif_a)) +
                                     sum(a != b for a, b in zip(instance_b, motif_b))) if clean_label else None,
                "causal_count": 2 if clean_label else 0,
                "causal_instances": json.dumps([instance_a, instance_b] if clean_label else []),
                "causal_orientations": json.dumps(["convergent_pair"] if clean_label else []),
                "causal_mutation_counts": json.dumps([]),
                "causal_pwm_match_probabilities": json.dumps([match_a, match_b] if clean_label else []),
                "decoy_start": left_start if not clean_label else None,
                "decoy_end": left_end if not clean_label else None,
                "decoy_instance": f"{left_instance}|{right_instance}" if not clean_label else None,
                "decoy_count": 2 if not clean_label else 0,
                "decoy_instances": json.dumps([instance_a, instance_b] if not clean_label else []),
                "shortcut_start": None, "shortcut_end": None, "shortcut_instance": None,
                "causal_intervals": json.dumps(causal_intervals),
                "decoy_intervals": json.dumps(decoy_intervals),
                "shortcut_intervals": json.dumps([]),
                "grammar_gap": gap, "grammar_valid": int(clean_label),
                "grammar_left_motif": left_name,
            })

    frame = pd.DataFrame(records)
    report = validate_splits(frame)
    report.update({
        "generation_duplicate_attempts_rejected": duplicate_attempts,
        "observed_label_flips": int(frame.label_flipped.sum()),
        "grammar_positive_gap": list(positive_gap),
        "grammar_negative_gap": list(negative_gap),
        "rc_invariant_grammar": True,
    })
    return frame, report
