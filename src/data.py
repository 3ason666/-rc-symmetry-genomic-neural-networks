from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .dna_utils import batch_one_hot_encode, one_hot_encode, reverse_complement


def _random_dna(length: int, rng: np.random.Generator) -> str:
    return "".join(rng.choice(list("ACGT"), size=length).tolist())


def generate_synthetic_data(config: dict) -> tuple[pd.DataFrame, dict]:
    length, motif = int(config["sequence_length"]), str(config["motif"]).upper()
    motif_rc = reverse_complement(motif)
    rng = np.random.default_rng(int(config["data_seed"]))
    records, seen_canonical = [], set()
    duplicate_attempts = 0
    for split, size_key in (("train", "train_size"), ("validation", "validation_size"), ("test", "test_size")):
        size = int(config[size_key])
        labels = np.array([1] * (size // 2) + [0] * (size - size // 2))
        rng.shuffle(labels)
        for local_i, label in enumerate(labels):
            while True:
                background = _random_dna(length, rng)
                if label:
                    orientation = "forward" if rng.random() < 0.5 else "reverse_complement"
                    inserted = motif if orientation == "forward" else motif_rc
                    start = int(rng.integers(0, length - len(motif) + 1))
                    sequence = background[:start] + inserted + background[start + len(motif):]
                    end = start + len(motif)
                else:
                    sequence = background
                    start = end = None
                    orientation = None
                    if motif in sequence or motif_rc in sequence:
                        continue
                canonical = min(sequence, reverse_complement(sequence))
                if canonical in seen_canonical:
                    duplicate_attempts += 1
                    continue
                seen_canonical.add(canonical)
                break
            records.append({
                "sample_id": f"{split}_{local_i:06d}", "sequence": sequence,
                "label": int(label), "motif_start": start, "motif_end": end,
                "motif_orientation": orientation, "split": split,
            })
    frame = pd.DataFrame(records)
    report = validate_splits(frame)
    report["generation_duplicate_attempts_rejected"] = duplicate_attempts
    return frame, report


def validate_splits(frame: pd.DataFrame) -> dict:
    duplicate_sequences = int(frame["sequence"].duplicated().sum())
    canonical_to_split: dict[str, str] = {}
    rc_leaks = []
    for row in frame.itertuples():
        canonical = min(row.sequence, reverse_complement(row.sequence))
        prior = canonical_to_split.get(canonical)
        if prior is not None and prior != row.split:
            rc_leaks.append((row.sample_id, prior, row.split))
        canonical_to_split[canonical] = row.split
    if duplicate_sequences or rc_leaks:
        raise ValueError(f"Split leakage detected: duplicates={duplicate_sequences}, rc_leaks={len(rc_leaks)}")
    return {"duplicate_sequences": duplicate_sequences, "cross_split_rc_leaks": len(rc_leaks),
            "total_sequences": int(len(frame))}


class SequenceDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, mode: str = "raw", augmentation_probability: float = 0.5):
        self.records = list(frame[["sequence", "label"]].itertuples(index=False, name=None))
        self.mode = mode
        self.augmentation_probability = augmentation_probability
        if mode == "pair":
            self.records = [item for record in self.records for item in (record, (reverse_complement(record[0]), record[1]))]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        sequence, label = self.records[index]
        if self.mode == "augment" and random.random() < self.augmentation_probability:
            sequence = reverse_complement(sequence)
        return one_hot_encode(sequence), torch.tensor(label, dtype=torch.float32)


def save_dataset(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8")
