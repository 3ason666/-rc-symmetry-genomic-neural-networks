from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

DNA_ALPHABET = "ACGT"
BASE_TO_INDEX = {base: i for i, base in enumerate(DNA_ALPHABET)}
COMPLEMENT_INDEX = np.array([3, 2, 1, 0])
_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def validate_dna_sequence(sequence: str) -> None:
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("DNA sequence must be a non-empty string")
    invalid = sorted(set(sequence.upper()) - set(DNA_ALPHABET))
    if invalid:
        raise ValueError(f"Invalid DNA character(s): {', '.join(invalid)}")


def reverse_complement(sequence: str) -> str:
    validate_dna_sequence(sequence)
    return sequence.upper().translate(_COMPLEMENT)[::-1]


def one_hot_encode(sequence: str) -> torch.Tensor:
    """Return float32 one-hot tensor shaped (4, length), channels A,C,G,T."""
    validate_dna_sequence(sequence)
    indices = torch.tensor([BASE_TO_INDEX[b] for b in sequence.upper()], dtype=torch.long)
    return torch.nn.functional.one_hot(indices, num_classes=4).T.to(torch.float32)


def batch_one_hot_encode(sequences: Sequence[str]) -> torch.Tensor:
    if len(sequences) == 0:
        raise ValueError("Cannot encode an empty sequence batch")
    lengths = {len(s) for s in sequences}
    if len(lengths) != 1:
        raise ValueError("All DNA sequences in a batch must have equal length")
    return torch.stack([one_hot_encode(s) for s in sequences])


def align_rc_position_attribution(attribution):
    """Map position-only attribution on RC(x) back to x coordinates."""
    if isinstance(attribution, torch.Tensor):
        return torch.flip(attribution, dims=(-1,))
    return np.flip(np.asarray(attribution), axis=-1).copy()


def align_rc_full_attribution(attribution):
    """Map (..., position, A/C/G/T) RC attribution to original coordinates."""
    if isinstance(attribution, torch.Tensor):
        if attribution.shape[-1] != 4:
            raise ValueError("Full attribution must have four nucleotide channels")
        return torch.flip(attribution, dims=(-2,)).index_select(
            -1, torch.tensor([3, 2, 1, 0], device=attribution.device)
        )
    array = np.asarray(attribution)
    if array.shape[-1] != 4:
        raise ValueError("Full attribution must have four nucleotide channels")
    return np.flip(array, axis=-2)[..., COMPLEMENT_INDEX].copy()

