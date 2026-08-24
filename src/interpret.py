from __future__ import annotations

import time

import numpy as np
import torch
from captum.attr import DeepLift, GradientShap, IntegratedGradients
from tqdm import tqdm

from .dna_utils import BASE_TO_INDEX, batch_one_hot_encode
from .training import DEVICE


@torch.inference_mode()
def ism_attribution(model, sequence: str, batch_size: int = 256, difference: str = "logit"):
    """Return (length, 4) reference-minus-mutant effects and position aggregates."""
    model.eval()
    original_index = np.array([BASE_TO_INDEX[b] for b in sequence], dtype=int)
    original_logit = model(batch_one_hot_encode([sequence]).to(DEVICE))[0]
    mutants, coordinates = [], []
    alphabet = "ACGT"
    for position, base_index in enumerate(original_index):
        for candidate_index, candidate in enumerate(alphabet):
            if candidate_index != base_index:
                mutants.append(sequence[:position] + candidate + sequence[position + 1:])
                coordinates.append((position, candidate_index))
    mutant_outputs = []
    for start in range(0, len(mutants), batch_size):
        logits = model(batch_one_hot_encode(mutants[start:start + batch_size]).to(DEVICE))
        if difference == "probability":
            logits = torch.sigmoid(logits)
        mutant_outputs.extend(logits.cpu().numpy())
    reference = torch.sigmoid(original_logit).item() if difference == "probability" else original_logit.item()
    matrix = np.zeros((len(sequence), 4), dtype=np.float32)
    for (position, candidate_index), mutant_output in zip(coordinates, mutant_outputs):
        matrix[position, candidate_index] = reference - float(mutant_output)
    mask = np.ones_like(matrix, dtype=bool); mask[np.arange(len(sequence)), original_index] = False
    signed = np.sum(np.where(mask, matrix, 0.0), axis=1) / 3.0
    absolute = np.sum(np.where(mask, np.abs(matrix), 0.0), axis=1) / 3.0
    return matrix, signed.astype(np.float32), absolute.astype(np.float32)


def run_ism_for_sequences(model, sequences: list[str], batch_size: int, difference: str, description: str):
    matrices, signed, absolute = [], [], []
    started = time.perf_counter()
    for sequence in tqdm(sequences, desc=description, leave=False):
        m, s, a = ism_attribution(model, sequence, batch_size, difference)
        matrices.append(m); signed.append(s); absolute.append(a)
    return (np.stack(matrices), np.stack(signed), np.stack(absolute), time.perf_counter() - started)


def _captum_method(model, method: str):
    if method == "integrated_gradients":
        return IntegratedGradients(model)
    if method == "deeplift":
        return DeepLift(model)
    if method == "gradient_shap":
        return GradientShap(model)
    raise ValueError(f"Unsupported Captum attribution method: {method}")


def run_captum_for_sequences(
    model,
    sequences: list[str],
    method: str,
    batch_size: int,
    n_steps: int = 32,
    n_samples: int = 16,
    random_seed: int = 20260822,
):
    """Return nucleotide, signed-position and absolute-position attributions.

    The zero baseline is individually RC invariant. GradientSHAP uses a two-point
    RC-compatible baseline distribution consisting of zero and uniform 0.25 DNA.
    """
    model.eval()
    attributor = _captum_method(model, method)
    matrices = []
    started = time.perf_counter()
    for start in tqdm(
        range(0, len(sequences), batch_size),
        desc=f"{method} attribution",
        leave=False,
    ):
        inputs = batch_one_hot_encode(sequences[start : start + batch_size]).to(DEVICE)
        zero = torch.zeros_like(inputs)
        torch.manual_seed(random_seed + start)
        if method == "integrated_gradients":
            attribution = attributor.attribute(
                inputs,
                baselines=zero,
                n_steps=int(n_steps),
                method="gausslegendre",
                internal_batch_size=max(len(inputs), len(inputs) * min(int(n_steps), 8)),
            )
        elif method == "deeplift":
            attribution = attributor.attribute(inputs, baselines=zero)
        else:
            baselines = torch.stack(
                (
                    torch.zeros_like(inputs[0]),
                    torch.full_like(inputs[0], 0.25),
                )
            )
            attribution = attributor.attribute(
                inputs,
                baselines=baselines,
                n_samples=int(n_samples),
                stdevs=0.0,
            )
        matrices.append(attribution.detach().cpu().permute(0, 2, 1).numpy())
    matrix = np.concatenate(matrices, axis=0).astype(np.float32)
    signed = matrix.sum(axis=-1).astype(np.float32)
    absolute = np.abs(matrix).sum(axis=-1).astype(np.float32)
    return matrix, signed, absolute, time.perf_counter() - started
