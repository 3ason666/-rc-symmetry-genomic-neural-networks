from __future__ import annotations

import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import SequenceDataset
from .dna_utils import batch_one_hot_encode
from .metrics import classification_metrics
from .models import PostHocConjoined, build_model, parameter_count

DEVICE = torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


@torch.inference_mode()
def predict_sequences(model, sequences: list[str], batch_size: int = 256) -> np.ndarray:
    model.eval(); outputs = []
    for start in range(0, len(sequences), batch_size):
        logits = model(batch_one_hot_encode(sequences[start:start + batch_size]).to(DEVICE))
        outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.array([])


@torch.inference_mode()
def evaluate(model, loader, threshold: float) -> dict:
    model.eval(); criterion = nn.BCEWithLogitsLoss(reduction="sum")
    labels, probabilities, total_loss = [], [], 0.0
    for inputs, targets in loader:
        logits = model(inputs.to(DEVICE)); total_loss += criterion(logits, targets.to(DEVICE)).item()
        labels.extend(targets.numpy()); probabilities.extend(torch.sigmoid(logits).cpu().numpy())
    return classification_metrics(labels, probabilities, total_loss / len(labels), threshold)


def train_model(model_type: str, seed: int, config: dict, data: pd.DataFrame, run_dir: Path):
    set_seed(seed); started = time.perf_counter(); run_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = config["training"]
    mode = {"CNN-Raw": "raw", "CNN-Aug": "augment", "CNN-Pair": "pair",
            "CNN-PostHoc": "augment", "CNN-RCPS": "raw"}[model_type]
    architecture = "rcps" if model_type == "CNN-RCPS" else "standard"
    train_ds = SequenceDataset(data[data.split == "train"], mode, train_cfg["augmentation_probability"])
    valid_ds = SequenceDataset(data[data.split == "validation"], "raw")
    test_ds = SequenceDataset(data[data.split == "test"], "raw")
    generator = torch.Generator().manual_seed(seed)
    loader_args = dict(batch_size=int(train_cfg["batch_size"]), num_workers=0)
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_args)
    valid_loader = DataLoader(valid_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)
    model = build_model(config["model"], architecture=architecture).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["learning_rate"]),
                                 weight_decay=float(train_cfg["weight_decay"]))
    criterion = nn.BCEWithLogitsLoss(); best_loss, best_epoch, stale, history = np.inf, 0, 0, []
    checkpoint = run_dir / "best_checkpoint.pt"
    for epoch in range(1, int(train_cfg["max_epochs"]) + 1):
        model.train(); train_loss, count = 0.0, 0
        for inputs, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs.to(DEVICE)); loss = criterion(logits, targets.to(DEVICE))
            loss.backward(); optimizer.step(); train_loss += loss.item() * len(targets); count += len(targets)
        validation_model = PostHocConjoined(model) if model_type == "CNN-PostHoc" else model
        val_metrics = evaluate(validation_model, valid_loader, float(train_cfg["threshold"]))
        history.append({"epoch": epoch, "train_loss": train_loss / count,
                        **{f"validation_{k}": v for k, v in val_metrics.items()}})
        if val_metrics["loss"] < best_loss - 1e-8:
            best_loss, best_epoch, stale = val_metrics["loss"], epoch, 0
            torch.save({"model_state": model.state_dict(), "model_config": config["model"],
                        "architecture": architecture, "model_type": model_type,
                        "seed": seed, "best_epoch": epoch}, checkpoint)
        else:
            stale += 1
            if stale >= int(train_cfg["patience"]): break
    saved = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    reloaded = build_model(saved["model_config"], architecture=saved.get("architecture", "standard")).to(DEVICE)
    reloaded.load_state_dict(saved["model_state"])
    if model_type == "CNN-PostHoc":
        reloaded = PostHocConjoined(reloaded)
    validation_metrics = evaluate(reloaded, valid_loader, float(train_cfg["threshold"]))
    test_metrics = evaluate(reloaded, test_loader, float(train_cfg["threshold"]))
    elapsed = time.perf_counter() - started
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False, encoding="utf-8")
    metadata = {"model_type": model_type, "training_strategy": mode,
                "architecture": architecture, "seed": seed,
                "best_epoch": best_epoch, "parameter_count": parameter_count(reloaded),
                "training_seconds": elapsed, "python": platform.python_version(),
                "pytorch": torch.__version__, "device": str(DEVICE),
                "validation_metrics": validation_metrics, "test_metrics": test_metrics}
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return reloaded, metadata


def train_model_validation_only(
    model_type: str,
    seed: int,
    config: dict,
    data: pd.DataFrame,
    run_dir: Path,
):
    """Train and select a checkpoint without constructing or evaluating a test loader.

    This separate entry point is used by the frozen Phase 2 pilot so accidental
    test-set evaluation cannot be introduced through the Phase 1 training path.
    """
    observed_splits = set(data["split"].astype(str).unique())
    if observed_splits != {"train", "validation"}:
        raise ValueError(
            "validation-only training requires exactly train and validation rows; "
            f"observed={sorted(observed_splits)}"
        )

    set_seed(seed)
    started = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = config["training"]
    mode = {
        "CNN-Raw": "raw",
        "CNN-Aug": "augment",
        "CNN-Pair": "pair",
        "CNN-RCPS": "raw",
        "Transformer-Raw": "raw",
        "Transformer-Aug-None": "augment",
        "Transformer-Aug-Absolute": "augment",
        "Transformer-Aug-Relative": "augment",
    }[model_type]
    architecture = {
        "CNN-RCPS": "rcps",
        "Transformer-Raw": "transformer_absolute",
        "Transformer-Aug-None": "transformer_none",
        "Transformer-Aug-Absolute": "transformer_absolute",
        "Transformer-Aug-Relative": "transformer_relative",
    }.get(model_type, "standard")
    train_ds = SequenceDataset(data[data.split == "train"], mode, train_cfg["augmentation_probability"])
    valid_ds = SequenceDataset(data[data.split == "validation"], "raw")
    generator = torch.Generator().manual_seed(seed)
    loader_args = {
        "batch_size": int(train_cfg["batch_size"]),
        "num_workers": int(train_cfg.get("num_workers", 0)),
    }
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_args)
    valid_loader = DataLoader(valid_ds, shuffle=False, **loader_args)
    model = build_model(config["model"], architecture=architecture).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss()
    best_loss, best_epoch, stale, history = np.inf, 0, 0, []
    checkpoint = run_dir / "best_checkpoint.pt"

    for epoch in range(1, int(train_cfg["max_epochs"]) + 1):
        model.train()
        train_loss, count = 0.0, 0
        for inputs, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs.to(DEVICE))
            loss = criterion(logits, targets.to(DEVICE))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(targets)
            count += len(targets)
        val_metrics = evaluate(model, valid_loader, float(train_cfg["threshold"]))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / count,
                **{f"validation_{key}": value for key, value in val_metrics.items()},
            }
        )
        if val_metrics["loss"] < best_loss - 1e-8:
            best_loss, best_epoch, stale = val_metrics["loss"], epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": config["model"],
                    "architecture": architecture,
                    "model_type": model_type,
                    "seed": seed,
                    "best_epoch": epoch,
                    "selection_split": "validation",
                    "test_evaluated": False,
                },
                checkpoint,
            )
        else:
            stale += 1
            if stale >= int(train_cfg["patience"]):
                break

    saved = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    reloaded = build_model(saved["model_config"], architecture=architecture).to(DEVICE)
    reloaded.load_state_dict(saved["model_state"])
    validation_metrics = evaluate(reloaded, valid_loader, float(train_cfg["threshold"]))
    elapsed = time.perf_counter() - started
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False, encoding="utf-8")
    metadata = {
        "model_type": model_type,
        "training_strategy": mode,
        "architecture": architecture,
        "seed": seed,
        "best_epoch": best_epoch,
        "parameter_count": parameter_count(reloaded),
        "training_seconds": elapsed,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "device": str(DEVICE),
        "selection_split": "validation",
        "test_evaluated": False,
        "validation_metrics": validation_metrics,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return reloaded, metadata
