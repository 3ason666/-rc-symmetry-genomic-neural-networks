from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dna_utils import batch_one_hot_encode
from src.models import build_model
from src.training import DEVICE, train_model_validation_only


CONFIG_PATH = PROJECT_ROOT / "configs" / "phase3a_external_replication.yaml"
MANIFEST_PATH = PROJECT_ROOT / "metadata" / "phase3b_training_manifest.json"
FINAL_GATE_PATH = PROJECT_ROOT / "results" / "phase3a_external_selection" / "phase3a_final_gate.json"
OUTPUT = PROJECT_ROOT / "results" / "phase3b_development_training"
TASKS = ("p3_gata1_fetal", "p3_ctcf_gm12878")
MODELS = ("CNN-Raw", "CNN-Aug", "CNN-RCPS")
SEEDS = (42, 123, 2026)
FOLDS = (1, 2, 3)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def validate_preflight() -> tuple[dict, dict, dict]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    gate = json.loads(FINAL_GATE_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_for_p3b_training" or not manifest.get("formal_training_allowed"):
        raise ValueError("P3B training manifest is not enabled")
    if gate.get("status") != "passed" or not gate.get("formal_training_allowed"):
        raise ValueError("P3A final gate did not authorize P3B training")
    if manifest.get("test_unsealing_allowed") is not False:
        raise ValueError("P3B development training requires a sealed test set")
    if sha256(FINAL_GATE_PATH) != manifest["gate_sha256"]:
        raise ValueError("P3A gate hash changed after the P3B manifest freeze")
    if sha256(CONFIG_PATH) != manifest["config_sha256"]:
        raise ValueError("Phase 3 config hash changed after the P3B manifest freeze")
    protocol_path = PROJECT_ROOT / config["protocol_path"]
    if sha256(protocol_path) != manifest["protocol_sha256"]:
        raise ValueError("Phase 3 protocol hash changed after the P3B manifest freeze")
    if tuple(manifest["models"]) != MODELS or tuple(manifest["seeds"]) != SEEDS or tuple(manifest["folds"]) != FOLDS:
        raise ValueError("P3B model, seed, or fold grid changed")
    return config, manifest, gate


def model_config(config: dict, *, smoke: bool) -> dict:
    models = config["models"]
    training = {
        "batch_size": int(models["batch_size"]),
        "max_epochs": 1 if smoke else int(models["max_epochs"]),
        "patience": 1 if smoke else int(models["patience"]),
        "learning_rate": float(models["learning_rate"]),
        "weight_decay": float(models["weight_decay"]),
        "augmentation_probability": float(models["augmentation_probability"]),
        "threshold": float(config["endpoints"]["prediction_threshold"]),
        "num_workers": 0,
    }
    return {
        "model": {
            "conv_channels": int(models["conv_channels"]),
            "kernel_size": int(models["kernel_size"]),
            "second_conv": bool(models["second_conv"]),
            "second_conv_channels": int(models["second_conv_channels"]),
            "second_kernel_size": int(models["second_kernel_size"]),
        },
        "training": training,
    }


def load_development(task_id: str, manifest: dict) -> pd.DataFrame:
    task = manifest["tasks"][task_id]
    path = PROJECT_ROOT / task["development_path"]
    if "sealed_test" in path.name.lower():
        raise ValueError("development path unexpectedly points at a sealed test file")
    if sha256(path) != task["development_sha256"]:
        raise ValueError(f"development dataset hash changed: {task_id}")
    frame = pd.read_csv(path)
    if set(frame["fixed_partition"].astype(str)) != {"development"}:
        raise ValueError(f"{task_id} development file contains test rows")
    if len(frame) != int(task["development_rows"]):
        raise ValueError(f"{task_id} development row count changed")
    return frame


def fold_view(frame: pd.DataFrame, fold: int, *, smoke: bool) -> pd.DataFrame:
    column = f"fold{fold}_partition"
    view = frame.copy()
    view["split"] = view[column].astype(str)
    if set(view["split"]) != {"train", "validation"}:
        raise ValueError(f"fold {fold} does not contain exactly train and validation rows")
    if smoke:
        selected = []
        for split in ("train", "validation"):
            for label in (0, 1):
                selected.append(
                    view[(view.split == split) & (view.label == label)]
                    .sort_values("sample_id", kind="mergesort")
                    .head(128)
                )
        view = pd.concat(selected, ignore_index=True)
    return view


def checkpoint_reload_max_error(checkpoint: Path, sequences: list[str]) -> float:
    saved = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    first = build_model(saved["model_config"], architecture=saved["architecture"]).to(DEVICE)
    second = build_model(saved["model_config"], architecture=saved["architecture"]).to(DEVICE)
    first.load_state_dict(saved["model_state"])
    second.load_state_dict(saved["model_state"])
    first.eval(); second.eval()
    inputs = batch_one_hot_encode(sequences).to(DEVICE)
    with torch.inference_mode():
        return float(torch.max(torch.abs(first(inputs) - second(inputs))).item())


def run_one(
    task_id: str,
    fold: int,
    model_type: str,
    seed: int,
    view: pd.DataFrame,
    config: dict,
    manifest_hash: str,
    output_root: Path,
) -> dict:
    run_name = f"{task_id}__fold_{fold}__{model_type.lower().replace('-', '_')}__seed_{seed}"
    run_dir = output_root / "runs" / run_name
    summary_path = run_dir / "run_summary.json"
    checkpoint = run_dir / "best_checkpoint.pt"
    if summary_path.exists() and checkpoint.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("status") == "completed"
            and summary.get("manifest_sha256") == manifest_hash
            and summary.get("checkpoint_sha256") == sha256(checkpoint)
            and summary.get("test_evaluated") is False
        ):
            print(f"SKIP completed {run_name}", flush=True)
            return summary

    started = time.perf_counter()
    _, metadata = train_model_validation_only(
        model_type,
        seed,
        config,
        view[["sample_id", "sequence", "label", "split"]],
        run_dir,
    )
    validation_sequences = view[view.split.eq("validation")].sequence.head(32).tolist()
    reload_error = checkpoint_reload_max_error(checkpoint, validation_sequences)
    if reload_error != 0.0:
        raise ValueError(f"checkpoint reload changed outputs: {run_name}, max_error={reload_error}")
    summary = {
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": task_id,
        "fold": fold,
        "model_type": model_type,
        "seed": seed,
        "train_rows": int((view.split == "train").sum()),
        "validation_rows": int((view.split == "validation").sum()),
        "best_epoch": int(metadata["best_epoch"]),
        "validation_metrics": metadata["validation_metrics"],
        "training_seconds": float(metadata["training_seconds"]),
        "total_run_seconds": time.perf_counter() - started,
        "device": str(DEVICE),
        "test_evaluated": False,
        "checkpoint_reload_max_abs_error": reload_error,
        "manifest_sha256": manifest_hash,
        "checkpoint_sha256": sha256(checkpoint),
        "history_sha256": sha256(run_dir / "training_history.csv"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"DONE {run_name} epoch={summary['best_epoch']} "
        f"val_auroc={summary['validation_metrics']['auroc']:.4f} "
        f"seconds={summary['total_run_seconds']:.1f}",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, action="append")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, manifest, _ = validate_preflight()
    torch.set_num_threads(max(1, min(6, torch.get_num_threads())))
    selected_tasks = tuple(args.task) if args.task else TASKS
    output_root = OUTPUT / ("smoke" if args.smoke else "full")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_hash = sha256(MANIFEST_PATH)
    training_config = model_config(config, smoke=args.smoke)
    summaries = []
    for task_id in selected_tasks:
        frame = load_development(task_id, manifest)
        folds = (1,) if args.smoke else FOLDS
        models = ("CNN-Raw",) if args.smoke else MODELS
        seeds = (42,) if args.smoke else SEEDS
        for fold in folds:
            view = fold_view(frame, fold, smoke=args.smoke)
            for model_type in models:
                for seed in seeds:
                    summaries.append(
                        run_one(
                            task_id, fold, model_type, seed, view,
                            training_config, manifest_hash, output_root,
                        )
                    )
    summary_frame = pd.DataFrame(summaries).sort_values(
        ["task_id", "fold", "model_type", "seed"], kind="mergesort"
    )
    summary_frame.to_json(
        output_root / "training_run_summary.json",
        orient="records",
        indent=2,
    )
    completed = int((summary_frame.status == "completed").sum())
    expected = len(selected_tasks) * (1 if args.smoke else len(FOLDS) * len(MODELS) * len(SEEDS))
    completion = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "smoke" if args.smoke else "full",
        "status": "completed" if completed == expected else "incomplete",
        "completed_runs": completed,
        "expected_runs": expected,
        "test_evaluated": False,
        "manifest_sha256": manifest_hash,
        "training_summary_sha256": sha256(output_root / "training_run_summary.json"),
    }
    (output_root / "completion.json").write_text(
        json.dumps(completion, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"status={completion['status']} runs={completed}/{expected} test_evaluated=False")
    return 0 if completion["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
