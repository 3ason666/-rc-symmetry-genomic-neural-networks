from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_phase3b import fold_view, model_config
from scripts.run_phase3c import (
    aggregate_predictions,
    attribution_stage,
    finalize_task,
    freeze_attribution_cohort,
    freeze_h1b_cohort,
    prepare_h1b_predictions,
    prediction_stage,
    task_runtime_config,
    validate_test_population,
)
from src.dna_utils import batch_one_hot_encode
from src.models import build_model
from src.training import DEVICE, train_model_validation_only


CONFIG_PATH = ROOT / "configs" / "phase3e_near_duplicate_sensitivity.yaml"
REVISION = "phase3e_near_duplicate_sensitivity_v1"
TASKS = ("p3_gata1_fetal", "p3_ctcf_gm12878")
MODELS = ("CNN-Raw", "CNN-Aug", "CNN-RCPS")
FOLDS = (1, 2, 3)
SEEDS = (42, 123, 2026)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_and_validate() -> tuple[dict, dict, dict, dict, dict[str, set[str]]]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("protocol_revision") != REVISION:
        raise ValueError("unexpected Phase 3E protocol revision")
    if config.get("device") != "cpu" or config["execution"].get("allow_force_rerun"):
        raise ValueError("Phase 3E device or rerun lock changed")
    training = config["training"]
    if tuple(training["model_types"]) != MODELS or tuple(training["folds"]) != FOLDS or tuple(training["seeds"]) != SEEDS:
        raise ValueError("Phase 3E model grid changed")
    if training.get("tune_hyperparameters") or training.get("select_models_after_test"):
        raise ValueError("post-primary tuning or model selection is forbidden")
    if config["execution"].get("primary_p3_results_replaced"):
        raise ValueError("sensitivity results cannot replace primary P3")

    frozen = config["frozen_inputs"]
    for key, hash_key in (
        ("phase3c_config_path", "phase3c_config_sha256"),
        ("phase3c_completion_path", "phase3c_completion_sha256"),
        ("phase3d_completion_path", "phase3d_completion_sha256"),
        ("exclusion_path", "exclusion_sha256"),
    ):
        path = ROOT / frozen[key]
        if sha256(path) != str(frozen[hash_key]).lower():
            raise ValueError(f"frozen input hash changed: {key}")
    p3d_completion_path = ROOT / frozen["phase3d_completion_path"]
    p3d_completion = json.loads(p3d_completion_path.read_text(encoding="utf-8"))
    p3d_summary_path = p3d_completion_path.parent / "audit_summary.json"
    if sha256(p3d_summary_path) != p3d_completion.get("audit_summary_sha256"):
        raise ValueError("Phase 3D summary hash no longer matches its completion lock")
    p3d_summary = json.loads(p3d_summary_path.read_text(encoding="utf-8"))
    if p3d_completion.get("status") != "completed_locked" or not p3d_summary.get("overall_triggers", {}).get("high_similarity_rerun"):
        raise ValueError("Phase 3D did not authorize the high-similarity rerun")
    p3c_completion = json.loads((ROOT / frozen["phase3c_completion_path"]).read_text(encoding="utf-8"))
    if p3c_completion.get("status") != "completed" or int(p3c_completion.get("test_execution_count", 0)) != 1:
        raise ValueError("locked primary Phase 3C result is missing")

    phase3a = yaml.safe_load((ROOT / frozen["phase3a_config_path"]).read_text(encoding="utf-8"))
    phase3c = yaml.safe_load((ROOT / frozen["phase3c_config_path"]).read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / frozen["phase3b_manifest_path"]).read_text(encoding="utf-8"))
    exclusions = pd.read_csv(ROOT / frozen["exclusion_path"])
    required = {"task_id", "p3_sample_id", "identity", "coverage"}
    if not required.issubset(exclusions.columns):
        raise ValueError("Phase 3D exclusion schema changed")
    rule = config["exclusion_rule"]
    if (exclusions.identity < float(rule["identity_min"])).any() or (exclusions.coverage < float(rule["bidirectional_coverage_min"])).any():
        raise ValueError("exclusion file contains an unconfirmed pair")
    exclusion_ids = {
        task: set(exclusions.loc[exclusions.task_id.eq(task), "p3_sample_id"].astype(str))
        for task in TASKS
    }
    expected = rule["expected_unique_ids"]
    for task in TASKS:
        if len(exclusion_ids[task]) != int(expected[task]):
            raise ValueError(f"exclusion count changed: {task}")
    return config, phase3a, phase3c, manifest, exclusion_ids


def load_development(task_id: str, manifest: dict) -> pd.DataFrame:
    record = manifest["tasks"][task_id]
    path = ROOT / record["development_path"]
    if sha256(path) != record["development_sha256"]:
        raise ValueError(f"development data changed: {task_id}")
    frame = pd.read_csv(path)
    if len(frame) != int(record["development_rows"]) or set(frame.fixed_partition.astype(str)) != {"development"}:
        raise ValueError(f"development population changed: {task_id}")
    return frame


def reload_error(checkpoint: Path, sequences: list[str]) -> float:
    saved = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    first = build_model(saved["model_config"], architecture=saved["architecture"]).to(DEVICE)
    second = build_model(saved["model_config"], architecture=saved["architecture"]).to(DEVICE)
    first.load_state_dict(saved["model_state"]); second.load_state_dict(saved["model_state"])
    first.eval(); second.eval()
    inputs = batch_one_hot_encode(sequences).to(DEVICE)
    with torch.inference_mode():
        return float(torch.max(torch.abs(first(inputs) - second(inputs))).item())


def train_one(task_id: str, fold: int, model: str, seed: int, view: pd.DataFrame, train_cfg: dict, output: Path, input_hash: str, excluded: int) -> dict:
    name = f"{task_id}__fold_{fold}__{model.lower().replace('-', '_')}__seed_{seed}"
    run_dir = output / "runs" / name
    summary_path = run_dir / "run_summary.json"
    checkpoint = run_dir / "best_checkpoint.pt"
    if summary_path.exists() and checkpoint.exists():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        if prior.get("status") == "completed" and prior.get("input_fingerprint") == input_hash and prior.get("checkpoint_sha256") == sha256(checkpoint):
            print(f"SKIP completed {name}", flush=True)
            return prior
    started = time.perf_counter()
    _, metadata = train_model_validation_only(model, seed, train_cfg, view[["sample_id", "sequence", "label", "split"]], run_dir)
    error = reload_error(checkpoint, view[view.split.eq("validation")].sequence.head(32).tolist())
    if error != 0.0:
        raise ValueError(f"checkpoint reload changed outputs: {name}")
    summary = {
        "status": "completed", "generated_at_utc": utc_now(), "sensitivity_only": True,
        "task_id": task_id, "fold": fold, "model_type": model, "seed": seed,
        "excluded_training_rows": excluded, "train_rows": int(view.split.eq("train").sum()),
        "validation_rows": int(view.split.eq("validation").sum()), "validation_rows_excluded": 0,
        "best_epoch": int(metadata["best_epoch"]), "validation_metrics": metadata["validation_metrics"],
        "training_seconds": float(metadata["training_seconds"]), "total_run_seconds": time.perf_counter() - started,
        "device": str(DEVICE), "test_evaluated": False, "checkpoint_reload_max_abs_error": error,
        "input_fingerprint": input_hash, "checkpoint_sha256": sha256(checkpoint),
        "history_sha256": sha256(run_dir / "training_history.csv"),
    }
    write_json(summary_path, summary)
    print(f"DONE {name} excluded={excluded} val_auroc={summary['validation_metrics']['auroc']:.4f} seconds={summary['total_run_seconds']:.1f}", flush=True)
    return summary


def run_training(config: dict, phase3a: dict, manifest: dict, exclusion_ids: dict[str, set[str]]) -> None:
    output = ROOT / config["execution"]["training_dir"]
    output.mkdir(parents=True, exist_ok=True)
    input_hash = sha256(CONFIG_PATH) + ":" + config["frozen_inputs"]["exclusion_sha256"]
    cfg = model_config(phase3a, smoke=False)
    summaries = []
    exclusions = []
    for task in TASKS:
        frame = load_development(task, manifest)
        for fold in FOLDS:
            view = fold_view(frame, fold, smoke=False)
            before_train = int(view.split.eq("train").sum())
            validation_ids = set(view.loc[view.split.eq("validation"), "sample_id"].astype(str))
            remove_mask = view.split.eq("train") & view.sample_id.astype(str).isin(exclusion_ids[task])
            removed_ids = sorted(view.loc[remove_mask, "sample_id"].astype(str))
            view = view.loc[~remove_mask].copy()
            if set(view.loc[view.split.eq("validation"), "sample_id"].astype(str)) != validation_ids:
                raise ValueError("validation population changed")
            exclusions.append({
                "task_id": task, "fold": fold, "original_train_rows": before_train,
                "excluded_train_rows": len(removed_ids), "retained_train_rows": int(view.split.eq("train").sum()),
                "validation_rows": int(view.split.eq("validation").sum()),
                "excluded_sample_id_sha256": hashlib.sha256("\n".join(removed_ids).encode("utf-8")).hexdigest(),
            })
            for model in MODELS:
                for seed in SEEDS:
                    summaries.append(train_one(task, fold, model, seed, view, cfg, output, input_hash, len(removed_ids)))
    summary = pd.DataFrame(summaries).sort_values(["task_id", "fold", "model_type", "seed"], kind="mergesort")
    summary.to_json(output / "training_run_summary.json", orient="records", indent=2)
    pd.DataFrame(exclusions).to_csv(output / "fold_exclusion_summary.csv", index=False)
    completion = {
        "status": "completed", "completed_at_utc": utc_now(), "sensitivity_only": True,
        "completed_runs": int(len(summary)), "expected_runs": int(config["training"]["expected_total_count"]),
        "test_evaluated": False, "input_fingerprint": input_hash,
        "training_summary_sha256": sha256(output / "training_run_summary.json"),
        "fold_exclusion_summary_sha256": sha256(output / "fold_exclusion_summary.csv"),
    }
    if completion["completed_runs"] != completion["expected_runs"]:
        raise ValueError("training grid incomplete")
    write_json(output / "completion.json", completion)


def checkpoint_records(config: dict) -> list[dict]:
    source = ROOT / config["execution"]["training_dir"] / "runs"
    records = []
    for path in sorted(source.glob("*/run_summary.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = path.parent / "best_checkpoint.pt"
        history = path.parent / "training_history.csv"
        if row.get("status") != "completed" or row.get("test_evaluated") is not False:
            raise ValueError(f"invalid sensitivity checkpoint: {path}")
        if sha256(checkpoint) != row["checkpoint_sha256"] or sha256(history) != row["history_sha256"]:
            raise ValueError(f"sensitivity checkpoint hash mismatch: {path.parent.name}")
        records.append({
            "run_name": path.parent.name, "task_id": row["task_id"], "fold_id": f"fold_{int(row['fold'])}",
            "fold": int(row["fold"]), "model_type": row["model_type"], "seed": int(row["seed"]),
            "checkpoint_path": checkpoint.relative_to(ROOT).as_posix(), "checkpoint_sha256": sha256(checkpoint),
            "validation_auroc": float(row["validation_metrics"]["auroc"]), "test_evaluated": False,
        })
    expected = {(t, f, m, s) for t in TASKS for f in FOLDS for m in MODELS for s in SEEDS}
    observed = {(r["task_id"], r["fold"], r["model_type"], r["seed"]) for r in records}
    if observed != expected:
        raise ValueError("sensitivity checkpoint grid incomplete")
    return records


def run_evaluation(config: dict, phase3c: dict) -> None:
    output = ROOT / config["execution"]["evaluation_dir"]
    completion_path = ROOT / config["execution"]["completion_path"]
    if completion_path.exists():
        raise RuntimeError("Phase 3E evaluation is already complete; rerun is forbidden")
    training_completion = ROOT / config["execution"]["training_dir"] / "completion.json"
    if not training_completion.exists() or json.loads(training_completion.read_text(encoding="utf-8")).get("status") != "completed":
        raise RuntimeError("Phase 3E training is incomplete")
    records = checkpoint_records(config)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG_PATH, output / "resolved_config.yaml")
    shutil.copy2(ROOT / config["protocol_path"], output / "frozen_protocol.md")
    results = []
    for task_id in TASKS:
        task_output = output / task_id
        task_output.mkdir(parents=True, exist_ok=True)
        task = phase3c["tasks"][task_id]
        test = validate_test_population(pd.read_csv(ROOT / task["sealed_test_path"]), task_id, phase3c)
        task_records = [row for row in records if row["task_id"] == task_id]
        runtime = task_runtime_config(phase3c, task_id)
        prediction_stage(test, runtime, {"checkpoints": task_records}, task_output)
        predictions, _ = aggregate_predictions(task_output)
        base = freeze_attribution_cohort(test, runtime, task_output)
        h1b, pool = freeze_h1b_cohort(test, predictions, phase3c, task_output)
        attribution_stage(base, runtime, {"checkpoints": task_records}, task_output)
        if len(h1b) == int(phase3c["attribution"]["h1b_target_samples"]):
            h1b_output = task_output / "h1b"
            raw = [row for row in task_records if row["model_type"] == "CNN-Raw"]
            prepare_h1b_predictions(task_output, h1b_output, h1b, raw)
            attribution_stage(h1b, task_runtime_config(phase3c, task_id, h1b=True), {"checkpoints": raw}, h1b_output)
        results.append(finalize_task(task_id, phase3c, task_records, task_output, pool))
    decision = {
        "sensitivity_only": True, "primary_p3_results_replaced": False,
        "tasks": {item["task_id"]: {row["hypothesis"]: row["status"] for row in item["decisions"]} for item in results},
    }
    write_json(output / "phase3e_sensitivity_decision.json", decision)
    completion = {
        "status": "completed_locked", "completed_at_utc": utc_now(), "protocol_revision": REVISION,
        "training_runs": len(records), "evaluation_execution_count": 1,
        "sensitivity_only": True, "primary_p3_results_replaced": False,
        "decision_sha256": sha256(output / "phase3e_sensitivity_decision.json"),
    }
    write_json(completion_path, completion)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3E cross-phase near-duplicate sensitivity")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    config, phase3a, phase3c, manifest, exclusions = load_and_validate()
    torch.set_num_threads(int(config["cpu_threads"]))
    if args.preflight:
        print(json.dumps({"status": "preflight_passed", "excluded_unique_ids": {k: len(v) for k, v in exclusions.items()}, "test_rows_loaded": False}, ensure_ascii=False))
        return
    if args.train:
        run_training(config, phase3a, manifest, exclusions)
        print(json.dumps({"status": "training_completed", "test_evaluated": False}))
        return
    if args.evaluate:
        run_evaluation(config, phase3c)
        print(json.dumps({"status": "completed_locked", "primary_p3_results_replaced": False}))
        return
    raise SystemExit("choose exactly one of --preflight, --train, or --evaluate")


if __name__ == "__main__":
    main()
